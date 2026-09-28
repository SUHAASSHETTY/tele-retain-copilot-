"""Tool-invocation logging middleware -> logs/tool_calls.jsonl.

`log_tool_call` decorates any sync or async tool function; `instrument_tool` applies it to an
existing LangChain BaseTool (used for the MCP tools loaded through langchain-mcp-adapters).
Each call appends one record:
  {timestamp, run_id, session_id, agent, tool_name, tool_source, args, result,
   result_truncated, latency_ms, status, error}
Args and results pass through src.guardrails.pii.mask_obj, and results are truncated.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.config import TOOL_CALLS_LOG
from src.guardrails.pii import mask, mask_obj
from src.run_context import current_agent, current_run_id, current_session_id

MAX_RESULT_CHARS = 1500
_IGNORED_ARGS = {"self", "cls", "runtime", "config", "callbacks", "run_manager"}
_lock = threading.Lock()
_log_path: Path = TOOL_CALLS_LOG


def set_log_path(path: Path) -> None:
    """Redirect the log (tests use a temp file so committed evidence is untouched)."""
    global _log_path
    _log_path = path


def normalize_result(result: Any) -> Any:
    """Turn tool outputs (pydantic, MCP content/artifact tuples, JSON text) into plain data."""
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, tuple) and len(result) == 2:  # (content, artifact) from MCP tools
        content, artifact = result
        if isinstance(artifact, dict) and artifact.get("structured_content") is not None:
            return artifact["structured_content"]
        result = content
    if isinstance(result, list) and all(isinstance(b, dict) for b in result):
        result = "\n".join(b.get("text", "") for b in result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _status(result: Any) -> str:
    if isinstance(result, dict) and result.get("ok") is False:
        code = (result.get("error") or {}).get("code", "")
        return "denied" if code.startswith("AUTH") else "error"
    return "ok"


def _write(record: dict) -> None:
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _record(name: str, source: str, agent: str | None, args: dict, result: Any, status: str,
            error: BaseException | None, started: float, span=None) -> None:
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if span is not None:  # the middleware's own span (auto-instrumented TOOL spans are not "current")
        span.set_attribute("copilot.tool_status", status)
        span.set_attribute("copilot.tool_latency_ms", latency_ms)
    masked = mask_obj(normalize_result(result)) if result is not None else None
    text = json.dumps(masked, ensure_ascii=False, default=str) if masked is not None else ""
    truncated = len(text) > MAX_RESULT_CHARS
    _write({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": current_run_id(),
        "session_id": current_session_id(),
        "agent": current_agent() or agent,
        "tool_name": name,
        "tool_source": source,
        "args": mask_obj(args),
        "result": (text[:MAX_RESULT_CHARS] + f"...[truncated {len(text) - MAX_RESULT_CHARS} chars]")
        if truncated else masked,
        "result_truncated": truncated,
        "latency_ms": latency_ms,
        "status": status,
        "error": None if error is None else {"type": type(error).__name__, "message": mask(str(error))[:300]},
    })


def _bind_args(fn: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        items = dict(bound.arguments)
    except TypeError:
        items = {"args": list(args), **kwargs}
    # flatten **kwargs parameters (e.g. MCP adapters' `call_tool(**arguments)`)
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind is inspect.Parameter.VAR_KEYWORD and pname in items:
            items.update(items.pop(pname))
    return {k: v for k, v in items.items() if k not in _IGNORED_ARGS}


def log_tool_call(tool_name: str | None = None, *, source: str = "local", agent: str | None = None):
    """Decorator: log every invocation of the wrapped tool function to logs/tool_calls.jsonl."""

    def decorator(fn: Callable) -> Callable:
        name = tool_name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                from src.observability.tracing import tool_middleware_span

                started, call_args = time.perf_counter(), _bind_args(fn, args, kwargs)
                with tool_middleware_span(name, source) as span:
                    try:
                        result = await fn(*args, **kwargs)
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        _record(name, source, agent, call_args, None, "timeout", exc, started, span)
                        raise
                    except asyncio.CancelledError as exc:
                        _record(name, source, agent, call_args, None, "cancelled", exc, started, span)
                        raise
                    except Exception as exc:
                        _record(name, source, agent, call_args, None, "error", exc, started, span)
                        raise
                    _record(name, source, agent, call_args, result, _status(normalize_result(result)), None,
                            started, span)
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            started, call_args = time.perf_counter(), _bind_args(fn, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                _record(name, source, agent, call_args, None, "error", exc, started)
                raise
            _record(name, source, agent, call_args, result, _status(normalize_result(result)), None, started)
            return result

        return sync_wrapper

    return decorator


def instrument_tool(tool: BaseTool, *, source: str) -> BaseTool:
    """Wrap an existing LangChain tool's implementation with `log_tool_call` (idempotent)."""
    if getattr(tool, "_tool_call_logged", False):
        return tool
    if getattr(tool, "coroutine", None) is not None:
        tool.coroutine = log_tool_call(tool.name, source=source)(tool.coroutine)
    if getattr(tool, "func", None) is not None:
        tool.func = log_tool_call(tool.name, source=source)(tool.func)
    object.__setattr__(tool, "_tool_call_logged", True)
    return tool
