"""MCP client for the telecom server via langchain-mcp-adapters MultiServerMCPClient (stdio).

- Auth: the client acts as the auth gateway. It signs a session token for the authenticated
  caller (`issue_session_token`), keeps it in the run context, and an interceptor injects it into
  every tool call. The `auth` parameter is stripped from the tool schemas the LLM sees, so the
  model can neither read nor forge it.
- Transcript: every MCP request/response (tools/list, tools/call, resources/read) is appended,
  masked via src.guardrails.pii.mask_obj, to logs/mcp_transcript.jsonl.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.types import CallToolResult, TextContent

from mcp_server.auth import SECRET_ENV, issue_token
from src.config import MCP_TRANSCRIPT_LOG, ROOT_DIR, settings
from src.audit.audit_middleware import audit
from src.guardrails.pii import mask_obj
from src.tools.logging_middleware import instrument_tool
from src.run_context import current_agent, current_run_id, current_session_id, mcp_auth_token_var

SERVER_NAME = "telecom"
AUTH_PARAM = "auth"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TranscriptWriter:
    """Append-only JSONL writer; every record is masked before it touches disk."""

    def __init__(self, path: Path = MCP_TRANSCRIPT_LOG):
        self.path = path
        self._lock = threading.Lock()

    def write(self, *, method: str, name: str, request: Any, response: Any, status: str,
              latency_ms: float, error_code: str | None = None) -> None:
        record = {
            "timestamp": _now(),
            "run_id": current_run_id(),
            "session_id": current_session_id(),
            "agent": current_agent(),
            "server": SERVER_NAME,
            "method": method,
            "name": name,
            "request": mask_obj(request),
            "response": mask_obj(response),
            "status": status,
            "error_code": error_code,
            "latency_ms": round(latency_ms, 1),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_tool_payload(result: Any) -> Any:
    """Best-effort structured payload from an MCP CallToolResult or LangChain tool output."""
    if isinstance(result, CallToolResult):
        if result.structuredContent is not None:
            return result.structuredContent
        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        result = "\n".join(texts)
    if isinstance(result, list):  # LangChain content blocks
        result = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"text": result}
    return result


def _status_of(payload: Any, is_error: bool) -> tuple[str, str | None]:
    if is_error:
        return "error", "MCP_TOOL_ERROR"
    if isinstance(payload, dict) and payload.get("ok") is False:
        code = (payload.get("error") or {}).get("code")
        return ("denied" if code in {"AUTHZ_DENIED", "AUTH_MISSING", "AUTH_INVALID"} else "error"), code
    return "ok", None


class AuthTranscriptInterceptor:
    """Injects the caller's session token and records the call in the MCP transcript."""

    def __init__(self, transcript: TranscriptWriter):
        self.transcript = transcript

    async def __call__(self, request: MCPToolCallRequest,
                       handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]
                       ) -> MCPToolCallResult:
        # Whatever the model supplied for `auth` is discarded; only the run context counts.
        args = {k: v for k, v in request.args.items() if k != AUTH_PARAM}
        args[AUTH_PARAM] = mcp_auth_token_var.get() or ""
        request = request.override(args=args)
        start = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as exc:
            self.transcript.write(method="tools/call", name=request.name, request=args,
                                  response={"exception": type(exc).__name__, "message": str(exc)},
                                  status="exception", latency_ms=(time.perf_counter() - start) * 1000,
                                  error_code=type(exc).__name__)
            raise
        payload = parse_tool_payload(result)
        status, code = _status_of(payload, bool(getattr(result, "isError", False)))
        self.transcript.write(method="tools/call", name=request.name, request=args, response=payload,
                              status=status, latency_ms=(time.perf_counter() - start) * 1000,
                              error_code=code)
        if status == "denied":
            err = payload.get("error") or {}
            audit("data_access_refused", "denied", tool=request.name, actor="mcp_client", reason=code,
                  policy_ref=err.get("policy_ref"),
                  details={"requested_customer": args.get("customer_id"), "message": err.get("message")})
        return result


def _prepare_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Remove the `auth` argument from the schema exposed to the LLM and wrap each tool with
    the tool-call logging middleware (logs/tool_calls.jsonl)."""
    for tool in tools:
        instrument_tool(tool, source="mcp")
        schema = tool.args_schema
        if isinstance(schema, dict) and AUTH_PARAM in schema.get("properties", {}):
            schema = copy.deepcopy(schema)
            schema["properties"].pop(AUTH_PARAM)
            schema["required"] = [r for r in schema.get("required", []) if r != AUTH_PARAM]
            tool.args_schema = schema
    return tools


class TelecomMCP:
    """Facade over MultiServerMCPClient for the telecom MCP server."""

    def __init__(self, transcript_path: Path = MCP_TRANSCRIPT_LOG, secret: str | None = None):
        self._secret = secret or os.getenv(SECRET_ENV) or secrets.token_hex(32)
        self.transcript = TranscriptWriter(transcript_path)
        self.interceptor = AuthTranscriptInterceptor(self.transcript)
        self.client = MultiServerMCPClient(
            {SERVER_NAME: self.connection()},
            tool_interceptors=[self.interceptor],
        )

    def connection(self) -> dict:
        return {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_server.server"],
            "cwd": str(ROOT_DIR),
            "env": {SECRET_ENV: self._secret, "PYTHONPATH": str(ROOT_DIR),
                    "OFFER_APPROVAL_THRESHOLD": str(settings.offer_approval_threshold),
                    "ANONYMIZED_TELEMETRY": "False"},
        }

    def issue_session_token(self, customer_id: str, session_id: str) -> str:
        """Called once the caller is authenticated; the token goes into run_scope()."""
        return issue_token(self._secret, session_id, customer_id)

    def _log_list(self, tools: list[BaseTool], latency_ms: float) -> None:
        self.transcript.write(method="tools/list", name="*", request={},
                              response={"tools": [t.name for t in tools]}, status="ok",
                              latency_ms=latency_ms)

    async def get_tools(self) -> list[BaseTool]:
        """Tools that open a fresh stdio session per call (simple, higher latency)."""
        start = time.perf_counter()
        tools = await asyncio.wait_for(self.client.get_tools(server_name=SERVER_NAME),
                                       timeout=settings.tool_timeout_s)
        self._log_list(tools, (time.perf_counter() - start) * 1000)
        return _prepare_tools(tools)

    @asynccontextmanager
    async def per_call_session(self) -> AsyncIterator["PerCallSession"]:
        """Baseline mode: no persistent session; each tool call spawns its own server process."""
        yield PerCallSession(self, await self.get_tools())

    @asynccontextmanager
    async def session(self) -> AsyncIterator["BoundSession"]:
        """One persistent stdio session (one server process) for a whole contact."""
        async with self.client.session(SERVER_NAME) as sess:
            start = time.perf_counter()
            tools = await load_mcp_tools(sess, tool_interceptors=[self.interceptor],
                                         server_name=SERVER_NAME)
            self._log_list(tools, (time.perf_counter() - start) * 1000)
            yield BoundSession(self, sess, _prepare_tools(tools))

    async def read_resource(self, uri: str, _session=None) -> Any:
        start = time.perf_counter()
        try:
            if _session is not None:
                blobs = await load_mcp_resources(_session, uris=[uri])
            else:
                blobs = await self.client.get_resources(SERVER_NAME, uris=[uri])
            text = blobs[0].as_string()
            payload = json.loads(text) if blobs[0].mimetype == "application/json" else text
        except Exception as exc:
            self.transcript.write(method="resources/read", name=uri, request={"uri": uri},
                                  response={"exception": type(exc).__name__, "message": str(exc)},
                                  status="exception", latency_ms=(time.perf_counter() - start) * 1000,
                                  error_code=type(exc).__name__)
            raise
        summary = payload
        if isinstance(payload, dict) and "docs" in payload:  # keep the transcript compact
            summary = {"doc_count": payload["doc_count"], "clause_count": payload["clause_count"],
                       "doc_ids": [d["doc_id"] for d in payload["docs"]]}
        self.transcript.write(method="resources/read", name=uri, request={"uri": uri},
                              response=summary, status="ok",
                              latency_ms=(time.perf_counter() - start) * 1000)
        return payload


class PerCallSession:
    """Same interface as BoundSession, but every tool call opens a fresh stdio session (new server
    process) via MultiServerMCPClient.get_tools(). Kept as the 'before' baseline for the latency
    optimization benchmark (scripts/optimization_benchmark.py)."""

    def __init__(self, owner: "TelecomMCP", tools: list[BaseTool]):
        self._owner = owner
        self.tools = tools
        self.by_name = {t.name: t for t in tools}

    @property
    def owner(self) -> "TelecomMCP":
        return self._owner

    async def call(self, name: str, **args: Any) -> Any:
        result = await asyncio.wait_for(self.by_name[name].ainvoke(args), timeout=settings.tool_timeout_s)
        return parse_tool_payload(result)

    async def read_resource(self, uri: str) -> Any:
        return await self._owner.read_resource(uri)


class BoundSession:
    """Tools and resources bound to one live MCP session."""

    def __init__(self, owner: TelecomMCP, session, tools: list[BaseTool]):
        self._owner = owner
        self._session = session
        self.tools = tools
        self.by_name = {t.name: t for t in tools}

    @property
    def owner(self) -> TelecomMCP:
        return self._owner

    async def call(self, name: str, **args: Any) -> Any:
        """Invoke a tool by name with a timeout; returns the parsed structured payload."""
        tool = self.by_name[name]
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=settings.tool_timeout_s)
        return parse_tool_payload(result)

    async def read_resource(self, uri: str) -> Any:
        return await self._owner.read_resource(uri, _session=self._session)
