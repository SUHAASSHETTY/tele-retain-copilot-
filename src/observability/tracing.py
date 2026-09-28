"""Arize Phoenix tracing, called from the CLI run path (`init_tracing()` in src/cli.py).

- Launches the local in-process Phoenix app on :6006 (data persisted under PHOENIX_WORKING_DIR so
  scripts/export_traces.py can export after the run) and registers it with `phoenix.otel.register`.
- `LangChainInstrumentor().instrument()` traces every LangGraph node (CHAIN), LLM call (LLM) and
  LangChain/MCP tool call (TOOL). Manual spans add what auto-instrumentation misses: a per-turn
  AGENT root span carrying run_id / masked customer / intent / outcome, GUARDRAIL spans for the
  input and output guards, and RETRIEVER spans for policy retrieval.
- PII: the default exporter is replaced by `MaskingSpanExporter`, which runs every string
  attribute (including JSON inputs/outputs) through src.guardrails.pii before a span leaves the
  process. No plaintext identifiers or amounts are exported.

Span classes used by the golden signals: thinking = LLM, acting = AGENT/CHAIN/GUARDRAIL,
tool = TOOL/RETRIEVER. The tool logging middleware adds `tool_middleware.<tool>` spans (kind UNKNOWN,
not counted as tools) carrying copilot.tool_status / copilot.tool_latency_ms.
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from src.config import settings
from src.guardrails.pii import mask, mask_customer_id, mask_obj
from src.run_context import current_agent, current_run_id, current_session_id

SPAN_KIND = "openinference.span.kind"
TRACER_NAME = "retention-copilot"
_state: dict[str, Any] = {"provider": None, "session": None}


def phoenix_url() -> str:
    return f"http://localhost:{settings.phoenix_port}"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_phoenix() -> str:
    """Start the in-process Phoenix app (reuses one already listening on the port)."""
    if _port_open(settings.phoenix_port):
        return phoenix_url()
    settings.phoenix_working_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(settings.phoenix_working_dir))
    os.environ["PHOENIX_PORT"] = str(settings.phoenix_port)
    import phoenix as px

    for attempt in (1, 2):
        try:
            _state["session"] = px.launch_app(use_temp_dir=False, run_in_thread=True)
        except RuntimeError:
            # First start in a new working dir runs DB migrations and can exceed Phoenix's own ~15 s
            # start-up timeout ("server took too long to start"); the server thread keeps booting.
            pass
        for _ in range(240):  # up to 60 s
            if _port_open(settings.phoenix_port):
                return phoenix_url()
            time.sleep(0.25)
    raise RuntimeError(f"Phoenix did not start on port {settings.phoenix_port}")


# --- masking exporter ------------------------------------------------------------

def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped[:1] in "{[":
            try:
                return json.dumps(mask_obj(json.loads(value)), ensure_ascii=False, default=str)
            except (json.JSONDecodeError, TypeError):
                pass
        return mask(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_value(v) for v in value)
    return value


def masked_span(span: ReadableSpan) -> ReadableSpan:
    attrs = {k: _mask_value(v) for k, v in (span.attributes or {}).items()}
    events = []
    for e in span.events or []:
        e_attrs = {k: _mask_value(v) for k, v in (e.attributes or {}).items()}
        events.append(type(e)(name=e.name, attributes=e_attrs, timestamp=e.timestamp))
    return ReadableSpan(name=span.name, context=span.context, parent=span.parent, resource=span.resource,
                        attributes=attrs, events=events, links=span.links, kind=span.kind, status=span.status,
                        start_time=span.start_time, end_time=span.end_time,
                        instrumentation_scope=span.instrumentation_scope)


class MaskingSpanExporter(SpanExporter):
    """Masks identifiers and amounts in every span before delegating to the real exporter."""

    def __init__(self, inner: SpanExporter):
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([masked_span(s) for s in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return getattr(self.inner, "force_flush", lambda *_: True)(timeout_millis)


class CopilotContextProcessor(SpanProcessor):
    """Stamps every span with run/session/agent from the run context so each run is findable."""

    def on_start(self, span, parent_context=None) -> None:
        for key, value in (("copilot.run_id", current_run_id()), ("copilot.session_id", current_session_id()),
                           ("copilot.agent", current_agent())):
            if value:
                span.set_attribute(key, value)

    def on_end(self, span) -> None:  # noqa: D401 - nothing to do
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# --- setup -----------------------------------------------------------------------

def init_tracing(launch: bool = True):
    """Register Phoenix + instrument LangChain/LangGraph. Idempotent. Returns the TracerProvider."""
    if _state["provider"] is not None:
        return _state["provider"]
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import HTTPSpanExporter, SimpleSpanProcessor, register

    if launch:
        launch_phoenix()
    endpoint = f"{phoenix_url()}/v1/traces"
    provider = register(project_name=settings.phoenix_project_name, endpoint=endpoint,
                        set_global_tracer_provider=True, verbose=False)
    # Phoenix's TracerProvider drops its default (unmasked) exporter as soon as a processor is added,
    # so both are registered explicitly: context stamping first, then the masking exporter.
    provider.add_span_processor(CopilotContextProcessor())
    provider.add_span_processor(SimpleSpanProcessor(
        span_exporter=MaskingSpanExporter(HTTPSpanExporter(endpoint=endpoint))))
    active = [type(x).__name__ for x in getattr(provider._active_span_processor, "_span_processors", [])]
    if "SimpleSpanProcessor" not in active:
        raise RuntimeError(f"tracing misconfigured: no exporting span processor ({active})")
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _state["provider"] = provider
    return provider


def tracing_enabled() -> bool:
    return _state["provider"] is not None


def flush() -> None:
    if _state["provider"] is not None:
        _state["provider"].force_flush()


def reset_project() -> bool:
    """Delete the project's existing spans so an export contains exactly the next run."""
    from phoenix.client import Client

    try:
        Client(base_url=phoenix_url()).projects.delete(project_name=settings.phoenix_project_name)
        return True
    except Exception:
        return False


def tracer():
    return trace.get_tracer(TRACER_NAME)


# --- manual spans ---------------------------------------------------------------

@contextmanager
def turn_span(*, run_id: str, session_id: str, customer_id: str, contact_id: str | None, turn: int) -> Iterator[Any]:
    """AGENT root span for one customer turn; run_id / masked customer propagate to all children."""
    from openinference.instrumentation import using_attributes

    customer = mask_customer_id(customer_id)
    meta = {"copilot_run_id": run_id, "customer": customer, "contact_id": contact_id or "", "turn": turn}
    with using_attributes(session_id=session_id, user_id=customer, metadata=meta):
        with tracer().start_as_current_span("copilot.turn", attributes={
            SPAN_KIND: "AGENT", "copilot.run_id": run_id, "copilot.session_id": session_id,
            "copilot.customer": customer, "copilot.contact_id": contact_id or "", "copilot.turn": turn,
        }) as span:
            yield span


def annotate_turn(span, state: dict) -> None:
    res = state.get("resolution") or {}
    for key, value in (("copilot.intent", state.get("intent")), ("copilot.outcome", res.get("outcome")),
                       ("copilot.guard_blocked", bool(state.get("guard_blocked"))),
                       ("copilot.steps", state.get("step_count"))):
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def guardrail_span(name: str) -> Iterator[Any]:
    with tracer().start_as_current_span(name, attributes={SPAN_KIND: "GUARDRAIL"}) as span:
        yield span


@contextmanager
def retriever_span(query: str) -> Iterator[Any]:
    with tracer().start_as_current_span("policy_retrieval.search", attributes={
            SPAN_KIND: "RETRIEVER", "input.value": query}) as span:
        yield span


def record_documents(span, hits: list[dict]) -> None:
    for i, h in enumerate(hits):
        span.set_attribute(f"retrieval.documents.{i}.document.id", h["citation"])
        span.set_attribute(f"retrieval.documents.{i}.document.score", float(h["score"]))
        span.set_attribute(f"retrieval.documents.{i}.document.content", h["text"][:500])


@contextmanager
def tool_middleware_span(tool_name: str, source: str) -> Iterator[Any]:
    """Span opened by the tool logging middleware; kind UNKNOWN so it is not double-counted as a tool."""
    with tracer().start_as_current_span(f"tool_middleware.{tool_name}", attributes={
            SPAN_KIND: "UNKNOWN", "copilot.tool_name": tool_name, "copilot.tool_source": source}) as span:
        yield span


def annotate_current_span(**attrs: Any) -> None:
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else json.dumps(v, default=str))


if __name__ == "__main__":  # python -m src.observability.tracing  -> browse persisted traces
    import signal

    print(f"Phoenix UI: {launch_phoenix()}  (projects: {settings.phoenix_project_name}, "
          f"retention-copilot-eval, retention-copilot-bench-*)  Ctrl-C to stop")
    signal.pause()
