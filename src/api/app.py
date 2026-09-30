"""Async FastAPI streaming API (bonus). Same graph, guards, tools and audit as the CLI.

  GET  /health
  POST /v1/contacts/stream              {"message": "...", "session_id": "optional"}
       header X-Customer-Id: CUST-######  (stands in for an authenticated session, as in the CLI)
       -> Server-Sent Events: `node` (one per graph node, masked summary), then either
          `approval_required` (the turn is paused at the human-approval interrupt) or `resolution`
  POST /v1/sessions/{session_id}/approval  {"approved": true|false, "approver": "name"}
       -> resumes the paused thread and streams the rest of the turn
  GET  /                                 browser UI (src/api/static/index.html; no external assets)
  GET  /v1/samples                       synthetic sample scenarios for the UI
  GET  /v1/account  (X-Customer-Id)      masked account summary via the MCP tool (same authorization)

Every event payload passes through src.guardrails.pii.mask_obj. The customer only ever receives the
output guard's final text. Run: uvicorn src.api.app:app --port 8000
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.common import Deps
from pathlib import Path

from src import ui
from src.config import SAMPLE_CONTACTS_PATH, settings
from src.graph import open_graph, redact_at_ingest
from src.guardrails.pii import mask_customer_id, mask_obj
from src.llm import llm_preflight
from src.memory.long_term import open_long_term_memory
from src.memory.short_term import thread_config
from src.observability.tracing import annotate_turn, turn_span
from src.run_context import llm_enabled_var, run_scope
from src.tools.mcp_client import TelecomMCP

CUSTOMER_RE = re.compile(r"^CUST-\d{6}$")
NODE_FIELDS = {  # what each node event reveals (already structured, then masked)
    "input_guard": ("guard_blocked", "input_flags"),
    "supervisor": ("next_worker", "step_count"),
    "intent_agent": ("intent", "intent_confidence"),
    "account_agent": ("account_failed",),
    "policy_retrieval_agent": ("policy_done",),
    "retention_offer_agent": ("needs_human_approval",),
    "resolution_agent": ("resolution",),
    "human_approval": ("approval",),
    "output_guard": ("output_flags", "output_risk"),
}


class ContactRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{3,64}$")


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(min_length=2, max_length=64)


class Runtime:
    """Long-lived resources shared by all requests."""
    app: Any = None
    mcp: TelecomMCP | None = None
    session: Any = None
    memory: Any = None
    use_llm: bool = False
    sessions: dict[str, dict] = {}   # session_id -> {customer_id, run_id}


rt = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with AsyncExitStack() as stack:
        from src.warmup import warm_up

        warm_up()
        rt.app = await stack.enter_async_context(open_graph())
        rt.mcp = TelecomMCP()
        rt.session = await stack.enter_async_context(rt.mcp.session())
        rt.memory = await stack.enter_async_context(open_long_term_memory())
        rt.use_llm, _ = await llm_preflight()
        yield


api = FastAPI(title="Retention Copilot API", version="1.0", lifespan=lifespan)
app = api


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(mask_obj(data, amounts=False), default=str)}\n\n"


def _summary(node: str, update: Any) -> dict:
    if not isinstance(update, dict):
        return {}
    fields = NODE_FIELDS.get(node, ())
    out = {k: update[k] for k in fields if k in update}
    if "resolution" in out and isinstance(out["resolution"], dict):
        out["resolution"] = {k: out["resolution"].get(k) for k in ("outcome", "reason")}
    return out


async def _stream(payload: Any, customer_id: str, session_id: str, run_id: str) -> AsyncIterator[str]:
    deps = Deps(mcp=rt.session, use_llm=rt.use_llm, max_steps=8, memory=rt.memory)
    token = rt.mcp.issue_session_token(customer_id, session_id)
    config = thread_config(session_id, run_id, settings.graph_recursion_limit)
    llm_tok = llm_enabled_var.set(rt.use_llm)
    try:
        with run_scope(run_id=run_id, session_id=session_id, mcp_auth_token=token), \
                turn_span(run_id=run_id, session_id=session_id, customer_id=customer_id, contact_id=session_id,
                          turn=1) as span:
            yield sse("start", {"session_id": session_id, "run_id": run_id, "customer": mask_customer_id(customer_id)})
            async for chunk in rt.app.astream(payload, config, context=deps, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        req = update[0].value if update else {}
                        yield sse("approval_required", {
                            "session_id": session_id, "request": req, "resume": f"/v1/sessions/{session_id}/approval",
                            "explanation": f"The copilot wants to offer {ui.describe_offer(req)} (total "
                                           f"{ui.money(req.get('offer_value_usd'))}). Offers above "
                                           f"{ui.money(ui.APPROVAL_THRESHOLD)} need a team lead's approval."})
                        return
                    yield sse("node", {"node": node, "label": ui.NODE_LABELS.get(node, node), **_summary(node, update)})
            state = (await rt.app.aget_state(config)).values
            annotate_turn(span, state)
            yield sse("resolution", _resolution_event(state))
    finally:
        llm_enabled_var.reset(llm_tok)


def _resolution_event(state: dict) -> dict:
    res = state.get("resolution") or {}
    icon, _color, label = ui.OUTCOME_LABELS.get(res.get("outcome"), ("", "", str(res.get("outcome"))))
    notes = []
    if state.get("guard_blocked"):
        notes.append(ui.GUARD_EXPLAIN.get(state.get("guard_reason"), "Blocked by a safety check."))
    for b in (state.get("offer_decision") or {}).get("blocked", []):
        if b.get("source") == "customer_request":
            notes.append(f"Requested {ui.describe_offer(b)} is above policy limits ({', '.join(b['policy_refs'][:2])}).")
    approval = state.get("approval")
    if approval:
        notes.append("Offer " + ("approved" if approval.get("approved") else "rejected") + " by the approver.")
    if res.get("ticket_id"):
        notes.append(f"Ticket {res['ticket_id']} opened for a specialist.")
    return {"outcome": res.get("outcome"), "outcome_label": f"{icon} {label}".strip(),
            "intent": ui.INTENT_LABELS.get(state.get("intent"), state.get("intent")),
            "risk_tier": (state.get("output_risk") or {}).get("tier"), "reply": state.get("final_response"),
            "notes": notes, "ticket_id": res.get("ticket_id"),
            "citations": [{"id": c, "title": ui.clause_titles().get(c, "")} for c in dict.fromkeys(res.get("citations") or [])]}


@api.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text()


@api.get("/v1/samples")
async def samples() -> list[dict]:
    out = []
    for line in SAMPLE_CONTACTS_PATH.read_text().splitlines():
        c = json.loads(line)
        out.append({"label": c["scenario"].replace("_", " "), "customer_id": c["customer_id"],
                    "customer_ref": mask_customer_id(c["customer_id"]), "message": c["turns"][-1]})
    return out


@api.get("/v1/account")
async def account(x_customer_id: str = Header(...)) -> dict:
    if not CUSTOMER_RE.match(x_customer_id):
        raise HTTPException(401, "X-Customer-Id must identify the authenticated caller (CUST-######)")
    token = rt.mcp.issue_session_token(x_customer_id, "WEB-ACCOUNT")
    with run_scope(session_id="WEB-ACCOUNT", mcp_auth_token=token):
        out = await rt.session.call("get_account", customer_id=x_customer_id)
    if not out.get("ok"):
        raise HTTPException(404, "account not found")
    return {"lines": ui.account_lines(out["account"]), "customer_ref": out["account"]["customer_ref"]}


@api.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm": "gemini" if rt.use_llm else "rules/templates", "model": settings.gemini_model}


@api.post("/v1/contacts/stream")
async def contact_stream(req: ContactRequest, x_customer_id: str = Header(...)) -> StreamingResponse:
    if not CUSTOMER_RE.match(x_customer_id):
        raise HTTPException(401, "X-Customer-Id must identify the authenticated caller (CUST-######)")
    session_id = req.session_id or f"API-{uuid.uuid4().hex[:8].upper()}"
    known = rt.sessions.get(session_id)
    if known and known["customer_id"] != x_customer_id:
        raise HTTPException(403, "session belongs to another customer")
    run_id = str(uuid.uuid4())
    rt.sessions[session_id] = {"customer_id": x_customer_id, "run_id": run_id}
    clean, flags = redact_at_ingest(req.message)
    payload = {"pending_input": clean, "ingest_flags": flags, "customer_id": x_customer_id,
               "session_id": session_id, "thread_id": session_id, "run_id": run_id}
    return StreamingResponse(_stream(payload, x_customer_id, session_id, run_id), media_type="text/event-stream")


@api.post("/v1/sessions/{session_id}/approval")
async def approval(session_id: str, req: ApprovalRequest) -> StreamingResponse:
    known = rt.sessions.get(session_id)
    if not known:
        raise HTTPException(404, "unknown session")
    snapshot = await rt.app.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.interrupts:
        raise HTTPException(409, "no approval pending for this session")
    decision = {"approved": req.approved, "approver": req.approver, "note": "via API"}
    return StreamingResponse(_stream(Command(resume=decision), known["customer_id"], session_id, known["run_id"]),
                             media_type="text/event-stream")
