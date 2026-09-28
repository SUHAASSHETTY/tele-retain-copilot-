"""LangGraph copilot graph.

    START -> input_guard -> context_manager -> supervisor --route_from_supervisor--> intent_agent | account_agent |
             policy_retrieval_agent | retention_offer_agent  (each returns to supervisor)
                                    \\-> resolution_agent --route_after_resolution--> request_approval -> human_approval |
                                                                                     escalate | output_guard
                                    \\-> clarify --route_after_clarify--> escalate | output_guard
                                    \\-> escalate (loop guard / tool failure)
             human_approval | escalate | resolution | clarify -> memory_writer -> output_guard -> END
    input_guard is always the first node and output_guard always the last.
    context_manager (after input_guard): compress history, extract session facts, select memories.
    memory_writer: persist facts / issue / offers to the customer's long-term LangMem store.

- Typed state (CopilotState); every node returns a validated Pydantic model (src/schemas.py).
- Short-term memory: AsyncSqliteSaver checkpointer (data/checkpoints.db), thread_id per conversation.
- Human gate: `interrupt()` in human_approval for offers/credits above OFFER_APPROVAL_THRESHOLD or
  that policy says need approval (fee waivers).
- Loop guards: max_steps in state (supervisor hops) and recursion_limit in the run config; hitting
  either escalates to a human instead of crashing.
"""

from __future__ import annotations

import operator
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from src.agents.account_agent import account_agent
from src.agents.common import Deps
from src.agents.human_loop import clarify, escalate, human_approval, request_approval, route_after_clarify
from src.agents.intent_agent import intent_agent
from src.agents.policy_retrieval_agent import policy_retrieval_agent
from src.agents.resolution_agent import resolution_agent, route_after_resolution
from src.agents.retention_offer_agent import retention_offer_agent
from src.agents.supervisor import route_from_supervisor, supervisor
from src.audit.audit_middleware import audit
from src.config import settings
from src.context.summarization import context_manager, memory_writer
from src.context.write import merge_facts, merge_scratchpad
from src.guardrails.input_guard import CARD_RE
from src.guardrails.nodes import input_guard, output_guard
from src.memory.short_term import CHECKPOINT_DB, open_checkpointer, thread_config
from src.observability.tracing import annotate_turn, turn_span
from src.run_context import llm_enabled_var, run_scope
DEFAULT_MAX_STEPS = 8


class CopilotState(TypedDict, total=False):
    # identity & bookkeeping
    run_id: str
    thread_id: str
    session_id: str
    customer_id: str                      # authenticated caller (never taken from customer text)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str | None
    step_count: int
    max_steps: int
    errors: Annotated[list[dict], operator.add]
    engine_log: Annotated[list[dict], operator.add]
    # context engineering / memory
    scratchpad: Annotated[dict, merge_scratchpad]     # write: per-node lanes (isolated per worker)
    session_facts: Annotated[list[dict], merge_facts] # write: facts stated in this conversation
    memories: list[dict]                              # select: long-term memories recalled this turn
    context_stats: dict                               # compress: token counts before/after
    memory_written: list[dict]
    # input guard / context
    pending_input: str | None
    ingest_flags: list[str]
    quarantined_input: str
    input_flags: list[str]
    guard_blocked: bool
    guard_reason: str | None
    # intent
    intent: str | None
    intent_confidence: float
    intent_rationale: str
    cancellation_intent: bool
    requested_offer: dict | None
    mentions_regulator: bool
    # supervisor
    next_worker: str | None
    supervisor_reason: str
    # account
    account_summary: dict | None
    billing: dict | None
    billing_findings: list[dict]
    account_failed: bool
    # policy
    policy_done: bool
    policy_question: str
    policy_answer: str | None
    policy_citations: list[dict]
    # offers
    offer_done: bool
    offer_proposal: dict | None
    offer_decision: dict | None
    needs_human_approval: bool
    approval_request: dict | None
    approval: dict | None
    # resolution / output
    resolution: dict | None
    clarify_count: int
    final_response: str | None
    output_flags: list[str]
    output_risk: dict | None


WORKERS = ["intent_agent", "account_agent", "policy_retrieval_agent", "retention_offer_agent"]


def build_graph(checkpointer=None):
    g = StateGraph(CopilotState, context_schema=Deps)
    g.add_node("input_guard", input_guard)
    g.add_node("context_manager", context_manager)
    g.add_node("memory_writer", memory_writer)
    g.add_node("supervisor", supervisor)
    g.add_node("intent_agent", intent_agent)
    g.add_node("account_agent", account_agent)
    g.add_node("policy_retrieval_agent", policy_retrieval_agent)
    g.add_node("retention_offer_agent", retention_offer_agent)
    g.add_node("resolution_agent", resolution_agent)
    g.add_node("request_approval", request_approval)
    g.add_node("human_approval", human_approval)
    g.add_node("clarify", clarify)
    g.add_node("escalate", escalate)
    g.add_node("output_guard", output_guard)

    g.add_edge(START, "input_guard")
    g.add_edge("input_guard", "context_manager")
    g.add_edge("context_manager", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            {w: w for w in WORKERS + ["resolution_agent", "clarify", "escalate"]})
    for w in WORKERS:
        g.add_edge(w, "supervisor")
    g.add_conditional_edges("resolution_agent", route_after_resolution,
                            {"human_approval": "request_approval", "escalate": "escalate",
                             "memory_writer": "memory_writer"})
    g.add_conditional_edges("clarify", route_after_clarify, {"escalate": "escalate", "memory_writer": "memory_writer"})
    g.add_edge("request_approval", "human_approval")
    g.add_edge("human_approval", "memory_writer")
    g.add_edge("escalate", "memory_writer")
    g.add_edge("memory_writer", "output_guard")
    g.add_edge("output_guard", END)  # the output guard is always the last node
    return g.compile(checkpointer=checkpointer, name="retention_copilot")


@asynccontextmanager
async def open_graph(db_path=CHECKPOINT_DB) -> AsyncIterator[Any]:
    """Compiled graph with the SQLite checkpointer (short-term memory)."""
    async with open_checkpointer(db_path) as saver:
        yield build_graph(checkpointer=saver)


ApprovalFn = Callable[[dict], Awaitable[dict]]


def redact_at_ingest(text: str) -> tuple[str, list[str]]:
    """Card numbers never enter the checkpointed state (PCI); everything else is the guard's job."""
    if CARD_RE.search(text):
        return CARD_RE.sub("[CARD-REDACTED]", text), ["payment_card_redacted"]
    return text, []


async def run_turn(app, *, text: str, customer_id: str, session_id: str, run_id: str, deps: Deps,
                   approve: ApprovalFn, contact_id: str | None = None, turn: int = 1) -> dict:
    """One customer turn through the graph inside an AGENT root span (run_id, masked customer,
    intent, outcome), resolving any approval interrupts."""
    with turn_span(run_id=run_id, session_id=session_id, customer_id=customer_id, contact_id=contact_id,
                   turn=turn) as span:
        state = await _run_turn(app, text=text, customer_id=customer_id, session_id=session_id, run_id=run_id,
                                deps=deps, approve=approve)
        annotate_turn(span, state)
        return state


async def _run_turn(app, *, text: str, customer_id: str, session_id: str, run_id: str, deps: Deps,
                    approve: ApprovalFn) -> dict:
    config = thread_config(session_id, run_id, settings.graph_recursion_limit)
    clean, flags = redact_at_ingest(text)
    payload: Any = {"pending_input": clean, "ingest_flags": flags, "customer_id": customer_id,
                    "session_id": session_id, "thread_id": session_id, "run_id": run_id}
    approvals = []
    try:
        while True:
            state = await app.ainvoke(payload, config, context=deps)
            interrupts = state.get("__interrupt__") or []
            if not interrupts:
                break
            decision = await approve(interrupts[0].value)
            approvals.append({"request": interrupts[0].value, "decision": decision})
            payload = Command(resume=decision)
    except GraphRecursionError:
        audit("loop_limit", "escalated", actor="system", reason="recursion_limit reached",
              policy_ref="POL-GOV-001 §2.2", details={"recursion_limit": settings.graph_recursion_limit})
        return {"final_response": "I wasn't able to complete this automatically, so I've passed it to a human "
                                  "agent who will follow up with you [POL-GOV-001 §2.2].",
                "resolution": {"outcome": "escalate", "reason": "recursion_limit"}, "approvals": approvals,
                "loop_guard": "recursion_limit"}
    state = dict(state)
    state["approvals"] = approvals
    return state


async def run_contact(app, mcp, contact: dict, *, approve: ApprovalFn, use_llm: bool | None = None,
                      max_steps: int = DEFAULT_MAX_STEPS, run_id: str | None = None, memory=None) -> dict:
    """Drive every turn of one contact. The contact's customer_id is the authenticated caller."""
    token = mcp.owner.issue_session_token(contact["customer_id"], contact["session_id"])
    deps_kwargs = {"mcp": mcp, "max_steps": max_steps, "memory": memory}
    if use_llm is not None:
        deps_kwargs["use_llm"] = use_llm
    deps = Deps(**deps_kwargs)
    llm_token = llm_enabled_var.set(deps.use_llm)
    try:
        return await _run_turns(app, contact, deps, token, approve, run_id)
    finally:
        llm_enabled_var.reset(llm_token)


async def _run_turns(app, contact: dict, deps: Deps, token: str, approve: ApprovalFn, run_id: str | None) -> dict:
    with run_scope(run_id=run_id, session_id=contact["session_id"], mcp_auth_token=token) as rid:
        turns = []
        for i, text in enumerate(contact["turns"], start=1):
            turns.append(await run_turn(app, text=text, customer_id=contact["customer_id"],
                                        session_id=contact["session_id"], run_id=rid, deps=deps,
                                        approve=approve, contact_id=contact.get("contact_id"), turn=i))
        return {"run_id": rid, "contact_id": contact.get("contact_id"), "turns": turns}
