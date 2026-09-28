"""Offer-policy retrieval worker: formulates a policy question from structured state (never from
raw customer text, which stays quarantined) and calls the agentic-RAG tool `policy_rag`."""

from __future__ import annotations

from langgraph.runtime import Runtime

from src.agents.common import Deps, engine_entry, error_entry
from src.context.isolate import view
from src.context.write import scratch
from src.run_context import agent_scope
from src.schemas import PolicyOutput
from src.tools.rag_tool import policy_rag_tool

NODE = "policy_retrieval_agent"


def policy_question(state: dict) -> tuple[str, list[str] | None]:
    """Question + optional doc filter, built only from trusted structured fields."""
    acct = state.get("account_summary") or {}
    plan = acct.get("plan") or {}
    intent = state.get("intent")
    if intent == "cancellation":
        return (f"What retention offers may be made to a {plan.get('tier', '')}-tier "
                f"{plan.get('plan_type', '')} customer with {acct.get('churn_risk_label', 'unknown')} churn "
                f"risk and {acct.get('tenure_months', 0)} months tenure who wants to cancel, what are the "
                f"discount limits, and when is human approval required?",
                ["POL-RET-001", "POL-RET-002", "POL-RET-003", "POL-RET-004", "POL-CAN-001"])
    if intent == "billing_query":
        kinds = {f["type"] for f in state.get("billing_findings") or []}
        if "duplicate_charge" in kinds:
            return ("How is a duplicate charge on an invoice disputed and credited, and does the credit "
                    "need approval?", ["POL-BIL-001", "POL-RET-003"])
        return ("How should invoice line items, taxes and postpaid data overage charges be explained, "
                "and how can future overage be reduced?", ["POL-BIL-002", "POL-PLN-001"])
    if intent == "complaint":
        return ("What are the complaint severity SLAs and when must repeat complaints be escalated?",
                ["POL-CMP-001"])
    if intent == "plan_change":
        return ("What are the rules for plan upgrades and downgrades and how should a plan be "
                "recommended based on usage?", ["POL-PLN-001", "POL-BIL-002"])
    return ("How should out-of-scope or ambiguous requests be handled?", ["POL-GOV-001"])


async def policy_retrieval_agent(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        state = view(NODE, state)  # isolate: intent, plan/risk facts, billing findings; no customer text
        question, doc_ids = policy_question(state)
        out = await policy_rag_tool.ainvoke({"question": question, "top_k": 4, "doc_ids": doc_ids})
        errors = [] if out["status"] != "error" else [error_entry(NODE, "policy_rag", "; ".join(out["notes"]))]
        return PolicyOutput(
            policy_question=question, policy_answer=out["answer"], policy_citations=out["citations"],
            engine_log=[engine_entry(NODE, out["grader"], f"status={out['status']} attempts={out['attempts']}"
                                     + (" degraded" if out["degraded"] else ""))],
            errors=errors,
            scratchpad=scratch(NODE, f"q={question[:120]}", f"status={out['status']} queries={len(out['queries'])}")["scratchpad"],
        ).update()
