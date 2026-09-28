"""Supervisor: chooses the next worker.

Design: the LLM proposes, pure functions dispose.
- `allowed_next(state)` (pure) lists the workers permitted from this state: prerequisites (intent ->
  account -> policy before resolution), the loop guard (max steps), guard blocks, and routing of
  ambiguous / out-of-scope intents to clarify.
- The supervisor node asks Gemini (structured output: SupervisorDecision) to pick among them only
  when more than one is allowed; otherwise the single option is forced. Without Gemini, a fixed
  priority order picks.
- `route_from_supervisor(state)` (pure) is the conditional edge: it returns the proposal if it is
  allowed, else the priority choice. Unit-testable without any LLM.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from src.agents.common import Deps, engine_entry
from src.llm import structured_call
from src.resilience import ExternalCallFailed
from src.run_context import agent_scope
from src.schemas import SupervisorDecision, SupervisorOutput

NODE = "supervisor"
MIN_CONFIDENCE = 0.6
DEFAULT_MAX_STEPS = 12
PRIORITY = ["escalate", "clarify", "intent_agent", "account_agent", "policy_retrieval_agent",
            "retention_offer_agent", "resolution_agent"]


def offer_relevant(state: dict) -> bool:
    """Retention offers for cancellations; credits for verified billing errors."""
    if state.get("intent") == "cancellation":
        return True
    return state.get("intent") == "billing_query" and bool(state.get("billing_findings"))


def allowed_next(state: dict) -> list[str]:
    max_steps = state.get("max_steps") or DEFAULT_MAX_STEPS
    if state.get("step_count", 0) >= max_steps:
        return ["escalate"]
    if state.get("guard_blocked"):
        return ["resolution_agent"]
    intent = state.get("intent")
    if intent is None:
        return ["intent_agent"]
    if intent in ("ambiguous", "out_of_scope") or state.get("intent_confidence", 0) < MIN_CONFIDENCE:
        return ["clarify"]
    if state.get("account_summary") is None:
        return ["escalate"] if state.get("account_failed") else ["account_agent"]
    options = []
    if not state.get("policy_done"):
        options.append("policy_retrieval_agent")
    if offer_relevant(state) and not state.get("offer_done"):
        options.append("retention_offer_agent")
    if state.get("policy_done"):
        options.append("resolution_agent")
    return options


def priority_choice(options: list[str]) -> str:
    return min(options, key=PRIORITY.index)


def route_from_supervisor(state: dict) -> str:
    """Conditional edge out of the supervisor (pure function of state)."""
    options = allowed_next(state)
    proposal = state.get("next_worker")
    return proposal if proposal in options else priority_choice(options)


_SYSTEM = (
    "You are the supervisor of a telecom customer-care copilot. Choose the next worker from the "
    "allowed list only. Workers: policy_retrieval_agent (look up the applicable policy), "
    "retention_offer_agent (check retention offers or billing credits against policy), "
    "resolution_agent (draft the resolution; requires policy to have been retrieved). Gather what "
    "the case needs before resolving: at-risk cancellations and verified billing errors need an offer "
    "or credit check."
)


async def supervisor(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        step = state.get("step_count", 0) + 1
        view = {**state, "step_count": step}  # same count the routing edge will see
        options = allowed_next(view)
        if len(options) == 1:
            choice, reason, engine = options[0], "only allowed option", "rules"
        elif runtime.context.use_llm:
            facts = {k: state.get(k) for k in ("intent", "intent_confidence", "cancellation_intent",
                                                "policy_done", "offer_done", "billing_findings")}
            facts["account"] = _account_brief(state.get("account_summary"))
            try:
                d = await structured_call(SupervisorDecision, [SystemMessage(_SYSTEM), HumanMessage(
                    f"State: {json.dumps(facts, default=str)}\nAllowed: {options}\nPick the next worker.")],
                    what="supervisor.route")
                choice, reason, engine = d.next_worker, d.reason, "gemini"
            except ExternalCallFailed as exc:
                choice, reason, engine = priority_choice(options), f"gemini failed ({exc.reason})", "rules"
        else:
            choice, reason, engine = priority_choice(options), "priority order (no GOOGLE_API_KEY)", "rules"
        if choice not in options:
            reason, choice = f"proposal '{choice}' not allowed; priority fallback", priority_choice(options)
        return SupervisorOutput(next_worker=choice, supervisor_reason=reason, step_count=step,
                                engine_log=[engine_entry(NODE, engine, f"{choice}: {reason}")]).update()


def _account_brief(acct: dict | None) -> dict | None:
    if not acct:
        return None
    return {k: acct.get(k) for k in ("tenure_months", "churn_risk_label", "complaints_90d")} | {
        "tier": acct.get("plan", {}).get("tier")}
