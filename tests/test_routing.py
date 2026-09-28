"""Routing logic: conditional edges pick the right worker for given states (no LLM needed).

`route_from_supervisor`, `route_after_resolution`, `route_after_clarify` and `decide_outcome` are pure
functions of state. The supervisor/intent nodes are also run with STUB LLMs to show a model proposal
is only accepted when the pure routing rules allow it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_server.schemas import CheckOfferEligibilityInput
from mcp_server.server import evaluate_offer
from src.agents import intent_agent as intent_mod
from src.agents import supervisor as sup_mod
from src.agents.common import Deps
from src.agents.human_loop import route_after_clarify
from src.agents.resolution_agent import decide_outcome, route_after_resolution
from src.agents.supervisor import allowed_next, route_from_supervisor
from src.schemas import IntentClassification, SupervisorDecision

ACCOUNT = {"plan": {"tier": "premium", "plan_type": "postpaid"}, "churn_risk_label": "high",
           "tenure_months": 48, "complaints_90d": 0}
DUPLICATE = [{"type": "duplicate_charge", "credit_due": 15.0, "disputable": True}]


def st(**kw) -> dict:
    return {"step_count": 0, "max_steps": 8, **kw}


@pytest.mark.parametrize("state,expected", [
    (st(), "intent_agent"),
    (st(intent="billing_query", intent_confidence=0.8), "account_agent"),
    (st(intent="billing_query", intent_confidence=0.8, account_summary=ACCOUNT), "policy_retrieval_agent"),
    (st(intent="billing_query", intent_confidence=0.8, account_summary=ACCOUNT, policy_done=True), "resolution_agent"),
    (st(intent="billing_query", intent_confidence=0.8, account_summary=ACCOUNT, policy_done=True,
        billing_findings=DUPLICATE), "retention_offer_agent"),
    (st(intent="cancellation", intent_confidence=0.9, account_summary=ACCOUNT, policy_done=True),
     "retention_offer_agent"),
    (st(intent="cancellation", intent_confidence=0.9, account_summary=ACCOUNT, policy_done=True, offer_done=True),
     "resolution_agent"),
    (st(intent="plan_change", intent_confidence=0.9, account_summary=ACCOUNT, policy_done=True), "resolution_agent"),
    (st(intent="complaint", intent_confidence=0.9, account_summary=ACCOUNT, policy_done=True), "resolution_agent"),
], ids=["no-intent", "needs-account", "needs-policy", "billing-resolve", "billing-dup-credit",
        "cancel-offer", "cancel-resolve", "plan-resolve", "complaint-resolve"])
def test_route_picks_the_right_worker(state, expected):
    assert route_from_supervisor(state) == expected


@pytest.mark.parametrize("state", [
    st(intent="ambiguous", intent_confidence=0.3),
    st(intent="plan_change", intent_confidence=0.4),       # low confidence
    st(intent="out_of_scope", intent_confidence=0.9),
], ids=["ambiguous", "low-confidence", "out-of-scope"])
def test_ambiguous_low_confidence_and_out_of_scope_go_to_clarify(state):
    assert route_from_supervisor(state) == "clarify"


def test_clarify_escalates_when_still_unclear():
    assert route_after_clarify({"resolution": {"outcome": "escalate"}}) == "escalate"
    assert route_after_clarify({"resolution": {"outcome": "clarify"}}) == "memory_writer"


@pytest.mark.parametrize("state,expected", [
    (st(guard_blocked=True, intent=None), "resolution_agent"),
    (st(step_count=8, intent="billing_query"), "escalate"),
    (st(intent="billing_query", intent_confidence=0.8, account_failed=True), "escalate"),
], ids=["guard-blocked", "step-limit", "account-tool-failed"])
def test_guards_and_failures(state, expected):
    assert route_from_supervisor(state) == expected


def test_model_proposal_is_used_only_when_allowed():
    base = st(intent="cancellation", intent_confidence=0.9, account_summary=ACCOUNT)
    assert allowed_next(base) == ["policy_retrieval_agent", "retention_offer_agent"]
    assert route_from_supervisor({**base, "next_worker": "retention_offer_agent"}) == "retention_offer_agent"
    # resolution before policy is not allowed: priority fallback instead
    assert route_from_supervisor({**base, "next_worker": "resolution_agent"}) == "policy_retrieval_agent"


def _offer(needs_approval: bool) -> dict:
    return {"offer_type": "discount_pct", "value": 20, "months": 6, "needs_approval": needs_approval,
            "policy_refs": ["POL-RET-003 §1.2"], "reasons": []}


def test_over_threshold_offer_routes_to_human_approval():
    state = {"intent": "cancellation", "account_summary": ACCOUNT, "offer_proposal": _offer(True),
             "offer_decision": {"checks": [], "blocked": []}}
    decided = decide_outcome(state)
    assert decided["outcome"] == "offer_pending_approval"
    assert route_after_resolution({"resolution": decided, "needs_human_approval": True}) == "human_approval"


def test_within_threshold_offer_skips_approval():
    state = {"intent": "cancellation", "account_summary": ACCOUNT, "offer_proposal": _offer(False),
             "offer_decision": {"checks": [], "blocked": []}}
    decided = decide_outcome(state)
    assert decided["outcome"] == "offer"
    assert route_after_resolution({"resolution": decided, "needs_human_approval": False}) == "memory_writer"


@pytest.mark.parametrize("pct,months,decision", [(20, 6, "needs_approval"), (10, 5, "allowed"), (50, 6, "blocked")])
def test_threshold_decision_comes_from_policy(pct, months, decision):
    customer = {"churn_risk_label": "high", "monthly_price": 85.0, "fraud_flag": 0, "late_payments_12m": 0,
                "tenure_months": 48, "last_retention_offer_date": None, "tier": "premium", "plan_type": "postpaid",
                "contract_end_date": "2026-10-15"}
    inp = CheckOfferEligibilityInput(customer_id="CUST-000397", offer_type="discount_pct", value=pct, months=months,
                                     cancellation_intent=True)
    assert evaluate_offer(customer, inp).decision == decision


@pytest.mark.parametrize("outcome,route", [("escalate", "escalate"), ("resolve", "memory_writer"),
                                           ("refuse", "memory_writer")])
def test_route_after_resolution(outcome, route):
    assert route_after_resolution({"resolution": {"outcome": outcome}}) == route


# --- nodes with stub LLMs ----------------------------------------------------------

def _runtime(use_llm: bool = True):
    return SimpleNamespace(context=Deps(mcp=None, use_llm=use_llm))


@pytest.mark.asyncio
async def test_supervisor_rejects_disallowed_llm_proposal(monkeypatch):
    async def stub(schema, messages, **kw):
        return SupervisorDecision(next_worker="resolution_agent", reason="stub wants to skip policy")
    monkeypatch.setattr(sup_mod, "structured_call", stub)
    state = st(intent="cancellation", intent_confidence=0.9, account_summary=ACCOUNT)
    out = await sup_mod.supervisor(state, _runtime())
    assert out["next_worker"] == "policy_retrieval_agent" and "not allowed" in out["supervisor_reason"]


@pytest.mark.asyncio
async def test_supervisor_accepts_allowed_llm_proposal(monkeypatch):
    async def stub(schema, messages, **kw):
        return SupervisorDecision(next_worker="retention_offer_agent", reason="check offers first")
    monkeypatch.setattr(sup_mod, "structured_call", stub)
    state = st(intent="cancellation", intent_confidence=0.9, account_summary=ACCOUNT)
    out = await sup_mod.supervisor(state, _runtime())
    assert out["next_worker"] == "retention_offer_agent" and out["engine_log"][0]["engine"] == "gemini"


@pytest.mark.asyncio
async def test_low_confidence_llm_intent_routes_to_clarify(monkeypatch):
    async def stub(schema, messages, **kw):
        return IntentClassification(intent="billing_query", confidence=0.35, rationale="unsure",
                                    cancellation_intent=False)
    monkeypatch.setattr(intent_mod, "structured_call", stub)
    from langchain_core.messages import HumanMessage
    out = await intent_mod.intent_agent({"messages": [HumanMessage("something about money?")]}, _runtime())
    assert out["intent_confidence"] == 0.35
    assert route_from_supervisor(st(**{k: out[k] for k in ("intent", "intent_confidence")})) == "clarify"
