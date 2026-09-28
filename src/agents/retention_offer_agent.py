"""Retention-offer worker: selects a compliant offer (or billing credit) and NEVER proposes anything
the MCP `check_offer_eligibility` tool did not allow.

Order of checks:
1. Billing contacts with a verified duplicate charge -> credit for the duplicate
   (purpose=billing_adjustment).
2. Cancellations -> the customer's explicit request first (e.g. "half off"); if blocked it is
   recorded with its policy reference. Then the candidate ladder: tier-maximum loyalty discount
   for 6 months at the customer's effective risk -> non-monetary data boost. The first candidate
   the tool allows becomes the proposal; `needs_approval` from the tool routes it to the human gate.
Every eligibility decision is written to the audit trail.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from src.agents.common import Deps, error_entry
from src.audit.audit_middleware import audit
from src.context.isolate import view
from src.context.write import scratch
from src.policy_limits import (
    DATA_BOOST_MAX_GB,
    DATA_BOOST_MAX_MONTHS,
    MAX_DISCOUNT_PCT_BY_TIER,
    MAX_OFFER_MONTHS,
)
from src.run_context import agent_scope
from src.schemas import OfferOutput

NODE = "retention_offer_agent"


def candidate_ladder(state: dict) -> list[dict]:
    acct = state["account_summary"]
    tier = acct["plan"]["tier"]
    risk = acct["churn_risk_label"]
    if state.get("cancellation_intent") and risk == "low":
        risk = "medium"
    tier_max = MAX_DISCOUNT_PCT_BY_TIER[tier]
    pct = tier_max if risk == "high" else tier_max // 2
    ladder = []
    if pct > 0:
        ladder.append({"offer_type": "discount_pct", "value": pct, "months": MAX_OFFER_MONTHS})
    ladder.append({"offer_type": "data_boost", "value": DATA_BOOST_MAX_GB, "months": DATA_BOOST_MAX_MONTHS})
    return ladder


async def _check(mcp, cid: str, offer: dict, *, cancel: bool, purpose: str, source: str) -> dict:
    res = await mcp.call("check_offer_eligibility", customer_id=cid, offer_type=offer["offer_type"],
                         value=offer["value"], months=offer.get("months", 1),
                         cancellation_intent=cancel, purpose=purpose)
    if not res.get("ok"):
        raise RuntimeError(res["error"]["code"])
    e = res["eligibility"]
    record = {**offer, "source": source, "purpose": purpose, **e}
    audit("offer_eligibility_check", e["decision"], tool="check_offer_eligibility",
          reason="; ".join(e["reasons"]), policy_ref=e["policy_refs"],
          details={"source": source, "offer_type": offer["offer_type"], "value": offer["value"],
                   "months": offer.get("months", 1), "offer_value_usd": e["offer_value_usd"],
                   "policy_refs": e["policy_refs"], "reasons": e["reasons"]})
    return record


async def retention_offer_agent(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        state = view(NODE, state)  # isolate: account facts, intent, requested offer; no history
        mcp, cid = runtime.context.mcp, state["customer_id"]
        cancel = bool(state.get("cancellation_intent"))
        checks: list[dict] = []
        proposal = None
        try:
            if state.get("intent") == "billing_query":
                for f in state.get("billing_findings") or []:
                    if f["type"] == "duplicate_charge" and f["disputable"]:
                        c = await _check(mcp, cid, {"offer_type": "credit", "value": f["credit_due"], "months": 1},
                                         cancel=False, purpose="billing_adjustment", source="duplicate_charge")
                        checks.append(c)
                        if c["allowed"]:
                            proposal = c
                        break
            else:
                requested = state.get("requested_offer")
                if requested:
                    c = await _check(mcp, cid, requested, cancel=cancel, purpose="retention",
                                     source="customer_request")
                    checks.append(c)
                    if c["allowed"]:
                        proposal = c
                if proposal is None:
                    for cand in candidate_ladder(state):
                        c = await _check(mcp, cid, cand, cancel=cancel, purpose="retention", source="ladder")
                        checks.append(c)
                        if c["allowed"]:
                            proposal = c
                            break
        except Exception as exc:
            audit("offer_selection_failed", "escalate", tool="check_offer_eligibility",
                  reason=str(exc) or type(exc).__name__)
            return OfferOutput(offer_proposal=None, needs_human_approval=False,
                               offer_decision={"checks": checks, "status": "tool_failure"},
                               errors=[error_entry(NODE, "check_offer_eligibility", str(exc) or type(exc).__name__)]
                               ).update()

        blocked = [c for c in checks if c["decision"] == "blocked"]
        decision = {
            "checks": checks,
            "blocked": [{"offer_type": b["offer_type"], "value": b["value"], "source": b["source"],
                         "policy_refs": b["policy_refs"], "reasons": b["reasons"]} for b in blocked],
            "status": "proposed" if proposal else "no_eligible_offer",
        }
        needs_approval = bool(proposal and proposal["needs_approval"])
        for b in blocked:
            audit("offer_blocked", "blocked", tool="check_offer_eligibility", reason="; ".join(b["reasons"]),
                  policy_ref=b["policy_refs"], details={"source": b["source"], "offer_type": b["offer_type"],
                                                        "value": b["value"], "months": b.get("months", 1)})
        if proposal:
            audit("offer_proposed", "pending_approval" if needs_approval else "within_limits",
                  tool="check_offer_eligibility", reason="; ".join(proposal["reasons"]),
                  policy_ref=proposal["policy_refs"],
                  details={k: proposal[k] for k in ("offer_type", "value", "months", "offer_value_usd", "source")})
        notes = [f"{c['source']}:{c['offer_type']}={c['value']:g} -> {c['decision']}" for c in checks]
        return OfferOutput(offer_proposal=proposal, offer_decision=decision, needs_human_approval=needs_approval,
                           scratchpad=scratch(NODE, *notes)["scratchpad"]).update()
