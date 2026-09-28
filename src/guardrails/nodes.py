"""Graph I/O nodes wrapping the input and output guardrails."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from src.agents.common import Deps
from src.audit.audit_middleware import audit
from src.guardrails.input_guard import check_input
from src.guardrails.output_guard import check_output
from src.guardrails.output_risk import classify_output_risk
from src.observability.tracing import guardrail_span
from src.run_context import agent_scope
from src.schemas import InputGuardOutput, OutputGuardOutput


async def input_guard(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope("input_guard"), guardrail_span("guardrail.input") as span:
        raw = state.get("pending_input") or ""
        g = check_input(raw, state["customer_id"])
        flags = list(dict.fromkeys((state.get("ingest_flags") or []) + g.flags))
        span.set_attribute("guardrail.decision", "blocked" if g.blocked else "sanitized" if flags else "passed")
        span.set_attribute("guardrail.flags", flags)
        span.set_attribute("guardrail.policy_ref", g.policy_ref or "")
        if g.blocked:
            audit("guardrail_block", "blocked", tool="input_guard", reason=g.reason, policy_ref=g.policy_ref,
                  details={"flags": flags})
        elif flags:
            audit("input_sanitized", "sanitized", tool="input_guard", reason=", ".join(flags),
                  details={"flags": flags})
        update = InputGuardOutput(quarantined_input=g.quarantined, input_flags=flags,
                                  guard_blocked=g.blocked, guard_reason=g.reason).update()
        update["messages"] = [HumanMessage(content=g.sanitized)]  # only sanitized text is stored
        update["pending_input"] = None
        update["max_steps"] = runtime.context.max_steps
        return update


async def output_guard(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope("output_guard"), guardrail_span("guardrail.output") as span:
        res = state.get("resolution") or {}
        proposal = state.get("offer_proposal") or {}
        approved = (state.get("approval") or {}).get("approved", False)
        max_pct = proposal.get("value") if proposal.get("offer_type") == "discount_pct" else None
        max_credit = proposal.get("value") if proposal.get("offer_type") == "credit" else None
        g = check_output(
            res.get("customer_message", ""),
            authenticated_customer_id=state["customer_id"],
            max_discount_pct=max_pct,
            max_credit_usd=max_credit,
            pending_approval=bool(proposal.get("needs_approval")) and not approved,
            safe_fallback="Thanks for your patience. A colleague will review your request and follow up "
                          "with you shortly [POL-GOV-001 §2.2].",
        )
        span.set_attribute("guardrail.decision", "replaced" if g.replaced else "sanitized" if g.flags else "passed")
        span.set_attribute("guardrail.flags", g.flags)
        if g.flags:
            audit("output_guard_intervention", "replaced" if g.replaced else "sanitized", tool="output_guard",
                  reason=", ".join(g.flags), policy_ref=g.policy_ref,
                  details={"flags": g.flags, "pii_entities": [f["entity"] for f in g.pii_findings]})
        risk = classify_output_risk(state)
        span.set_attribute("copilot.output_risk_tier", risk["tier"])
        audit("contact_resolved", res.get("outcome", "unknown"), reason=res.get("reason"),
              policy_ref=res.get("citations"),
              details={"citations": res.get("citations"), "ticket_id": res.get("ticket_id"),
                       "output_flags": g.flags, "risk_tier": risk["tier"], "risk_gate": risk["gate"],
                       "gate_satisfied": risk["gate_satisfied"]})
        return OutputGuardOutput(final_response=g.text, output_flags=g.flags, output_risk=risk,
                                 messages=[AIMessage(content=g.text)]).update()
