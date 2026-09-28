"""Output-risk classification (docs/output-risk.md).

Every resolution is assigned a tier from structured state (not from the drafted text):
  low     informational: bill/plan/policy explanations, clarifying questions, out-of-scope declines
  medium  changes the customer's situation within policy and within the auto-approval threshold:
          plan recommendations, complaints logged, credits/discounts at or below the threshold,
          non-monetary offers
  high    monetary value above the threshold or any fee waiver, retention during cancellation,
          escalations to humans (fraud, safety, regulator, repeat complaint), refusals of
          injection / cross-customer requests
High-tier outputs are gated: above-threshold offers must pass the human approval interrupt
(src/agents/human_loop.py::human_approval) before being described as applied; prohibited requests are
refused; escalations hand off to a person. `gate_satisfied` reports whether that happened.
"""

from __future__ import annotations

from typing import Literal

Tier = Literal["low", "medium", "high"]


def classify_output_risk(state: dict) -> dict:
    res = state.get("resolution") or {}
    outcome = res.get("outcome")
    proposal = state.get("offer_proposal") or {}
    approval = state.get("approval") or {}
    intent = state.get("intent")

    if outcome in ("refuse", "escalate") or state.get("guard_blocked"):
        return {"tier": "high", "reason": f"{outcome}: {res.get('reason')}", "gate": "refusal_or_human_handoff",
                "gate_satisfied": True}
    if proposal and (proposal.get("needs_approval") or proposal.get("offer_type") == "fee_waiver"):
        return {"tier": "high", "reason": "offer above the auto-approval threshold (or fee waiver)",
                "gate": "human_approval_interrupt", "gate_satisfied": bool(approval)}
    if intent == "cancellation":
        return {"tier": "high", "reason": "retention attempt during a cancellation", "gate": "policy_check",
                "gate_satisfied": state.get("offer_decision") is not None}
    if proposal or intent in ("plan_change", "complaint"):
        return {"tier": "medium", "reason": "within-policy change or commitment", "gate": "policy_check_and_audit",
                "gate_satisfied": True}
    return {"tier": "low", "reason": "informational", "gate": "output_guard", "gate_satisfied": True}
