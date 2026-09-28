"""Pydantic models for node outputs (structured output at every node boundary) and LLM calls.

Every graph node builds one of the `*Output` models (validated on construction) and returns
`model.update()`, the dict of state keys it writes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["billing_query", "plan_change", "complaint", "cancellation", "ambiguous", "out_of_scope"]
Worker = Literal["intent_agent", "account_agent", "policy_retrieval_agent", "retention_offer_agent",
                 "resolution_agent", "clarify", "escalate"]
Outcome = Literal["resolve", "offer", "offer_pending_approval", "escalate", "clarify", "decline", "refuse"]
Engine = Literal["gemini", "rules", "heuristic", "tool", "human"]


class NodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scratchpad: dict[str, list[str]] | None = Field(
        default=None, description="Notes for this node's own scratchpad lane (context write)")

    def update(self) -> dict[str, Any]:
        out = self.model_dump(exclude_none=False)
        if out.get("scratchpad") is None:
            out.pop("scratchpad", None)
        return out


# --- LLM structured outputs ----------------------------------------------------

class RequestedOffer(BaseModel):
    offer_type: Literal["discount_pct", "credit", "data_boost", "plan_upgrade", "fee_waiver"]
    value: float = Field(ge=0, description="percent for discount_pct, USD for credit/fee_waiver")
    months: int = Field(default=1, ge=1, le=36)


class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=300)
    cancellation_intent: bool = Field(description="Customer explicitly says they want to cancel/leave")
    requested_offer: RequestedOffer | None = Field(
        default=None, description="A specific discount/credit the customer explicitly asked for")
    mentions_regulator: bool = False


class SupervisorDecision(BaseModel):
    next_worker: Worker
    reason: str = Field(max_length=300)


class ResolutionMessage(BaseModel):
    customer_message: str = Field(max_length=1500)
    citations: list[str] = Field(description="Policy clause IDs cited, e.g. 'POL-RET-003 §1.2'")


# --- Node outputs ----------------------------------------------------------------

class InputGuardOutput(NodeOutput):
    quarantined_input: str
    input_flags: list[str]
    guard_blocked: bool
    guard_reason: str | None
    # per-turn resets
    intent: None = None
    intent_confidence: float = 0.0
    requested_offer: None = None
    cancellation_intent: bool = False
    mentions_regulator: bool = False
    next_worker: None = None
    policy_done: bool = False
    offer_done: bool = False
    policy_citations: list = []
    policy_answer: None = None
    offer_proposal: None = None
    offer_decision: None = None
    resolution: None = None
    needs_human_approval: bool = False
    approval_request: None = None
    approval: None = None
    final_response: None = None
    step_count: int = 0
    account_summary: None = None
    billing: None = None
    billing_findings: list = []
    account_failed: bool = False


class SupervisorOutput(NodeOutput):
    next_worker: Worker
    supervisor_reason: str
    step_count: int
    engine_log: list[dict]


class IntentOutput(NodeOutput):
    intent: Intent
    intent_confidence: float
    intent_rationale: str
    cancellation_intent: bool
    requested_offer: dict | None
    mentions_regulator: bool
    engine_log: list[dict]


class AccountOutput(NodeOutput):
    account_summary: dict | None
    billing: dict | None
    billing_findings: list[dict]
    account_failed: bool = False
    errors: list[dict] = []


class PolicyOutput(NodeOutput):
    policy_done: bool = True
    policy_question: str
    policy_answer: str
    policy_citations: list[dict]
    engine_log: list[dict]
    errors: list[dict] = []


class OfferOutput(NodeOutput):
    offer_done: bool = True
    offer_proposal: dict | None
    offer_decision: dict
    needs_human_approval: bool
    errors: list[dict] = []


class ResolutionOutput(NodeOutput):
    resolution: dict
    engine_log: list[dict]


class ApprovalRequestOutput(NodeOutput):
    approval_request: dict


class ApprovalOutput(NodeOutput):
    approval: dict
    resolution: dict
    needs_human_approval: bool


class HandoffOutput(NodeOutput):
    resolution: dict
    clarify_count: int | None = None
    errors: list[dict] = []

    def update(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ContextOutput(NodeOutput):
    summary: str | None
    messages: list = Field(default_factory=list, description="RemoveMessage entries for compressed history")
    session_facts: list[dict]
    memories: list[dict]
    context_stats: dict
    engine_log: list[dict]


class MemoryWriteOutput(NodeOutput):
    memory_written: list[dict]


class OutputGuardOutput(NodeOutput):
    final_response: str
    output_flags: list[str]
    output_risk: dict
    messages: list
