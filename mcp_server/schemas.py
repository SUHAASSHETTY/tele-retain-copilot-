"""Pydantic input/output schemas for the telecom MCP server tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.policy_limits import OFFER_TYPES

CustomerId = Field(pattern=r"^CUST-\d{6}$", description="Synthetic customer ID, e.g. CUST-000123")
OfferType = Literal[OFFER_TYPES]  # type: ignore[valid-type]


class ToolError(BaseModel):
    code: Literal[
        "AUTH_MISSING", "AUTH_INVALID", "AUTH_UNAVAILABLE", "AUTHZ_DENIED",
        "VALIDATION_ERROR", "NOT_FOUND", "INTERNAL_ERROR",
    ]
    message: str
    policy_ref: str | None = None


class _Result(BaseModel):
    ok: bool
    error: ToolError | None = None


# --- Inputs --------------------------------------------------------------------

class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = CustomerId


class GetAccountInput(_Input):
    pass


class GetBillingHistoryInput(_Input):
    months: int = Field(default=3, ge=1, le=3, description="Number of most recent invoices (1-3)")


class CheckOfferEligibilityInput(_Input):
    offer_type: OfferType
    value: float = Field(ge=0, description=(
        "discount_pct: percent off the plan price; credit / fee_waiver: USD; "
        "data_boost: GB per month; plan_upgrade: ignored (use 0)"))
    months: int = Field(default=1, ge=1, le=24, description="Duration in billing months")
    cancellation_intent: bool = Field(
        default=False, description="Customer explicitly said they intend to cancel (POL-RET-001 §1.3)")
    purpose: Literal["retention", "billing_adjustment"] = Field(
        default="retention",
        description="billing_adjustment = credit for a verified billing error (POL-BIL-001 §2.1); "
                    "skips retention eligibility but still applies the hard cap and approval threshold")


class CreateEscalationTicketInput(_Input):
    reason: Literal["approval_required", "repeat_complaint", "fraud_or_collections", "out_of_policy",
                    "ambiguous_request", "regulator_mention", "tool_failure", "safety_threat", "other"]
    summary: str = Field(min_length=5, max_length=500)
    priority: Literal["P1", "P2", "P3"] = "P3"


# --- Outputs -------------------------------------------------------------------

class PlanSummary(BaseModel):
    plan_id: str
    name: str
    plan_type: Literal["prepaid", "postpaid"]
    tier: Literal["entry", "mid", "premium"]
    monthly_price: float
    data_cap_gb: int | None = Field(description="None means unlimited")
    contract_months: int


class AccountSummary(BaseModel):
    customer_ref: str = Field(description="Masked customer ID")
    account_ref: str = Field(description="Masked account number")
    first_initial: str
    email_masked: str
    phone_masked: str
    region: str
    plan: PlanSummary
    tenure_months: int
    contract_end_date: str | None
    auto_pay: bool
    data_used_gb_last_cycle: float
    avg_data_used_gb_3m: float
    churn_risk_label: Literal["low", "medium", "high"]
    churn_risk_score: float
    complaints_90d: int
    open_complaints: int
    last_retention_offer_date: str | None


class GetAccountOutput(_Result):
    account: AccountSummary | None = None


class LineItem(BaseModel):
    item: str
    amount: float


class Invoice(BaseModel):
    invoice_id: str
    period_start: str
    issue_date: str
    status: Literal["paid", "due"]
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float
    disputable: bool = Field(description="Within the POL-BIL-001 §1.1 dispute window")


class GetBillingHistoryOutput(_Result):
    customer_ref: str | None = None
    invoices: list[Invoice] = []


class EligibilityDecision(BaseModel):
    decision: Literal["allowed", "needs_approval", "blocked"]
    allowed: bool = Field(description="True if the offer may be presented (possibly pending approval)")
    needs_approval: bool
    offer_type: str
    offer_value_usd: float
    max_allowed: float | None = Field(
        default=None, description="Highest permitted value for this offer type for this customer")
    effective_risk: Literal["low", "medium", "high"]
    reasons: list[str]
    policy_refs: list[str]


class CheckOfferEligibilityOutput(_Result):
    eligibility: EligibilityDecision | None = None


class CreateEscalationTicketOutput(_Result):
    ticket_id: str | None = None
    queue: str | None = None
    priority: str | None = None
