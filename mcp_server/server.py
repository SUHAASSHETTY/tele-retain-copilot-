"""Telecom account MCP server (FastMCP, stdio).

Tools (every customer-scoped tool is authorized against the caller's session token):
  get_account               masked account + plan summary
  get_billing_history       most recent invoices with line items
  check_offer_eligibility   allowed / needs_approval / blocked + policy_refs
  create_escalation_ticket  open a ticket for a human agent
Resources:
  policy://catalog          policy docs and their clause IDs
  plans://catalog           plans and retention-offer catalogue

Errors are returned as structured `{"ok": false, "error": {...}}` results, never stack traces.
Run: python -m mcp_server.server   (stdio; stdout is reserved for the MCP protocol)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import date, timedelta
from typing import Annotated, Any, Callable

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from mcp_server.auth import AuthError, verify_token
from mcp_server.schemas import (
    AccountSummary,
    CheckOfferEligibilityInput,
    CheckOfferEligibilityOutput,
    CreateEscalationTicketInput,
    CreateEscalationTicketOutput,
    EligibilityDecision,
    GetAccountInput,
    GetAccountOutput,
    GetBillingHistoryInput,
    GetBillingHistoryOutput,
    Invoice,
    PlanSummary,
    ToolError,
)
from src.config import POLICY_CORPUS_DIR, RUNTIME_DIR, SYNTHETIC_DIR
from src.guardrails.pii import mask, mask_customer_id
from src.policy_limits import (
    ABSOLUTE_MAX_DISCOUNT_PCT,
    APPROVAL_THRESHOLD,
    AS_OF,
    COLLECTIONS_LATE_PAYMENTS,
    CREDIT_HARD_CAP,
    DATA_BOOST_MAX_GB,
    DATA_BOOST_MAX_MONTHS,
    DISPUTE_WINDOW_DAYS,
    ETF_PCT_OF_PRICE,
    MAX_DISCOUNT_PCT_BY_TIER,
    MAX_OFFER_MONTHS,
    MIN_TENURE_FOR_DISCOUNT,
    OFFER_COOLDOWN_MONTHS,
    PLAN_UPGRADE_MAX_MONTHS,
    POLICY_REFS,
    TIER_ORDER,
)

DB_PATH = SYNTHETIC_DIR / "telecom.db"
TICKETS_DB = RUNTIME_DIR / "escalations.db"

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="[mcp_server] %(message)s")
log = logging.getLogger("mcp_server")

mcp = FastMCP(
    "telecom-account",
    instructions="Synthetic telecom account, billing and retention-offer tools. "
                 "All customer data is synthetic; identifiers in responses are masked.",
)

AuthParam = Annotated[str, Field(description="Session token. Injected by the client; never set by the model.")]
CustomerIdParam = Annotated[str, Field(description="Customer ID of the authenticated caller, e.g. CUST-000123")]


# --- helpers -------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _error(output_cls, code: str, message: str, policy_ref: str | None = None) -> dict:
    return output_cls(ok=False, error=ToolError(code=code, message=message, policy_ref=policy_ref)).model_dump()


def _authorized_call(output_cls, input_cls, auth: str, fn: Callable[[Any], Any], **raw) -> dict:
    """Validate input, verify the session token, enforce account-holder-only access, run fn."""
    try:
        inp = input_cls(**raw)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _error(output_cls, "VALIDATION_ERROR", mask(detail, amounts=False))
    try:
        _session, authenticated = verify_token(auth)
    except AuthError as exc:
        return _error(output_cls, exc.code, exc.message)
    if inp.customer_id != authenticated:
        return _error(output_cls, "AUTHZ_DENIED",
                      "Access is limited to the authenticated account holder's own data.",
                      POLICY_REFS["account_holder_only"])
    try:
        return fn(inp)
    except Exception as exc:  # never leak internals to the caller
        log.error("tool failure %s: %s", output_cls.__name__, mask(repr(exc)))
        return _error(output_cls, "INTERNAL_ERROR", "The tool failed unexpectedly; escalate to a human agent.")


def _load_customer(con: sqlite3.Connection, customer_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT c.*, p.name AS plan_name, p.plan_type, p.tier, p.monthly_price, p.data_cap_gb, "
        "p.contract_months FROM customers c JOIN plans p USING (plan_id) WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()


def _months_between(earlier: date, later: date) -> float:
    return (later - earlier).days / 30.44


# --- tools ---------------------------------------------------------------------

@mcp.tool(description="Masked account and plan summary for the authenticated customer.")
def get_account(auth: AuthParam, customer_id: CustomerIdParam) -> GetAccountOutput:
    def run(inp: GetAccountInput) -> dict:
        with closing(_db()) as con:
            c = _load_customer(con, inp.customer_id)
            if c is None:
                return _error(GetAccountOutput, "NOT_FOUND", "No account found for the caller.")
            open_cmp = con.execute(
                "SELECT COUNT(*) FROM complaints WHERE customer_id = ? AND status = 'open'",
                (inp.customer_id,)).fetchone()[0]
        account = AccountSummary(
            customer_ref=mask_customer_id(c["customer_id"]),
            account_ref=mask(c["account_number"], amounts=False),
            first_initial=c["first_name"][0],
            email_masked=mask(c["email"], amounts=False),
            phone_masked=mask(c["phone"], amounts=False),
            region=c["region"],
            plan=PlanSummary(plan_id=c["plan_id"], name=c["plan_name"], plan_type=c["plan_type"],
                             tier=c["tier"], monthly_price=c["monthly_price"],
                             data_cap_gb=c["data_cap_gb"], contract_months=c["contract_months"]),
            tenure_months=c["tenure_months"],
            contract_end_date=c["contract_end_date"],
            auto_pay=bool(c["auto_pay"]),
            data_used_gb_last_cycle=c["data_used_gb"],
            avg_data_used_gb_3m=c["avg_data_used_gb_3m"],
            churn_risk_label=c["churn_risk_label"],
            churn_risk_score=c["churn_risk_score"],
            complaints_90d=c["complaints_90d"],
            open_complaints=open_cmp,
            last_retention_offer_date=c["last_retention_offer_date"],
        )
        return GetAccountOutput(ok=True, account=account).model_dump()

    return _authorized_call(GetAccountOutput, GetAccountInput, auth, run, customer_id=customer_id)


@mcp.tool(description="Most recent invoices (1-3) with line items for the authenticated customer.")
def get_billing_history(auth: AuthParam, customer_id: CustomerIdParam,
                        months: Annotated[int, Field(description="Number of recent invoices, 1-3")] = 3,
                        ) -> GetBillingHistoryOutput:
    def run(inp: GetBillingHistoryInput) -> dict:
        with closing(_db()) as con:
            rows = con.execute(
                "SELECT * FROM invoices WHERE customer_id = ? ORDER BY period_start DESC LIMIT ?",
                (inp.customer_id, inp.months)).fetchall()
        if not rows:
            return _error(GetBillingHistoryOutput, "NOT_FOUND", "No invoices found for the caller.")
        window_start = AS_OF - timedelta(days=DISPUTE_WINDOW_DAYS)
        invoices = [Invoice(
            invoice_id=r["invoice_id"], period_start=r["period_start"], issue_date=r["issue_date"],
            status=r["status"], line_items=json.loads(r["line_items_json"]), subtotal=r["subtotal"],
            tax=r["tax"], total=r["total"],
            disputable=date.fromisoformat(r["issue_date"]) >= window_start,
        ) for r in rows]
        return GetBillingHistoryOutput(ok=True, customer_ref=mask_customer_id(inp.customer_id),
                                       invoices=invoices).model_dump()

    return _authorized_call(GetBillingHistoryOutput, GetBillingHistoryInput, auth, run,
                            customer_id=customer_id, months=months)


def evaluate_offer(c: sqlite3.Row | dict, inp: CheckOfferEligibilityInput) -> EligibilityDecision:
    """Deterministic policy check for one proposed offer (POL-RET-001..004, POL-CAN-001, POL-BIL-001)."""
    reasons: list[str] = []
    refs: list[str] = []
    risk = c["churn_risk_label"]
    eff_risk = risk
    price = float(c["monthly_price"])

    def decide(decision: str, value: float = 0.0, max_allowed: float | None = None) -> EligibilityDecision:
        return EligibilityDecision(
            decision=decision, allowed=decision != "blocked", needs_approval=decision == "needs_approval",
            offer_type=inp.offer_type, offer_value_usd=round(value, 2), max_allowed=max_allowed,
            effective_risk=eff_risk, reasons=reasons, policy_refs=list(dict.fromkeys(refs)))

    def block(reason: str, *ref_keys: str, max_allowed: float | None = None) -> EligibilityDecision:
        reasons.append(reason)
        refs.extend(POLICY_REFS[k] for k in ref_keys)
        return decide("blocked", max_allowed=max_allowed)

    if inp.purpose == "billing_adjustment":
        if inp.offer_type != "credit":
            return block("Billing adjustments must be credits.", "dispute_credit")
        refs.append(POLICY_REFS["dispute_credit"])
        value = inp.value
        if value > CREDIT_HARD_CAP:
            return block(f"Credit exceeds the hard cap of ${CREDIT_HARD_CAP:.2f}.", "hard_cap",
                         max_allowed=CREDIT_HARD_CAP)
    else:
        if inp.cancellation_intent and risk == "low":
            eff_risk = "medium"
            reasons.append("Explicit cancellation intent raises effective risk to medium.")
            refs.append(POLICY_REFS["cancel_intent"])
        if c["fraud_flag"] or c["late_payments_12m"] >= COLLECTIONS_LATE_PAYMENTS:
            return block("Account is flagged for fraud or collections; no retention offers.", "fraud")

        if inp.offer_type == "data_boost":
            if inp.value > DATA_BOOST_MAX_GB or inp.months > DATA_BOOST_MAX_MONTHS:
                return block(f"Data boost is limited to {DATA_BOOST_MAX_GB} GB for "
                             f"{DATA_BOOST_MAX_MONTHS} months.", "tenure", max_allowed=DATA_BOOST_MAX_GB)
            reasons.append("Non-monetary offer; available to any eligible customer.")
            refs.extend([POLICY_REFS["tenure"], POLICY_REFS["auto_approve"]])
            return decide("allowed", 0.0, max_allowed=DATA_BOOST_MAX_GB)

        # Absolute prohibitions first: they apply to everyone, whatever their eligibility.
        if inp.offer_type == "discount_pct" and inp.value > ABSOLUTE_MAX_DISCOUNT_PCT:
            return block(f"Discounts above {ABSOLUTE_MAX_DISCOUNT_PCT}% are prohibited.",
                         "ceiling", "prohibited_excessive")
        if inp.offer_type == "discount_pct" and inp.months > MAX_OFFER_MONTHS:
            return block(f"Discounts may run for at most {MAX_OFFER_MONTHS} months.",
                         "duration", "prohibited_excessive")
        if inp.offer_type == "credit" and inp.value > CREDIT_HARD_CAP:
            return block(f"Credit exceeds the hard cap of ${CREDIT_HARD_CAP:.2f}.", "hard_cap",
                         max_allowed=CREDIT_HARD_CAP)

        # Monetary retention offers
        if c["tenure_months"] < MIN_TENURE_FOR_DISCOUNT:
            return block(f"Tenure below {MIN_TENURE_FOR_DISCOUNT} months: non-monetary offers only.", "tenure")
        last = c["last_retention_offer_date"]
        if last and _months_between(date.fromisoformat(last), AS_OF) < OFFER_COOLDOWN_MONTHS:
            return block(f"A monetary retention offer was made in the last {OFFER_COOLDOWN_MONTHS} months.",
                         "cooldown")
        if eff_risk == "low":
            return block("Low churn risk: non-monetary offers only.", "risk")

        tier_ref = f"tier_{c['tier']}"
        if inp.offer_type == "discount_pct":
            tier_max = MAX_DISCOUNT_PCT_BY_TIER[c["tier"]]
            max_pct = tier_max if eff_risk == "high" else tier_max // 2
            if inp.value > ABSOLUTE_MAX_DISCOUNT_PCT:
                return block(f"Discounts above {ABSOLUTE_MAX_DISCOUNT_PCT}% are prohibited.",
                             "ceiling", "prohibited_excessive", max_allowed=max_pct)
            if inp.months > MAX_OFFER_MONTHS:
                return block(f"Discounts may run for at most {MAX_OFFER_MONTHS} months.",
                             "duration", "prohibited_excessive", max_allowed=max_pct)
            if inp.value > max_pct:
                return block(f"{c['tier'].title()} tier with {eff_risk} risk allows at most {max_pct}%.",
                             tier_ref, "risk", max_allowed=max_pct)
            value = inp.value / 100 * price * inp.months
            refs.extend([POLICY_REFS[tier_ref], POLICY_REFS["offer_value"]])
        elif inp.offer_type == "credit":
            if inp.value > CREDIT_HARD_CAP:
                return block(f"Credit exceeds the hard cap of ${CREDIT_HARD_CAP:.2f}.", "hard_cap",
                             max_allowed=CREDIT_HARD_CAP)
            value = inp.value
        elif inp.offer_type == "plan_upgrade":
            next_tier = TIER_ORDER.index(c["tier"]) + 1
            if next_tier >= len(TIER_ORDER):
                return block("Already on the top tier; no upgrade available.", tier_ref)
            if inp.months > PLAN_UPGRADE_MAX_MONTHS:
                return block(f"Upgrade-at-current-price is limited to {PLAN_UPGRADE_MAX_MONTHS} months.",
                             "duration", max_allowed=PLAN_UPGRADE_MAX_MONTHS)
            with closing(_db()) as con:
                nxt = con.execute("SELECT monthly_price FROM plans WHERE plan_type = ? AND tier = ?",
                                  (c["plan_type"], TIER_ORDER[next_tier])).fetchone()
            value = (nxt["monthly_price"] - price) * inp.months
            refs.append(POLICY_REFS["offer_value"])
        else:  # fee_waiver
            if c["plan_type"] == "prepaid" or not c["contract_end_date"]:
                return block("Prepaid plans have no early-termination fee.", "prepaid_no_etf")
            remaining = max(0, math.ceil(_months_between(AS_OF, date.fromisoformat(c["contract_end_date"]))))
            etf = round(remaining * ETF_PCT_OF_PRICE * price, 2)
            if inp.value > etf + 0.005:
                return block("Waiver cannot exceed the early-termination fee owed.", "etf", max_allowed=etf)
            value = inp.value or etf
            reasons.append("Fee waivers always require human approval.")
            refs.extend([POLICY_REFS["etf"], POLICY_REFS["etf_waiver"], POLICY_REFS["human_approval"]])
            return decide("needs_approval", value, max_allowed=etf)

    if value > APPROVAL_THRESHOLD:
        reasons.append(f"Offer value exceeds the ${APPROVAL_THRESHOLD:.2f} auto-approval threshold.")
        refs.append(POLICY_REFS["human_approval"])
        return decide("needs_approval", value)
    reasons.append(f"Within policy limits and at or below the ${APPROVAL_THRESHOLD:.2f} threshold.")
    refs.append(POLICY_REFS["auto_approve"])
    return decide("allowed", value)


@mcp.tool(description="Check a proposed retention offer or billing credit against policy. Returns "
                      "allowed / needs_approval / blocked with the policy clauses that decide it.")
def check_offer_eligibility(
    auth: AuthParam,
    customer_id: CustomerIdParam,
    offer_type: Annotated[str, Field(description="discount_pct | credit | data_boost | plan_upgrade | fee_waiver")],
    value: Annotated[float, Field(description="discount_pct: percent; credit/fee_waiver: USD; data_boost: GB; plan_upgrade: 0")],
    months: Annotated[int, Field(description="Duration in billing months")] = 1,
    cancellation_intent: Annotated[bool, Field(description="Customer explicitly intends to cancel")] = False,
    purpose: Annotated[str, Field(description="retention | billing_adjustment")] = "retention",
) -> CheckOfferEligibilityOutput:
    def run(inp: CheckOfferEligibilityInput) -> dict:
        with closing(_db()) as con:
            c = _load_customer(con, inp.customer_id)
        if c is None:
            return _error(CheckOfferEligibilityOutput, "NOT_FOUND", "No account found for the caller.")
        return CheckOfferEligibilityOutput(ok=True, eligibility=evaluate_offer(c, inp)).model_dump()

    return _authorized_call(CheckOfferEligibilityOutput, CheckOfferEligibilityInput, auth, run,
                            customer_id=customer_id, offer_type=offer_type, value=value, months=months,
                            cancellation_intent=cancellation_intent, purpose=purpose)


@mcp.tool(description="Open an escalation ticket for a human agent (approval, repeat complaint, fraud, "
                      "out-of-policy, ambiguous request, regulator mention, tool failure).")
def create_escalation_ticket(
    auth: AuthParam,
    customer_id: CustomerIdParam,
    reason: Annotated[str, Field(description="approval_required | repeat_complaint | fraud_or_collections | "
                                             "out_of_policy | ambiguous_request | regulator_mention | "
                                             "tool_failure | other")],
    summary: Annotated[str, Field(description="Short case summary (5-500 chars); PII is masked on storage")],
    priority: Annotated[str, Field(description="P1 | P2 | P3")] = "P3",
) -> CreateEscalationTicketOutput:
    queues = {"approval_required": "retention-approvals", "repeat_complaint": "complaints-resolution",
              "fraud_or_collections": "fraud-review", "regulator_mention": "complaints-resolution"}

    def run(inp: CreateEscalationTicketInput) -> dict:
        ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        queue = queues.get(inp.reason, "care-tier2")
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(TICKETS_DB)) as con, con:
            con.execute("CREATE TABLE IF NOT EXISTS tickets (ticket_id TEXT PRIMARY KEY, customer_ref TEXT, "
                        "reason TEXT, queue TEXT, priority TEXT, summary TEXT, created_as_of TEXT)")
            con.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?)",
                        (ticket_id, mask_customer_id(inp.customer_id), inp.reason, queue, inp.priority,
                         mask(inp.summary), AS_OF.isoformat()))
        return CreateEscalationTicketOutput(ok=True, ticket_id=ticket_id, queue=queue,
                                            priority=inp.priority).model_dump()

    return _authorized_call(CreateEscalationTicketOutput, CreateEscalationTicketInput, auth, run,
                            customer_id=customer_id, reason=reason, summary=summary, priority=priority)


# --- resources -----------------------------------------------------------------

_CLAUSE_RE = re.compile(r"^### (POL-[A-Z]+-\d{3} §\d+\.\d+) — (.+)$", re.M)
_FRONT_RE = re.compile(r"^(doc_id|title|version|effective_date|owner): (.+)$", re.M)


def policy_catalog() -> dict:
    docs = []
    for path in sorted(POLICY_CORPUS_DIR.glob("*.md")):
        text = path.read_text()
        meta = dict(_FRONT_RE.findall(text.split("---", 2)[1]))
        docs.append({**meta, "path": f"data/policy_corpus/{path.name}",
                     "clauses": [{"id": cid, "title": title} for cid, title in _CLAUSE_RE.findall(text)]})
    return {"docs": docs, "doc_count": len(docs), "clause_count": sum(len(d["clauses"]) for d in docs)}


def plans_catalog() -> dict:
    with closing(_db()) as con:
        plans = [dict(r) for r in con.execute("SELECT * FROM plans ORDER BY plan_type, monthly_price")]
        offers = [dict(r) for r in con.execute("SELECT * FROM retention_offers ORDER BY offer_id")]
    return {"plans": plans, "retention_offers": offers,
            "approval_threshold_usd": APPROVAL_THRESHOLD, "credit_hard_cap_usd": CREDIT_HARD_CAP}


@mcp.resource("policy://catalog", name="policy_catalog", mime_type="application/json",
              description="Offer & retention policy documents and their citable clause IDs.")
def policy_catalog_resource() -> str:
    return json.dumps(policy_catalog(), ensure_ascii=False)


@mcp.resource("plans://catalog", name="plans_catalog", mime_type="application/json",
              description="Plan catalogue and retention-offer catalogue with approval limits.")
def plans_catalog_resource() -> str:
    return json.dumps(plans_catalog())


if __name__ == "__main__":
    mcp.run("stdio")
