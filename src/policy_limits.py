"""Policy limits: the single source of truth shared by the data generator (which renders them
into data/policy_corpus/*.md), the MCP eligibility tool, and the retention-offer agent.

Clause IDs in POLICY_REFS must match the headings generated in data/policy_corpus/.
"""

from __future__ import annotations

from datetime import date

from src.config import settings

AS_OF = date(2026, 9, 15)  # fixed synthetic "today" so decisions never depend on the wall clock

APPROVAL_THRESHOLD = round(settings.offer_approval_threshold, 2)  # USD; above -> human approval
CREDIT_HARD_CAP = 150.00          # USD; above -> prohibited, copilot may not offer at all
MAX_DISCOUNT_PCT_BY_TIER = {"entry": 10, "mid": 15, "premium": 20}
ABSOLUTE_MAX_DISCOUNT_PCT = 25
MAX_OFFER_MONTHS = 6
MIN_TENURE_FOR_DISCOUNT = 6       # months
OFFER_COOLDOWN_MONTHS = 12
COLLECTIONS_LATE_PAYMENTS = 3     # late payments in 12 months -> no offers
DATA_BOOST_MAX_GB = 10
DATA_BOOST_MAX_MONTHS = 3
PLAN_UPGRADE_MAX_MONTHS = 3
ETF_PCT_OF_PRICE = 0.25           # early-termination fee per remaining month
DISPUTE_WINDOW_DAYS = 90
DISPUTE_SLA_BUSINESS_DAYS = 10
OVERAGE_PER_GB = 10.00
TAX_RATE = 0.08
REPEAT_COMPLAINT_ESCALATION = 3   # complaints in 90 days -> mandatory escalation
COMPLAINT_SLAS = {
    "P1": ("Total loss of service or safety issue", "4 hours", "48 hours"),
    "P2": ("Degraded service, repeated billing error", "24 hours", "5 business days"),
    "P3": ("General dissatisfaction, information request", "24 hours", "10 business days"),
}

PLANS = [
    # plan_id, name, type, tier, monthly_price, data_cap_gb (None = unlimited), minutes, contract_months
    ("PRE-BASIC", "Prepaid Basic", "prepaid", "entry", 15.00, 5, 300, 0),
    ("PRE-PLUS", "Prepaid Plus", "prepaid", "mid", 25.00, 15, 1000, 0),
    ("PRE-MAX", "Prepaid Max", "prepaid", "premium", 35.00, 40, None, 0),
    ("POST-ESSENTIAL", "Postpaid Essential", "postpaid", "entry", 45.00, 20, None, 12),
    ("POST-PREMIUM", "Postpaid Premium", "postpaid", "mid", 65.00, 60, None, 24),
    ("POST-UNLIMITED", "Postpaid Unlimited", "postpaid", "premium", 85.00, None, None, 24),
]
PLAN_BY_ID = {p[0]: p for p in PLANS}
TIER_ORDER = ["entry", "mid", "premium"]

RETENTION_OFFERS = [
    # offer_id, name, type, max_value, unit, max_months, monetary, notes
    ("OFF-DISC-PCT", "Loyalty discount", "discount_pct", None, "percent_of_plan_price",
     MAX_OFFER_MONTHS, 1, "Max % by plan tier per POL-RET-002 §1; value = pct x plan price x months"),
    ("OFF-CREDIT", "One-time bill credit", "credit", CREDIT_HARD_CAP, "usd", 1, 1,
     "Above approval threshold needs human approval per POL-RET-003 §1"),
    ("OFF-DATA-BOOST", "Data boost +10 GB", "data_boost", DATA_BOOST_MAX_GB, "gb_per_month",
     DATA_BOOST_MAX_MONTHS, 0, "Non-monetary; available to any eligible customer"),
    ("OFF-PLAN-UPGRADE", "Upgrade at current price", "plan_upgrade", None, "price_difference",
     PLAN_UPGRADE_MAX_MONTHS, 1, "Value = (next-tier price - current price) x months"),
    ("OFF-ETF-WAIVE", "Early-termination fee waiver", "fee_waiver", None, "usd", 1, 1,
     "Always requires human approval per POL-CAN-001 §3.2"),
]
OFFER_TYPES = tuple(o[2] for o in RETENTION_OFFERS)

# Clause IDs cited by automated decisions (must exist in data/policy_corpus).
POLICY_REFS = {
    "tenure": "POL-RET-001 §1.1",
    "risk": "POL-RET-001 §1.2",
    "cancel_intent": "POL-RET-001 §1.3",
    "cooldown": "POL-RET-001 §2.1",
    "fraud": "POL-RET-001 §2.2",
    "tier_entry": "POL-RET-002 §1.1",
    "tier_mid": "POL-RET-002 §1.2",
    "tier_premium": "POL-RET-002 §1.3",
    "duration": "POL-RET-002 §2.1",
    "offer_value": "POL-RET-002 §2.2",
    "ceiling": "POL-RET-002 §2.3",
    "auto_approve": "POL-RET-003 §1.1",
    "human_approval": "POL-RET-003 §1.2",
    "hard_cap": "POL-RET-003 §1.3",
    "prohibited_excessive": "POL-RET-004 §1.1",
    "prepaid_no_etf": "POL-CAN-001 §2.1",
    "etf": "POL-CAN-001 §2.2",
    "etf_waiver": "POL-CAN-001 §3.2",
    "dispute_credit": "POL-BIL-001 §2.1",
    "account_holder_only": "POL-PRV-001 §1.1",
}
