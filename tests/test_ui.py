"""User-facing presentation: masked output, friendly labels, robust input handling (no API key needed)."""

from __future__ import annotations

import pytest
from rich.console import Console

from src import ui
from src.cli import parse_command

STATE = {
    "intent": "cancellation", "intent_confidence": 0.85, "input_flags": [], "guard_blocked": False,
    "account_summary": {"customer_ref": "CUST-***397", "account_ref": "ACC-*****949", "tenure_months": 48,
                        "churn_risk_label": "high", "complaints_90d": 2, "data_used_gb_last_cycle": 70.0,
                        "avg_data_used_gb_3m": 16.3,
                        "plan": {"name": "Postpaid Unlimited", "plan_type": "postpaid", "tier": "premium",
                                 "monthly_price": 85.0, "data_cap_gb": None}},
    "offer_decision": {"checks": [
        {"offer_type": "discount_pct", "value": 50, "months": 6, "source": "customer_request", "decision": "blocked",
         "reasons": ["Discounts above 25% are prohibited."], "policy_refs": ["POL-RET-002 §2.3", "POL-RET-004 §1.1"],
         "offer_value_usd": 0},
        {"offer_type": "discount_pct", "value": 20, "months": 6, "source": "ladder", "decision": "needs_approval",
         "reasons": ["Offer value exceeds the $50.00 auto-approval threshold."],
         "policy_refs": ["POL-RET-002 §1.3", "POL-RET-003 §1.2"], "offer_value_usd": 102.0}],
        "blocked": [{"offer_type": "discount_pct", "value": 50, "source": "customer_request",
                     "policy_refs": ["POL-RET-002 §2.3"], "reasons": []}]},
    "offer_proposal": {"offer_type": "discount_pct", "value": 20, "months": 6, "needs_approval": True},
    "approval": {"approved": True, "approver": "cli-auto-approve"},
    "resolution": {"outcome": "offer", "citations": ["POL-RET-003 §1.2", "POL-CAN-001 §1.3"]},
    "final_response": "Good news: 20% off for 6 months [POL-RET-003 §1.2].",
}


def render(fn, *a, **kw) -> str:
    rec = Console(record=True, width=120)
    old, ui.console = ui.console, rec
    try:
        fn(*a, **kw)
    finally:
        ui.console = old
    return rec.export_text()


def test_contact_report_is_readable_and_masked():
    out = render(ui.contact_report, "CT-0006", "Half off or I'm gone. Call me on +1-555-0142, id CUST-000397.", STATE)
    for heading in ("Customer request", "Detected intent", "Account", "Policy decision", "Resolution",
                    "Human approval", "Reply to the customer", "Policy clause"):
        assert heading in out
    assert "CUST-000397" not in out and "+1-555-0142" not in out and "CUST-***397" in out
    assert "blocked" in out and "needs approval" in out and "APPROVED" in out
    assert "Human approval required" in out            # clause title, not just the id
    assert "details:" not in out                        # no internal run ids / engine logs by default


def test_blocked_request_without_duration_is_not_guessed():
    assert ui.describe_offer({"offer_type": "discount_pct", "value": 50}) == "50% off"
    assert ui.describe_offer({"offer_type": "discount_pct", "value": 20, "months": 6}) == "20% off for 6 month(s)"


@pytest.mark.parametrize("text,expected", [
    ("", ("empty", "")), ("   ", ("empty", "")), ("help", ("help", "")), ("?", ("help", "")),
    ("QUIT", ("quit", "")), ("exit", ("quit", "")), ("account", ("account", "")), ("history", ("history", "")),
    ("reset", ("reset", "")), ("policy when do credits need approval?", ("policy", "when do credits need approval?")),
    ("my account is wrong", ("message", "my account is wrong")),    # a sentence, not a command
])
def test_chat_commands(text, expected):
    assert parse_command(text) == expected


@pytest.mark.parametrize("answer,expected", [("y", True), ("YES", True), ("approve", True), ("n", False),
                                             ("reject", False), ("maybe", None), ("", None)])
def test_approval_answers(answer, expected):
    assert ui.parse_yes_no(answer) is expected


def test_approval_prompt_is_plain_language():
    out = render(lambda: ui.console.print(ui.approval_prompt_text(
        {"customer_ref": "CUST-***397", "offer_type": "discount_pct", "value": 20, "months": 6,
         "offer_value_usd": 102.0, "policy_refs": ["POL-RET-003 §1.2"]})))
    assert "20% off for 6 month(s)" in out and "$102.00" in out and "team lead" in out


def test_api_resolution_event_explains_and_cites():
    from src.api.app import _resolution_event
    ev = _resolution_event({**STATE, "guard_blocked": False})
    assert ev["outcome_label"].endswith("Retention offer made")
    assert {"id": "POL-RET-003 §1.2", "title": "Human approval required"} in ev["citations"]
    assert any("above policy limits" in n for n in ev["notes"]) and any("approved" in n for n in ev["notes"])
