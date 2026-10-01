"""Streamlit UI (app.py, src/streamlit_ui/): read-only data layer, error handling and page smoke tests.

The pages are run with Streamlit's AppTest against an API that is not running, so the error paths
are exercised without starting the agent backend.
"""

from __future__ import annotations

import httpx
import pytest

from mcp_server.schemas import CheckOfferEligibilityInput
from mcp_server.server import DB_PATH, _load_customer, evaluate_offer
from src.config import ROOT_DIR
from src.streamlit_ui import backend, data
from src.streamlit_ui.components import plain_reason

OFFLINE = "http://127.0.0.1:9"  # nothing listens on the discard port


def test_customers_are_masked_and_sorted_by_risk():
    rows = data.customers()
    assert rows and rows[0]["risk"] == "high"
    order = {"high": 0, "medium": 1, "low": 2}
    assert [order[r["risk"]] for r in rows] == sorted(order[r["risk"]] for r in rows)
    for r in rows:
        assert r["customer_ref"].startswith("CUST-***")
        assert len(r["initial"]) == 1
        assert not {"first_name", "last_name", "email", "phone", "account_number"} & set(r)


def test_risk_factors_explain_themselves_and_lead_with_the_most_severe():
    c = data.customer(data.customers()[0]["customer_id"])
    sev = {"high": 0, "medium": 1, "low": 2, "info": 3}
    assert [sev[f["severity"]] for f in c["factors"]] == sorted(sev[f["severity"]] for f in c["factors"])
    assert all(f["label"] and f["detail"] for f in c["factors"])
    assert any("churn" in f["label"].lower() for f in c["factors"])


def test_opportunities_use_the_agents_policy_engine():
    for o in data.opportunities():
        assert o["risk"] != "low"
        if o["best"]:
            assert o["best"]["decision"] in ("allowed", "needs_approval")
        else:
            assert o["checks"] and all(c["decision"] == "blocked" for c in o["checks"])


def test_recheck_reproduces_the_policy_decision_with_real_amounts():
    """The audit log masks amounts; the UI re-runs the deterministic engine to show the real cost."""
    import sqlite3
    from contextlib import closing

    cid = data.customers()[0]["customer_id"]
    offer = {"offer_type": "discount_pct", "value": 10, "months": 6, "source": "ladder"}
    got = data.recheck(cid, offer, cancellation=True)
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        want = evaluate_offer(_load_customer(con, cid), CheckOfferEligibilityInput(
            customer_id=cid, offer_type="discount_pct", value=10, months=6, cancellation_intent=True))
    assert got["decision"] == want.decision and got["cost"] == want.offer_value_usd
    assert isinstance(got["cost"], float) and "**" not in " ".join(got["reasons"])
    assert data.recheck(cid, {"offer_type": "discount_pct"}, cancellation=False) is None  # incomplete record


def test_plain_reason_hides_internal_names():
    assert plain_reason("discount_pct allowed by check_offer_eligibility") == \
        "The discount passed the policy eligibility check"
    assert plain_reason("3 complaints in 90 days") == "3 complaints in 90 days"
    assert plain_reason(None) is None


def test_backend_errors_are_friendly_and_masked():
    req = httpx.Request("POST", "http://x/v1/contacts/stream")
    err = backend._explain(httpx.HTTPStatusError(
        "403", request=req, response=httpx.Response(403, json={"detail": "session of CUST-123456"}, request=req)))
    assert err.message == "This case belongs to another customer."
    assert "CUST-123456" not in err.detail and "HTTP 403" in err.detail
    assert "not reachable" in backend._explain(httpx.ConnectError("refused")).message
    assert "too long" in backend._explain(httpx.ReadTimeout("slow")).message
    assert backend._explain(RuntimeError("boom")).message.startswith("Unable to complete the analysis")


@pytest.fixture
def offline_api(monkeypatch):
    monkeypatch.setenv("COPILOT_API_URL", OFFLINE)
    backend.base_url.clear()
    backend.samples.clear()
    yield
    backend.base_url.clear()


def test_app_starts_cleanly_when_the_api_is_offline(offline_api):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Customer Service &amp; Retention Copilot" in " ".join(m.value for m in at.markdown)
    assert any("API offline" in m.value for m in at.sidebar.markdown)
    assert [m.label for m in at.metric][:2] == ["Customers profiled", "At-risk customers"]


def _copilot_page():
    from src.streamlit_ui.pages import customer_copilot

    customer_copilot.render()


def test_copilot_validates_input_and_reports_backend_failure(offline_api):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_copilot_page, default_timeout=60).run()
    assert not at.exception
    run = next(b for b in at.button if "Run full analysis" in b.label)
    run.click().run()
    assert any("enter the customer's message" in w.value for w in at.warning)

    at.text_area(key="cp_message").set_value("I want to cancel my plan.").run()
    next(b for b in at.button if "Run full analysis" in b.label).click().run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)  # clean message, no traceback
