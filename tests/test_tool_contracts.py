"""Tool contracts: every tool's input/output schema validates, and every tool has error paths
(bad customer id, cross-customer access, over-limit offer, RAG no-hit) that return structured
results rather than raising or leaking stack traces. No API key needed."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import mcp_server.server as server
from mcp_server.auth import SECRET_ENV, issue_token
from mcp_server.schemas import (
    CheckOfferEligibilityOutput,
    CreateEscalationTicketOutput,
    GetAccountOutput,
    GetBillingHistoryOutput,
)
from src.config import SAMPLE_CONTACTS_PATH
from src.guardrails.output_guard import known_citations
from src.resilience import ExternalCallFailed
from src.tools.rag_tool import PolicyRAGInput, PolicyRAGOutput, run_policy_rag

SECRET = "test-secret"
CONTACTS = {json.loads(l)["scenario"]: json.loads(l) for l in SAMPLE_CONTACTS_PATH.read_text().splitlines()}
ME = CONTACTS["cancellation_at_risk_needs_approval"]["customer_id"]   # premium tier, high risk
OTHER = "CUST-000123"


@pytest.fixture(autouse=True)
def mcp_secret(monkeypatch, tmp_path):
    monkeypatch.setenv(SECRET_ENV, SECRET)
    monkeypatch.setattr(server, "TICKETS_DB", tmp_path / "tickets.db")


def tok(cid: str = ME) -> str:
    return issue_token(SECRET, "SES-TEST", cid)


def _no_trace(payload: dict) -> None:
    assert "Traceback" not in json.dumps(payload)


# --- happy paths: outputs validate against their Pydantic schemas ------------------

def test_get_account_contract():
    out = GetAccountOutput.model_validate(server.get_account(tok(), ME))
    assert out.ok and out.account.customer_ref.startswith("CUST-***") and out.account.plan.plan_id


def test_get_billing_history_contract():
    out = GetBillingHistoryOutput.model_validate(server.get_billing_history(tok(), ME, months=2))
    assert out.ok and len(out.invoices) == 2 and all(i.total > 0 for i in out.invoices)


def test_check_offer_eligibility_contract():
    out = CheckOfferEligibilityOutput.model_validate(
        server.check_offer_eligibility(tok(), ME, "discount_pct", 20, months=6, cancellation_intent=True))
    e = out.eligibility
    assert out.ok and e.decision == "needs_approval" and e.needs_approval and "POL-RET-003 §1.2" in e.policy_refs


def test_create_escalation_ticket_contract():
    out = CreateEscalationTicketOutput.model_validate(
        server.create_escalation_ticket(tok(), ME, "repeat_complaint", "Third complaint in 90 days", "P2"))
    assert out.ok and out.ticket_id.startswith("ESC-") and out.queue == "complaints-resolution"


@pytest.mark.asyncio
async def test_mcp_tool_schemas_are_published():
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert set(tools) == {"get_account", "get_billing_history", "check_offer_eligibility", "create_escalation_ticket"}
    for t in tools.values():
        assert {"auth", "customer_id"} <= set(t.inputSchema["properties"])
        assert t.outputSchema and "ok" in t.outputSchema["properties"]


# --- error paths ---------------------------------------------------------------------

@pytest.mark.parametrize("call,code", [
    (lambda: server.get_account(tok(), "CUST-12"), "VALIDATION_ERROR"),                     # bad customer id
    (lambda: server.get_account(tok(), OTHER), "AUTHZ_DENIED"),                             # cross-customer
    (lambda: server.get_account(issue_token("wrong-secret", "S", OTHER), OTHER), "AUTH_INVALID"),  # forged token
    (lambda: server.get_account("", ME), "AUTH_MISSING"),
    (lambda: server.get_billing_history(tok(), ME, months=12), "VALIDATION_ERROR"),
    (lambda: server.get_billing_history(tok(), OTHER, months=1), "AUTHZ_DENIED"),
    (lambda: server.check_offer_eligibility(tok(), ME, "free_iphone", 1), "VALIDATION_ERROR"),
    (lambda: server.check_offer_eligibility(tok(), OTHER, "credit", 10), "AUTHZ_DENIED"),
    (lambda: server.create_escalation_ticket(tok(), ME, "because", "Some summary text"), "VALIDATION_ERROR"),
    (lambda: server.create_escalation_ticket(tok(), ME, "other", "hi"), "VALIDATION_ERROR"),
], ids=["account-bad-id", "account-cross-customer", "account-forged-token", "account-no-token",
        "billing-bad-months", "billing-cross-customer", "offer-bad-type", "offer-cross-customer",
        "ticket-bad-reason", "ticket-short-summary"])
def test_tool_error_paths_are_structured(call, code):
    payload = call()
    assert payload["ok"] is False and payload["error"]["code"] == code
    _no_trace(payload)
    if code == "AUTHZ_DENIED":
        assert payload["error"]["policy_ref"] == "POL-PRV-001 §1.1"


@pytest.mark.parametrize("offer_type,value,months,ref", [
    ("discount_pct", 90, 6, "POL-RET-004 §1.1"),      # above absolute ceiling
    ("discount_pct", 20, 12, "POL-RET-002 §2.1"),     # too long
    ("credit", 400, 1, "POL-RET-003 §1.3"),           # above hard cap
])
def test_over_limit_offers_are_blocked_with_policy_ref(offer_type, value, months, ref):
    out = CheckOfferEligibilityOutput.model_validate(
        server.check_offer_eligibility(tok(), ME, offer_type, value, months=months, cancellation_intent=True))
    assert out.ok and out.eligibility.decision == "blocked" and not out.eligibility.allowed
    assert ref in out.eligibility.policy_refs


# --- policy RAG tool ---------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"question": "hi"}, {"question": "valid question", "top_k": 0},
                                 {"question": "valid question", "top_k": 9}, {"question": "x" * 501}])
def test_rag_input_schema_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        PolicyRAGInput(**bad)


@pytest.mark.asyncio
async def test_rag_output_contract_and_citations_resolve():
    out = PolicyRAGOutput.model_validate((await run_policy_rag("When does a credit need human approval?")).model_dump())
    assert out.status == "answered" and out.citations and out.grader == "heuristic"
    assert all(c.citation in known_citations() for c in out.citations)


@pytest.mark.asyncio
async def test_rag_no_hit_returns_no_relevant_policy():
    out = await run_policy_rag("What's the weather forecast in Paris tomorrow?")
    assert out.status == "no_relevant_policy" and out.citations == [] and "escalate" in out.answer.lower()


@pytest.mark.asyncio
async def test_rag_judge_failure_degrades_instead_of_raising():
    class Failing:
        name = "gemini"

        async def grade(self, *a):
            raise ExternalCallFailed("rag.grade", "429 RESOURCE_EXHAUSTED", 3)

        rewrite = answer = grade

    out = await run_policy_rag("When does a credit need human approval?", judge=Failing())
    assert out.degraded and out.grader == "heuristic" and out.status == "answered"


# --- through the real MCP client (stdio + langchain-mcp-adapters) --------------------------

@pytest.mark.asyncio
async def test_mcp_client_hides_auth_and_logs_every_call(tmp_path, isolated_logs):
    from src.run_context import run_scope
    from src.tools.mcp_client import TelecomMCP

    mcp = TelecomMCP(transcript_path=tmp_path / "transcript.jsonl", secret=SECRET)
    async with mcp.session() as s:
        assert all("auth" not in t.args for t in s.tools)
        with run_scope(session_id="SES-TEST", mcp_auth_token=mcp.issue_session_token(ME, "SES-TEST")):
            ok = await s.call("get_account", customer_id=ME)
            denied = await s.call("get_account", customer_id=OTHER, auth="model-supplied-token")
    assert ok["ok"] and denied["error"]["code"] == "AUTHZ_DENIED"
    records = [json.loads(l) for l in (isolated_logs / "tool_calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["ok", "denied"]
    required = {"timestamp", "run_id", "agent", "tool_name", "args", "result", "latency_ms", "status", "error"}
    assert all(required <= r.keys() for r in records)
    # a model-supplied `auth` is discarded by the interceptor and never logged in clear
    assert all(r["args"].get("auth") in (None, "<redacted>") for r in records)
    assert "model-supplied-token" not in (isolated_logs / "tool_calls.jsonl").read_text()
