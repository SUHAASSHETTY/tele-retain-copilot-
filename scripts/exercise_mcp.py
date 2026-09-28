"""Exercise every MCP tool and resource once (plus the authorization and error paths) through
langchain-mcp-adapters, producing logs/mcp_transcript.jsonl from code.

Run: python -m scripts.exercise_mcp [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3

from src.config import MCP_TRANSCRIPT_LOG, SAMPLE_CONTACTS_PATH, SYNTHETIC_DIR
from src.guardrails.pii import mask_customer_id
from src.run_context import agent_scope, run_scope
from src.tools.mcp_client import TelecomMCP, parse_tool_payload


def _contact(scenario: str) -> dict:
    for line in SAMPLE_CONTACTS_PATH.read_text().splitlines():
        c = json.loads(line)
        if c["scenario"] == scenario:
            return c
    raise KeyError(scenario)


def _summary(payload: dict) -> str:
    if not payload.get("ok"):
        return f"ok=False error={payload['error']['code']}"
    if "eligibility" in payload:
        e = payload["eligibility"]
        return f"decision={e['decision']} refs={e['policy_refs']}"
    if "account" in payload:
        a = payload["account"]
        return f"ok=True {a['customer_ref']} plan={a['plan']['plan_id']} risk={a['churn_risk_label']}"
    if "invoices" in payload:
        return f"ok=True invoices={len(payload['invoices'])}"
    if "ticket_id" in payload:
        return f"ok=True ticket queue={payload['queue']}"
    return str(payload)[:120]


async def main(reset: bool) -> None:
    if reset:
        MCP_TRANSCRIPT_LOG.unlink(missing_ok=True)
    mcp = TelecomMCP()

    cases = [
        # (scenario whose caller is authenticated, label, tool, args-builder)
        ("cancellation_at_risk_needs_approval", "get_account (own account)", "get_account",
         lambda cid: {"customer_id": cid}),
        ("billing_dispute", "get_billing_history (own, 2 invoices)", "get_billing_history",
         lambda cid: {"customer_id": cid, "months": 2}),
        ("billing_dispute", "check_offer_eligibility: $15 duplicate-charge credit", "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "credit", "value": 15, "purpose": "billing_adjustment"}),
        ("cancellation_at_risk_within_threshold", "check_offer_eligibility: 10% x 6 mo (entry, high risk)",
         "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "discount_pct", "value": 10, "months": 6,
                      "cancellation_intent": True}),
        ("cancellation_at_risk_needs_approval", "check_offer_eligibility: 20% x 6 mo (premium, high risk)",
         "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "discount_pct", "value": 20, "months": 6,
                      "cancellation_intent": True}),
        ("cancellation_at_risk_needs_approval", "check_offer_eligibility: 50% ('half off')",
         "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "discount_pct", "value": 50, "months": 6,
                      "cancellation_intent": True}),
        ("prompt_injection", "check_offer_eligibility: 90% off (injection)", "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "discount_pct", "value": 90, "months": 24}),
        ("cancellation_fraud_flag", "check_offer_eligibility: fraud-flagged account", "check_offer_eligibility",
         lambda cid: {"customer_id": cid, "offer_type": "discount_pct", "value": 5, "months": 3,
                      "cancellation_intent": True}),
        ("complaint_repeat", "create_escalation_ticket (repeat complaint)", "create_escalation_ticket",
         lambda cid: {"customer_id": cid, "reason": "repeat_complaint", "priority": "P2",
                      "summary": "Third dropped-call complaint in 90 days; caller CUST-000411 reachable "
                                 "on +1-555-0142."}),
        ("cross_customer_request", "get_billing_history for another customer's ID",
         "get_billing_history", lambda cid: {"customer_id": "CUST-000123", "months": 1}),
        ("billing_query", "get_billing_history months=12 (validation error)", "get_billing_history",
         lambda cid: {"customer_id": cid, "months": 12}),
    ]

    async with mcp.session() as session:
        print(f"tools/list -> {[t.name for t in session.tools]}")
        print(f"LLM-visible params: " + ", ".join(f"{t.name}({', '.join(t.args)})" for t in session.tools))
        for scenario, label, tool, build in cases:
            contact = _contact(scenario)
            token = mcp.issue_session_token(contact["customer_id"], contact["session_id"])
            with run_scope(session_id=contact["session_id"], mcp_auth_token=token), agent_scope("exercise_mcp"):
                payload = await session.call(tool, **build(contact["customer_id"]))
            print(f"  {label:<58} caller={mask_customer_id(contact['customer_id'])} -> {_summary(payload)}")

        # Forged token: a caller signs their own token with a guessed secret.
        contact = _contact("cross_customer_request")
        forged = TelecomMCP(secret="not-the-server-secret").issue_session_token("CUST-000123", "SES-X")
        with run_scope(session_id=contact["session_id"], mcp_auth_token=forged), agent_scope("exercise_mcp"):
            payload = await session.call("get_account", customer_id="CUST-000123")
        print(f"  {'get_account with forged token':<58} -> {_summary(payload)}")

        # The model tries to pass its own `auth` argument: it is discarded by the interceptor.
        with run_scope(session_id="SES-NOAUTH", mcp_auth_token=None), agent_scope("exercise_mcp"):
            payload = await session.call("get_account", customer_id="CUST-000123", auth="anything")
        print(f"  {'get_account, no session token, model-supplied auth':<58} -> {_summary(payload)}")

        with run_scope(session_id="SES-RESOURCES"), agent_scope("exercise_mcp"):
            policy = await session.read_resource("policy://catalog")
            plans = await session.read_resource("plans://catalog")
        print(f"resources/read policy://catalog -> {policy['doc_count']} docs, {policy['clause_count']} clauses")
        print(f"resources/read plans://catalog  -> {len(plans['plans'])} plans, "
              f"{len(plans['retention_offers'])} offers, threshold=${plans['approval_threshold_usd']}")

    # Multi-session path (MultiServerMCPClient.get_tools: one stdio session per call)
    tools = {t.name: t for t in await mcp.get_tools()}
    contact = _contact("plan_change")
    token = mcp.issue_session_token(contact["customer_id"], contact["session_id"])
    with run_scope(session_id=contact["session_id"], mcp_auth_token=token), agent_scope("exercise_mcp"):
        payload = parse_tool_payload(await tools["get_account"].ainvoke({"customer_id": contact["customer_id"]}))
    print(f"MultiServerMCPClient.get_tools path: get_account -> {_summary(payload)}")

    # Report on the transcript this run produced.
    records = [json.loads(line) for line in MCP_TRANSCRIPT_LOG.read_text().splitlines()]
    by_method: dict[str, int] = {}
    for r in records:
        by_method[r["method"]] = by_method.get(r["method"], 0) + 1
    statuses = {s: sum(r["status"] == s for r in records) for s in {r["status"] for r in records}}
    print(f"\n{MCP_TRANSCRIPT_LOG.relative_to(MCP_TRANSCRIPT_LOG.parents[1])}: {len(records)} records "
          f"{by_method} statuses={statuses}")

    # Leak check: no plaintext customer IDs, account numbers, phones or emails in the transcript.
    con = sqlite3.connect(SYNTHETIC_DIR / "telecom.db")
    secrets_ = [v for row in con.execute("SELECT customer_id, account_number, phone, email FROM customers")
                for v in row]
    text = MCP_TRANSCRIPT_LOG.read_text()
    leaks = [v for v in secrets_ if v in text]
    print(f"plaintext PII leak check over {len(secrets_)} identifiers: {'PASS' if not leaks else f'FAIL {len(leaks)}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="truncate logs/mcp_transcript.jsonl first")
    asyncio.run(main(ap.parse_args().reset))
