"""Account worker: pulls the authenticated caller's account/plan (and invoices for billing
contacts) through the MCP server, and derives billing findings (duplicate charges, overage)."""

from __future__ import annotations

import asyncio
from collections import Counter

from langgraph.runtime import Runtime

from src.agents.common import Deps, error_entry
from src.audit.audit_middleware import audit
from src.context.isolate import view
from src.run_context import agent_scope
from src.schemas import AccountOutput

NODE = "account_agent"


def billing_findings(billing: dict | None) -> list[dict]:
    """Deterministic checks on the latest invoice."""
    if not billing or not billing.get("invoices"):
        return []
    latest = billing["invoices"][0]
    findings = []
    counts = Counter((li["item"], li["amount"]) for li in latest["line_items"])
    for (item, amount), n in counts.items():
        if n > 1 and "monthly charge" not in item:
            findings.append({"type": "duplicate_charge", "invoice_id": latest["invoice_id"], "item": item,
                             "amount": amount, "occurrences": n, "credit_due": round(amount * (n - 1), 2),
                             "disputable": latest["disputable"]})
    for li in latest["line_items"]:
        if li["item"].lower().startswith("data overage"):
            findings.append({"type": "overage", "invoice_id": latest["invoice_id"], "item": li["item"],
                             "amount": li["amount"]})
    return findings


async def account_agent(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        state = view(NODE, state)  # isolate: customer_id + intent only
        mcp, cid = runtime.context.mcp, state["customer_id"]
        try:
            calls = [mcp.call("get_account", customer_id=cid)]
            if state.get("intent") == "billing_query":
                calls.append(mcp.call("get_billing_history", customer_id=cid, months=3))
            results = await asyncio.gather(*calls)
        except Exception as exc:  # timeout / transport failure -> escalate with reason
            audit("account_lookup_failed", "escalate", tool="get_account", reason=type(exc).__name__)
            return AccountOutput(account_summary=None, billing=None, billing_findings=[], account_failed=True,
                                 errors=[error_entry(NODE, "get_account", type(exc).__name__)]).update()
        account = results[0]
        if not account.get("ok"):
            code = account["error"]["code"]
            audit("account_lookup_failed", "escalate", tool="get_account", reason=code,
                  policy_ref=account["error"].get("policy_ref"))
            return AccountOutput(account_summary=None, billing=None, billing_findings=[], account_failed=True,
                                 errors=[error_entry(NODE, "get_account", code)]).update()
        billing = results[1] if len(results) > 1 and results[1].get("ok") else None
        return AccountOutput(account_summary=account["account"], billing=billing,
                             billing_findings=billing_findings(billing)).update()
