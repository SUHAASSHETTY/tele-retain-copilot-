"""Read-only data for the Streamlit views. No agent, LLM or write path is involved here.

Every number comes from data the project already produces:
  * data/synthetic/telecom.db          customers, plans, complaints (opened read-only)
  * mcp_server.server.evaluate_offer   the deterministic policy engine the agents call through MCP,
                                        used to size retention opportunities (nothing is offered)
  * src.agents.retention_offer_agent.candidate_ladder   the agents' own offer ladder
  * logs/agent_actions.jsonl           the audit trail (cases, decisions, per-run evidence)
  * reports/*.json                     evaluation, red-team and golden-signal evidence

Names, emails, phones and account numbers never leave this module: a customer is shown as the masked
customer reference plus a first initial. The raw customer ID is kept only as the handle sent in the
X-Customer-Id header (the demo stand-in for an authenticated caller, as in the CLI and /v1/samples).

Risk factors are transparent rules over account facts; each carries the evidence it used.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date
from pathlib import Path

from mcp_server.schemas import CheckOfferEligibilityInput
from mcp_server.server import DB_PATH, _load_customer, evaluate_offer
from src import ui
from src.agents.retention_offer_agent import candidate_ladder
from src.audit import audit_middleware
from src.tools import logging_middleware
from src.config import EVAL_REPORT_PATH, GOLDEN_SIGNALS_PATH, REPORTS_DIR
from src.guardrails.pii import mask, mask_customer_id, mask_obj
from src.policy_limits import AS_OF, COLLECTIONS_LATE_PAYMENTS, REPEAT_COMPLAINT_ESCALATION

CONTRACT_SOON_DAYS = 60
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
COMPLAINT_SEVERITY = {"P1": "critical", "P2": "high", "P3": "normal"}


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


# --- risk factors ------------------------------------------------------------------

def risk_factors(c: sqlite3.Row, open_complaints: int) -> list[dict]:
    """Rule-based factors from account facts, most severe first. severity: high | medium | low | info."""
    out: list[dict] = []

    def add(severity: str, label: str, detail: str) -> None:
        out.append({"severity": severity, "label": label, "detail": detail})

    risk = c["churn_risk_label"]
    add(risk, f"{risk.capitalize()} churn risk score", f"Churn model score {c['churn_risk_score']:.2f} on the account.")
    if c["complaints_90d"] >= REPEAT_COMPLAINT_ESCALATION:
        add("high", "Repeat complaints",
            f"{c['complaints_90d']} complaints in 90 days (policy escalates at {REPEAT_COMPLAINT_ESCALATION}).")
    elif c["complaints_90d"]:
        add("medium", "Recent complaints", f"{c['complaints_90d']} complaint(s) in the last 90 days.")
    if open_complaints:
        add("medium", "Unresolved complaint", f"{open_complaints} complaint(s) still open.")
    if c["competitor_quote"]:
        add("high", "Competitor quote on file", "The customer has mentioned a competitor's offer.")
    if c["contract_end_date"]:
        days = (date.fromisoformat(c["contract_end_date"]) - AS_OF).days
        if 0 <= days <= CONTRACT_SOON_DAYS:
            add("medium", "Contract ending soon", f"Contract ends in {days} days; the customer can leave freely.")
    cap = c["data_cap_gb"]
    if cap and c["data_used_gb"] > cap:
        add("medium", "Plan does not fit usage", f"{c['data_used_gb']:.0f} GB used vs a {cap} GB cap.")
    if c["late_payments_12m"] >= COLLECTIONS_LATE_PAYMENTS:
        add("high", "Collections hold", f"{c['late_payments_12m']} late payments in 12 months; no offers allowed.")
    elif c["late_payments_12m"]:
        add("low", "Late payment", f"{c['late_payments_12m']} late payment(s) in 12 months.")
    if c["fraud_flag"]:
        add("high", "Fraud flag", "Account is flagged; retention offers are not permitted.")
    return sorted(out, key=lambda f: SEVERITY_ORDER[f["severity"]])


def best_offer(c: sqlite3.Row) -> dict:
    """Walk the agents' candidate ladder through the policy engine (read-only; nothing is offered)."""
    state = {"account_summary": {"plan": {"tier": c["tier"]}, "churn_risk_label": c["churn_risk_label"]},
             "cancellation_intent": False}
    checks = []
    for cand in candidate_ladder(state):
        e = evaluate_offer(c, CheckOfferEligibilityInput(
            customer_id=c["customer_id"], offer_type=cand["offer_type"], value=cand["value"],
            months=cand["months"], purpose="retention"))
        checks.append({"offer": ui.describe_offer(cand), "decision": e.decision, "cost_usd": e.offer_value_usd,
                       "reasons": e.reasons, "policy_refs": e.policy_refs})
        if e.allowed:
            break
    return {"best": next((x for x in checks if x["decision"] != "blocked"), None), "checks": checks}


def recheck(customer_id: str, offer: dict, cancellation: bool) -> dict | None:
    """Re-run one offer from the audit trail through the policy engine. The audit log masks amounts by
    design; the engine is deterministic (fixed AS_OF date), so this reproduces the real cost and reasons."""
    try:
        purpose = "billing_adjustment" if offer.get("source") == "duplicate_charge" else "retention"
        with closing(_db()) as con:
            c = _load_customer(con, customer_id)
        e = evaluate_offer(c, CheckOfferEligibilityInput(
            customer_id=customer_id, offer_type=offer["offer_type"], value=float(offer["value"]),
            months=int(offer.get("months") or 1), cancellation_intent=cancellation, purpose=purpose))
    except (KeyError, TypeError, ValueError):
        return None
    return {"decision": e.decision, "cost": e.offer_value_usd, "reasons": e.reasons, "policy_refs": e.policy_refs}


# --- customers ---------------------------------------------------------------------

def _row(c: sqlite3.Row, open_cmp: int) -> dict:
    factors = risk_factors(c, open_cmp)
    return {
        "customer_id": c["customer_id"], "customer_ref": mask_customer_id(c["customer_id"]),
        "initial": c["first_name"][0], "region": c["region"], "plan": c["plan_name"], "tier": c["tier"],
        "plan_type": c["plan_type"], "tenure_months": c["tenure_months"],
        "monthly_bill": round(float(c["monthly_bill"]), 2), "risk": c["churn_risk_label"],
        "risk_score": round(float(c["churn_risk_score"]), 2), "complaints_90d": c["complaints_90d"],
        "open_complaints": open_cmp, "contract_end_date": c["contract_end_date"], "auto_pay": bool(c["auto_pay"]),
        "data_used_gb": c["data_used_gb"], "data_cap_gb": c["data_cap_gb"], "factors": factors,
        "needs_attention": any(f["severity"] == "high" for f in factors) or open_cmp > 0,
    }


def customers() -> list[dict]:
    """All customers, highest risk first."""
    with closing(_db()) as con:
        rows = con.execute(
            "SELECT c.*, p.name AS plan_name, p.plan_type, p.tier, p.monthly_price, p.data_cap_gb, p.contract_months "
            "FROM customers c JOIN plans p USING (plan_id)").fetchall()
        open_counts = dict(con.execute("SELECT customer_id, COUNT(*) FROM complaints WHERE status='open' GROUP BY 1"))
    out = [_row(c, open_counts.get(c["customer_id"], 0)) for c in rows]
    return sorted(out, key=lambda r: (RISK_ORDER[r["risk"]], -r["risk_score"]))


def customer(customer_id: str) -> dict | None:
    with closing(_db()) as con:
        c = _load_customer(con, customer_id)
        if c is None:
            return None
        complaints = [dict(r) for r in con.execute(
            "SELECT opened_date, category, severity, status, summary FROM complaints "
            "WHERE customer_id = ? ORDER BY opened_date DESC", (customer_id,))]
    for x in complaints:
        x["summary"] = mask(x["summary"] or "", amounts=False)
        x["severity"] = COMPLAINT_SEVERITY.get(x["severity"], x["severity"])
    row = _row(c, sum(1 for x in complaints if x["status"] == "open"))
    return {**row, "complaints": complaints, "retention": best_offer(c)}


def opportunities(rows: list[dict] | None = None) -> list[dict]:
    """Best policy-eligible retention offer for every medium/high-risk customer."""
    wanted = {r["customer_id"] for r in (rows or customers()) if r["risk"] != "low"}
    out = []
    with closing(_db()) as con:
        for cid in wanted:
            c = _load_customer(con, cid)
            out.append({"customer_id": cid, "customer_ref": mask_customer_id(cid), "initial": c["first_name"][0],
                        "plan": c["plan_name"], "risk": c["churn_risk_label"],
                        "risk_score": round(float(c["churn_risk_score"]), 2),
                        "monthly_bill": round(float(c["monthly_bill"]), 2), **best_offer(c)})
    return sorted(out, key=lambda o: (RISK_ORDER[o["risk"]], -o["risk_score"]))


def complaint_mix() -> dict:
    with closing(_db()) as con:
        rows = [dict(r) for r in con.execute("SELECT category, status FROM complaints")]
    return {"by_category": dict(Counter(r["category"] for r in rows).most_common()),
            "open": sum(1 for r in rows if r["status"] == "open"), "total": len(rows)}


# --- audit trail -------------------------------------------------------------------

def audit_rows() -> list[dict]:
    return _read_jsonl(audit_middleware._log_path)


def cases() -> list[dict]:
    """One row per resolved contact (`contact_resolved`), newest first."""
    rows = sorted((r for r in audit_rows() if r.get("action") == "contact_resolved"),
                  key=lambda r: r.get("timestamp", ""), reverse=True)
    return mask_obj([{"time": r["timestamp"][:19].replace("T", " "), "session": r.get("session_id"),
                      "outcome": ui.OUTCOME_LABELS.get(r.get("decision"), ("", "", r.get("decision")))[2],
                      "reason": r.get("reason"), "policy": ", ".join(r.get("policy_ref") or []),
                      "run_id": r.get("run_id")} for r in rows], amounts=True)


def run_trail(run_id: str) -> list[dict]:
    """Every audited decision of one run, in order (the evidence behind a recommendation)."""
    keep = ("timestamp", "actor", "actor_type", "action", "tool", "decision", "reason", "policy_ref", "details")
    return mask_obj([{k: r.get(k) for k in keep} for r in audit_rows() if r.get("run_id") == run_id], amounts=False)


def run_tools(run_id: str) -> list[dict]:
    """Tool calls of one run from the logging middleware (arguments and results omitted)."""
    keep = ("agent", "tool_name", "tool_source", "status", "latency_ms")
    return [{k: r.get(k) for k in keep} for r in _read_jsonl(logging_middleware._log_path) if r.get("run_id") == run_id]


def decision_counts() -> dict:
    return dict(Counter(r.get("action") for r in audit_rows()))


# --- evaluation evidence ---------------------------------------------------------------

def evaluation() -> dict:
    return {"eval": _read_json(EVAL_REPORT_PATH), "redteam": _read_json(REPORTS_DIR / "redteam_results.json"),
            "signals": _read_json(GOLDEN_SIGNALS_PATH), "regenerate": _read_json(REPORTS_DIR / "regenerate_summary.json")}
