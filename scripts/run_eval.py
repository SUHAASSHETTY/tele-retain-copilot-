"""Agent evaluation over data/golden_set.jsonl -> reports/eval_report.json.

1. Runs every golden case through the copilot graph (same path as the CLI; approvals auto-approved).
2. Deterministic metrics: intent accuracy, action accuracy, policy-ref recall, citation validity.
3. DeepEval LLM-as-judge with Gemini ONLY (deepeval.models.GeminiModel, eval_mode="llm"):
     HallucinationMetric   actual output vs ground-truth context (expected policy clauses + account facts)
     FaithfulnessMetric    actual output vs what the agent retrieved (cited clauses + tool results)
     AnswerRelevancyMetric actual output vs the customer's input
   Judge calls go through CachedGeminiJudge: on-disk cache (.cache/eval_judge_cache.sqlite, so reruns
   are free), the shared Gemini rate limiter, and retries with backoff on 429/503. If the judge is
   unusable, judge metrics are reported as not run with the reason; no score is ever invented.

Run: python -m scripts.run_eval [--limit N] [--no-judge] [--agent-llm]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM, GeminiModel

from src.config import EVAL_REPORT_PATH, GOLDEN_SET_PATH, POLICY_CORPUS_DIR, ROOT_DIR, SYNTHETIC_DIR, settings
from src.graph import open_graph, run_contact
from src.guardrails.output_guard import CITATION_RE, known_citations
from src.guardrails.pii import mask
from src.llm import RATE_LIMITER, llm_preflight
from src.memory.long_term import open_long_term_memory
from src.policy_limits import PLAN_BY_ID
from src.resilience import ExternalCallFailed, resilient_call
from src.tools.mcp_client import TelecomMCP

CACHE_PATH = ROOT_DIR / ".cache" / "eval_judge_cache.sqlite"
EVAL_MEMORY = ROOT_DIR / "data" / "runtime" / "eval_memory.db"
THRESHOLD = 0.5


# --- Gemini judge with cache + rate limiting ------------------------------------

class CachedGeminiJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by deepeval.models.GeminiModel, with an on-disk response cache."""

    def __init__(self, model_name: str, cache_only: bool = False):
        self.inner = GeminiModel(model=model_name, api_key=settings.google_api_key or "unused-in-cache-replay",
                                 temperature=0.0)
        self.model_name = model_name
        self.cache_only = cache_only  # judge unreachable: replay cached verdicts, never call the API
        self.db = open_cache()
        self.hits = self.misses = 0
        super().__init__(model=model_name)

    def load_model(self):
        return self.inner.load_model()

    def get_model_name(self) -> str:
        return f"{self.model_name} (gemini, cached)"

    def _key(self, prompt: str, schema) -> str:
        schema_id = json.dumps(schema.model_json_schema(), sort_keys=True) if schema is not None else ""
        return hashlib.sha256(f"{self.model_name}\n{schema_id}\n{prompt}".encode()).hexdigest()

    def _get(self, key: str, schema):
        row = self.db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        self.hits += 1
        value = json.loads(row[0])
        return (schema.model_validate(value) if schema is not None else value), 0.0

    def _put(self, key: str, result) -> None:
        value = result.model_dump() if hasattr(result, "model_dump") else result
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, self.model_name, json.dumps(value)))
        self.db.commit()

    async def a_generate(self, prompt: str, schema=None):
        key = self._key(prompt, schema)
        cached = self._get(key, schema)
        if cached is not None:
            return cached
        self.misses += 1
        if self.cache_only:
            raise RuntimeError("judge unavailable and this judge prompt is not cached (no score invented)")

        async def call():
            await RATE_LIMITER.aacquire()
            return await self.inner.a_generate(prompt, schema=schema)

        result, cost = await resilient_call(call, what="eval.judge", attempts=6, max_wait_s=60,
                                            timeout_s=settings.llm_timeout_s)
        self._put(key, result)
        return result, cost

    def generate(self, prompt: str, schema=None):
        return asyncio.get_event_loop().run_until_complete(self.a_generate(prompt, schema))


def open_cache() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, value TEXT)")
    return db


def cached_responses(model_name: str) -> int:
    return open_cache().execute("SELECT COUNT(*) FROM responses WHERE model = ?", (model_name,)).fetchone()[0]


async def judge_preflight(model_name: str) -> tuple[bool, str]:
    if not settings.google_api_key:
        return False, "GOOGLE_API_KEY not set"
    try:
        probe = GeminiModel(model=model_name, api_key=settings.google_api_key, temperature=0.0)

        async def call():
            await RATE_LIMITER.aacquire()
            return await probe.a_generate("Reply with the single word: ok")

        await resilient_call(call, what="eval.judge.preflight", attempts=3, timeout_s=60)
        return True, f"{model_name} reachable"
    except ExternalCallFailed as exc:
        return False, f"{model_name} unusable ({exc.reason})"


# --- context builders ------------------------------------------------------------

def clause_texts() -> dict[str, str]:
    out = {}
    for path in POLICY_CORPUS_DIR.glob("*.md"):
        for m in re.finditer(r"^### (POL-[A-Z]{3}-\d{3} §\d+\.\d+) — (.+?)\n\n(.+?)(?=\n\n##|\n\n###|\Z)",
                             path.read_text(), re.M | re.S):
            out[m.group(1)] = f"[{m.group(1)}] {m.group(2)}: {m.group(3).strip()}"
    return out


def account_facts(customer_id: str) -> str:
    """Ground-truth account facts from the synthetic DB (no identifiers)."""
    con = sqlite3.connect(SYNTHETIC_DIR / "telecom.db")
    con.row_factory = sqlite3.Row
    c = con.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    inv = con.execute("SELECT * FROM invoices WHERE customer_id = ? ORDER BY period_start DESC", (customer_id,)).fetchall()
    plan = PLAN_BY_ID[c["plan_id"]]
    cap = "unlimited data" if plan[5] is None else f"{plan[5]} GB data cap"
    lines = [f"Account facts: plan {plan[1]} ({plan[2]}, {plan[3]} tier, ${plan[4]:.2f}/month, {cap}); tenure "
             f"{c['tenure_months']} months; last cycle data use {c['data_used_gb']} GB; 3-month average "
             f"{c['avg_data_used_gb_3m']} GB; churn risk {c['churn_risk_label']}; complaints in last 90 days "
             f"{c['complaints_90d']}; fraud flag {bool(c['fraud_flag'])}."]
    for i in inv[:2]:
        items = ", ".join(f"{li['item']} ${li['amount']:.2f}" for li in json.loads(i["line_items_json"]))
        lines.append(f"Invoice for period starting {i['period_start']}: {items}; tax ${i['tax']:.2f}; "
                     f"total ${i['total']:.2f}.")
    return "\n".join(lines)


def retrieved_context(state: dict, clauses: dict[str, str]) -> list[str]:
    """What the agent actually retrieved/used: RAG clauses, decision clauses and tool results."""
    refs = [c["citation"] for c in state.get("policy_citations") or []]
    refs += (state.get("resolution") or {}).get("citations") or []
    for chk in (state.get("offer_decision") or {}).get("checks", []):
        refs += chk.get("policy_refs", [])
    ctx = [clauses[r] for r in dict.fromkeys(refs) if r in clauses]
    acct = state.get("account_summary")
    if acct:
        plan = acct["plan"]
        cap = "unlimited" if plan["data_cap_gb"] is None else f"{plan['data_cap_gb']} GB"
        ctx.append(f"Tool get_account: plan {plan['name']} ({plan['plan_type']}, {plan['tier']} tier, "
                   f"${plan['monthly_price']:.2f}/month, cap {cap}); tenure {acct['tenure_months']} months; "
                   f"last cycle {acct['data_used_gb_last_cycle']} GB; 3-month average {acct['avg_data_used_gb_3m']} GB; "
                   f"complaints in 90 days {acct['complaints_90d']}.")
    for inv in ((state.get("billing") or {}).get("invoices") or [])[:2]:
        items = ", ".join(f"{li['item']} ${li['amount']:.2f}" for li in inv["line_items"])
        ctx.append(f"Tool get_billing_history: period {inv['period_start']}: {items}; total ${inv['total']:.2f}.")
    for chk in (state.get("offer_decision") or {}).get("checks", []):
        ctx.append(f"Tool check_offer_eligibility: {chk['offer_type']} {chk['value']} for {chk.get('months', 1)} "
                   f"month(s) -> {chk['decision']} ({'; '.join(chk['reasons'])}).")
    return ctx or ["(no context retrieved)"]


# --- main -------------------------------------------------------------------------

async def approve(_request: dict) -> dict:
    return {"approved": True, "approver": "eval-auto-approve", "note": "golden-set evaluation"}


async def run_agent(cases: list[dict], use_llm: bool) -> list[dict]:
    rows = []
    mcp = TelecomMCP()
    async with open_graph() as app, mcp.session() as session, open_long_term_memory(EVAL_MEMORY) as memory:
        for case in cases:
            await app.checkpointer.adelete_thread(case["session_id"])
            await memory.forget_customer(case["customer_id"])
            contact = {"contact_id": case["case_id"], "customer_id": case["customer_id"],
                       "session_id": case["session_id"], "turns": [case["input"]]}
            t0 = time.perf_counter()
            result = await run_contact(app, session, contact, approve=approve, use_llm=use_llm, memory=memory)
            rows.append({"case": case, "run_id": result["run_id"], "state": result["turns"][-1],
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return rows


def deterministic(case: dict, state: dict) -> dict:
    res = state.get("resolution") or {}
    reply = state.get("final_response") or ""
    cited = set(res.get("citations") or []) | set(CITATION_RE.findall(reply))
    exp_refs = case["expected_policy_refs"]
    return {
        "intent_correct": None if case["expected_intent"] is None else state.get("intent") == case["expected_intent"],
        "action_correct": res.get("outcome") == case["expected_action"],
        "policy_ref_recall": round(sum(r in cited for r in exp_refs) / len(exp_refs), 3) if exp_refs else None,
        "citations_valid": all(c in known_citations() for c in cited),
    }


ALL_METRICS = ("hallucination", "faithfulness", "answer_relevancy")


async def judge_case(judge, case: dict, state: dict, clauses: dict[str, str], metric_names=ALL_METRICS,
                     include_reason: bool = True) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase

    reply = state.get("final_response") or ""
    context = [clauses[r] for r in case["expected_policy_refs"] if r in clauses] + [account_facts(case["customer_id"])]
    tc = LLMTestCase(input=case["input"], actual_output=reply, expected_output=case["reference_answer"],
                     context=context, retrieval_context=retrieved_context(state, clauses))
    kw = dict(model=judge, eval_mode="llm", threshold=THRESHOLD, async_mode=False, include_reason=include_reason)
    builders = {"hallucination": HallucinationMetric, "faithfulness": FaithfulnessMetric,
                "answer_relevancy": AnswerRelevancyMetric}
    metrics = {name: builders[name](**kw) for name in metric_names}
    out = {}
    for name, metric in metrics.items():
        assert isinstance(metric.model, CachedGeminiJudge), "judge must be Gemini"
        try:
            await metric.a_measure(tc, _show_indicator=False)
            out[name] = {"score": round(float(metric.score), 3), "passed": bool(metric.is_successful()),
                         "reason": mask(str(metric.reason or ""))[:400]}
        except Exception as exc:  # judge failure after retries: record, never invent a score
            out[name] = {"score": None, "passed": None, "error": f"{type(exc).__name__}: {mask(str(exc))[:200]}"}
    return out


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


EVAL_SPANS_PATH = ROOT_DIR / "traces" / "eval_spans.parquet"


async def main(limit: int | None, no_judge: bool, agent_llm: bool, trace: bool = True, judge_model: str | None = None,
               metric_names: tuple[str, ...] = ALL_METRICS, include_reason: bool = True,
               skip_preflight: bool = False) -> int:
    cases = [json.loads(l) for l in GOLDEN_SET_PATH.read_text().splitlines() if l.strip()][:limit]
    use_llm, agent_why = await llm_preflight() if agent_llm else (False, "rules/templates (default; --agent-llm for Gemini)")
    print(f"Golden set: {len(cases)} cases | agent LLM: {'Gemini' if use_llm else 'OFF: ' + agent_why}")
    if trace:  # eval runs are traced too, so eval findings can cite run_id + span_id
        from src.observability import tracing
        tracing.launch_phoenix()
        tracing.reset_project()
        tracing.init_tracing(launch=False)
        print(f"Phoenix tracing -> project '{settings.phoenix_project_name}'")
    from src.warmup import warm_up
    print(f"warm-up: {warm_up()}")
    rows = await run_agent(cases, use_llm)
    if trace:
        from scripts.export_traces import export_spans
        tracing.flush()
        export_spans(EVAL_SPANS_PATH)

    judge_model = judge_model or settings.gemini_judge_model
    if no_judge:
        judge_ok, judge_why = False, "disabled with --no-judge"
    elif skip_preflight:
        judge_ok, judge_why = True, "preflight skipped (quota-saving)"
    else:
        judge_ok, judge_why = await judge_preflight(judge_model)
    judge_mode = "live" if judge_ok else "not_run"
    if not judge_ok and not no_judge and cached_responses(judge_model):
        judge_mode, judge_why = "cache_replay", f"{judge_why}; replaying {cached_responses(judge_model)} cached verdicts"
    print(f"Judge: GeminiModel {judge_model} [{judge_mode}] ({judge_why}) metrics={list(metric_names)}")
    judge = CachedGeminiJudge(judge_model, cache_only=judge_mode == "cache_replay") if judge_mode != "not_run" else None
    clauses = clause_texts()

    per_case = []
    for row in rows:
        case, state = row["case"], row["state"]
        det = deterministic(case, state)
        scores = await judge_case(judge, case, state, clauses, metric_names, include_reason) if judge else {}
        per_case.append({
            "case_id": case["case_id"], "category": case["category"], "run_id": row["run_id"],
            "expected_intent": case["expected_intent"], "predicted_intent": state.get("intent"),
            "expected_action": case["expected_action"], "predicted_action": (state.get("resolution") or {}).get("outcome"),
            "expected_policy_refs": case["expected_policy_refs"],
            "cited": (state.get("resolution") or {}).get("citations"),
            "engines": sorted({f"{e['node']}:{e['engine']}" for e in state.get("engine_log") or []}),
            "latency_ms": row["latency_ms"], **det,
            "judge": scores or {"status": "not_run", "reason": judge_why},
            "output_preview": mask(state.get("final_response") or "")[:240],
        })
        j = " ".join(f"{k}={v.get('score')}" for k, v in scores.items()) if scores else "judge=not_run"
        print(f"  {case['case_id']} {case['category']:<10} intent={'✓' if det['intent_correct'] else '✗' if det['intent_correct'] is False else '-'} "
              f"action={'✓' if det['action_correct'] else '✗'} refs={det['policy_ref_recall']} {j}")

    def metric_agg(name: str) -> dict:
        scores = [c["judge"].get(name, {}).get("score") for c in per_case if isinstance(c["judge"].get(name), dict)]
        passed = [c["judge"][name].get("passed") for c in per_case if isinstance(c["judge"].get(name), dict)]
        scored = [s for s in scores if s is not None]
        if not scores:
            return {"status": "not_run", "reason": judge_why if name in metric_names else "metric not selected"}
        if not scored:
            first = next(c["judge"][name].get("error") for c in per_case if isinstance(c["judge"].get(name), dict))
            return {"status": "not_run", "reason": first, "errors": len(scores)}
        return {"status": "ran" if len(scored) == len(scores) else "partial", "judge_model": judge_model,
                "mean_score": _mean(scored), "pass_rate": _mean([float(p) for p in passed if p is not None]),
                "scored_cases": len(scored), "errors": len(scores) - len(scored)}

    intents = [c["intent_correct"] for c in per_case if c["intent_correct"] is not None]
    halluc = metric_agg("hallucination")
    aggregate = {
        "cases": len(per_case),
        "intent_accuracy": _mean([float(x) for x in intents]),
        "action_accuracy": _mean([float(c["action_correct"]) for c in per_case]),
        "policy_ref_recall": _mean([c["policy_ref_recall"] for c in per_case]),
        "citation_validity": _mean([float(c["citations_valid"]) for c in per_case]),
        "hallucination": halluc,
        # DeepEval 4.x HallucinationMetric: 1 = no contradicted context (pass), 0 = contradicted; threshold is
        # a minimum (the metric flipped direction in 4.x). hallucination_rate = share of SCORED cases failing it;
        # see hallucination.scored_cases / errors for coverage.
        "hallucination_rate": (round(1 - halluc["pass_rate"], 3) if halluc.get("pass_rate") is not None else None),
        "faithfulness": metric_agg("faithfulness"),
        "answer_relevancy": metric_agg("answer_relevancy"),
        "latency_ms_mean": _mean([c["latency_ms"] for c in per_case]),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/run_eval.py", "golden_set": "data/golden_set.jsonl",
        "agent_engine": "gemini" if use_llm else "rules/templates",
        "judge": {"provider": "google-gemini", "class": "deepeval.models.GeminiModel via CachedGeminiJudge",
                  "model": judge_model, "eval_mode": "llm", "mode": judge_mode,
                  "status": "ran" if judge else "not_run", "metrics": list(metric_names),
                  "include_reason": include_reason,
                  "reason": judge_why, "cache": str(CACHE_PATH.relative_to(ROOT_DIR)),
                  "cache_hits": judge.hits if judge else 0, "cache_misses": judge.misses if judge else 0},
        "threshold": THRESHOLD, "run_ids": [c["run_id"] for c in per_case],
        "traces": str(EVAL_SPANS_PATH.relative_to(ROOT_DIR)) if trace else None,
        "phoenix_project": settings.phoenix_project_name if trace else None,
        "aggregate": aggregate, "cases": per_case,
    }
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\naggregate: intent_accuracy={aggregate['intent_accuracy']} action_accuracy={aggregate['action_accuracy']} "
          f"policy_ref_recall={aggregate['policy_ref_recall']} citation_validity={aggregate['citation_validity']}")
    for m in ("hallucination", "faithfulness", "answer_relevancy"):
        print(f"  {m}: {aggregate[m]}")
    print(f"-> {EVAL_REPORT_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    ap.add_argument("--agent-llm", action="store_true", help="run the agent with Gemini (default: rules)")
    ap.add_argument("--no-trace", action="store_true", help="do not trace eval runs in Phoenix")
    ap.add_argument("--judge-model", help="Gemini judge model (default GEMINI_JUDGE_MODEL)")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS), help="comma list of judge metrics")
    ap.add_argument("--no-reason", action="store_true", help="skip judge reason calls (saves quota)")
    ap.add_argument("--skip-judge-preflight", action="store_true", help="do not spend a call on the preflight")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.limit, a.no_judge, a.agent_llm, not a.no_trace, a.judge_model,
                                      tuple(m for m in a.metrics.split(",") if m), not a.no_reason,
                                      a.skip_judge_preflight)))
