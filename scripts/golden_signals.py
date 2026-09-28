"""Golden signals from the exported Phoenix spans (+ quality from the eval report).

Reads traces/phoenix_spans.parquet and reports/eval_report.json and writes reports/golden_signals.json:
  latency   p50 / p95 / mean per span class: thinking (LLM), acting (AGENT/CHAIN/GUARDRAIL), tool
            (TOOL/RETRIEVER), plus end-to-end turn latency
  traffic   runs, turns, spans, LLM calls, tool calls
  errors    spans with ERROR status, tool calls with non-ok status
  tokens    prompt / completion / total from LLM span attributes (llm.token_count.*)
  cost      tokens x price table (src/observability/cost.py, prices dated in that file)
  per_run   totals per run_id
  quality   intent / action accuracy, hallucination rate, faithfulness, relevancy (from the eval)

Run: python -m scripts.golden_signals [--spans PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import EVAL_REPORT_PATH, GOLDEN_SIGNALS_PATH, PHOENIX_SPANS_PATH, ROOT_DIR
from src.observability.cost import PRICE_PAGE_UPDATED, PRICE_RETRIEVED, PRICE_SOURCE, PRICE_TABLE, price_for, token_cost

KIND_CLASS = {"LLM": "thinking", "AGENT": "acting", "CHAIN": "acting", "GUARDRAIL": "acting",
              "TOOL": "tool", "RETRIEVER": "tool"}  # UNKNOWN (tool_middleware.*) -> "other"


def _json(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v[:1] == "{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return {}


def _llm_field(row: pd.Series, *path: str):
    """llm.* attribute from either flattened columns or the nested JSON column."""
    flat = "attributes.llm." + ".".join(path)
    if flat in row.index and pd.notna(row[flat]):
        return row[flat]
    node = _json(row.get("attributes.llm"))
    for p in path:
        node = node.get(p, {}) if isinstance(node, dict) else {}
    return node if node != {} else None


def span_table(df: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per span: identity, class, latency, tokens and cost (used by the dashboard too)."""
    kind = df["span_kind"].fillna("UNKNOWN")
    start, end = pd.to_datetime(df["start_time"]), pd.to_datetime(df["end_time"])
    out = pd.DataFrame({
        "span_id": df["context.span_id"], "trace_id": df["context.trace_id"],
        "run_id": df.get("attributes.copilot.run_id"), "contact_id": None,
        "agent": df.get("copilot_agent"), "name": df["name"], "span_kind": kind,
        "span_class": kind.map(lambda k: KIND_CLASS.get(k, "other")),
        "start_time": start, "latency_ms": ((end - start).dt.total_seconds() * 1000).round(3),
        "status": df.get("status_code", pd.Series(["UNSET"] * len(df), index=df.index)),
        "tool_status": df.get("attributes.copilot.tool_status"),
    })
    # contact id lives on the turn span; propagate it to every span of that trace
    turn = df["name"] == "copilot.turn"
    if "attributes.copilot.contact_id" in df.columns:
        contact_by_trace = df[turn].set_index("context.trace_id")["attributes.copilot.contact_id"].to_dict()
        out["contact_id"] = out["trace_id"].map(contact_by_trace)
        run_by_trace = df[turn].set_index("context.trace_id")["attributes.copilot.run_id"].to_dict()
        out["run_id"] = out["run_id"].fillna(out["trace_id"].map(run_by_trace))
    llm_rows = df[kind == "LLM"]
    out["model"] = None
    out["prompt_tokens"] = 0
    out["completion_tokens"] = 0
    for idx, row in llm_rows.iterrows():
        out.at[idx, "model"] = _llm_field(row, "model_name")
        out.at[idx, "prompt_tokens"] = int(_llm_field(row, "token_count", "prompt") or 0)
        out.at[idx, "completion_tokens"] = int(_llm_field(row, "token_count", "completion") or 0)
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    out["cost_usd"] = [token_cost(p, c, m) if k == "LLM" else 0.0
                       for p, c, m, k in zip(out["prompt_tokens"], out["completion_tokens"], out["model"], kind)]
    return out.reset_index(drop=True)


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {"count": int(s.size), "p50_ms": round(float(s.quantile(0.5)), 1),
            "p95_ms": round(float(s.quantile(0.95)), 1), "mean_ms": round(float(s.mean()), 1)}


def compute(spans_path: Path = PHOENIX_SPANS_PATH, eval_path: Path = EVAL_REPORT_PATH) -> dict:
    t = span_table(pd.read_parquet(spans_path))
    llm = t[t["span_kind"] == "LLM"]
    tools = t[t["span_kind"] == "TOOL"]
    middleware = t[t["name"].str.startswith("tool_middleware.")]
    turns = t[t["name"] == "copilot.turn"].sort_values("start_time")
    warm_turns = turns.iloc[1:] if len(turns) > 1 else turns
    models = sorted(m for m in llm["model"].dropna().unique())
    notes = []
    if llm.empty:
        notes.append("No LLM spans in this export: the traced run used deterministic rules/templates (Gemini was "
                     "unavailable: model/quota), so thinking latency, tokens and cost are empty rather than estimated.")

    per_run = []
    for run_id, g in t.dropna(subset=["run_id"]).groupby("run_id"):
        per_run.append({
            "run_id": run_id, "contact_id": next((c for c in g["contact_id"].dropna()), None),
            "turns": int((g["name"] == "copilot.turn").sum()), "spans": int(len(g)),
            "e2e_latency_ms": round(float(g[g["name"] == "copilot.turn"]["latency_ms"].sum()), 1),
            "llm_calls": int((g["span_kind"] == "LLM").sum()), "tool_calls": int((g["span_kind"] == "TOOL").sum()),
            "prompt_tokens": int(g["prompt_tokens"].sum()), "completion_tokens": int(g["completion_tokens"].sum()),
            "cost_usd": round(float(g["cost_usd"].sum()), 6),
        })
    per_run.sort(key=lambda r: r["contact_id"] or "")

    quality = {"source": str(eval_path.relative_to(ROOT_DIR)), "status": "missing"}
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        agg = ev["aggregate"]
        quality = {
            "source": str(eval_path.relative_to(ROOT_DIR)), "generated_at": ev["generated_at"],
            "cases": agg["cases"], "agent_engine": ev["agent_engine"],
            "intent_accuracy": agg["intent_accuracy"], "action_accuracy": agg["action_accuracy"],
            "policy_ref_recall": agg["policy_ref_recall"], "citation_validity": agg["citation_validity"],
            "hallucination_rate": agg["hallucination_rate"], "hallucination": agg["hallucination"],
            "faithfulness": agg["faithfulness"], "answer_relevancy": agg["answer_relevancy"],
            "judge": {k: ev["judge"][k] for k in ("model", "status", "reason")},
        }
        if ev["judge"]["status"] != "ran":
            notes.append(f"LLM-as-judge metrics not available: {ev['judge']['reason']} (see eval_report.json).")
        for m in ("hallucination", "faithfulness", "answer_relevancy"):
            st = agg[m].get("status")
            if st == "partial":
                notes.append(f"{m}: judged on only {agg[m]['scored_cases']} of {agg['cases']} cases "
                             f"({agg[m]['errors']} judge calls failed: quota / 503); treat the rate as indicative only.")
            elif st == "not_run":
                notes.append(f"{m}: not run ({agg[m].get('reason')}).")
        quality["hallucination_coverage"] = (f"{agg['hallucination'].get('scored_cases', 0)}/{agg['cases']} cases judged"
                                             if agg["hallucination"].get("status") != "not_run" else "not judged")

    tokens = {"prompt": int(t["prompt_tokens"].sum()), "completion": int(t["completion_tokens"].sum()),
              "total": int(t["total_tokens"].sum()), "llm_spans": int(len(llm)),
              "per_llm_call_mean": round(float(llm["total_tokens"].mean()), 1) if len(llm) else None}
    runs = max(len(per_run), 1)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/golden_signals.py",
        "sources": {"spans": str(spans_path.relative_to(ROOT_DIR)), "eval": str(eval_path.relative_to(ROOT_DIR))},
        "span_classes": {"thinking": ["LLM"], "acting": ["AGENT", "CHAIN", "GUARDRAIL"], "tool": ["TOOL", "RETRIEVER"]},
        "latency": {
            "thinking": _stats(t[t["span_class"] == "thinking"]["latency_ms"]),
            "acting": _stats(t[t["span_class"] == "acting"]["latency_ms"]),
            "tool": _stats(t[t["span_class"] == "tool"]["latency_ms"]),
            "tool_by_name": {n: _stats(g["latency_ms"]) for n, g in tools.groupby("name")},
            "end_to_end_turn": _stats(turns["latency_ms"]),
            "end_to_end_turn_warm": _stats(warm_turns["latency_ms"]),
            "cold_start_first_turn_ms": round(float(turns["latency_ms"].iloc[0]), 1) if len(turns) else None,
        },
        "traffic": {"runs": len(per_run), "turns": int(len(turns)), "spans": int(len(t)),
                    "llm_calls": int(len(llm)), "tool_calls": int(len(tools)),
                    "spans_by_kind": t["span_kind"].value_counts().to_dict()},
        "errors": {"error_spans": int((t["status"] == "ERROR").sum()),
                   "error_span_rate": round(float((t["status"] == "ERROR").mean()), 4),
                   "tool_calls_not_ok": int((middleware["tool_status"].fillna("unknown") != "ok").sum()),
                   "tool_status_counts": middleware["tool_status"].fillna("unknown").value_counts().to_dict(),
                   "tool_status_source": "tool_middleware.* spans (logging middleware)"},
        "tokens": tokens,
        "cost": {"currency": "USD", "total": round(float(t["cost_usd"].sum()), 6),
                 "per_run_mean": round(float(t["cost_usd"].sum()) / runs, 6),
                 "models_seen": models, "pricing_basis": {m: price_for(m)[1] for m in models},
                 "price_table_usd_per_1m_tokens": PRICE_TABLE, "price_source": PRICE_SOURCE,
                 "price_page_updated": PRICE_PAGE_UPDATED, "price_retrieved": PRICE_RETRIEVED},
        "per_run": per_run,
        "quality": quality,
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, default=PHOENIX_SPANS_PATH)
    ap.add_argument("--eval", type=Path, default=EVAL_REPORT_PATH)
    ap.add_argument("--out", type=Path, default=GOLDEN_SIGNALS_PATH)
    a = ap.parse_args()
    gs = compute(a.spans.resolve(), a.eval.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(gs, indent=2, default=str) + "\n")
    lat = gs["latency"]
    print(f"{a.out.resolve().relative_to(ROOT_DIR)} <- {gs['sources']['spans']}")
    for k in ("thinking", "acting", "tool", "end_to_end_turn", "end_to_end_turn_warm"):
        print(f"  latency {k:<16} {lat[k]}")
    print(f"  traffic {gs['traffic']['runs']} runs, {gs['traffic']['turns']} turns, {gs['traffic']['spans']} spans, "
          f"{gs['traffic']['llm_calls']} LLM / {gs['traffic']['tool_calls']} tool calls")
    print(f"  errors  {gs['errors']}")
    print(f"  tokens  {gs['tokens']} | cost ${gs['cost']['total']} total")
    q = gs["quality"]
    print(f"  quality intent={q.get('intent_accuracy')} action={q.get('action_accuracy')} "
          f"hallucination_rate={q.get('hallucination_rate')}")
    for n in gs["notes"]:
        print(f"  NOTE: {n}")


if __name__ == "__main__":
    main()
