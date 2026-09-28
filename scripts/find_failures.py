"""Mine the project's evidence for REAL failures and print citable candidates.

Sources and what counts as a candidate:
  traces/phoenix_spans.parquet, traces/eval_spans.parquet
      span status ERROR; latency outliers (copilot.turn spans > 5x the warm median)
  logs/tool_calls.jsonl        tool status timeout / error / denied (deliberate probes by the
                               exercise scripts are labelled `expected_probe`, not failures)
  logs/agent_actions.jsonl     loop_limit, account_lookup_failed, offer_selection_failed, ticket_failed,
                               output_guard_intervention (guardrail blocks are listed as defenses, not failures)
  reports/eval_report.json     wrong intent / wrong action vs the golden set, judge not run, low faithfulness
  reports/redteam_results.json attacks that were not detected, or that caused harm
  logs/memory_test.log         FAIL lines
Retries: resilient_call does not persist retry attempts, so retries cannot be mined (reported as such).

Each candidate carries run_id, trace_id, span_id(s) and `file:Lnn` references to the records.
--snapshot NAME copies every mined source file to evidence/NAME/ (with a sha256 manifest) and rewrites
the references to point there, so a failure write-up keeps resolving after evidence is regenerated.

Run: python -m scripts.find_failures [--snapshot pre_fix]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

SOURCES = ["traces/phoenix_spans.parquet", "traces/eval_spans.parquet", "logs/tool_calls.jsonl",
           "logs/agent_actions.jsonl", "logs/mcp_transcript.jsonl", "logs/memory_test.log",
           "reports/eval_report.json", "reports/redteam_results.json", "reports/golden_signals.json",
           "reports/regenerate_summary.json", "logs/regenerate/13_pii.log"]  # step logs with raw console output are NOT copied
FAILURE_ACTIONS = {"loop_limit", "account_lookup_failed", "offer_selection_failed", "output_guard_intervention"}


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


class Miner:
    def __init__(self, base: Path):
        self.base = base
        self.spans = pd.concat([self._spans("traces/phoenix_spans.parquet"), self._spans("traces/eval_spans.parquet")],
                               ignore_index=True)
        self.tool_lines = _lines(base / "logs/tool_calls.jsonl")
        self.audit_lines = _lines(base / "logs/agent_actions.jsonl")

    def _spans(self, rel: str) -> pd.DataFrame:
        p = self.base / rel
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df["_source"] = rel
        df["_ms"] = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds() * 1000
        return df

    def ref(self, rel: str, line: int) -> str:
        return f"{self.rel(rel)}:L{line}"

    def rel(self, rel: str) -> str:
        return str((self.base / rel).relative_to(ROOT_DIR))

    def records_for_run(self, run_id: str, limit: int = 6) -> dict:
        tool = [self.ref("logs/tool_calls.jsonl", i) for i, l in enumerate(self.tool_lines, 1) if run_id in l][:limit]
        audit = [self.ref("logs/agent_actions.jsonl", i) for i, l in enumerate(self.audit_lines, 1) if run_id in l][:limit]
        return {"tool_calls": tool, "agent_actions": audit}

    def spans_for_run(self, run_id: str, names: tuple[str, ...] = ("copilot.turn",)) -> dict:
        if self.spans.empty or "attributes.copilot.run_id" not in self.spans.columns:
            return {}
        s = self.spans[self.spans["attributes.copilot.run_id"] == run_id]
        if s.empty:
            return {}
        out = {"trace_id": s["context.trace_id"].iloc[0], "source": self.rel(s["_source"].iloc[0]), "spans": {}}
        for n in names:
            hit = s[s["name"] == n]
            if not hit.empty:
                out["spans"][n] = {"span_id": hit["context.span_id"].iloc[0], "latency_ms": round(float(hit["_ms"].iloc[0]), 1)}
        return out

    # --- miners ------------------------------------------------------------------------

    def span_errors(self) -> list[dict]:
        if self.spans.empty:
            return []
        err = self.spans[self.spans.get("status_code") == "ERROR"]
        return [{"category": "span_error", "run_id": r.get("attributes.copilot.run_id"), "trace_id": r["context.trace_id"],
                 "span_id": r["context.span_id"], "span": r["name"], "source": self.rel(r["_source"]),
                 "detail": str(r.get("status_message"))[:200]} for _, r in err.iterrows()]

    def latency_outliers(self) -> list[dict]:
        out = []
        for src, g in self.spans.groupby("_source") if not self.spans.empty else []:
            turns = g[g["name"] == "copilot.turn"].sort_values("start_time")
            if len(turns) < 3:
                continue
            warm = turns["_ms"].iloc[1:].median()
            for _, r in turns[turns["_ms"] > 5 * warm].iterrows():
                out.append({"category": "latency_outlier", "run_id": r["attributes.copilot.run_id"],
                            "contact_id": r.get("attributes.copilot.contact_id"), "trace_id": r["context.trace_id"],
                            "span_id": r["context.span_id"], "span": "copilot.turn", "source": self.rel(src),
                            "latency_ms": round(float(r["_ms"]), 1), "warm_median_ms": round(float(warm), 1),
                            "is_first_turn_of_process": bool(r["context.span_id"] == turns["context.span_id"].iloc[0])})
        return out

    def tool_failures(self) -> list[dict]:
        out = []
        for i, line in enumerate(self.tool_lines, 1):
            r = json.loads(line)
            if r["status"] != "ok":
                out.append({"category": "tool_" + r["status"], "run_id": r["run_id"], "tool": r["tool_name"],
                            "record": self.ref("logs/tool_calls.jsonl", i),
                            "error_code": (r.get("result") or {}).get("error", {}).get("code") if isinstance(r.get("result"), dict) else None,
                            "expected_probe": r.get("agent") == "exercise_mcp"})
        return out

    def audit_findings(self) -> tuple[list[dict], dict]:
        out, defenses = [], {}
        for i, line in enumerate(self.audit_lines, 1):
            r = json.loads(line)
            a = r["action"]
            if a == "guardrail_block":
                defenses.setdefault(r["reason"], []).append(self.ref("logs/agent_actions.jsonl", i))
            if a in FAILURE_ACTIONS or (a == "escalation" and r["decision"] == "ticket_failed"):
                out.append({"category": a, "run_id": r["run_id"], "decision": r["decision"], "reason": r["reason"],
                            "record": self.ref("logs/agent_actions.jsonl", i)})
        return out, defenses

    def eval_findings(self) -> list[dict]:
        p = self.base / "reports/eval_report.json"
        if not p.exists():
            return []
        ev = json.loads(p.read_text())
        text = p.read_text().splitlines()
        out = []
        if ev["judge"]["status"] != "ran":
            line = next(i for i, l in enumerate(text, 1) if '"reason"' in l)
            out.append({"category": "eval_judge_not_run", "reason": ev["judge"]["reason"], "model": ev["judge"]["model"],
                        "record": self.ref("reports/eval_report.json", line)})
        for c in ev["cases"]:
            line = next(i for i, l in enumerate(text, 1) if f'"case_id": "{c["case_id"]}"' in l)
            wrong = [k for k in ("intent_correct", "action_correct") if c.get(k) is False]
            faith = (c.get("judge") or {}).get("faithfulness") or {}
            if wrong or (isinstance(faith, dict) and faith.get("score") is not None and faith["score"] < 0.5):
                out.append({"category": "eval_wrong_" + "_and_".join(w.split("_")[0] for w in wrong) if wrong else "eval_low_faithfulness",
                            "case_id": c["case_id"], "run_id": c["run_id"], "expected_intent": c["expected_intent"],
                            "predicted_intent": c["predicted_intent"], "expected_action": c["expected_action"],
                            "predicted_action": c["predicted_action"], "record": self.ref("reports/eval_report.json", line),
                            "trace": self.spans_for_run(c["run_id"], ("copilot.turn", "intent_agent", "clarify")),
                            "logs": self.records_for_run(c["run_id"])})
        return out

    def redteam_findings(self) -> list[dict]:
        p = self.base / "reports/redteam_results.json"
        if not p.exists():
            return []
        rt = json.loads(p.read_text())
        text = p.read_text().splitlines()
        out = []
        for r in rt["results"]:
            if not r["detected"] or not r["pass"]:
                line = next(i for i, l in enumerate(text, 1) if f'"attack_id": "{r["attack_id"]}"' in l)
                out.append({"category": "redteam_undetected" if r["pass"] else "redteam_harm", "attack_id": r["attack_id"],
                            "technique": r["technique"], "run_id": r["run_id"], "outcome": r["outcome"],
                            "violations": r["violations"], "record": self.ref("reports/redteam_results.json", line),
                            "logs": self.records_for_run(r["run_id"])})
        return out

    def pipeline_findings(self) -> list[dict]:
        p = self.base / "reports/regenerate_summary.json"
        if not p.exists():
            return []
        text = p.read_text().splitlines()
        out = []
        for s in json.loads(p.read_text())["steps"]:
            if s["status"].startswith("FAILED"):
                line = next(i for i, l in enumerate(text, 1) if f'"name": "{s["name"]}"' in l)
                out.append({"category": "pipeline_step_failed", "step": s["name"], "detail": s["detail"],
                            "record": self.ref("reports/regenerate_summary.json", line),
                            "step_log": self.rel(f"logs/regenerate/{s['step']:02d}_{s['name']}.log")})
        return out

    def memory_test_findings(self) -> list[dict]:
        return [{"category": "memory_test_fail", "record": self.ref("logs/memory_test.log", i), "line": l.strip()}
                for i, l in enumerate(_lines(self.base / "logs/memory_test.log"), 1) if "[FAIL]" in l]


def snapshot(name: str) -> Path:
    dest = ROOT_DIR / "evidence" / name
    if dest.exists():
        shutil.rmtree(dest)
    manifest = {"snapshot": name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator": "scripts/find_failures.py --snapshot", "files": {}}
    for rel in SOURCES:
        src = ROOT_DIR / rel
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
            manifest["files"][rel] = {"sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "bytes": src.stat().st_size}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="freeze the mined evidence under evidence/<name>/ and cite that copy")
    a = ap.parse_args()
    base = snapshot(a.snapshot) if a.snapshot else ROOT_DIR
    m = Miner(base)
    audit, defenses = m.audit_findings()
    tool = m.tool_failures()
    found = {
        "span_errors": m.span_errors(), "latency_outliers": m.latency_outliers(),
        "tool_failures": [t for t in tool if not t["expected_probe"]], "expected_tool_probes": [t for t in tool if t["expected_probe"]],
        "audit_failures": audit, "eval": m.eval_findings(), "redteam": m.redteam_findings(),
        "memory_test": m.memory_test_findings(), "pipeline": m.pipeline_findings(),
    }
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "generator": "scripts/find_failures.py",
              "base": str(base.relative_to(ROOT_DIR)) if base != ROOT_DIR else ".",
              "note_retries": "retry attempts are not persisted by src/resilience.py, so they cannot be mined",
              "guardrail_blocks_by_reason": {k: {"count": len(v), "first": v[0]} for k, v in defenses.items()},
              "candidates": found}
    out = (base / "failure_candidates.json") if a.snapshot else (ROOT_DIR / "reports" / "failure_candidates.json")
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"evidence base: {report['base']}")
    for cat, items in found.items():
        print(f"\n== {cat}: {len(items)}")
        for it in items[:8]:
            keys = ("case_id", "attack_id", "contact_id", "tool", "span", "error_code", "reason", "latency_ms",
                    "predicted_intent", "expected_intent", "predicted_action", "technique", "step", "detail")
            desc = " ".join(f"{k}={it[k]}" for k in keys if it.get(k) is not None)
            ids = f"run_id={it.get('run_id')}"
            if it.get("trace_id"):
                ids += f" trace_id={it['trace_id']} span_id={it['span_id']}"
            elif (it.get("trace") or {}).get("trace_id"):
                t = it["trace"]
                ids += f" trace_id={t['trace_id']} spans=" + ",".join(f"{n}:{s['span_id']}" for n, s in t["spans"].items())
            print(f"  - {it['category']}: {desc}\n      {ids}\n      record={it.get('record')} logs={it.get('logs', '')}")
        if len(items) > 8:
            print(f"  ... {len(items) - 8} more")
    print(f"\nguardrail blocks (defenses, not failures): { {k: v['count'] for k, v in report['guardrail_blocks_by_reason'].items()} }")
    print(f"retries: {report['note_retries']}")
    print(f"-> {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
