"""Regenerate ALL evidence with one command. Stops at the first failing step and exits nonzero.

  python -m scripts.regenerate_evidence [--llm] [--skip-benchmark]

Steps (each a committed script; console output of every step is saved under logs/regenerate/):
   1 data          synthetic data, policy corpus, sample/golden/red-team sets (only if missing)
   2 index         local Chroma index over the policy corpus
   3 phoenix       start the in-process Phoenix app on :6006 (kept running for all later steps)
   4 tools         exercise every MCP tool/resource + policy RAG (resets tool_calls / mcp_transcript)
   5 contacts      run all sample contacts, approvals --auto-approve (decisions recorded in the audit log)
                   and export traces -> traces/phoenix_spans.parquet
   6 redteam       adversarial set -> reports/redteam_results.json
   7 eval          golden set + DeepEval Gemini judge (traced) -> reports/eval_report.json
   8 tests         pytest (routing, loops, tool contracts, memory -> logs/memory_test.log, guardrails)
   9 signals       golden signals -> reports/golden_signals.json
  10 dashboard     reports/dashboard_data.csv, dashboard_charts.png, dashboard.png (Phoenix screenshot)
  11 benchmark     optimization before/after -> reports/optimization_note.md (skip with --skip-benchmark)
  12 api           FastAPI SSE streaming demo -> logs/api_demo.log (bonus)
  13 reconcile     tool names across code / logs / spans -> reports/tool_reconciliation.json
  14 pii           plaintext identifier scan of logs/, traces/, reports/, evidence/
  15 citations     every citation in docs/*.md and README.md resolves to a committed artifact

Default is deterministic (rules/templates, no Gemini) so it reproduces without quota; --llm lets the
agent use Gemini when the preflight passes. The eval judge always tries Gemini and records the outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.guardrails.pii import mask
from src.config import AGENT_ACTIONS_LOG, GOLDEN_SET_PATH, LOGS_DIR, REPORTS_DIR, ROOT_DIR, SAMPLE_CONTACTS_PATH, SYNTHETIC_DIR

PY = sys.executable
# Judge pinned so reruns replay the on-disk judge cache instead of spending free-tier quota.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-3.6-flash")
STEP_LOGS = LOGS_DIR / "regenerate"


def steps(args) -> list[tuple[str, list[str] | None, dict]]:
    llm = [] if args.llm else ["--no-llm"]
    data_missing = not all(p.exists() for p in (SYNTHETIC_DIR / "telecom.db", SAMPLE_CONTACTS_PATH, GOLDEN_SET_PATH,
                                                 ROOT_DIR / "data" / "redteam_set.jsonl"))
    s = [
        ("data", [PY, "-m", "scripts.generate_synthetic_data"] if data_missing else None, {}),
        ("index", [PY, "-m", "scripts.build_policy_index"], {}),
        ("phoenix", None, {}),
        ("tools", [PY, "-m", "scripts.exercise_tools", "--reset", *llm], {}),
        ("contacts", [PY, "-m", "src.cli", "run", "--input", "data/sample_contacts.jsonl", "--auto-approve",
                      "--export-traces", *llm], {}),
        ("redteam", [PY, "-m", "scripts.run_redteam", *([] if not args.llm else ["--llm"])], {}),
        ("eval", [PY, "-m", "scripts.run_eval", "--judge-model", EVAL_JUDGE_MODEL, "--no-reason",
                  *(["--agent-llm"] if args.llm else [])],
         {"PHOENIX_PROJECT_NAME": "retention-copilot-eval"}),
        ("tests", [PY, "-m", "pytest", "-q", "-p", "no:warnings"], {}),
        ("signals", [PY, "-m", "scripts.golden_signals"], {}),
        ("dashboard", [PY, "-m", "scripts.dashboard"], {}),
        ("benchmark", None if args.skip_benchmark else [PY, "-m", "scripts.optimization_benchmark"], {}),
        ("api", [PY, "-m", "scripts.api_demo"], {}),
        ("reconcile", [PY, "-m", "scripts.reconcile_tools"], {}),
        ("pii", [PY, "-m", "scripts.check_pii_leaks"], {}),
        ("citations", [PY, "-m", "scripts.verify_citations"], {}),
    ]
    return s


def last_line(text: str) -> str:
    noise = ("HF Hub", "🌍", "💽", "📖", "Warning", "⚠️", "boto3", "fork", "ev_poll")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not any(n in l for n in noise)]
    return lines[-1][:110] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="let the agent use Gemini (default: deterministic)")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()
    STEP_LOGS.mkdir(parents=True, exist_ok=True)
    results, started = [], time.perf_counter()
    print(f"regenerate_evidence: {'Gemini allowed' if args.llm else 'deterministic (no Gemini for the agent)'}\n")

    for i, (name, cmd, env) in enumerate(steps(args), start=1):
        t0 = time.perf_counter()
        if name == "phoenix":
            from src.observability import tracing
            status, detail = "ok", tracing.launch_phoenix()
        elif cmd is None:
            status, detail = "skipped", "already present" if name == "data" else "skipped by flag"
        else:
            if name == "tools":
                AGENT_ACTIONS_LOG.unlink(missing_ok=True)  # the audit log restarts with this regeneration
            proc = subprocess.run(cmd, cwd=ROOT_DIR, env={**os.environ, **env}, capture_output=True, text=True,
                                  timeout=3600)
            raw = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
            (STEP_LOGS / f"{i:02d}_{name}.log").write_text(mask(raw, amounts=True))  # step logs live under logs/: masked
            status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            detail = mask(last_line(proc.stdout) or last_line(proc.stderr), amounts=True)
        secs = time.perf_counter() - t0
        results.append({"step": i, "name": name, "status": status, "seconds": round(secs, 1), "detail": detail})
        print(f"  {i:>2}. {name:<10} {status:<18} {secs:6.1f}s  {detail}")
        if status.startswith("FAILED"):
            print(f"\nstopped: step '{name}' failed; see {(STEP_LOGS / f'{i:02d}_{name}.log').relative_to(ROOT_DIR)}")
            break

    ok = all(not r["status"].startswith("FAILED") for r in results) and len(results) == len(steps(args))
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "generator": "scripts/regenerate_evidence.py", "mode": "llm" if args.llm else "deterministic",
               "status": "PASS" if ok else "FAIL", "total_seconds": round(time.perf_counter() - started, 1),
               "steps": results}
    (REPORTS_DIR / "regenerate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(r['status'] == 'ok' for r in results)} ok, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped, "
          f"{sum(r['status'].startswith('FAILED') for r in results)} failed in {summary['total_seconds']}s "
          f"-> reports/regenerate_summary.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
