"""Exercise every tool (MCP + policy RAG) through the logging middleware, producing
logs/tool_calls.jsonl and logs/mcp_transcript.jsonl from code.

Run: python -m scripts.exercise_tools [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from collections import Counter

from scripts import exercise_mcp
from src.config import MCP_TRANSCRIPT_LOG, SYNTHETIC_DIR, TOOL_CALLS_LOG
from src.llm import llm_preflight
from src.run_context import llm_enabled_var
from src.run_context import agent_scope, run_scope
from src.tools.rag_tool import policy_rag_tool

RAG_QUESTIONS = [
    "What is the maximum loyalty discount for a premium tier customer at high churn risk?",
    "When does a retention credit or discount need human approval?",
    "How should a request to see another customer's account be handled?",
    "What are the acknowledgement and resolution SLAs for a P2 complaint?",
    "customer wants to leave, what now",                            # vague -> rewrite loop
    "What is the policy on international device insurance claims?",  # not covered by the corpus
]


async def run_rag(no_llm: bool) -> None:
    use_llm, why = (False, "disabled with --no-llm") if no_llm else await llm_preflight()
    llm_enabled_var.set(use_llm)
    print(f"\npolicy_rag (judge: {'gemini' if use_llm else 'heuristic -- ' + why})")
    for i, q in enumerate(RAG_QUESTIONS, start=1):
        with run_scope(session_id=f"SES-RAG-{i}"), agent_scope("policy_retrieval_agent"):
            out = await policy_rag_tool.ainvoke({"question": q})
        cites = [c["citation"] for c in out["citations"]]
        print(f"  {q[:62]:<62} -> {out['status']:<18} attempts={out['attempts']} "
              f"grader={out['grader']} citations={cites}")
        if out["attempts"] > 1:
            print(f"      queries: {out['queries']}")


async def main(reset: bool, no_llm: bool = False) -> None:
    if reset:
        TOOL_CALLS_LOG.unlink(missing_ok=True)
        MCP_TRANSCRIPT_LOG.unlink(missing_ok=True)
    await exercise_mcp.main(reset=False)
    await run_rag(no_llm)

    records = [json.loads(line) for line in TOOL_CALLS_LOG.read_text().splitlines()]
    print(f"\nlogs/tool_calls.jsonl: {len(records)} records")
    print(f"  by tool:   {dict(Counter(r['tool_name'] for r in records))}")
    print(f"  by status: {dict(Counter(r['status'] for r in records))}")
    required = {"timestamp", "run_id", "agent", "tool_name", "args", "result", "latency_ms", "status", "error"}
    missing = [i for i, r in enumerate(records) if not required <= r.keys()]
    print(f"  schema check (all required fields present): {'PASS' if not missing else f'FAIL {missing}'}")

    con = sqlite3.connect(SYNTHETIC_DIR / "telecom.db")
    ids = [v for row in con.execute("SELECT customer_id, account_number, phone, email FROM customers")
           for v in row]
    text = TOOL_CALLS_LOG.read_text()
    leaks = [v for v in ids if v in text]
    print(f"  plaintext PII leak check over {len(ids)} identifiers: {'PASS' if not leaks else f'FAIL {len(leaks)}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="truncate tool_calls.jsonl and mcp_transcript.jsonl")
    ap.add_argument("--no-llm", action="store_true", help="heuristic RAG judge only")
    a = ap.parse_args()
    asyncio.run(main(a.reset, a.no_llm))
