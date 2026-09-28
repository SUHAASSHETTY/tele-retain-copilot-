"""Helpers to run copilot turns against explicit checkpoint / memory stores (used by the memory
tests). Run as a module to execute one session in a separate OS process:

    python -m tests.memory_driver --checkpoints X --memory Y --logs Z --customer C --session S --text T
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from src.audit import audit_middleware
from src.graph import open_graph, run_contact
from src.memory.long_term import open_long_term_memory
from src.run_context import llm_enabled_var
from src.tools import logging_middleware
from src.tools.mcp_client import TelecomMCP


async def _reject(_request: dict) -> dict:
    return {"approved": False, "approver": "test", "note": "memory test"}


def isolate_logs(logs_dir: Path) -> None:
    """Send tool/audit logs to a scratch dir so tests never touch the project's evidence."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging_middleware.set_log_path(logs_dir / "tool_calls.jsonl")
    audit_middleware.set_log_path(logs_dir / "agent_actions.jsonl")


async def run_session(*, checkpoints: Path, memory_path: Path, logs_dir: Path, customer_id: str,
                      session_id: str, turns: list[str]) -> dict:
    """Open a fresh graph instance + stores, run the turns (rules only, no LLM), return a summary."""
    isolate_logs(logs_dir)
    llm_enabled_var.set(False)
    mcp = TelecomMCP(transcript_path=logs_dir / "mcp_transcript.jsonl")
    async with open_graph(checkpoints) as app, mcp.session() as session, \
            open_long_term_memory(memory_path) as memory:
        contact = {"contact_id": session_id, "customer_id": customer_id, "session_id": session_id, "turns": turns}
        result = await run_contact(app, session, contact, approve=_reject, use_llm=False, memory=memory)
        snapshot = await app.aget_state({"configurable": {"thread_id": session_id}})
        stored = await memory.all(customer_id)
    turns_out = []
    for t in result["turns"]:
        turns_out.append({
            "intent": t.get("intent"), "outcome": (t.get("resolution") or {}).get("outcome"),
            "reply": t.get("final_response"),
            "session_facts": [f["content"] for f in t.get("session_facts") or []],
            "memories": [{"kind": m["kind"], "content": m["content"], "score": m.get("score"),
                          "selected_by": m.get("selected_by", "semantic")} for m in t.get("memories") or []],
            "memory_written": t.get("memory_written") or [],
            "context_stats": t.get("context_stats"), "summary": t.get("summary"),
            "message_count": len(t.get("messages") or []),
            "message_texts": [str(m.content) for m in t.get("messages") or []],
            "customer_texts": [str(m.content) for m in t.get("messages") or [] if m.type == "human"],
        })
    return {"pid": os.getpid(), "run_id": result["run_id"], "session_id": session_id, "turns": turns_out,
            "thread_message_count": len(snapshot.values.get("messages", [])),
            "stored_memories": [{"kind": m["kind"], "content": m["content"], "session_id": m["session_id"]}
                                for m in stored]}


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--checkpoints", "--memory", "--logs", "--customer", "--session"):
        ap.add_argument(a, required=True)
    ap.add_argument("--text", required=True, action="append")
    a = ap.parse_args()
    out = asyncio.run(run_session(checkpoints=Path(a.checkpoints), memory_path=Path(a.memory),
                                  logs_dir=Path(a.logs), customer_id=a.customer, session_id=a.session,
                                  turns=a.text))
    print("RESULT_JSON=" + json.dumps(out))


if __name__ == "__main__":
    main()
