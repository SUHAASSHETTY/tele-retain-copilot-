"""Command-line interface.

  python -m src.cli run --input data/sample_contacts.jsonl [--auto-approve | --auto-reject]
  python -m src.cli chat --customer CUST-000397 [--session SES-...]

`run` drives every sample contact end-to-end (interactive approval by default; the flags make it
non-interactive and reproducible) and writes masked results to reports/sample_run_results.jsonl.
The contact's customer_id stands in for an upstream authenticated session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections import Counter

from src.config import REPORTS_DIR, ROOT_DIR
from src.graph import open_graph, run_contact, run_turn
from src.guardrails.pii import mask, mask_customer_id, mask_obj
from src.llm import llm_preflight
from src.memory.long_term import open_long_term_memory
from src.observability import tracing
from src.run_context import run_scope
from src.tools.mcp_client import TelecomMCP

RESULTS_PATH = REPORTS_DIR / "sample_run_results.jsonl"
# EU AI Act Art. 50(1)-style transparency: people must know they are interacting with an AI system.
AI_DISCLOSURE = ("You are chatting with an AI assistant for (synthetic) telecom customer care. It can make "
                 "mistakes; discounts or credits above the approval threshold are reviewed by a human, and you "
                 "can ask for a human agent at any time.")


def _display_path(path) -> str:
    """Project-relative when inside the project, otherwise just the file name (no local paths printed)."""
    path = path.resolve()
    return str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else path.name


def approver(mode: str):
    async def approve(request: dict) -> dict:
        if mode == "approve":
            return {"approved": True, "approver": "cli-auto-approve", "note": "non-interactive run"}
        if mode == "reject":
            return {"approved": False, "approver": "cli-auto-reject", "note": "non-interactive run"}
        print("\n  >>> HUMAN APPROVAL REQUIRED")
        print(f"      {request['offer_type']} value={request['value']} months={request['months']} "
              f"offer_value=${request['offer_value_usd']:.2f} refs={request['policy_refs']}")
        answer = (await asyncio.to_thread(input, "      Approve? [y/N] ")).strip().lower()
        return {"approved": answer in ("y", "yes"), "approver": "cli-interactive", "note": ""}
    return approve


def _summarize(contact: dict, result: dict) -> dict:
    last = result["turns"][-1]
    res = last.get("resolution") or {}
    engines = Counter(f"{e['node']}:{e['engine']}" for e in last.get("engine_log") or [])  # cumulative per thread
    return {
        "contact_id": contact["contact_id"], "scenario": contact["scenario"], "run_id": result["run_id"],
        "customer_ref": mask_customer_id(contact["customer_id"]), "turns": len(contact["turns"]),
        "expected_intent": contact.get("expected_intent"), "intent": last.get("intent"),
        "intent_confidence": last.get("intent_confidence"),
        "expected_outcome": contact.get("expected_outcome"), "outcome": res.get("outcome"),
        "reason": res.get("reason"), "input_flags": last.get("input_flags"),
        "output_flags": last.get("output_flags"),
        "offer": {k: (last.get("offer_proposal") or {}).get(k) for k in ("offer_type", "value", "months",
                                                                         "decision", "policy_refs")}
        if last.get("offer_proposal") else None,
        "blocked_offers": (last.get("offer_decision") or {}).get("blocked"),
        "approvals": [a["decision"] for t in result["turns"] for a in t.get("approvals", [])],
        "citations": res.get("citations"), "ticket_id": res.get("ticket_id"),
        "steps": last.get("step_count"), "engines": dict(engines),
        "session_facts": [f["content"] for f in last.get("session_facts") or []],
        "memories_recalled": [{"kind": m["kind"], "content": m["content"]} for m in last.get("memories") or []],
        "memory_written": last.get("memory_written"), "context_stats": last.get("context_stats"),
        "errors": [e for t in result["turns"] for e in t.get("errors") or []],
        "final_response": last.get("final_response"),
        "output_risk_tier": (last.get("output_risk") or {}).get("tier"),
    }


def _expected_match(s: dict) -> bool:
    exp, got = s["expected_outcome"], s["outcome"]
    if exp == "offer_pending_approval":  # after the gate the outcome is offer (approved) or resolve (rejected)
        return bool(s["approvals"])
    return exp == got


async def cmd_run(args) -> int:
    contacts = [json.loads(line) for line in open(args.input) if line.strip()]
    if args.only:
        contacts = [c for c in contacts if c["contact_id"] in args.only]
    mode = "approve" if args.auto_approve else "reject" if args.auto_reject else "interactive"
    if not args.no_trace:  # observability is part of the run path, not just imported
        url = tracing.launch_phoenix()
        if not args.keep_traces:
            tracing.reset_project()
        tracing.init_tracing(launch=False)
        print(f"Phoenix tracing -> {url} (project '{tracing.settings.phoenix_project_name}')")
    use_llm, why = (False, "disabled with --no-llm") if args.no_llm else await llm_preflight()
    from src.warmup import warm_up
    print(f"warm-up: {warm_up()}")
    print(f"Running {len(contacts)} contact(s) | LLM: {'Gemini (' + why + ')' if use_llm else 'OFF: ' + why + ' -> rules/templates'}"
          f" | approval: {mode}")
    if args.logs_dir:  # benchmark / scratch runs never touch the project's evidence logs
        from pathlib import Path

        from src.audit import audit_middleware
        from src.tools import logging_middleware
        logs = Path(args.logs_dir)
        logs.mkdir(parents=True, exist_ok=True)
        logging_middleware.set_log_path(logs / "tool_calls.jsonl")
        audit_middleware.set_log_path(logs / "agent_actions.jsonl")
        mcp = TelecomMCP(transcript_path=logs / "mcp_transcript.jsonl")
    else:
        mcp = TelecomMCP()
    summaries = []
    mcp_ctx = mcp.per_call_session() if args.mcp_mode == "per-call" else mcp.session()
    print(f"MCP mode: {args.mcp_mode}")
    async with open_graph() as app, mcp_ctx as session, open_long_term_memory() as memory:
        if not args.keep_memory:  # reproducible runs: start each customer's long-term memory empty
            cleared = sum([await memory.forget_customer(c) for c in {c["customer_id"] for c in contacts}])
            print(f"cleared {cleared} long-term memories for {len({c['customer_id'] for c in contacts})} customer(s)")
        for contact in contacts:
            if not args.keep_history:
                await app.checkpointer.adelete_thread(contact["session_id"])
            result = await run_contact(app, session, contact, approve=approver(mode), max_steps=args.max_steps,
                                       memory=memory, use_llm=use_llm)
            s = _summarize(contact, result)
            summaries.append(s)
            ok = "OK " if _expected_match(s) else "DIFF"
            print(f"\n[{ok}] {s['contact_id']} {s['scenario']} ({s['customer_ref']}, run {s['run_id'][:8]})")
            print(f"      intent: {s['intent']} ({s['intent_confidence']}) expected {s['expected_intent']} | "
                  f"outcome: {s['outcome']} expected {s['expected_outcome']} | steps={s['steps']}")
            if s["input_flags"]:
                print(f"      input_flags: {s['input_flags']}")
            if s["blocked_offers"]:
                print(f"      blocked: {[(b['offer_type'], b['value'], b['policy_refs']) for b in s['blocked_offers']]}")
            if s["offer"]:
                print(f"      offer: {s['offer']}")
            if s["approvals"]:
                print(f"      approvals: {s['approvals']}")
            if s["memories_recalled"]:
                print(f"      memories recalled: {[m['content'] for m in s['memories_recalled']]}")
            print(f"      engines: {s['engines']}")
            print(f"      reply: {s['final_response']}")

    results_path = RESULTS_PATH if not args.results_path else __import__("pathlib").Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as fh:
        for s in summaries:
            fh.write(json.dumps(mask_obj(s, amounts=True), ensure_ascii=False) + "\n")  # amounts tied to a person masked
    tracing.flush()
    if args.export_traces and tracing.tracing_enabled():
        from scripts.export_traces import export_spans
        export_spans(__import__("pathlib").Path(args.export_path).resolve() if args.export_path else None)
    matched = sum(_expected_match(s) for s in summaries)
    print(f"\n{matched}/{len(summaries)} contacts matched the expected outcome; "
          f"results -> {_display_path(results_path)}")
    return 0


async def cmd_chat(args) -> int:
    session_id = args.session or f"CHAT-{uuid.uuid4().hex[:8].upper()}"
    print(AI_DISCLOSURE)
    print(f"Chat as {mask_customer_id(args.customer)} (session {session_id}). Empty line or 'quit' to exit.")
    mode = "approve" if args.auto_approve else "reject" if args.auto_reject else "interactive"
    if not args.no_trace:
        tracing.init_tracing()
    mcp = TelecomMCP()
    async with open_graph() as app, mcp.session() as session, open_long_term_memory() as memory:
        from src.agents.common import Deps
        token = mcp.issue_session_token(args.customer, session_id)
        use_llm, why = await llm_preflight()
        print(f"LLM: {'Gemini' if use_llm else 'OFF'} ({why})")
        deps = Deps(mcp=session, max_steps=args.max_steps, memory=memory, use_llm=use_llm)
        from src.run_context import llm_enabled_var
        llm_enabled_var.set(use_llm)
        with run_scope(session_id=session_id, mcp_auth_token=token) as rid:
            while True:
                try:
                    text = (await asyncio.to_thread(input, "you> ")).strip()
                except EOFError:
                    break
                if not text or text.lower() in ("quit", "exit"):
                    break
                state = await run_turn(app, text=text, customer_id=args.customer, session_id=session_id,
                                       run_id=rid, deps=deps, approve=approver(mode), contact_id=session_id)
                tracing.flush()
                print(f"copilot> {state.get('final_response')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description="Customer Service & Retention Copilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run contacts from a JSONL file")
    run.add_argument("--input", required=True)
    run.add_argument("--only", nargs="*", help="contact_ids to run")
    run.add_argument("--keep-history", action="store_true", help="continue existing checkpointed threads")
    run.add_argument("--keep-memory", action="store_true", help="keep existing long-term memories")
    run.add_argument("--no-llm", action="store_true", help="deterministic rules/templates only")
    run.add_argument("--keep-traces", action="store_true", help="do not clear the Phoenix project first")
    run.add_argument("--export-traces", action="store_true", help="write traces/phoenix_spans.parquet at the end")
    run.add_argument("--export-path", help="alternative parquet path for --export-traces")
    run.add_argument("--results-path", help="alternative results JSONL path")
    run.add_argument("--logs-dir", help="write tool/audit/MCP logs here instead of logs/")
    run.add_argument("--mcp-mode", choices=["session", "per-call"], default="session",
                     help="session = one persistent MCP session per run (default); per-call = baseline")
    run.add_argument("--max-steps", type=int, default=8)
    chat = sub.add_parser("chat", help="interactive chat as one customer")
    chat.add_argument("--customer", required=True)
    chat.add_argument("--session")
    chat.add_argument("--max-steps", type=int, default=8)
    for p in (run, chat):
        p.add_argument("--no-trace", action="store_true", help="disable Phoenix tracing")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--auto-approve", action="store_true")
        g.add_argument("--auto-reject", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "chat" and not mask(args.customer, amounts=False).startswith("CUST-***"):
        ap.error("--customer must look like CUST-123456")
    if not args.no_trace:
        tracing.launch_phoenix()  # start the in-process app before the event loop starts
    return asyncio.run(cmd_run(args) if args.cmd == "run" else cmd_chat(args))


if __name__ == "__main__":
    sys.exit(main())
