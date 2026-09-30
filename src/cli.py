"""Command-line interface.

  python -m src.cli demo                         guided demo of representative scenarios (no API key needed)
  python -m src.cli chat --customer CUST-000397  interactive support chat (commands: help, account, history,
                                                 policy, reset, quit)
  python -m src.cli run --input data/sample_contacts.jsonl [--auto-approve | --auto-reject] [--details]

`run` drives every sample contact end-to-end (interactive approval by default; the flags make it
non-interactive and reproducible) and writes masked results to reports/sample_run_results.jsonl.
The contact's customer_id stands in for an upstream authenticated session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from src import ui
from src.config import REPORTS_DIR, ROOT_DIR, RUNTIME_DIR, SAMPLE_CONTACTS_PATH
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
DEMO_SCENARIOS = [  # (sample contact, narration)
    ("CT-0001", "A customer asks why their bill went up. The copilot reads the invoice through the secure "
                "account tool and explains it with policy citations."),
    ("CT-0003", "A customer keeps running out of data. The copilot recommends a plan based on real usage."),
    ("CT-0006", "An at-risk customer demands half off. 50% is above the policy ceiling, so it is blocked; the "
                "strongest compliant offer is worth more than the auto-approval limit, so a team lead must approve."),
    ("CT-0009", "A prompt-injection attempt ('ignore previous instructions ... 90% off'). The input guard refuses it "
                "before any tool is called."),
    ("CT-0010", "A customer asks for someone else's bill. Cross-customer access is refused."),
    ("CT-0014", "A cancellation on an account flagged for fraud review: every offer is blocked and the case goes "
                "to a human specialist."),
]


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
        ui.console.print()
        ui.console.print(ui.approval_prompt_text(request))
        while True:
            try:
                answer = await asyncio.to_thread(input, "  Approve this offer? [y]es / [n]o: ")
            except EOFError:
                answer = "n"
            decision = ui.parse_yes_no(answer)
            if decision is not None:
                break
            ui.status(ui.WARN, "Please answer y (approve) or n (reject).")
        ui.status(ui.OK if decision else ui.BAD, "Approved by the team lead." if decision else "Rejected by the team lead.")
        return {"approved": decision, "approver": "cli-interactive", "note": ""}
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


def _redirect_logs(logs_dir) -> TelecomMCP:
    """Scratch/benchmark/demo runs never touch the project's evidence logs."""
    from src.audit import audit_middleware
    from src.tools import logging_middleware
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    logging_middleware.set_log_path(logs / "tool_calls.jsonl")
    audit_middleware.set_log_path(logs / "agent_actions.jsonl")
    return TelecomMCP(transcript_path=logs / "mcp_transcript.jsonl")


async def _prepare(no_llm: bool) -> tuple[bool, str]:
    use_llm, why = (False, "disabled with --no-llm") if no_llm else await llm_preflight()
    from src.warmup import warm_up
    t0 = time.perf_counter()
    with ui.console.status("Loading local models (embeddings, PII detector)…"):
        warm_up()
    ui.status(ui.OK, f"Local models ready ({time.perf_counter() - t0:.1f}s).")
    ui.engine_notice(use_llm, why)
    return use_llm, why


async def cmd_run(args) -> int:
    contacts = [json.loads(line) for line in open(args.input) if line.strip()]
    if args.only:
        contacts = [c for c in contacts if c["contact_id"] in args.only]
    mode = "approve" if args.auto_approve else "reject" if args.auto_reject else "interactive"
    ui.banner("Customer Service & Retention Copilot", f"Processing {len(contacts)} customer contact(s) · "
              f"approvals: {'automatic (approve)' if mode == 'approve' else 'automatic (reject)' if mode == 'reject' else 'you decide'}")
    if not args.no_trace:  # observability is part of the run path, not just imported
        url = tracing.launch_phoenix()
        if not args.keep_traces:
            tracing.reset_project()
        tracing.init_tracing(launch=False)
        ui.status(ui.OK, f"Tracing to Arize Phoenix: {url} (project '{tracing.settings.phoenix_project_name}')")
    use_llm, _why = await _prepare(args.no_llm)
    mcp = _redirect_logs(args.logs_dir) if args.logs_dir else TelecomMCP()
    summaries = []
    mcp_ctx = mcp.per_call_session() if args.mcp_mode == "per-call" else mcp.session()
    if args.mcp_mode != "session":
        ui.status(ui.INFO, f"MCP mode: {args.mcp_mode} (benchmark baseline)")
    async with open_graph() as app, mcp_ctx as session, open_long_term_memory() as memory:
        if not args.keep_memory:  # reproducible runs: start each customer's long-term memory empty
            await asyncio.gather(*(memory.forget_customer(c) for c in {c["customer_id"] for c in contacts}))
        for i, contact in enumerate(contacts, start=1):
            if not args.keep_history:
                await app.checkpointer.adelete_thread(contact["session_id"])
            with ui.console.status(f"[{i}/{len(contacts)}] {contact['contact_id']}: working…"):
                result = await run_contact(app, session, contact, approve=approver(mode),
                                           max_steps=args.max_steps, memory=memory, use_llm=use_llm) \
                    if mode != "interactive" else None
            if result is None:  # interactive approval cannot run under a spinner
                result = await run_contact(app, session, contact, approve=approver(mode),
                                           max_steps=args.max_steps, memory=memory, use_llm=use_llm)
            s = _summarize(contact, result)
            summaries.append(s)
            ui.contact_report(f"{contact['contact_id']} · {contact['scenario'].replace('_', ' ')} · "
                              f"{mask_customer_id(contact['customer_id'])}",
                              "  /  ".join(contact["turns"]), result["turns"][-1],
                              expected=contact.get("expected_outcome"), matched=_expected_match(s),
                              details=args.details, run_id=result["run_id"])

    results_path = RESULTS_PATH if not args.results_path else Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as fh:
        for s in summaries:
            fh.write(json.dumps(mask_obj(s, amounts=True), ensure_ascii=False) + "\n")  # amounts tied to a person masked
    tracing.flush()
    if args.export_traces and tracing.tracing_enabled():
        from scripts.export_traces import export_spans
        export_spans(Path(args.export_path).resolve() if args.export_path else None)
    ui.console.print()
    ui.console.print(ui.summary_table([{**s, "match": _expected_match(s)} for s in summaries]))
    matched = sum(_expected_match(s) for s in summaries)
    print(f"{matched}/{len(summaries)} contacts matched the expected outcome; "
          f"results -> {_display_path(results_path)}")
    return 0


async def cmd_demo(args) -> int:
    ui.banner("Customer Service & Retention Copilot · guided demo",
              "Six real scenarios run through the full agent graph: guardrails → intent → secure account "
              "lookup (MCP) → policy search (RAG) → offer check → approval gate → reply.\n"
              "All customers are synthetic. Identifiers are masked everywhere.")
    contacts = {c["contact_id"]: c for c in (json.loads(l) for l in SAMPLE_CONTACTS_PATH.read_text().splitlines())}
    if args.trace:
        url = tracing.launch_phoenix()
        tracing.reset_project()
        tracing.init_tracing(launch=False)
        ui.status(ui.OK, f"Tracing to Arize Phoenix: {url}")
    use_llm, _ = await _prepare(args.no_llm)
    mode = "interactive" if args.interactive else "approve"
    if mode == "approve":
        ui.status(ui.INFO, "Approvals: an automated stand-in approves (use --interactive to decide yourself).")
    mcp = _redirect_logs(RUNTIME_DIR / "demo_logs")
    async with open_graph(RUNTIME_DIR / "demo_checkpoints.db") as app, mcp.session() as session, \
            open_long_term_memory(RUNTIME_DIR / "demo_memory.db") as memory:
        for i, (cid, story) in enumerate(DEMO_SCENARIOS, start=1):
            contact = contacts[cid]
            await app.checkpointer.adelete_thread(contact["session_id"])
            await memory.forget_customer(contact["customer_id"])
            ui.console.print()
            ui.section(f"Scenario {i} of {len(DEMO_SCENARIOS)}")
            ui.console.print(ui.Text(story, style="italic"))
            if mode == "approve":
                with ui.console.status("Working…"):
                    result = await run_contact(app, session, contact, approve=approver(mode), memory=memory,
                                               use_llm=use_llm)
            else:
                result = await run_contact(app, session, contact, approve=approver(mode), memory=memory,
                                           use_llm=use_llm)
            ui.contact_report(f"{cid} · {mask_customer_id(contact['customer_id'])}", "  /  ".join(contact["turns"]),
                              result["turns"][-1], details=args.details, run_id=result["run_id"])
            if args.pause and i < len(DEMO_SCENARIOS):
                try:
                    await asyncio.to_thread(input, "\nPress Enter for the next scenario…")
                except EOFError:
                    pass
    ui.console.print()
    ui.status(ui.OK, "Demo complete. Try it yourself:  python -m src.cli chat --customer CUST-000397")
    return 0


CHAT_HELP = """[bold]Commands[/bold]
  [cyan]help[/cyan]              show this help
  [cyan]account[/cyan]           your plan, usage and account summary (masked)
  [cyan]history[/cyan]           the conversation so far
  [cyan]policy[/cyan] \\[question] look up company policy (e.g. [italic]policy when do credits need approval?[/italic])
  [cyan]reset[/cyan]             start a new conversation
  [cyan]quit[/cyan] / [cyan]exit[/cyan]       leave the chat
Anything else is sent to the copilot as your message.

[bold]Try asking[/bold]
  • Why is my bill higher this month?
  • I keep running out of data. Which plan should I be on?
  • I want to cancel unless you can give me a discount.
  • My signal keeps dropping at home and nothing has been fixed."""


def parse_command(text: str) -> tuple[str, str]:
    """('help'|'account'|'history'|'policy'|'reset'|'quit'|'empty'|'message', argument)."""
    t = (text or "").strip()
    if not t:
        return "empty", ""
    head, _, rest = t.partition(" ")
    cmd = head.lower().lstrip("/")
    if cmd in ("quit", "exit", "bye", "q"):
        return "quit", ""
    if cmd in ("help", "?", "account", "history", "reset") and not rest:
        return ("help" if cmd == "?" else cmd), ""
    if cmd == "policy":
        return "policy", rest.strip()
    return "message", t


async def cmd_chat(args) -> int:
    session_id = args.session or f"CHAT-{uuid.uuid4().hex[:8].upper()}"
    mode = "approve" if args.auto_approve else "reject" if args.auto_reject else "interactive"
    ui.banner("Customer Support Chat", AI_DISCLOSURE)
    if not args.no_trace:
        tracing.init_tracing()
        ui.status(ui.OK, f"Conversation traced in Arize Phoenix: {tracing.phoenix_url()}")
    mcp = TelecomMCP()
    async with open_graph() as app, mcp.session() as session, open_long_term_memory() as memory:
        from src.agents.common import Deps
        from src.run_context import llm_enabled_var
        from src.tools.rag_tool import policy_rag_tool

        use_llm, _ = await _prepare(no_llm=False)
        llm_enabled_var.set(use_llm)
        deps = Deps(mcp=session, max_steps=args.max_steps, memory=memory, use_llm=use_llm)
        ui.status(ui.INFO, f"Signed in as customer {mask_customer_id(args.customer)}. Type 'help' for commands, "
                           "or just describe what you need.")
        last_message = ""
        while True:
            token = mcp.issue_session_token(args.customer, session_id)
            try:
                text = await asyncio.to_thread(input, "\nYou › ")
            except (EOFError, KeyboardInterrupt):
                break
            cmd, arg = parse_command(text)
            if cmd == "quit":
                break
            if cmd == "empty":
                ui.status(ui.INFO, "Type a message, or 'help' for commands.")
                continue
            if cmd == "help":
                ui.console.print(ui.Panel(CHAT_HELP, border_style="blue", title="Help", title_align="left"))
                continue
            if cmd == "reset":
                session_id = f"CHAT-{uuid.uuid4().hex[:8].upper()}"
                ui.status(ui.OK, "Started a new conversation.")
                continue
            try:
                with run_scope(session_id=session_id, mcp_auth_token=token) as rid:
                    if cmd == "account":
                        out = await session.call("get_account", customer_id=args.customer)
                        if out.get("ok"):
                            ui.console.print(ui.Panel("\n".join(ui.account_lines(out["account"])),
                                                      title="Your account", title_align="left", border_style="blue"))
                        else:
                            ui.status(ui.BAD, "Your account could not be loaded right now.")
                        continue
                    if cmd == "history":
                        snap = await app.aget_state({"configurable": {"thread_id": session_id}})
                        msgs = (snap.values or {}).get("messages", [])
                        if not msgs:
                            ui.status(ui.INFO, "No messages in this conversation yet.")
                        for m in msgs:
                            who = "You" if m.type == "human" else "Copilot"
                            ui.console.print(ui.Text(f"{who}: ", style="bold") + ui.Text(ui.safe(str(m.content))))
                        continue
                    if cmd == "policy":
                        question = arg or last_message
                        if not question:
                            ui.status(ui.INFO, "Ask a policy question, e.g. 'policy when does a credit need approval?'")
                            continue
                        with ui.console.status("Searching company policy…"):
                            out = await policy_rag_tool.ainvoke({"question": ui.safe(question)[:500]})
                        ui.console.print(ui.Panel(ui.Text(out["answer"]), title="Policy", title_align="left",
                                                  border_style="cyan"))
                        table = ui.citations_table([c["citation"] for c in out["citations"]])
                        if table:
                            ui.console.print(table)
                        continue
                    last_message = arg
                    if mode == "interactive":
                        state = await run_turn(app, text=arg, customer_id=args.customer, session_id=session_id,
                                               run_id=rid, deps=deps, approve=approver(mode), contact_id=session_id)
                    else:
                        with ui.console.status("Working on it…"):
                            state = await run_turn(app, text=arg, customer_id=args.customer, session_id=session_id,
                                                   run_id=rid, deps=deps, approve=approver(mode), contact_id=session_id)
                tracing.flush()
                _chat_reply(state, args.details, rid)
            except Exception as exc:  # never crash the conversation
                ui.status(ui.BAD, f"Something went wrong handling that ({type(exc).__name__}). Please try again, "
                                  "or type 'help'.")
    ui.status(ui.OK, "Goodbye. Thanks for contacting us.")
    return 0


def _chat_reply(state: dict, details: bool, run_id: str) -> None:
    res = state.get("resolution") or {}
    icon, color, label = ui.OUTCOME_LABELS.get(res.get("outcome"), (ui.INFO, "white", ""))
    ui.console.print(ui.Panel(ui.Text(state.get("final_response") or ""), title=f"Copilot · {icon} {label}",
                              title_align="left", border_style=color, padding=(0, 1)))
    if state.get("guard_blocked"):
        ui.status(ui.BAD, ui.GUARD_EXPLAIN.get(state.get("guard_reason"), "Blocked by a safety check."), style="dim")
    blocked = [b for b in (state.get("offer_decision") or {}).get("blocked", []) if b.get("source") == "customer_request"]
    for b in blocked:
        ui.status(ui.BAD, f"Your request ({ui.describe_offer(b)}) is above what policy allows "
                          f"({', '.join(b['policy_refs'][:2])}).", style="dim")
    if (state.get("offer_proposal") or {}).get("needs_approval"):
        ui.console.print(ui.approval_line(state))
    if res.get("ticket_id"):
        ui.status(ui.INFO, f"Ticket {res['ticket_id']} opened; a specialist will follow up.", style="dim")
    table = ui.citations_table(res.get("citations") or [])
    if table:
        ui.console.print(table)
    if details:
        ui.console.print(ui.Text(f"details: run {run_id} · intent {state.get('intent')} · steps "
                                 f"{state.get('step_count')}", style="dim"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.cli", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Customer Service & Retention Copilot: a guarded, audited multi-agent assistant for telecom care.",
        epilog="examples:\n"
               "  python -m src.cli demo                                   guided demo (no API key needed)\n"
               "  python -m src.cli chat --customer CUST-000397            talk to the copilot as a customer\n"
               "  python -m src.cli run --input data/sample_contacts.jsonl --auto-approve   batch run + evidence")
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
    chat = sub.add_parser("chat", help="interactive support chat as one customer")
    chat.add_argument("--customer", required=True, help="synthetic customer ID, e.g. CUST-000397")
    chat.add_argument("--session", help="resume an earlier conversation")
    chat.add_argument("--max-steps", type=int, default=8)
    demo = sub.add_parser("demo", help="guided demo of six representative scenarios")
    demo.add_argument("--interactive", action="store_true", help="you approve/reject offers yourself")
    demo.add_argument("--pause", action="store_true", help="wait for Enter between scenarios")
    demo.add_argument("--trace", action="store_true", help="also trace the demo in Phoenix")
    demo.add_argument("--no-llm", action="store_true", help="deterministic rules/templates only")
    demo.add_argument("--details", action="store_true", help="show internal run details")
    for p in (run, chat):
        p.add_argument("--no-trace", action="store_true", help="disable Phoenix tracing")
        p.add_argument("--details", action="store_true", help="show internal run details (run id, engines)")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--auto-approve", action="store_true")
        g.add_argument("--auto-reject", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "chat" and not mask(args.customer, amounts=False).startswith("CUST-***"):
        ap.error("--customer must look like CUST-123456 (see data/sample_contacts.jsonl for synthetic customers)")
    tracing_on = (args.cmd != "demo" and not args.no_trace) or (args.cmd == "demo" and args.trace)
    if tracing_on:
        with ui.console.status("Starting Arize Phoenix (first start can take a minute)…"):
            tracing.launch_phoenix()  # start the in-process app before the event loop starts
    commands = {"run": cmd_run, "chat": cmd_chat, "demo": cmd_demo}
    try:
        return asyncio.run(commands[args.cmd](args))
    except KeyboardInterrupt:
        ui.status(ui.INFO, "Stopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
