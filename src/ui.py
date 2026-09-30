"""Terminal presentation for the copilot (rich). Pure formatting: no graph, policy or tool logic here.

Everything shown passes through src.guardrails.pii (customer IDs, account numbers, emails, phones and
cards masked). Internal identifiers (run ids, engine logs) are shown only with --details.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.config import POLICY_CORPUS_DIR
from src.guardrails.pii import mask, mask_customer_id
from src.policy_limits import APPROVAL_THRESHOLD

# Real terminals use their own width; piped output (logs, CI) gets a readable fixed width.
console = Console(highlight=False, soft_wrap=False, width=None if sys.stdout.isatty() else 110)

OK, BAD, WAIT, INFO, WARN = "✔", "✖", "⏳", "ℹ", "⚠"

INTENT_LABELS = {
    "billing_query": "Billing question", "plan_change": "Plan / data question", "complaint": "Complaint",
    "cancellation": "Cancellation / retention", "ambiguous": "Unclear request", "out_of_scope": "Out of scope",
    None: "Not classified (stopped by a safety check)",
}
OUTCOME_LABELS = {
    "resolve": (OK, "green", "Resolved"),
    "offer": (OK, "green", "Retention offer made"),
    "offer_pending_approval": (WAIT, "yellow", "Offer waiting for a team lead's approval"),
    "escalate": (INFO, "cyan", "Handed over to a human specialist"),
    "clarify": (INFO, "cyan", "Asked the customer to clarify"),
    "decline": (INFO, "cyan", "Politely declined (outside telecom support)"),
    "refuse": (BAD, "red", "Request refused by a safety / privacy check"),
}
GUARD_EXPLAIN = {
    "prompt_injection": "The message tried to override the assistant's instructions, so it was treated as an attack "
                        "and refused (no tools were called).",
    "cross_customer_request": "The message asked about another customer's account. Only the verified account "
                              "holder's data can be discussed.",
    "safety_threat": "The message contained a threat, so it was handed to a human immediately.",
    "input_too_long": "The message was too long to process safely.",
}
OFFER_WORDS = {"discount_pct": "discount", "credit": "bill credit", "data_boost": "data boost",
               "plan_upgrade": "plan upgrade", "fee_waiver": "fee waiver"}
NODE_LABELS = {
    "input_guard": "Checked the message for safety and privacy",
    "context_manager": "Loaded conversation context and memory",
    "supervisor": "Planned the next step",
    "intent_agent": "Understood the request",
    "account_agent": "Looked up the account (secure tool call)",
    "policy_retrieval_agent": "Searched company policy",
    "retention_offer_agent": "Checked offers against policy limits",
    "resolution_agent": "Drafted a resolution",
    "request_approval": "Requested a team lead's approval",
    "human_approval": "Recorded the approval decision",
    "clarify": "Prepared a clarifying question",
    "escalate": "Opened a ticket for a human specialist",
    "memory_writer": "Saved useful notes for next time",
    "output_guard": "Checked the reply for privacy and policy limits",
}


@lru_cache(maxsize=1)
def clause_titles() -> dict[str, str]:
    """'POL-RET-003 §1.2' -> 'Human approval required' (from the policy corpus headings)."""
    titles = {}
    for p in POLICY_CORPUS_DIR.glob("*.md"):
        for cid, title in re.findall(r"^### (POL-[A-Z]{3}-\d{3} §\d+\.\d+) — (.+)$", p.read_text(), re.M):
            titles[cid] = title.strip()
    return titles


def safe(text: str | None) -> str:
    """Mask identifiers for display (amounts stay: the verified customer sees their own bill)."""
    return mask(text or "", amounts=False)


def money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


# --- generic blocks -----------------------------------------------------------------

def banner(title: str, subtitle: str = "") -> None:
    body = Text(title, style="bold")
    if subtitle:
        body.append("\n" + subtitle, style="dim")
    console.print(Panel(body, box=box.DOUBLE, border_style="blue", padding=(1, 2)))


def section(title: str) -> None:
    console.print(Rule(Text(title, style="bold"), style="blue"))


def status(icon: str, message: str, style: str = "") -> None:
    colors = {OK: "green", BAD: "red", WAIT: "yellow", INFO: "cyan", WARN: "yellow"}
    console.print(Text(f"{icon} ", style=colors.get(icon, "")) + Text(message, style=style))


def engine_notice(use_llm: bool, why: str) -> None:
    if use_llm:
        status(OK, f"AI engine: Google Gemini ({why}).")
    else:
        status(INFO, "AI engine: built-in rules & templates (Gemini not in use: "
                     f"{why}). Every policy check, guardrail and approval gate still applies.")


def describe_offer(o: dict, plan_name: str | None = None) -> str:
    kind = o.get("offer_type")
    v, months = o.get("value"), o.get("months")
    period = f" for {months} month(s)" if months else ""  # unknown duration is omitted, never guessed
    if kind == "discount_pct":
        return f"{v:g}% off{(' ' + plan_name) if plan_name else ''}{period}"
    if kind == "credit":
        return f"a one-time bill credit of {money(v)}"
    if kind == "data_boost":
        return f"+{v:g} GB data per month{period}, free"
    if kind == "plan_upgrade":
        return f"next-tier plan at the current price{period}"
    return OFFER_WORDS.get(kind, str(kind))


def citations_table(cites: list[str]) -> Table | None:
    if not cites:
        return None
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold", pad_edge=False)
    t.add_column("Policy clause", style="cyan", no_wrap=True)
    t.add_column("What it says")
    for c in dict.fromkeys(cites):
        t.add_row(c, clause_titles().get(c, ""))
    return t


# --- contact report ---------------------------------------------------------------------

def account_lines(acct: dict | None) -> list[str]:
    if not acct:
        return ["Not looked up (not needed, or stopped by a safety check)."]
    plan = acct.get("plan") or {}
    cap = "unlimited data" if plan.get("data_cap_gb") is None else f"{plan.get('data_cap_gb')} GB/month"
    return [
        f"Customer   {acct.get('customer_ref')}   (account {acct.get('account_ref')})",
        f"Plan       {plan.get('name')} · {plan.get('plan_type')} · {plan.get('tier')} tier · "
        f"{money(plan.get('monthly_price'))}/month · {cap}",
        f"Usage      last cycle {acct.get('data_used_gb_last_cycle')} GB · 3-month average "
        f"{acct.get('avg_data_used_gb_3m')} GB",
        f"Profile    {acct.get('tenure_months')} months tenure · churn risk {acct.get('churn_risk_label')} · "
        f"{acct.get('complaints_90d')} complaint(s) in 90 days",
    ]


def policy_block(state: dict) -> Group | Text:
    decision = state.get("offer_decision") or {}
    checks = decision.get("checks") or []
    if not checks:
        cites = [c["citation"] for c in state.get("policy_citations") or []]
        if cites:
            return Text(f"Relevant policy found: {', '.join(cites[:4])}. No offer or credit was needed.")
        return Text("No policy lookup was needed.")
    plan_name = ((state.get("account_summary") or {}).get("plan") or {}).get("name")
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False, expand=True)
    t.add_column("Option checked", ratio=3, overflow="fold")
    t.add_column("Decision", no_wrap=True)
    t.add_column("Deciding reason", ratio=5, overflow="fold")
    t.add_column("Clause", style="cyan", no_wrap=True)
    for c in checks:
        label = describe_offer(c) + (" (customer's request)" if c.get("source") == "customer_request" else "")
        dec = c.get("decision")
        dec_text = {"allowed": Text(f"{OK} allowed", style="green"),
                    "needs_approval": Text(f"{WAIT} needs approval", style="yellow"),
                    "blocked": Text(f"{BAD} blocked", style="red")}.get(dec, Text(str(dec)))
        reasons, refs = c.get("reasons") or [""], c.get("policy_refs") or [""]
        why = reasons[-1]  # the check appends the deciding reason / clause last
        if c.get("offer_value_usd") and dec != "blocked":
            why = f"Worth {money(c['offer_value_usd'])}. " + why
        t.add_row(label, dec_text, safe(why), refs[-1])
    note = Text(f"Offers worth more than {money(APPROVAL_THRESHOLD)} need a team lead's approval "
                f"(POL-RET-003 §1.2); blocked options are never offered.", style="dim")
    return Group(t, note)


def approval_line(state: dict) -> Text:
    approval = state.get("approval") or {}
    proposal = state.get("offer_proposal") or {}
    if not proposal or not proposal.get("needs_approval"):
        return Text(f"Not required: nothing above the {money(APPROVAL_THRESHOLD)} auto-approval limit.", style="dim")
    if not approval:
        return Text(f"{WAIT} Required: waiting for a team lead.", style="yellow")
    who = approval.get("approver", "")
    kind = "automated demo approver" if str(who).endswith(("auto-approve", "auto-reject")) else "team lead"
    if approval.get("approved"):
        return Text(f"{OK} Required, and APPROVED by the {kind}.", style="green")
    return Text(f"{BAD} Required, and REJECTED by the {kind}. The offer was withdrawn.", style="red")


def contact_report(title: str, request: str, state: dict, *, expected: str | None = None,
                   matched: bool | None = None, details: bool = False, run_id: str | None = None) -> None:
    res = state.get("resolution") or {}
    icon, color, label = OUTCOME_LABELS.get(res.get("outcome"), (INFO, "white", str(res.get("outcome"))))
    header = Text(title, style="bold")
    if matched is not None:
        header.append(f"   {OK} as expected" if matched else f"   {BAD} expected {expected}",
                      style="green" if matched else "red")
    console.print()
    console.print(Rule(header, style=color, align="left"))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True, width=18)
    grid.add_column()
    grid.add_row("Customer request", Text(f"“{safe(request)}”", style="italic"))
    intent = state.get("intent")
    conf = state.get("intent_confidence")
    grid.add_row("Detected intent", f"{INTENT_LABELS.get(intent, intent)}"
                 + (f"  (confidence {conf:.0%})" if intent and conf else ""))
    flags = [f for f in state.get("input_flags") or [] if not f.startswith("injection:")]
    if state.get("guard_blocked"):
        grid.add_row("Safety check", Text(f"{BAD} {GUARD_EXPLAIN.get(state.get('guard_reason'), 'Blocked.')}",
                                          style="red"))
    elif flags:
        grid.add_row("Safety check", Text(f"{WARN} Cleaned up: {', '.join(flags).replace('_', ' ')}",
                                          style="yellow"))
    else:
        grid.add_row("Safety check", Text(f"{OK} Passed", style="green"))
    grid.add_row("Account", "\n".join(account_lines(state.get("account_summary"))))
    grid.add_row("Policy decision", policy_block(state))
    grid.add_row("Resolution", Text(f"{icon} {label}", style=f"bold {color}"))
    grid.add_row("Human approval", approval_line(state))
    if res.get("ticket_id"):
        grid.add_row("Ticket", f"{res['ticket_id']} (a specialist will follow up)")
    console.print(grid)
    console.print(Panel(Text(state.get("final_response") or "(no reply)"), title="Reply to the customer",
                        title_align="left", border_style=color, padding=(0, 1)))
    cites = citations_table(res.get("citations") or [])
    if cites:
        console.print(cites)
    if details:
        engines = sorted({f"{e['node']}:{e['engine']}" for e in state.get("engine_log") or []})
        console.print(Text(f"details: run {run_id} · steps {state.get('step_count')} · "
                           f"risk tier {(state.get('output_risk') or {}).get('tier')} · engines {', '.join(engines)}",
                           style="dim"))


def approval_prompt_text(request: dict) -> Panel:
    """Plain-language explanation for a (possibly non-technical) team lead."""
    what = describe_offer(request)
    refs = ", ".join(request.get("policy_refs") or [])
    body = Text()
    body.append(f"The assistant wants to offer {request.get('customer_ref')} {what}.\n", style="bold")
    body.append(f"Total value: {money(request.get('offer_value_usd'))}. Offers above {money(APPROVAL_THRESHOLD)} "
                f"need a team lead's sign-off before they are applied.\n")
    body.append(f"The offer is within policy limits ({refs}).\n", style="dim")
    body.append("Approve = the offer is applied and the customer is told. "
                "Reject = the offer is withdrawn and the customer can still cancel.")
    return Panel(body, title=f"{WAIT} Team-lead approval needed", title_align="left", border_style="yellow")


def parse_yes_no(answer: str) -> bool | None:
    a = (answer or "").strip().lower()
    if a in ("y", "yes", "approve", "a"):
        return True
    if a in ("n", "no", "reject", "r"):
        return False
    return None


def summary_table(rows: list[dict]) -> Table:
    t = Table(title="Run summary", box=box.ROUNDED, header_style="bold")
    for col in ("Contact", "Scenario", "Intent", "Outcome", "Approval", "Check"):
        t.add_column(col)
    for r in rows:
        icon, color, label = OUTCOME_LABELS.get(r["outcome"], (INFO, "white", str(r["outcome"])))
        appr = ("approved" if r["approvals"][0].get("approved") else "rejected") if r["approvals"] else "—"
        t.add_row(r["contact_id"], r["scenario"].replace("_", " "), INTENT_LABELS.get(r["intent"], r["intent"]),
                  Text(label, style=color), appr, Text(OK, style="green") if r["match"] else Text(BAD, style="red"))
    return t
