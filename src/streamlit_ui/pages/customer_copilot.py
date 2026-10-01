"""Customer Copilot: select a customer -> review risk -> run the real agent workflow -> action + reply."""

from __future__ import annotations

import streamlit as st

from src import ui
from src.policy_limits import APPROVAL_THRESHOLD
from src.streamlit_ui import backend, data, state
from src.streamlit_ui.backend import BackendError
from src.streamlit_ui.components import (RISK_ICON, plain_reason, citations, esc, factors, key_values, money, page_header, pill,
                                         section, show_error)

MAX_CHARS = 6000
EXAMPLES = {
    "Cancellation": "I'm thinking of cancelling. Another provider offered me a cheaper plan.",
    "Billing": "Why is my bill higher this month? I think I was charged twice.",
    "Complaint": "This is the third time I'm complaining about dropped calls. Nobody has fixed it.",
    "Plan fit": "I keep running out of data before the month ends. What can I do?",
}
ACTION_TEXT = {
    "resolve": "Resolve now: answer from account data and policy",
    "escalate": "Escalate to a human specialist",
    "clarify": "Ask the customer to clarify the request",
    "decline": "Politely decline: outside telecom support",
    "refuse": "Refuse: a safety or privacy check blocked the request",
}
NEXT_STEP = {
    "resolve": "The customer's question is answered in this contact; no retention cost.",
    "offer": "Confirm the customer accepts, apply the offer in billing and schedule a check-in before it ends.",
    "offer_pending_approval": "A team lead approves or rejects the offer before it is quoted.",
    "escalate": "A specialist owns the case under the complaint SLA and keeps the customer informed.",
    "clarify": "Wait for the customer's answer, then run the analysis again.",
    "decline": "No follow-up: the request is outside telecom support.",
    "refuse": "No action: the attempt is recorded in the audit trail for review.",
}


# --- running the real workflow ----------------------------------------------------------

def _step_detail(ev: dict) -> str:
    node = ev.get("node")
    if node == "input_guard":
        return ("Blocked: " + ", ".join(ev.get("input_flags") or [])) if ev.get("guard_blocked") else \
            ("Flags: " + ", ".join(ev["input_flags"]) if ev.get("input_flags") else "No safety or privacy issues")
    if node == "intent_agent" and ev.get("intent"):
        conf = ev.get("intent_confidence")
        return ui.INTENT_LABELS.get(ev["intent"], ev["intent"]) + (f" ({conf:.0%} confidence)" if conf else "")
    if node == "account_agent":
        return "Lookup failed" if ev.get("account_failed") else "Account and billing retrieved via MCP"
    if node == "retention_offer_agent":
        return "Offer needs team-lead approval" if ev.get("needs_human_approval") else "Offers checked against policy"
    if node == "resolution_agent" and ev.get("resolution"):
        return (ev["resolution"].get("reason") or "")[:140]
    if node == "human_approval" and ev.get("approval"):
        return "Approved" if ev["approval"].get("approved") else "Rejected"
    if node == "output_guard":
        tier = (ev.get("output_risk") or {}).get("tier")
        return f"Output risk tier: {tier}" + (f" · flags: {', '.join(ev['output_flags'])}" if ev.get("output_flags") else "")
    if node == "supervisor" and ev.get("next_worker"):
        return f"Routed to {ev['next_worker']}"
    return ""


def _consume(events, result: dict, status) -> None:
    for event, payload in events:
        if event == "start":
            result.update(session_id=payload["session_id"], run_id=payload["run_id"])
        elif event == "node":
            step = {"node": payload.get("node"), "label": payload.get("label"), "detail": _step_detail(payload)}
            result["steps"].append(step)
            if step["node"] != "supervisor":
                status.write(f"✓ **{step['label']}**" + (f" — {step['detail']}" if step["detail"] else ""))
        elif event == "approval_required":
            result["approval"] = payload
            status.update(label="Waiting for a team lead's approval", state="complete", expanded=False)
        elif event == "resolution":
            result["resolution"] = payload
            result.pop("approval", None)
            status.update(label="Analysis complete", state="complete", expanded=False)


def run_analysis(customer_id: str, message: str) -> None:
    result = {"customer_id": customer_id, "message": message, "steps": []}
    with st.status("Analyzing customer…", expanded=True) as status:
        try:
            _consume(backend.analyze(customer_id, message), result, status)
        except BackendError as err:
            status.update(label="Analysis failed", state="error", expanded=False)
            st.session_state["error"] = err
            return
    st.session_state.pop("error", None)
    st.session_state["result"] = result
    st.session_state.setdefault("history", []).insert(0, result)


def run_decision(approved: bool, approver: str) -> None:
    result = st.session_state["result"]
    with st.status("Recording the team lead's decision…", expanded=True) as status:
        try:
            _consume(backend.decide(result["session_id"], approved, approver), result, status)
        except BackendError as err:
            status.update(label="Could not record the decision", state="error", expanded=False)
            st.session_state["error"] = err
            return
    st.session_state.pop("error", None)


# --- reading the evidence of one run ----------------------------------------------------

def _evidence(result: dict) -> dict:
    """What the audit trail recorded for this run. Amounts are masked in the audit log by design, so
    offers are re-checked with the same deterministic policy engine to show their real cost and reasons."""
    trail = data.run_trail(result.get("run_id", ""))
    cid = result["customer_id"]
    cancel = (result.get("resolution") or {}).get("intent") == ui.INTENT_LABELS["cancellation"]
    proposed = next((r for r in trail if r["action"] == "offer_proposed"), None)
    check = data.recheck(cid, proposed["details"], cancel) if proposed else None
    blocked = []
    for r in trail:
        if r["action"] == "offer_blocked":
            again = data.recheck(cid, r["details"], cancel)
            blocked.append({"offer": ui.describe_offer(r["details"]),
                            "reason": "; ".join(again["reasons"]) if again else r["reason"]})
    return {
        "trail": trail,
        "resolved": next((r for r in reversed(trail) if r["action"] == "contact_resolved"), None),
        "offer_text": ui.describe_offer(proposed["details"]) if proposed else None,
        "cost": check["cost"] if check else None,
        "why": check["reasons"] if check else [],
        "blocked": blocked,
    }


def _recommended_action(outcome: str, ev: dict) -> str:
    if outcome in ("offer", "offer_pending_approval") and ev["offer_text"]:
        text = f"Offer {ev['offer_text']}"
        return text + (" (pending team-lead approval)" if outcome == "offer_pending_approval" else "")
    return ACTION_TEXT.get(outcome, ui.OUTCOME_LABELS.get(outcome, ("", "", outcome or "No action"))[2])


def _priority(c: dict, outcome: str, risk_tier: str | None) -> str:
    if c["risk"] == "high" or outcome in ("escalate", "refuse") or risk_tier == "high":
        return "High"
    if c["risk"] == "medium" or outcome in ("offer", "offer_pending_approval"):
        return "Medium"
    return "Low"


def _impact(c: dict, outcome: str, ev: dict, res: dict) -> str:
    if outcome in ("offer", "offer_pending_approval") and ev["cost"] is not None:
        limit = "within" if ev["cost"] <= APPROVAL_THRESHOLD else "above"
        return (f"If the customer stays, {money(c['monthly_bill'])}/month ({money(c['monthly_bill'] * 12)}/year) of "
                f"billing is retained for a total offer cost of {money(ev['cost'])}, {limit} the "
                f"{money(APPROVAL_THRESHOLD)} auto-approval limit.")
    if outcome == "escalate" and res.get("ticket_id"):
        return f"Ticket {res['ticket_id']} gives a specialist ownership of the case. " + NEXT_STEP["escalate"]
    return NEXT_STEP.get(outcome, "")


# --- page ------------------------------------------------------------------------------

def _on_customer() -> None:
    state.select_customer(st.session_state["cp_customer"])
    st.session_state.pop("error", None)


def _on_message() -> None:
    st.session_state["_message"] = st.session_state["cp_message"]


def _on_scenario() -> None:
    label = st.session_state.get("cp_scenario")
    sample = next((s for s in _samples() if s["label"] == label), None)
    if sample:
        state.select_customer(sample["customer_id"], sample["message"])


def _on_example() -> None:
    choice = st.session_state.get("cp_example")
    if choice:
        st.session_state["_message"] = EXAMPLES[choice]


def _samples() -> list[dict]:
    try:
        return backend.samples()
    except BackendError:
        return []


def render() -> None:
    page_header("Customer Copilot", "Analyze a customer and get the next best action",
                "Select a customer, review their risk, paste what they said, and run the full agent workflow.")
    try:
        people = state.customers()
    except Exception as err:  # noqa: BLE001
        show_error(err, "Customer data is unavailable.")
        return
    by_id = {p["customer_id"]: p for p in people}
    st.session_state.setdefault("_customer", people[0]["customer_id"])
    st.session_state["cp_customer"] = st.session_state["_customer"]
    st.session_state["cp_message"] = st.session_state.get("_message", "")

    # 1 · customer
    section("1 · Select customer")
    left, right = st.columns([3, 2], gap="medium")
    left.selectbox(
        "Customer", options=list(by_id), key="cp_customer", on_change=_on_customer,
        format_func=lambda cid: f"{RISK_ICON[by_id[cid]['risk']]}  {by_id[cid]['customer_ref']} · "
                                f"{by_id[cid]['initial']}. · {by_id[cid]['plan']} · {by_id[cid]['risk']} risk",
        help="Type to search. Sorted by churn risk, highest first. Names are masked (PII policy).")
    samples = _samples()
    right.selectbox("Or load a demo scenario", options=[s["label"] for s in samples], index=None, key="cp_scenario", on_change=_on_scenario,
                    placeholder="Choose a scenario…", format_func=str.capitalize,
                    help="Sample contacts from data/sample_contacts.jsonl: sets the customer and the message.")

    c = state.customer(st.session_state["_customer"])
    if c is None:
        st.warning("This customer could not be found. Please pick another one.")
        return

    col_a, col_b = st.columns(2, gap="medium")
    with col_a.container(border=True, height="stretch"):
        section("Customer summary")
        status = pill("Needs attention", "high") if c["needs_attention"] else pill("Stable", "low")
        st.markdown(f"#### {esc(c['initial'])}. •••••  {pill(c['customer_ref'], 'info')} {status}",
                    unsafe_allow_html=True)
        latest = c["complaints"][0] if c["complaints"] else None
        key_values([
            ("Customer ID", c["customer_ref"]),
            ("Plan", f"{c['plan']} · {c['tier']} · {c['plan_type']}"),
            ("Tenure", f"{c['tenure_months']} months · {c['region']}"),
            ("Monthly bill", money(c["monthly_bill"])),
            ("Contract", f"ends {c['contract_end_date']}" if c["contract_end_date"] else "no contract"),
            ("Interactions", f"{c['complaints_90d']} complaint(s) in 90 days, {c['open_complaints']} open"),
            ("Latest", f"{latest['category'].replace('_', ' ')} complaint, {latest['status']} "
                       f"({latest['opened_date']})" if latest else "no complaints on record"),
        ])
    with col_b.container(border=True, height="stretch"):
        section("Customer risk")
        score = f"score {c['risk_score']:.2f}"
        st.markdown(f"#### {RISK_ICON[c['risk']]} {c['risk'].capitalize()} risk  {pill(score, c['risk'])}",
                    unsafe_allow_html=True)
        st.progress(min(max(c["risk_score"], 0.0), 1.0), text="Churn risk score (0 = safe, 1 = likely to leave)")
        st.markdown("**Key risk factors**")
        factors(c["factors"], limit=4)
        st.caption("Factors are transparent rules over the account record.")

    # 2 · message
    st.write("")
    section("2 · What did the customer say?")
    st.pills("Examples", list(EXAMPLES), key="cp_example", on_change=_on_example, label_visibility="collapsed")
    st.text_area("Customer message", key="cp_message", on_change=_on_message, height=110, max_chars=MAX_CHARS,
                 placeholder="Paste or type the customer's words, or pick an example above.",
                 label_visibility="collapsed")
    row = st.container(horizontal=True, gap="small")
    run = row.button("▶  Run full analysis", type="primary")
    result = st.session_state.get("result")
    same = bool(result and result["customer_id"] == c["customer_id"])
    again = row.button("↻  Analyze again", disabled=not same,
                      help="Run the last message again as a new case")
    if run or again:
        message = (st.session_state.get("_message") or "").strip() if run else result["message"]
        if not message:
            st.warning("Please enter the customer's message first.", icon="✍️")
        else:
            run_analysis(c["customer_id"], message)
            st.rerun()

    if st.session_state.get("error"):
        show_error(st.session_state["error"])
    result = st.session_state.get("result")
    if not result or result["customer_id"] != c["customer_id"]:
        st.info("Run the analysis to see the risk explanation, the recommended action and a ready-to-send reply.",
                icon="💡")
        return
    _results(c, result)


def _results(c: dict, result: dict) -> None:
    st.divider()
    ev = _evidence(result)
    if result.get("approval"):
        _approval_card(result["approval"])
        return
    res = result.get("resolution") or {}
    outcome = res.get("outcome") or ""
    reason = plain_reason((ev["resolved"] or {}).get("reason")) or res.get("outcome_label")

    section("3 · AI analysis & recommendation")
    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True, height="stretch"):
        st.markdown("##### Why is this customer at risk?")
        bullets = [f"**{f['label']}** — {f['detail']}" for f in c["factors"][:3]]
        if res.get("intent"):
            bullets.append(f"**This contact:** {res['intent']} — {reason}")
        st.markdown("\n".join(f"- {b}" for b in bullets).replace("$", "\\$"))
        st.markdown("##### What should the agent do?")
        st.markdown(_recommended_action(outcome, ev).replace("$", "\\$"))
        st.markdown("##### Why this action?")
        why = list(ev["why"])
        why += [f"Not offered: {b['offer']} — {b['reason']}" for b in ev["blocked"]]
        why += [n for n in res.get("notes") or [] if not (ev["blocked"] and n.startswith("Requested "))]
        st.markdown("\n".join(f"- {w}" for w in why if w).replace("$", "\\$") or "Decided from the policy "
                    "clauses cited below.")
        st.markdown("##### Expected impact")
        st.markdown(_impact(c, outcome, ev, res).replace("$", "\\$"))
    with right.container(border=True, height="stretch"):
        section("Recommended action")
        st.markdown(f'<div class="rc-action">{esc(_recommended_action(outcome, ev))}</div>', unsafe_allow_html=True)
        prio = _priority(c, outcome, res.get("risk_tier"))
        st.markdown(f"{pill(res.get('outcome_label') or outcome, 'info')} "
                    f"{pill('Priority: ' + prio, prio.lower())}", unsafe_allow_html=True)
        st.write("")
        key_values([("Reason", reason or "—"), ("Expected outcome", NEXT_STEP.get(outcome, "—")),
                    ("Ticket", res.get("ticket_id") or "none")], stacked=True)
        st.caption("Priority is High for high churn risk, escalations or high output risk; Medium for medium "
                   "risk or offers; otherwise Low.")

    st.write("")
    section("4 · Customer-service response")
    with st.container(border=True):
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            st.markdown((res.get("reply") or "_No reply was produced._").replace("$", "\\$"))
        st.markdown(" ".join([pill(res.get("intent") or "—", "info"),
                              pill(f"output risk: {res.get('risk_tier') or 'n/a'}", ""),
                              pill("passed output guard", "low")]), unsafe_allow_html=True)
        with st.expander("Policy clauses cited"):
            citations(res.get("citations") or [])
        with st.expander("Copy response"):
            st.caption("Use the copy icon in the top-right corner of the box.")
            st.code(res.get("reply") or "", language=None, wrap_lines=True)
    if st.button("↻  Regenerate response", help="Run the same message through the full workflow again"):
        run_analysis(c["customer_id"], result["message"])
        st.rerun()

    _activity(result, ev)


def _approval_card(req: dict) -> None:
    section("3 · Team-lead approval required")
    with st.container(border=True):
        request = req.get("request") or {}
        st.warning(req.get("explanation") or "This offer needs a team lead's approval.", icon="⏸️")
        key_values([("Offer", ui.describe_offer(request)), ("Total value", money(request.get("offer_value_usd"))),
                    ("Why", "; ".join(request.get("reasons") or []) or "Above the auto-approval threshold"),
                    ("Policy", ", ".join(request.get("policy_refs") or []) or "POL-RET-003")])
        approver = st.text_input("Approver", value="team-lead", max_chars=64,
                                 help="Recorded in the audit trail as the human who decided.")
        row = st.container(horizontal=True, gap="small")
        if row.button("✓  Approve offer", type="primary", disabled=len(approver.strip()) < 2):
            run_decision(True, approver.strip())
            st.rerun()
        if row.button("✕  Reject offer", disabled=len(approver.strip()) < 2):
            run_decision(False, approver.strip())
            st.rerun()


def _activity(result: dict, ev: dict) -> None:
    st.write("")
    section("Agent activity")
    steps = [s for s in result["steps"] if s["node"] != "supervisor"]
    cols = st.columns(2)
    for i, s in enumerate(steps):
        cols[i % 2].markdown(f"✅ **{s['label']}**" + (f"  \n<span class='rc-muted'>{esc(s['detail'])}</span>"
                                                       if s["detail"] else ""), unsafe_allow_html=True)
    with st.expander("View details — audit trail and tool calls for this run"):
        st.caption(f"Session `{result.get('session_id')}` · run `{result.get('run_id')}` · "
                   f"{sum(1 for s in result['steps'] if s['node'] == 'supervisor')} supervisor routing decisions")
        if ev["trail"]:
            st.dataframe([{"actor": r["actor"], "type": r["actor_type"], "action": r["action"],
                           "decision": r["decision"], "reason": r["reason"],
                           "policy": ", ".join(r["policy_ref"]) if isinstance(r["policy_ref"], list) else r["policy_ref"]}
                          for r in ev["trail"]], hide_index=True, width="stretch")
        tools = data.run_tools(result.get("run_id", ""))
        if tools:
            st.dataframe(tools, hide_index=True, width="stretch")
