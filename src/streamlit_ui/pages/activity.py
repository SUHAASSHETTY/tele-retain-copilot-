"""Agent Activity: the real agent pipeline, the latest run in this session, and the audit trail."""

from __future__ import annotations

import streamlit as st

from src.config import ROOT_DIR
from src.streamlit_ui import data, state
from src.streamlit_ui.components import esc, page_header, plain_reason, section, show_error

PIPELINE = [  # graph nodes in src/graph.py, in the order a contact normally flows through them
    ("🛡️", "Input guard", "Blocks prompt injection, threats and requests for other customers' data; masks PII."),
    ("🧠", "Context & memory", "Loads the conversation and what the customer told us before (LangMem)."),
    ("🧭", "Supervisor", "Plans the next step and routes work to the specialist agents."),
    ("🔎", "Intent agent", "Classifies the request: billing, plan, complaint, cancellation…"),
    ("👤", "Account agent", "Fetches account and billing through authorized MCP tools."),
    ("📚", "Policy agent", "Retrieves the relevant policy clauses (agentic RAG)."),
    ("🎁", "Retention offer agent", "Checks every offer with the policy engine; above $50 needs a team lead."),
    ("✍️", "Resolution agent", "Decides the outcome and drafts the reply with citations."),
    ("✅", "Output guard", "Re-checks the reply for PII leaks and policy limits, then audits the decision."),
]
ACTIONS = [("contact_resolved", "Contacts resolved"), ("offer_proposed", "Offers proposed"),
           ("offer_blocked", "Offers blocked by policy"), ("guardrail_block", "Guardrail blocks"),
           ("escalation", "Escalations")]


def render() -> None:
    page_header("Agent Activity", "What the agents actually did",
                "Nine cooperating agents handle every contact. Each decision is written to an audit trail.")

    section("The agent pipeline")
    cols = st.columns(3, gap="small")
    for i, (icon, name, what) in enumerate(PIPELINE):
        with cols[i % 3].container(border=True, height="stretch"):
            st.markdown(f"**{icon} {esc(name)}**")
            st.caption(what)
    diagram = ROOT_DIR / "docs" / "assets" / "architecture.png"
    if diagram.exists():
        with st.expander("View architecture diagram"):
            st.image(str(diagram), width="stretch")

    st.write("")
    section("Latest analysis in this session")
    history = st.session_state.get("history") or []
    if not history:
        st.info("No analysis yet in this session. Run one in Customer Copilot to see its live steps here.", icon="💡")
        if st.button("Open Customer Copilot"):
            state.go("copilot")
    else:
        last = history[0]
        outcome = (last.get("resolution") or {}).get("outcome_label") or "Waiting for approval"
        with st.container(border=True):
            st.markdown(f"**{esc(outcome)}** · session `{esc(last.get('session_id'))}`", unsafe_allow_html=True)
            for s in last["steps"]:
                if s["node"] != "supervisor":
                    st.markdown(f"✅ **{s['label']}**" + (f" — {s['detail']}" if s["detail"] else ""))
            with st.expander("View details"):
                st.dataframe(last["steps"], hide_index=True, width="stretch")

    st.write("")
    section("Audit trail")
    try:
        counts = data.decision_counts()
        cases = data.cases()
    except Exception as err:  # noqa: BLE001
        show_error(err, "The audit trail is unavailable.")
        return
    for col, (key, label) in zip(st.columns(len(ACTIONS)), ACTIONS):
        col.metric(label, counts.get(key, 0), border=True)
    if not cases:
        st.info("The audit trail is empty. Run an analysis to create the first entry.")
        return
    st.caption("Select a case to see every decision the agents made for it.")
    picked = st.dataframe([{**{k: c[k] for k in ("time", "session", "outcome", "policy")},
                            "reason": plain_reason(c["reason"])} for c in cases],
                          column_order=("time", "session", "outcome", "reason", "policy"),
                          hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          height=320)
    if picked.selection.rows:
        case = cases[picked.selection.rows[0]]
        with st.expander(f"Decisions for session {case['session']}", expanded=True):
            trail = data.run_trail(case["run_id"])
            st.dataframe([{"actor": r["actor"], "type": r["actor_type"], "action": r["action"],
                           "decision": r["decision"], "reason": r["reason"]} for r in trail],
                         hide_index=True, width="stretch")
