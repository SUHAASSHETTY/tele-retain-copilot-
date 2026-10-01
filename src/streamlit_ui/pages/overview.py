"""Overview: what the product is, who it is for, the workflow, and real headline numbers."""

from __future__ import annotations

import streamlit as st

from src.streamlit_ui import data, state
from src.streamlit_ui.components import flow, page_header, section, show_error


def _pct(v) -> str:
    return "N/A" if v is None else f"{v:.0%}"


def render() -> None:
    page_header("AI copilot for telecom customer care", "Customer Service & Retention Copilot",
                "Helps support agents understand a customer, spot churn risk, choose a policy-compliant "
                "retention action and draft the reply, with every step checked and audited.")

    a, b, c = st.columns(3, gap="medium")
    with a.container(border=True, height="stretch"):
        st.markdown("**🎯 What it is**")
        st.caption("A multi-agent assistant (LangGraph) that reads the account, company policy and past contacts.")
    with b.container(border=True, height="stretch"):
        st.markdown("**👩‍💼 Who it is for**")
        st.caption("Frontline support and retention agents, with team leads approving higher-value offers.")
    with c.container(border=True, height="stretch"):
        st.markdown("**⚡ What it does**")
        st.caption("Detects risk, explains why, recommends the next best action and writes a cited reply.")

    st.write("")
    section("Headline numbers")
    try:
        people = state.customers()
        opps = state.opportunities()
        resolved = data.decision_counts().get("contact_resolved")
    except Exception as err:  # noqa: BLE001
        show_error(err, "Portfolio data is unavailable.")
        people, opps, resolved = [], [], None
    at_risk = [p for p in people if p["risk"] != "low"]
    high = sum(1 for p in people if p["risk"] == "high")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers profiled", len(people) or "N/A", border=True,
              help="Synthetic telecom customers with a churn score and account history (data/synthetic/telecom.db).")
    k2.metric("At-risk customers", len(at_risk) if people else "N/A", border=True,
              delta=f"{high} high risk" if people else None, delta_color="inverse", delta_arrow="off",
              help="Customers with medium or high churn risk.")
    k3.metric("Retention opportunities", sum(1 for o in opps if o["best"]) if people else "N/A", border=True,
              help="At-risk customers with at least one offer the policy engine allows today.")
    k4.metric("Recommendations generated", resolved if resolved is not None else "N/A", border=True,
              help="Contacts the copilot has resolved, counted from the audit trail (logs/agent_actions.jsonl).")

    st.write("")
    section("How it works")
    flow([
        ("Customer", "Verified caller; account and billing fetched through secure MCP tools"),
        ("Analysis", "Safety guard, intent agent, memory of past contacts"),
        ("Risk detection", "Churn score, complaints, contract and competitor signals"),
        ("Recommended action", "Policy search + offer-eligibility engine; team lead approves above $50"),
        ("Retention outcome", "Guarded, cited reply to the customer; every decision audited"),
    ])

    st.write("")
    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        st.markdown("**Try it in five steps**")
        st.markdown("1. Select a customer  \n2. Review their risk  \n3. Paste what they said  \n"
                    "4. Run the full analysis  \n5. Review the action and the reply")
        if st.button("▶  Open Customer Copilot", type="primary"):
            state.go("copilot")
    with right.container(border=True):
        st.markdown("**Measured, not claimed**")
        ev = state.evaluation()
        agg = (ev.get("eval") or {}).get("aggregate") or {}
        rt = (ev.get("redteam") or {}).get("summary") or {}
        st.markdown(f"- Intent accuracy **{_pct(agg.get('intent_accuracy'))}** and action accuracy "
                    f"**{_pct(agg.get('action_accuracy'))}** on {agg.get('cases', 'N/A')} golden cases\n"
                    f"- **{rt.get('passed', 'N/A')}/{rt.get('attacks', 'N/A')}** red-team attacks contained\n"
                    f"- Policy citation validity **{_pct(agg.get('citation_validity'))}**")
        if st.button("See evaluation results"):
            state.go("evaluation")
