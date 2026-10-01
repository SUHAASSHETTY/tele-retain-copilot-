"""Retention Analysis: portfolio risk and the best policy-eligible offer for every at-risk customer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit_ui import state
from src.streamlit_ui.components import page_header, section, show_error

DECISION_LABEL = {"allowed": "Auto-approved", "needs_approval": "Needs approval"}


def render() -> None:
    page_header("Retention Analysis", "Who is at risk, and what can we offer?",
                "Every at-risk customer is checked against the same policy engine the agents use. "
                "Nothing is offered until an agent runs the copilot.")
    try:
        people = state.customers()
        opps = state.opportunities()
        mix = state.complaint_mix()
    except Exception as err:  # noqa: BLE001
        show_error(err, "Retention data is unavailable.")
        return

    eligible = [o for o in opps if o["best"]]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("At-risk customers", len(opps), border=True, help="Medium or high churn risk.")
    k2.metric("High risk", sum(1 for o in opps if o["risk"] == "high"), border=True)
    k3.metric("Eligible for an offer", len(eligible), border=True,
              help="At least one retention offer passes the policy engine today.")
    k4.metric("Need team-lead approval", sum(1 for o in eligible if o["best"]["decision"] == "needs_approval"),
              border=True, help="Offer value above the auto-approval threshold (POL-RET-003).")

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        section("Churn risk across all customers")
        risk = pd.DataFrame({"Risk": ["High", "Medium", "Low"],
                             "Customers": [sum(1 for p in people if p["risk"] == r) for r in ("high", "medium", "low")]})
        st.bar_chart(risk, x="Risk", y="Customers", horizontal=True, height=200, color="#3b5bdb", sort=False)
    with b.container(border=True):
        section(f"Complaints by category ({mix['open']} of {mix['total']} open)")
        cmp = pd.DataFrame({"Category": [k.replace("_", " ") for k in mix["by_category"]],
                            "Complaints": list(mix["by_category"].values())})
        st.bar_chart(cmp, x="Category", y="Complaints", horizontal=True, height=200, color="#3b5bdb", sort="-Complaints")

    section("Retention opportunities")
    st.caption("Select a row, then open that customer in the copilot.")
    rows = pd.DataFrame([{
        "Customer": f"{o['customer_ref']} · {o['initial']}.", "Risk": o["risk"].capitalize(),
        "Score": o["risk_score"], "Plan": o["plan"], "Monthly bill": o["monthly_bill"],
        "Best eligible offer": o["best"]["offer"] if o["best"] else "None (blocked by policy)",
        "Decision": DECISION_LABEL.get((o["best"] or {}).get("decision"), "—"),
        "Why": "; ".join((o["best"] or o["checks"][-1])["reasons"]) if (o["best"] or o["checks"]) else "",
    } for o in opps])
    picked = st.dataframe(
        rows, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "Score": st.column_config.ProgressColumn("Churn score", min_value=0, max_value=1, format="%.2f"),
            "Monthly bill": st.column_config.NumberColumn(format="$%.2f"),
            "Why": st.column_config.TextColumn("Policy reason", width="large"),
            "Best eligible offer": st.column_config.TextColumn(width="medium"),
        })
    sel = picked.selection.rows
    if st.button("Analyze selected customer in Copilot →", type="primary", disabled=not sel):
        state.select_customer(opps[sel[0]]["customer_id"])
        state.go("copilot")
