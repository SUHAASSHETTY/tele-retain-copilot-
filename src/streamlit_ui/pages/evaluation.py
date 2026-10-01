"""Evaluation / Results: only metrics that exist in reports/ (nothing is estimated here)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.streamlit_ui import state
from src.streamlit_ui.components import page_header, section


def _pct(v) -> str:
    return "N/A" if v is None else f"{v:.0%}"


def render() -> None:
    page_header("Evaluation", "How well does it work?",
                "Results from the project's evaluation, red-team and tracing runs (reports/).")
    ev = state.evaluation()
    report, rt, sig = ev.get("eval") or {}, ev.get("redteam") or {}, ev.get("signals") or {}
    if not (report or rt or sig):
        st.info("No evaluation results yet. Run `python -m scripts.regenerate_evidence` to produce them.", icon="📊")
        return
    agg = report.get("aggregate") or {}
    summ = rt.get("summary") or {}
    errs = sig.get("errors") or {}
    traffic = sig.get("traffic") or {}
    tool_lat = (sig.get("latency") or {}).get("tool") or {}

    section(f"Answer quality · {agg.get('cases', 'N/A')} golden cases")
    k = st.columns(4)
    k[0].metric("Intent accuracy", _pct(agg.get("intent_accuracy")), border=True)
    k[1].metric("Action accuracy", _pct(agg.get("action_accuracy")), border=True,
                help="Resolve / offer / escalate / clarify / refuse matched the expected action.")
    k[2].metric("Policy citation recall", _pct(agg.get("policy_ref_recall")), border=True,
                help="Expected policy clauses that the reply cited.")
    k[3].metric("Citation validity", _pct(agg.get("citation_validity")), border=True,
                help="Cited clauses that exist in the policy corpus.")

    section("Safety & reliability")
    k = st.columns(4)
    k[0].metric("Red-team attacks contained", f"{summ.get('passed', 'N/A')}/{summ.get('attacks', 'N/A')}", border=True)
    k[1].metric("Attack success rate", _pct(summ.get("attack_success_rate")), border=True)
    k[2].metric("Tool success rate", _pct(1 - errs["tool_calls_not_ok"] / traffic["tool_calls"])
                if traffic.get("tool_calls") and "tool_calls_not_ok" in errs else "N/A", border=True,
                help="MCP tool calls that returned ok, from the traced runs.")
    k[3].metric("Tool latency (p95)", f"{tool_lat['p95_ms']:.0f} ms" if tool_lat.get("p95_ms") else "N/A", border=True)

    hal = agg.get("hallucination") or {}
    if hal.get("status") != "ok":
        st.info(f"Hallucination judging (Gemini via DeepEval) covered {hal.get('scored_cases', 0)} of "
                f"{agg.get('cases', '?')} cases because the judge model was unavailable; no scores were invented "
                f"for the rest. Agent engine during evaluation: {report.get('agent_engine', 'N/A')}.", icon="ℹ️")

    by_tool = (sig.get("latency") or {}).get("tool_by_name") or {}
    if by_tool:
        with st.container(border=True):
            section("Tool latency by tool (ms)")
            df = pd.DataFrame([{"Tool": name, "p50": v.get("p50_ms"), "p95": v.get("p95_ms")}
                               for name, v in by_tool.items()])
            st.bar_chart(df, x="Tool", y=["p50", "p95"], stack=False, horizontal=True, height=300, color=["#91a7ff", "#3b5bdb"])

    with st.expander("View detailed evaluation"):
        cases = report.get("cases") or []
        if cases:
            st.markdown("**Golden set cases**")
            st.dataframe([{"case": c["case_id"], "category": c["category"], "expected": c["expected_action"],
                           "predicted": c["predicted_action"], "intent ok": c["intent_correct"],
                           "action ok": c["action_correct"], "citation recall": c["policy_ref_recall"],
                           "latency ms": c["latency_ms"]} for c in cases], hide_index=True, width="stretch")
        attacks = rt.get("results") or []
        if attacks:
            st.markdown("**Red-team attacks**")
            st.dataframe([{"attack": a["attack_id"], "category": a["category"], "technique": a["technique"],
                           "outcome": a["outcome"], "detected": a["detected"], "passed": a["pass"]} for a in attacks],
                         hide_index=True, width="stretch")
        steps = (ev.get("regenerate") or {}).get("steps") or []
        if steps:
            st.markdown("**Evidence regeneration** (`python -m scripts.regenerate_evidence`)")
            st.dataframe([{"step": s["step"], "name": s["name"], "status": s["status"], "seconds": s["seconds"]}
                          for s in steps], hide_index=True, width="stretch")
        st.caption(f"Evaluation generated {report.get('generated_at', 'N/A')} · red team {rt.get('generated_at', 'N/A')}"
                   f" · signals {sig.get('generated_at', 'N/A')}")
