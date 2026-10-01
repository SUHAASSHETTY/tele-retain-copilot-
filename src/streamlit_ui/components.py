"""Reusable UI pieces. Every value placed into HTML goes through esc()."""

from __future__ import annotations

import re
from html import escape

import streamlit as st

from src import ui
from src.streamlit_ui import backend
from src.streamlit_ui.backend import BackendError

RISK_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def esc(value) -> str:
    return escape("" if value is None else str(value))


def pill(text: str, kind: str = "") -> str:
    return f'<span class="rc-pill {esc(kind)}">{esc(text)}</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="rc-section">{esc(title)}</div>', unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="rc-hero"><div class="rc-eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)
    st.write("")


def flow(steps: list[tuple[str, str]]) -> None:
    parts = []
    for i, (title, detail) in enumerate(steps, 1):
        if i > 1:
            parts.append('<div class="arrow">→</div>')
        parts.append(f'<div class="step"><div class="n">{i}</div><b>{esc(title)}</b><span>{esc(detail)}</span></div>')
    st.markdown(f'<div class="rc-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def key_values(pairs: list[tuple[str, str]], stacked: bool = False) -> None:
    """Label/value grid; `stacked` puts each label above its value (for narrow cards)."""
    rows = "".join(f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in pairs)
    st.markdown(f'<div class="rc-kv{" stacked" if stacked else ""}">{rows}</div>', unsafe_allow_html=True)


def factors(items: list[dict], limit: int | None = None) -> None:
    rows = "".join(f'<div class="rc-factor"><span class="rc-dot {esc(f["severity"])}"></span>'
                   f'<div><b>{esc(f["label"])}</b><small>{esc(f["detail"])}</small></div></div>'
                   for f in items[:limit])
    st.markdown(rows or '<span class="rc-muted">No risk factors on record.</span>', unsafe_allow_html=True)


def citations(cites: list[dict]) -> None:
    rows = "".join(f'<div class="rc-cite"><code>{esc(c["id"])}</code> {esc(c.get("title", ""))}</div>' for c in cites)
    st.markdown(rows or '<span class="rc-muted">No policy clauses cited.</span>', unsafe_allow_html=True)


def show_error(err: Exception, context: str = "") -> None:
    """A clean message on screen; masked technical details in an expander. Never a traceback."""
    if not isinstance(err, BackendError):
        err = BackendError("Unable to complete the analysis. Please check the configuration and try again.",
                           f"{type(err).__name__}: {err}")
    st.error(f"{context} {err.message}".strip(), icon="⚠️")
    if err.detail:
        with st.expander("Technical details"):
            st.code(err.detail, language=None, wrap_lines=True)


def plain_reason(reason: str | None) -> str | None:
    """Turn the resolution agent's terse reason into plain English (e.g. tool and field names)."""
    if not reason:
        return reason
    reason = re.sub(r"\b(\w+) allowed by check_offer_eligibility",
                    lambda m: f"The {ui.OFFER_WORDS.get(m.group(1), m.group(1))} passed the policy eligibility check",
                    reason)
    return reason.replace("_", " ")


def money(v) -> str:
    return ui.money(v)


def sidebar_status() -> None:
    with st.sidebar.container(border=True):
        st.markdown('<div class="rc-section">System status</div>', unsafe_allow_html=True)
        try:
            h = backend.health()
        except BackendError as err:
            st.markdown("🔴 **API offline**")
            st.caption(err.message)
            return
        st.markdown("🟢 **API online**")
        if h.get("llm") == "gemini":
            st.markdown(f"🟢 **Gemini** · `{h.get('model')}`")
        else:
            st.markdown("🟠 **Rules & templates engine**")
            st.caption("Gemini is not available (no key, quota or model error), so the copilot uses its "
                       "deterministic fallback. Guardrails, policy checks and audit work the same.")
        st.caption("🔒 Synthetic data · PII masked")
