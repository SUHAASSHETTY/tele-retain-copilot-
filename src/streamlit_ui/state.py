"""Session state, cached read-only data and page navigation shared by the pages."""

from __future__ import annotations

import streamlit as st

from src.streamlit_ui import data

PAGES: dict[str, st.Page] = {}  # filled by app.py


def go(name: str) -> None:
    st.switch_page(PAGES[name])


@st.cache_data(ttl=300, show_spinner=False)
def customers() -> list[dict]:
    return data.customers()


@st.cache_data(ttl=300, show_spinner=False)
def customer(customer_id: str) -> dict | None:
    return data.customer(customer_id)


@st.cache_data(ttl=300, show_spinner=False)
def opportunities() -> list[dict]:
    return data.opportunities(customers())


@st.cache_data(ttl=300, show_spinner=False)
def complaint_mix() -> dict:
    return data.complaint_mix()


@st.cache_data(ttl=60, show_spinner=False)
def evaluation() -> dict:
    return data.evaluation()


def select_customer(customer_id: str, message: str | None = None) -> None:
    """Remember the customer (and optionally a message) for the Customer Copilot page."""
    st.session_state["_customer"] = customer_id
    if message is not None:
        st.session_state["_message"] = message
