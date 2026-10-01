"""Streamlit UI for the Customer Service & Retention Copilot.

    streamlit run app.py

A front-end layer only: the agent workflow runs in the existing FastAPI app (src/api/app.py), which
this UI starts in-process (or reaches at COPILOT_API_URL). See src/streamlit_ui/.
COPILOT_NO_LLM=1 forces the deterministic rules/templates engine, like the CLI's --no-llm.
"""

import os
from pathlib import Path

if os.getenv("COPILOT_NO_LLM") == "1":  # must run before src.config reads .env (override=False)
    os.environ["GOOGLE_API_KEY"] = ""

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Retention Copilot", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

from src.streamlit_ui import backend, components, state, styles  # noqa: E402
from src.streamlit_ui.pages import activity, customer_copilot, evaluation, overview, retention  # noqa: E402

styles.inject()
st.logo(str(Path(__file__).parent / "src" / "streamlit_ui" / "assets" / "logo.svg"), size="large")

state.PAGES.update(
    overview=st.Page(overview.render, title="Overview", icon=":material/space_dashboard:", url_path="overview",
                     default=True),
    copilot=st.Page(customer_copilot.render, title="Customer Copilot", icon=":material/support_agent:",
                    url_path="copilot"),
    retention=st.Page(retention.render, title="Retention Analysis", icon=":material/trending_down:",
                      url_path="retention"),
    activity=st.Page(activity.render, title="Agent Activity", icon=":material/account_tree:", url_path="activity"),
    evaluation=st.Page(evaluation.render, title="Evaluation / Results", icon=":material/fact_check:",
                       url_path="evaluation"),
)
page = st.navigation(list(state.PAGES.values()))

try:
    backend.base_url()  # starts the existing API once per process
except backend.BackendError as err:
    components.show_error(err)
    st.stop()

components.sidebar_status()
page.run()
