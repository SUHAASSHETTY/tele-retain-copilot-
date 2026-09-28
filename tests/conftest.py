"""Shared fixtures. Tests never need GOOGLE_API_KEY (Gemini is disabled or stubbed) and never write
to the committed evidence logs: tool and audit logs are redirected to a temp dir per test."""

from __future__ import annotations

import pytest

from src.audit import audit_middleware
from src.config import AGENT_ACTIONS_LOG, TOOL_CALLS_LOG
from src.run_context import llm_enabled_var
from src.tools import logging_middleware


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path):
    logging_middleware.set_log_path(tmp_path / "tool_calls.jsonl")
    audit_middleware.set_log_path(tmp_path / "agent_actions.jsonl")
    token = llm_enabled_var.set(False)
    yield tmp_path
    llm_enabled_var.reset(token)
    logging_middleware.set_log_path(TOOL_CALLS_LOG)
    audit_middleware.set_log_path(AGENT_ACTIONS_LOG)
