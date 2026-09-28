"""Per-contact run context held in contextvars.

A single `run_id` (uuid) per contact is propagated into Phoenix span attributes, tool_calls.jsonl,
agent_actions.jsonl and mcp_transcript.jsonl so every artifact cross-references. The MCP session
token for the authenticated caller also lives here, so tool auth never passes through the LLM.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
agent_var: ContextVar[str | None] = ContextVar("agent", default=None)
mcp_auth_token_var: ContextVar[str | None] = ContextVar("mcp_auth_token", default=None)
llm_enabled_var: ContextVar[bool] = ContextVar("llm_enabled", default=True)  # False after a failed preflight


def new_run_id() -> str:
    return str(uuid.uuid4())


def current_run_id() -> str | None:
    return run_id_var.get()


def current_session_id() -> str | None:
    return session_id_var.get()


def current_agent() -> str | None:
    return agent_var.get()


@contextmanager
def run_scope(run_id: str | None = None, session_id: str | None = None,
              mcp_auth_token: str | None = None) -> Iterator[str]:
    """Bind run/session/auth context for the duration of one contact."""
    rid = run_id or new_run_id()
    tokens = [run_id_var.set(rid), session_id_var.set(session_id),
              mcp_auth_token_var.set(mcp_auth_token)]
    try:
        yield rid
    finally:
        for var, tok in zip((run_id_var, session_id_var, mcp_auth_token_var), tokens):
            var.reset(tok)


@contextmanager
def agent_scope(agent: str) -> Iterator[None]:
    """Mark which agent/node is currently acting (used by logging and audit middleware)."""
    tok = agent_var.set(agent)
    try:
        yield
    finally:
        agent_var.reset(tok)
