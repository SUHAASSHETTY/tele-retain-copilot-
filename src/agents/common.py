"""Shared helpers for agent nodes: runtime dependencies, engine bookkeeping, context selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from src.llm import llm_available


@dataclass
class Deps:
    """Runtime context for one contact (not checkpointed)."""
    mcp: Any                              # src.tools.mcp_client.BoundSession
    use_llm: bool = field(default_factory=llm_available)
    max_steps: int = 12
    memory: Any = None                    # src.memory.long_term.LongTermMemory (None = disabled)


def engine_entry(node: str, engine: str, detail: str = "") -> dict:
    return {"node": node, "engine": engine, "detail": detail[:200]}


def customer_turns(state: dict, last_n: int = 3) -> list[str]:
    """Select: the most recent sanitized customer turns (already masked by the input guard)."""
    turns = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    return turns[-last_n:]


def error_entry(node: str, what: str, reason: str) -> dict:
    return {"node": node, "what": what, "reason": reason}
