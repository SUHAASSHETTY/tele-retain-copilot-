"""Compress context: summarization middleware for long conversations.

When the thread's messages exceed CONTEXT_TOKEN_BUDGET (approximate token count), everything but
the last KEEP_LAST_MESSAGES messages is folded into a running summary and removed from state
(RemoveMessage). With Gemini available the summary is produced by LangMem's `summarize_messages`;
otherwise an extractive summary is used. Durable facts survive separately in `session_facts`.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.resilience import ExternalCallFailed, resilient_call

CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "600"))
KEEP_LAST_MESSAGES = int(os.getenv("CONTEXT_KEEP_LAST_MESSAGES", "4"))


@dataclass
class Compression:
    tokens_before: int
    tokens_after: int
    summary: str | None
    removals: list[RemoveMessage] = field(default_factory=list)
    engine: str = "none"


def _extractive(previous: str | None, messages: list[BaseMessage]) -> str:
    lines = [previous] if previous else []
    for m in messages:
        who = "Customer" if isinstance(m, HumanMessage) else "Copilot" if isinstance(m, AIMessage) else "Other"
        text = " ".join(str(m.content).split())
        lines.append(f"- {who}: {text[:200]}{'...' if len(text) > 200 else ''}")
    return "\n".join(lines)[-2500:]


async def _langmem_summary(previous: str | None, messages: list[BaseMessage]) -> str:
    from langmem.short_term import RunningSummary, summarize_messages

    from src.llm import get_chat_model

    running = RunningSummary(summary=previous, summarized_message_ids=set(), last_summarized_message_id=None) \
        if previous else None
    result = await resilient_call(lambda: asyncio.to_thread(
        summarize_messages, messages, running_summary=running, model=get_chat_model(),
        max_tokens=64, max_tokens_before_summary=1, max_summary_tokens=200), what="context.summarize")
    if result.running_summary is None:
        raise ExternalCallFailed("context.summarize", "no summary produced", 1)
    return result.running_summary.summary


async def compress_messages(messages: list[BaseMessage], summary: str | None, *, use_llm: bool,
                            budget: int | None = None, keep_last: int | None = None) -> Compression:
    budget = budget or CONTEXT_TOKEN_BUDGET
    keep_last = keep_last or KEEP_LAST_MESSAGES
    before = count_tokens_approximately(messages)
    if before <= budget or len(messages) <= keep_last:
        return Compression(before, before, summary)
    old, recent = messages[:-keep_last], messages[-keep_last:]
    engine = "extractive"
    if use_llm:
        try:
            new_summary, engine = await _langmem_summary(summary, old), "gemini(langmem)"
        except ExternalCallFailed:
            new_summary = _extractive(summary, old)
    else:
        new_summary = _extractive(summary, old)
    return Compression(before, count_tokens_approximately(recent), new_summary,
                       [RemoveMessage(id=m.id) for m in old if m.id], engine)
