"""Select context: pull only what is relevant into a worker's prompt.

- Long-term memories: semantic top-k within the caller's own namespace, plus the most recent
  facts when the customer explicitly refers back to an earlier contact ("last time", "I told you").
- Session facts: facts stated earlier in this conversation (kept even after compression).
- Policy chunks: only clauses the RAG judge graded relevant, capped per worker.
"""

from __future__ import annotations

import re

from src.memory.long_term import LongTermMemory

REFERS_BACK_RE = re.compile(r"\b(last time|previous(ly)?|before|earlier|i told you|i mentioned|again|"
                            r"what i said|as i said)\b", re.I)
PERSONAL_KINDS = ("fact", "preference")


async def select_memories(memory: LongTermMemory | None, customer_id: str, query: str,
                          limit: int = 3, exclude_session: str | None = None) -> list[dict]:
    """Memories from earlier sessions (this session's facts are already in state)."""
    if memory is None or not query:
        return []
    hits = [h for h in await memory.recall(customer_id, query, limit=limit + 3)
            if not exclude_session or h.get("session_id") != exclude_session][:limit]
    if REFERS_BACK_RE.search(query):  # an explicit reference to the past: include personal facts
        personal = [m for m in await memory.all(customer_id) if m["kind"] in PERSONAL_KINDS
                    and (not exclude_session or m.get("session_id") != exclude_session)]
        personal.sort(key=lambda m: m.get("recorded_at") or "", reverse=True)
        ids = {h["id"] for h in hits}
        hits += [m | {"score": None, "selected_by": "refers_back"} for m in personal[:limit] if m["id"] not in ids]
    return hits


def personal_context(ctx) -> list[dict]:
    """Facts/preferences available to the resolution: this session's, then long-term ones."""
    facts = list(ctx.get("session_facts") or [])
    seen = {f["content"].lower() for f in facts}
    for m in ctx.get("memories") or []:
        if m.get("kind") in PERSONAL_KINDS and m["content"].lower() not in seen:
            facts.append({**m, "source": "long_term"})
            seen.add(m["content"].lower())
    return facts


def select_policy(citations: list[dict], limit: int = 4) -> list[dict]:
    return sorted(citations or [], key=lambda c: -(c.get("score") or 0))[:limit]
