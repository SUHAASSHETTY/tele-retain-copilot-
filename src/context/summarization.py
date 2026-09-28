"""Context middleware nodes.

`context_manager` (after the input guard, every turn):
  compress  - summarize old messages when over the token budget (src.context.compress)
  write     - extract durable facts/preferences from this turn into `session_facts`
  select    - recall the relevant long-term memories for this customer (src.context.select)
`memory_writer` (after the output guard): write this turn's facts, issue and offers to the
customer's long-term LangMem store. Turns blocked by the input guard are never written or mined,
so injected text cannot poison memory.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from src.agents.common import Deps, engine_entry
from src.audit.audit_middleware import audit
from src.context.compress import compress_messages
from src.context.select import select_memories
from src.context.write import write_long_term
from src.memory.long_term import extract_llm, extract_rules
from src.resilience import ExternalCallFailed
from src.run_context import agent_scope
from src.schemas import ContextOutput, MemoryWriteOutput


async def context_manager(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope("context_manager"):
        deps = runtime.context
        comp = await compress_messages(state.get("messages", []), state.get("summary"), use_llm=deps.use_llm)
        humans = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        latest = humans[-1].content if humans else ""
        sid = state.get("session_id")
        facts, memories, extract_engine = [], [], "skipped"
        if not state.get("guard_blocked") and latest:
            extract_engine = "rules"
            if deps.use_llm:
                try:
                    found, extract_engine = await extract_llm(latest, sid), "gemini(langmem)"
                except ExternalCallFailed:
                    found = extract_rules(latest, sid)
            else:
                found = extract_rules(latest, sid)
            turn_no = len(humans)
            facts = [{"kind": f.kind, "content": f.content, "session_id": sid, "turn": turn_no} for f in found]
            memories = await select_memories(deps.memory, state["customer_id"], latest, exclude_session=sid)
        if memories:
            audit("memory_recalled", f"{len(memories)} item(s)", reason="relevant long-term memories selected",
                  details={"kinds": [m["kind"] for m in memories],
                           "selected_by": [m.get("selected_by", "semantic") for m in memories]})
        stats = {"tokens_before": comp.tokens_before, "tokens_after": comp.tokens_after,
                 "compressed_messages": len(comp.removals), "summary_engine": comp.engine}
        return ContextOutput(
            summary=comp.summary, messages=comp.removals, session_facts=facts, memories=memories,
            context_stats=stats,
            engine_log=[engine_entry("context_manager", extract_engine,
                                     f"facts={len(facts)} memories={len(memories)} compressed={len(comp.removals)}")],
            scratchpad={"context_manager": [f"compress:{comp.engine} {comp.tokens_before}->{comp.tokens_after} tokens"]},
        ).update()


async def memory_writer(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope("memory_writer"):
        memory = runtime.context.memory
        if memory is None:
            return MemoryWriteOutput(memory_written=[]).update()
        if state.get("guard_blocked"):
            audit("memory_write", "skipped", reason="input guard blocked this turn; nothing persisted")
            return MemoryWriteOutput(memory_written=[]).update()
        written = await write_long_term(memory, state["customer_id"], state)
        if written:
            audit("memory_write", f"{len(written)} item(s)", reason="turn facts/issue/offers persisted",
                  details={"items": [{"kind": w["kind"], "content": w["content"]} for w in written]})
        return MemoryWriteOutput(memory_written=[{"kind": w["kind"], "content": w["content"]} for w in written]).update()
