"""Write context: persist notes outside the prompt window.

- Scratchpad: each node appends notes to its own `scratchpad[node]` lane in graph state (merged by
  the `merge_scratchpad` reducer). Other workers never see a lane that is not theirs (isolate.py).
- Session facts: durable facts/preferences stated in this conversation are written to state
  (`session_facts`), so they survive history compression within the thread.
- Long-term: at the end of each turn, facts, preferences, the issue handled and any offers are
  written to the customer's LangMem store (src.memory.long_term).
"""

from __future__ import annotations

from src.memory.long_term import CustomerMemory, LongTermMemory

MAX_NOTES_PER_LANE = 20


def merge_scratchpad(left: dict | None, right: dict | None) -> dict:
    """Reducer: append notes per node lane, keeping the most recent MAX_NOTES_PER_LANE."""
    merged = {k: list(v) for k, v in (left or {}).items()}
    for node, notes in (right or {}).items():
        merged[node] = (merged.get(node, []) + list(notes))[-MAX_NOTES_PER_LANE:]
    return merged


def merge_facts(left: list | None, right: list | None) -> list:
    """Reducer: union of session facts by content (first occurrence wins)."""
    out, seen = [], set()
    for f in (left or []) + (right or []):
        key = f["content"].lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def scratch(node: str, *notes: str) -> dict:
    """State update appending notes to this node's own scratchpad lane."""
    return {"scratchpad": {node: [n[:300] for n in notes]}}


def turn_memories(state: dict) -> list[CustomerMemory]:
    """What this turn contributes to long-term memory (from trusted, structured state)."""
    sid = state.get("session_id")
    mems = [CustomerMemory(kind=f["kind"], content=f["content"], session_id=sid)
            for f in state.get("session_facts") or [] if f.get("session_id") == sid]
    res = state.get("resolution") or {}
    if state.get("intent") and res.get("outcome"):
        mems.append(CustomerMemory(kind="past_issue", session_id=sid,
                                   content=f"{state['intent'].replace('_', ' ')} contact resolved as "
                                           f"{res['outcome']}: {res.get('reason', '')}"[:400]))
    for chk in (state.get("offer_decision") or {}).get("checks", []):
        desc = (f"{chk['offer_type']} {chk['value']:g}{'%' if chk['offer_type'] == 'discount_pct' else ''}"
                f" for {chk.get('months', 1)} month(s)")
        mems.append(CustomerMemory(kind="past_offer", session_id=sid,
                                   content=f"{desc} was {chk['decision']} ({', '.join(chk['policy_refs'])})"[:400]))
    approval = state.get("approval")
    if approval:
        mems.append(CustomerMemory(kind="past_offer", session_id=sid,
                                   content=f"proposed offer {'approved' if approval['approved'] else 'rejected'} "
                                           f"by a human approver"))
    return mems


async def write_long_term(memory: LongTermMemory, customer_id: str, state: dict) -> list[dict]:
    return await memory.remember(customer_id, turn_memories(state))
