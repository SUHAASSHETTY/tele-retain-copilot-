"""Isolate context: every worker gets its own scoped view of state.

`view(worker, state)` builds a ScopedState containing only the keys that worker is allowed to
read, plus derived values (e.g. the last few sanitized customer turns instead of the full raw
history) and only that worker's own scratchpad lane. Reading any other key raises IsolationError,
so a worker cannot silently depend on context it was not given.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

# key -> allowed for worker. Derived keys: recent_turns, own_scratch.
WORKER_SCOPES: dict[str, frozenset[str]] = {
    "intent_agent": frozenset({"recent_turns", "summary", "own_scratch"}),
    "account_agent": frozenset({"customer_id", "intent", "own_scratch"}),
    "policy_retrieval_agent": frozenset({"intent", "account_summary", "billing_findings", "own_scratch"}),
    "retention_offer_agent": frozenset({"customer_id", "intent", "cancellation_intent", "requested_offer",
                                        "account_summary", "billing_findings", "own_scratch"}),
    "resolution_agent": frozenset({"intent", "guard_blocked", "guard_reason", "input_flags", "account_failed", "mentions_regulator",
                                   "account_summary", "billing", "billing_findings", "offer_proposal",
                                   "offer_decision", "policy_answer", "policy_citations", "session_facts",
                                   "memories", "summary", "recent_turns", "own_scratch"}),
    "clarify": frozenset({"intent", "intent_confidence", "clarify_count", "own_scratch"}),
    "escalate": frozenset({"customer_id", "intent", "resolution", "account_failed", "step_count", "own_scratch"}),
    "human_approval": frozenset({"customer_id", "offer_proposal", "resolution", "approval_request", "own_scratch"}),
}
RECENT_TURNS = 3


class IsolationError(KeyError):
    pass


class ScopedState(dict):
    """A dict restricted to one worker's scope."""

    def __init__(self, worker: str, allowed: frozenset[str], data: dict[str, Any]):
        super().__init__({k: v for k, v in data.items() if k in allowed})
        self.worker = worker
        self.allowed = allowed

    def _check(self, key: str) -> None:
        if key not in self.allowed:
            raise IsolationError(f"{self.worker} may not read '{key}' (not in its scoped context)")

    def __getitem__(self, key: str) -> Any:
        self._check(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._check(key)
        return super().get(key, default)


def view(worker: str, state: dict) -> ScopedState:
    allowed = WORKER_SCOPES[worker]
    data = dict(state)
    if "recent_turns" in allowed:
        turns = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        data["recent_turns"] = turns[-RECENT_TURNS:]
    data["own_scratch"] = (state.get("scratchpad") or {}).get(worker, [])
    return ScopedState(worker, allowed, data)
