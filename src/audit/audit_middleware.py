"""Audit trail of consequential agent actions -> logs/agent_actions.jsonl.

Record: {timestamp, run_id, session_id, actor, actor_type, action, tool, decision, reason,
         policy_ref, details}
- actor_type: "agent" (a worker node), "human" (an approver) or "system" (guards, middleware).
- Every value passes through src.guardrails.pii.mask_obj: no plaintext identifiers or amounts.

Consequential actions (ACTIONS below): guardrail blocks / sanitisation, data access refused,
offers proposed / blocked, approvals requested / granted / denied, escalations, output-guard
interventions, loop-limit hits, memory writes and the final resolution of each contact.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import AGENT_ACTIONS_LOG
from src.guardrails.pii import mask, mask_obj
from src.run_context import current_agent, current_run_id, current_session_id

ActorType = Literal["agent", "human", "system"]
SYSTEM_ACTORS = {"input_guard", "output_guard", "context_manager", "memory_writer", "supervisor", "system",
                 "mcp_client"}
ACTIONS = {
    "guardrail_block", "input_sanitized", "data_access_refused", "offer_eligibility_check",
    "offer_proposed", "offer_blocked", "offer_selection_failed", "approval_requested", "approval_granted",
    "approval_denied", "escalation", "output_guard_intervention", "loop_limit", "account_lookup_failed",
    "clarification_requested", "request_declined", "resolution_drafted", "contact_resolved",
    "memory_write", "memory_recalled",
}

_lock = threading.Lock()
_log_path: Path = AGENT_ACTIONS_LOG


def set_log_path(path: Path) -> None:
    global _log_path
    _log_path = path


def _actor(actor: str | None, actor_type: ActorType | None) -> tuple[str, ActorType]:
    name = actor or current_agent() or "system"
    if actor_type:
        return name, actor_type
    if name.startswith("human:"):
        return name, "human"
    return name, "system" if name in SYSTEM_ACTORS else "agent"


def audit(action: str, decision: str, *, tool: str | None = None, actor: str | None = None,
          actor_type: ActorType | None = None, reason: str | None = None, policy_ref: str | list[str] | None = None,
          details: dict[str, Any] | None = None) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown audit action '{action}'")
    name, kind = _actor(actor, actor_type)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": current_run_id(),
        "session_id": current_session_id(),
        "actor": name,
        "actor_type": kind,
        "action": action,
        "tool": tool,
        "decision": decision,
        "reason": mask(reason, amounts=True) if reason else None,
        "policy_ref": policy_ref,
        "details": mask_obj(details or {}),
    }
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record
