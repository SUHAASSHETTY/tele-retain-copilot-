"""Human-in-the-loop nodes: approval gate (LangGraph interrupt), clarify/decline, escalate."""

from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.agents.common import Deps, error_entry
from src.audit.audit_middleware import audit
from src.context.isolate import view
from src.guardrails.pii import mask_customer_id
from src.run_context import agent_scope
from src.schemas import ApprovalOutput, ApprovalRequestOutput, HandoffOutput


async def request_approval(state: dict, runtime: Runtime[Deps]) -> dict:
    """Build and audit the approval request. Kept separate from `human_approval` because a node
    that calls interrupt() re-executes from the top on resume; side effects must not live there."""
    with agent_scope("human_approval"):
        state = view("human_approval", state)
        p = state["offer_proposal"]
        request = {
            "type": "offer_approval",
            "customer_ref": mask_customer_id(state["customer_id"]),
            "offer_type": p["offer_type"], "value": p["value"], "months": p["months"],
            "offer_value_usd": p["offer_value_usd"], "policy_refs": p["policy_refs"],
            "reasons": p["reasons"],
        }
        audit("approval_requested", "pending", tool="interrupt", reason="; ".join(p["reasons"]),
              policy_ref=p["policy_refs"], details=request)
        return ApprovalRequestOutput(approval_request=request).update()


async def human_approval(state: dict, runtime: Runtime[Deps]) -> dict:
    """Pause the graph (interrupt) until a human approves or rejects an above-threshold offer/credit."""
    with agent_scope("human_approval"):
        state = view("human_approval", state)
        request = state["approval_request"]
        answer = interrupt(request)  # resumes with {"approved": bool, "approver": str, "note": str}
        answer = answer if isinstance(answer, dict) else {}
        approved = bool(answer.get("approved"))
        approver = answer.get("approver", "unknown")
        automated = str(approver).endswith(("auto-approve", "auto-reject"))  # scripted stand-ins, not people
        audit("approval_granted" if approved else "approval_denied", "granted" if approved else "denied",
              actor=f"{'system' if automated else 'human'}:{approver}",
              actor_type="system" if automated else "human", tool="interrupt", reason=answer.get("note") or None,
              policy_ref="POL-RET-003 §1.2", details=request)
        p = state["offer_proposal"]
        res = dict(state["resolution"])
        if approved:
            res["outcome"] = "offer"
            res["customer_message"] = (
                f"Good news: your offer has been approved by a team lead and is now applied - "
                f"{_desc(p)} [POL-RET-003 §1.2]. If you'd still prefer to cancel, that's your choice "
                f"[POL-CAN-001 §1.3].")
        else:
            res["outcome"] = "resolve"
            res["customer_message"] = (
                "I'm sorry, the offer I put forward was not approved [POL-RET-003 §1.2]. If you'd still like "
                "to cancel, I'll respect that decision [POL-CAN-001 §1.3].")
        res["citations"] = ["POL-RET-003 §1.2", "POL-CAN-001 §1.3"]
        res["approval"] = {"approved": approved, "approver": approver}
        return ApprovalOutput(approval={"approved": approved, "approver": approver}, resolution=res,
                              needs_human_approval=False).update()


def _desc(p: dict) -> str:
    if p["offer_type"] == "discount_pct":
        return f"{p['value']:g}% off your plan for {p['months']} months"
    if p["offer_type"] == "credit":
        return f"a one-time credit of ${p['value']:,.2f}"
    return p["offer_type"].replace("_", " ")


async def clarify(state: dict, runtime: Runtime[Deps]) -> dict:
    """Ambiguous / low-confidence -> one clarifying question (then escalate); out of scope -> decline."""
    with agent_scope("clarify"):
        state = view("clarify", state)
        count = state.get("clarify_count", 0)
        if state.get("intent") == "out_of_scope":
            res = {"outcome": "decline", "reason": "out_of_scope", "escalation_reason": None, "engine": "rules",
                   "customer_message": "I'm sorry, I can only help with your telecom account and services "
                                       "[POL-GOV-001 §2.1]. I can help with your bill, plan, a complaint or "
                                       "your contract.", "citations": ["POL-GOV-001 §2.1"]}
            audit("request_declined", "declined", reason="out_of_scope", policy_ref="POL-GOV-001 §2.1",
                  details={"intent_confidence": state.get("intent_confidence")})
            return HandoffOutput(resolution=res).update()
        if count >= 1:
            audit("escalation", "requested", reason="still unclear after one clarification",
                  policy_ref="POL-GOV-001 §2.2", details={"clarify_count": count})
            return HandoffOutput(resolution={
                "outcome": "escalate", "reason": "still unclear after clarification",
                "escalation_reason": "ambiguous_request", "engine": "rules",
                "customer_message": "", "citations": ["POL-GOV-001 §2.2"]}, clarify_count=count).update()
        audit("clarification_requested", "clarify", reason="ambiguous or low-confidence intent",
              policy_ref="POL-GOV-001 §2.2", details={"intent_confidence": state.get("intent_confidence")})
        return HandoffOutput(resolution={
            "outcome": "clarify", "reason": "ambiguous or low-confidence intent", "escalation_reason": None,
            "engine": "rules", "citations": ["POL-GOV-001 §2.2"],
            "customer_message": "Happy to help. Could you tell me a bit more: is this about your bill, your "
                                "plan or data, a service problem (for example calls or signal), or your "
                                "contract? [POL-GOV-001 §2.2]"}, clarify_count=count + 1).update()


def route_after_clarify(state: dict) -> str:
    return "escalate" if (state.get("resolution") or {}).get("outcome") == "escalate" else "memory_writer"


_PRIORITY = {"repeat_complaint": "P2", "fraud_or_collections": "P2", "regulator_mention": "P1",
             "tool_failure": "P2", "safety_threat": "P1"}


async def escalate(state: dict, runtime: Runtime[Deps]) -> dict:
    """Open an escalation ticket via MCP and tell the customer; never fails the contact."""
    with agent_scope("escalate"):
        state = view("escalate", state)
        res = dict(state.get("resolution") or {})
        if not res or res.get("outcome") != "escalate":  # reached from the supervisor (loop guard / failure)
            reason = "tool_failure" if state.get("account_failed") else "other"
            why = ("account lookup failed" if state.get("account_failed")
                   else f"step limit reached ({state.get('step_count')})")
            res = {"outcome": "escalate", "reason": why, "escalation_reason": reason, "engine": "rules",
                   "citations": ["POL-GOV-001 §2.2"],
                   "customer_message": "I wasn't able to complete this automatically, so I've passed it to a "
                                       "human agent who will follow up with you [POL-GOV-001 §2.2]."}
        reason = res.get("escalation_reason") or "other"
        summary = f"{state.get('intent') or 'unknown'} contact escalated: {res.get('reason')}"
        errors = []
        ticket = None
        try:
            out = await runtime.context.mcp.call(
                "create_escalation_ticket", customer_id=state["customer_id"], reason=reason,
                summary=summary[:480], priority=_PRIORITY.get(reason, "P3"))
            ticket = out.get("ticket_id") if out.get("ok") else None
            if not out.get("ok"):
                errors.append(error_entry("escalate", "create_escalation_ticket", out["error"]["code"]))
        except Exception as exc:
            errors.append(error_entry("escalate", "create_escalation_ticket", type(exc).__name__))
        if not res.get("customer_message"):
            res["customer_message"] = ("I want to make sure you get the right help, so I've passed this to a "
                                       "colleague who will contact you [POL-GOV-001 §2.2].")
        if ticket:
            res["customer_message"] += f" Your reference is {ticket}."
        res["ticket_id"] = ticket
        audit("escalation", "ticket_created" if ticket else "ticket_failed", tool="create_escalation_ticket",
              reason=f"{reason}: {res.get('reason')}", policy_ref=res.get("citations"),
              details={"reason": reason, "ticket_id": ticket, "summary": summary})
        return HandoffOutput(resolution=res, errors=errors).update()
