"""Resolution worker: decides resolve / offer / escalate / refuse and drafts the customer message.

- The OUTCOME is decided by `decide_outcome(state)`, a pure function over guard flags, tool results,
  eligibility decisions and policy thresholds (unit-testable, no LLM).
- The MESSAGE is drafted by Gemini (structured output: ResolutionMessage) from structured facts,
  citing only clauses in the allowed set; unknown citations are dropped. Without Gemini, or if it
  fails, a deterministic policy-grounded template is used.
"""

from __future__ import annotations

import json
import re

from langgraph.runtime import Runtime

from src.agents.common import Deps, engine_entry
from src.audit.audit_middleware import audit
from src.context.isolate import view
from src.context.quarantine import untrusted_prompt
from src.context.select import personal_context
from src.guardrails.output_guard import CITATION_RE, known_citations
from src.llm import structured_call
from src.policy_limits import (
    APPROVAL_THRESHOLD,
    ETF_PCT_OF_PRICE,
    OVERAGE_PER_GB,
    PLANS,
    REPEAT_COMPLAINT_ESCALATION,
)
from src.resilience import ExternalCallFailed
from src.run_context import agent_scope
from src.schemas import ResolutionMessage, ResolutionOutput

NODE = "resolution_agent"
FRAUD_REF = "POL-RET-001 §2.2"


# --- outcome (pure) ------------------------------------------------------------

def decide_outcome(state: dict) -> dict:
    """Return {outcome, reason, escalation_reason} from state alone."""
    def out(outcome, reason, esc=None):
        return {"outcome": outcome, "reason": reason, "escalation_reason": esc}

    if state.get("guard_blocked"):
        if state.get("guard_reason") == "safety_threat":
            return out("escalate", "safety threat in customer message", "safety_threat")
        return out("refuse", state.get("guard_reason") or "input_guard")
    if state.get("account_failed") or (state.get("offer_decision") or {}).get("status") == "tool_failure":
        return out("escalate", "a required tool failed", "tool_failure")
    acct = state.get("account_summary") or {}
    intent = state.get("intent")
    if state.get("mentions_regulator"):
        return out("escalate", "customer mentioned a regulator or legal action", "regulator_mention")
    if intent == "complaint" and acct.get("complaints_90d", 0) + 1 >= REPEAT_COMPLAINT_ESCALATION:
        return out("escalate", f"{acct.get('complaints_90d', 0) + 1} complaints in 90 days", "repeat_complaint")

    decision = state.get("offer_decision") or {}
    proposal = state.get("offer_proposal")
    if intent == "cancellation":
        if proposal:
            return out("offer_pending_approval" if proposal["needs_approval"] else "offer",
                       f"{proposal['offer_type']} allowed by check_offer_eligibility")
        if any(FRAUD_REF in b["policy_refs"] for b in decision.get("blocked", [])):
            return out("escalate", "account flagged for fraud/collections review", "fraud_or_collections")
        return out("resolve", "no eligible retention offer; cancellation handled per policy")
    if intent == "billing_query" and proposal:
        return out("offer_pending_approval" if proposal["needs_approval"] else "resolve",
                   "billing credit for verified duplicate charge")
    return out("resolve", f"{intent} answered from account data and policy")


# --- templates -----------------------------------------------------------------

def _money(v: float) -> str:
    return f"${v:,.2f}"


HEAVY_USE_RE = re.compile(r"video|stream|work(?:ing)? from home|gaming|hotspot|upload|tether", re.I)
DEFAULT_HEADROOM, HEAVY_HEADROOM = 1.10, 1.50


def second_person(text: str) -> str:
    """Turn the customer's own words into a reply fragment ('I'm moving' -> 'you're moving')."""
    swaps = [(r"\bI'm\b", "you're"), (r"\bI am\b", "you are"), (r"\bI'll\b", "you'll"), (r"\bI'd\b", "you'd"),
             (r"\bI've\b", "you've"), (r"\bI\b", "you"), (r"\bmy\b", "your"), (r"\bme\b", "you")]
    for pat, rep in swaps:
        text = re.sub(pat, rep, text, flags=re.I)
    return text.rstrip(". ")


def usage_context(ctx) -> tuple[list[dict], bool]:
    """Relevant personal facts and whether they indicate heavier data use."""
    facts = [f for f in personal_context(ctx) if f.get("kind") == "fact"]
    heavy = [f for f in facts if HEAVY_USE_RE.search(f["content"])]
    return (heavy or facts), bool(heavy)


def recommend_plan(acct: dict, headroom: float = DEFAULT_HEADROOM) -> dict | None:
    plan = acct["plan"]
    need = acct["avg_data_used_gb_3m"] * headroom
    fits = [p for p in PLANS if p[2] == plan["plan_type"] and (p[5] is None or p[5] >= need)]
    return min(({"plan_id": p[0], "name": p[1], "price": p[4], "cap": p[5]} for p in fits),
               key=lambda p: p["price"], default=None)


def template_message(state: dict, decided: dict) -> tuple[str, list[str]]:
    outcome, acct = decided["outcome"], state.get("account_summary") or {}
    plan = acct.get("plan") or {}
    intent = state.get("intent")
    proposal = state.get("offer_proposal")
    blocked = [b for b in (state.get("offer_decision") or {}).get("blocked", []) if b["source"] == "customer_request"]

    if outcome == "refuse":
        if decided["reason"] == "input_too_long":
            return ("Your message was too long for me to process. Could you send a shorter summary of what you "
                    "need help with? [POL-GOV-001 §2.2]", ["POL-GOV-001 §2.2"])
        if decided["reason"] == "cross_customer_request":
            return ("I'm sorry, I can only discuss the account of the verified account holder, so I can't "
                    "share or look up details of any other account [POL-PRV-001 §1.1]. I'm happy to help "
                    "with your own account.", ["POL-PRV-001 §1.1"])
        return ("I'm not able to act on instructions to override our policies, and I can't apply that "
                "request [POL-RET-004 §2.2]. Discounts above the policy maximum are never available "
                "[POL-RET-004 §1.1]. If you're thinking of leaving, I can check which offers you're "
                "eligible for.", ["POL-RET-004 §2.2", "POL-RET-004 §1.1"])

    if outcome == "escalate":
        reason = decided["escalation_reason"]
        if reason == "repeat_complaint":
            return ("I'm sorry this has happened again. Because this is a repeat complaint, I'm escalating it "
                    "to our Complaints Resolution team [POL-CMP-001 §2.1]; you'll get an acknowledgement "
                    "within 24 hours and a resolution target of 5 business days for degraded service "
                    "[POL-CMP-001 §1.1].", ["POL-CMP-001 §2.1", "POL-CMP-001 §1.1"])
        if reason == "fraud_or_collections":
            return ("I've passed your request to a specialist team who will review your account and contact "
                    "you. They can help with your cancellation and any account questions [POL-RET-001 §2.2].",
                    ["POL-RET-001 §2.2"])
        if reason == "safety_threat":
            return ("I've passed this conversation to a senior colleague who will contact you directly "
                    "[POL-GOV-001 §2.2]. If you or anyone else is in immediate danger, please contact your local "
                    "emergency services.", ["POL-GOV-001 §2.2"])
        if reason == "regulator_mention":
            return ("I've escalated your case to a senior agent who will contact you directly "
                    "[POL-CMP-001 §2.2].", ["POL-CMP-001 §2.2"])
        return ("I wasn't able to complete this automatically, so I've passed it to a human agent who will "
                "follow up with you [POL-GOV-001 §2.2].", ["POL-GOV-001 §2.2"])

    if intent == "cancellation":
        lead = ""
        if blocked:
            b = blocked[0]
            lead = (f"I'm not able to offer {b['value']:g}% off, as that is above what our policy allows "
                    f"[{b['policy_refs'][0]}]. ")
        if proposal:
            desc = _offer_desc(proposal, plan)
            refs = list(dict.fromkeys(proposal["policy_refs"]))
            if outcome == "offer_pending_approval":
                return (lead + f"I'd like to keep you with us: I can put forward {desc}. Offers of this value "
                        f"need a team lead's approval [POL-RET-003 §1.2], so it is pending approval and I'll "
                        f"confirm once it's reviewed. If you'd still prefer to cancel, that's your choice "
                        f"[POL-CAN-001 §1.3].", refs + ["POL-RET-003 §1.2", "POL-CAN-001 §1.3"])
            return (lead + f"Before you go, I can offer you {desc} [{refs[0]}]. If you'd still prefer to "
                    f"cancel, I'll process that for you [POL-CAN-001 §1.3].", refs + ["POL-CAN-001 §1.3"])
        etf = ""
        if plan.get("plan_type") == "postpaid" and acct.get("contract_end_date"):
            etf = (f" As a postpaid customer, an early-termination fee of {ETF_PCT_OF_PRICE:.0%} of your "
                   f"monthly price applies for each remaining contract month [POL-CAN-001 §2.2].")
        return ("I understand you'd like to cancel, and I'll respect that decision [POL-CAN-001 §1.3]." + etf,
                ["POL-CAN-001 §1.3"] + (["POL-CAN-001 §2.2"] if etf else []))

    if intent == "billing_query":
        findings = state.get("billing_findings") or []
        invoices = (state.get("billing") or {}).get("invoices") or []
        dup = next((f for f in findings if f["type"] == "duplicate_charge"), None)
        if dup and proposal:
            status = ("is pending a team lead's approval [POL-RET-003 §1.2]" if proposal["needs_approval"]
                      else "has been applied to your account [POL-RET-003 §1.1]")
            return (f"You're right: the \"{dup['item']}\" was billed {dup['occurrences']} times on your latest "
                    f"invoice. A credit of {_money(dup['credit_due'])} for the duplicate {status}, as verified "
                    f"duplicate charges are credited in full [POL-BIL-001 §2.1].",
                    ["POL-BIL-001 §2.1", "POL-RET-003 §1.2" if proposal["needs_approval"] else "POL-RET-003 §1.1"])
        over = next((f for f in findings if f["type"] == "overage"), None)
        if over and invoices:
            prev = invoices[1]["total"] if len(invoices) > 1 else None
            rec = recommend_plan(acct)
            msg = (f"Your latest bill is {_money(invoices[0]['total'])}"
                   + (f", up from {_money(prev)}" if prev else "")
                   + f", because it includes a {over['item'].split(' @')[0][0].lower() + over['item'].split(' @')[0][1:]} charged at "
                     f"{_money(OVERAGE_PER_GB)}/GB above your {plan.get('data_cap_gb')} GB cap [POL-BIL-002 §1.2].")
            cites = ["POL-BIL-002 §1.2"]
            if rec and rec["plan_id"] != plan.get("plan_id"):
                msg += (f" Your 3-month average is {acct['avg_data_used_gb_3m']} GB, so {rec['name']} "
                        f"({_money(rec['price'])}/month) would avoid overage [POL-BIL-002 §2.1] [POL-PLN-001 §2.2].")
                cites += ["POL-BIL-002 §2.1", "POL-PLN-001 §2.2"]
            return msg, cites
        if invoices:
            items = ", ".join(li["item"] for li in invoices[0]["line_items"])
            return (f"Your latest invoice totals {_money(invoices[0]['total'])}: {items}, plus taxes "
                    f"[POL-BIL-002 §1.1].", ["POL-BIL-002 §1.1"])
        return ("I couldn't find a recent invoice to explain; a colleague will follow up [POL-GOV-001 §2.2].",
                ["POL-GOV-001 §2.2"])

    if intent == "complaint":
        return ("I'm sorry about the trouble. I've logged your complaint; you'll get an acknowledgement within "
                "24 hours and we aim to resolve degraded-service issues within 5 business days [POL-CMP-001 §1.1].",
                ["POL-CMP-001 §1.1"])

    if intent == "plan_change":
        cap = plan.get("data_cap_gb")
        cap_txt = "unlimited data" if cap is None else f"{cap} GB of data per month"
        facts, heavy = usage_context(state)
        rec = recommend_plan(acct, HEAVY_HEADROOM if heavy else DEFAULT_HEADROOM)
        msg = ""
        if facts:
            recalled = "; ".join(second_person(f["content"]) for f in facts[:2])
            when = "Last time you told me" if any(f.get("source") == "long_term" for f in facts[:2]) else "You mentioned"
            msg = f"{when} that {recalled}. " + ("I've allowed extra headroom for heavier data use. " if heavy else "")
        msg += (f"You're on {plan.get('name')} with {cap_txt}; last cycle you used "
                f"{acct.get('data_used_gb_last_cycle')} GB (3-month average {acct.get('avg_data_used_gb_3m')} GB).")
        if rec and rec["plan_id"] != plan.get("plan_id"):
            rec_cap = "unlimited data" if rec["cap"] is None else f"{rec['cap']} GB"
            msg += (f" Based on your usage I'd recommend {rec['name']} ({rec_cap}, {_money(rec['price'])}/month) "
                    f"[POL-PLN-001 §2.2]. Upgrades take effect immediately with pro-rated pricing "
                    f"[POL-PLN-001 §1.1].")
            cites = ["POL-PLN-001 §2.2", "POL-PLN-001 §1.1"]
        elif rec is None:  # nothing of this plan type covers the need
            msg += (" None of our " + str(plan.get("plan_type")) + " plans covers that usage with enough headroom "
                    "[POL-PLN-001 §2.2]. Moving from prepaid to postpaid needs a credit check that a colleague "
                    "can arrange for you [POL-PLN-001 §2.1].")
            cites = ["POL-PLN-001 §2.2", "POL-PLN-001 §2.1"]
        else:
            msg += " Your current plan covers your usage with headroom, so no change is needed [POL-PLN-001 §2.2]."
            cites = ["POL-PLN-001 §2.2"]
        return msg, cites

    return ("Thanks for getting in touch. A colleague will follow up [POL-GOV-001 §2.2].", ["POL-GOV-001 §2.2"])


def _offer_desc(p: dict, plan: dict) -> str:
    if p["offer_type"] == "discount_pct":
        return f"{p['value']:g}% off your {plan.get('name', 'plan')} for {p['months']} months"
    if p["offer_type"] == "data_boost":
        return f"an extra {p['value']:g} GB of data per month for {p['months']} months at no charge"
    if p["offer_type"] == "credit":
        return f"a one-time bill credit of {_money(p['value'])}"
    if p["offer_type"] == "plan_upgrade":
        return f"an upgrade to the next tier at your current price for {p['months']} months"
    return "an early-termination fee waiver"


# --- drafting with Gemini --------------------------------------------------------

_SYSTEM = (
    "You write replies for a telecom customer-care copilot. The outcome is already decided; write a "
    "concise, empathetic customer message (max 120 words) consistent with it. Use only the facts given. "
    "Cite every policy statement inline like [POL-RET-003 §1.2], using ONLY citations from the allowed "
    "list. Never mention internal IDs, other customers, or offers not in the facts. If an offer is "
    "pending approval, say it is pending, never that it is applied. Customer identifiers are masked."
)


def allowed_citations(state: dict, template_refs: list[str]) -> list[str]:
    refs = [c["citation"] for c in state.get("policy_citations") or []]
    for chk in (state.get("offer_decision") or {}).get("checks", []):
        refs += chk.get("policy_refs", [])
    refs += template_refs
    return [r for r in dict.fromkeys(refs) if r in known_citations()]


async def resolution_agent(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        state = view(NODE, state)  # isolate: structured results + selected personal context only
        decided = decide_outcome(state)
        template, template_refs = template_message(state, decided)
        allowed = allowed_citations(state, template_refs)
        message, cites, engine, detail = template, template_refs, "rules", "template (no GOOGLE_API_KEY)"
        if runtime.context.use_llm:
            facts = {
                "outcome": decided, "intent": state.get("intent"),
                "account": state.get("account_summary"), "billing_findings": state.get("billing_findings"),
                "offer_proposal": state.get("offer_proposal"),
                "blocked_offers": (state.get("offer_decision") or {}).get("blocked"),
                "policy_answer": state.get("policy_answer"), "approval_threshold_usd": APPROVAL_THRESHOLD,
                "reference_reply": template,
            }
            # facts/preferences the customer stated are untrusted text: they go in the quarantined block
            personal = "\n".join(f"- {f['content']}" for f in personal_context(state)) or "(none)"
            try:
                draft = await structured_call(ResolutionMessage, untrusted_prompt(
                    _SYSTEM, personal,
                    task=f"Allowed citations: {allowed}. Write the customer message. You may refer to the "
                         f"customer's own stated circumstances above where relevant.",
                    trusted_context=f"Facts: {json.dumps(facts, default=str)}\nCustomer-stated context:"),
                    what="resolution.draft")
                used = list(dict.fromkeys(draft.citations + CITATION_RE.findall(draft.customer_message)))
                bad = [c for c in used if c not in allowed]
                if bad or not used:
                    detail = f"gemini draft rejected (citations {bad or 'missing'}); template used"
                else:
                    message, cites, engine, detail = draft.customer_message, used, "gemini", "gemini draft"
            except ExternalCallFailed as exc:
                detail = f"gemini failed ({exc.reason}); template used"
        resolution = {**decided, "customer_message": message, "citations": cites, "engine": engine}
        audit("resolution_drafted", decided["outcome"], reason=decided["reason"], policy_ref=cites,
              details={"escalation_reason": decided["escalation_reason"], "engine": engine})
        return ResolutionOutput(resolution=resolution, engine_log=[engine_entry(NODE, engine, detail)]).update()


def route_after_resolution(state: dict) -> str:
    outcome = (state.get("resolution") or {}).get("outcome")
    if outcome == "escalate":
        return "escalate"
    if outcome == "offer_pending_approval" and state.get("needs_human_approval"):
        return "human_approval"
    return "memory_writer"
