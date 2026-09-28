"""Intent-classification worker.

Gemini (structured output: IntentClassification) classifies the quarantined customer text into
billing_query / plan_change / complaint / cancellation / ambiguous / out_of_scope, with a
confidence, any explicitly requested offer, and whether the customer wants to cancel. If Gemini
is unavailable or fails after retries, deterministic keyword rules are used instead.
"""

from __future__ import annotations

import re

from langgraph.runtime import Runtime

from src.agents.common import Deps, engine_entry
from src.context.isolate import view
from src.context.quarantine import untrusted_prompt
from src.llm import structured_call
from src.resilience import ExternalCallFailed
from src.run_context import agent_scope
from src.schemas import IntentClassification, IntentOutput, RequestedOffer

NODE = "intent_agent"

_RULES = [
    ("out_of_scope", 0.9, r"\b(restaurant|table for|flight|hotel|weather|recipe|movie|taxi|pizza|laptop recommendation)\b"),
    ("cancellation", 0.85, r"\b(cancel\w*|leaving|i'?m gone|switch(ing)? (provider|carrier)|port (my )?number|terminate|close my account)\b"),
    ("complaint", 0.8, r"\b(complain(ing|t)?|fed up|dropped calls?|nothing has been fixed|terrible service|unacceptable|outage)\b"
                       r"|\b(signal|reception|coverage|no service)\b|\b(keeps?|kept) (dropping|cutting out|disconnecting)\b"
                       r"|\bcalls? (keep )?(drop|dropping|cut(ting)? out)\b|\bslow (data|internet|speeds?)\b"
                       r"|\b(data|internet|connection|speeds?)\b.{0,15}\bslow\b|\bbuffering\b"),
    ("billing_query", 0.8, r"\b(bill|charged?|charges|invoice|refund|overcharg\w*|payment)\b"),
    ("plan_change", 0.75, r"\b(plan|data allowance|allowance|upgrade|downgrade|more data|running out of data|data cap)\b"),
]
_HALF_RE = re.compile(r"\bhalf (off|price)\b", re.I)
_PCT_RE = re.compile(r"(\d{1,3})\s?%\s?(off|discount)", re.I)
_CREDIT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:bill |account )?(?:credit|refund)|(?:credit|refund) of \$\s?(\d[\d,]*)", re.I)
_REGULATOR_RE = re.compile(r"\b(regulator|ombudsman|lawyer|legal action|sue|fcc|ofcom|trai)\b", re.I)


def classify_rules(text: str) -> IntentClassification:
    intent, conf = "ambiguous", 0.3
    for name, c, pattern in _RULES:
        if re.search(pattern, text, re.I):
            intent, conf = name, c
            break
    requested = None
    if _HALF_RE.search(text):
        requested = RequestedOffer(offer_type="discount_pct", value=50, months=6)
    elif m := _PCT_RE.search(text):
        requested = RequestedOffer(offer_type="discount_pct", value=float(m.group(1)), months=6)
    elif m := _CREDIT_RE.search(text):
        requested = RequestedOffer(offer_type="credit", value=float((m.group(1) or m.group(2)).replace(",", "")))
    cancel = bool(re.search(_RULES[1][2], text, re.I))
    return IntentClassification(
        intent=intent, confidence=conf, cancellation_intent=cancel, requested_offer=requested,
        mentions_regulator=bool(_REGULATOR_RE.search(text)),
        rationale=f"keyword rules matched '{intent}'" if intent != "ambiguous" else "no intent keywords matched",
    )


_SYSTEM = (
    "You classify telecom customer-care contacts. Classify the customer's CURRENT request (the most "
    "recent turn), using earlier turns only as context. Intents: billing_query (charges, bills, refunds, disputes), plan_change (plan, data allowance, "
    "upgrades), complaint (service problems, dissatisfaction), cancellation (wants to leave/cancel, or "
    "demands a discount to stay), ambiguous (unclear what they need), out_of_scope (not about their "
    "telecom service). Use a confidence below 0.6 when unsure. requested_offer only if the customer "
    "explicitly asks for a specific discount or credit ('half off' = discount_pct 50)."
)


async def intent_agent(state: dict, runtime: Runtime[Deps]) -> dict:
    with agent_scope(NODE):
        ctx = view(NODE, state)  # isolate: recent sanitized turns + running summary only
        text = "\n".join(ctx["recent_turns"])
        engine, detail = "rules", ""
        if runtime.context.use_llm:
            summary = ctx.get("summary")
            untrusted = (f"[summary of earlier turns]\n{summary}\n[recent turns]\n" if summary else "") + text
            try:
                result = await structured_call(
                    IntentClassification,
                    untrusted_prompt(_SYSTEM, untrusted, "Classify the customer's current request."),
                    what="intent.classify")
                engine = "gemini"
            except ExternalCallFailed as exc:
                result, detail = classify_rules(text), f"gemini failed ({exc.reason}); rules fallback"
        else:
            result, detail = classify_rules(text), "no GOOGLE_API_KEY"
        return IntentOutput(
            intent=result.intent,
            intent_confidence=result.confidence,
            intent_rationale=result.rationale,
            cancellation_intent=result.cancellation_intent,
            requested_offer=result.requested_offer.model_dump() if result.requested_offer else None,
            mentions_regulator=result.mentions_regulator,
            engine_log=[engine_entry(NODE, engine, detail or result.rationale)],
        ).update()
