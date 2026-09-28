"""Input guardrail, wired into the graph's entry node.

Policy functions (no external service):
- prompt-injection / role-override detection (src.context.quarantine) -> block (POL-RET-004 §2.2)
- requests for another customer's data                  -> block (POL-PRV-001 §1.1)
- payment card numbers and other identifiers in the text -> redacted before any model sees them
- length: > HARD_MAX_CHARS blocked, > MAX_INPUT_CHARS truncated (flagged)
- toxicity: threats of violence / self-harm -> block + escalate to a human (P1);
            abusive language -> flagged, customer still served
- quarantine of the remaining text as untrusted data     (src.context.quarantine)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.context.quarantine import detect_injection, quarantine
from src.guardrails.pii import CARD_RE, CUSTOMER_ID_RE, mask

CROSS_CUSTOMER_PATTERNS = [
    r"\b(another|other|different) customer'?s?\b", r"\bsomeone else'?s (account|bill|data)",
    r"\b(my )?(neighbou?r|friend|ex|wife|husband|partner|boss)'s (account|bill|data|number|plan)",
    r"\b(calling |asking )?(for|on behalf of) my (wife|husband|partner|mother|mum|mom|father|dad|son|daughter|"
    r"friend|neighbou?r|boss|colleague)\b.{0,60}\b(her|his|their) (account|bill|number|plan|data|details)",
]
_XC = [re.compile(p, re.I) for p in CROSS_CUSTOMER_PATTERNS]

MAX_INPUT_CHARS = 2000
HARD_MAX_CHARS = 6000
THREAT_PATTERNS = [
    r"\b(kill|hurt|harm|attack|shoot|stab|beat up)\b.{0,40}\b(you|your staff|staff|agents?|employees|them|people)\b",
    r"\bfind (out )?where (you|your staff|they|your agents?) live\b", r"\bbomb\b",
    r"\b(kill|hurt|harm) myself\b", r"\bend (it all|my life)\b",
]
ABUSE_PATTERNS = [r"\b(idiot|stupid|useless|moron|incompetent|pathetic)s?\b", r"\bf+u+c+k\w*", r"\bsh[i1]t\w*",
                  r"\bcrap\b", r"\bbastard\w*"]
_THREAT = [re.compile(p, re.I) for p in THREAT_PATTERNS]
_ABUSE = [re.compile(p, re.I) for p in ABUSE_PATTERNS]


@dataclass
class InputGuardResult:
    sanitized: str
    quarantined: str
    flags: list[str] = field(default_factory=list)
    blocked: bool = False
    reason: str | None = None
    policy_ref: str | None = None


def check_input(text: str, authenticated_customer_id: str) -> InputGuardResult:
    flags: list[str] = []
    text = text or ""
    if len(text) > HARD_MAX_CHARS:
        return InputGuardResult(sanitized=mask(text[:200], amounts=False) + " ...", quarantined=quarantine(""),
                                flags=["input_too_long"], blocked=True, reason="input_too_long",
                                policy_ref="POL-GOV-001 §2.2")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        flags.append("input_truncated")
    if any(p.search(text) for p in _THREAT):
        flags.append("safety_threat")
    elif any(p.search(text) for p in _ABUSE):
        flags.append("abusive_language")
    if CARD_RE.search(text):
        flags.append("payment_card_redacted")
    other_ids = {m.group(0) for m in CUSTOMER_ID_RE.finditer(text)} - {authenticated_customer_id}
    if other_ids or any(p.search(text) for p in _XC):
        flags.append("cross_customer_request")
    injection = detect_injection(text)
    if injection:
        flags.append("prompt_injection")
        flags.extend(f"injection:{name}" for name in injection)

    # identifiers masked before any model sees the text; amounts kept (the customer's own words)
    sanitized = mask(text, amounts=False)
    result = InputGuardResult(sanitized=sanitized, quarantined=quarantine(sanitized), flags=flags)
    if "safety_threat" in flags:
        result.blocked, result.reason, result.policy_ref = True, "safety_threat", "POL-GOV-001 §2.2"
    elif "prompt_injection" in flags:
        result.blocked, result.reason, result.policy_ref = True, "prompt_injection", "POL-RET-004 §2.2"
    elif "cross_customer_request" in flags:
        result.blocked, result.reason, result.policy_ref = True, "cross_customer_request", "POL-PRV-001 §1.1"
    return result
