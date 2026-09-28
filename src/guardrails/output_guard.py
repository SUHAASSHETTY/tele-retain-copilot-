"""Output guardrail, wired into the graph's LAST node. It blocks or sanitizes; it never just logs.

Checks on every drafted customer message before release, in order:
1. another customer's identifier in the answer                  -> BLOCK: replaced by a refusal
2. a discount % above the eligibility-checked value             -> BLOCK: replaced by the safe template
3. a credit $ amount above the eligibility-checked value        -> BLOCK: replaced by the safe template
4. an offer awaiting approval described as applied              -> BLOCK: replaced by the safe template
5. PII leakage: Presidio (custom CUST-/ACC-/phone recognizers +
   email/phone/card/IBAN/SSN) and regex masking                  -> SANITIZE: masked in place
6. cited clause IDs must exist in the policy corpus             -> SANITIZE: unknown citations stripped
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import POLICY_CORPUS_DIR
from src.guardrails.pii import CUSTOMER_ID_RE, detect_pii, mask

CITATION_RE = re.compile(r"POL-[A-Z]{3}-\d{3} §\d+\.\d+")
PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
CREDIT_RE = re.compile(r"credit(?: of)?\s+\$\s?(\d[\d,]*(?:\.\d+)?)|\$\s?(\d[\d,]*(?:\.\d+)?)\s+(?:bill |account )?credit", re.I)
APPLIED_RE = re.compile(r"\b(has been|have been|is now|I've|I have) (applied|approved|added|granted)\b", re.I)
_REFUSAL_RE = re.compile(r"\b(can't|cannot|can not|unable|not able|not possible|exceeds?|above|over|"
                         r"more than|prohibited|not permitted|tax(es)?|asked for|requested)\b", re.I)


@lru_cache(maxsize=1)
def known_citations() -> frozenset[str]:
    text = "".join(p.read_text() for p in POLICY_CORPUS_DIR.glob("*.md"))
    return frozenset(re.findall(r"^### (POL-[A-Z]{3}-\d{3} §\d+\.\d+)", text, re.M))


@dataclass
class OutputGuardResult:
    text: str
    flags: list[str] = field(default_factory=list)
    replaced: bool = False
    policy_ref: str | None = None
    pii_findings: list[dict] = field(default_factory=list)


def _refusal_context(text: str, pos: int) -> bool:
    """A figure is fine when quoted while refusing it ('we can't offer 90%') or as the tax rate."""
    clause = re.split(r"[.;,:!?]|\bbut\b|\binstead\b", text[max(0, pos - 60):pos])[-1]
    return bool(_REFUSAL_RE.search(clause))


def check_output(text: str, *, authenticated_customer_id: str, max_discount_pct: float | None,
                 pending_approval: bool, safe_fallback: str, max_credit_usd: float | None = None,
                 use_presidio: bool = True) -> OutputGuardResult:
    other_ids = {m.group(0) for m in CUSTOMER_ID_RE.finditer(text)} - {authenticated_customer_id}
    if other_ids:
        return OutputGuardResult(
            "I can only discuss the account of the verified account holder [POL-PRV-001 §1.1].",
            ["other_customer_identifier"], True, "POL-PRV-001 §1.1")

    flags: list[str] = []
    pct_cap = max_discount_pct or 0.0
    if any(float(m.group(1)) > pct_cap + 1e-9 and not _refusal_context(text, m.start())
           for m in PERCENT_RE.finditer(text)):
        flags.append("discount_above_checked_value")
    credit_cap = max_credit_usd or 0.0
    for m in CREDIT_RE.finditer(text):
        amount = float((m.group(1) or m.group(2)).replace(",", ""))
        if amount > credit_cap + 0.005 and not _refusal_context(text, m.start()):
            flags.append("credit_above_checked_value")
            break
    if pending_approval and APPLIED_RE.search(text):
        flags.append("pending_offer_described_as_applied")
    if flags:
        ref = "POL-RET-003 §1.2" if "pending_offer_described_as_applied" in flags else "POL-RET-002 §2.3"
        return OutputGuardResult(mask(safe_fallback, amounts=False), flags, True, ref)

    unknown = [c for c in CITATION_RE.findall(text) if c not in known_citations()]
    for c in unknown:
        text = text.replace(f"[{c}]", "").replace(c, "")
    if unknown:
        flags.append("unknown_citation_removed")

    findings = detect_pii(text) if use_presidio else []
    masked = mask(text, amounts=False, use_presidio=use_presidio)
    if findings or masked != text:
        flags.append("pii_masked")
    return OutputGuardResult(masked, flags, False, "POL-PRV-001 §2.1" if "pii_masked" in flags else None,
                             [{"entity": f["entity"], "score": f["score"]} for f in findings])
