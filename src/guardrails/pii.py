"""PII masking: `mask()` / `mask_obj()` are the ONLY way customer data reaches logs.

Formats follow POL-PRV-001 §2.1: IDs and account numbers keep the last 3 characters, phone
numbers the last 2 digits, emails the first character of the local part. Payment card numbers
and credentials are fully redacted. For logs, money amounts tied to a person are masked too.

Regex rules cover the synthetic identifier formats deterministically; Presidio (`use_presidio=True`,
used by the output guard) adds custom recognizers for CUST-/ACC-/synthetic phone formats plus its
built-in email, phone, card, IBAN and SSN recognizers.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

CUSTOMER_ID_RE = re.compile(r"\bCUST-(\d{3})(\d{3})\b")
ACCOUNT_RE = re.compile(r"\bACC-(\d{5})(\d{3})\b")
EMAIL_RE = re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# +CC-555-0142 / +1 555 014 2222 style, or US (555) 123-4567 / 555-123-4567. Never ISO dates (4-2-2).
PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+\d{1,3}[ -]?\d{3}[ -]?\d{3,4}(?:[ -]?\d{4})?|\(?\d{3}\)?[ -]\d{3}[ -]\d{4})\b"
)
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")

REDACT_KEYS = {"auth", "token", "session_token", "secret", "api_key", "password", "authorization"}
MONEY_KEYS = {
    "amount", "subtotal", "tax", "total", "monthly_bill", "balance", "offer_value_usd",
    "credit_usd", "amount_usd", "etf_usd", "total_billed",
}
NAME_KEYS = {"first_name", "last_name", "name_on_account", "full_name"}
# `value` in an offer is money only for these offer types (a percent / GB count is not PII).
MONEY_OFFER_TYPES = {"credit", "fee_waiver"}

MASKED_AMOUNT = "<masked:amount>"


def _mask_text(text: str, *, amounts: bool) -> str:
    text = CARD_RE.sub("[CARD-REDACTED]", text)
    text = CUSTOMER_ID_RE.sub(lambda m: f"CUST-***{m.group(2)}", text)
    text = ACCOUNT_RE.sub(lambda m: f"ACC-*****{m.group(2)}", text)
    text = EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)
    text = PHONE_RE.sub(lambda m: f"***-***-**{m.group(0)[-2:]}", text)
    if amounts:
        text = MONEY_RE.sub("$**.**", text)
    return text


PRESIDIO_ENTITIES = ["CUSTOMER_ID", "ACCOUNT_NUMBER", "SYNTH_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER",
                     "CREDIT_CARD", "IBAN_CODE", "US_SSN"]


@lru_cache(maxsize=1)
def presidio_analyzer():
    """Presidio AnalyzerEngine (small spaCy model) + custom recognizers for the synthetic formats."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="CUSTOMER_ID", patterns=[Pattern("customer_id", r"\bCUST-\d{6}\b", 0.95)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="ACCOUNT_NUMBER", patterns=[Pattern("account_number", r"\bACC-\d{8}\b", 0.95)]))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="SYNTH_PHONE", patterns=[Pattern("synthetic_phone", r"\+1-555-01\d{2}\b", 0.9)]))
    return engine


def detect_pii(text: str, min_score: float = 0.5) -> list[dict]:
    """Presidio findings (entity, span, score) for identifiers in free text."""
    if not text:
        return []
    results = presidio_analyzer().analyze(text=text, language="en", entities=PRESIDIO_ENTITIES)
    return [{"entity": r.entity_type, "start": r.start, "end": r.end, "score": round(r.score, 2),
             "text": text[r.start:r.end]} for r in results if r.score >= min_score]


def _mask_presidio(text: str) -> str:
    """Mask every Presidio finding with the same formats as the regex masker."""
    for f in sorted(detect_pii(text), key=lambda f: f["start"], reverse=True):
        span = f["text"]
        masked = _mask_text(span, amounts=False)
        if masked == span:  # entity the regexes do not know: generic placeholder
            masked = f"<{f['entity']}>"
        text = text[:f["start"]] + masked + text[f["end"]:]
    return text


def mask(text: str, *, amounts: bool = True, use_presidio: bool = False) -> str:
    """Mask identifiers (and, by default, money amounts) in free text."""
    if not text:
        return text
    if use_presidio:
        text = _mask_presidio(text)
    return _mask_text(text, amounts=amounts)


def mask_obj(obj: Any, *, amounts: bool = True, use_presidio: bool = False, _parent: dict | None = None) -> Any:
    """Recursively mask a JSON-like structure for logging."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            k = str(key).lower()
            if k in REDACT_KEYS:
                out[key] = "<redacted>" if value else value
            elif amounts and value is not None and (
                k in MONEY_KEYS or (k == "value" and obj.get("offer_type") in MONEY_OFFER_TYPES)
            ):
                out[key] = MASKED_AMOUNT
            elif k in NAME_KEYS and isinstance(value, str) and value:
                out[key] = value[0] + "***"
            else:
                out[key] = mask_obj(value, amounts=amounts, use_presidio=use_presidio, _parent=obj)
        return out
    if isinstance(obj, (list, tuple)):
        return [mask_obj(v, amounts=amounts, use_presidio=use_presidio) for v in obj]
    if isinstance(obj, str):
        return mask(obj, amounts=amounts, use_presidio=use_presidio)
    return obj


def mask_customer_id(customer_id: str) -> str:
    return _mask_text(customer_id, amounts=False)
