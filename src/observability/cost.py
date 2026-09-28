"""Gemini price table and token-to-cost estimates for golden signals.

Prices: USD per 1M tokens, paid tier, standard (non-batch), text input; output includes thinking tokens.
Source: https://ai.google.dev/gemini-api/docs/pricing (page "last updated 2026-09-24 UTC"),
retrieved 2026-09-28. The 3.6/3.7/3.8-flash prices below apply through 2026-12-31
($1.50 / $7.50 from 2027-01-01). Unknown models fall back to GEMINI_INPUT/OUTPUT_PRICE_PER_MTOK.
The free tier used during development is not billed; these figures estimate paid-tier cost.
"""

from __future__ import annotations

from src.config import settings

PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICE_PAGE_UPDATED = "2026-09-24"
PRICE_RETRIEVED = "2026-09-28"

PRICE_TABLE: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


def price_for(model: str | None) -> tuple[dict[str, float], str]:
    """(prices, basis) for a model name such as 'gemini-3.8-flash' or 'models/gemini-3.8-flash-001'."""
    name = (model or "").removeprefix("models/")
    for key in sorted(PRICE_TABLE, key=len, reverse=True):
        if name.startswith(key):
            return PRICE_TABLE[key], f"price_table:{key}"
    return ({"input": settings.gemini_input_price_per_mtok, "output": settings.gemini_output_price_per_mtok},
            "env_fallback")


def token_cost(prompt_tokens: int, completion_tokens: int, model: str | None) -> float:
    p, _ = price_for(model)
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000
