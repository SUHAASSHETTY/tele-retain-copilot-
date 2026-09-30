"""Gemini chat-model factory (langchain-google-genai), the application's only LLM provider.

Retries are handled by src.resilience (tenacity), so the client's own retries are disabled to
avoid multiplying attempts. Structured output uses Pydantic schemas.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.config import require_google_api_key, settings
from src.resilience import ExternalCallFailed, resilient_call

S = TypeVar("S", bound=BaseModel)
logging.getLogger("google_genai").setLevel(logging.ERROR)       # SDK advisory warnings are not user-facing
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# One limiter shared by every Gemini client in the process (agents, RAG judge, memory, eval judge).
RATE_LIMITER = InMemoryRateLimiter(requests_per_second=settings.gemini_rpm / 60.0,
                                   check_every_n_seconds=0.1, max_bucket_size=1)


def llm_available() -> bool:
    return bool(settings.google_api_key)


@lru_cache(maxsize=8)
def get_chat_model(model: str | None = None, temperature: float | None = None,
                   thinking_budget: int | None = None) -> ChatGoogleGenerativeAI:
    kwargs = {}
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        google_api_key=require_google_api_key(),
        temperature=settings.gemini_temperature if temperature is None else temperature,
        timeout=settings.llm_timeout_s,
        max_retries=0,
        rate_limiter=RATE_LIMITER,
        **kwargs,
    )


async def structured_call(schema: type[S], messages: list[BaseMessage], *, what: str,
                          model: BaseChatModel | None = None) -> S:
    """Invoke the model with structured output, a timeout and retries."""
    chat = model or get_chat_model()
    runnable = chat.with_structured_output(schema)
    result = await resilient_call(lambda: runnable.ainvoke(messages), what=what)
    if isinstance(result, dict):  # some providers return dicts for json-schema output
        result = schema.model_validate(result)
    return result


async def llm_preflight() -> tuple[bool, str]:
    """One cheap call to confirm the configured model is usable before a batch run."""
    if not llm_available():
        return False, "GOOGLE_API_KEY not set"
    try:
        await resilient_call(lambda: get_chat_model().ainvoke("Reply with: ok"), what="llm.preflight",
                             attempts=3, timeout_s=30)
        return True, f"{settings.gemini_model} reachable"
    except ExternalCallFailed as exc:
        return False, f"{settings.gemini_model} unusable ({exc.reason})"
