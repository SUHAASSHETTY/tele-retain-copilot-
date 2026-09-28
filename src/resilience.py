"""Timeouts, retries with backoff, and graceful fallback for every external call.

`resilient_call` wraps an async call in asyncio.wait_for and tenacity retries (exponential backoff
with jitter) for transient failures: timeouts, HTTP 429 / RESOURCE_EXHAUSTED, 5xx / UNAVAILABLE and
connection errors. Non-transient errors fail fast. Callers catch `ExternalCallFailed` and degrade
(e.g. heuristic fallback or escalate to a human with the reason) instead of crashing.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import settings

T = TypeVar("T")

_TRANSIENT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota", "503", "unavailable",
                      "500", "internal", "deadline", "timed out", "timeout", "connection reset",
                      "clientconnector", "dns", "gaierror", "servname", "connection refused",
                      "server disconnected", "remoteprotocolerror")


class ExternalCallFailed(Exception):
    """An external call failed after retries; `reason` is safe to show in logs."""

    def __init__(self, what: str, reason: str, attempts: int):
        super().__init__(f"{what} failed after {attempts} attempt(s): {reason}")
        self.what = what
        self.reason = reason
        self.attempts = attempts


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


async def resilient_call(
    fn: Callable[[], Awaitable[T]],
    *,
    what: str,
    timeout_s: float | None = None,
    attempts: int | None = None,
    max_wait_s: float = 20.0,
) -> T:
    """Run `fn()` with a per-attempt timeout and retries on transient errors."""
    timeout_s = timeout_s or settings.llm_timeout_s
    attempts = attempts or settings.max_retries
    tried = 0
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=max_wait_s),
            retry=retry_if_exception(is_transient),
            reraise=True,
        ):
            with attempt:
                tried += 1
                return await asyncio.wait_for(fn(), timeout=timeout_s)
    except RetryError as exc:  # pragma: no cover - reraise=True makes this rare
        raise ExternalCallFailed(what, repr(exc.last_attempt.exception()), tried) from exc
    except Exception as exc:
        reason = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else type(exc).__name__
        raise ExternalCallFailed(what, reason, tried) from exc
    raise ExternalCallFailed(what, "no attempt made", tried)  # pragma: no cover
