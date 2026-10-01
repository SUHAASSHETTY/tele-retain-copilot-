"""Client for the existing copilot API (src/api/app.py). The Streamlit UI never runs agent logic itself.

By default the unchanged FastAPI app is started once, in-process, on a free local port (its lifespan
opens the LangGraph graph, the MCP session, memory and the Gemini preflight exactly as `uvicorn`
would). Set COPILOT_API_URL to use an API that is already running instead.
COPILOT_NO_LLM=1 (handled in app.py) forces the deterministic rules/templates engine, like the CLI's --no-llm.

Every error is converted to BackendError: a short message for the screen plus masked technical
details for the "Technical details" expander. Secrets never appear in either.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from typing import Iterator

import httpx
import streamlit as st

from src.guardrails.pii import mask

STARTUP_TIMEOUT_S = 240
STREAM_TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class BackendError(Exception):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = mask(detail, amounts=False)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@st.cache_resource(show_spinner="Starting the copilot backend (agents, MCP tools, policy index)…")
def base_url() -> str:
    """Start the existing API once per Streamlit process and return its URL."""
    external = os.getenv("COPILOT_API_URL")
    if external:
        return external.rstrip("/")
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config("src.api.app:app", host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="copilot-api", daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server.started:
        if not thread.is_alive():
            raise BackendError("The copilot backend could not start.",
                               "The API process exited during startup; run `uvicorn src.api.app:app` to see why.")
        if time.monotonic() > deadline:
            raise BackendError("The copilot backend is taking too long to start.",
                               f"Not ready after {STARTUP_TIMEOUT_S}s.")
        time.sleep(0.2)
    return f"http://127.0.0.1:{port}"


def _explain(exc: Exception) -> BackendError:
    if isinstance(exc, BackendError):
        return exc
    if isinstance(exc, httpx.ConnectError):
        return BackendError("The copilot service is not reachable. Please restart the app and try again.", repr(exc))
    if isinstance(exc, httpx.TimeoutException):
        return BackendError("The analysis took too long to complete. Please try again.", repr(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        try:
            detail = exc.response.json().get("detail")
        except ValueError:
            detail = exc.response.text[:300]
        friendly = {401: "The selected customer could not be verified.",
                    403: "This case belongs to another customer.",
                    404: "No matching customer or case was found.",
                    409: "There is no approval waiting for this case any more.",
                    422: "The request was not valid. Please check the message and try again."}
        return BackendError(friendly.get(code, "The copilot service returned an error. Please try again."),
                            f"HTTP {code}: {detail}")
    return BackendError("Unable to complete the analysis. Please check the configuration and try again.",
                        f"{type(exc).__name__}: {exc}")


def _get(path: str, headers: dict | None = None) -> dict | list:
    try:
        r = httpx.get(base_url() + path, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a clean on-screen message
        raise _explain(exc) from exc


def health() -> dict:
    return _get("/health")


@st.cache_data(show_spinner=False)
def samples() -> list[dict]:
    return _get("/v1/samples")


def account(customer_id: str) -> dict:
    """Masked account summary through the MCP `get_account` tool (the same authorization as the agents)."""
    return _get("/v1/account", headers={"X-Customer-Id": customer_id})


def _events(method: str, path: str, **kwargs) -> Iterator[tuple[str, dict]]:
    try:
        with httpx.stream(method, base_url() + path, timeout=STREAM_TIMEOUT, **kwargs) as r:
            if r.status_code >= 400:
                r.read()
                r.raise_for_status()
            event = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    yield event or "message", json.loads(line[6:])
    except Exception as exc:  # noqa: BLE001
        raise _explain(exc) from exc


def analyze(customer_id: str, message: str) -> Iterator[tuple[str, dict]]:
    """Run one real copilot turn. Yields SSE events: start, node*, then approval_required or resolution."""
    session_id = f"ST-{uuid.uuid4().hex[:10].upper()}"
    yield from _events("POST", "/v1/contacts/stream", headers={"X-Customer-Id": customer_id},
                       json={"message": message, "session_id": session_id})


def decide(session_id: str, approved: bool, approver: str) -> Iterator[tuple[str, dict]]:
    """Resume a turn paused at the human-approval gate with the team lead's decision."""
    yield from _events("POST", f"/v1/sessions/{session_id}/approval",
                       json={"approved": approved, "approver": approver})
