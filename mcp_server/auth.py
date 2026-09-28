"""Session tokens binding an MCP tool call to the authenticated caller.

The client (acting as the auth gateway) signs `session_id:customer_id` with a shared secret
passed to the server subprocess via the MCP_AUTH_SECRET env var. The server verifies the HMAC
and derives the authenticated customer from the token, never from an LLM-supplied argument.
If MCP_AUTH_SECRET is unset, the client generates a random per-process secret.
"""

from __future__ import annotations

import hashlib
import hmac
import os

SECRET_ENV = "MCP_AUTH_SECRET"


class AuthError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(secret: str, session_id: str, customer_id: str) -> str:
    payload = f"{session_id}:{customer_id}"
    return f"{payload}:{_sign(secret, payload)}"


def verify_token(token: str | None, secret: str | None = None) -> tuple[str, str]:
    """Return (session_id, customer_id) for a valid token, else raise AuthError."""
    secret = secret or os.getenv(SECRET_ENV)
    if not secret:
        raise AuthError("AUTH_UNAVAILABLE", "Server has no auth secret configured.")
    if not token:
        raise AuthError("AUTH_MISSING", "No session token supplied with the tool call.")
    try:
        session_id, customer_id, sig = token.rsplit(":", 2)
    except ValueError:
        raise AuthError("AUTH_INVALID", "Malformed session token.") from None
    if not hmac.compare_digest(sig, _sign(secret, f"{session_id}:{customer_id}")):
        raise AuthError("AUTH_INVALID", "Session token signature is invalid.")
    return session_id, customer_id
