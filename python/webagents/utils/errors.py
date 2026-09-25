"""
One line that says what failed, and where (2026-09-24).

The OpenAI and Anthropic clients raise ``APIConnectionError("Connection
error.")`` and keep the reason (a refused connection, an unknown host, a
certificate) in ``__cause__`` and the request in ``.request``. A mistyped
``OPENAI_BASE_URL`` therefore reached the person as "Error: Connection
error." and nothing else, found walking a first run. The TypeScript SDK had
the same gap ("fetch failed") and got the same fix (``src/skills/llm/request.ts``).

Only the ORIGIN of the request is named: a path or a query can carry a key on
some gateways, and this text is shown to people and written to logs.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit


def _origin(url: object) -> str:
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return ""
    if not parts.scheme or not parts.hostname:
        return ""
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{parts.scheme}://{host}{f':{port}' if port else ''}"


def _request_url(error: BaseException) -> Optional[object]:
    try:
        request = getattr(error, "request", None)
        return getattr(request, "url", None) if request is not None else None
    except Exception:  # a client's `.request` can raise when it was never set
        return None


def describe_exception(error: BaseException) -> str:
    """The error's message; for a request that got no answer, the server and the reason."""
    message = str(error).strip() or type(error).__name__
    cause = error.__cause__
    reason = str(cause).strip() if cause is not None else ""
    url = _request_url(error)
    origin = _origin(url) if url is not None else ""
    # A request that never got an answer: name where it was going. One that
    # did (an HTTP status error) already says what the server answered.
    if origin and getattr(error, "response", None) is None:
        return f"Could not reach {origin}: {reason or message}"
    if reason and reason not in message:
        return f"{message} ({reason})"
    return message
