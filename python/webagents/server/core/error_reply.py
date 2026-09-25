"""
What a caller is told when a run fails, and what the log keeps (2026-09-24, S-228).

THE LEAK. A failed run answered the caller with the exception's own text:
`Response(500, str(e))` from a skill's `@http` route (the one daemon- and
deployed-agent `/chat/completions` takes, via `CompletionsTransportSkill`),
`{"error": str(e)}` in the dedicated `/{agent}/chat/completions` stream,
`HTTPException(500, str(e))` from a static agent's mounted handlers, the same
text as a WebSocket close reason. What that text holds is whatever the failing
code put in it: a provider's error body (OpenAI's 401 quotes a masked form of
the rejected key), a tool's file paths on the agent's host, library wording.
Anyone who could make a served agent's run fail could read it. Measured on a
daemon whose model was unreachable: `500 Connection error.`.

THE REPLY NOW is the portal's S-028 fix: the full error, traceback included,
goes to this server's log under a short reference, and the caller gets a fixed
sentence that carries the same reference, so an operator can find the one
from the other. Status codes and body shapes are unchanged; only the text is.

WHAT KEEPS ITS TEXT:

  * errors raised in order to be shown: this SDK's own auth refusals, its
    payment and commerce errors (which carry an HTTP status and, mostly, a
    `to_dict()`), and FastAPI's `HTTPException`. The class's MODULE is checked,
    not only its name: `openai.AuthenticationError` has our name, and its text
    is exactly what must not travel.
  * anything at all, on a server started with `error_detail=True`. Only the
    local daemon asks for that, and only while bound to loopback
    (`webagents daemon start`, `dev`): its answers go to the developer's own
    terminal, where "Connection error." is the message the chat exists to show
    (`cli/client/daemon_client.py:daemon_error_detail`). Loopback is enough
    because S-218 keeps the daemon there unless `--host` says otherwise, and
    S-224's origin rule keeps other sites' pages from reading its answers. The
    decision is never taken from the caller's address: behind a proxy or a
    sidecar every caller arrives from 127.0.0.1.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

#: What a caller reads when a run fails, ahead of the reference.
INTERNAL_ERROR_MESSAGE = "The agent could not complete this request."

#: This SDK's refusals, whose messages are written for the caller.
_SHOWN_NAMES = frozenset({"AuthenticationError", "AuthorizationError", "AuthError"})

_LOG = logging.getLogger("webagents.server.errors")


def new_reference() -> str:
    """Eight hex characters: short enough to read out, unique enough to grep for."""
    return secrets.token_hex(4)


def is_meant_to_be_shown(error: BaseException) -> bool:
    """Whether `error` was raised to be shown to the caller (module docstring)."""
    try:
        from starlette.exceptions import HTTPException
    except ImportError:  # pragma: no cover - starlette ships with the server
        HTTPException = None  # type: ignore[assignment]
    if HTTPException is not None and isinstance(error, HTTPException):
        return True
    module = type(error).__module__ or ""
    if module != "webagents" and not module.startswith("webagents."):
        return False
    if type(error).__name__ in _SHOWN_NAMES:
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 600


def reply_text(
    error: BaseException,
    *,
    detail: bool = False,
    where: str = "request",
    logger: Optional[logging.Logger] = None,
    prefix: str = "",
) -> str:
    """What the caller reads for a failed run.

    `prefix` + the error's own text when it is meant to be shown or `detail`
    is on (so the local daemon's wording stays exactly what it was);
    otherwise `INTERNAL_ERROR_MESSAGE` and a fresh reference. Every failure
    that is not meant to be shown is logged with its traceback under that
    reference, whichever of the two the caller gets, so the log always has
    the whole story.
    """
    if is_meant_to_be_shown(error):
        return f"{prefix}{error}"
    reference = new_reference()
    (logger or _LOG).error(
        "%s failed [ref %s]: %s: %s", where, reference, type(error).__name__, error, exc_info=error
    )
    if detail:
        return f"{prefix}{error}"
    return f"{INTERNAL_ERROR_MESSAGE} Reference: {reference}"
