"""
What a failed turn says, in the chat and in `-p` (2026-09-25).

A headline in plain words and one line of what to do. Until now each CLI
printed the same refusal its own way: asked for a model by a CLI sign-in, a
Robutler that does not fund one answered, and the TypeScript chat headed the
error with the platform's protocol text ("session.create requires
X-Payment-Token in session.extensions [WebSocket closed unexpectedly
(code=4001)]") while this one printed its own sentence, the original cause in
brackets after it, and then the same sentence again as the hint.

One function in each SDK now decides both lines, and both run the same cases
(`tests/fixtures/cli/failure_presentation.json`); the TypeScript twin is
`typescript/src/cli/failures.ts`. The SDKs still raise their own words (the
TypeScript proxy skill keeps the socket's close code in its message, which the
portal reads), so the rules match either wording.
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple, Optional

from ..config_store import cli_command

def sign_in_hint() -> str:
    """A function, not a constant: the command names the active profile (`cli_command`)."""
    return f"Use a provider key: {cli_command('secrets set OPENAI_API_KEY')}."


def credits_hint() -> str:
    return f"Add credits in Robutler, or use a provider key of your own ({cli_command('secrets set OPENAI_API_KEY')})."
EXPIRED_HINT = "Your Robutler sign-in may have expired. Type /login to sign in again."
UNREACHABLE_HINT = "Check the network, or platform.url."

#: The socket's close code the TypeScript proxy skill appends to the platform's reason.
_CLOSE_NOTE = re.compile(r"\s*\[WebSocket closed unexpectedly \(code=\d+\)\]\s*$")
_UNREACHABLE = re.compile(
    r"could not reach|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|fetch failed|websocket error|connection timeout"
    r"|connection refused|connect call failed|timed out|nodename nor servname|name or service not known",
    re.I,
)


class FailureText(NamedTuple):
    headline: str
    hint: Optional[str] = None
    #: A stable code for Robutler's own refusals, which `-p` reports.
    code: Optional[str] = None


def present_failure(
    message: str,
    *,
    proxy_url: Optional[str] = None,
    generic_hint: Optional[Callable[[str], Optional[str]]] = None,
) -> FailureText:
    """The headline, hint and code for a failed turn. `proxy_url` is the
    Robutler socket when the turn ran on Robutler's models; `generic_hint` is
    the chat's advice for a provider's own errors."""
    text = (message or "").strip() or "The model returned an error."
    if proxy_url is not None:
        # A Robutler from before CLI sign-ins asks for a payment token alone; a
        # later one also names the Bearer token `webagents login` stores.
        if ("X-Payment-Token" in text and "Bearer" not in text) or "does not run models for a CLI sign-in" in text:
            return FailureText(f"Robutler at {proxy_url} does not run models for a CLI sign-in.", sign_in_hint(), "sign_in_not_supported")
        if re.search(r"not enough credits|insufficient credits|payment_required|\b4002\b", text, re.I):
            return FailureText("Not enough credits to start a model call.", credits_hint(), "insufficient_credits")
        if re.search(r"payment token|bearer token|unauthori[sz]ed|\b4001\b|\b401\b|expired", text, re.I):
            return FailureText("Robutler did not accept your sign-in.", EXPIRED_HINT, "sign_in_refused")
        if _UNREACHABLE.search(text):
            return FailureText(f"Could not reach Robutler at {proxy_url}.", UNREACHABLE_HINT, "unreachable")
    headline = _CLOSE_NOTE.sub("", text).splitlines()[0].strip() or "The model returned an error."
    return FailureText(headline, generic_hint(text) if generic_hint else None)
