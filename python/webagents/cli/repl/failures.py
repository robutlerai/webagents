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

def provider_key_for(model: Optional[str]) -> str:
    """The key variable the hints name for `model` (2026-09-27): its provider's
    first, when this SDK has a client for it; OPENAI_API_KEY otherwise, the
    chat's historical default. Every hint said OPENAI_API_KEY, on a Google
    model too. `auto/balanced` names no provider, so it keeps the default."""
    from webagents.agents.skills.core.llm.providers import provider_for_model

    provider = provider_for_model(model)
    if provider is not None and provider.credential == "api_key" and provider.env_vars:
        return provider.env_vars[0]
    return "OPENAI_API_KEY"


def sign_in_hint(model: Optional[str] = None) -> str:
    """A function, not a constant: the command names the active profile (`cli_command`)."""
    return f"Use a provider key: {cli_command(f'secrets set {provider_key_for(model)}')}."


def credits_hint(model: Optional[str] = None) -> str:
    return f"Add credits in Robutler, or use a provider key of your own ({cli_command(f'secrets set {provider_key_for(model)}')})."
EXPIRED_HINT = "Your Robutler sign-in may have expired. Type /login to sign in again."


def for_prompt(hint: Optional[str]) -> Optional[str]:
    """A hint as `-p` says it (2026-09-28, the e2e pass): `/login` is a chat
    command, so `-p` names the command a shell runs instead. The TypeScript
    `forPrompt` says the same (fixture `cli/final_sdk_low_items.json`)."""
    if hint == EXPIRED_HINT:
        return f"Your Robutler sign-in may have expired. Run `{cli_command('login')}` to sign in again."
    return hint
UNREACHABLE_HINT = "Check the network, or platform.url."
#: What to do about a reply with nothing in it: the same line in both chats.
EMPTY_REPLY_HINT = "/model <provider/model> tries another model."


def present_empty_reply(
    reason: Optional[str],
    *,
    blocked: bool = False,
    retried: bool = False,
    thinking: bool = False,
    rounds: Optional[int] = None,
    tool: Optional[str] = None,
) -> "FailureText":
    """What the chat says about a turn that produced no answer (2026-09-27).

    The agent used to answer an empty completion with a canned apology about
    "content filtering" and add it to the conversation, whatever had happened
    (the real reason, MALFORMED_FUNCTION_CALL, was in the platform's log
    alone), and a reply that was only thinking said nothing at all. Both chats
    now print one truthful line: the provider's own finish reason, that the
    prompt was blocked, that the request was sent twice, or that the model only
    produced thinking. The cases both SDKs run are in
    `tests/fixtures/cli/chat_fixes_empty_reply.json`; the TypeScript twin is
    `cli/failures.ts` (`presentEmptyReply`).

    `tool_round_limit` and `tool_loop` are the AGENT's reasons, not the
    provider's (2026-09-28, `core/tool_budget.py`): the turn spent its tool
    rounds (`rounds` of them), or called `tool` three times with the same
    arguments, and its last, tool-less call brought no answer. The chat used
    to name the finish reason of the model's last tool call instead ("the
    provider reported STOP"), which blamed the provider for the agent's cap.
    """
    from webagents.agents.core.tool_budget import TOOL_LOOP, TOOL_ROUND_LIMIT, tool_loop_sentence, tool_round_limit_sentence

    if reason == TOOL_ROUND_LIMIT:
        return FailureText(tool_round_limit_sentence(rounds), EMPTY_REPLY_HINT)
    if reason == TOOL_LOOP:
        return FailureText(tool_loop_sentence(tool), EMPTY_REPLY_HINT)
    twice = ", twice" if retried else ""
    if blocked:
        headline = f"The model returned no answer: the provider blocked the prompt{f' ({reason})' if reason else ''}."
    elif thinking:
        headline = (
            "The model returned no answer: it only produced thinking"
            f"{f' (the provider reported {reason}{twice})' if reason else ''}."
        )
    elif reason:
        headline = f"The model returned no answer: the provider reported {reason}{twice}."
    else:
        headline = "The model returned no answer, and the provider gave no reason."
    return FailureText(headline, EMPTY_REPLY_HINT)

#: The socket's close code the TypeScript proxy skill appends to the platform's reason.
_CLOSE_NOTE = re.compile(r"\s*\[WebSocket closed unexpectedly \(code=\d+\)\]\s*$")
_UNREACHABLE = re.compile(
    r"could not reach|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|fetch failed|websocket error|connection timeout"
    r"|connection refused|connect call failed|timed out|nodename nor servname|name or service not known",
    re.I,
)


#: Robutler's own credit refusals that say more than "not enough credits"
#: (2026-09-27, the portal's `cliCreditsRefusal`): what the call needs
#: reserved, and what unfinished calls hold and when it returns. The chat used
#: to print "Not enough credits to start a model call." over every one of
#: them, beside a balance that was plainly there. Shown as Robutler wrote
#: them: the first sentence is the headline, the rest the hint.
_SPECIFIC_CREDITS = re.compile(r"^Not enough credits (?:for this model call|to start a model call right now)\b")
_HELD_CREDITS = "held by unfinished model calls"
#: The first sentence, and the rest: a full stop followed by space or the end
#: (so the decimal point in "0.358 credits" does not end one).
_FIRST_SENTENCE = re.compile(r"^(.*?\.)(?=\s|$)\s*(.*)$", re.S)


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
    model: Optional[str] = None,
) -> FailureText:
    """The headline, hint and code for a failed turn. `proxy_url` is the
    Robutler socket when the turn ran on Robutler's models; `generic_hint` is
    the chat's advice for a provider's own errors; `model` is the model the
    turn ran on, so a key hint names its provider's variable."""
    text = (message or "").strip() or "The model returned an error."
    if proxy_url is not None:
        # The platform's own provider key was refused (a 401, 402 or 429 from
        # the maker, `providerErrorForClient` in the portal, 2026-09-27): not
        # the person's problem, so the way out is another model.
        if "unavailable right now" in text.lower():
            return FailureText("This model is unavailable right now.", EMPTY_REPLY_HINT, "provider_unavailable")
        # A Robutler from before CLI sign-ins asks for a payment token alone; a
        # later one also names the Bearer token `webagents login` stores.
        if ("X-Payment-Token" in text and "Bearer" not in text) or "does not run models for a CLI sign-in" in text:
            return FailureText(f"Robutler at {proxy_url} does not run models for a CLI sign-in.", sign_in_hint(model), "sign_in_not_supported")
        said = _CLOSE_NOTE.sub("", text).strip()
        if _SPECIFIC_CREDITS.match(said):
            sentence = _FIRST_SENTENCE.match(said)
            headline, rest = (sentence.group(1), sentence.group(2).strip()) if sentence else (said, "")
            if _HELD_CREDITS in said:
                return FailureText(headline, rest or None, "credits_held")
            return FailureText(headline, credits_hint(model), "insufficient_credits")
        if re.search(r"not enough credits|insufficient credits|payment_required|\b4002\b", text, re.I):
            return FailureText("Not enough credits to start a model call.", credits_hint(model), "insufficient_credits")
        if re.search(r"payment token|bearer token|unauthori[sz]ed|\b4001\b|\b401\b|expired", text, re.I):
            return FailureText("Robutler did not accept your sign-in.", EXPIRED_HINT, "sign_in_refused")
        if _UNREACHABLE.search(text):
            return FailureText(f"Could not reach Robutler at {proxy_url}.", UNREACHABLE_HINT, "unreachable")
    headline = _CLOSE_NOTE.sub("", text).splitlines()[0].strip() or "The model returned an error."
    return FailureText(headline, generic_hint(text) if generic_hint else None)
