"""
Compaction of a conversation that is filling the model's context (2026-09-29,
the owner: "what's the best strategy for auto-compaction? ... logic/settings
with good defaults and command and api surface too?").

WHY HERE. Compaction lived in the `memory` skill (a `before_llm_call` hook), so
an agent without `memory` never compacted and a long chat ran until the
provider refused it; and the hook compacted the run's copy of the history
while the chat kept and re-sent the whole of it, so past the threshold every
turn paid for a new summary and saved another episode. It is now the agent's:
the chat compacts its own history between turns (and keeps the result), a run
compacts only as a safety stop inside one long turn, and `Agent.compact()` is
there for any other host. The memory skill keeps the summary as an episode
(`on_compaction`), once per compaction.

THE POLICY (`CompactionPolicy`, the agent file's `compaction:` block):

  * `auto` (true): compact without being asked.
  * `at` (0.8): when the conversation reaches this part of the model's
    context window (a fraction), or this many tokens (a number above 1).
  * `keep` (0.25): the recent part kept as it is, a fraction of the window or
    tokens; the cut never separates a tool call from its results.
  * `hard` (0.95): inside one long turn, the point a run compacts at by
    itself, leaving the turn in progress alone.
  * `clear_tool_results` (true): first clear the output of earlier tool
    calls, which costs no model call and keeps the thread of the conversation.
  * `model`: the model that writes the summary (default: the agent's own);
    `instructions`: what the summary should also keep; `window`: the context
    window, for a model the table below does not know.

THE STRATEGY, cheapest first. Nothing happens under `at`. Past it: (1) the
output of the earlier tool calls is cleared (a one-line note names the tool
and the size); if that brings the conversation under three quarters of `at`,
that is all. (2) Otherwise the earlier part becomes one summary written by the
model (what was asked, decided, done, what is open, the names and numbers),
an earlier summary rolled into it. (3) If no summary can be made, the oldest
whole turns are dropped until it fits, and a note says how many. `/compact`
(the chat) forces step 2. The system prompt is never touched.

WHAT IS COUNTED: `ceil(characters / 4)` per message's text and tool calls,
plus 4, the same in both SDKs. A budget guard, not a tokenizer.

The TypeScript twin is `typescript/src/core/context-compaction.ts`; both are
pinned by `tests/fixtures/context/compaction.json`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

WORDS: Dict[str, str] = {
    "compacted": "Compacted the conversation: {summarized} earlier messages became a summary, the last {kept} stay as they were. Context {percent}% full.",
    "cleared": "Made room by clearing the output of {cleared} earlier tool calls. Context {percent}% full.",
    "dropped": "Could not summarize the conversation ({reason}); dropped its {dropped} oldest messages instead. Context {percent}% full.",
    "nothing": "Nothing to compact yet: the conversation is {percent}% of the context.",
    "stub": "[output cleared to make room: {tool}, {chars} characters]",
    "summaryPrefix": "Summary of the earlier part of this conversation, written when it grew long:\n",
    "droppedNote": "[{dropped} earlier messages were dropped to make room]",
    "instructions": (
        "Summarize the conversation below so that it can be continued from the summary alone. Keep: what the person "
        "asked for; what was decided, and why; what was done (files, commands, results); where things stand and what is "
        "still open; and the names, numbers, paths and preferences worth keeping. Plain prose or short lists, under 400 "
        "words, and nothing else."
    ),
    "focus": "Pay particular attention to: {focus}",
    "emptySummary": "the model answered with nothing",
}

#: The context windows compaction measures against, by model prefix, first
#: match wins. A claim to check when the providers' catalogs change; a model
#: not listed gets `DEFAULT_WINDOW`, and `compaction.window` overrides both.
CONTEXT_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("openai/gpt-4.1", 1047576),
    ("openai/gpt-5", 400000),
    ("openai/o1", 200000),
    ("openai/o3", 200000),
    ("openai/o4", 200000),
    ("openai/", 128000),
    ("anthropic/", 200000),
    ("google/", 1048576),
    ("xai/grok-4-fast", 2000000),
    ("xai/grok-4", 256000),
    ("xai/", 131072),
    ("fireworks/", 131072),
    ("ollama/", 32768),
)
DEFAULT_WINDOW = 128000

#: Set on a run's context while it writes a compaction summary (the
#: TypeScript agent summarizes in a run of its own), so nothing in it
#: compacts again and the memory notes stay out of it.
COMPACTION_RUN = "context_compaction"

#: A tool output shorter than this is left alone by the clearing step.
CLEAR_MIN_CHARS = 400
#: Where compaction aims to land, as a part of `at`, so it does not run again on the next turn.
SETTLE = 0.75

POLICY_KEYS = ("auto", "at", "keep", "hard", "clear_tool_results", "model", "instructions", "window")
POLICY_WORDS: Dict[str, str] = {
    "notMapping": "compaction must be a mapping of auto, at, keep, hard, clear_tool_results, model, instructions and window.",
    "unknownKey": 'compaction: unknown key "{key}". It takes auto, at, keep, hard, clear_tool_results, model, instructions and window.',
    "notBool": "compaction.{key} must be true or false.",
    "notAmount": "compaction.{key} must be a part of the context between 0 and 1, or a number of tokens.",
    "notText": "compaction.{key} must be text.",
    "notWindow": "compaction.window must be a number of tokens.",
    "atBelowHard": "compaction.at must be below compaction.hard.",
}


@dataclass(frozen=True)
class CompactionPolicy:
    auto: bool = True
    at: float = 0.8
    keep: float = 0.25
    hard: float = 0.95
    clear_tool_results: bool = True
    model: Optional[str] = None
    instructions: Optional[str] = None
    window: Optional[int] = None


class CompactionPolicyError(ValueError):
    """A `compaction:` block that cannot be read; the message says why."""


def parse_policy(raw: Any) -> CompactionPolicy:
    """The agent file's `compaction:` block as a policy; None or absent is every default."""
    if raw is None:
        return CompactionPolicy()
    if not isinstance(raw, dict):
        raise CompactionPolicyError(POLICY_WORDS["notMapping"])
    values: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in POLICY_KEYS:
            raise CompactionPolicyError(POLICY_WORDS["unknownKey"].replace("{key}", str(key)))
        if key in ("auto", "clear_tool_results"):
            if not isinstance(value, bool):
                raise CompactionPolicyError(POLICY_WORDS["notBool"].replace("{key}", key))
        elif key in ("at", "keep", "hard"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 or (value > 1 and value != int(value)):
                raise CompactionPolicyError(POLICY_WORDS["notAmount"].replace("{key}", key))
            value = float(value)
        elif key in ("model", "instructions"):
            if not isinstance(value, str) or not value.strip():
                raise CompactionPolicyError(POLICY_WORDS["notText"].replace("{key}", key))
            value = value.strip()
        elif key == "window":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CompactionPolicyError(POLICY_WORDS["notWindow"])
        values[key] = value
    policy = CompactionPolicy(**values)
    at, hard = policy.at, policy.hard
    if (at <= 1) == (hard <= 1) and at >= hard:
        raise CompactionPolicyError(POLICY_WORDS["atBelowHard"])
    return policy


def context_window(model: Optional[str], override: Optional[int] = None) -> int:
    """The context window of `provider/model` (`CONTEXT_WINDOWS`), or `override`."""
    if override:
        return int(override)
    name = (model or "").lower()
    return next((size for prefix, size in CONTEXT_WINDOWS if name.startswith(prefix)), DEFAULT_WINDOW)


def tokens_of(amount: float, window: int) -> int:
    """A policy amount in tokens: a fraction of the window, or tokens already."""
    return int(amount) if amount > 1 else int(amount * window)


# -- counting ---------------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _tool_calls_of(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def _call_parts(call: Dict[str, Any]) -> Tuple[str, str]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") if isinstance(function.get("name"), str) else ""
    args = function.get("arguments") if isinstance(function.get("arguments"), str) else ""
    return name, args


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """`ceil(characters / 4) + 4` for one message (module docstring)."""
    chars = len(_text_of(message.get("content")))
    for call in _tool_calls_of(message):
        name, args = _call_parts(call)
        chars += len(name) + len(args)
    return math.ceil(chars / 4) + 4


def estimate_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def percent_of(tokens: int, window: int) -> int:
    return min(999, round(100 * tokens / window)) if window else 0


def transcript_of(messages: Sequence[Dict[str, Any]]) -> str:
    """The earlier turns as the summarizing model reads them: one line per message, one per tool call."""
    lines: List[str] = []
    for m in messages:
        text = _text_of(m.get("content"))
        if text:
            lines.append(f"{m.get('role')}: {text}")
        for call in _tool_calls_of(m):
            name, args = _call_parts(call)
            lines.append(f"{m.get('role')} called {name or 'tool'}({args})")
    return "\n".join(lines)


# -- the steps --------------------------------------------------------------------------------


def _split_head(messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The agent's system prompt (the first message, when it is one) and the rest."""
    if messages and messages[0].get("role") == "system" and not _is_compaction_note(messages[0]):
        return [messages[0]], list(messages[1:])
    return [], list(messages)


def _is_compaction_note(message: Dict[str, Any]) -> bool:
    text = _text_of(message.get("content"))
    return message.get("role") == "system" and text.startswith(WORDS["summaryPrefix"])


def recent_cut(body: Sequence[Dict[str, Any]], keep_tokens: int, protect_from: Optional[int] = None) -> int:
    """Where the recent part starts: walking back from the end while it fits
    in `keep_tokens` (the last message always kept), never after `protect_from`
    (the turn in progress), and never on a tool result, so a call and its
    results stay together."""
    if not body:
        return 0
    cut = len(body) - 1
    used = estimate_message_tokens(body[cut])
    while cut > 0 and used + estimate_message_tokens(body[cut - 1]) <= keep_tokens:
        cut -= 1
        used += estimate_message_tokens(body[cut])
    if protect_from is not None:
        cut = min(cut, max(0, protect_from))
    while cut > 0 and body[cut].get("role") == "tool":
        cut -= 1
    return cut


def clear_tool_results(messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """`messages` with the output of each tool call of `CLEAR_MIN_CHARS` or more
    replaced by a one-line note naming the tool and the size, and how many."""
    names: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    cleared = 0
    for m in messages:
        for call in _tool_calls_of(m):
            if isinstance(call.get("id"), str):
                names[call["id"]] = _call_parts(call)[0] or "tool"
        text = _text_of(m.get("content"))
        if m.get("role") == "tool" and len(text) >= CLEAR_MIN_CHARS and not text.startswith("[output cleared"):
            copy_ = dict(m)
            tool = names.get(str(m.get("tool_call_id")), m.get("name") if isinstance(m.get("name"), str) else "tool")
            copy_["content"] = WORDS["stub"].replace("{tool}", tool).replace("{chars}", str(len(text)))
            out.append(copy_)
            cleared += 1
        else:
            out.append(m)
    return out, cleared


def drop_oldest_turns(older: Sequence[Dict[str, Any]], fits: Callable[[List[Dict[str, Any]]], bool]) -> Tuple[List[Dict[str, Any]], int]:
    """`older` without its oldest whole turns (a turn starts at a person's
    message), dropped until `fits` says the rest fits; and how many went."""
    rest = list(older)
    dropped = 0
    while rest and not fits(rest):
        end = next((i for i in range(1, len(rest)) if rest[i].get("role") == "user"), len(rest))
        dropped += end
        rest = rest[end:]
    return rest, dropped


Summarizer = Callable[[str, str], Awaitable[str]]


@dataclass
class Compaction:
    """What compaction did: `stage` is none, cleared, summarized or dropped."""

    stage: str
    messages: List[Dict[str, Any]]
    before: int
    after: int
    window: int
    summary: Optional[str] = None
    summarized: int = 0
    kept: int = 0
    cleared: int = 0
    dropped: int = 0
    reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.stage != "none"

    def sentence(self) -> str:
        """The chat's one line for it (`WORDS`)."""
        percent = percent_of(self.after, self.window)
        if self.stage == "summarized":
            return WORDS["compacted"].replace("{summarized}", str(self.summarized)).replace("{kept}", str(self.kept)).replace("{percent}", str(percent))
        if self.stage == "cleared":
            return WORDS["cleared"].replace("{cleared}", str(self.cleared)).replace("{percent}", str(percent))
        if self.stage == "dropped":
            return (
                WORDS["dropped"].replace("{reason}", self.reason or WORDS["emptySummary"]).replace("{dropped}", str(self.dropped)).replace("{percent}", str(percent))
            )
        return WORDS["nothing"].replace("{percent}", str(percent_of(self.before, self.window)))


def summary_instructions(policy: CompactionPolicy, focus: Optional[str] = None) -> str:
    parts = [WORDS["instructions"]]
    if policy.instructions:
        parts.append(policy.instructions)
    if focus and focus.strip():
        parts.append(WORDS["focus"].replace("{focus}", focus.strip()))
    return "\n".join(parts)


async def compact_messages(
    messages: Sequence[Dict[str, Any]],
    policy: CompactionPolicy,
    window: int,
    summarize: Summarizer,
    *,
    force: bool = False,
    focus: Optional[str] = None,
    protect_from: Optional[int] = None,
    threshold: Optional[int] = None,
) -> Compaction:
    """The conversation after compaction (module docstring). `force` is
    `/compact`: it summarizes whatever the size. `protect_from` is the index,
    in `messages`, where the turn in progress starts: nothing from there on is
    touched. `threshold` defaults to the policy's `at`."""
    before = estimate_tokens(messages)
    limit = threshold if threshold is not None else tokens_of(policy.at, window)
    none = Compaction("none", list(messages), before, before, window)
    if not force and before <= limit:
        return none
    settle = int(tokens_of(policy.at, window) * SETTLE)
    head, body = _split_head(messages)
    protect = None if protect_from is None else protect_from - len(head)
    last_ask = turn_start(body)
    if force and last_ask:
        # `/compact`: everything before the latest exchange becomes the summary,
        # whatever the `keep` budget would have kept.
        cut = last_ask if protect is None else min(last_ask, max(0, protect))
    else:
        cut = recent_cut(body, tokens_of(policy.keep, window), protect)
    older, recent = body[:cut], body[cut:]
    if not older:
        return none
    cleared_older, cleared = clear_tool_results(older) if policy.clear_tool_results else (list(older), 0)
    if cleared and not force:
        candidate = head + cleared_older + recent
        after = estimate_tokens(candidate)
        if after <= settle:
            return Compaction("cleared", candidate, before, after, window, cleared=cleared, kept=len(recent))
    reason: Optional[str] = None
    try:
        summary = (await summarize(transcript_of(cleared_older), summary_instructions(policy, focus))).strip()
    except Exception as error:  # noqa: BLE001 - the fallback below says it
        summary, reason = "", str(error) or error.__class__.__name__
    if summary:
        out = head + [{"role": "system", "content": WORDS["summaryPrefix"] + summary}] + recent
        return Compaction("summarized", out, before, estimate_tokens(out), window, summary=summary, summarized=len(older), kept=len(recent), cleared=cleared)

    def fits(rest: List[Dict[str, Any]]) -> bool:
        return estimate_tokens(head + rest + recent) + 12 <= settle

    rest, dropped = drop_oldest_turns(cleared_older, fits)
    if not dropped:
        if not cleared:
            return Compaction("none", list(messages), before, before, window, reason=reason or WORDS["emptySummary"])
        out = head + cleared_older + recent
        return Compaction("cleared", out, before, estimate_tokens(out), window, cleared=cleared, kept=len(recent), reason=reason or WORDS["emptySummary"])
    note = [{"role": "system", "content": WORDS["droppedNote"].replace("{dropped}", str(dropped))}]
    out = head + note + rest + recent
    return Compaction("dropped", out, before, estimate_tokens(out), window, dropped=dropped, cleared=cleared, kept=len(recent), reason=reason or WORDS["emptySummary"])


def turn_start(messages: Sequence[Dict[str, Any]]) -> Optional[int]:
    """The index of the last message from the person: where the turn in progress starts."""
    return next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)


def copy_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(list(messages))
