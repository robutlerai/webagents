"""
Automatic context compaction (gap-closure plan items 2.1 and 2.4, 2026-09-26):
past a token threshold, the older turns of the working conversation become one
summary written by the agent's own model, and the last ``keep`` turns stay
verbatim. The summary also goes into the caller's episodic memory
(``caller_scoped.py``), so a compacted conversation is still searchable later.

Pure functions here, so both agents can be pinned by the same cases
(``tests/fixtures/memory_tool/definition.json``, ``compaction``); the TypeScript
twin is ``typescript/src/skills/memory/compaction.ts``.

WHAT IS COUNTED. The estimate is ``ceil(characters / 4)`` over a message's
text and its tool calls' names and arguments, plus 4 per message. It is a
budget guard, not a tokenizer: the point is to compact well before the model's
window, and the same estimate in both SDKs.

WHERE THE CUT GOES. The first system message (the agent's prompt) is never
summarized. Of the rest, the last ``keep`` messages are kept, and the cut moves
earlier while it would land on a tool result, so a tool call and its results
are never split. An earlier summary sits among the older turns and is rolled
into the new one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

COMPACTION_PREFIX = "Summary of the earlier part of this conversation, written by you when it grew long:\n"

COMPACTION_INSTRUCTIONS = (
    "Summarize the conversation below for your own later use: what was asked, what was decided, what is "
    "still open, and any names, numbers or preferences worth keeping. Write plain prose, under 300 words, "
    "and nothing else."
)

DEFAULT_COMPACTION_THRESHOLD = 60000
DEFAULT_COMPACTION_KEEP = 12

Summarizer = Callable[[str, str], Awaitable[str]]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _tool_calls_of(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """The estimate of one message (module docstring)."""
    chars = len(_text_of(message.get("content")))
    for call in _tool_calls_of(message):
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") if isinstance(function.get("name"), str) else ""
        args = function.get("arguments") if isinstance(function.get("arguments"), str) else ""
        chars += len(name) + len(args)
    return math.ceil(chars / 4) + 4


def estimate_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def transcript_of(messages: Sequence[Dict[str, Any]]) -> str:
    """The older turns as the summarizing model reads them: one line per message, one per tool call."""
    lines: List[str] = []
    for m in messages:
        text = _text_of(m.get("content"))
        if text:
            lines.append(f"{m.get('role')}: {text}")
        for call in _tool_calls_of(m):
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") if isinstance(function.get("name"), str) else "tool"
            args = function.get("arguments") if isinstance(function.get("arguments"), str) else ""
            lines.append(f"{m.get('role')} called {name}({args})")
    return "\n".join(lines)


@dataclass
class CompactionPlan:
    head: Optional[Dict[str, Any]]
    older: List[Dict[str, Any]]
    recent: List[Dict[str, Any]]


def plan_compaction(messages: Sequence[Dict[str, Any]], threshold: int, keep: int) -> Optional[CompactionPlan]:
    """Where the cut goes, or None when there is nothing to summarize (module docstring)."""
    if not messages or estimate_tokens(messages) <= threshold:
        return None
    head = messages[0] if messages[0].get("role") == "system" else None
    body = list(messages[1:] if head else messages)
    keep = max(0, int(keep))
    if len(body) <= keep:
        return None
    cut = len(body) - keep
    while cut > 0 and body[cut].get("role") == "tool":
        cut -= 1
    if cut <= 0:
        return None
    return CompactionPlan(head=head, older=body[:cut], recent=body[cut:])


@dataclass
class CompactionResult:
    compacted: bool
    messages: List[Dict[str, Any]]
    summary: Optional[str] = None
    summarized: int = 0
    _: Any = field(default=None, repr=False)


async def compact_conversation(
    messages: Sequence[Dict[str, Any]],
    threshold: int,
    keep: int,
    summarize: Summarizer,
) -> CompactionResult:
    """The conversation after compaction: the head, one system message carrying
    the summary, then the recent turns. ``summarize`` is the agent's model (or
    a stub under test); an empty summary leaves the conversation as it was."""
    plan = plan_compaction(messages, threshold, keep)
    if plan is None:
        return CompactionResult(compacted=False, messages=list(messages))
    summary = (await summarize(transcript_of(plan.older), COMPACTION_INSTRUCTIONS)).strip()
    if not summary:
        return CompactionResult(compacted=False, messages=list(messages))
    out: List[Dict[str, Any]] = []
    if plan.head is not None:
        out.append(plan.head)
    out.append({"role": "system", "content": f"{COMPACTION_PREFIX}{summary}"})
    out.extend(plan.recent)
    return CompactionResult(compacted=True, messages=out, summary=summary, summarized=len(plan.older))
