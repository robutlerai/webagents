"""
What the chat keeps of a turn's tool calls, and how much of it goes back to
the model (2026-09-28).

WHY. The chat kept only the person's message and the final answer of each
turn; the tool calls and what they returned were dropped. On the next message
the model had no record of the folder it had listed or the files it had read,
and a tool-eager model (Gemini 3.5 Flash, behind ``auto/balanced``) listed and
read them again, every message: three messages of small talk cost 21k, 27k and
45k tokens. The chat now keeps each turn's tool rounds between the message and
the answer, the way the agent loop itself sends them.

THE BUDGET. Kept results add up, so what goes back to the model keeps the
newest results whole up to ``TOOL_HISTORY_BUDGET_CHARS`` and replaces older
ones with a one-line note that says what was left out; the call itself stays,
so the model still knows what it did and can call again. The conversation on
disk keeps everything; only the copy sent is trimmed.

ONE BEHAVIOUR IN BOTH SDKS: ``typescript/src/cli/turn-history.ts``, and the
cases in ``tests/fixtures/chat/turn_history.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

#: Characters of tool results the model gets back whole, newest first.
TOOL_HISTORY_BUDGET_CHARS = 40_000


def left_out_result(chars: int) -> str:
    """What a result past the budget becomes (agent-facing)."""
    return f"[An earlier tool result was left out to save space ({chars} characters). Call the tool again if you need it.]"


class TurnRecorder:
    """Collects one turn's tool calls and results from its stream, in rounds: a
    call that arrives after a result of the current round starts the next
    round. A call that never got a result (the turn was stopped) is not kept,
    so every kept call has its result, as the providers require."""

    def __init__(self) -> None:
        self._rounds: List[Dict[str, Any]] = []

    def observe(self, event: Any) -> None:
        """Takes the chat's render events (``render.ToolCall``, ``render.ToolResult``)."""
        kind = type(event).__name__
        if kind == "ToolCall" and getattr(event, "id", ""):
            if not self._rounds or self._rounds[-1]["results"]:
                self._rounds.append({"calls": [], "results": {}})
            args = event.arguments if isinstance(event.arguments, str) else json.dumps(event.arguments or {})
            self._rounds[-1]["calls"].append({"id": event.id, "name": event.name, "arguments": args})
        elif kind == "ToolResult" and getattr(event, "id", ""):
            for round_ in self._rounds:
                if any(c["id"] == event.id for c in round_["calls"]):
                    round_["results"][event.id] = str(event.result if event.result is not None else "")
                    break

    def messages(self) -> List[Dict[str, Any]]:
        """The rounds as the agent loop sends them: an assistant message with the
        calls, then one tool message per result."""
        out: List[Dict[str, Any]] = []
        for round_ in self._rounds:
            answered = [c for c in round_["calls"] if c["id"] in round_["results"]]
            if not answered:
                continue
            out.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in answered
                    ],
                }
            )
            for c in answered:
                out.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": round_["results"][c["id"]]})
        return out


def history_for_model(messages: List[Dict[str, Any]], budget: int = TOOL_HISTORY_BUDGET_CHARS) -> List[Dict[str, Any]]:
    """The copy of the conversation sent to the model: tool results whole,
    newest first, until ``budget`` characters; older ones replaced by
    ``left_out_result``."""
    left = budget
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if not isinstance(m, dict) or m.get("role") != "tool" or not isinstance(m.get("content"), str):
            continue
        size = len(m["content"])
        if size <= left:
            left -= size
            continue
        left = 0
        out[i] = {**m, "content": left_out_result(size)}
    return out


def spoken_count(messages: List[Dict[str, Any]]) -> int:
    """The person's and the agent's words in a conversation: what "N messages" counts."""
    return sum(
        1
        for m in messages
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    )
