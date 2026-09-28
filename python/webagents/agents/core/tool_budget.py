"""
The tool-round budget of one turn (2026-09-28): the TypeScript agent's
`maxToolIterations` (`typescript/src/core/tool-budget.ts`), in this SDK.

The agentic loop calls the model, runs the tools it asks for, and calls it
again, until the model answers or the budget is spent. This SDK hard-coded
five rounds and stopped silently after the fifth, without calling the model
again: asked "nice", the built-in agent explored the folder for five rounds
and the turn ended with no answer, while the chat blamed the provider ("the
provider reported STOP", the finish reason of the model's last tool call).

Both SDKs now share one budget (`max_tool_iterations`, default 50), warn the
model at the same round in the same words (half way through a budget of 20 or
less, so the warning fires before a short cap; 80% of a larger one), and end a
turn that spent the budget with the same reason, `tool_round_limit`, which the
chat says as "The agent stopped after N tool rounds without an answer."
(`cli/repl/failures.py`, `present_empty_reply`). The cases both SDKs run are
in `tests/fixtures/agent_loop/tool_round_budget.json`.

THE CAP IS NEVER A SILENT STOP (the owner, 2026-09-28): a turn that reaches
its budget makes ONE more model call with tools off and a wrap-up message, so
the model answers from what it gathered; the turn's finish reason is still
`tool_round_limit`. A turn that makes the same tool call, arguments and all,
`REPEAT_LIMIT` times stops the same way early, with the reason `tool_loop`.
The budget is `max_tool_rounds` in the agent file, `--max-tool-rounds` on the
command line and `/rounds` in the chat (`parse_max_tool_rounds`).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

#: The budget when the agent names none (TypeScript `AgentConfig.maxToolIterations`).
DEFAULT_MAX_TOOL_ITERATIONS = 50
#: The finish reason of a turn that spent its budget.
TOOL_ROUND_LIMIT = "tool_round_limit"
#: The finish reason of a turn stopped for repeating one tool call.
TOOL_LOOP = "tool_loop"
#: How many identical calls (same tool, same arguments) one turn may make.
REPEAT_LIMIT = 3
#: The bounds of `max_tool_rounds` (the agent file, `--max-tool-rounds`, `/rounds`).
MIN_TOOL_ROUNDS = 1
MAX_TOOL_ROUNDS = 1000
#: What a Yes to the chat's question sends as the next message.
CONTINUE_MESSAGE = "Keep going."


def budget_warning_round(limit: int) -> int:
    """The round at which the model is told to stop calling tools."""
    fraction = 0.5 if limit <= 20 else 0.8
    return max(1, math.ceil(limit * fraction))


def budget_warning(used: int, limit: int) -> str:
    """The system message the model gets at that round, word for word the TypeScript agent's."""
    return (
        f"You have used {used}/{limit} of your tool-call budget for this response. "
        "Stop delegating and calling tools. Summarize what you have done so far and "
        "deliver the final answer to the user now. Do not start new workflows."
    )


def tool_round_limit_sentence(rounds: Optional[int], answered: bool = False) -> str:
    """What a turn that spent its budget says: the chat's headline (fixture
    `cli/chat_fixes_empty_reply.json`) and a non-streaming turn's answer when
    the last call brought none; `answered`, what `-p` adds after an answer."""
    tail = "." if answered else " without an answer."
    if isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 0:
        return f"The agent stopped after {rounds} tool round{'' if rounds == 1 else 's'}{tail}"
    return f"The agent stopped at its tool-round limit{tail}"


def tool_round_limit_finish(rounds: int) -> Dict[str, Any]:
    """`webagents_finish` for a turn that spent its budget: the shape the LLM
    proxy skill sends (`reason`, `blocked`, `retried`), plus how many rounds
    ran. The TypeScript agent sends the same object as its `max_iterations`
    error's `details.finish`."""
    return {"reason": TOOL_ROUND_LIMIT, "blocked": False, "retried": False, "rounds": rounds}


def final_answer_message(limit: int) -> str:
    """The system message of the last, tool-less call of a turn that spent its budget."""
    return (
        f"You have used all {limit} tool rounds for this response, and tools are now off. "
        "Answer the user now from what you have gathered so far: what you found, what is still open, "
        "and what to try next."
    )


def loop_answer_message(tool: str) -> str:
    """The system message of the last, tool-less call of a turn stopped for repeating itself."""
    return (
        f"You called {tool} {REPEAT_LIMIT} times with the same arguments, and tools are now off. "
        "Answer the user now from what you have gathered so far, and say what is still open."
    )


def tool_loop_finish(tool: str, rounds: int) -> Dict[str, Any]:
    """`webagents_finish` for a turn stopped for repeating `tool`."""
    return {"reason": TOOL_LOOP, "blocked": False, "retried": False, "rounds": rounds, "tool": tool}


def tool_loop_sentence(tool: Optional[str]) -> str:
    """What a turn stopped for repeating a tool call says, answer or not."""
    if tool:
        return f"The agent stopped early: it called {tool} {REPEAT_LIMIT} times with the same arguments."
    return "The agent stopped early: it repeated the same tool call."


def continue_question(rounds: int) -> str:
    """What the interactive chat asks after a turn that spent its budget."""
    return f"Used {rounds} tool round{'' if rounds == 1 else 's'}. Keep going? [Y/n]"


def canonical_arguments(arguments: Any) -> str:
    """A tool call's arguments as one string, key order aside: parsed JSON is
    written back with sorted keys, anything else is kept as given."""
    value = arguments
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return arguments
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


class RepeatedCalls:
    """Counts the calls of one turn by tool and arguments. `record` answers
    the tool's name once one call has been made `REPEAT_LIMIT` times."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[str, str], int] = {}

    def record(self, name: str, arguments: Any) -> Optional[str]:
        key = (name, canonical_arguments(arguments))
        self._counts[key] = self._counts.get(key, 0) + 1
        return name if self._counts[key] >= REPEAT_LIMIT else None


def rounds_refusal(name: str, value: Any) -> str:
    """The sentence a bad `max_tool_rounds` is refused with."""
    try:
        shown = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        shown = json.dumps(str(value))
    return f"{name} must be a whole number from {MIN_TOOL_ROUNDS} to {MAX_TOOL_ROUNDS}, not {shown}."


def parse_max_tool_rounds(value: Any, name: str = "max_tool_rounds") -> int:
    """A tool-round budget from the agent file, a flag or `/rounds`: a whole
    number (or its digits) within the bounds, else ValueError with the sentence."""
    number: Optional[int] = None
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    if number is None or not MIN_TOOL_ROUNDS <= number <= MAX_TOOL_ROUNDS:
        raise ValueError(rounds_refusal(name, value))
    return number


class TurnBudget:
    """One turn's tool rounds, the way both agent loops count them (the
    TypeScript twin is `TurnBudget` in `core/tool-budget.ts`).

    `begin_call` runs before every model call. It counts the round, sends the
    budget warning at its round, and decides when the NEXT call is the turn's
    last, with tools off: the budget is spent (`tool_round_limit`), or one
    tool call was made `REPEAT_LIMIT` times (`tool_loop`). Either way the
    wrap-up message goes in first, after every tool result of the round (a
    system message between an assistant's tool calls and their results is
    refused by OpenAI-shaped APIs). `final` is then the turn's finish.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.warn_at = budget_warning_round(limit)
        self.warned = False
        self.rounds = 0
        self.final: Optional[Dict[str, Any]] = None
        self._repeats = RepeatedCalls()
        self._looped: Optional[str] = None

    def begin_call(self, messages: list) -> bool:
        """True when this model call is the turn's last, with tools off."""
        if self.final is None and self._looped is not None:
            self.final = tool_loop_finish(self._looped, self.rounds)
            messages.append({"role": "system", "content": loop_answer_message(self._looped)})
        elif self.final is None and self.rounds >= self.limit:
            self.final = tool_round_limit_finish(self.rounds)
            messages.append({"role": "system", "content": final_answer_message(self.limit)})
        if self.final is not None:
            return True
        self.rounds += 1
        if not self.warned and self.rounds >= self.warn_at:
            self.warned = True
            messages.append({"role": "system", "content": budget_warning(self.rounds, self.limit)})
        return False

    def record_call(self, name: str, arguments: Any) -> None:
        """After the model asked for a tool: the first call to reach the repeat limit is remembered."""
        repeated = self._repeats.record(name, arguments)
        if repeated is not None and self._looped is None:
            self._looped = repeated


#: Where `--max-tool-rounds` puts its value for the agent builders (and a
#: child process): the flag is validated once, at the root.
MAX_TOOL_ROUNDS_ENV = "WEBAGENTS_MAX_TOOL_ROUNDS"

#: What the chat says about the budget (`/rounds`, `/status`); the fixture's
#: `rounds_words`, the TypeScript chat's `ROUNDS_WORDS`.
ROUNDS_WORDS: Dict[str, str] = {
    "show": "Tool rounds: {rounds} per turn ({source}).",
    "showHint": "Change with /rounds <n>; add --save to keep it in the agent file.",
    "set": "Tool rounds set to {rounds} per turn, for this chat.",
    "setHint": "/rounds {rounds} --save keeps it in {file}.",
    "kept": "Tool rounds {rounds} kept in {file}.",
    "already": "{file} already keeps max_tool_rounds: {rounds}.",
    "status": "{rounds} per turn ({source})",
    "session": "set in this chat",
    "flag": "--max-tool-rounds",
    "file": "{file}",
    "default": "the default",
}


def rounds_source_words(source: str, file: Optional[str] = None) -> str:
    """Where the budget came from, in the chat's words."""
    return ROUNDS_WORDS.get(source, source).format(file=file or "the agent file")


def effective_max_tool_rounds(file_value: Optional[int], environ: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """The budget an agent built here runs with, and where it came from, most
    specific first: `--max-tool-rounds` (its environment value), the agent
    file's `max_tool_rounds`, the default. The chat's `/rounds` sits above
    all three. A caller of a served agent has no say: no request field reaches it."""
    import os

    env = (environ if environ is not None else os.environ).get(MAX_TOOL_ROUNDS_ENV)
    if env is not None and env.strip():
        return parse_max_tool_rounds(env, "--max-tool-rounds"), "flag"
    if file_value is not None:
        return file_value, "file"
    return DEFAULT_MAX_TOOL_ITERATIONS, "default"
