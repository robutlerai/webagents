"""
The tool-round budget of one turn (2026-09-28), against the shared fixture
`tests/fixtures/agent_loop/tool_round_budget.json`, which the TypeScript suite
reads too (`tests/unit/core/tool-round-budget.test.ts`).

The owner asked the built-in agent "nice". The model explored the folder for
five rounds, the agent's hard-coded cap, and the turn ended with no answer and
no further model call, while the chat said "The model returned no answer: the
provider reported STOP" (the finish reason of the model's last tool call).
Pinned here:

  * the budget is the agent's (`max_tool_iterations`, default 50, the
    TypeScript `maxToolIterations`), and the model is warned at the same round
    in the same words as the TypeScript agent's;
  * a model that keeps asking for tools stops at the cap, and the turn's last
    chunk carries the agent's reason, `tool_round_limit`, which the chat says
    as "The agent stopped after N tool rounds without an answer.";
  * a model that answers in the LAST round answered: no such reason;
  * a non-streaming turn answers the same sentence, not an apology.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.core.tool_budget import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    REPEAT_LIMIT,
    TOOL_ROUND_LIMIT,
    budget_warning,
    budget_warning_round,
    canonical_arguments,
    continue_question,
    final_answer_message,
    loop_answer_message,
    parse_max_tool_rounds,
    tool_loop_sentence,
)
from webagents.agents.skills.base import Handoff, Skill
from webagents.agents.tools.decorators import tool
from webagents.cli.repl.failures import present_empty_reply
from webagents.cli.repl.render import Finish, events_from_chunk

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "agent_loop" / "tool_round_budget.json").read_text())
SCENARIO = FIXTURE["scenario"]
#: What the proxy skill says about every model call, a tool call included:
#: Gemini finishes a function call with STOP (the owner's session).
PROVIDER_STOP = {"reason": "STOP", "blocked": False, "retried": False}


class KeepsAskingForTools(Skill):
    """A model that asks for `list_directory` whenever it has tools (a new
    path each call, or the same one with `same_args`), answers at call
    `answer_at`, and answers `SCENARIO["answer"]` when it has no tools."""

    def __init__(self, streaming: bool, answer_at: Optional[int] = None, same_args: bool = False) -> None:
        super().__init__({})
        self.streaming = streaming
        self.answer_at = answer_at
        self.same_args = same_args
        self.calls = 0
        self.seen: List[List[Dict[str, Any]]] = []
        self.tools_offered: List[bool] = []

    async def initialize(self, agent):
        self.agent = agent
        function = self.chat_completion_stream if self.streaming else self.chat_completion
        agent.register_handoff(
            Handoff(target="stub_model", description="stub", scope="all", metadata={"function": function, "priority": 10}),
            source="stub",
        )

    def _reply(self, tools) -> Optional[str]:
        if not tools:
            return SCENARIO["answer"]
        if self.answer_at is not None and self.calls >= self.answer_at:
            return "done"
        return None

    def _call(self) -> Dict[str, Any]:
        path = "." if self.same_args else f"dir-{self.calls}"
        return {"id": f"call-{self.calls}", "type": "function", "function": {"name": "list_directory", "arguments": json.dumps({"path": path})}}

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        self.tools_offered.append(bool(tools))
        reply = self._reply(tools)
        if reply is not None:
            message = {"role": "assistant", "content": reply}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [self._call()]}
        return {"id": "c", "created": 1, "model": "stub", "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    async def chat_completion_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        self.tools_offered.append(bool(tools))
        base = {"id": "c", "created": 1, "model": "stub", "object": "chat.completion.chunk"}
        reply = self._reply(tools)
        if reply is not None:
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
        else:
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, **self._call()}]}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "webagents_finish": dict(PROVIDER_STOP)}


class FolderTools(Skill):
    def __init__(self) -> None:
        super().__init__({})

    async def initialize(self, agent):
        self.agent = agent
        self.register_tool(self.list_directory)

    @tool(name="list_directory", description="Lists the folder")
    async def list_directory(self, path: str = ".") -> str:
        return "README.md\nsrc"


def _agent(model: KeepsAskingForTools, limit: Optional[int] = SCENARIO["limit"]) -> BaseAgent:
    return BaseAgent(name="budget", instructions="", skills={"model": model, "tools": FolderTools()}, max_tool_iterations=limit)


def _stream(agent: BaseAgent) -> List[Dict[str, Any]]:
    async def go():
        return [chunk async for chunk in agent.run_streaming([{"role": "user", "content": "nice"}])]

    return asyncio.run(go())


def _warnings(messages: List[Dict[str, Any]]) -> List[str]:
    return [m["content"] for m in messages if m.get("role") == "system" and "tool-call budget" in str(m.get("content"))]


# ---------------------------------------------------------------------------
# The budget both SDKs share
# ---------------------------------------------------------------------------


def test_the_default_the_reason_and_the_words_are_the_fixtures():
    assert DEFAULT_MAX_TOOL_ITERATIONS == FIXTURE["default_limit"]
    assert TOOL_ROUND_LIMIT == FIXTURE["finish_reason"]
    assert budget_warning(7, 9) == FIXTURE["warning"].format(used=7, limit=9)
    assert "—" not in FIXTURE["warning"]


def test_the_limits_the_refusals_and_the_question_are_the_fixtures():
    for case in FIXTURE["refusals"]:
        with pytest.raises(ValueError) as refused:
            parse_max_tool_rounds(case["value"], case["name"])
        assert str(refused.value) == case["says"]
    for case in FIXTURE["accepted"]:
        assert parse_max_tool_rounds(case["value"]) == case["rounds"]
    assert continue_question(7) == FIXTURE["continue_question"].format(rounds=7)
    for case in FIXTURE["continue_questions"]:
        assert continue_question(case["rounds"]) == case["asks"]


def test_the_warning_round_is_the_fixtures():
    for row in FIXTURE["warn_at"]:
        assert budget_warning_round(row["limit"]) == row["round"], row


def test_the_agent_takes_a_budget_and_defaults_to_fifty():
    assert BaseAgent(name="plain").max_tool_iterations == FIXTURE["default_limit"]
    assert BaseAgent(name="short", max_tool_iterations=3).max_tool_iterations == 3


# ---------------------------------------------------------------------------
# A model that keeps asking for tools
# ---------------------------------------------------------------------------


def _text(chunks: List[Dict[str, Any]]) -> str:
    return "".join(
        ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or "" for chunk in chunks if chunk.get("choices")
    )


def test_a_model_that_keeps_asking_answers_at_the_cap_with_tools_off():
    model = KeepsAskingForTools(streaming=True)
    chunks = _stream(_agent(model))

    # The rounds of the budget, then ONE more call with tools off.
    assert model.calls == SCENARIO["model_calls"]
    assert model.tools_offered == [True] * SCENARIO["rounds"] + [False]
    # The warning arrives before the call the fixture names, once, in the
    # TypeScript agent's words; the last call gets the wrap-up message too.
    warned_at = SCENARIO["warned_at_call"]
    for index, seen in enumerate(model.seen, start=1):
        expected = [budget_warning(warned_at, SCENARIO["limit"])] if index >= warned_at else []
        assert _warnings(seen) == expected, index
    assert model.seen[warned_at - 1][-1] == {"role": "system", "content": budget_warning(warned_at, SCENARIO["limit"])}
    final_message = FIXTURE["final_answer"].format(limit=SCENARIO["limit"])
    assert model.seen[SCENARIO["final_call"] - 1][-1] == {"role": "system", "content": final_message}
    assert final_answer_message(SCENARIO["limit"]) == final_message

    # The answer the last call gave, then the agent's reason on the last chunk.
    assert _text(chunks).strip() == SCENARIO["answer"]
    assert chunks[-1]["webagents_finish"] == {"reason": TOOL_ROUND_LIMIT, "blocked": False, "retried": False, "rounds": SCENARIO["rounds"]}
    finishes = [event for chunk in chunks for event in events_from_chunk(chunk) if isinstance(event, Finish)]
    assert finishes[-1] == Finish(TOOL_ROUND_LIMIT, False, False, SCENARIO["rounds"])


def test_a_last_call_with_no_answer_says_the_agents_cap_not_the_providers():
    model = KeepsAskingForTools(streaming=True)
    model._reply = lambda tools: None if tools else ""  # the last call says nothing
    chunks = _stream(_agent(model))
    finishes = [event for chunk in chunks for event in events_from_chunk(chunk) if isinstance(event, Finish)]
    last = finishes[-1]
    explained = present_empty_reply(last.reason, blocked=last.blocked, retried=last.retried, rounds=last.rounds)
    assert explained.headline == SCENARIO["headline"]
    assert "provider" not in explained.headline


def test_the_same_call_three_times_stops_early_with_the_loop_reason():
    loop = FIXTURE["loop"]
    model = KeepsAskingForTools(streaming=True, same_args=True)
    chunks = _stream(_agent(model, limit=loop["limit"]))
    assert model.calls == loop["model_calls"]
    assert model.tools_offered[-1] is False
    assert model.seen[-1][-1] == {"role": "system", "content": FIXTURE["loop_answer"].format(tool=loop["tool"])}
    assert loop_answer_message(loop["tool"]) == FIXTURE["loop_answer"].format(tool=loop["tool"])
    assert _text(chunks).strip() == SCENARIO["answer"]
    assert chunks[-1]["webagents_finish"] == {
        "reason": FIXTURE["loop_finish_reason"], "blocked": False, "retried": False, "rounds": loop["rounds"], "tool": loop["tool"],
    }
    assert tool_loop_sentence(loop["tool"]) == loop["says"]


def test_same_arguments_are_the_fixtures():
    assert REPEAT_LIMIT == FIXTURE["repeat_limit"]
    for case in FIXTURE["same_arguments"]:
        assert (canonical_arguments(case["a"]) == canonical_arguments(case["b"])) is case["same"], case


def test_an_answer_in_the_last_round_is_an_answer():
    model = KeepsAskingForTools(streaming=True, answer_at=SCENARIO["limit"])
    chunks = _stream(_agent(model))
    assert model.calls == SCENARIO["limit"]
    assert _text(chunks).strip() == "done"
    assert all(chunk.get("webagents_finish", {}).get("reason") not in (TOOL_ROUND_LIMIT, "tool_loop") for chunk in chunks)


def test_the_default_budget_lets_a_sixth_round_run():
    # Five rounds used to be the end of the turn, with no answer.
    model = KeepsAskingForTools(streaming=True, answer_at=7)
    chunks = _stream(_agent(model, limit=None))
    assert model.calls == 7
    assert all(chunk.get("webagents_finish", {}).get("reason") != TOOL_ROUND_LIMIT for chunk in chunks)


def test_a_non_streaming_turn_answers_at_the_cap_too():
    model = KeepsAskingForTools(streaming=False)
    response = asyncio.run(_agent(model).run([{"role": "user", "content": "nice"}]))
    assert model.calls == SCENARIO["model_calls"]
    assert model.tools_offered[-1] is False
    assert _warnings(model.seen[SCENARIO["warned_at_call"] - 1]) == [budget_warning(SCENARIO["warned_at_call"], SCENARIO["limit"])]
    assert response["choices"][0]["message"]["content"] == SCENARIO["answer"]
    assert response["webagents_finish"]["reason"] == TOOL_ROUND_LIMIT
    assert response["webagents_finish"]["rounds"] == SCENARIO["rounds"]
    source = (Path(__file__).resolve().parents[1] / "webagents" / "agents" / "core" / "base_agent.py").read_text()
    assert "technical difficulties" not in source
    from webagents.server.core.app import openai_completion_body

    assert openai_completion_body(response)["webagents_finish"]["reason"] == TOOL_ROUND_LIMIT


def test_a_non_streaming_answer_in_the_last_round_is_kept():
    model = KeepsAskingForTools(streaming=False, answer_at=SCENARIO["limit"])
    response = asyncio.run(_agent(model).run([{"role": "user", "content": "nice"}]))
    assert response["choices"][0]["message"]["content"] == "done"
    assert "webagents_finish" not in response


def test_p_prints_the_answer_says_the_cap_and_exits_1(monkeypatch, capsys):
    # `-p` prints the last call's answer, says the cap on stderr (text) or in
    # `finish` (json), and exits 1, as the TypeScript `-p` does.
    from types import SimpleNamespace

    import webagents.cli.agent_builder as agent_builder
    from webagents.cli.one_shot import _run

    agent = _agent(KeepsAskingForTools(streaming=True))

    async def build_agent(*_args, **_kwargs):
        return SimpleNamespace(agent=agent, model_problem=None, access=SimpleNamespace(kind="direct", model="stub"), model_label="stub")

    monkeypatch.setattr(agent_builder, "build_agent", build_agent)
    assert asyncio.run(_run(None, "nice", None, "text")) == 1
    out, err = capsys.readouterr()
    assert out.strip() == SCENARIO["answer"]
    assert f"Error: {SCENARIO['answered']}" in err
    assert "provider reported" not in err
    assert asyncio.run(_run(None, "nice", None, "json")) == 1
    body = json.loads(capsys.readouterr().out)
    assert body["content"].strip() == SCENARIO["answer"]
    assert body["finish"] == {"reason": TOOL_ROUND_LIMIT, "rounds": SCENARIO["rounds"]}


# ---------------------------------------------------------------------------
# The built-in agent answers small talk without tools
# ---------------------------------------------------------------------------

SMALL_TALK_RULE = (
    "- Answer greetings, thanks and small talk directly, without tools. Use a tool only when the request needs one, "
    "and do not explore the folder unless the person asks you to."
)


def test_both_robutler_files_carry_the_small_talk_rule_and_match():
    root = Path(__file__).resolve().parents[2]
    python_copy = (root / "python" / "webagents" / "agents" / "builtin" / "ROBUTLER.md").read_text()
    typescript_copy = (root / "typescript" / "src" / "agents" / "ROBUTLER.md").read_text()
    assert python_copy == typescript_copy
    how_to_answer = python_copy.split("## How to answer", 1)[1]
    assert SMALL_TALK_RULE in how_to_answer
    assert "—" not in python_copy
