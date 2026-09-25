"""A tool's answer reaches the model as the TypeScript agent sends it (2026-09-25).

Text as it is; anything else as JSON with `JSON.stringify`'s bytes (no spaces
after separators, non-ASCII as itself). Python's `str()` spelling
(`{'ok': True, 'error': None}`) is what the model used to read here.
"""

import asyncio

import pytest

from webagents.agents.core.base_agent import BaseAgent, tool_result_text
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool


@pytest.mark.parametrize(
    "value,text",
    [
        ("plain text", "plain text"),
        ({"ok": True, "error": None}, '{"ok":true,"error":null}'),
        ({"results": [{"name": "café", "score": 0.5}]}, '{"results":[{"name":"café","score":0.5}]}'),
        ([1, "two", False], '[1,"two",false]'),
        (42, "42"),
        (None, "null"),
    ],
)
def test_the_text_is_json_stringify_s(value, text):
    assert tool_result_text(value) == text


def test_a_value_json_cannot_hold_falls_back_to_its_str():
    class Opaque:
        def __str__(self):
            return "opaque"

    assert tool_result_text(Opaque()) == "opaque"


class Answers(Skill):
    @tool(name="answer", description="Answers a dict.", scope="all")
    async def answer(self) -> dict:
        return {"found": ["a", "b"], "error": None}


def test_the_agent_sends_a_dict_answer_as_json():
    agent = BaseAgent(name="a", instructions="x", skills={"answers": Answers()})
    message = asyncio.run(
        agent._execute_single_tool({"id": "call_1", "function": {"name": "answer", "arguments": "{}"}})
    )
    assert message == {"tool_call_id": "call_1", "role": "tool", "content": '{"found":["a","b"],"error":null}'}
