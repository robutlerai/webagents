"""
The shape of a served agent's chat completion (2026-09-26, the new-developer
e2e run): `created` was null, a tool turn's `usage` counted only the last
model call, and a tool call handed back to the client came out labelled
`chat.completion.chunk` with a `delta`. Pinned by
`tests/fixtures/completions/response_shapes.json`, which the TypeScript suite
runs too (`tests/unit/server/completion-shapes-e2efix.test.ts`), through
`BaseAgent.run()` and through the server's route.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import handoff, tool
from webagents.server.core.app import WebAgentsServer

SHAPES = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "completions" / "response_shapes.json").read_text())
CALLS = SHAPES["model_calls"]


class ScriptedLLM(Skill):
    """A model that answers a tool call on the first call of a turn and text on the second,
    each call reporting the fixture's usage and no `created` (as some servers send none)."""

    def __init__(self, tool_name: str = "ping"):
        super().__init__({})
        self.tool_name = tool_name

    @handoff(name="scripted")
    async def answer(self, messages: List[Dict[str, Any]], tools=None, **kwargs) -> Dict[str, Any]:
        second = any(m.get("role") == "tool" for m in messages)
        usage = dict(CALLS[1] if second else CALLS[0])
        if second:
            message: Dict[str, Any] = {"role": "assistant", "content": "Tool said: pong"}
            finish = "stop"
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": self.tool_name, "arguments": "{}"}}]}
            finish = "tool_calls"
        return {"id": "cmpl-scripted", "object": "chat.completion", "model": "scripted", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


class StreamedLLM(Skill):
    """The same model, streaming as the OpenAI skill does (it registers its
    streaming function as the handoff, and a non-streaming `run` rebuilds one
    response from the chunks): tool-call deltas or content deltas, the finish,
    then the usage-only chunk. The rebuilt tool-call response dropped that
    chunk, so a two-call turn reported one call's usage (the e2e run's h2 step)."""

    def __init__(self, tool_name: str = "ping"):
        super().__init__({})
        self.tool_name = tool_name

    @handoff(name="scripted")
    async def answer(self, messages: List[Dict[str, Any]], tools=None, **kwargs):
        second = any(m.get("role") == "tool" for m in messages)
        usage = dict(CALLS[1] if second else CALLS[0])
        head = {"id": "cmpl-scripted", "object": "chat.completion.chunk", "model": "scripted"}
        if second:
            yield {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Tool said: "}, "finish_reason": None}]}
            yield {**head, "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}]}
            yield {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        else:
            call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": self.tool_name, "arguments": ""}}
            yield {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [call]}, "finish_reason": None}]}
            yield {**head, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": None}]}
            yield {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        yield {**head, "choices": [], "usage": usage}


class Tools(Skill):
    @tool(name="ping", description="Answers pong.")
    async def ping(self) -> str:
        return "pong"


def _agent(tool_name: str = "ping", streamed: bool = False) -> BaseAgent:
    llm = StreamedLLM(tool_name) if streamed else ScriptedLLM(tool_name)
    agent = BaseAgent(name="shaped", instructions="x", skills={"llm": llm, "tools": Tools()})
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def test_a_tool_turn_sums_every_model_call_and_created_is_a_timestamp():
    result = asyncio.run(_agent().run([{"role": "user", "content": "TOOL ping"}]))
    assert result["object"] == SHAPES["completion"]["object"]
    assert isinstance(result["created"], int) and result["created"] > 0
    assert result["usage"] == SHAPES["turn_usage"]
    assert result["choices"][0]["message"]["content"] == "Tool said: pong"


def test_a_tool_call_handed_back_to_the_client_is_a_completion_not_a_chunk():
    expected = SHAPES["handed_back_tool_call"]
    result = asyncio.run(_agent(expected["tool"]).run([{"role": "user", "content": "FORCE not_ours"}]))
    assert result["object"] == expected["object"]
    choice = result["choices"][0]
    assert choice["finish_reason"] == expected["finish_reason"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == expected["tool"]
    for key in expected["never_carries"]:
        assert key not in choice
    assert isinstance(result["created"], int) and result["created"] > 0
    # One model call: its usage, not zero.
    assert result["usage"] == CALLS[0]


def test_a_streamed_tool_turn_sums_every_model_call():
    # Through the chunk reconstruction, the path a served OpenAI-backed agent
    # takes: the tool-call stream's usage-only chunk counts too.
    result = asyncio.run(_agent(streamed=True).run([{"role": "user", "content": "TOOL ping"}]))
    assert result["object"] == SHAPES["completion"]["object"]
    assert result["usage"] == SHAPES["turn_usage"]
    assert result["choices"][0]["message"]["content"] == "Tool said: pong"


def test_a_streamed_tool_call_handed_back_carries_its_usage():
    expected = SHAPES["handed_back_tool_call"]
    result = asyncio.run(_agent(expected["tool"], streamed=True).run([{"role": "user", "content": "FORCE not_ours"}]))
    assert result["object"] == expected["object"]
    choice = result["choices"][0]
    assert choice["finish_reason"] == expected["finish_reason"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == expected["tool"]
    for key in expected["never_carries"]:
        assert key not in choice
    assert result["usage"] == CALLS[0]


def test_the_served_route_answers_the_same_shape():
    server = WebAgentsServer(agents=[_agent()], enable_monitoring=False, enable_prometheus=False, enable_rate_limiting=False, enable_request_logging=False, heartbeat=False, quiet=True, agent_card=False)
    client = TestClient(server.app, headers={"Authorization": "Bearer test-service-token"})
    response = client.post("/shaped/chat/completions", json={"messages": [{"role": "user", "content": "TOOL ping"}]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body) == SHAPES["completion"]["top_level_keys"]
    assert body["object"] == SHAPES["completion"]["object"]
    assert isinstance(body["created"], int) and body["created"] > 0
    assert body["usage"] == SHAPES["turn_usage"]
