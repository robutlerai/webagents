"""
`created` is a positive Unix timestamp on every streamed chunk and on the
completion the daemon's route answers (2026-09-27, the final e2e re-run:
both were `null`). A model client whose server sent no `created` yields chunks
with `created: None`; the agent now stamps every chunk of a stream with one
timestamp (`BaseAgent.run_streaming`, as the TypeScript server stamps its
chunks), and the completions transport's merge (the daemon's dynamic route)
takes a timestamp when the first chunk carries none. The fixture
`tests/fixtures/completions/response_shapes.json` names the rule.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff, Skill
from webagents.agents.skills.core.transport import CompletionsTransportSkill
from webagents.server.core.app import WebAgentsServer

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "completions" / "response_shapes.json").read_text())
AUTHED = {"Authorization": "Bearer any-value"}


class NoCreatedModel(Skill):
    """A model whose server sent no `created`: the OpenAI client then yields chunks with `created: None`."""

    async def initialize(self, agent):
        self.agent = agent
        agent.register_handoff(
            Handoff(target="stub", description="stub", scope="all", metadata={"function": self.chat_completion_stream, "priority": 10}),
            source="stub",
        )

    async def chat_completion_stream(self, messages, tools=None, **kwargs):
        base = {"id": "cmpl-e2e", "created": None, "model": "gpt-4o-mini", "object": "chat.completion.chunk"}
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {"content": "You said: hi."}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {**base, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}


def _agent(name: str = "reporter") -> BaseAgent:
    return BaseAgent(name=name, instructions="Echo.", skills={"model": NoCreatedModel(), "completions": CompletionsTransportSkill()})


def _is_timestamp(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value - int(time.time())) < 600


def test_the_fixture_names_the_rule():
    assert "created" in FIXTURE and "timestamp" in FIXTURE["created"]["rule"]


def test_every_streamed_chunk_carries_one_timestamp():
    agent = _agent()

    async def collect():
        return [chunk async for chunk in agent.run_streaming([{"role": "user", "content": "hi"}])]

    chunks = asyncio.run(collect())
    assert len(chunks) >= 3
    stamps = {chunk["created"] for chunk in chunks if isinstance(chunk, dict) and chunk.get("object") == "chat.completion.chunk"}
    assert len(stamps) == 1 and _is_timestamp(next(iter(stamps)))


def test_a_finished_run_carries_a_timestamp():
    agent = _agent()

    async def run():
        return await agent.run([{"role": "user", "content": "hi"}])

    response = asyncio.run(run())
    assert response["object"] == "chat.completion" and _is_timestamp(response["created"])


def test_the_static_route_streams_timestamps():
    server = WebAgentsServer(agents=[_agent()], enable_monitoring=False)
    with TestClient(server.app) as client:
        res = client.post("/reporter/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}, headers=AUTHED)
        assert res.status_code == 200, res.text
        events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        assert events and all(_is_timestamp(event["created"]) for event in events), events
        assert len({event["created"] for event in events}) == 1


def test_the_daemons_dynamic_route_answers_timestamps_streamed_and_merged():
    # The daemon serves file-loaded agents through the dynamic catch-all and
    # the completions transport (`server/core/app.py`), not the static route.
    agent = _agent()

    async def resolve(name: str):
        return agent if name == "reporter" else None

    server = WebAgentsServer(dynamic_agents=resolve, enable_monitoring=False)
    with TestClient(server.app) as client:
        merged = client.post("/reporter/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}, headers=AUTHED)
        assert merged.status_code == 200, merged.text
        body = merged.json()
        assert body["object"] == "chat.completion" and _is_timestamp(body["created"]), body
        assert body["choices"][0]["message"]["content"] == "You said: hi."

        streamed = client.post("/reporter/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}, headers=AUTHED)
        assert streamed.status_code == 200, streamed.text
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        assert events and all(_is_timestamp(event["created"]) for event in events), events
