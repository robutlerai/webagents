"""
`stream` defaults to FALSE on chat/completions, per the OpenAI API (2026-09-23).

It defaulted to True. The official OpenAI SDKs OMIT `stream` unless the caller
sets it, so a completely standard `client.chat.completions.create(...)` against
a Python agent got `text/event-stream` and failed to parse it. The TypeScript
server has always defaulted to false, so the two SDKs disagreed about the
endpoint whose whole promise is OpenAI compatibility.

Nothing covered the default, which is how it survived: the whole suite passed
before and after the fix. Found by the cross-SDK campaign, where a TypeScript
agent calling a Python agent got SSE where it expected a completion.
"""

from starlette.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff, Skill
from webagents.server.core.app import create_server

AUTH = {"Authorization": "Bearer test"}


class EchoLLM(Skill):
    """A generator handoff, which is how EVERY real provider skill registers
    (`is_generator: True` in openai, anthropic, google, xai, fireworks, proxy)."""

    async def initialize(self, agent):
        await super().initialize(agent)
        agent.register_handoff(
            Handoff(target="echo", description="echo", scope="all",
                    metadata={"function": self.completion, "priority": 10, "is_generator": True}),
            source="echo",
        )

    async def completion(self, messages, tools=None, **kwargs):
        yield {"choices": [{"delta": {"content": "echoed"}}]}


def client():
    agent = BaseAgent(name="echo", instructions="x", skills={"llm": EchoLLM()})
    return TestClient(create_server(agents=[agent]).app)


def test_omitting_stream_returns_a_json_completion():
    r = client().post("/echo/chat/completions", headers=AUTH,
                      json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["choices"][0]["message"]["content"] == "echoed"


def test_stream_false_returns_a_json_completion():
    r = client().post("/echo/chat/completions", headers=AUTH,
                      json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.headers["content-type"].startswith("application/json")


def test_stream_true_still_streams():
    r = client().post("/echo/chat/completions", headers=AUTH,
                      json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in r.text


def test_the_typescript_server_agrees():
    # The two SDKs serve the same endpoint and must not disagree about it.
    from pathlib import Path

    ts = (Path(__file__).resolve().parents[3] / "typescript" / "src" / "server" / "multi.ts").read_text()
    # TypeScript streams only when asked: `if (body.stream)`.
    assert "if (body.stream)" in ts


# ---------------------------------------------------------------------------
# The same contract through the COMPLETIONS TRANSPORT (2026-09-24)
# ---------------------------------------------------------------------------
#
# Every agent the daemon loads from an AGENT.md gets `CompletionsTransportSkill`
# auto-added, and a transport skill OVERRIDES the route above. The tests above
# never had one, so they passed while every daemon-served agent broke the
# contract twice over: `stream: false` answered 500 (`extend(None)` on the SDK's
# `"tool_calls": null`) and, once that was fixed, `text/event-stream` holding
# one `data:` event and a `[DONE]`. Found by the sandbox end-to-end run.


def transport_client():
    from webagents.agents.skills.core.transport.completions.skill import (
        CompletionsTransportSkill,
    )

    agent = BaseAgent(
        name="echo",
        instructions="x",
        skills={"llm": EchoLLM(), "completions": CompletionsTransportSkill()},
    )
    # RESOLVED DYNAMICALLY, the way the daemon serves an AGENT.md. A static
    # `agents=[...]` agent is answered by the plain route above even when it
    # carries the transport, so a test built that way passes without touching
    # the transport at all (the first version of these did exactly that).
    return TestClient(
        create_server(dynamic_agents=lambda name: agent if name == "echo" else None).app
    )


def test_transport_stream_false_returns_a_json_completion():
    r = transport_client().post("/echo/chat/completions", headers=AUTH,
                                json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "echoed"


def test_transport_omitting_stream_returns_a_json_completion():
    r = transport_client().post("/echo/chat/completions", headers=AUTH,
                                json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["choices"][0]["message"]["content"] == "echoed"


def test_transport_stream_true_still_streams():
    r = transport_client().post("/echo/chat/completions", headers=AUTH,
                                json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in r.text
