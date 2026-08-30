"""Sender attribution, driven through a REAL `POST /{agent}/chat/completions`.

The platform relays every turn under a service token whose `sub` is
`service:robutler-router`; the PERSON is named only by the request body's
`metadata.sender`. `AuthSkill._extract_platform_sender_id` has always read
that off the context — and nothing put it there, so the scope was permanently
USER and all 54 `scope="owner"`/`"admin"` tool registrations were unreachable
on platform-routed calls.

A test built on a hand-made Context cannot catch that: it tests the reader,
not the wiring. This one posts an actual HTTP request at the server and looks
at what the agent's hooks saw.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff, Skill
from webagents.agents.tools.decorators import hook
from webagents.server.context.context_vars import get_context
from webagents.server.core.app import create_server


class ProbeSkill(Skill):
    """Records what the context looked like at on_connection — the same
    lifecycle point AuthSkill authenticates in."""

    def __init__(self):
        super().__init__()
        self.seen_metadata = None

    @hook("on_connection")
    async def capture(self, context):
        ctx = context or get_context()
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = ctx.get("metadata") if hasattr(ctx, "get") else None
        self.seen_metadata = metadata
        return context


class StubLLM(Skill):
    """A handoff so the agent can complete a turn without a model provider."""

    async def initialize(self, agent):
        await super().initialize(agent)
        agent.register_handoff(
            Handoff(
                target="stub_llm",
                description="Stub completion",
                scope="all",
                metadata={"function": self.completion, "priority": 10, "is_generator": True},
            ),
            source="stub_llm",
        )

    async def completion(self, messages, tools=None, **kwargs):
        yield {"choices": [{"delta": {"content": "ok"}}]}


@pytest.fixture
def probe_agent():
    probe = ProbeSkill()
    agent = BaseAgent(
        name="mini",
        instructions="be helpful",
        skills={"probe": probe, "llm": StubLLM()},
    )
    return agent, probe


@pytest.mark.asyncio
async def test_request_metadata_sender_reaches_the_context(probe_agent):
    agent, probe = probe_agent
    server = create_server(agents=[agent])

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mini/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "metadata": {
                    "chat_id": "chat-1",
                    "chat_type": "direct",
                    "platform": "robutler",
                    "sender": {"id": "user-42", "username": "alice"},
                },
            },
            headers={"Authorization": "Bearer service-token"},
        )

    assert response.status_code == 200, response.text
    assert probe.seen_metadata is not None, "request metadata never reached the context"
    assert probe.seen_metadata["sender"]["id"] == "user-42"
    assert probe.seen_metadata["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_no_metadata_is_not_an_error(probe_agent):
    """A plain OpenAI client sends no `metadata`; that must still run."""
    agent, probe = probe_agent
    server = create_server(agents=[agent])

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mini/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
            # No `metadata`, but still a credential: completions refuses a
            # request that presents none (it bills the owner's model).
            headers={"Authorization": "Bearer service-token"},
        )

    assert response.status_code == 200, response.text
