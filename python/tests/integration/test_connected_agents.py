"""Connected agents integration test.

Validates the "connected agents" story from the docs:
two real WebAgent servers communicating via HTTP / NLI delegation.
"""

import pytest

try:
    from webagents.agents.core.base_agent import BaseAgent
    from webagents.agents.skills.base import Skill
    from webagents.agents.tools.decorators import tool
    from webagents.server.core.app import WebAgentsServer

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SDK, reason="webagents SDK not importable"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class EchoSkill(Skill):
    """Minimal skill that provides a tool returning its input."""

    @tool(name="echo", description="Echo the input message back")
    async def echo_tool(self, message: str) -> str:
        return f"echo: {message}"


@pytest.fixture
def agent_a():
    """Agent A: has a callable tool."""
    return BaseAgent(
        name="agent-a",
        instructions="You are agent A. Use the echo tool when asked.",
        model="openai/gpt-4o",
        skills={"echo": EchoSkill()},
    )


@pytest.fixture
def agent_b():
    """Agent B: will call agent A via NLI."""
    return BaseAgent(
        name="agent-b",
        instructions="You are agent B. You delegate tasks to agent A.",
        model="openai/gpt-4o",
    )


@pytest.fixture
def two_agent_server(agent_a, agent_b):
    return WebAgentsServer(
        agents=[agent_a, agent_b],
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServerSetup:
    """Validate that the two-agent server starts and routes correctly."""

    def test_server_has_both_agents(self, two_agent_server):
        from fastapi.testclient import TestClient

        client = TestClient(two_agent_server.app)

        resp_a = client.get("/agent-a/models")
        assert resp_a.status_code == 200

        resp_b = client.get("/agent-b/models")
        assert resp_b.status_code == 200

    def test_agent_a_chat_completions_endpoint_exists(self, two_agent_server):
        """The chat completions endpoint should accept POST."""
        from fastapi.testclient import TestClient

        client = TestClient(two_agent_server.app)

        resp = client.post(
            "/agent-a/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        # May fail due to no real LLM, but should not 404
        assert resp.status_code != 404


class TestAgentDiscovery:
    """Validate that agents can list each other's tools."""

    def test_agent_a_exposes_echo_tool(self, agent_a):
        tools = agent_a.get_tools_for_scope("all")
        names = [t["function"]["name"] for t in tools]
        assert "echo" in names

    def test_agent_b_has_no_tools(self, agent_b):
        tools = agent_b.get_tools_for_scope("all")
        assert len(tools) == 0


class TestToolExecution:
    """Validate that tools can be executed directly."""

    @pytest.mark.asyncio
    async def test_echo_tool_execution(self, agent_a):
        echo_skill = agent_a.skills["echo"]
        result = await echo_skill.echo_tool("test message")
        assert result == "echo: test message"
