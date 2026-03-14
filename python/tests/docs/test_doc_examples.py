"""Runnable tests that mirror the highest-value doc examples.

Each test references the doc file and section it validates.
No real API keys or LLM calls needed -- tests validate SDK wiring only.
"""

import pytest

try:
    from webagents.agents.core.base_agent import BaseAgent
    from webagents.agents.skills.base import Skill
    from webagents.agents.tools.decorators import tool, hook, http

    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    BaseAgent = Skill = tool = hook = http = None

try:
    from webagents.agents.skills.robutler.payments.skill import pricing

    HAS_PRICING = True
except ImportError:
    HAS_PRICING = False
    pricing = None

pytestmark = pytest.mark.skipif(not HAS_SDK, reason="webagents SDK not importable")


# ---------------------------------------------------------------------------
# Validates: quickstart.md -- "Create Your First Agent"
# ---------------------------------------------------------------------------


class TestQuickstartAgent:
    def test_create_agent_with_model_string(self):
        agent = BaseAgent(
            name="quickstart-agent",
            instructions="You are a helpful assistant.",
            model="openai/gpt-4o",
        )
        assert agent.name == "quickstart-agent"
        assert agent.model == "openai/gpt-4o"

    def test_create_agent_with_skills_dict(self):
        """quickstart.md -- 'Connect to the Network' (Python)"""
        skill = Skill()
        agent = BaseAgent(
            name="with-skills",
            model="openai/gpt-4o",
            skills={"test_skill": skill},
        )
        assert "test_skill" in agent.skills


# ---------------------------------------------------------------------------
# Validates: agent/tools.md -- @pricing decorator
# ---------------------------------------------------------------------------


class TestPricingDecorator:
    @pytest.mark.skipif(not HAS_PRICING, reason="pricing not importable")
    def test_pricing_attaches_metadata(self):
        """pricing() should set _pricing_config on the function."""

        @pricing(credits_per_call=0.10)
        @tool(name="lookup", description="Look something up")
        async def lookup(query: str) -> str:
            return query

        assert hasattr(lookup, "_pricing_config")
        assert lookup._pricing_config["credits_per_call"] == 0.10


# ---------------------------------------------------------------------------
# Validates: agent/tools.md -- @tool(scope=...)
# ---------------------------------------------------------------------------


class TestToolScoping:
    def test_tool_scope_is_stored(self):
        @tool(name="admin_reset", description="Reset data", scope="admin")
        async def admin_reset() -> str:
            return "done"

        assert admin_reset._tool_scope == "admin"

    def test_tool_scope_default_is_all(self):
        @tool(name="public_info", description="Get info")
        async def public_info() -> str:
            return "info"

        assert public_info._tool_scope == "all"

    def test_agent_filters_tools_by_scope(self):
        """get_tools_for_scope should exclude tools above the caller's level."""

        @tool(name="public_tool", description="Public", scope="all")
        async def public_tool() -> str:
            return "public"

        @tool(name="owner_tool", description="Owner only", scope="owner")
        async def owner_tool() -> str:
            return "owner"

        agent = BaseAgent(
            name="scoped-agent",
            model="openai/gpt-4o",
            tools=[public_tool, owner_tool],
        )

        all_scope_tools = agent.get_tools_for_scope("all")
        owner_scope_tools = agent.get_tools_for_scope("owner")

        all_names = {t["function"]["name"] for t in all_scope_tools}
        owner_names = {t["function"]["name"] for t in owner_scope_tools}

        assert "public_tool" in all_names
        assert "owner_tool" not in all_names
        assert "public_tool" in owner_names
        assert "owner_tool" in owner_names


# ---------------------------------------------------------------------------
# Validates: agent/endpoints.md -- @http decorator
# ---------------------------------------------------------------------------


class TestHttpEndpoints:
    def test_http_decorator_marks_function(self):
        @http("/health", method="get")
        async def health_check(request):
            return {"status": "ok"}

        assert hasattr(health_check, "_http_route")
        assert health_check._http_route["subpath"] == "/health"
        assert health_check._http_route["method"] == "get"


# ---------------------------------------------------------------------------
# Validates: agent/lifecycle.md -- @hook decorator and hook names
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    def test_hook_decorator_stores_event(self):
        @hook("on_connection", priority=1)
        async def on_conn(context):
            return context

        assert on_conn._hook_event == "on_connection"
        assert on_conn._hook_priority == 1

    def test_valid_hook_names_accepted(self):
        valid_hooks = [
            "on_connection",
            "before_llm_call",
            "after_llm_call",
            "before_toolcall",
            "after_toolcall",
            "on_message",
            "on_chunk",
            "finalize_connection",
        ]
        for name in valid_hooks:

            @hook(name)
            async def handler(context):
                return context

            assert handler._hook_event == name

    def test_skill_registers_hooks(self):
        """A skill with @hook methods should collect them on init."""

        class LogSkill(Skill):
            @hook("on_connection", priority=10)
            async def on_connect(self, context):
                return context

            @hook("finalize_connection", priority=99)
            async def on_finalize(self, context):
                return context

        skill = LogSkill()
        hook_events = [h["event"] for h in skill._registered_hooks]
        assert "on_connection" in hook_events
        assert "finalize_connection" in hook_events
