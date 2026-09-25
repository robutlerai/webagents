"""
The one scope rule (ADR-0045 section 5, 2026-09-25), from the table both SDKs run,
and where a Python agent applies it: tools, prompts, the tool that would run, widgets
and commands.

Before this an unknown scope took the `all` level in Python, so a tool declared
`scope="group:friends"` was shown to, and run for, every caller.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.core.base_agent import BaseAgent, CommandForbidden
from webagents.agents.core.scopes import caller_scopes, scope_allows
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command, prompt, tool
from webagents.server.context.context_vars import create_context, set_context

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "scopes" / "scope_allows.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['required']!r}<-{c['caller']!r}")
def test_the_shared_table(case):
    assert scope_allows(case["required"], case["caller"]) is case["allowed"]


class TestCallerScopes:
    def test_anonymous_holds_nothing(self):
        assert caller_scopes(None) == frozenset()

    def test_the_tier_and_the_groups(self):
        auth = SimpleNamespace(scope=SimpleNamespace(value="user"), groups=["friends", "family"])
        assert caller_scopes(auth) == {"user", "group:friends", "group:family"}

    def test_a_tokens_own_scope_list_is_never_read(self):
        # The local AOAuth context carries the `scope` claim an issuer chose.
        auth = SimpleNamespace(scopes=["owner", "admin", "group:friends"])
        assert caller_scopes(auth) == frozenset()


class Guarded(Skill):
    @tool(scope="group:friends")
    async def friends_tool(self) -> str:
        """For friends."""
        return "friends"

    @tool(scope="project")
    async def odd_tool(self) -> str:
        """A scope nobody grants."""
        return "odd"

    @tool(scope=["group:friends", "group:family"])
    async def kin_tool(self) -> str:
        """For friends or family."""
        return "kin"

    @tool(scope="owner")
    async def owner_tool(self) -> str:
        """For the owner."""
        return "owner"

    @prompt(scope="group:friends")
    def friends_prompt(self) -> str:
        return "FRIENDS PROMPT"

    @prompt(scope="project")
    def odd_prompt(self) -> str:
        return "ODD PROMPT"

    @command("/friends/thing", description="For friends", scope="group:friends")
    async def friends_command(self) -> str:
        return "friends ran"


def _agent() -> BaseAgent:
    agent = BaseAgent(name="guarded", instructions="Guarded.", skills={"guarded": Guarded()})
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _as(tier=None, groups=()):
    ctx = create_context(messages=[], stream=False)
    if tier or groups:
        ctx.auth = SimpleNamespace(scope=SimpleNamespace(value=tier or "user"), groups=list(groups))
    set_context(ctx)
    return ctx


def _visible(agent) -> set:
    return {t["name"] for t in agent.get_tools_for_scopes(agent._caller_scopes_of(_current()))}


def _current():
    from webagents.server.context.context_vars import get_context

    return get_context()


class TestTheAgentAppliesIt:
    def test_a_group_tool_is_the_groups(self):
        agent = _agent()
        _as()
        assert "friends_tool" not in _visible(agent)
        _as("user")
        assert "friends_tool" not in _visible(agent)
        _as("user", ["friends"])
        assert "friends_tool" in _visible(agent)
        _as("owner")
        assert "friends_tool" in _visible(agent)

    def test_a_list_is_any_of(self):
        agent = _agent()
        _as("user", ["family"])
        assert "kin_tool" in _visible(agent)
        assert "friends_tool" not in _visible(agent)

    def test_an_unknown_scope_fails_closed(self):
        agent = _agent()
        for tier in (None, "user", "owner", "admin"):
            _as(tier)
            assert "odd_tool" not in _visible(agent), tier

    def test_the_tool_that_would_run_is_checked_the_same_way(self):
        agent = _agent()
        _as("user")
        assert agent._get_tool_function_by_name("friends_tool") is None
        _as("user", ["friends"])
        assert agent._get_tool_function_by_name("friends_tool") is not None

    def test_prompts(self):
        agent = _agent()
        ctx = _as("user")
        text = asyncio.run(agent._execute_prompts(ctx))
        assert "FRIENDS PROMPT" not in text and "ODD PROMPT" not in text
        ctx = _as("user", ["friends"])
        text = asyncio.run(agent._execute_prompts(ctx))
        assert "FRIENDS PROMPT" in text and "ODD PROMPT" not in text

    def test_commands(self):
        agent = _agent()
        with pytest.raises(CommandForbidden):
            asyncio.run(agent.execute_command("/friends/thing", {}, context=_as("user")))
        assert asyncio.run(agent.execute_command("/friends/thing", {}, context=_as("user", ["friends"]))) == "friends ran"
