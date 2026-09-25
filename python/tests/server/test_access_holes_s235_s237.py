"""
S-235, S-236 and S-237 (2026-09-25): three access holes in the Python server,
each pinned here against the code that had it.

  * S-235: the command routes needed no credential, and `execute_command`
    checked a command's scope only when a context was passed, which no route
    did; any page the person opened could run the local daemon's commands with
    a `text/plain` POST.
  * S-236: the Robutler AuthSkill's refusal was a bare exception, which the
    hook runner logs and walks past, so the request ran anyway at `all` scope.
  * S-237: a request tool named like a hidden owner-only tool made the agent
    run its own owner-only function for the caller.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent, CommandForbidden
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command, tool
from webagents.server.context.context_vars import create_context, set_context
from webagents.server.core.app import WebAgentsServer


class OwnerThings(Skill):
    @command("/owner/thing", description="An owner command", scope="owner")
    async def owner_thing(self) -> str:
        return "owner ran"

    @command("/any/thing", description="Anyone's command", scope="all")
    async def any_thing(self) -> str:
        return "anyone ran"

    @command("/admin/thing", description="An admin command", scope="admin")
    async def admin_thing(self) -> str:
        return "admin ran"

    @tool(scope="owner")
    async def secret_tool(self) -> str:
        """An owner-only tool."""
        return "secret"


def _agent(**skills) -> BaseAgent:
    agent = BaseAgent(name="guarded", instructions="Guarded.", skills={"things": OwnerThings(), **skills})
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _as(scope: str):
    ctx = create_context(messages=[], stream=False)
    ctx.auth = SimpleNamespace(scope=SimpleNamespace(value=scope))
    set_context(ctx)
    return ctx


# -- S-235 ---------------------------------------------------------------------------------


class TestCommandScope:
    def test_no_context_is_the_least_privilege(self):
        agent = _agent()
        with pytest.raises(CommandForbidden):
            asyncio.run(agent.execute_command("/owner/thing", {}))
        assert asyncio.run(agent.execute_command("/any/thing", {})) == "anyone ran"

    def test_the_owner_may(self):
        agent = _agent()
        assert asyncio.run(agent.execute_command("/owner/thing", {}, context=_as("owner"))) == "owner ran"

    def test_admin_scope_is_checked_too(self):
        # It was not checked at all: only `owner` was.
        agent = _agent()
        with pytest.raises(CommandForbidden):
            asyncio.run(agent.execute_command("/admin/thing", {}, context=_as("owner")))
        assert asyncio.run(agent.execute_command("/admin/thing", {}, context=_as("admin"))) == "admin ran"


class TestCommandRoutes:
    @pytest.fixture
    def client(self):
        agent = _agent()

        def resolve(name, **kwargs):
            return agent if name == "guarded" else None

        return TestClient(WebAgentsServer(agents=[], dynamic_agents=resolve).app)

    def test_no_credential_no_command(self, client):
        response = client.post("/guarded/command/any/thing", json={})
        assert response.status_code == 401

    def test_a_plain_text_post_is_refused(self, client):
        # The cross-origin form a web page can send without a preflight.
        response = client.post(
            "/guarded/command/any/thing",
            content=b"{}",
            headers={"Authorization": "Bearer x", "Content-Type": "text/plain"},
        )
        assert response.status_code == 415

    def test_an_owner_command_is_refused_to_a_caller_who_is_not_one(self, client):
        response = client.post("/guarded/command/owner/thing", json={}, headers={"Authorization": "Bearer x"})
        assert response.status_code == 403

    def test_an_all_command_still_runs(self, client):
        response = client.post("/guarded/command/any/thing", json={}, headers={"Authorization": "Bearer x"})
        assert response.status_code == 200
        assert response.json() == {"result": "anyone ran"}


def test_the_owner_acting_commands_say_so():
    # Declared `all`, so any caller with a credential could run them.
    from webagents.agents.skills.local.auth.skill import AuthSkill as LocalAuth
    from webagents.agents.skills.local.mcp.skill import LocalMcpSkill
    from webagents.agents.skills.local.session.skill import SessionManagerSkill

    wanted = {"/auth/token", "/mcp/call", "/session/save", "/session/load", "/session/new", "/session/history"}
    seen = set()
    for cls in (LocalAuth, LocalMcpSkill, SessionManagerSkill):
        for attr in dir(cls):
            fn = getattr(cls, attr, None)
            path = getattr(fn, "_command_path", None)
            if path in wanted:
                seen.add(path)
                assert getattr(fn, "_command_scope", None) == "owner", path
    assert seen == wanted


# -- S-236 ---------------------------------------------------------------------------------


def _refusing_auth_skill(monkeypatch):
    from webagents.agents.skills.robutler.auth.skill import AuthSkill

    skill = AuthSkill({"require_auth": True})

    async def nothing(*args, **kwargs):
        return None

    for name in ("_authenticate_api_key", "_authenticate_with_owner_assertion_only", "_authenticate_service_token"):
        monkeypatch.setattr(skill, name, nothing)
    return skill


class TestARefusedCredentialStopsTheRequest:
    def test_the_hook_runner_raises_it(self, monkeypatch):
        from webagents.agents.skills.robutler.auth.skill import AuthenticationError

        agent = _agent(auth=_refusing_auth_skill(monkeypatch))
        ctx = create_context(messages=[], stream=False, agent=agent)
        with pytest.raises(AuthenticationError):
            asyncio.run(agent._execute_hooks("on_connection", ctx))

    @pytest.mark.parametrize("stream", [False, True])
    def test_the_server_answers_401_and_the_model_is_never_called(self, monkeypatch, stream):
        agent = _agent(auth=_refusing_auth_skill(monkeypatch))
        called = []

        async def never(*args, **kwargs):
            called.append(True)
            yield {"choices": [{"index": 0, "delta": {"content": "should not run"}}]}

        monkeypatch.setattr(agent, "_execute_handoff", never, raising=False)
        client = TestClient(WebAgentsServer(agents=[agent]).app)
        response = client.post(
            "/guarded/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": stream},
            headers={"Authorization": "Bearer not-a-real-key"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert called == []


# -- S-237 ---------------------------------------------------------------------------------


class TestAHiddenToolIsNeverRunForTheCaller:
    def test_a_shadowing_request_tool_is_the_callers_own(self):
        agent = _agent()
        _as("all")
        shadow = {"type": "function", "function": {"name": "secret_tool", "parameters": {"type": "object", "properties": {}}}}
        agent._merge_tools([shadow])
        assert agent._get_tool_function_by_name("secret_tool") is None

    def test_scope_is_checked_when_the_tool_would_run(self):
        agent = _agent()
        _as("all")
        assert agent._get_tool_function_by_name("secret_tool") is None
        _as("owner")
        assert agent._get_tool_function_by_name("secret_tool") is not None


# -- S-244 -----------------------------------------------------------------------------------


def test_a_handoffs_tool_keeps_the_handoffs_scope_s244():
    """The auto-registered `use_<target>` tool carried the handoff's scope in
    an attribute `register_tool` does not read, so an owner-only handoff was
    offered to every caller (S-244, 2026-09-25). And every such tool of a skill
    switched to the skill's LAST handoff: the target was read late."""
    from webagents.agents.tools.decorators import handoff

    class TwoRoutes(Skill):
        @handoff(name="private_route", scope="owner", auto_tool=True, priority=90)
        async def private_route(self, messages, **kwargs):
            yield {"choices": [{"delta": {"content": "private"}}]}

        @handoff(name="public_route", scope="all", auto_tool=True, priority=91)
        async def public_route(self, messages, **kwargs):
            yield {"choices": [{"delta": {"content": "public"}}]}

    skill = TwoRoutes()
    agent = BaseAgent(name="routes", instructions="Routes.", skills={"routes": skill})
    tools = {t["name"]: t for t in agent._registered_tools}
    assert tools["use_private_route"]["scope"] == "owner"
    assert tools["use_public_route"]["scope"] == "all"

    anyone = [t["name"] for t in agent.get_tools_for_scopes(["all"])]
    assert "use_private_route" not in anyone and "use_public_route" in anyone

    private = asyncio.run(tools["use_private_route"]["function"]())
    public = asyncio.run(tools["use_public_route"]["function"]())
    assert private == skill.request_handoff("private_route")
    assert public == skill.request_handoff("public_route")
