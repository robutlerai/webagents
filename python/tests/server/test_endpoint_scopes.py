"""
S-242/S-243 (2026-09-25): a scoped `@http` or websocket endpoint is checked
against the caller the agent verified, on every route that serves one, and an
endpoint with no scope stays open to anyone, exactly as before.

Until then the static mount read an endpoint's scope and never checked it (so
the OpenAI skill's owner-only setup form saved credentials for anyone), and
the dynamic route checked it through an import of a module that does not
exist, so there it refused everyone. The refusal bodies are the ones the
TypeScript servers answer (`tests/fixtures/endpoint_gate/refusals.json`).

Also here: the static agent's command routes. The generic mount stopped at the
agent's own `/command/{path:path}` handler with "'path:path' is not a valid
parameter name" at every startup; the fix is NOT to mount that handler as a
plain endpoint (that would drop the S-235 guards) but to give static agents the
same guarded command routes dynamic agents have.
"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from webagents.access.caller import CallerAuth
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import command, hook, http, websocket
from webagents.server.core.app import WebAgentsServer

REFUSALS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "endpoint_gate" / "refusals.json").read_text()
)
NO_CALLER = REFUSALS["no_caller"]
NOT_OPEN = REFUSALS["not_open"]


class HeaderIdentity(Skill):
    """Stands in for an auth skill: `X-Test-Caller` names the caller's tier,
    and `Bearer owner-token` is the owner."""

    identifies_caller = True

    @hook("on_connection", priority=0, scope="all")
    async def who(self, context):
        headers = getattr(getattr(context, "request", None), "headers", None) or {}
        tier = headers.get("x-test-caller")
        if not tier and headers.get("authorization") == "Bearer owner-token":
            tier = "owner"
        if tier:
            context.auth = CallerAuth(scope=tier, provider="test")
        return context


class Payments(Skill):
    """A connection hook that is not identification: an endpoint call must not run it."""

    ran: list = []

    @hook("on_connection", priority=0, scope="all")
    async def charge(self, context):
        Payments.ran.append(True)
        return context


class Endpoints(Skill):
    saved: list = []

    @http("/open", method="get")
    async def open_endpoint(self) -> dict:
        return {"open": True}

    @http("/mine", method="get", scope="owner")
    async def owner_endpoint(self) -> dict:
        return {"mine": True}

    @http("/admins", method="get", scope="admin")
    async def admin_endpoint(self) -> dict:
        return {"admins": True}

    @http("/save", method="post", scope=["owner"])
    async def save(self, value: str = "") -> dict:
        Endpoints.saved.append(value)
        return {"saved": value}

    @http("/items/{item_id}", method="get", scope="owner")
    async def item(self, item_id: str) -> dict:
        return {"item": item_id}

    @http("/files/{rest:path}", method="get")
    async def files(self, rest: str) -> dict:
        return {"rest": rest}

    @websocket("/live", scope="owner")
    async def live(self, ws) -> None:
        await ws.accept()
        await ws.send_json({"live": True})
        await ws.close()

    @websocket("/public")
    async def public(self, ws) -> None:
        await ws.accept()
        await ws.send_json({"public": True})
        await ws.close()

    @command("/owner/thing", description="An owner command", scope="owner")
    async def owner_thing(self) -> str:
        return "owner ran"

    @command("/any/thing", description="Anyone's command", scope="all")
    async def any_thing(self) -> str:
        return "anyone ran"


def _agent(identity: bool = True) -> BaseAgent:
    skills = {"endpoints": Endpoints(), "payments": Payments()}
    if identity:
        skills["identity"] = HeaderIdentity()
    agent = BaseAgent(name="guarded", instructions="Guarded.", skills=skills)
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _static(agent: BaseAgent, **kwargs) -> TestClient:
    return TestClient(WebAgentsServer(agents=[agent], quiet=True).app, **kwargs)


def _dynamic(agent: BaseAgent, **kwargs) -> TestClient:
    def resolve(name, **_):
        return agent if name == "guarded" else None

    return TestClient(WebAgentsServer(agents=[], dynamic_agents=resolve, quiet=True).app, **kwargs)


@pytest.fixture(autouse=True)
def _reset():
    Endpoints.saved.clear()
    Payments.ran.clear()


@pytest.fixture(params=["static", "dynamic"])
def serve(request):
    return _static if request.param == "static" else _dynamic


# -- @http --------------------------------------------------------------------------------


class TestHttpEndpoints:
    def test_an_endpoint_with_no_scope_is_open(self, serve):
        response = serve(_agent()).get("/guarded/open")
        assert response.status_code == 200
        assert response.json() == {"open": True}

    def test_an_anonymous_caller_is_asked_to_identify(self, serve):
        response = serve(_agent()).get("/guarded/mine")
        assert response.status_code == NO_CALLER["status"]
        assert response.json() == NO_CALLER["body"]

    def test_a_verified_caller_outside_the_scope_is_forbidden(self, serve):
        response = serve(_agent()).get("/guarded/mine", headers={"X-Test-Caller": "user"})
        assert response.status_code == NOT_OPEN["status"]
        assert response.json() == NOT_OPEN["body"]

    def test_the_owner_may(self, serve):
        client = serve(_agent())
        assert client.get("/guarded/mine", headers={"X-Test-Caller": "owner"}).json() == {"mine": True}
        # An admin passes an owner scope; an owner does not pass an admin one.
        assert client.get("/guarded/mine", headers={"X-Test-Caller": "admin"}).status_code == 200
        assert client.get("/guarded/admins", headers={"X-Test-Caller": "owner"}).status_code == 403

    def test_path_parameters_reach_a_scoped_handler(self, serve):
        response = serve(_agent()).get("/guarded/items/42", headers={"X-Test-Caller": "owner"})
        assert response.json() == {"item": "42"}

    def test_a_refused_post_never_runs_the_handler(self, serve):
        client = serve(_agent())
        assert client.post("/guarded/save", json={"value": "theirs"}).status_code == 401
        assert client.post("/guarded/save", json={"value": "theirs"}, headers={"X-Test-Caller": "user"}).status_code == 403
        assert Endpoints.saved == []
        assert client.post("/guarded/save", json={"value": "mine"}, headers={"X-Test-Caller": "owner"}).json() == {"saved": "mine"}
        assert Endpoints.saved == ["mine"]

    def test_an_agent_that_verifies_no_one_keeps_scoped_endpoints_shut(self, serve):
        response = serve(_agent(identity=False)).get("/guarded/mine", headers={"Authorization": "Bearer anything"})
        assert response.status_code == 401

    def test_identifying_the_caller_takes_no_payment(self, serve):
        serve(_agent()).get("/guarded/mine", headers={"X-Test-Caller": "owner"})
        assert Payments.ran == []

    def test_a_path_converter_is_one_parameter(self, serve):
        # `{rest:path}` was read as a parameter named "rest:path".
        response = serve(_agent()).get("/guarded/files/a/b/c.txt")
        assert response.json() == {"rest": "a/b/c.txt"}

    def test_the_agents_own_key_in_a_url_makes_no_one_its_owner(self, serve):
        # The dynamic route made a localhost `?token` equal to the agent's own
        # platform credential its owner; the OpenAI setup link relied on it and
        # handed that credential to every caller (S-246). The rule is gone.
        agent = _agent(identity=False)
        agent.api_key = "agent-key-1"
        client = serve(agent, base_url="http://localhost")
        assert client.get("/guarded/mine?token=agent-key-1").status_code == 401


class TestAccessBlock:
    """The access block is identification too: its groups are scopes."""

    def _agent(self, block):
        from webagents.access.install import add_access, finish_access

        skills = {"endpoints": Endpoints()}
        policy = add_access(skills, block, None)
        agent = BaseAgent(name="guarded", instructions="Guarded.", skills=skills)
        finish_access(agent, policy, skills)
        asyncio.run(agent._ensure_skills_initialized())
        return agent

    def test_a_refused_caller_gets_the_blocks_answer(self, serve):
        client = serve(self._agent({"default": "none"}))
        response = client.get("/guarded/mine")
        assert response.status_code == 403
        assert response.json() == {"error": {"code": "forbidden", "message": "This agent does not accept requests from this caller."}}
        # An endpoint with no scope is still open: the block decides turns and scoped endpoints.
        assert client.get("/guarded/open").status_code == 200


# -- websockets -----------------------------------------------------------------------------


class TestWebsockets:
    def test_an_open_socket_needs_nothing(self):
        with _dynamic(_agent()).websocket_connect("/guarded/public") as ws:
            assert ws.receive_json() == {"public": True}

    def test_an_anonymous_upgrade_is_refused_with_the_gates_answer(self):
        with pytest.raises(WebSocketDenialResponse) as refused:
            with _dynamic(_agent()).websocket_connect("/guarded/live"):
                pass
        assert refused.value.status_code == NO_CALLER["status"]
        assert refused.value.json() == NO_CALLER["body"]

    def test_a_caller_outside_the_scope_is_forbidden(self):
        with pytest.raises(WebSocketDenialResponse) as refused:
            with _dynamic(_agent()).websocket_connect("/guarded/live", headers={"X-Test-Caller": "user"}):
                pass
        assert refused.value.status_code == NOT_OPEN["status"]
        assert refused.value.json() == NOT_OPEN["body"]

    def test_a_browser_sends_its_credential_as_token(self):
        with _dynamic(_agent()).websocket_connect("/guarded/live?token=owner-token") as ws:
            assert ws.receive_json() == {"live": True}


# -- the static agent's command routes -------------------------------------------------------


class TestStaticCommands:
    def test_startup_mounts_every_endpoint_without_a_warning(self, capsys):
        WebAgentsServer(agents=[_agent()])
        out = capsys.readouterr().out
        assert "not a valid parameter name" not in out
        assert "Could not mount" not in out
        assert "/guarded/files/{rest:path}" in out

    def test_commands_run_with_the_same_guards_as_on_a_dynamic_agent(self):
        client = _static(_agent())
        # Every command path needs a credential (the floor, S-235).
        assert client.get("/guarded/command").status_code == 401
        listed = client.get("/guarded/command", headers={"Authorization": "Bearer x"}).json()
        assert {c["path"] for c in listed["commands"]} >= {"/owner/thing", "/any/thing"}
        assert client.post("/guarded/command/any/thing", json={}).status_code == 401
        plain = client.post(
            "/guarded/command/any/thing",
            content=b"{}",
            headers={"Authorization": "Bearer x", "Content-Type": "text/plain"},
        )
        assert plain.status_code == 415
        assert client.post("/guarded/command/no/such", json={}, headers={"Authorization": "Bearer x"}).status_code == 404
        ran = client.post("/guarded/command/any/thing", json={}, headers={"Authorization": "Bearer x"})
        assert ran.json() == {"result": "anyone ran"}

    @pytest.mark.parametrize("serve_name", ["static", "dynamic"])
    def test_an_owner_command_runs_for_the_verified_owner_only(self, serve_name):
        # No route passed a context, so an owner command refused its owner too.
        client = (_static if serve_name == "static" else _dynamic)(_agent())
        refused = client.post("/guarded/command/owner/thing", json={}, headers={"Authorization": "Bearer x"})
        assert refused.status_code == 403
        ran = client.post("/guarded/command/owner/thing", json={}, headers={"Authorization": "Bearer owner-token"})
        assert ran.json() == {"result": "owner ran"}

    def test_the_generic_mount_never_serves_the_command_handlers(self):
        # A command handler reached as a plain endpoint would skip every guard.
        routes = [r for r in WebAgentsServer(agents=[_agent()], quiet=True).app.routes if "/command" in getattr(r, "path", "")]
        names = {getattr(r.endpoint, "__name__", "") for r in routes}
        assert not any("execute_command_handler" in n or "get_command_docs_handler" in n for n in names)


# -- the in-tree example: the OpenAI skill's setup form (S-243, S-246) ------------------------


class FakeKV(Skill):
    def __init__(self):
        super().__init__({}, scope="all")
        self.store = {}

    async def kv_set(self, key, value, namespace=None):
        self.store[(namespace, key)] = value

    async def kv_get(self, key, namespace=None):
        return self.store.get((namespace, key))


class TestOpenAISetupForm:
    """The form saves the OpenAI key and workflow the agent's later turns use.
    It answered anyone (S-243); the link to it carried the agent's own platform
    credential and went to every caller (S-246). Now the owner asks for a
    one-time link, and the form takes only its code."""

    def _agent(self):
        from webagents.agents.skills.ecosystem.openai import OpenAIAgentBuilderSkill

        kv = FakeKV()
        skill = OpenAIAgentBuilderSkill({})
        skill.api_key = None  # not from this machine's environment
        agent = BaseAgent(name="guarded", instructions="Guarded.", skills={"openai_workflow": skill, "kv": kv})
        asyncio.run(agent._ensure_skills_initialized())
        agent.api_key = "agent-key-1"
        return agent, skill, kv

    @staticmethod
    def _code(link):
        from urllib.parse import parse_qs, urlsplit

        return parse_qs(urlsplit(link.rsplit(" ", 1)[-1]).query)["setup"][0]

    def test_no_link_no_form(self):
        agent, _, kv = self._agent()
        client = _static(agent)
        assert client.get("/guarded/setup/openai").status_code == 403
        response = client.post("/guarded/setup/openai", data={"api_key": "sk-theirs", "workflow_id": "wf_theirs"})
        assert response.status_code == 403
        # The agent's own key never opens it either.
        response = client.post("/guarded/setup/openai?token=agent-key-1", data={"api_key": "sk-theirs", "workflow_id": "wf_theirs"})
        assert response.status_code == 403
        assert ("openai", "openai_credentials") not in kv.store

    def test_the_owners_link_works_once(self):
        agent, skill, kv = self._agent()
        link = asyncio.run(skill.openai_setup_link())
        assert "agent-key-1" not in link
        code = self._code(link)
        client = _static(agent)
        form = client.get("/guarded/setup/openai", params={"setup": code})
        assert form.status_code == 200
        assert f'name="setup" value="{code}"' in form.text
        saved = client.post("/guarded/setup/openai", data={"api_key": "sk-mine", "workflow_id": "wf_mine", "setup": code})
        assert saved.status_code == 200
        assert json.loads(kv.store[("openai", "openai_credentials")]) == {"api_key": "sk-mine", "workflow_id": "wf_mine"}
        again = client.post("/guarded/setup/openai", data={"api_key": "sk-theirs", "workflow_id": "wf_theirs", "setup": code})
        assert again.status_code == 403
        assert json.loads(kv.store[("openai", "openai_credentials")])["api_key"] == "sk-mine"

    def test_an_expired_link_is_refused(self, monkeypatch):
        import webagents.agents.skills.ecosystem.openai.skill as openai_skill

        agent, skill, _ = self._agent()
        code = self._code(asyncio.run(skill.openai_setup_link()))
        now = openai_skill.time.time()
        monkeypatch.setattr(openai_skill.time, "time", lambda: now + openai_skill.SETUP_LINK_TTL_SECONDS + 1)
        assert _static(agent).get("/guarded/setup/openai", params={"setup": code}).status_code == 403

    def test_only_the_owner_hears_of_it(self):
        agent, skill, _ = self._agent()
        # The prompt is the owner's, the link tool is the owner's, and the tool
        # any caller may use names no link and no key.
        prompts = {p["function"].__name__: p for p in agent._registered_prompts}
        assert prompts["openai_prompt"]["scope"] == "owner"
        tools = {t["name"]: t for t in agent._registered_tools}
        assert tools["openai_setup_link"]["scope"] == "owner"
        answer = asyncio.run(skill.use_openai_workflow())
        assert "http" not in answer and "agent-key-1" not in answer


# -- the real Robutler auth skill and a real platform service token ---------------------------


class TestWithThePlatformAuthSkill:
    """The gate against the actual credential path: the Robutler AuthSkill
    verifying an RS256 service token, which names its sender (S-240)."""

    AGENT_URL = "https://agent.example.com/agents/guarded"
    PLATFORM = "https://robutler.test"

    @pytest.fixture(scope="class")
    def keyring(self, tmp_path_factory):
        from webagents.crypto.jwks import JWKSManager

        manager = JWKSManager({"keys_dir": str(tmp_path_factory.mktemp("keys"))})
        return manager, manager.ensure_keys("test-platform")

    def _token(self, keyring, sender):
        import time

        import jwt as pyjwt

        manager, kid = keyring
        now = int(time.time())
        claims = {
            "sub": "service:robutler-router", "iss": self.PLATFORM, "aud": self.AGENT_URL,
            "iat": now, "exp": now + 300, "sender": {"id": sender},
        }
        return pyjwt.encode(claims, manager.get_signing_key(), algorithm="RS256", headers={"kid": kid})

    def _client(self, keyring):
        import time
        from unittest.mock import MagicMock

        from webagents.agents.skills.robutler.auth.skill import AuthSkill

        manager, _ = keyring
        auth = AuthSkill({
            "require_auth": True,
            "platform_api_url": self.PLATFORM,
            "platform_issuer": self.PLATFORM,
            "agent_url": self.AGENT_URL,
        })
        agent = BaseAgent(name="guarded", instructions="Guarded.", skills={"endpoints": Endpoints(), "auth": auth})
        agent.owner_user_id = "owner-1"
        # What `initialize` would set, without its platform health check.
        auth.agent = agent
        auth.logger = MagicMock()
        auth.client = None
        auth._jwks_keys = [manager.get_public_jwk()]
        auth._jwks_fetched_at = time.monotonic()
        return _static(agent)

    def test_the_owner_the_platform_relays_for_gets_in(self, keyring):
        client = self._client(keyring)
        response = client.get("/guarded/mine", headers={"Authorization": f"Bearer {self._token(keyring, 'owner-1')}"})
        assert response.json() == {"mine": True}

    def test_anyone_else_it_relays_for_is_forbidden(self, keyring):
        client = self._client(keyring)
        response = client.get("/guarded/mine", headers={"Authorization": f"Bearer {self._token(keyring, 'stranger-2')}"})
        assert response.status_code == NOT_OPEN["status"]
        assert response.json() == NOT_OPEN["body"]

    def test_a_credential_the_skill_cannot_verify_gets_its_refusal(self, keyring):
        response = self._client(keyring).get("/guarded/mine", headers={"Authorization": "Bearer made-up"})
        assert response.status_code == 401
        assert response.json() == {
            "error": {
                "code": "unauthorized",
                "message": "Authentication failed (API key, owner assertion, or service token required)",
            }
        }
