"""
The access block, end to end through the Python server (ADR-0045).

A caller is placed by what it can PROVE: a Web Bot Auth signature this agent
verifies (the key set here is a stub), or the tier something trusted already
set (the local chat's owner). The model sees only the tools and prompts its
group may use, and a signature that does not verify is a 401, never
"anonymous".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from webagents.access.install import add_access, finish_access
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff, Skill
from webagents.agents.skills.local.rest.skill import RestSkill
from webagents.crypto.http_signature import SigningKey, ed25519_public_jwk, sign_request
from webagents.crypto.web_bot_auth_verify import KeySetOutcome, parse_key_set
from webagents.server.core.app import WebAgentsServer

FRIEND = "https://bot.acme.com/agents/scout"
STRANGER = "https://stranger.example/agents/x"
SPAMMER = "https://spam.example/agents/x"


class RecordingLLM(Skill):
    """A handoff that answers with what it was given: the system text and the tool names."""

    def __init__(self):
        super().__init__()
        self.seen = []

    async def initialize(self, agent):
        await super().initialize(agent)
        agent.register_handoff(
            Handoff(
                target="recording_llm",
                description="Records the turn",
                scope="all",
                metadata={"function": self.completion, "priority": 10, "is_generator": True},
            ),
            source="recording_llm",
        )

    async def completion(self, messages, tools=None, **kwargs):
        system = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "system")
        names = sorted((t.get("function") or {}).get("name", "") for t in (tools or []))
        self.seen.append({"system": system, "tools": names})
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}}]}


class StubKeySets:
    def __init__(self, jwks):
        self.jwks = jwks

    async def get(self, discovery):
        keys, reason = parse_key_set({"keys": self.jwks}, well_known_directory=False)
        return KeySetOutcome(keys=tuple(keys), ttl_s=300) if keys else KeySetOutcome(code="key_set_invalid", reason=reason)


def _key():
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())


KEY = _key()


def _agent(tmp_path: Path, access: dict):
    (tmp_path / "FRIENDS.md").write_text("FRIENDS-ONLY GUIDANCE")
    llm = RecordingLLM()
    skills = {"rest": RestSkill({}), "llm": llm}
    policy = add_access(skills, access, tmp_path / "AGENT.md")
    skills["access"].public_url = "https://agent.example"
    skills["access"].key_sets = StubKeySets([ed25519_public_jwk(KEY.private_key.public_key())])
    agent = BaseAgent(name="mini", instructions="Mini.", skills=skills)
    finish_access(agent, policy, skills)
    return agent, llm


ACCESS = {
    "deny": ["agent:https://spam.example/**"],
    "groups": {"friends": ["agent:https://*.acme.com/**"]},
    "instructions": {"friends": "FRIENDS.md"},
    "tools": {"friends": ["rest"]},
}


def _signed(agent_url: str, body: bytes, *, host="agent.example"):
    signed = sign_request([KEY], agent_url, "POST", f"https://{host}/mini/chat/completions", body)
    return {"host": host, "content-type": "application/json", **signed.headers}


def _post(client, headers, body):
    return client.post("/mini/chat/completions", content=body, headers=headers)


BODY = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()


def test_a_friend_gets_the_friends_tools_and_instructions(tmp_path):
    agent, llm = _agent(tmp_path, ACCESS)
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    response = _post(client, _signed(FRIEND, BODY), BODY)
    assert response.status_code == 200, response.text
    seen = llm.seen[-1]
    assert "rest_request" in seen["tools"]
    assert "FRIENDS-ONLY GUIDANCE" in seen["system"]
    assert f"This turn is from the agent {FRIEND}, which proved it with a Web Bot Auth signature. Groups: friends." in seen["system"]
    assert "## Calling web APIs" in seen["system"]


def test_a_stranger_gets_the_default_group_and_nothing_granted(tmp_path):
    agent, llm = _agent(tmp_path, ACCESS)
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    response = _post(client, _signed(STRANGER, BODY), BODY)
    assert response.status_code == 200, response.text
    seen = llm.seen[-1]
    assert "rest_request" not in seen["tools"]
    assert "FRIENDS-ONLY GUIDANCE" not in seen["system"]
    assert "## Calling web APIs" not in seen["system"]
    assert "Groups: everyone." in seen["system"]


def test_deny_is_a_403(tmp_path):
    agent, llm = _agent(tmp_path, ACCESS)
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    response = _post(client, _signed(SPAMMER, BODY), BODY)
    assert response.status_code == 403
    assert response.json() == {"error": {"code": "forbidden", "message": "This agent does not accept requests from this caller."}}
    assert llm.seen == []


def test_default_none_keeps_out_a_caller_in_no_group(tmp_path):
    agent, llm = _agent(tmp_path, dict(ACCESS, default="none"))
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    assert _post(client, _signed(STRANGER, BODY), BODY).status_code == 403
    # A bearer this agent cannot verify names no one: refused too.
    assert _post(client, {"authorization": "Bearer made-up", "content-type": "application/json"}, BODY).status_code == 403
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200


def test_a_signature_that_does_not_verify_is_a_401_not_anonymous(tmp_path):
    agent, llm = _agent(tmp_path, ACCESS)
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    headers = _signed(FRIEND, BODY)
    other = json.dumps({"messages": [{"role": "user", "content": "something else"}]}).encode()
    response = _post(client, headers, other)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "content_digest_mismatch"
    assert llm.seen == []


def test_a_signature_for_another_host_is_a_401(tmp_path):
    agent, _ = _agent(tmp_path, ACCESS)
    client = TestClient(WebAgentsServer(agents=[agent]).app)
    response = _post(client, _signed(FRIEND, BODY, host="other.example"), BODY)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "signature_authority_mismatch"


def test_the_local_owner_gets_everything(tmp_path):
    from webagents.access import run_as_local_owner

    agent, llm = _agent(tmp_path, ACCESS)

    async def turn():
        run_as_local_owner(agent)
        async for _ in agent.run_streaming([{"role": "user", "content": "hi"}]):
            pass

    asyncio.run(turn())
    seen = llm.seen[-1]
    assert "rest_request" in seen["tools"]
    assert "FRIENDS-ONLY GUIDANCE" in seen["system"]
    assert "This turn is from this agent's owner." in seen["system"]


class TestTheFileLoader:
    def _file(self, tmp_path, access_yaml: str):
        (tmp_path / "AGENT.md").write_text(
            "---\nname: gated\nskills:\n  - rest\n" + access_yaml + "---\nGated.\n"
        )
        return tmp_path / "AGENT.md"

    def test_the_block_scopes_the_named_skills_tools(self, tmp_path):
        from webagents.cli.agent_builder import build_agent

        (tmp_path / "F.md").write_text("friends text")
        path = self._file(
            tmp_path,
            "access:\n  groups:\n    friends: ['agent:https://*.acme.com/**']\n  tools:\n    friends: [rest]\n  instructions:\n    friends: F.md\n",
        )
        built = asyncio.run(build_agent(path, working_dir=tmp_path, initialize=False))
        tools = {t["name"]: t for t in built.agent.get_all_tools()}
        assert tools["rest_request"]["scope"] == ["group:friends"]
        assert "access" in built.agent.skills

    def test_an_unknown_tool_name_stops_the_load(self, tmp_path):
        from webagents.cli.agent_builder import build_agent
        from webagents.cli.loader import AgentFormatError

        path = self._file(tmp_path, "access:\n  groups:\n    friends: []\n  tools:\n    friends: [nope]\n")
        with pytest.raises(AgentFormatError, match='access.tools.friends: "nope" is not a skill in this agent file or one of its tools.'):
            asyncio.run(build_agent(path, working_dir=tmp_path, initialize=False))

    def test_a_missing_instructions_file_stops_the_load(self, tmp_path):
        from webagents.cli.agent_builder import build_agent
        from webagents.cli.loader import AgentFormatError

        path = self._file(tmp_path, "access:\n  instructions:\n    everyone: GONE.md\n")
        with pytest.raises(AgentFormatError, match="access.instructions.everyone: GONE.md was not found next to the agent file."):
            asyncio.run(build_agent(path, working_dir=tmp_path, initialize=False))

    def test_a_malformed_block_stops_the_load_with_its_sentence(self, tmp_path):
        from webagents.cli.loader import AgentFormatError, load_agent

        path = self._file(tmp_path, "access:\n  groupz: {}\n")
        with pytest.raises(AgentFormatError, match='access: unknown key "groupz". It takes deny, groups, default, instructions and tools.'):
            load_agent(path)


class TestServiceTokenPrincipals:
    """S-240 (2026-09-25): a platform service token names its sender to the
    access block only through the platform's signed claim, and only when the
    token is addressed to this agent's own URL. The body's sender names no one."""

    def _auth(self, *, audience_verified, sender=None):
        from webagents.agents.skills.robutler.auth.skill import AuthContext, AuthScope

        claims = {"sub": "service:robutler-router"}
        if sender is not None:
            claims["sender"] = sender
        return AuthContext(
            user_id="from-the-body",
            authenticated=True,
            scope=AuthScope.USER,
            assertion=claims,
            audience_verified=audience_verified,
        )

    def test_a_token_addressed_here_names_its_signed_sender(self):
        from webagents.agents.skills.local.access.skill import _user_principals

        auth = self._auth(audience_verified=True, sender={"id": "user-alice", "username": "alice"})
        assert _user_principals(auth) == ["user:user-alice", "user:@alice"]

    def test_a_token_not_addressed_here_names_no_one(self):
        from webagents.agents.skills.local.access.skill import _user_principals

        auth = self._auth(audience_verified=False, sender={"id": "user-alice", "username": "alice"})
        assert _user_principals(auth) == []

    def test_the_body_sender_names_no_one(self):
        from webagents.agents.skills.local.access.skill import _user_principals

        assert _user_principals(self._auth(audience_verified=True)) == []
