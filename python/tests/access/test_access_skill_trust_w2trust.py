"""
Trust-gated groups end to end through the Python server (plan item 2.5,
2026-09-26); the same cases as TypeScript's
tests/unit/access/access-skill-trust-w2trust.test.ts.

The platform is a stub on the skill (`trust`): the access skill must ask it
about the VERIFIED calling agent (the one the Web Bot Auth signature proved),
on the topic the block names, and place the caller by the answer. When the
stub raises, the platform is "unreachable" and the trust-gated group is not
joined: the caller keeps its other groups or the default, and `default: none`
refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path

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


class RecordingLLM(Skill):
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


class StubPlatform:
    """A platform that answers per agent and topic, records what it was asked, or raises."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    async def lookup(self, agent, topic=None):
        self.asked.append((agent, topic))
        if self.answers == "unreachable":
            raise ConnectionError("ECONNREFUSED")
        key = f"{agent}|{topic or ''}"
        if key not in self.answers:
            raise LookupError("not found")
        score = self.answers[key]
        return {"score": 0.1, "topic": {"query": topic, "score": score}} if topic else {"score": score if score is not None else 0}


KEY = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())

ACCESS = {
    "groups": {
        "friends": ["agent:https://*.acme.com/**"],
        "billing": {"trust": {"min": 0.6, "topic": "billing"}},
    },
    "tools": {"billing": ["rest"]},
}


def _agent(tmp_path: Path, access: dict, platform):
    llm = RecordingLLM()
    skills = {"rest": RestSkill({}), "llm": llm}
    policy = add_access(skills, access, tmp_path / "AGENT.md")
    skills["access"].public_url = "https://agent.example"
    skills["access"].key_sets = StubKeySets([ed25519_public_jwk(KEY.private_key.public_key())])
    skills["access"].trust = platform
    agent = BaseAgent(name="mini", instructions="Mini.", skills=skills)
    finish_access(agent, policy, skills)
    return TestClient(WebAgentsServer(agents=[agent]).app), llm


def _signed(agent_url: str, body: bytes, *, host="agent.example"):
    signed = sign_request([KEY], agent_url, "POST", f"https://{host}/mini/chat/completions", body)
    return {"host": host, "content-type": "application/json", **signed.headers}


def _post(client, headers, body):
    return client.post("/mini/chat/completions", content=body, headers=headers)


BODY = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()


def test_a_verified_agent_above_the_threshold_joins_the_group_and_gets_its_tools(tmp_path):
    platform = StubPlatform({f"{FRIEND}|billing": 0.7})
    client, llm = _agent(tmp_path, ACCESS, platform)
    response = _post(client, _signed(FRIEND, BODY), BODY)
    assert response.status_code == 200, response.text
    seen = llm.seen[-1]
    assert "rest_request" in seen["tools"]
    assert "Groups: friends, billing." in seen["system"]
    # Asked about the agent the SIGNATURE proved, on the block's topic, and nothing else.
    assert platform.asked == [(FRIEND, "billing")]


def test_below_the_threshold_the_caller_keeps_only_its_identity_groups(tmp_path):
    client, llm = _agent(tmp_path, ACCESS, StubPlatform({f"{FRIEND}|billing": 0.59}))
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200
    assert "rest_request" not in llm.seen[-1]["tools"]
    assert "Groups: friends." in llm.seen[-1]["system"]


def test_the_platform_unreachable_fails_closed_the_rest_untouched(tmp_path):
    platform = StubPlatform("unreachable")
    client, llm = _agent(tmp_path, ACCESS, platform)
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200
    assert "rest_request" not in llm.seen[-1]["tools"]
    assert "Groups: friends." in llm.seen[-1]["system"]
    assert platform.asked == [(FRIEND, "billing")]


def test_a_topic_the_platform_could_not_score_does_not_admit(tmp_path):
    client, llm = _agent(tmp_path, ACCESS, StubPlatform({f"{FRIEND}|billing": None}))
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200
    assert "Groups: friends." in llm.seen[-1]["system"]


def test_default_none_with_only_a_trust_group_refuses_when_unreachable(tmp_path):
    closed = {"groups": {"billing": {"trust": {"min": 0.6, "topic": "billing"}}}, "default": "none"}
    client, llm = _agent(tmp_path, closed, StubPlatform("unreachable"))
    assert _post(client, _signed(STRANGER, BODY), BODY).status_code == 403
    assert llm.seen == []
    client, llm = _agent(tmp_path, closed, StubPlatform({f"{STRANGER}|billing": 0.8}))
    assert _post(client, _signed(STRANGER, BODY), BODY).status_code == 200
    assert "Groups: billing." in llm.seen[-1]["system"]


def test_a_caller_with_no_verified_agent_is_never_looked_up(tmp_path):
    platform = StubPlatform({f"{FRIEND}|billing": 0.9})
    client, llm = _agent(tmp_path, ACCESS, platform)
    # A bearer this agent cannot verify names no one (the existing access suite's anonymous shape).
    assert _post(client, {"authorization": "Bearer made-up", "content-type": "application/json"}, BODY).status_code == 200
    assert "Groups: everyone." in llm.seen[-1]["system"]
    assert platform.asked == []


def test_a_block_without_trust_groups_asks_the_platform_nothing(tmp_path):
    platform = StubPlatform({f"{FRIEND}|billing": 0.9})
    client, _ = _agent(tmp_path, {"groups": {"friends": ["agent:https://*.acme.com/**"]}}, platform)
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200
    assert platform.asked == []


def test_the_lookup_is_made_once_per_topic_key_not_once_per_group(tmp_path):
    platform = StubPlatform({f"{FRIEND}|billing": 0.9, f"{FRIEND}|": 0.5})
    access = {
        "groups": {
            "a": {"trust": {"min": 0.1, "topic": "billing"}},
            "b": {"trust": {"min": 0.2, "topic": "billing"}},
            "c": {"members": ["agent:https://*.acme.com/**"], "trust": {"min": 0.4}},
        }
    }
    client, llm = _agent(tmp_path, access, platform)
    assert _post(client, _signed(FRIEND, BODY), BODY).status_code == 200
    assert len(platform.asked) == 2
    assert set(platform.asked) == {(FRIEND, None), (FRIEND, "billing")}
    assert "Groups: a, b, c." in llm.seen[-1]["system"]
