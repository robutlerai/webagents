"""
The trust skill (plan items 2.5 and 2.7, 2026-09-26): the `trust` tool both
SDKs offer (`tests/fixtures/trust/trust_tool_definition.json`; TypeScript
tests/unit/skills/trust/trust-skill-w2trust.test.ts), what it answers, how it
fails, and that an agent file can name it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.trust.trust_skill import TrustSkill
from webagents.cli.agent_builder import SKILL_CLASSES, load_skills
from webagents.crypto.http_signature import SigningKey
from webagents.trustflow.trust_lookup import TrustLookupError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trust"
FIXTURE = json.loads((FIXTURES / "trust_tool_definition.json").read_text())
RECORD_FIXTURE = json.loads((FIXTURES / "trustflow_record.json").read_text())
PORTAL = "https://portal.test"


@pytest.fixture
def platform(monkeypatch):
    class Platform:
        answer = httpx.Response(404, json={})
        seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        Platform.seen.append(request)
        if isinstance(Platform.answer, Exception):
            raise Platform.answer
        return Platform.answer

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw))
    for name in ("WEBAGENTS_API_KEY", "WEBAGENTS_AGENT_TOKEN", "SERVICE_TOKEN", "ROBUTLER_API_URL", "ROBUTLER_INTERNAL_API_URL"):
        monkeypatch.delenv(name, raising=False)
    return Platform


async def skill(**config):
    s = TrustSkill({"robutler_api_url": PORTAL, **config})
    await s.initialize(SimpleNamespace(name="t"))
    return s


def test_offers_the_definition_in_the_shared_fixture():
    trust = TrustSkill({"robutler_api_url": PORTAL, "robutler_api_key": "k"})
    agent = BaseAgent(name="t", instructions="x", skills={"trust": trust})
    offered = [t for t in agent.get_all_tools() if getattr(t["function"], "__self__", None) is trust]
    assert [t["name"] for t in offered] == ["trust"]
    assert offered[0]["definition"] == FIXTURE["definition"]


async def test_answers_the_platform_record_unchanged(platform):
    platform.answer = httpx.Response(200, json=FIXTURE["lookup_response"])
    s = await skill(robutler_api_key="k")
    assert await s.trust(agent="@scout", topic="billing") == FIXTURE["lookup_response"]
    assert str(platform.seen[0].url) == f"{PORTAL}/api/trust/lookup?agent=%40scout&topic=billing"


async def test_says_the_fixture_sentence_when_the_platform_has_no_answer(platform):
    s = await skill(robutler_api_key="k")
    platform.answer = httpx.Response(404, json={"error": "Agent not found"})
    assert await s.trust(agent="@nobody") == {"error": FIXTURE["messages"]["not_found"].replace("{agent}", "@nobody")}
    platform.answer = httpx.ConnectError("refused")
    assert await s.trust(agent="@nobody") == {"error": FIXTURE["messages"]["unreachable"].replace("{agent}", "@nobody")}


async def test_with_neither_identity_nor_key_it_says_so_and_dials_nothing(platform, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent)
    s = await skill()
    assert s.credential().kind == "none"
    assert await s.trust(agent="@scout") == {"error": FIXTURE["no_credential"]}
    assert platform.seen == []


async def test_needs_an_agent_to_look_up(platform):
    s = await skill(robutler_api_key="k")
    assert await s.trust(agent="  ") == {"error": "agent is required: the agent URL, @username or platform id."}


async def test_uses_the_agent_identity_to_sign_when_the_server_attached_one(platform):
    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    identity = SimpleNamespace(issuer="https://agents.example.com/agents/scout", held_keys=lambda: [key])
    s = TrustSkill({"robutler_api_url": PORTAL})
    await s.initialize(SimpleNamespace(name="scout", signing_identity=identity))
    assert s.credential().kind == "signature"


async def test_fetches_this_agent_record_by_its_identity_url_and_verifies_a_record_from_anyone(platform):
    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    identity = SimpleNamespace(issuer="https://agents.example.com/agents/scout", held_keys=lambda: [key])
    answer = {"record": RECORD_FIXTURE["records"]["valid"], "payload": RECORD_FIXTURE["payload"], "jwks_url": f"{PORTAL}/.well-known/jwks.json"}
    platform.answer = httpx.Response(200, json=answer)
    s = TrustSkill({"robutler_api_url": PORTAL, "identity": identity})
    await s.initialize(SimpleNamespace(name="scout"))
    assert await s.record() == answer
    assert platform.seen[0].url.path == "/api/trust/record"
    assert platform.seen[0].url.params["agent"] == "https://agents.example.com/agents/scout"
    verified = await s.verify(answer["record"], keys=RECORD_FIXTURE["jwks"]["keys"], issuer=RECORD_FIXTURE["issuer"], now=RECORD_FIXTURE["now"])
    assert verified.ok


async def test_without_an_identity_there_is_no_own_record_to_fetch(platform):
    s = await skill(robutler_api_key="k")
    with pytest.raises(TrustLookupError) as raised:
        await s.record()
    assert raised.value.code == "no_credential"


def test_the_agent_file_entry_resolves_to_the_skill():
    assert "trust" in SKILL_CLASSES
    loaded = load_skills([{"trust": {"robutler_api_url": PORTAL, "robutler_api_key": "k"}}], "t")
    assert isinstance(loaded["trust"], TrustSkill)
