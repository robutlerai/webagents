"""
The TrustFlow lookup client (plan item 2.5, 2026-09-26): which route, which
query, which credential, what is held and for how long, and the sentence for
every failure, as the shared contract pins them
(`tests/fixtures/trust/trust_tool_definition.json`; TypeScript
tests/unit/trustflow/trust-lookup-w2trust.test.ts).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import SigningKey
from webagents.trustflow.trust_lookup import (
    NO_TRUST_CREDENTIAL,
    TRUST_CACHE_TTL_MS,
    TRUST_LOOKUP_PATH,
    TRUST_RECORD_PATH,
    PlatformCredential,
    TrustLookup,
    TrustLookupError,
    platform_credential_for,
    trust_failure_message,
)

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "trust" / "trust_tool_definition.json").read_text())
PORTAL = "https://portal.test"
SCOUT = "https://bot.acme.com/agents/scout"


@pytest.fixture
def platform(monkeypatch):
    """A stub platform behind `httpx.AsyncClient`: `answer` is a response, a
    callable returning one, or an exception to raise; every request is recorded."""

    class Platform:
        answer = httpx.Response(404, json={})
        seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        Platform.seen.append(request)
        answer = Platform.answer
        if isinstance(answer, Exception):
            raise answer
        return answer(request) if callable(answer) else answer

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw))
    return Platform


def bearer(**kwargs):
    return TrustLookup(platform_url=PORTAL, credential=PlatformCredential(bearer="k1"), **kwargs)


def test_the_contract():
    assert TRUST_LOOKUP_PATH == FIXTURE["routes"]["lookup"]
    assert TRUST_RECORD_PATH == FIXTURE["routes"]["record"]
    assert TRUST_CACHE_TTL_MS == FIXTURE["cache_ttl_ms"]
    assert NO_TRUST_CREDENTIAL == FIXTURE["no_credential"]
    m = FIXTURE["messages"]
    assert trust_failure_message("unreachable", "@x") == m["unreachable"].replace("{agent}", "@x")
    assert trust_failure_message("not_found", "@x") == m["not_found"].replace("{agent}", "@x")
    assert trust_failure_message("refused", "@x", 503) == m["refused"].replace("{agent}", "@x").replace("{status}", "503")
    assert trust_failure_message("unreadable", "@x") == m["unreadable"].replace("{agent}", "@x")
    assert trust_failure_message("no_credential", "@x") == FIXTURE["no_credential"]


async def test_asks_the_lookup_route_as_a_bearer_and_returns_the_answer_unchanged(platform):
    platform.answer = httpx.Response(200, json=FIXTURE["lookup_response"])
    assert await bearer().lookup(SCOUT, "billing") == FIXTURE["lookup_response"]
    [request] = platform.seen
    assert request.method == "GET"
    assert request.url.path == TRUST_LOOKUP_PATH
    assert request.url.params["agent"] == SCOUT
    assert request.url.params["topic"] == "billing"
    assert request.headers["authorization"] == "Bearer k1"


async def test_leaves_the_topic_out_when_none_is_asked(platform):
    platform.answer = httpx.Response(200, json=FIXTURE["lookup_response"])
    await bearer().lookup("@scout")
    assert str(platform.seen[0].url) == f"{PORTAL}{TRUST_LOOKUP_PATH}?agent=%40scout"


async def test_holds_an_answer_for_the_ttl_per_agent_and_topic(platform):
    platform.answer = httpx.Response(200, json=FIXTURE["lookup_response"])
    clock = {"t": 1_000_000.0}
    client = bearer(now=lambda: clock["t"])
    await client.lookup(SCOUT, "billing")
    await client.lookup(SCOUT, "billing")
    assert len(platform.seen) == 1
    await client.lookup(SCOUT)
    assert len(platform.seen) == 2
    clock["t"] += TRUST_CACHE_TTL_MS + 1
    await client.lookup(SCOUT, "billing")
    assert len(platform.seen) == 3
    client.clear_cache()
    await client.lookup(SCOUT, "billing")
    assert len(platform.seen) == 4


async def test_signs_with_an_identity_that_can_sign_and_sends_no_bearer(platform):
    platform.answer = httpx.Response(200, json=FIXTURE["lookup_response"])
    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    identity = SimpleNamespace(issuer=SCOUT, held_keys=lambda: [key])
    client = TrustLookup(platform_url=PORTAL, credential=platform_credential_for(identity, None))
    await client.lookup("@other")
    [request] = platform.seen
    assert request.headers.get("signature-input")
    assert "authorization" not in request.headers
    assert str(request.url) == f"{PORTAL}{TRUST_LOOKUP_PATH}?agent=%40other"


async def test_is_refused_up_front_with_no_credential_and_dials_nothing(platform):
    client = TrustLookup(platform_url=PORTAL)
    with pytest.raises(TrustLookupError) as raised:
        await client.lookup(SCOUT)
    assert raised.value.code == "no_credential"
    assert str(raised.value) == FIXTURE["no_credential"]
    assert platform.seen == []


@pytest.mark.parametrize(
    "answer, code, message",
    [
        (httpx.Response(404, json={"error": "Agent not found"}), "not_found", FIXTURE["messages"]["not_found"]),
        (httpx.Response(503, json={}), "refused", FIXTURE["messages"]["refused"].replace("{status}", "503")),
        (httpx.ConnectError("refused"), "unreachable", FIXTURE["messages"]["unreachable"]),
        (httpx.Response(200, content=b"not json"), "unreadable", FIXTURE["messages"]["unreadable"]),
        (httpx.Response(200, json={"subject": {}}), "unreadable", FIXTURE["messages"]["unreadable"]),
    ],
    ids=["not_found", "refused", "unreachable", "not json", "no score"],
)
async def test_names_each_failure_and_holds_none_of_them(platform, answer, code, message):
    platform.answer = answer
    client = bearer()
    with pytest.raises(TrustLookupError) as raised:
        await client.lookup(SCOUT)
    assert raised.value.code == code
    assert str(raised.value) == message.replace("{agent}", SCOUT)
    with pytest.raises(TrustLookupError):
        await client.lookup(SCOUT)
    assert len(platform.seen) == 2


async def test_record_asks_the_record_route_and_holds_the_answer(platform):
    answer = {"record": "a.b.c", "payload": {"exp": 1}, "jwks_url": f"{PORTAL}/.well-known/jwks.json"}
    platform.answer = httpx.Response(200, json=answer)
    client = bearer()
    assert await client.record(SCOUT) == answer
    assert await client.record(SCOUT) == answer
    assert len(platform.seen) == 1
    assert platform.seen[0].url.path == TRUST_RECORD_PATH
    assert platform.seen[0].url.params["agent"] == SCOUT


async def test_an_answer_without_a_record_is_unreadable(platform):
    platform.answer = httpx.Response(200, json={"payload": {}})
    with pytest.raises(TrustLookupError) as raised:
        await bearer().record(SCOUT)
    assert raised.value.code == "unreadable"


def test_the_credential_rule():
    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    identity = SimpleNamespace(issuer=SCOUT, held_keys=lambda: [key])
    assert platform_credential_for(identity, "k").kind == "signature"
    assert platform_credential_for(None, " k ") == PlatformCredential(bearer="k")
    assert platform_credential_for(None, None) == PlatformCredential(refusal=NO_TRUST_CREDENTIAL)
    loopback = SimpleNamespace(issuer="http://127.0.0.1:3000/agents/scout", held_keys=lambda: [key])
    assert platform_credential_for(loopback, "k") == PlatformCredential(bearer="k")
    none = platform_credential_for(loopback, None)
    assert none.kind == "none" and "loopback" in none.refusal
