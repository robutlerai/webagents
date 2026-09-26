"""DiscoverySkill signs with the agent's identity and needs no platform key
(2026-09-23).

The platform's `authenticateAgentRequest` takes an RFC 9421 signed request
before it looks for a bearer, on every route this skill calls. Until this
change the skill only ever sent `Authorization: Bearer <WEBAGENTS_API_KEY>`,
so an agent that already held a signing identity was told to obtain a key for
a call the platform would have accepted signed. Pinned here, in order:

  1. an identity signs, and no bearer rides beside the signature;
  2. the identity is the one `create_server` serves: the key under
     WEBAGENTS_KEYS_DIR for this agent name, LOADED and never minted, and the
     agent URL composed exactly as the server composes it;
  3. a configured key is still presented, and still works, when there is no
     identity, or when the identity cannot sign (loopback);
  4. neither is refused up front with a sentence that names both fixes.

The requests are signed for real and captured by an `httpx.MockTransport`,
and every signature is verified the way the platform verifies it (rebuild the
base from the request as sent, check it under the published key), so a
regression in WHAT is signed cannot pass.
"""

import logging

import httpx
import pytest

from webagents.agents.skills.robutler.discovery.skill import (
    NO_DISCOVERY_CREDENTIAL,
    NO_DISCOVERY_SIGN_IN,
    DiscoverySkill,
)
from webagents.crypto.jwks import JWKSManager

from .crypto.support import verify_signed_request

PLATFORM = "https://platform.example.com"
PUBLIC_URL = "https://agent.example.com"
AGENT_URL = f"{PUBLIC_URL}/translator"
KEY_SET = f"{AGENT_URL}/.well-known/jwks.json"

#: One row as `POST /api/intents/search` answers it; the tool hands rows on as they are.
INTENT_ROW = {
    "id": "i-1",
    "intent": "translate legal documents into German",
    "agentId": "a-jurist",
    "description": "Certified legal translation",
    "url": "https://legal.example.com/agents/jurist",
    "similarity": 0.8712,
}


class FakeAgent:
    """What the skill reads off the agent: its name (for the key file and
    the principal) and, here, no `api_key` and no `intents`."""

    name = "translator"
    api_key = None
    intents = None


@pytest.fixture
def platform(monkeypatch):
    """A stub platform behind `httpx.AsyncClient`: records every request and
    answers the two routes the skill calls. Scrubs every credential variable
    so the test decides what the skill holds."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/intents/search":
            return httpx.Response(200, json={"results": [INTENT_ROW]})
        if request.url.path == "/api/discovery/announce":
            return httpx.Response(200, json={"ok": True, "intents": {"count": 1}})
        return httpx.Response(404, json={"error": "not found"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handler), **kw)
    )
    for name in ("WEBAGENTS_API_KEY", "SERVICE_TOKEN", "WEBAGENTS_PUBLIC_URL", "ROBUTLER_API_KEY", "ROBUTLER_INTERNAL_API_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROBUTLER_API_URL", PLATFORM)
    return seen


async def skill_with(monkeypatch, tmp_path, *, public_url=PUBLIC_URL, key=None, served_key=True, config=None):
    """A skill attached to `FakeAgent`, with the identity halves the test asks
    for: `public_url` (the agent URL's base, None for unset) and `served_key`
    (whether `create_server` has written the agent's key under
    WEBAGENTS_KEYS_DIR). Returns the skill and the public keys by thumbprint."""
    keys_dir = tmp_path / "keys"
    monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(keys_dir))
    if public_url is not None:
        monkeypatch.setenv("WEBAGENTS_PUBLIC_URL", public_url)
    public = {}
    if served_key:
        # What `create_server` does at start (`_register_well_known`).
        manager = JWKSManager({"keys_dir": str(keys_dir)})
        thumbprint = manager.ensure_ed25519_key(FakeAgent.name)
        public[thumbprint] = manager.get_ed25519_signing_key().public_key()
    cfg = dict(config or {})
    if key is not None:
        cfg["robutler_api_key"] = key
    skill = DiscoverySkill(cfg)
    await skill.initialize(FakeAgent())
    return skill, public


def assert_signed(request: httpx.Request, public: dict, key_set: str = KEY_SET) -> dict:
    """The request carries a verifying signature naming `key_set`, and no bearer."""
    assert "authorization" not in request.headers, "a bearer rode beside the signature"
    assert request.headers["signature-agent"] == f'sig1="{key_set}";type=jwks_uri'
    verified = verify_signed_request(request.headers, request.method, str(request.url), request.content, public)
    record = verified["sig1"]
    assert record["keyid"] in public
    assert record["tag"] == "web-bot-auth"
    assert record["expires"] - record["created"] == 60
    return record


def assert_bearer(request: httpx.Request, key: str) -> None:
    assert request.headers["authorization"] == f"Bearer {key}"
    for name in ("signature-input", "signature", "signature-agent", "content-digest"):
        assert name not in request.headers, name


# ---------------------------------------------------------------------------
# 1. An identity signs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signs_the_search_with_the_served_identity_and_sends_no_bearer(platform, monkeypatch, tmp_path):
    skill, public = await skill_with(monkeypatch, tmp_path)
    assert skill.credential().kind == "signature"

    result = await skill.search(query="translate a contract into German", types=["intents"])

    assert result == {"intents": [INTENT_ROW]}, result
    (request,) = platform
    assert request.method == "POST"
    assert str(request.url) == f"{PLATFORM}/api/intents/search"
    # `JSON.stringify`'s bytes, so the signed body is the TypeScript skill's.
    assert request.content == b'{"query":"translate a contract into German","limit":10}'
    record = assert_signed(request, public)
    lines = record["base"].split("\n")
    assert lines[0] == '"@method": POST'
    assert lines[1] == '"@authority": platform.example.com'
    assert lines[2] == '"@path": /api/intents/search'


@pytest.mark.asyncio
async def test_signs_rather_than_sending_the_key_when_it_has_both(platform, monkeypatch, tmp_path):
    skill, public = await skill_with(monkeypatch, tmp_path, key="rok_configured")
    assert skill.credential().kind == "signature"

    await skill.search(query="translate", types=["intents"])

    (request,) = platform
    assert_signed(request, public)


@pytest.mark.asyncio
async def test_the_principal_is_composed_as_the_server_composes_it(platform, monkeypatch, tmp_path):
    # `create_server(url_prefix="/agents")` mounts the agent at
    # `/agents/{name}`; `agent_path` is how the skill learns the prefix.
    skill, public = await skill_with(monkeypatch, tmp_path, config={"agent_path": "/agents"})
    await skill.search(query="translate", types=["intents"])
    assert_signed(platform[0], public, key_set=f"{PUBLIC_URL}/agents/translator/.well-known/jwks.json")


@pytest.mark.asyncio
async def test_an_explicit_agent_url_overrides_the_composed_principal(platform, monkeypatch, tmp_path):
    skill, public = await skill_with(
        monkeypatch, tmp_path, config={"agent_url": "https://Agents.Example.com:443/agents/other/"}
    )
    await skill.search(query="translate", types=["intents"])
    # Canonical spelling: host lowercased, default port dropped, no trailing slash.
    assert_signed(platform[0], public, key_set="https://agents.example.com/agents/other/.well-known/jwks.json")


@pytest.mark.asyncio
async def test_publish_signs_the_announce_and_announces_the_signed_principal(platform, monkeypatch, tmp_path):
    skill, public = await skill_with(monkeypatch, tmp_path)

    result = await skill.publish_intents(
        intents=["translate documents between English and German"], description="Translates text."
    )

    assert result["success"] is True, result
    (request,) = platform
    assert str(request.url) == f"{PLATFORM}/api/discovery/announce"
    assert_signed(request, public)
    import json

    body = json.loads(request.content)
    # The endpoint announced is the URL the signature names, which is where
    # the platform just fetched the key set from: the agent's own mount, not
    # the bare origin WEBAGENTS_PUBLIC_URL holds.
    assert body["url"] == AGENT_URL
    assert body["intents"] == [
        {"intent": "translate documents between English and German", "description": "Translates text."}
    ]
    assert result["agent_url"] == AGENT_URL


@pytest.mark.asyncio
async def test_auto_publish_on_initialize_signs_with_the_served_identity(platform, monkeypatch, tmp_path):
    monkeypatch.setattr(DiscoverySkill, "AUTO_PUBLISH_DELAY_S", 0)
    skill, public = await skill_with(
        monkeypatch, tmp_path, config={"intents": ["proofread German correspondence"], "description": "Proofreads."}
    )
    assert skill._auto_publish_task is not None, "no auto-publish was scheduled"

    await skill._auto_publish_task

    (request,) = platform
    assert str(request.url) == f"{PLATFORM}/api/discovery/announce"
    assert_signed(request, public)


# ---------------------------------------------------------------------------
# 2. The key is loaded, never minted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_public_url_without_a_served_key_is_not_an_identity(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(monkeypatch, tmp_path, served_key=False)

    credential = skill.credential()
    assert credential.kind == "none"
    assert credential.refusal.startswith("This agent cannot sign its platform calls: no Ed25519 key for 'translator'")
    assert str(tmp_path / "keys") in credential.refusal
    # Nothing was written: the skill piggybacks on the server's identity and
    # never creates one of its own.
    assert not (tmp_path / "keys").exists() or not any((tmp_path / "keys").iterdir())

    # With a key it falls back to the bearer.
    with_key, _ = await skill_with(monkeypatch, tmp_path, served_key=False, key="rok_configured")
    await with_key.search(query="translate", types=["intents"])
    assert_bearer(platform[0], "rok_configured")


# ---------------------------------------------------------------------------
# 3. A key is a bearer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presents_the_configured_key_when_there_is_no_identity(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(monkeypatch, tmp_path, public_url=None, served_key=False, key="rok_configured")
    assert skill.credential().kind == "bearer"

    result = await skill.search(query="translate", types=["intents"])

    assert result == {"intents": [INTENT_ROW]}, result
    assert_bearer(platform[0], "rok_configured")


@pytest.mark.asyncio
async def test_takes_the_key_from_webagents_api_key(platform, monkeypatch, tmp_path):
    monkeypatch.setenv("WEBAGENTS_API_KEY", "rok_from_env")
    # A public URL but no served key for this agent: no identity, so the
    # environment's key is the credential.
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url="https://agent.example.com/agents/mini", served_key=False
    )
    assert skill.credential().kind == "bearer"

    result = await skill.publish_intents(intents=["translate"], description="d")

    assert result["success"] is True, result
    (request,) = platform
    assert_bearer(request, "rok_from_env")
    # With a bearer the announced endpoint is what it always was: the
    # WEBAGENTS_PUBLIC_URL value, verbatim.
    import json

    assert json.loads(request.content)["url"] == "https://agent.example.com/agents/mini"


@pytest.mark.asyncio
async def test_falls_back_to_the_key_when_the_identity_cannot_sign_and_says_why_when_it_cannot(
    platform, monkeypatch, tmp_path
):
    # What an agent has with WEBAGENTS_PUBLIC_URL pointing at loopback: a key
    # the server serves, at an address the platform refuses by name.
    with_key, _ = await skill_with(monkeypatch, tmp_path, public_url="http://localhost:8000", key="rok_configured")
    assert with_key.credential().kind == "bearer"
    await with_key.search(query="translate", types=["intents"])
    assert_bearer(platform[0], "rok_configured")

    without_key, _ = await skill_with(monkeypatch, tmp_path, public_url="http://localhost:8000")
    credential = without_key.credential()
    assert credential.kind == "none"
    assert "WEBAGENTS_PUBLIC_URL" in credential.refusal
    assert "loopback" in credential.refusal


# ---------------------------------------------------------------------------
# 4. Neither is refused, naming the fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_naming_both_fixes_and_dials_nothing(platform, monkeypatch, tmp_path):
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    # The skill's own logger, which `initialize` warns on, exactly once.
    target = logging.getLogger("webagents.skill.webagents.discovery")
    handler = Collect(level=logging.WARNING)
    target.addHandler(handler)
    try:
        skill, _ = await skill_with(monkeypatch, tmp_path, public_url=None, served_key=False)
    finally:
        target.removeHandler(handler)

    credential = skill.credential()
    assert credential.kind == "none"
    # The sentence both SDKs answer, naming the ways a person at the CLI has.
    expected = NO_DISCOVERY_CREDENTIAL
    assert credential.refusal == expected
    assert "`webagents publish`" in expected and "WEBAGENTS_PUBLIC_URL" in expected and "WEBAGENTS_AGENT_TOKEN" in expected
    assert [r.getMessage() for r in records if expected in r.getMessage()], records

    # The TypeScript skill's answer: the refusal and nothing else.
    searched = await skill.search(query="translate", types=["intents"])
    assert searched == {"error": expected}
    published = await skill.publish_intents(intents=["translate"], description="d")
    assert published == {"success": False, "error": expected}
    assert platform == [], "a call was dialled with no credential"


@pytest.mark.asyncio
async def test_a_whitespace_only_key_is_no_key(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(monkeypatch, tmp_path, public_url=None, served_key=False, key="   ")
    assert skill.credential().kind == "none"


# ---------------------------------------------------------------------------
# 5. In the chat, the signed-in person (2026-09-25)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_presents_the_persons_token_when_the_agent_has_no_credential(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url=None, served_key=False, config={"person_token": lambda: "person-token"}
    )

    result = await skill.search(query="translate", types=["intents"])

    assert result == {"intents": [INTENT_ROW]}, result
    assert_bearer(platform[0], "person-token")


@pytest.mark.asyncio
async def test_the_agents_own_key_comes_first(platform, monkeypatch, tmp_path):
    asked = []
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url=None, served_key=False, key="rok_agent",
        config={"person_token": lambda: asked.append(1) or "person-token"},
    )

    await skill.search(query="translate", types=["intents"])

    assert_bearer(platform[0], "rok_agent")
    assert asked == []


@pytest.mark.asyncio
async def test_the_person_stands_in_for_an_identity_that_cannot_sign_from_loopback(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url="http://localhost:8000", config={"person_token": lambda: "person-token"}
    )

    await skill.search(query="translate", types=["intents"])

    assert_bearer(platform[0], "person-token")


@pytest.mark.asyncio
async def test_nobody_signed_in_is_the_chats_sentence_and_nothing_is_dialled(platform, monkeypatch, tmp_path):
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url=None, served_key=False, config={"person_token": lambda: None}
    )

    assert await skill.search(query="translate", types=["intents"]) == {"error": NO_DISCOVERY_SIGN_IN}
    assert platform == []


@pytest.mark.asyncio
async def test_publishing_intents_never_speaks_as_the_person(platform, monkeypatch, tmp_path):
    asked = []
    skill, _ = await skill_with(
        monkeypatch, tmp_path, public_url=None, served_key=False,
        config={"person_token": lambda: asked.append(1) or "person-token"},
    )

    published = await skill.publish_intents(intents=["translate"], description="d")

    assert published == {"success": False, "error": NO_DISCOVERY_CREDENTIAL}
    assert asked == [] and platform == []


def test_only_the_chat_and_dash_p_hand_discovery_the_person():
    """`load_skills` gives `discovery` the person only when asked, and `serve`
    (every caller would search as the owner) never asks."""
    from pathlib import Path as _Path

    from webagents.cli.agent_builder import load_skills

    def person():
        return "person-token"

    assert load_skills(["discovery"], agent_name="a", person_token=person)["discovery"].person_token is person
    assert load_skills(["discovery"], agent_name="a")["discovery"].person_token is None
    cli = _Path(__file__).resolve().parents[1] / "webagents" / "cli"
    assert "person_token=get_token" in (cli / "repl" / "session.py").read_text()
    assert "person_token=get_token" in (cli / "one_shot.py").read_text()
    for host in ("serve.py", "doctor.py"):
        assert "person_token" not in (cli / host).read_text(), host
