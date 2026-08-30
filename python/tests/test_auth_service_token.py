"""The Python service-token verifier: fail closed, pinned issuer, exact kid,
per-target audience with a one-release no-aud transition window.

Before M4 this path accepted any JWT whose UNVERIFIED sub started with
"service:", fetched JWKS from a URL the portal does not serve (so the fetch
404'd), fell back to keys[0] on a kid mismatch, decoded with audience
verification OFF, and returned blanket ADMIN — lateral admin across every
on-prem agent from one leaked (or forged) bearer.
"""

import time
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest

from webagents.agents.skills.robutler.auth.skill import (
    PLATFORM_FALLBACK_AUDIENCE,
    AuthScope,
    AuthSkill,
)
from webagents.crypto.jwks import JWKSManager

PLATFORM = "https://robutler.test"
AGENT_URL = "https://agent.example.com/agents/mini"


@pytest.fixture(scope="module")
def keyring(tmp_path_factory):
    """A real RSA key + its JWK, via the SDK's own JWKSManager."""
    mgr = JWKSManager({"keys_dir": str(tmp_path_factory.mktemp("keys"))})
    kid = mgr.ensure_keys("test-platform")
    return mgr, kid


def make_skill(keyring, **config):
    mgr, kid = keyring
    skill = AuthSkill({
        "require_auth": True,
        "platform_api_url": PLATFORM,
        "platform_issuer": PLATFORM,
        "agent_url": AGENT_URL,
        **config,
    })
    skill.logger = MagicMock()
    skill.agent = MagicMock()
    skill.agent.owner_user_id = "owner-1"
    # Warm the JWKS cache with the platform key — the loader must never be
    # given a reason to fall back to anything else.
    skill._jwks_keys = [mgr.get_public_jwk()]
    skill._jwks_fetched_at = time.monotonic()
    return skill


def sign(keyring, *, sub="service:robutler-router", iss=PLATFORM, aud=None,
         kid_override=None, alg="RS256", key=None, ttl=300):
    mgr, kid = keyring
    now = int(time.time())
    payload = {"sub": sub, "iss": iss, "iat": now, "exp": now + ttl, "scopes": ["agents:*"]}
    if aud is not None:
        payload["aud"] = aud
    return pyjwt.encode(
        payload,
        key if key is not None else mgr.get_signing_key(),
        algorithm=alg,
        headers={"kid": kid_override or kid},
    )


class TestServiceTokenVerifier:
    async def test_valid_no_aud_token_is_user_scoped_never_admin(self, keyring):
        skill = make_skill(keyring)
        ctx = await skill._authenticate_service_token(sign(keyring))
        assert ctx is not None and ctx.authenticated
        assert ctx.scope is AuthScope.USER
        assert ctx.scope is not AuthScope.ADMIN

    async def test_valid_token_with_matching_audience(self, keyring):
        skill = make_skill(keyring)
        ctx = await skill._authenticate_service_token(sign(keyring, aud=AGENT_URL))
        assert ctx is not None and ctx.authenticated

    async def test_wrong_audience_is_always_refused(self, keyring):
        """A token minted for ANOTHER agent's URL must never authenticate
        here, transition window or not."""
        skill = make_skill(keyring)
        token = sign(keyring, aud="https://other-agent.example.net")
        assert await skill._authenticate_service_token(token) is None

    async def test_platform_fallback_audience_is_accepted(self, keyring):
        """`getServiceToken()` with no target mints
        SERVICE_TOKEN_FALLBACK_AUD (lib/agents/router.ts) — the voice relay's
        outbound leg does exactly that. Refusing it stops a legitimate
        platform leg from authenticating at all."""
        skill = make_skill(keyring)
        token = sign(keyring, aud=PLATFORM_FALLBACK_AUDIENCE)
        ctx = await skill._authenticate_service_token(token)
        assert ctx is not None and ctx.authenticated

    async def test_public_url_trailing_slash_still_matches(self, keyring):
        """The platform rstrips the audience it mints; a WEBAGENTS_PUBLIC_URL
        with a trailing slash used to refuse every token, silently."""
        skill = make_skill(keyring, agent_url=AGENT_URL + "/")
        ctx = await skill._authenticate_service_token(sign(keyring, aud=AGENT_URL))
        assert ctx is not None and ctx.authenticated

    async def test_no_aud_refused_once_transition_flag_is_on(self, keyring):
        skill = make_skill(keyring, require_service_aud="1")
        assert await skill._authenticate_service_token(sign(keyring)) is None

    async def test_wrong_issuer_refused(self, keyring):
        skill = make_skill(keyring)
        token = sign(keyring, iss="https://attacker.example")
        assert await skill._authenticate_service_token(token) is None

    async def test_kid_mismatch_refused_no_first_key_fallback(self, keyring, monkeypatch):
        """The old code took keys[0] whenever the kid did not match — key
        rotation became a verification bypass. An unknown kid (after one
        refetch) must yield NO key and NO auth."""
        skill = make_skill(keyring)

        async def refetch_misses(kid):
            return None

        # Simulate the refetch also missing (the exact-match loader returns
        # None rather than any listed key).
        monkeypatch.setattr(skill, "_load_platform_jwk", refetch_misses)
        token = sign(keyring, kid_override="rotated-away")
        assert await skill._authenticate_service_token(token) is None

    async def test_exact_kid_required_by_loader(self, keyring, monkeypatch):
        """_load_platform_jwk returns only the EXACT kid; a miss refetches
        once and then gives up — never keys[0]."""
        mgr, kid = keyring
        skill = make_skill(keyring)
        fetches = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"keys": [mgr.get_public_jwk()]}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                fetches.append(url)
                return FakeResponse()

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        skill._jwks_keys = None  # cold cache

        assert await skill._load_platform_jwk(kid) is not None
        assert await skill._load_platform_jwk("not-a-kid") is None
        assert await skill._load_platform_jwk(None) is None
        # The JWKS endpoint is the one the portal actually serves.
        assert all(u == f"{PLATFORM}/.well-known/jwks.json" for u in fetches)

    async def test_non_service_sub_refused(self, keyring):
        skill = make_skill(keyring)
        assert await skill._authenticate_service_token(sign(keyring, sub="user-1")) is None

    async def test_hs256_refused(self, keyring):
        skill = make_skill(keyring)
        token = sign(keyring, alg="HS256", key="a" * 40)
        assert await skill._authenticate_service_token(token) is None

    async def test_expired_refused(self, keyring):
        skill = make_skill(keyring)
        assert await skill._authenticate_service_token(sign(keyring, ttl=-60)) is None

    async def test_sender_metadata_attributes_and_elevates_owner(self, keyring, monkeypatch):
        """The platform relays a chat turn for metadata.sender; the auth
        context belongs to that sender, OWNER only when they own this agent."""
        skill = make_skill(keyring)
        monkeypatch.setattr(skill, "_extract_platform_sender_id", lambda: "owner-1")
        ctx = await skill._authenticate_service_token(sign(keyring))
        assert ctx is not None
        assert ctx.user_id == "owner-1"
        assert ctx.scope is AuthScope.OWNER

        monkeypatch.setattr(skill, "_extract_platform_sender_id", lambda: "stranger-2")
        ctx = await skill._authenticate_service_token(sign(keyring))
        assert ctx is not None
        assert ctx.user_id == "stranger-2"
        assert ctx.scope is AuthScope.USER
