"""
`claim_url`: the link a person opens to take ownership of an agent, the twin
of the TypeScript SDK's `claimUrl` (2026-09-19).

Python had only `JWKSManager.mint_claim_token`, which left building the link
to each operator, and with it the one thing that matters about the link: a
claim token is a bearer until it is spent, so it goes in the URL FRAGMENT,
which a browser never sends, and never in the query, which reaches the
platform's access logs and any `Referer`. What is pinned here is that
property, from the outside: what a server would be sent for this URL carries
no token. The shape and the lifetime are the TypeScript ones
(`typescript/tests/unit/server/claim-url.test.ts` pins the same).
"""

import json
from urllib.parse import urlsplit

import jwt as pyjwt
import pytest

from webagents.crypto.jwks import JWKSManager
from webagents.server.core.registration import CLAIM_URL_TTL_SECONDS, claim_url

AGENT_NAME = "mini"
AGENT_USER_ID = "3f0c2a9e-5b7d-4c1e-9a66-0d2f8e1b7c44"
AGENT_URL = "https://agent.example.com/agents/mini"
PLATFORM = "https://robutler.ai"


@pytest.fixture(autouse=True)
def no_platform_in_the_environment(monkeypatch):
    monkeypatch.delenv("ROBUTLER_API_URL", raising=False)
    monkeypatch.delenv("ROBUTLER_INTERNAL_API_URL", raising=False)


@pytest.fixture
def identity(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    manager.ensure_ed25519_key(AGENT_NAME)
    return manager


def _claims(identity, token, audience=f"{PLATFORM}/claim"):
    # What the claim route does: select the key by `kid` from the key set, EdDSA only.
    jwk = identity.get_ed25519_public_jwk()
    assert pyjwt.get_unverified_header(token) == {"alg": "EdDSA", "kid": jwk["kid"], "typ": "JWT"}
    key = pyjwt.algorithms.OKPAlgorithm.from_jwk(json.dumps(jwk))
    return pyjwt.decode(token, key, algorithms=["EdDSA"], audience=audience)


def test_the_link_is_platform_claim_id_with_the_token_in_the_fragment(identity):
    url = claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)
    assert url is not None
    prefix, token = url.split("#")  # exactly one `#`
    assert prefix == f"{PLATFORM}/claim/{AGENT_USER_ID}"

    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path, parts.query) == ("https", "robutler.ai", f"/claim/{AGENT_USER_ID}", "")
    assert parts.fragment == token

    claims = _claims(identity, token)
    assert claims["sub"] == AGENT_NAME
    assert claims["aud"] == f"{PLATFORM}/claim"
    assert claims["scope"] == "agent:claim"
    assert claims["jti"]


def test_nothing_a_server_is_sent_for_the_link_carries_the_token(identity):
    url = claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)
    token = urlsplit(url).fragment
    # A browser sends the request target and the host, never the fragment:
    # this is everything an access log or a `Referer` can ever hold.
    sent = urlsplit(url)._replace(fragment="").geturl()
    assert sent == f"{PLATFORM}/claim/{AGENT_USER_ID}"
    assert token not in sent
    for part in token.split("."):
        assert part not in sent


def test_the_lifetime_is_ten_minutes_unless_the_caller_says_otherwise(identity):
    assert CLAIM_URL_TTL_SECONDS == 600

    def lifetime(**kw):
        claims = _claims(identity, urlsplit(claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM, **kw)).fragment)
        return claims["exp"] - claims["iat"]

    assert lifetime() == 600
    assert lifetime(ttl_seconds=None) == 600  # TypeScript's `ttlSeconds ?? 600`
    assert lifetime(ttl_seconds=60) == 60


def test_trailing_slashes_on_the_platform_url_change_neither_the_link_nor_the_audience(identity):
    url = claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=f"{PLATFORM}///")
    assert url.startswith(f"{PLATFORM}/claim/{AGENT_USER_ID}#")
    assert _claims(identity, urlsplit(url).fragment)["aud"] == f"{PLATFORM}/claim"


def test_the_agent_url_is_the_issuer_when_given_and_absent_otherwise(identity):
    with_issuer = claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM, agent_url=AGENT_URL + "/")
    assert _claims(identity, urlsplit(with_issuer).fragment)["iss"] == AGENT_URL
    without = claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)
    assert "iss" not in _claims(identity, urlsplit(without).fragment)


def test_every_link_is_its_own_single_use_token(identity):
    first = _claims(identity, urlsplit(claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)).fragment)
    second = _claims(identity, urlsplit(claim_url(identity, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)).fragment)
    assert first["jti"] != second["jti"]


class _Recording:
    """An identity that records what it was asked to mint."""

    def __init__(self):
        self.calls = []

    def mint_claim_token(self, agent_id, platform_url, ttl_seconds=600, *, issuer=None):
        self.calls.append((agent_id, platform_url, ttl_seconds, issuer))
        return "header.payload.signature"


def test_the_platform_comes_from_the_argument_then_the_environment_as_registration_resolves_it(monkeypatch):
    identity = _Recording()
    monkeypatch.setenv("ROBUTLER_INTERNAL_API_URL", "https://internal.example/")
    assert claim_url(identity, AGENT_NAME, "u-1") == "https://internal.example/claim/u-1#header.payload.signature"
    monkeypatch.setenv("ROBUTLER_API_URL", "https://env.example")
    assert claim_url(identity, AGENT_NAME, "u-1") == "https://env.example/claim/u-1#header.payload.signature"
    assert claim_url(identity, AGENT_NAME, "u-1", platform_url=PLATFORM) == f"{PLATFORM}/claim/u-1#header.payload.signature"
    # The audience is minted from the same base the link is built on, slash stripped.
    assert [call[1] for call in identity.calls] == ["https://internal.example", "https://env.example", PLATFORM]


def test_no_platform_url_is_no_link_and_nothing_is_minted():
    identity = _Recording()
    assert claim_url(identity, AGENT_NAME, AGENT_USER_ID) is None
    assert identity.calls == []


def test_an_identity_without_its_key_is_a_clear_error_not_a_link(tmp_path):
    manager = JWKSManager({"keys_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="ensure_ed25519_key"):
        claim_url(manager, AGENT_NAME, AGENT_USER_ID, platform_url=PLATFORM)
