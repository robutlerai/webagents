"""
`JWKSManager.mint_claim_token`: the one JWT the SDK still mints (ADR 0038
step 5, W2 design sections 7.2 and 9.2, 2026-09-17).

It replaced `mint_aoauth_token`, the RS256 bearer the platform verified
against a PEM on the agent card. Registration is now a signed request
(`webagents.crypto.http_signature`); the claim token stays a JWT because a
person pastes it into the claim route, so it is not a request the agent
makes. What the claim route reads: `EdDSA`, the `kid` selecting the
registered key by thumbprint, the audience `{platform}/claim`, the scope
`agent:claim`, and a `jti` it spends once.
"""

import json
import time

import jwt as pyjwt
import pytest

from webagents.crypto.jwks import JWKSManager

AGENT_ID = "mini"
PLATFORM = "https://robutler.ai"
AGENT_URL = "https://agent.example.com/agents/mini"


@pytest.fixture
def manager(tmp_path):
    mgr = JWKSManager({"keys_dir": str(tmp_path)})
    mgr.ensure_ed25519_key(AGENT_ID)
    return mgr


def _verify(manager, token, audience=f"{PLATFORM}/claim"):
    # Exactly what the platform does: select the key by `kid` from the key
    # set and verify the EdDSA signature against it.
    jwk = manager.get_ed25519_public_jwk()
    header = pyjwt.get_unverified_header(token)
    assert header["kid"] == jwk["kid"]
    public_key = pyjwt.algorithms.OKPAlgorithm.from_jwk(json.dumps(jwk))
    return pyjwt.decode(token, public_key, algorithms=["EdDSA"], audience=audience)


def test_the_token_verifies_by_kid_with_the_published_key(manager):
    token = manager.mint_claim_token(AGENT_ID, PLATFORM)
    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "EdDSA"
    assert header["kid"] == manager.get_ed25519_thumbprint()

    claims = _verify(manager, token)
    assert claims["sub"] == AGENT_ID
    assert claims["aud"] == f"{PLATFORM}/claim"
    assert claims["scope"] == "agent:claim"
    assert claims["jti"]
    assert claims["exp"] - claims["iat"] == 600
    assert claims["nbf"] == claims["iat"]
    assert claims["exp"] > time.time()
    assert "iss" not in claims


def test_the_audience_is_derived_from_the_platform_url_never_the_agent_url(manager):
    token = manager.mint_claim_token(AGENT_ID, PLATFORM + "/")
    assert _verify(manager, token)["aud"] == f"{PLATFORM}/claim"
    with pytest.raises(pyjwt.InvalidAudienceError):
        _verify(manager, token, audience=f"{AGENT_URL}/claim")


def test_ttl_and_issuer_are_honoured(manager):
    token = manager.mint_claim_token(AGENT_ID, PLATFORM, ttl_seconds=30, issuer=AGENT_URL + "/")
    claims = _verify(manager, token)
    assert claims["exp"] - claims["iat"] == 30
    assert claims["iss"] == AGENT_URL


def test_every_token_has_its_own_jti(manager):
    first = _verify(manager, manager.mint_claim_token(AGENT_ID, PLATFORM))["jti"]
    second = _verify(manager, manager.mint_claim_token(AGENT_ID, PLATFORM))["jti"]
    assert first != second


def test_the_rsa_key_does_not_sign_it(manager):
    token = manager.mint_claim_token(AGENT_ID, PLATFORM)
    manager.ensure_keys(AGENT_ID)  # the RSA key exists beside it and is not used
    # PyJWT refuses before it verifies (an RSA key is not an EdDSA key); any
    # refusal is the point, so the base class is what is expected.
    with pytest.raises(pyjwt.exceptions.PyJWTError):
        pyjwt.decode(
            token,
            manager.get_signing_key().public_key(),
            algorithms=["RS256", "EdDSA"],
            audience=f"{PLATFORM}/claim",
        )


def test_minting_without_the_key_is_a_clear_error(tmp_path):
    mgr = JWKSManager({"keys_dir": str(tmp_path)})
    mgr.ensure_keys(AGENT_ID)  # the RSA key alone does not mint it
    with pytest.raises(RuntimeError, match="ensure_ed25519_key"):
        mgr.mint_claim_token(AGENT_ID, PLATFORM)


def test_the_bearer_assertion_is_gone():
    # No compatibility window (ADR 0038 operator decisions, 2026-09-17): the
    # RS256 registration bearer must not come back beside the signer.
    assert not hasattr(JWKSManager, "mint_aoauth_token")
