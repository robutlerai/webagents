"""`JWKSManager.mint_aoauth_token` — the key-possession proof.

The token is EXPERIMENTAL (nothing in the SDK presents it yet), so this pins
the one thing that matters if a developer follows the docstring: the token
must verify against the SPKI public key the agent card publishes, with the
claims platform auto-registration reads. A minting helper nobody calls and
nobody tests is how a "registration is supported" claim survives being false.
"""

import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from webagents.crypto.jwks import JWKSManager

AGENT_ID = "mini"
ISSUER = "https://agent.example.com/agents/mini"
PLATFORM = "https://robutler.ai"


@pytest.fixture
def manager(tmp_path):
    mgr = JWKSManager({"keys_dir": str(tmp_path)})
    mgr.ensure_keys(AGENT_ID)
    return mgr


def test_token_verifies_against_the_card_public_key(manager):
    token = manager.mint_aoauth_token(AGENT_ID, issuer=ISSUER, audience=PLATFORM)

    # Exactly what the platform does: take metadata.publicKey off the card
    # (SPKI PEM) and verify the bearer with it.
    spki_pem = manager.get_public_key_spki_pem()
    assert spki_pem.startswith("-----BEGIN PUBLIC KEY-----")
    public_key = load_pem_public_key(spki_pem.encode())

    claims = pyjwt.decode(token, public_key, algorithms=["RS256"], audience=PLATFORM)
    assert claims["sub"] == AGENT_ID
    assert claims["client_id"] == AGENT_ID
    assert claims["iss"] == ISSUER
    assert claims["aud"] == PLATFORM
    assert claims["token_type"] == "Bearer"
    assert claims["exp"] > time.time()

    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "RS256"
    # The kid must name the key the JWKS lists, or a verifier that pins kid
    # (this SDK's own does) can never find it.
    assert header["kid"] == manager.get_public_jwk()["kid"]


def test_issuer_trailing_slash_is_normalised(manager):
    token = manager.mint_aoauth_token(AGENT_ID, issuer=ISSUER + "/", audience=PLATFORM)
    claims = pyjwt.decode(
        token, load_pem_public_key(manager.get_public_key_spki_pem().encode()),
        algorithms=["RS256"], audience=PLATFORM,
    )
    assert claims["iss"] == ISSUER


def test_minting_without_keys_is_a_clear_error(tmp_path):
    mgr = JWKSManager({"keys_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="ensure_keys"):
        mgr.mint_aoauth_token(AGENT_ID, issuer=ISSUER, audience=PLATFORM)
