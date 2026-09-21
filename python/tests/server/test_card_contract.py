"""The agent card contract (ADR 0038 step 5, W2 design section 3.3, 2026-09-17).

The card is the registration-time metadata document, read once at
``{principal}/.well-known/agent.json``, and the platform checks three fields
by simple string comparison: ``client_id`` equals the URL the card was
fetched from, ``url`` equals the principal, ``jwks_uri`` equals the key-set
URL the signature was resolved through. A card failing any of them is
``card_not_self_naming`` or ``card_key_set_mismatch``. The key itself comes
from the key set, so the card carries NO key material: the ``publicKey``
and ``metadata.publicKey`` PEM placements the bearer flow read are gone,
and a card that still published them would be advertising a credential
shape nothing verifies.
"""

import pytest

from webagents.server.core.registration import build_agent_card, compose_principal

PRINCIPAL = "https://agent.example.com/agents/card"


class _Agent:
    name = "card"
    instructions = "You are helpful."


def test_the_card_self_names():
    card = build_agent_card(_Agent(), PRINCIPAL)
    assert card["client_id"] == PRINCIPAL + "/.well-known/agent.json"
    assert card["url"] == PRINCIPAL
    assert card["jwks_uri"] == PRINCIPAL + "/.well-known/jwks.json"


def test_the_card_carries_no_key_material():
    card = build_agent_card(_Agent(), PRINCIPAL)
    assert "publicKey" not in card
    assert "metadata" not in card
    assert "jwks" not in card
    assert "-----BEGIN" not in str(card)


def test_the_card_names_the_signature_scheme_and_keeps_its_metadata():
    card = build_agent_card(_Agent(), PRINCIPAL)
    assert card["name"] == "card"
    assert card["description"] == "You are helpful."
    assert card["authentication"] == {"schemes": ["HTTPSig"]}
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False}


def test_a_trailing_slash_never_reaches_the_card():
    # The platform compares `url` to the principal by string equality, and
    # the principal has no trailing slash.
    card = build_agent_card(_Agent(), PRINCIPAL + "/")
    assert card["url"] == PRINCIPAL
    assert card["client_id"] == PRINCIPAL + "/.well-known/agent.json"


@pytest.mark.parametrize(
    "base,agent_path,expected",
    [
        ("https://agent.example.com", None, "https://agent.example.com/card"),
        ("https://agent.example.com", "", "https://agent.example.com/card"),
        ("https://agent.example.com/", "/agents", "https://agent.example.com/agents/card"),
        ("https://agent.example.com", "/agents/", "https://agent.example.com/agents/card"),
        # The relative last resort is already `/{name}`; the prefix goes in front.
        ("/card", None, "/card"),
        ("/card", "/agents", "/agents/card"),
    ],
)
def test_the_principal_is_the_mount_path(base, agent_path, expected):
    assert compose_principal(base, "card", agent_path) == expected


def test_the_principal_is_where_the_card_and_key_set_are_served():
    principal = compose_principal("https://agent.example.com", "card", "/agents")
    card = build_agent_card(_Agent(), principal)
    assert card["client_id"] == "https://agent.example.com/agents/card/.well-known/agent.json"
    assert card["jwks_uri"] == "https://agent.example.com/agents/card/.well-known/jwks.json"


def test_the_principal_is_spelled_the_way_the_platform_derives_it():
    # 2026-09-18: the platform derives the principal from `Signature-Agent`
    # through a WHATWG parse, host lowercased and a default port dropped, and
    # compares the card by string equality. The card must be built from that
    # same spelling, and it is the spelling the signer uses.
    from webagents.crypto.http_signature import normalize_agent_url

    principal = compose_principal("https://Agents.Example.com:443", "card", "/agents")
    assert principal == "https://agents.example.com/agents/card"
    assert principal == normalize_agent_url("https://Agents.Example.com:443/agents/card")
    card = build_agent_card(_Agent(), principal)
    assert card["url"] == principal
    assert card["client_id"] == principal + "/.well-known/agent.json"
    assert card["jwks_uri"] == principal + "/.well-known/jwks.json"
