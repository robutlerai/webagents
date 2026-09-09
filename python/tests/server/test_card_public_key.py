"""The agent card publishes the SPKI PEM at BOTH ``publicKey`` and
``metadata.publicKey`` (build plan 1M-00, ADR-0038 step 1).

The platform's verifier reads the card's TOP-LEVEL ``publicKey`` (portal
``lib/auth/agent-auth.ts``: ``metadata?.publicKey`` where ``metadata`` is
the whole fetched card). This SDK wrote only the nested ``metadata.publicKey``,
so a card that looked complete to every reader in this repository was a card
with no key to the one reader that matters, and no Python-served agent could
auto-register. These tests pin the shape the verifier reads, and pin that the
two placements never drift apart.
"""

from webagents.server.core.registration import build_agent_card

PEM = "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAtest\n-----END PUBLIC KEY-----\n"


class _Agent:
    name = "card"
    instructions = "You are helpful."


def test_public_key_is_published_at_the_top_level_where_the_platform_reads_it():
    card = build_agent_card(_Agent(), "/card", PEM)
    assert card["publicKey"] == PEM


def test_public_key_is_still_published_under_metadata_for_readers_of_that_shape():
    card = build_agent_card(_Agent(), "/card", PEM)
    assert card["metadata"]["publicKey"] == PEM


def test_the_two_placements_carry_the_same_key():
    card = build_agent_card(_Agent(), "/card", PEM)
    assert card["publicKey"] == card["metadata"]["publicKey"]


def test_no_key_means_neither_placement_not_an_empty_value():
    # A card with no key must not tempt the verifier with an empty string:
    # ``importSPKI('')`` fails and negative-caches the issuer for the TTL.
    card = build_agent_card(_Agent(), "/card", None)
    assert "publicKey" not in card
    assert "metadata" not in card
