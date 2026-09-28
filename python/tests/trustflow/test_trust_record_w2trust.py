"""
Verifying the exportable TrustFlow record (plan item 2.7, 2026-09-26): the
vectors both SDKs verify (`tests/fixtures/trust/trustflow_record.json`, signed
by the portal's own signer byte for byte, pinned there by
tests/unit/reputation/trust-record-w2trust.test.ts; TypeScript
tests/unit/trustflow/trust-record-w2trust.test.ts), the key set fetch, and
the A2A card extension.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from webagents.trustflow.trust_record import (
    TRUSTFLOW_RECORD_EXTENSION_URI,
    _reset_trust_key_sets,
    decode_trust_record,
    key_set_url_for_issuer,
    trust_record_extension,
    trust_record_from_card,
    verify_trust_record,
    with_trust_record_extension,
)

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "trust" / "trustflow_record.json").read_text())
HELD = {"keys": FIXTURE["jwks"]["keys"], "issuer": FIXTURE["issuer"], "now": FIXTURE["now"]}
JWKS_URL = f"{FIXTURE['issuer']}/.well-known/jwks.json"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
async def test_the_shared_vectors(case):
    options = dict(HELD)
    if "issuer" in case:
        options["issuer"] = case["issuer"]
    if "subject" in case:
        options["subject"] = case["subject"]
    result = await verify_trust_record(FIXTURE["records"][case["record"]], **options)
    assert result.ok is case["expect"]["ok"], result
    if result.ok:
        assert result.record == FIXTURE["payload"]
        if "kid" in case["expect"]:
            assert result.kid == case["expect"]["kid"]
    else:
        assert result.code == case["expect"]["code"]


def test_decodes_the_header_and_claims_without_deciding_anything():
    assert decode_trust_record(FIXTURE["records"]["valid"]) == {"header": FIXTURE["header"], "payload": FIXTURE["payload"]}
    assert decode_trust_record(FIXTURE["records"]["tampered"])["payload"]["score"] == 0.99
    assert decode_trust_record("nope") is None


class TestTheKeySet:
    def setup_method(self):
        _reset_trust_key_sets()

    async def test_fetched_from_the_expected_issuer_held_and_refetched_once_for_an_unknown_kid(self):
        fetched = []

        async def fetch_json(url):
            fetched.append(url)
            assert url == JWKS_URL
            return FIXTURE["jwks"]

        assert (await verify_trust_record(FIXTURE["records"]["valid"], issuer=FIXTURE["issuer"], now=FIXTURE["now"], fetch_json=fetch_json)).ok
        assert (await verify_trust_record(FIXTURE["records"]["valid"], issuer=FIXTURE["issuer"], now=FIXTURE["now"], fetch_json=fetch_json)).ok
        assert len(fetched) == 1
        unknown = await verify_trust_record(FIXTURE["records"]["unknown_kid"], issuer=FIXTURE["issuer"], now=FIXTURE["now"], fetch_json=fetch_json)
        assert (unknown.ok, unknown.code) == (False, "no_key")
        assert len(fetched) == 2

    async def test_never_fetches_from_an_issuer_the_filter_refuses(self):
        async def must_not_fetch(url):
            raise AssertionError(f"fetched {url}")

        for issuer in ("http://169.254.169.254", "http://evil.example", "https://127.0.0.1", "https://robutler.ai?x=1"):
            result = await verify_trust_record(FIXTURE["records"]["valid"], issuer=issuer, now=FIXTURE["now"], fetch_json=must_not_fetch)
            assert (result.ok, result.code) == (False, "key_set"), issuer
        assert key_set_url_for_issuer("http://localhost:3000") == "http://localhost:3000/.well-known/jwks.json"
        assert key_set_url_for_issuer("https://robutler.ai/") == JWKS_URL

    async def test_a_key_set_that_cannot_be_read_is_a_refusal_not_a_raise(self):
        async def fetch_json(url):
            raise RuntimeError("503")

        result = await verify_trust_record(FIXTURE["records"]["valid"], issuer=FIXTURE["issuer"], now=FIXTURE["now"], fetch_json=fetch_json)
        assert (result.ok, result.code) == (False, "key_set")

    async def test_the_default_issuer_is_configuration_never_the_record(self, monkeypatch):
        monkeypatch.setenv("ROBUTLER_PLATFORM_ISSUER", FIXTURE["issuer"])
        assert (await verify_trust_record(FIXTURE["records"]["valid"], keys=FIXTURE["jwks"]["keys"], now=FIXTURE["now"])).ok
        monkeypatch.setenv("ROBUTLER_PLATFORM_ISSUER", "https://staging.robutler.net")
        result = await verify_trust_record(FIXTURE["records"]["valid"], keys=FIXTURE["jwks"]["keys"], now=FIXTURE["now"])
        assert (result.ok, result.code) == (False, "issuer")


class TestTheCardExtension:
    def bare(self):
        card = copy.deepcopy(FIXTURE["extension"]["card_with_record"])
        del card["capabilities"]["extensions"]
        return card

    def test_writes_the_fixture_entry_under_the_robutler_uri(self):
        assert TRUSTFLOW_RECORD_EXTENSION_URI == FIXTURE["extension"]["uri"]
        valid = FIXTURE["records"]["valid"]
        assert trust_record_extension(valid) == {"uri": FIXTURE["extension"]["uri"], "description": FIXTURE["extension"]["description"], "params": {"record": valid}}
        bare = self.bare()
        assert with_trust_record_extension(bare, valid) == FIXTURE["extension"]["card_with_record"]
        assert "extensions" not in bare["capabilities"]

    def test_replaces_an_earlier_record_and_keeps_other_extensions(self):
        other = {"uri": "https://other.example/ext", "params": {"a": 1}}
        card = dict(self.bare(), capabilities={"streaming": True, "extensions": [other]})
        once = with_trust_record_extension(card, "old.record.x")
        twice = with_trust_record_extension(once, FIXTURE["records"]["valid"])
        assert twice["capabilities"]["extensions"] == [other, trust_record_extension(FIXTURE["records"]["valid"])]

    async def test_reads_the_record_back_and_verifies_it_for_the_card_subject(self):
        record = trust_record_from_card(FIXTURE["extension"]["card_with_record"])
        assert record == FIXTURE["records"]["valid"]
        assert trust_record_from_card(self.bare()) is None
        assert (await verify_trust_record(record, subject={"url": "https://agents.example.com/agents/scout"}, **HELD)).ok
        foreign = await verify_trust_record(record, subject={"url": "https://elsewhere.example/agents/scout"}, **HELD)
        assert (foreign.ok, foreign.code) == (False, "subject")
