"""
The Web Bot Auth request signer, `webagents.crypto.http_signature` (ADR 0038
step 5, W2 design sections 2.1 to 2.7, 9.2 and 10.2, 2026-09-17).

What is pinned, in the design's own order: the RFC 9651 serialiser subset
(strings with escaping, tokens, integers, byte sequences, flag parameters,
inner lists, dictionaries); the derived components of a request target; the
three `Signature-Agent` forms and their covered component; every signature
parameter and every covered component; `Content-Digest` present iff there
is a body; rotation with two labels; the end-to-end example of design
section 2.7 byte for byte; and `WebBotAuth` applied by a real httpx client,
sync and async, over a mock transport. The signatures are checked with the
test-only verifier in `support.py`, which rebuilds the base from the headers
and the request rather than trusting the signer's own record of it.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from webagents.crypto.http_signature import (
    DEFAULT_SIGNATURE_AGENT_FORM,
    SIGNATURE_AGENT_FORMS,
    SIGNATURE_HEADERS,
    InnerList,
    Item,
    SigningError,
    SigningKey,
    Token,
    WebBotAuth,
    agent_card_url,
    content_digest,
    ed25519_public_jwk,
    jwk_thumbprint,
    key_set_url,
    normalize_agent_url,
    request_target,
    serialize_bare_item,
    serialize_dictionary,
    serialize_inner_list,
    serialize_item,
    sign_request,
    signature_agent_component,
    signature_agent_header,
    signature_labels,
)

from .support import split_dictionary, verify_signed_request

# RFC 9421 Appendix B.1.4, test-key-ed25519 (the key of the draft's own
# vectors, denylisted by the platform's profile layer, P section 6.8).
B14_D = "n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU"
B14_X = "JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs"
B14_THUMBPRINT = "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U"

AGENT_URL = "https://agent.example/agents/mini"
PLATFORM_URL = "https://robutler.ai/api/auth/cli/token"
CREATED = 1758067200
EXPIRES = 1758067260
# 64 bytes, 0x00 to 0x3f: fixed, and obviously not random.
FIXED_NONCE = base64.b64encode(bytes(range(64))).decode()
DIGEST_OF_EMPTY_OBJECT = "sha-256=:RBNvo1WzZ4oRRq0W9+hknpT7T8If536DEMBg9hyq/4o=:"


def b14_key() -> SigningKey:
    raw = base64.urlsafe_b64decode(B14_D + "=")
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.from_private_bytes(raw))


def fresh_key() -> SigningKey:
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())


def public_keys(*keys: SigningKey):
    return {key.thumbprint: key.private_key.public_key() for key in keys}


# ---------------------------------------------------------------------------
# RFC 9651 serialiser subset
# ---------------------------------------------------------------------------


class TestSerializer:
    def test_string_is_quoted_and_escaped(self):
        assert serialize_bare_item('a"b\\c') == '"a\\"b\\\\c"'
        assert serialize_bare_item("") == '""'

    def test_string_must_be_printable_ascii(self):
        with pytest.raises(SigningError):
            serialize_bare_item("café")
        with pytest.raises(SigningError):
            serialize_bare_item("line\nbreak")

    def test_token_is_bare_and_validated(self):
        assert serialize_bare_item(Token("jwks_uri")) == "jwks_uri"
        assert serialize_bare_item(Token("*a/b:c")) == "*a/b:c"
        with pytest.raises(SigningError):
            serialize_bare_item(Token("1abc"))
        with pytest.raises(SigningError):
            serialize_bare_item(Token('has"quote'))

    def test_integer_and_its_range(self):
        assert serialize_bare_item(1758067200) == "1758067200"
        assert serialize_bare_item(-5) == "-5"
        with pytest.raises(SigningError):
            serialize_bare_item(10**15)

    def test_byte_sequence_is_standard_base64_between_colons(self):
        assert serialize_bare_item(b"\xfb\xff") == ":+/8=:"
        assert serialize_bare_item(b"") == "::"

    def test_booleans(self):
        assert serialize_bare_item(True) == "?1"
        assert serialize_bare_item(False) == "?0"

    def test_parameters_flag_form_and_ordering(self):
        item = Item("v", (("flag", True), ("type", Token("jwks_uri")), ("n", 3), ("off", False)))
        assert serialize_item(item) == '"v";flag;type=jwks_uri;n=3;off=?0'

    def test_parameter_keys_are_validated(self):
        with pytest.raises(SigningError):
            serialize_item(Item("v", (("Type", Token("x")),)))
        with pytest.raises(SigningError):
            serialize_dictionary([("Sig1", Item("v"))])

    def test_inner_list_with_parameters(self):
        inner = InnerList((Item("@method"), Item("signature-agent", (("key", "sig1"),))), (("created", 1),))
        assert serialize_inner_list(inner) == '("@method" "signature-agent";key="sig1");created=1'
        assert serialize_inner_list(InnerList(())) == "()"

    def test_dictionary_members_and_true_member(self):
        members = [("a", Item(1)), ("b", Item(True, (("p", Token("q")),))), ("c", InnerList((Item("x"),)))]
        assert serialize_dictionary(members) == 'a=1, b;p=q, c=("x")'

    def test_unknown_bare_item_type_is_refused(self):
        with pytest.raises(SigningError):
            serialize_bare_item(1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Keys, thumbprints, digests, targets, agent URLs
# ---------------------------------------------------------------------------


class TestKeysAndDigests:
    def test_b14_thumbprint_is_the_rfc_value(self):
        key = b14_key()
        assert key.thumbprint == B14_THUMBPRINT
        assert ed25519_public_jwk(key.private_key.public_key()) == {"kty": "OKP", "crv": "Ed25519", "x": B14_X}
        assert jwk_thumbprint({"x": B14_X, "kty": "OKP", "crv": "Ed25519", "kid": "ignored"}) == B14_THUMBPRINT

    def test_published_jwk_shape(self):
        jwk = b14_key().public_jwk()
        assert jwk == {"kty": "OKP", "crv": "Ed25519", "x": B14_X, "kid": B14_THUMBPRINT, "use": "sig"}
        assert "alg" not in jwk

    def test_thumbprint_shape(self):
        thumbprint = fresh_key().thumbprint
        assert len(thumbprint) == 43
        assert "=" not in thumbprint and "+" not in thumbprint and "/" not in thumbprint

    def test_only_ed25519_signs(self):
        with pytest.raises(SigningError, match="Ed25519"):
            SigningKey.from_private_key(rsa.generate_private_key(public_exponent=65537, key_size=2048))  # type: ignore[arg-type]

    def test_content_digest_matches_the_design_example(self):
        assert content_digest(b"{}") == DIGEST_OF_EMPTY_OBJECT
        assert content_digest(b"") == "sha-256=:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=:"


class TestRequestTarget:
    def test_method_uppercased_host_lowercased_default_port_dropped(self):
        target = request_target("post", "https://Robutler.AI:443/api/auth/cli/token")
        assert (target.method, target.authority, target.path, target.query) == (
            "POST",
            "robutler.ai",
            "/api/auth/cli/token",
            "?",
        )

    def test_non_default_port_kept_and_http_default_dropped(self):
        assert request_target("GET", "https://h.example:8443/").authority == "h.example:8443"
        assert request_target("GET", "http://h.example:80/").authority == "h.example"
        assert request_target("GET", "http://h.example:8080/").authority == "h.example:8080"

    def test_ipv6_literal_keeps_its_brackets(self):
        assert request_target("GET", "https://[::1]:8443/x").authority == "[::1]:8443"

    def test_query_keeps_percent_encoding_and_leading_question_mark(self):
        target = request_target("GET", "https://h.example/a%20b/c?x=1&y=%2F")
        assert target.path == "/a%20b/c"
        assert target.query == "?x=1&y=%2F"

    def test_empty_query_and_absent_query_are_both_a_question_mark(self):
        assert request_target("GET", "https://h.example/p?").query == "?"
        assert request_target("GET", "https://h.example/p").query == "?"

    def test_empty_path_is_root(self):
        assert request_target("GET", "https://h.example").path == "/"

    def test_httpx_url_reads_the_same_as_the_string(self):
        for url in ("https://Robutler.AI:443/api/auth/cli/token", "https://h.example:8443/a%20b?x=%2F", "https://h.example/p?"):
            assert request_target("POST", httpx.URL(url)) == request_target("POST", url)

    def test_relative_and_non_http_urls_are_refused(self):
        for bad in ("/api/auth/cli/token", "ftp://h.example/x", "https:///x"):
            with pytest.raises(SigningError):
                request_target("GET", bad)

    @pytest.mark.parametrize(
        "url, path, query, sent",
        [
            # 2026-09-19: `@path` and `@query` are the WHATWG spelling, which is
            # what the TypeScript signer signs and the platform rebuilds. This
            # signer used to sign the caller's spelling (`/a/../b`) while httpx
            # sent `/b`, and to sign AND send `%2e%2e`, which the platform resolves.
            ("https://h.example/a/../b", "/b", "?", "https://h.example/b"),
            ("https://h.example/a/./b/", "/a/b/", "?", "https://h.example/a/b/"),
            ("https://h.example/a/%2e%2E/b/%2e", "/b/", "?", "https://h.example/b/"),
            ("https://h.example/a/b/..", "/a/", "?", "https://h.example/a/"),
            ("https://h.example/../../x", "/x", "?", "https://h.example/x"),
            ("https://h.example//a//b", "//a//b", "?", "https://h.example//a//b"),
            ("https://h.example/a/.../b", "/a/.../b", "?", "https://h.example/a/.../b"),
            ("https://h.example/a b/{c}^`", "/a%20b/%7Bc%7D%5E%60", "?", "https://h.example/a%20b/%7Bc%7D%5E%60"),
            ("https://h.example/%7euser/%zz", "/%7euser/%zz", "?", "https://h.example/%7euser/%zz"),
            ("https://h.example/p?a=1 2&b='x'", "/p", "?a=1%202&b=%27x%27", "https://h.example/p?a=1%202&b=%27x%27"),
            ("https://h.example/p?a=1#frag?x", "/p", "?a=1", "https://h.example/p?a=1"),
            ("https://h.example/p?a?b", "/p", "?a?b", "https://h.example/p?a?b"),
            ("https://h.example?x", "/", "?x", "https://h.example/?x"),
            ("  https://h.example/a\tb\n/c  ", "/ab/c", "?", "https://h.example/ab/c"),
            ("https://h.example\\a\\..\\b?x\\y", "/b", "?x\\y", "https://h.example/b?x\\y"),
            ("https://h.example/caf\u00e9?q=\u00fc", "/caf%C3%A9", "?q=%C3%BC", "https://h.example/caf%C3%A9?q=%C3%BC"),
        ],
    )
    def test_path_and_query_are_the_whatwg_spelling(self, url, path, query, sent):
        target = request_target("GET", url)
        assert (target.path, target.query, target.url) == (path, query, sent)
        # And that spelling is a fixed point: signing what was sent signs the same thing.
        again = request_target("GET", target.url)
        assert (again.path, again.query, again.url) == (path, query, sent)


class TestAgentUrl:
    def test_trailing_slash_is_dropped_and_well_known_urls_derive(self):
        assert normalize_agent_url(AGENT_URL + "/") == AGENT_URL
        assert key_set_url(AGENT_URL + "/") == AGENT_URL + "/.well-known/jwks.json"
        assert agent_card_url(AGENT_URL) == AGENT_URL + "/.well-known/agent.json"

    @pytest.mark.parametrize(
        "bad",
        ["/agents/mini", "https://user:pw@agent.example/x", "https://agent.example/x?y=1", "https://agent.example/x#f", "", "https://agént.example/x"],
    )
    def test_refused_shapes(self, bad):
        with pytest.raises(SigningError):
            normalize_agent_url(bad)


# ---------------------------------------------------------------------------
# The three Signature-Agent forms (design section 2.4)
# ---------------------------------------------------------------------------


class TestSignatureAgentForms:
    def test_the_default_is_the_typed_dictionary(self):
        assert DEFAULT_SIGNATURE_AGENT_FORM == "dictionary-typed"
        assert SIGNATURE_AGENT_FORMS == ("dictionary-typed", "dictionary-untyped", "legacy-string")

    def test_header_per_form(self):
        assert (
            signature_agent_header(AGENT_URL, "dictionary-typed", ["sig1"])
            == 'sig1="https://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri'
        )
        assert (
            signature_agent_header(AGENT_URL, "dictionary-untyped", ["sig1"])
            == 'sig1="https://agent.example/agents/mini/.well-known/jwks.json"'
        )
        assert signature_agent_header(AGENT_URL, "legacy-string", ["sig1"]) == '"https://agent.example"'

    def test_legacy_string_is_the_origin_with_a_non_default_port(self):
        assert signature_agent_header("https://agent.example:8443/agents/mini", "legacy-string", ["sig1"]) == '"https://agent.example:8443"'
        assert signature_agent_header("https://agent.example:443/agents/mini", "legacy-string", ["sig1"]) == '"https://agent.example"'

    def test_covered_component_is_keyed_except_for_the_legacy_string(self):
        assert serialize_item(signature_agent_component("dictionary-typed", "sig1")) == '"signature-agent";key="sig1"'
        assert serialize_item(signature_agent_component("dictionary-untyped", "sig2")) == '"signature-agent";key="sig2"'
        assert serialize_item(signature_agent_component("legacy-string", "sig1")) == '"signature-agent"'

    def test_unknown_form_is_refused(self):
        with pytest.raises(SigningError, match="dictionary-typed"):
            signature_agent_header(AGENT_URL, "cimd", ["sig1"])
        with pytest.raises(SigningError):
            sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", form="directory")


# ---------------------------------------------------------------------------
# sign_request: parameters, components, digest, rotation
# ---------------------------------------------------------------------------


class TestSignRequest:
    def test_every_parameter_and_component_in_order(self):
        key = fresh_key()
        signed = sign_request([key], AGENT_URL, "POST", PLATFORM_URL, b"{}", created=CREATED)
        verified = verify_signed_request(signed.headers, "POST", PLATFORM_URL, b"{}", public_keys(key))
        assert list(verified) == ["sig1"]
        record = verified["sig1"]
        assert record["created"] == CREATED
        assert record["expires"] == CREATED + 60
        assert record["keyid"] == key.thumbprint
        assert record["alg"] == "ed25519"
        assert record["tag"] == "web-bot-auth"
        nonce = record["nonce"]
        assert len(base64.b64decode(nonce, validate=True)) == 64
        assert record["components"] == [
            ("@method", ""),
            ("@authority", ""),
            ("@path", ""),
            ("@query", ""),
            ("content-digest", ""),
            ("signature-agent", "sig1"),
        ]
        # The parameter ORDER is part of the bytes (design section 2.3).
        member = split_dictionary(signed.headers["Signature-Input"])["sig1"]
        assert member.index(";created=") < member.index(";expires=") < member.index(";keyid=") < member.index(';alg="ed25519"') < member.index(";nonce=") < member.index(';tag="web-bot-auth"')
        assert set(signed.headers) == set(SIGNATURE_HEADERS)

    def test_created_defaults_to_now_and_lifetime_is_60(self):
        import time

        key = fresh_key()
        before = int(time.time())
        signed = sign_request([key], AGENT_URL, "GET", "https://robutler.ai/api/feed")
        record = signed.signatures[0]
        assert before <= record.created <= int(time.time())
        assert record.expires - record.created == 60
        signed = sign_request([key], AGENT_URL, "GET", "https://robutler.ai/api/feed", lifetime=300)
        assert signed.signatures[0].expires - signed.signatures[0].created == 300

    def test_lifetime_bounds(self):
        for bad in (0, -1, 3601):
            with pytest.raises(SigningError):
                sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", lifetime=bad)

    def test_lifetime_is_a_whole_number_of_seconds(self):
        # 2026-09-19: `0.5` passed `0 < lifetime <= 3600` and `int()` then made
        # expires == created, which the platform refuses; the TypeScript signer
        # has always required an integer. `True` is an int in Python.
        for bad in (0.5, 1.5, 59.9, 60.0, True, "60", None):
            with pytest.raises(SigningError, match="integer between 1 and 3600"):
                sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", lifetime=bad)  # type: ignore[arg-type]
            with pytest.raises(SigningError, match="integer between 1 and 3600"):
                WebBotAuth([fresh_key()], AGENT_URL, lifetime=bad)  # type: ignore[arg-type]
        for good in (1, 60, 3600):
            signed = sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", lifetime=good, created=CREATED)
            assert signed.signatures[0].expires == CREATED + good
        with pytest.raises(SigningError, match="created must be an integer"):
            sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", created=1758067200.5)  # type: ignore[arg-type]

    def test_content_digest_present_iff_there_is_a_body(self):
        key = fresh_key()
        without = sign_request([key], AGENT_URL, "GET", "https://robutler.ai/api/feed")
        assert "Content-Digest" not in without.headers
        assert '"content-digest"' not in without.headers["Signature-Input"]
        verify_signed_request(without.headers, "GET", "https://robutler.ai/api/feed", b"", public_keys(key))

        body = json.dumps({"hello": "world"}).encode()
        with_body = sign_request([key], AGENT_URL, "POST", "https://robutler.ai/api/feed", body)
        assert with_body.headers["Content-Digest"] == content_digest(body)
        assert '"content-digest" "signature-agent";key="sig1"' in with_body.headers["Signature-Input"]
        verify_signed_request(with_body.headers, "POST", "https://robutler.ai/api/feed", body, public_keys(key))

    def test_the_digest_binds_the_body(self):
        key = fresh_key()
        signed = sign_request([key], AGENT_URL, "POST", PLATFORM_URL, b"{}")
        with pytest.raises(AssertionError):
            verify_signed_request(signed.headers, "POST", PLATFORM_URL, b'{"a":1}', public_keys(key))

    def test_the_signature_binds_method_authority_path_and_query(self):
        key = fresh_key()
        url = "https://robutler.ai/api/feed?cursor=abc"
        signed = sign_request([key], AGENT_URL, "GET", url)
        verify_signed_request(signed.headers, "GET", url, b"", public_keys(key))
        for method, other in (
            ("POST", url),
            ("GET", "https://attacker.example/api/feed?cursor=abc"),
            ("GET", "https://robutler.ai/api/feeds?cursor=abc"),
            ("GET", "https://robutler.ai/api/feed?cursor=abd"),
        ):
            with pytest.raises(InvalidSignature):
                verify_signed_request(signed.headers, method, other, b"", public_keys(key))

    @pytest.mark.parametrize("form", SIGNATURE_AGENT_FORMS)
    def test_each_form_verifies(self, form):
        key = fresh_key()
        signed = sign_request([key], AGENT_URL, "POST", PLATFORM_URL, b"{}", form=form)
        verified = verify_signed_request(signed.headers, "POST", PLATFORM_URL, b"{}", public_keys(key))
        expected_key = "" if form == "legacy-string" else "sig1"
        assert verified["sig1"]["components"][-1] == ("signature-agent", expected_key)

    def test_signature_is_64_bytes_standard_base64(self):
        signed = sign_request([fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x")
        member = split_dictionary(signed.headers["Signature"])["sig1"]
        assert member.startswith(":") and member.endswith(":")
        assert len(base64.b64decode(member[1:-1], validate=True)) == 64
        assert len(signed.signatures[0].signature) == 64

    def test_rotation_signs_once_per_held_key(self):
        current, previous = fresh_key(), fresh_key()
        signed = sign_request([current, previous], AGENT_URL, "POST", PLATFORM_URL, b"{}")
        verified = verify_signed_request(signed.headers, "POST", PLATFORM_URL, b"{}", public_keys(current, previous))
        assert list(verified) == ["sig1", "sig2"]
        assert verified["sig1"]["keyid"] == current.thumbprint
        assert verified["sig2"]["keyid"] == previous.thumbprint
        assert verified["sig1"]["nonce"] != verified["sig2"]["nonce"]
        assert verified["sig1"]["created"] == verified["sig2"]["created"]
        assert verified["sig1"]["components"][-1] == ("signature-agent", "sig1")
        assert verified["sig2"]["components"][-1] == ("signature-agent", "sig2")
        agents = split_dictionary(signed.headers["Signature-Agent"])
        assert list(agents) == ["sig1", "sig2"]
        assert agents["sig1"] == agents["sig2"] == '"https://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri'
        assert len(signed.signatures) == 2

    def test_rotation_with_the_legacy_string_keeps_one_bare_value(self):
        current, previous = fresh_key(), fresh_key()
        signed = sign_request([current, previous], AGENT_URL, "GET", "https://robutler.ai/x", form="legacy-string")
        assert signed.headers["Signature-Agent"] == '"https://agent.example"'
        verified = verify_signed_request(signed.headers, "GET", "https://robutler.ai/x", b"", public_keys(current, previous))
        assert [v["components"][-1] for v in verified.values()] == [("signature-agent", ""), ("signature-agent", "")]

    def test_labels(self):
        assert signature_labels("sig1", 1) == ["sig1"]
        assert signature_labels("sig1", 3) == ["sig1", "sig2", "sig3"]
        assert signature_labels("agent", 2) == ["agent", "agent2"]
        with pytest.raises(SigningError):
            signature_labels("Sig1", 1)

    def test_refusals(self):
        key = fresh_key()
        with pytest.raises(SigningError, match="ensure_ed25519_key"):
            sign_request([], AGENT_URL, "GET", "https://robutler.ai/x")
        with pytest.raises(SigningError, match="one nonce per key"):
            sign_request([key, fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", nonces=[FIXED_NONCE])
        # One nonce PER key: the platform spends each nonce once, so two labels
        # sharing one are `signature_replayed` on the second (2026-09-19).
        with pytest.raises(SigningError, match="every label needs its own nonce"):
            sign_request([key, fresh_key()], AGENT_URL, "GET", "https://robutler.ai/x", nonces=[FIXED_NONCE, FIXED_NONCE])
        with pytest.raises(SigningError, match="non-empty string"):
            sign_request([key], AGENT_URL, "GET", "https://robutler.ai/x", nonces=[""])
        with pytest.raises(SigningError):
            sign_request([key], "/agents/mini", "GET", "https://robutler.ai/x")
        with pytest.raises(SigningError):
            sign_request([key], AGENT_URL, "GET", "/x")


# ---------------------------------------------------------------------------
# Design section 2.7, byte for byte
# ---------------------------------------------------------------------------


EXAMPLE_SIGNATURE_INPUT = (
    'sig1=("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="sig1")'
    f';created={CREATED};expires={EXPIRES};keyid="{B14_THUMBPRINT}";alg="ed25519";nonce="{FIXED_NONCE}";tag="web-bot-auth"'
)

EXAMPLE_BASE = "\n".join(
    [
        '"@method": POST',
        '"@authority": robutler.ai',
        '"@path": /api/auth/cli/token',
        '"@query": ?',
        f'"content-digest": {DIGEST_OF_EMPTY_OBJECT}',
        '"signature-agent";key="sig1": "https://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri',
        f'"@signature-params": {EXAMPLE_SIGNATURE_INPUT[len("sig1="):]}',
    ]
)


class TestDesignExample:
    """Section 2.7 with `<thumbprint>` the B.1.4 thumbprint and `<b64>` the fixed nonce."""

    def test_headers_and_base_are_reproduced(self):
        key = b14_key()
        signed = sign_request([key], AGENT_URL, "POST", PLATFORM_URL, b"{}", created=CREATED, nonces=[FIXED_NONCE])
        assert signed.headers["Content-Digest"] == DIGEST_OF_EMPTY_OBJECT
        assert signed.headers["Signature-Agent"] == 'sig1="https://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri'
        assert signed.headers["Signature-Input"] == EXAMPLE_SIGNATURE_INPUT
        assert signed.signatures[0].base == EXAMPLE_BASE
        assert "\n" not in signed.headers["Signature-Input"]
        assert not EXAMPLE_BASE.endswith("\n")
        # Ed25519 is deterministic, so the signature bytes are a fixed value
        # every other implementation of this profile must produce too.
        assert signed.headers["Signature"] == (
            "sig1=:CjoHhKjA3wQEhEH/S/atFm0zCnqGtAoY3wCVAHcqs5HLUG15VAQcCKU9dVBM4Xt+4LdB9IW5WnHpukoD35vAAA==:"
        )
        verify_signed_request(signed.headers, "POST", PLATFORM_URL, b"{}", public_keys(key))


# ---------------------------------------------------------------------------
# WebBotAuth over a real httpx client
# ---------------------------------------------------------------------------


def _capture():
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), dict(request.headers), request.content))
        return httpx.Response(200, json={"ok": True})

    return seen, handler


class TestWebBotAuth:
    def test_requires_the_request_body(self):
        assert WebBotAuth.requires_request_body is True

    def test_signs_a_sync_post_over_the_bytes_sent(self):
        key = fresh_key()
        seen, handler = _capture()
        with httpx.Client(transport=httpx.MockTransport(handler), auth=WebBotAuth([key], AGENT_URL)) as client:
            response = client.post(PLATFORM_URL, json={})
        assert response.status_code == 200
        method, url, headers, content = seen[0]
        assert content == b"{}"
        assert "authorization" not in headers
        verified = verify_signed_request(headers, method, url, content, public_keys(key))
        assert verified["sig1"]["keyid"] == key.thumbprint

    async def test_signs_an_async_get_without_a_digest(self):
        key = fresh_key()
        seen, handler = _capture()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=WebBotAuth([key], AGENT_URL)) as client:
            await client.get("https://robutler.ai:8443/api/feed?cursor=1")
        method, url, headers, content = seen[0]
        assert content == b""
        assert "content-digest" not in headers
        verified = verify_signed_request(headers, method, url, content, public_keys(key))
        assert verified["sig1"]["base"].splitlines()[1:4] == ['"@authority": robutler.ai:8443', '"@path": /api/feed', '"@query": ?cursor=1']

    def test_a_streamed_body_is_read_before_it_is_digested(self):
        key = fresh_key()
        seen, handler = _capture()

        def chunks():
            yield b'{"part":'
            yield b" 1}"

        with httpx.Client(transport=httpx.MockTransport(handler), auth=WebBotAuth([key], AGENT_URL)) as client:
            client.send(client.build_request("POST", PLATFORM_URL, content=chunks()))
        method, url, headers, content = seen[0]
        assert content == b'{"part": 1}'
        verify_signed_request(headers, method, url, content, public_keys(key))

    def test_a_resent_request_is_signed_afresh(self):
        key = fresh_key()
        seen, handler = _capture()
        auth = WebBotAuth([key], AGENT_URL)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            request = client.build_request("POST", PLATFORM_URL, json={})
            client.send(request, auth=auth)
            client.send(request, auth=auth)
        first, second = seen[0][2], seen[1][2]
        assert first["signature"] != second["signature"]
        assert first["signature-input"] != second["signature-input"]
        for headers in (first, second):
            assert headers["signature-input"].count("sig1=") == 1
            verify_signed_request(headers, "POST", PLATFORM_URL, b"{}", public_keys(key))

    def test_form_and_rotation_reach_the_wire(self):
        current, previous = fresh_key(), fresh_key()
        seen, handler = _capture()
        auth = WebBotAuth([current, previous], AGENT_URL, form="dictionary-untyped", lifetime=120)
        with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as client:
            client.post(PLATFORM_URL, json={})
        method, url, headers, content = seen[0]
        assert headers["signature-agent"] == (
            'sig1="https://agent.example/agents/mini/.well-known/jwks.json", sig2="https://agent.example/agents/mini/.well-known/jwks.json"'
        )
        verified = verify_signed_request(headers, method, url, content, public_keys(current, previous))
        assert list(verified) == ["sig1", "sig2"]
        assert all(v["expires"] - v["created"] == 120 for v in verified.values())

    def test_constructor_refusals(self):
        with pytest.raises(SigningError):
            WebBotAuth([], AGENT_URL)
        with pytest.raises(SigningError):
            WebBotAuth([fresh_key()], "/agents/mini")
        with pytest.raises(SigningError):
            WebBotAuth([fresh_key()], AGENT_URL, form="cimd")


# ---------------------------------------------------------------------------
# 2026-09-18, W2 review: one spelling of the agent URL, the signable rule
# shared with registration, and the held-key cap
# ---------------------------------------------------------------------------

from webagents.crypto.http_signature import (  # noqa: E402 - grouped with its cases
    MAX_HELD_KEYS,
    assert_signable_agent_url,
    canonical_agent_url,
    normalize_agent_url,
    signature_agent_value,
)


class TestAgentUrlSpelling:
    """The platform derives the principal through a WHATWG parse (host
    lowercased, default port dropped) and compares the card to it by string
    equality; the signer and the card must therefore agree on that one
    spelling, or a verifying signature is refused `card_not_self_naming`."""

    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("https://Agents.Example.com:443/agents/a", "https://agents.example.com/agents/a"),
            ("HTTPS://AGENTS.EXAMPLE.COM/agents/a/", "https://agents.example.com/agents/a"),
            ("http://Host.Example:80/x", "http://host.example/x"),
            ("https://host.example:8443/x", "https://host.example:8443/x"),
            ("https://[2606:4700::1111]:443/agents/a", "https://[2606:4700::1111]/agents/a"),
            ("https://agent.example/agents/mini", "https://agent.example/agents/mini"),
            ("https://agent.example", "https://agent.example"),
            # Not an absolute http(s) URL: only the trailing slashes go.
            ("/agents/mini/", "/agents/mini"),
            ("", ""),
        ],
    )
    def test_canonical_agent_url(self, raw, canonical):
        assert canonical_agent_url(raw) == canonical
        assert canonical_agent_url(canonical) == canonical  # idempotent

    def test_normalize_returns_the_canonical_spelling(self):
        assert normalize_agent_url("https://Agents.Example.com:443/agents/a/") == "https://agents.example.com/agents/a"
        with pytest.raises(SigningError, match="port"):
            normalize_agent_url("https://agent.example:99999/agents/a")

    def test_the_wire_value_and_the_signature_use_the_canonical_spelling(self):
        key = fresh_key()
        raw = "https://Agent.Example:443/agents/mini"
        signed = sign_request([key], raw, "POST", PLATFORM_URL, b"{}", created=CREATED)
        expected = signature_agent_value(AGENT_URL, DEFAULT_SIGNATURE_AGENT_FORM)
        assert signed.headers["Signature-Agent"] == f'sig1="{expected}";type=jwks_uri'
        assert f'"signature-agent";key="sig1": "{expected}";type=jwks_uri' in signed.signatures[0].base
        # Byte-identical to signing for the canonical spelling directly.
        direct = sign_request([key], AGENT_URL, "POST", PLATFORM_URL, b"{}", created=CREATED, nonces=[signed.signatures[0].nonce])
        assert direct.signatures[0].base == signed.signatures[0].base


class TestSignableAgentUrl:
    """`assert_signable_agent_url`, the TypeScript `assertSignableAgentUrl`
    word for word: loopback by name refused always, plain http only under
    `allow_http` or ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1."""

    @pytest.fixture(autouse=True)
    def _no_allow_private(self, monkeypatch):
        monkeypatch.delenv("ROBUTLER_AGENT_URL_ALLOW_PRIVATE", raising=False)

    def test_https_passes_canonical(self):
        assert assert_signable_agent_url("https://Agent.Example:443/agents/mini") == AGENT_URL

    @pytest.mark.parametrize(
        "url",
        [
            "https://localhost/agents/mini",
            "http://localhost:8000/agents/mini",
            "https://mini.localhost/agents/mini",
            "https://127.0.0.1/agents/mini",
            "http://[::1]:8000/agents/mini",
        ],
    )
    def test_loopback_is_refused_by_name_even_when_http_is_allowed(self, url):
        with pytest.raises(SigningError, match="loopback"):
            assert_signable_agent_url(url, allow_http=True)

    def test_plain_http_is_refused_by_default_and_names_the_switch(self):
        with pytest.raises(SigningError, match="ROBUTLER_AGENT_URL_ALLOW_PRIVATE"):
            assert_signable_agent_url("http://agent.example/agents/mini")
        with pytest.raises(SigningError, match="ROBUTLER_AGENT_URL_ALLOW_PRIVATE"):
            sign_request([fresh_key()], "http://agent.example/agents/mini", "GET", "https://robutler.ai/x")
        with pytest.raises(SigningError, match="ROBUTLER_AGENT_URL_ALLOW_PRIVATE"):
            WebBotAuth([fresh_key()], "http://agent.example/agents/mini")

    def test_plain_http_is_allowed_by_argument(self):
        assert assert_signable_agent_url("http://agent.example/agents/mini", allow_http=True) == "http://agent.example/agents/mini"
        signed = sign_request([fresh_key()], "http://agent.example/agents/mini", "GET", "https://robutler.ai/x", allow_http=True)
        assert signed.headers["Signature-Agent"] == 'sig1="http://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri'
        assert WebBotAuth([fresh_key()], "http://agent.example/agents/mini", allow_http=True).agent_url == "http://agent.example/agents/mini"

    def test_plain_http_is_allowed_by_the_environment_the_platform_reads(self, monkeypatch):
        monkeypatch.setenv("ROBUTLER_AGENT_URL_ALLOW_PRIVATE", "1")
        assert assert_signable_agent_url("http://agent.example/agents/mini") == "http://agent.example/agents/mini"
        sign_request([fresh_key()], "http://agent.example/agents/mini", "GET", "https://robutler.ai/x")
        # An explicit False wins over the environment.
        with pytest.raises(SigningError, match="not https"):
            assert_signable_agent_url("http://agent.example/agents/mini", allow_http=False)


class TestHeldKeyCap:
    """The platform verifies at most two labels per request and refuses the
    request above that, so a third held key is refused before signing."""

    def test_the_cap_is_the_platforms(self):
        assert MAX_HELD_KEYS == 2

    def test_two_keys_sign_and_three_are_refused(self):
        keys = [fresh_key(), fresh_key(), fresh_key()]
        signed = sign_request(keys[:2], AGENT_URL, "GET", "https://robutler.ai/x")
        assert [s.label for s in signed.signatures] == ["sig1", "sig2"]
        with pytest.raises(SigningError, match="at most 2"):
            sign_request(keys, AGENT_URL, "GET", "https://robutler.ai/x")
