"""
The inbound Web Bot Auth verifier (ADR-0045 section 3).

`verify-cases.json` (written by `make_verify_cases.py`) holds signed requests and
the outcome both SDK verifiers must reach; the TypeScript suite runs the same file
(tests/unit/crypto/web-bot-auth-verify.test.ts). Then: the SDK's own signer
round-trips through the verifier, the vectors the portal verifies still build
the same base here, and the key-set fetcher's refusals.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import SigningKey, ed25519_public_jwk, sign_request
from webagents.crypto.structured_fields import StructuredFieldParseError, parse_dictionary, parse_item
from webagents.crypto.web_bot_auth_verify import (
    Discovery,
    InboundRequest,
    KeySetFetcher,
    KeySetOutcome,
    MemoryNonceStore,
    parse_key_set,
    verify_web_bot_auth,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web_bot_auth"
CASES = json.loads((FIXTURES / "verify-cases.json").read_text())


class _StubKeySets:
    def __init__(self, keys):
        self.keys = keys
        self.asked = []

    async def get(self, discovery):
        self.asked.append(discovery)
        keys, reason = parse_key_set({"keys": self.keys}, well_known_directory=discovery.type == "directory")
        return KeySetOutcome(keys=tuple(keys or ()), ttl_s=300) if keys else KeySetOutcome(code="key_set_invalid", reason=reason)


async def _verify(case, nonces=None):
    request = case["request"]
    inbound = InboundRequest(
        method=request["method"],
        target=request["target"],
        headers=request["headers"],
        body=base64.b64decode(request["body"]),
    )
    return await verify_web_bot_auth(
        inbound,
        authorities=[CASES["authority"]],
        scheme=CASES["scheme"],
        key_sets=_StubKeySets(case.get("keys") or [CASES["signerKey"]]),
        nonces=nonces or MemoryNonceStore(),
        now=case["now"],
    )


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["name"])
def test_the_shared_cases(case):
    async def run():
        nonces = MemoryNonceStore()
        outcome = None
        for _ in range(case.get("repeat", 1)):
            outcome = await _verify(case, nonces)
        return outcome

    outcome = asyncio.run(run())
    expect = case["expect"]
    if expect["ok"]:
        assert outcome.refusal is None, outcome.refusal
        assert outcome.agent.principal == expect["principal"]
        assert list(outcome.agent.thumbprints) == expect["thumbprints"]
        assert outcome.agent.identifier == expect["identifier"]
    else:
        assert outcome.agent is None
        assert outcome.refusal.code == expect["code"]
        assert outcome.refusal.description == expect["description"]


def _key():
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())


def test_the_sdk_signer_round_trips():
    key = _key()
    signed = sign_request([key], "https://caller.example/agents/scout", "POST", "https://agent.example/agents/mini/chat/completions?stream=1", b"{}")
    headers = {"host": "agent.example", **signed.headers}
    entry = ed25519_public_jwk(key.private_key.public_key())
    outcome = asyncio.run(
        verify_web_bot_auth(
            InboundRequest("POST", "/agents/mini/chat/completions?stream=1", headers, b"{}"),
            authorities=["agent.example"],
            scheme="https",
            key_sets=_StubKeySets([entry]),
            nonces=MemoryNonceStore(),
        )
    )
    assert outcome.ok, outcome.refusal
    assert outcome.agent.principal == "https://caller.example/agents/scout"
    assert outcome.agent.thumbprints == (key.thumbprint,)


def test_the_portal_vectors_verify_here():
    """The vectors the portal verifies (the W2 cross-language set) build the same
    base here and verify under their key; only the test-key denylist would stop
    them at discovery, which is the key-set fetcher's job, not the base's."""
    vectors = json.loads((FIXTURES / "vectors.json").read_text())
    request = vectors["request"]
    for vector in vectors["vectors"]:
        headers = {"host": request["authority"], **{k.lower(): v for k, v in vector["headers"].items()}}
        body = request["body"].encode() if isinstance(request["body"], str) else b""
        outcome = asyncio.run(
            verify_web_bot_auth(
                InboundRequest(request["method"], request["path"] + (request["query"] if request["query"] != "?" else ""), headers, body),
                authorities=[request["authority"]],
                scheme="https",
                key_sets=_AnyKeySets(vectors["key"]["publicJwk"], vectors["key"]["thumbprint"]),
                nonces=MemoryNonceStore(),
                now=vectors["params"]["created"],
            )
        )
        assert outcome.ok, (vector["id"], outcome.refusal)


class _AnyKeySets:
    """Answers with one key whatever the discovery, skipping the denylist the
    real fetcher applies, so the vectors' B.1.4 key can be used."""

    def __init__(self, jwk, thumbprint):
        from webagents.crypto.web_bot_auth_verify import DiscoveredKey

        self.key = DiscoveredKey(thumbprint, jwk["x"])

    async def get(self, discovery):
        return KeySetOutcome(keys=(self.key,), ttl_s=300)


class TestStructuredFields:
    def test_a_dictionary_with_an_inner_list(self):
        (label, member), = parse_dictionary('sig1=("@method" "signature-agent";key="sig1");created=1;tag="web-bot-auth"')
        assert label == "sig1"
        assert [item.value for item in member.items] == ["@method", "signature-agent"]
        assert dict(member.params) == {"created": 1, "tag": "web-bot-auth"}

    @pytest.mark.parametrize(
        "text",
        [
            "a=1, a=2",
            "a=1;x;x",
            "a=1.5",
            "a=@1700000000",
            'a=%"x"',
            "a=:YWI:",
            "a=:YWJjZA:",
            "a=1,",
            "A=1",
            'a="\x01"',
            "a=(1 2",
            "a=" + "1" * 16,
            ", ".join(f"k{i}=1" for i in range(17)),
            "a=" + "x" * 9000,
        ],
    )
    def test_refusals(self, text):
        with pytest.raises(StructuredFieldParseError):
            parse_dictionary(text)

    def test_a_legacy_string(self):
        assert parse_item('"https://caller.example"').value == "https://caller.example"


# -- the key-set fetcher, against a local server ------------------------------------------------


class _KeyRoutes(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    key_set = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path
        if path == "/agents/scout/.well-known/jwks.json":
            body, ctype, status, extra = json.dumps(self.key_set).encode(), "application/json", 200, [("Cache-Control", "max-age=10")]
        elif path == "/redirect/.well-known/jwks.json":
            body, ctype, status, extra = b"", "text/plain", 302, [("Location", "/agents/scout/.well-known/jwks.json")]
        elif path == "/big/.well-known/jwks.json":
            body, ctype, status, extra = b"x" * (70 * 1024), "application/json", 200, []
        elif path == "/notjson/.well-known/jwks.json":
            body, ctype, status, extra = b"<html>", "text/html", 200, []
        elif path == "/.well-known/http-message-signatures-directory":
            body, ctype, status, extra = json.dumps(self.key_set).encode(), "application/json", 200, []
        else:
            body, ctype, status, extra = b"no", "text/plain", 404, []
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for name, value in extra:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def key_server():
    key = _key()
    entry = ed25519_public_jwk(key.private_key.public_key())
    _KeyRoutes.key_set = {"keys": [dict(entry, kid=key.thumbprint, use="sig")]}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _KeyRoutes)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], key
    httpd.shutdown()


def _jwks(port, prefix):
    url = f"http://127.0.0.1:{port}{prefix}/.well-known/jwks.json"
    return Discovery("jwks_uri", url.rsplit("/.well-known", 1)[0], url, url)


class TestKeySetFetcher:
    def test_fetches_and_parses(self, key_server):
        port, key = key_server
        outcome = asyncio.run(KeySetFetcher(allow_private=True).get(_jwks(port, "/agents/scout")))
        assert outcome.ok and outcome.keys[0].thumbprint == key.thumbprint
        assert outcome.ttl_s == 300  # max-age=10 is clamped up to the floor

    def test_private_addresses_need_the_switch(self, key_server):
        port, _ = key_server
        outcome = asyncio.run(KeySetFetcher().get(_jwks(port, "/agents/scout")))
        assert outcome.code == "key_set_unreachable"
        assert outcome.reason == "127.0.0.1 is not a public address, so it is not called."

    def test_no_redirects(self, key_server):
        port, _ = key_server
        outcome = asyncio.run(KeySetFetcher(allow_private=True).get(_jwks(port, "/redirect")))
        assert (outcome.code, outcome.reason) == ("key_set_unreachable", "it answered with a redirect, which is not followed")

    def test_size_cap(self, key_server):
        port, _ = key_server
        outcome = asyncio.run(KeySetFetcher(allow_private=True).get(_jwks(port, "/big")))
        assert (outcome.code, outcome.reason) == ("key_set_invalid", "it is larger than 64 KiB")

    def test_not_json(self, key_server):
        port, _ = key_server
        outcome = asyncio.run(KeySetFetcher(allow_private=True).get(_jwks(port, "/notjson")))
        assert (outcome.code, outcome.reason) == ("key_set_invalid", "it is not JSON")

    def test_a_directory_needs_its_media_type(self, key_server):
        port, _ = key_server
        origin = f"http://127.0.0.1:{port}"
        url = f"{origin}/.well-known/http-message-signatures-directory"
        discovery = Discovery("directory", origin, url, url, "application/http-message-signatures-directory+json")
        outcome = asyncio.run(KeySetFetcher(allow_private=True).get(discovery))
        assert outcome.code == "key_set_invalid"
        assert outcome.reason == "a directory must be served as application/http-message-signatures-directory+json"

    def test_the_rfc_test_key_poisons_the_set(self):
        vectors = json.loads((FIXTURES / "vectors.json").read_text())
        keys, reason = parse_key_set({"keys": [vectors["key"]["publicJwk"]]}, well_known_directory=False)
        assert keys is None and reason == "it carries a known test key (RFC 9421 Appendix B.1)"
