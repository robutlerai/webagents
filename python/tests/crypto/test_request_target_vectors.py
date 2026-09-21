"""
Cross-language vectors for the REQUEST TARGET (2026-09-19):
`tests/fixtures/web_bot_auth/vectors-request-target.json`.

The first vector file signs one ordinary URL, so it never noticed that the
two SDKs read `@path` and `@query` differently: the TypeScript signer takes
them from the WHATWG `URL` parser (`pathname`, `search`), which is also how
the platform rebuilds them (`new URL(request.url)`, lib/auth/agent-auth.ts
step 10), while this signer took them from `urlsplit` or `httpx`'s
`raw_path` untouched. A dot-segment path therefore signed `/a/../b` here and
`/b` there, and the platform verified `/b`. This file pins the cases the
first one lacks, signed with the same RFC 9421 Appendix B.1.4 test key and
the same fixed `created`, `expires` and `nonce`:

  * dot segments, plain and `%2e`-spelled, in the middle and in last position
  * a request with a body (`Content-Digest` covered) and a query string
  * a query that needs the WHATWG query percent-encoding, under a host and
    port that need the authority rule
  * dot segments together with a body and a query
  * an empty query (`?` alone), a backslash path

Each case records the inputs, the normalised `target` (what goes in the base
AND on the wire), every header and every base line. The TypeScript twin is
`typescript/tests/unit/crypto/http-signature-request-target.test.ts`; both
regenerate the whole document and compare it with the file byte for byte,
and neither overwrites it. The top-level `source` sentence is prose and is
read back from the file, as in `test_vectors.py`. The three older vector
files are untouched: the portal reads them.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import (
    SIGNATURE_ALG,
    SIGNATURE_TAG,
    SigningKey,
    WebBotAuth,
    apply_signed_target,
    request_target,
    sign_request,
)

from .support import verify_signed_request

VECTORS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "web_bot_auth" / "vectors-request-target.json"

KEY_D = "n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU"
KEY_X = "JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs"
KEY_THUMBPRINT = "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U"
AGENT_URL = "https://agent.example/agents/mini"
FORM = "dictionary-typed"
LABEL = "sig1"
CREATED = 1758067200
EXPIRES = 1758067260
NONCE = base64.b64encode(bytes(range(64))).decode()

#: `(id, method, url, body)`. ASCII only: the file is compared byte for byte
#: across `json.dumps` and `JSON.stringify`, which differ on non-ASCII.
CASES = (
    ("dot-segments", "GET", "https://robutler.ai/api/./agents/x/../mini/feed", ""),
    ("encoded-dot-segments", "GET", "https://robutler.ai/api/agents/%2e%2E/agents/mini/%2e/feed/..", ""),
    ("body-and-query", "POST", "https://robutler.ai/api/mpp/credits?amount=500&note=a%20b", '{"amount":500}'),
    ("query-encoding", "GET", "https://Robutler.AI:443/api/search?q=a b'c\"d<e>&k=%7e", ""),
    ("dot-segments-body-and-query", "POST", "https://robutler.ai/api/agents/../mpp/./credits?x=1", "{}"),
    ("empty-query", "GET", "https://robutler.ai/api/feed?", ""),
    ("backslash-path", "GET", "https://robutler.ai\\api\\feed\\..\\me", ""),
)

#: What the WHATWG parser makes of each case: `(path, query, url)`. Spelled
#: out so a change to the normaliser fails here and not only against the file.
EXPECTED_TARGETS = {
    "dot-segments": ("/api/agents/mini/feed", "?", "https://robutler.ai/api/agents/mini/feed"),
    "encoded-dot-segments": ("/api/agents/mini/", "?", "https://robutler.ai/api/agents/mini/"),
    "body-and-query": ("/api/mpp/credits", "?amount=500&note=a%20b", "https://robutler.ai/api/mpp/credits?amount=500&note=a%20b"),
    "query-encoding": ("/api/search", "?q=a%20b%27c%22d%3Ce%3E&k=%7e", "https://robutler.ai/api/search?q=a%20b%27c%22d%3Ce%3E&k=%7e"),
    "dot-segments-body-and-query": ("/api/mpp/credits", "?x=1", "https://robutler.ai/api/mpp/credits?x=1"),
    "empty-query": ("/api/feed", "?", "https://robutler.ai/api/feed?"),
    "backslash-path": ("/api/me", "?", "https://robutler.ai/api/me"),
}

PYTHON_SOURCE = (
    "Request-target cross-language vectors, written 2026-09-19 by the Python SDK signer "
    "(webagents/python/tests/crypto/test_request_target_vectors.py). Inputs: the RFC 9421 Appendix B.1.4 "
    "Ed25519 test key (denylisted by the platform profile), created 1758067200, expires 1758067260, nonce = "
    "bytes 0x00..0x3f in standard base64, agent URL https://agent.example/agents/mini, the dictionary-typed "
    "Signature-Agent form. Each case gives a method, a URL as a caller might spell it and a body; `target` is "
    "what the WHATWG URL parser makes of the URL, which is what both SDKs sign as @authority, @path and @query, "
    "what they put on the wire, and what the platform rebuilds from the request it received. A consumer signs "
    "the inputs and compares the target, every header and every base line; Ed25519 is deterministic so the "
    "signature bytes are pinned too."
)


def signing_key() -> SigningKey:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(KEY_D + "="))
    return SigningKey.from_private_key(key)


def build_vectors(source: str = PYTHON_SOURCE) -> dict:
    key = signing_key()
    assert key.thumbprint == KEY_THUMBPRINT
    cases = []
    for case_id, method, url, body in CASES:
        signed = sign_request(
            [key], AGENT_URL, method, url, body.encode("ascii"), form=FORM, label=LABEL, created=CREATED, nonces=[NONCE]
        )
        assert signed.signatures[0].expires == EXPIRES
        assert signed.target is not None
        headers = {}
        if "Content-Digest" in signed.headers:
            headers["content-digest"] = signed.headers["Content-Digest"]
        headers["signature-agent"] = signed.headers["Signature-Agent"]
        headers["signature-input"] = signed.headers["Signature-Input"]
        headers["signature"] = signed.headers["Signature"]
        cases.append(
            {
                "id": case_id,
                "request": {"method": method, "url": url, "body": body},
                "target": {
                    "authority": signed.target.authority,
                    "path": signed.target.path,
                    "query": signed.target.query,
                    "url": signed.target.url,
                },
                "headers": headers,
                "signatureBaseLines": signed.signatures[0].base.split("\n"),
            }
        )
    return {
        "source": source,
        "key": {
            "name": "rfc9421-B.1.4",
            "publicJwk": {"kty": "OKP", "crv": "Ed25519", "x": KEY_X},
            "privateJwk": {"kty": "OKP", "crv": "Ed25519", "x": KEY_X, "d": KEY_D},
            "thumbprint": KEY_THUMBPRINT,
        },
        "agentUrl": AGENT_URL,
        "form": FORM,
        "params": {
            "label": LABEL,
            "created": CREATED,
            "expires": EXPIRES,
            "nonce": NONCE,
            "alg": SIGNATURE_ALG,
            "tag": SIGNATURE_TAG,
        },
        "cases": cases,
    }


def serialize(vectors: dict) -> str:
    return json.dumps(vectors, indent=2) + "\n"


def test_every_case_signs_the_whatwg_target_and_verifies():
    built = build_vectors()
    public = {KEY_THUMBPRINT: signing_key().private_key.public_key()}
    assert [c["id"] for c in built["cases"]] == [c[0] for c in CASES]
    for case in built["cases"]:
        path, query, url = EXPECTED_TARGETS[case["id"]]
        assert case["target"] == {"authority": "robutler.ai", "path": path, "query": query, "url": url}, case["id"]
        lines = case["signatureBaseLines"]
        assert lines[1] == '"@authority": robutler.ai'
        assert lines[2] == f'"@path": {path}'
        assert lines[3] == f'"@query": {query}'
        body = case["request"]["body"].encode("ascii")
        assert ("content-digest" in case["headers"]) == bool(body)
        # A verifier that received the NORMALISED URL (what is sent) rebuilds the same base.
        verified = verify_signed_request(case["headers"], case["request"]["method"], url, body, public)
        assert verified[LABEL]["base"] == "\n".join(lines)


def test_the_url_sent_is_the_url_signed():
    """`WebBotAuth` moves the request onto the signed spelling, so a verifier
    that reads the request line (the platform) rebuilds the signed base. Until
    2026-09-19 `%2e%2e` was signed AND sent as written, and the platform's
    parser resolved it to another path."""
    key = signing_key()
    public = {KEY_THUMBPRINT: key.private_key.public_key()}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.raw_path.decode("ascii"), dict(request.headers), request.content))
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler), auth=WebBotAuth([key], AGENT_URL)) as client:
        for case_id, method, url, body in CASES:
            client.request(method, url, content=body.encode("ascii") or None)
    assert len(seen) == len(CASES)
    for (case_id, _, _, body), (method, raw_path, headers, content) in zip(CASES, seen):
        path, query, url = EXPECTED_TARGETS[case_id]
        assert raw_path == url[len("https://robutler.ai") :], case_id
        base = verify_signed_request(headers, method, "https://robutler.ai" + raw_path, content, public)[LABEL]["base"]
        assert f'"@path": {path}\n"@query": {query}\n' in base, case_id


def test_a_string_and_the_httpx_url_built_from_it_sign_the_same_target():
    for _, method, url, _ in CASES:
        assert request_target(method, httpx.URL(url)) == request_target(method, url)


def test_apply_signed_target_leaves_an_ordinary_url_alone_and_moves_a_non_canonical_one():
    ordinary = httpx.Request("GET", "https://robutler.ai/api/feed?x=1")
    before = ordinary.url
    apply_signed_target(ordinary, request_target("GET", str(ordinary.url)))
    assert ordinary.url is before

    spelled = httpx.Request("GET", "https://robutler.ai/api/%2e%2e/feed")
    assert spelled.url.raw_path == b"/api/%2e%2e/feed"
    apply_signed_target(spelled, request_target("GET", str(spelled.url)))
    assert spelled.url.raw_path == b"/feed"
    assert spelled.url.host == "robutler.ai"


def test_the_fixture_file_matches_byte_for_byte():
    """Writes the file when it does not exist; compares, never overwrites, when it does."""
    if not VECTORS_PATH.exists():
        VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTORS_PATH.write_text(serialize(build_vectors()), encoding="ascii", newline="\n")

    committed = VECTORS_PATH.read_bytes()
    source = json.loads(committed)["source"]
    assert isinstance(source, str) and source
    produced = serialize(build_vectors(source)).encode("ascii")
    assert committed == produced, (
        f"{VECTORS_PATH} differs from what this signer produces. The file is shared with the "
        "TypeScript SDK; do not overwrite it, find which side drifted."
    )
    assert json.loads(committed) == build_vectors(source)
