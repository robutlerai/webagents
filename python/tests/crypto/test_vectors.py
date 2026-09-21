"""
Cross-language signing vectors (ADR 0038 step 5, W2 design section 10.3,
2026-09-17): `tests/fixtures/web_bot_auth/vectors.json`.

One file, three consumers. This test signs a fixed request with the RFC 9421
Appendix B.1.4 Ed25519 test key, a fixed `created`, `expires` and `nonce`,
once per `Signature-Agent` form, and holds the result against the fixture.
The TypeScript SDK's signer test produces the same vectors from the same
inputs and compares byte for byte; the portal verifies every vector through
its pure signature layer. Because Ed25519 is deterministic, the signature
bytes themselves are pinned, not only the base.

WHO WRITES THE FILE. Whichever SDK's test runs first on a tree where the
file does not exist writes it; every later run, in either language,
regenerates the document and fails on any byte that differs, and never
overwrites. On 2026-09-17 the TypeScript signer wrote it first, so this test
reproduces its layout: the serialisation is `JSON.stringify(value, null, 2)`
plus one trailing newline, which `json.dumps(value, indent=2) + "\\n"`
reproduces exactly for this ASCII-only content, and the key order below is
the file's. The top-level `source` sentence is prose about the file, not a
vector: it is read back from the file rather than pinned here, so a reworded
sentence on one side does not read as a signing difference on the other.
Everything else in the document, inputs and outputs, is compared byte for
byte.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import (
    SIGNATURE_AGENT_FORMS,
    SIGNATURE_ALG,
    SIGNATURE_TAG,
    SigningKey,
    content_digest,
    request_target,
    sign_request,
)

from .support import verify_signed_request
from .test_http_signature import EXAMPLE_BASE, EXAMPLE_SIGNATURE_INPUT

VECTORS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "web_bot_auth" / "vectors.json"

KEY_D = "n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU"
KEY_X = "JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs"
KEY_THUMBPRINT = "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U"
METHOD = "POST"
URL = "https://robutler.ai/api/auth/cli/token"
BODY = b"{}"
AGENT_URL = "https://agent.example/agents/mini"
LABEL = "sig1"
CREATED = 1758067200
EXPIRES = 1758067260
NONCE = base64.b64encode(bytes(range(64))).decode()

# The sentence written when THIS test creates the file on a fresh tree.
PYTHON_SOURCE = (
    "W2 design section 10.3 cross-language vectors, written 2026-09-17 by the Python SDK signer "
    "(webagents/python/tests/crypto/test_vectors.py). Inputs: the RFC 9421 Appendix B.1.4 Ed25519 "
    "test key (denylisted by the platform profile, P section 6.8), POST /api/auth/cli/token at "
    "robutler.ai with body {}, created 1758067200, expires 1758067260, nonce = bytes 0x00..0x3f in "
    "standard base64, agent URL https://agent.example/agents/mini, one vector per Signature-Agent "
    "form. A consumer signs the inputs and compares every header and every base line; Ed25519 is "
    "deterministic so the signature bytes are pinned too."
)


def signing_key() -> SigningKey:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(KEY_D + "="))
    return SigningKey.from_private_key(key)


def build_vectors(source: str = PYTHON_SOURCE) -> dict:
    key = signing_key()
    assert key.thumbprint == KEY_THUMBPRINT
    target = request_target(METHOD, URL)
    vectors = []
    for form in SIGNATURE_AGENT_FORMS:
        signed = sign_request([key], AGENT_URL, METHOD, URL, BODY, form=form, label=LABEL, created=CREATED, nonces=[NONCE])
        assert signed.signatures[0].expires == EXPIRES
        headers = signed.headers
        vectors.append(
            {
                "id": form,
                "form": form,
                "label": LABEL,
                "headers": {
                    "content-digest": headers["Content-Digest"],
                    "signature-agent": headers["Signature-Agent"],
                    "signature-input": headers["Signature-Input"],
                    "signature": headers["Signature"],
                },
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
        "request": {
            "method": METHOD,
            "url": URL,
            "authority": target.authority,
            "path": target.path,
            "query": urlsplit(URL).query,
            "body": BODY.decode("ascii"),
        },
        "params": {
            "label": LABEL,
            "created": CREATED,
            "expires": EXPIRES,
            "nonce": NONCE,
            "alg": SIGNATURE_ALG,
            "tag": SIGNATURE_TAG,
        },
        "vectors": vectors,
    }


def serialize(vectors: dict) -> str:
    return json.dumps(vectors, indent=2) + "\n"


def test_the_vectors_verify_and_match_the_design_example():
    built = build_vectors()
    public = {KEY_THUMBPRINT: signing_key().private_key.public_key()}
    assert [v["form"] for v in built["vectors"]] == list(SIGNATURE_AGENT_FORMS)
    for vector in built["vectors"]:
        verified = verify_signed_request(vector["headers"], METHOD, URL, BODY, public)
        assert verified[LABEL]["base"] == "\n".join(vector["signatureBaseLines"])
        assert vector["headers"]["content-digest"] == content_digest(BODY)
    typed = built["vectors"][0]
    assert typed["headers"]["signature-input"] == EXAMPLE_SIGNATURE_INPUT
    assert "\n".join(typed["signatureBaseLines"]) == EXAMPLE_BASE
    untyped, legacy = built["vectors"][1], built["vectors"][2]
    assert untyped["headers"]["signature-agent"] == 'sig1="https://agent.example/agents/mini/.well-known/jwks.json"'
    assert legacy["headers"]["signature-agent"] == '"https://agent.example"'
    assert '"signature-agent");' in legacy["headers"]["signature-input"]


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
        "TypeScript SDK and the portal; do not overwrite it, find which side drifted."
    )
    # And it round-trips as the same object, so a reader in either language
    # sees the same inputs and outputs.
    assert json.loads(committed) == build_vectors(source)
