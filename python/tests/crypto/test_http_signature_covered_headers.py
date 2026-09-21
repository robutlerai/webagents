"""
Covered headers on the Web Bot Auth signer (machine-purchase design section
6.4; pass P9b, 2026-09-18). The paid retry of a purchase carries
`Payment-Authorization` and `Robutler-Terms-Accepted`, and the platform
admits them only when both are among the signature's covered components.
This suite pins how `sign_request(covered_headers=...)` and
`WebBotAuth(covered_headers=...)` place and value them, what they refuse,
and the cross-language vectors for them. The TypeScript twin is
`webagents/typescript/tests/unit/crypto/http-signature-covered-headers.test.ts`.

THE VECTOR FILE. `tests/fixtures/web_bot_auth/vectors-covered-headers.json`
is a SECOND file beside `vectors.json`, which stays byte-identical. The
TypeScript signer wrote it first (2026-09-18), so this suite REBUILDS THE
WHOLE DOCUMENT from Python-only inputs (the fixed challenge is minted here
with this SDK's JCS, base64url and SHA-256, the credential with this SDK's
encoder, the signatures with this SDK's signer) and compares the bytes, the
`source` sentence alone read back from the file. So one assertion pins the
signer, the credential encoder and the challenge encoding across languages.
It writes the file only when it does not exist and never overwrites it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.agents.skills.robutler.payments_x402.mpp_buyer import (
    base64url_encode,
    encode_mpp_credential,
    jcs_canonicalize,
)
from webagents.crypto.http_signature import (
    COVERED_HEADERS_RESERVED,
    SIGNATURE_AGENT_FORMS,
    SIGNATURE_ALG,
    SIGNATURE_TAG,
    SigningError,
    SigningKey,
    WebBotAuth,
    normalize_covered_headers,
    sign_request,
)

from .covered_support import covered_of, verify_with_covered_headers

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "web_bot_auth"
VECTORS_PATH = FIXTURES / "vectors-covered-headers.json"
FIRST_VECTORS_PATH = FIXTURES / "vectors.json"

KEY_D = "n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU"
KEY_X = "JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs"
KEY_THUMBPRINT = "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U"
AGENT_URL = "https://agent.example/agents/mini"
PURCHASE_URL = "https://robutler.ai/api/mpp/credits"
CREATED = 1758067200
EXPIRES = 1758067260
NONCE = base64.b64encode(bytes(range(64))).decode()
TERMS_VERSION = "2026-07-31"
COVERED = ["payment-authorization", "robutler-terms-accepted"]

# The sentence written when THIS suite creates the file on a fresh tree.
PYTHON_SOURCE = (
    "Machine-purchase design section 6.4 cross-language vectors with covered headers, written 2026-09-18 by the "
    "Python SDK signer (webagents/python/tests/crypto/test_http_signature_covered_headers.py). Inputs as in "
    "vectors.json, POST /api/mpp/credits at robutler.ai with body {}, and two covered headers: "
    "payment-authorization (the MPP credential built from `credential`) and robutler-terms-accepted."
)


def signing_key() -> SigningKey:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(KEY_D + "="))
    return SigningKey.from_private_key(key)


def fixed_challenge() -> dict:
    """The fixed challenge the vector's credential echoes, minted here the way
    the platform mints one, in the TypeScript suite's key order."""
    request = {"amount": "500", "currency": "usd", "methodDetails": {"networkId": "profile_vector", "paymentMethodTypes": ["card"]}}
    opaque = {
        "packId": "mpp_5",
        "userId": "00000000-0000-4000-8000-000000000001",
        "principal": AGENT_URL,
        "thumbprint": KEY_THUMBPRINT,
        "kind": "pack",
        "terms": TERMS_VERSION,
        "origin": hashlib.sha256(b"POST /api/mpp/credits").hexdigest(),
    }
    return {
        "id": base64url_encode(hashlib.sha256(b"mpp covered-headers vector").digest()),
        "realm": "robutler.ai",
        "method": "stripe",
        "intent": "charge",
        "request": base64url_encode(jcs_canonicalize(request)),
        "expires": "2026-09-18T12:05:00.000Z",
        "opaque": base64url_encode(jcs_canonicalize(opaque)),
        "header": "Payment-Authorization",
    }


def build_document(source: str = PYTHON_SOURCE) -> dict:
    key = signing_key()
    assert key.thumbprint == KEY_THUMBPRINT
    challenge = fixed_challenge()
    payload = {"spt": "spt_vector_0000000000000001"}
    credential = encode_mpp_credential(challenge, payload)
    request_headers = {"payment-authorization": credential, "robutler-terms-accepted": TERMS_VERSION}
    vectors = []
    for form in SIGNATURE_AGENT_FORMS:
        signed = sign_request(
            [key], AGENT_URL, "POST", PURCHASE_URL, b"{}",
            form=form, label="sig1", created=CREATED, lifetime=EXPIRES - CREATED, nonces=[NONCE],
            headers=request_headers, covered_headers=COVERED,
        )
        vectors.append(
            {
                "id": form,
                "form": form,
                "label": "sig1",
                "headers": {
                    "content-digest": signed.headers["Content-Digest"],
                    "signature-agent": signed.headers["Signature-Agent"],
                    "signature-input": signed.headers["Signature-Input"],
                    "signature": signed.headers["Signature"],
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
            "method": "POST",
            "url": PURCHASE_URL,
            "authority": "robutler.ai",
            "path": "/api/mpp/credits",
            "query": "",
            "body": "{}",
            "headers": request_headers,
        },
        "coveredHeaders": list(COVERED),
        "credential": {"challenge": challenge, "payload": payload, "header": credential},
        "params": {
            "label": "sig1",
            "created": CREATED,
            "expires": EXPIRES,
            "nonce": NONCE,
            "alg": SIGNATURE_ALG,
            "tag": SIGNATURE_TAG,
        },
        "vectors": vectors,
    }


def serialize(document: dict) -> str:
    # JSON.stringify(value, null, 2) plus one newline, which this reproduces
    # for ASCII-only content (tests/crypto/test_vectors.py, "WHO WRITES").
    return json.dumps(document, indent=2) + "\n"


def public_keys() -> dict:
    return {KEY_THUMBPRINT: signing_key().private_key.public_key()}


# --------------------------------------------------------------------------
# normalize_covered_headers
# --------------------------------------------------------------------------


def test_normalize_trims_lowercases_and_deduplicates_in_first_seen_order():
    assert normalize_covered_headers([" Payment-Authorization ", "robutler-terms-accepted", "PAYMENT-AUTHORIZATION"]) == COVERED
    assert normalize_covered_headers(None) == []
    assert normalize_covered_headers([]) == []


@pytest.mark.parametrize("name", sorted(COVERED_HEADERS_RESERVED) + ["Content-Digest", "Signature"])
def test_normalize_refuses_the_reserved_names(name):
    with pytest.raises(SigningError, match="covered by the signer itself"):
        normalize_covered_headers([name])


@pytest.mark.parametrize("name", ["@method", "@query-param", "bad header", "", "x;y", "caf\u00e9"])
def test_normalize_refuses_derived_components_and_non_field_names(name):
    with pytest.raises(SigningError):
        normalize_covered_headers([name])


def test_normalize_refuses_non_strings():
    with pytest.raises(SigningError, match="header field names"):
        normalize_covered_headers([42])  # type: ignore[list-item]


# --------------------------------------------------------------------------
# sign_request with covered_headers
# --------------------------------------------------------------------------


def _sign(headers, covered, *, body=b"{}", form="dictionary-typed", url=PURCHASE_URL):
    return sign_request(
        [signing_key()], AGENT_URL, "POST", url, body,
        form=form, created=CREATED, nonces=[NONCE], headers=headers, covered_headers=covered,
    )


def test_places_covered_headers_after_content_digest_and_before_the_member():
    headers = {"Payment-Authorization": "Payment abc", "Robutler-Terms-Accepted": TERMS_VERSION}
    signed = _sign(headers, ["Payment-Authorization", "Robutler-Terms-Accepted"])
    assert covered_of(signed.headers["Signature-Input"]) == [
        '"@method"', '"@authority"', '"@path"', '"@query"', '"content-digest"',
        '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent";key="sig1"',
    ]
    verify_with_covered_headers({**headers, **signed.headers}, "POST", PURCHASE_URL, b"{}", public_keys())


def test_values_the_covered_lines_with_the_field_value_trimmed():
    headers = {"payment-authorization": "  Payment abc \t", "robutler-terms-accepted": TERMS_VERSION}
    signed = _sign(headers, COVERED)
    lines = signed.signatures[0].base.split("\n")
    assert lines[5] == '"payment-authorization": Payment abc'
    assert lines[6] == f'"robutler-terms-accepted": {TERMS_VERSION}'
    verify_with_covered_headers({**headers, **signed.headers}, "POST", PURCHASE_URL, b"{}", public_keys())


def test_without_a_body_the_covered_headers_follow_query_directly():
    headers = {"payment-authorization": "Payment abc", "robutler-terms-accepted": TERMS_VERSION}
    signed = _sign(headers, COVERED, body=b"")
    assert "Content-Digest" not in signed.headers
    assert covered_of(signed.headers["Signature-Input"]) == [
        '"@method"', '"@authority"', '"@path"', '"@query"',
        '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent";key="sig1"',
    ]
    verify_with_covered_headers({**headers, **signed.headers}, "POST", PURCHASE_URL, b"", public_keys())


def test_legacy_string_form_puts_covered_headers_before_the_bare_member():
    headers = {"payment-authorization": "Payment abc", "robutler-terms-accepted": TERMS_VERSION}
    signed = _sign(headers, COVERED, form="legacy-string")
    assert covered_of(signed.headers["Signature-Input"])[-3:] == [
        '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent"',
    ]


def test_reads_the_value_case_insensitively_from_a_mapping_and_from_httpx_headers():
    mapping = _sign({"PAYMENT-AUTHORIZATION": "Payment abc"}, ["payment-authorization"])
    headers = _sign(httpx.Headers({"Payment-Authorization": "Payment abc"}), ["Payment-Authorization"])
    assert mapping.headers == headers.headers


def test_httpx_headers_join_a_repeated_field_and_the_joined_value_is_covered():
    headers = httpx.Headers([("robutler-terms-accepted", TERMS_VERSION), ("robutler-terms-accepted", "2026-01-01")])
    signed = _sign(headers, ["robutler-terms-accepted"])
    assert signed.signatures[0].base.split("\n")[5] == f'"robutler-terms-accepted": {TERMS_VERSION}, 2026-01-01'


def test_refuses_a_covered_header_the_message_does_not_carry():
    with pytest.raises(SigningError, match="does not carry it"):
        _sign({"payment-authorization": "Payment abc"}, COVERED)
    with pytest.raises(SigningError, match="does not carry it"):
        _sign(None, COVERED)


@pytest.mark.parametrize("value", ["Payment a\r\nforged: 1", "Payment \u00e9"])
def test_refuses_a_covered_value_that_is_not_printable_ascii(value):
    with pytest.raises(SigningError, match="not printable ASCII"):
        _sign({"payment-authorization": value}, ["payment-authorization"])


def test_with_no_covered_headers_the_signature_is_byte_identical_to_before():
    plain = sign_request([signing_key()], AGENT_URL, "POST", PURCHASE_URL, b"{}", created=CREATED, nonces=[NONCE])
    empty = _sign({"payment-authorization": "Payment abc"}, [])
    none = _sign({"payment-authorization": "Payment abc"}, None)
    assert plain.headers == empty.headers == none.headers
    assert plain.signatures[0].base == empty.signatures[0].base


# --------------------------------------------------------------------------
# WebBotAuth with covered_headers
# --------------------------------------------------------------------------


async def test_web_bot_auth_covers_the_named_headers_on_a_real_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    auth = WebBotAuth([signing_key()], AGENT_URL, covered_headers=["Robutler-Terms-Accepted"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await client.post(PURCHASE_URL, content=b"{}", headers={"Robutler-Terms-Accepted": TERMS_VERSION}, auth=auth)
    [request] = seen
    assert '"robutler-terms-accepted"' in covered_of(request.headers["signature-input"])
    verify_with_covered_headers(request.headers, "POST", str(request.url), request.content, public_keys())


async def test_web_bot_auth_refuses_a_request_that_lacks_a_covered_header():
    auth = WebBotAuth([signing_key()], AGENT_URL, covered_headers=["robutler-terms-accepted"])
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SigningError, match="does not carry it"):
            await client.post(PURCHASE_URL, content=b"{}", auth=auth)


def test_web_bot_auth_refuses_a_reserved_name_at_construction():
    with pytest.raises(SigningError, match="covered by the signer itself"):
        WebBotAuth([signing_key()], AGENT_URL, covered_headers=["content-digest"])


# --------------------------------------------------------------------------
# Cross-language vectors with covered headers (design section 6.4)
# --------------------------------------------------------------------------


def test_the_built_vectors_verify_and_carry_both_covered_lines():
    document = build_document()
    assert [v["form"] for v in document["vectors"]] == list(SIGNATURE_AGENT_FORMS)
    for vector in document["vectors"]:
        headers = {**document["request"]["headers"], **vector["headers"]}
        verified = verify_with_covered_headers(headers, "POST", PURCHASE_URL, b"{}", public_keys())
        assert verified["sig1"]["base"] == "\n".join(vector["signatureBaseLines"])
        assert f'"payment-authorization": {document["credential"]["header"]}' in vector["signatureBaseLines"]
        assert f'"robutler-terms-accepted": {TERMS_VERSION}' in vector["signatureBaseLines"]


def test_the_fixture_file_matches_byte_for_byte():
    """Writes the file when it does not exist; compares, never overwrites, when it does."""
    if not VECTORS_PATH.exists():
        VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTORS_PATH.write_text(serialize(build_document()), encoding="ascii", newline="\n")

    committed = VECTORS_PATH.read_bytes()
    source = json.loads(committed)["source"]
    assert isinstance(source, str) and source
    produced = serialize(build_document(source)).encode("ascii")
    assert committed == produced, (
        f"{VECTORS_PATH} differs from what this SDK produces. The file is shared with the TypeScript SDK "
        "and the portal; do not overwrite it, find which side drifted (signer, JCS, base64url or the credential encoder)."
    )


def test_the_committed_credential_is_what_this_encoder_makes_of_the_committed_challenge():
    committed = json.loads(VECTORS_PATH.read_text(encoding="ascii"))
    credential = committed["credential"]
    assert encode_mpp_credential(credential["challenge"], credential["payload"]) == credential["header"]
    assert committed["request"]["headers"]["payment-authorization"] == credential["header"]


def test_the_first_vector_file_carries_no_covered_headers():
    committed = json.loads(FIRST_VECTORS_PATH.read_text(encoding="ascii"))
    assert "headers" not in committed["request"]
    for vector in committed["vectors"]:
        assert not any(line.startswith('"payment-authorization"') for line in vector["signatureBaseLines"])
