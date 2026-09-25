"""
Writes `verify-cases.json`: signed requests and the outcome BOTH SDK verifiers
must reach (ADR-0045). Deterministic: a fixed seed key, fixed `created` and
nonces, so the file only changes when this script does.

    python tests/fixtures/web_bot_auth/make_verify_cases.py

Each case: the request as the verifying agent receives it (`method`, `target`,
`headers`, `body`), the verifier's `now`, the key set its stub fetcher answers
with (`keys`, default the signer's key), how many times the same request is
verified (`repeat`, the last answer is compared), and `expect`: `{ok: true,
principal, thumbprints, identifier}` or `{ok: false, code, description}`.
The verifier is configured with authority `agent.example`, scheme https.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.crypto.http_signature import SigningKey, ed25519_public_jwk, sign_request

HERE = Path(__file__).resolve().parent
SEED = bytes(range(1, 33))
OTHER_SEED = bytes(range(101, 133))
CALLER = "https://caller.example/agents/scout"
CREATED = 1_790_000_000
NOW = CREATED + 5


def key(seed: bytes) -> SigningKey:
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.from_private_bytes(seed))


SIGNER = key(SEED)
OTHER = key(OTHER_SEED)


def jwk(k: SigningKey) -> dict:
    entry = ed25519_public_jwk(k.private_key.public_key())
    entry["kid"] = k.thumbprint
    entry["use"] = "sig"
    return entry


def signed(method, url, body=b"", *, form="dictionary-typed", nonce="n-0001", agent=CALLER, allow_http=None, sign_body=None):
    out = sign_request(
        [SIGNER],
        agent,
        method,
        url,
        body if sign_body is None else sign_body,
        form=form,
        created=CREATED,
        nonces=[nonce],
        allow_http=allow_http,
    )
    target = out.target.request_line or out.target.path
    headers = {"host": out.target.authority, **{k.lower(): v for k, v in out.headers.items()}}
    return {"method": method, "target": target, "headers": headers, "body": base64.b64encode(body).decode("ascii")}


OK = {
    "ok": True,
    "principal": CALLER,
    "thumbprints": [SIGNER.thumbprint],
    "identifier": f"{CALLER}/.well-known/jwks.json",
}


def refused(code, description):
    return {"ok": False, "code": code, "description": description}


MALFORMED = (
    "Signature-Input and Signature must be RFC 9651 dictionaries whose members match label for label, "
    "each Signature member a 64-byte Ed25519 signature."
)

POST_URL = "https://agent.example/agents/mini/chat/completions"
BODY = b'{"messages":[{"role":"user","content":"hi"}]}'

cases = []


def case(name, request, expect, **extra):
    cases.append({"name": name, "request": request, "now": extra.pop("now", NOW), "expect": expect, **extra})


case("a signed POST verifies", signed("POST", POST_URL, BODY, nonce="n-post"), OK)
case("a signed GET with a query verifies", signed("GET", "https://agent.example/agents/mini/info?x=1&y=2", nonce="n-get"), OK)

r = signed("POST", POST_URL, BODY, nonce="n-path")
r["target"] = "/agents/mini/info"
case("a changed path does not verify", r, refused("signature_invalid", "The signature does not verify under the published key."))

r = signed("POST", POST_URL, BODY, nonce="n-body")
r["body"] = base64.b64encode(b'{"messages":[]}').decode("ascii")
case("a changed body fails the digest", r, refused("content_digest_mismatch", "Content-Digest does not match the request body."))

r = signed("POST", POST_URL, BODY, nonce="n-host")
r["headers"]["host"] = "other.example"
case(
    "a signature for another host is not this agent's",
    r,
    refused(
        "signature_authority_mismatch",
        "This agent verifies signatures made for its own address only; sign the request for agent.example and send it there.",
    ),
)

case(
    "an expired signature",
    signed("POST", POST_URL, BODY, nonce="n-old"),
    refused("signature_expired", "The signature expired more than 60 seconds ago; sign a fresh request."),
    now=CREATED + 60 + 61,
)
case(
    "a signature from the future",
    signed("POST", POST_URL, BODY, nonce="n-future"),
    refused("signature_expired", "The created parameter is more than 60 seconds in the future; check the signer's clock."),
    now=CREATED - 61,
)

r = signed("POST", POST_URL, BODY, nonce="n-tag")
r["headers"]["signature-input"] = r["headers"]["signature-input"].replace(';tag="web-bot-auth"', "")
case(
    "no web-bot-auth tag",
    r,
    refused(
        "signature_malformed",
        'No signature carries tag="web-bot-auth". Each Signature-Input member this agent verifies must set that tag.',
    ),
)

r = signed("POST", POST_URL, BODY, nonce="n-tokentag")
r["headers"]["signature-input"] = r["headers"]["signature-input"].replace(';tag="web-bot-auth"', ";tag=web-bot-auth")
case(
    "a token is not the tag string",
    r,
    refused(
        "signature_malformed",
        'No signature carries tag="web-bot-auth". Each Signature-Input member this agent verifies must set that tag.',
    ),
)

r = signed("POST", POST_URL, BODY, nonce="n-keyid")
r["headers"]["signature-input"] = r["headers"]["signature-input"].replace(f'keyid="{SIGNER.thumbprint}"', 'keyid="short"')
case(
    "a keyid that is no thumbprint",
    r,
    refused(
        "signature_params_invalid",
        "The keyid parameter must be the RFC 7638 SHA-256 thumbprint of the signing key: base64url, no padding, 43 characters.",
    ),
)

r = signed("POST", POST_URL, BODY, nonce="n-created")
r["headers"]["signature-input"] = r["headers"]["signature-input"].replace(f"created={CREATED}", "created=?1")
case(
    "a boolean is not a created time",
    r,
    refused("signature_params_invalid", "The created parameter is required and must be an integer number of seconds since the epoch."),
)

case(
    "a key set without the signing key",
    signed("POST", POST_URL, BODY, nonce="n-unknown"),
    refused("signature_key_unknown", "No key in the published key set has the thumbprint this signature names as keyid."),
    keys=[jwk(OTHER)],
)

case(
    "the same signature twice is a replay",
    signed("POST", POST_URL, BODY, nonce="n-replay"),
    refused("signature_replayed", "This signature's nonce was already used; sign each request afresh."),
    repeat=2,
)

r = signed("POST", POST_URL, BODY, nonce="n-nodigest", sign_body=b"")
case(
    "a body the signature does not cover",
    r,
    refused(
        "signature_coverage_insufficient",
        'The request has a body, so it must send Content-Digest (sha-256 over the body bytes) and cover "content-digest" in the signature.',
    ),
)

r = signed("POST", POST_URL, BODY, nonce="n-garbage")
r["headers"]["signature-input"] = "not a dictionary ("
case("an unparseable Signature-Input", r, refused("signature_malformed", MALFORMED))

r = signed("GET", "https://agent.example/agents/mini/info", nonce="n-legacy", form="legacy-string")
case(
    "the legacy string form resolves to the origin's directory",
    r,
    {
        "ok": True,
        "principal": "https://caller.example",
        "thumbprints": [SIGNER.thumbprint],
        "identifier": "https://caller.example/.well-known/http-message-signatures-directory",
    },
)

case("the untyped dictionary form", signed("GET", "https://agent.example/agents/mini/info", nonce="n-untyped", form="dictionary-untyped"), OK)

case(
    "a plain http Signature-Agent is refused",
    signed("GET", "https://agent.example/agents/mini/info", nonce="n-http", agent="http://caller.example/agents/scout", allow_http=True),
    refused("signature_agent_invalid", "The Signature-Agent member this signature covers is not usable: the value must use https."),
)

r = signed("GET", "https://agent.example/agents/mini/info", nonce="n-cimd")
r["headers"]["signature-agent"] = r["headers"]["signature-agent"].replace(";type=jwks_uri", ";type=cimd").replace(
    "/.well-known/jwks.json", "/.well-known/agent.json"
)
case(
    "card-based discovery is not supported",
    r,
    refused(
        "signature_agent_invalid",
        "The Signature-Agent member this signature covers is not usable: card-based key discovery (type=cimd) is not supported by this verifier.",
    ),
)

r = signed("GET", "https://agent.example/agents/mini/info", nonce="n-member")
r["headers"]["signature-agent"] = r["headers"]["signature-agent"].replace("sig1=", "other=", 1)
case("the covered member is absent", r, refused("signature_agent_invalid", "The Signature-Agent dictionary has no member keyed sig1."))

doc = {
    "about": __doc__.strip().splitlines()[0] + " Generated by make_verify_cases.py; do not edit by hand.",
    "authority": "agent.example",
    "scheme": "https",
    "signerKey": jwk(SIGNER),
    "cases": cases,
}
(HERE / "verify-cases.json").write_text(json.dumps(doc, indent=2) + "\n")
print(f"{len(cases)} cases")
