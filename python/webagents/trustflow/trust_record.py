"""
The exportable TrustFlow record: verifying one, and carrying one in an A2A
card (webagents gap-closure plan item 2.7, 2026-09-26). The TypeScript twin is
`typescript/src/trustflow/trust-record.ts`; both verify the vectors in
`tests/fixtures/trust/trustflow_record.json`, which the portal's signer
produces byte for byte.

A RECORD is a compact JWS the platform signs with its keyring (RS256, the `kid`
published at `<issuer>/.well-known/jwks.json`): `iss`, `sub` (the agent URL),
`aud` (`urn:robutler:trustflow-record`), `iat`, `exp`, and the record itself
(`agent`, `score` in [0, 1], `tier`, `topics`, `computed_at`, `methodology`).
It travels with the agent, so anyone holding the platform key set can verify
it with no call to the platform: that is the point of it, and why the checks
here are strict.

WHAT VERIFYING CHECKS, in order: the three-part shape; `alg` RS256 and `typ`
`trustflow+jwt` (a platform token is not a record); the key by `kid` from the
held keys or the ISSUER's key set; the signature; `iss` equal to the issuer
the CALLER expects; `aud`; `exp` and `iat`; the claim shape; and, when asked,
that the record is about the subject the caller has in hand (URL, id or
username). The expected issuer comes from configuration (`issuer`, else
ROBUTLER_PLATFORM_ISSUER, else the platform URL the skills resolve), NEVER from
the record's own `iss`, and the key set URL is derived from that issuer under
the same floor `crypto.jwks` applies to a configured key-set URL: a record must
not be able to send its verifier to a host of its choosing (S-135).

THE CARD EXTENSION. An agent references its record in its A2A v1.0 card
(`transport/a2a/card.py`) as `capabilities.extensions[]` with the Robutler URI
below and `params.record` the JWS. The card's own signature covers the
extension, so a peer reading the card gets both proofs.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlsplit

TRUSTFLOW_RECORD_TYP = "trustflow+jwt"
TRUSTFLOW_RECORD_AUDIENCE = "urn:robutler:trustflow-record"
TRUSTFLOW_RECORD_EXTENSION_URI = "https://robutler.ai/a2a/extensions/trustflow-record/v1"
TRUSTFLOW_RECORD_EXTENSION_DESCRIPTION = (
    "A TrustFlow record signed by Robutler: params.record is a compact JWS verifiable against the issuer key set."
)
#: How long a fetched key set is held (the platform serves it with max-age 3600).
TRUST_KEY_SET_TTL_S = 3600
#: How far ahead of the clock an `iat` may be.
TRUST_RECORD_LEEWAY_S = 60

_KEY_SET_PATH = "/.well-known/jwks.json"


@dataclass
class VerifyTrustRecordResult:
    ok: bool
    #: The verified claims, when `ok`.
    record: Optional[Dict[str, Any]] = None
    kid: Optional[str] = None
    #: Why not, when not: malformed, alg, typ, key_set, no_key, signature,
    #: issuer, audience, expired, not_yet_valid, shape, subject.
    code: Optional[str] = None
    reason: Optional[str] = None


def _refuse(code: str, reason: str) -> VerifyTrustRecordResult:
    return VerifyTrustRecordResult(ok=False, code=code, reason=reason)


# ---------------------------------------------------------------------------
# base64url and JSON
# ---------------------------------------------------------------------------


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))


def _decode_json(segment: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(_b64url_decode(segment).decode("utf-8"))
    except Exception:  # noqa: BLE001 - not JSON is one refusal, whatever the cause
        return None
    return parsed if isinstance(parsed, dict) else None


def decode_trust_record(jws: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """The header and claims of a record, unverified: for display, never for a decision."""
    parts = jws.split(".") if isinstance(jws, str) else []
    if len(parts) != 3 or not all(parts):
        return None
    header, payload = _decode_json(parts[0]), _decode_json(parts[1])
    return {"header": header, "payload": payload} if header is not None and payload is not None else None


def canonical_record_url(value: str) -> str:
    """The one spelling of an agent URL: lower-case origin plus path, no trailing slash."""
    trimmed = value.strip().rstrip("/")
    parts = urlsplit(trimmed)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return trimmed
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    default = 443 if parts.scheme == "https" else 80
    netloc = host if port is None or port == default else f"{host}:{port}"
    return f"{parts.scheme}://{netloc}{parts.path}".rstrip("/")


def _trimmed_url(value: Any) -> Optional[str]:
    url = str(value or "").strip().rstrip("/")
    return url or None


# ---------------------------------------------------------------------------
# The key set
# ---------------------------------------------------------------------------

_key_sets: Dict[str, Any] = {}


def _reset_trust_key_sets() -> None:
    """Tests: forget every fetched key set."""
    _key_sets.clear()


def key_set_url_for_issuer(issuer: str) -> Optional[str]:
    """`<issuer>/.well-known/jwks.json` when the issuer is a URL a verifier may
    fetch from: http or https, a host, no userinfo, no query or fragment, and
    plain http to `localhost` only (the TypeScript `keySetUrlFromIssuer`)."""
    from webagents.crypto.jwks import is_public_key_set_url, is_well_formed_key_set_url

    base = _trimmed_url(issuer)
    if not base:
        return None
    url = f"{base}{_KEY_SET_PATH}"
    if not is_well_formed_key_set_url(url):
        return None
    parts = urlsplit(url)
    hostname = (parts.hostname or "").lower()
    if parts.scheme == "http":
        return url if hostname == "localhost" or hostname.endswith(".localhost") else None
    return url if is_public_key_set_url(url) else None


async def _default_fetch_json(url: str) -> Dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        return response.json()


async def _fetch_key_set(url: str, fetch_json: Callable[[str], Awaitable[Dict[str, Any]]], force: bool) -> List[Dict[str, Any]]:
    hit = _key_sets.get(url)
    if hit is not None and not force and hit["expires"] > time.time():
        return hit["keys"]
    body = await fetch_json(url)
    keys = [k for k in (body.get("keys") or []) if isinstance(k, dict)]
    _key_sets[url] = {"keys": keys, "expires": time.time() + TRUST_KEY_SET_TTL_S}
    return keys


def _find_key(keys: Optional[List[Dict[str, Any]]], kid: str) -> Optional[Dict[str, Any]]:
    for key in keys or []:
        if isinstance(key, dict) and key.get("kid") == kid:
            return key
    return None


def default_trust_issuer() -> str:
    """The issuer to expect when the caller names none (module docstring)."""
    from webagents.agents.skills.robutler.platform_url import resolve_platform_url

    return _trimmed_url(os.environ.get("ROBUTLER_PLATFORM_ISSUER")) or resolve_platform_url()


# ---------------------------------------------------------------------------
# Verifying
# ---------------------------------------------------------------------------


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def trust_record_shape(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The claims when their shape is a record's, else None."""
    agent = payload.get("agent")
    if (
        not isinstance(payload.get("iss"), str)
        or not isinstance(payload.get("sub"), str)
        or not isinstance(payload.get("aud"), str)
        or not _number(payload.get("iat"))
        or not _number(payload.get("exp"))
        or not isinstance(agent, dict)
        or not isinstance(agent.get("id"), str)
        or not agent.get("id")
        or not isinstance(agent.get("username"), str)
        or not (agent.get("url") is None or isinstance(agent.get("url"), str))
        or not _number(payload.get("score"))
        or not (0 <= payload["score"] <= 1)
        or not isinstance(payload.get("tier"), str)
        or not isinstance(payload.get("topics"), list)
        or not isinstance(payload.get("computed_at"), str)
        or not isinstance(payload.get("methodology"), str)
        or not payload.get("methodology")
    ):
        return None
    for topic in payload["topics"]:
        if not isinstance(topic, dict) or not isinstance(topic.get("id"), str) or not isinstance(topic.get("label"), str) or not _number(topic.get("score")):
            return None
    return payload


def record_matches_subject(record: Dict[str, Any], subject: Dict[str, Any]) -> bool:
    """Whether `record` is about `subject`: every field the caller named must match."""
    named = 0
    if "url" in subject and subject["url"] is not None:
        named += 1
        wanted = canonical_record_url(str(subject["url"]))
        recorded = canonical_record_url(record["agent"]["url"]) if record["agent"].get("url") else None
        if wanted != recorded and wanted != canonical_record_url(record["sub"]):
            return False
    if "id" in subject and subject["id"] is not None:
        named += 1
        if subject["id"] != record["agent"]["id"]:
            return False
    if "username" in subject and subject["username"] is not None:
        named += 1
        if str(subject["username"]).lstrip("@").lower() != str(record["agent"]["username"]).lower():
            return False
    return named > 0


def _verify_rs256(jwk: Dict[str, Any], data: bytes, signature: bytes) -> bool:
    if jwk.get("kty") != "RSA" or not isinstance(jwk.get("n"), str) or not isinstance(jwk.get("e"), str):
        return False
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    numbers = rsa.RSAPublicNumbers(
        int.from_bytes(_b64url_decode(jwk["e"]), "big"), int.from_bytes(_b64url_decode(jwk["n"]), "big")
    )
    try:
        numbers.public_key().verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, ValueError):
        return False


async def verify_trust_record(
    jws: str,
    *,
    issuer: Optional[str] = None,
    keys: Optional[List[Dict[str, Any]]] = None,
    jwks_url: Optional[str] = None,
    fetch_json: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
    subject: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> VerifyTrustRecordResult:
    """Verify `jws` as a TrustFlow record (module docstring: what is checked, in order). Never raises."""
    parts = jws.split(".") if isinstance(jws, str) else []
    if len(parts) != 3 or not all(parts):
        return _refuse("malformed", "not a compact JWS")
    header = _decode_json(parts[0])
    if header is None:
        return _refuse("malformed", "protected header is not base64url JSON")
    if header.get("alg") != "RS256":
        return _refuse("alg", f"alg {header.get('alg') or '(none)'} is not RS256")
    if header.get("typ") != TRUSTFLOW_RECORD_TYP:
        return _refuse("typ", f"typ {header.get('typ') or '(none)'} is not {TRUSTFLOW_RECORD_TYP}")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        return _refuse("no_key", "protected header has no kid")
    payload = _decode_json(parts[1])
    if payload is None:
        return _refuse("malformed", "payload is not base64url JSON")

    expected_issuer = _trimmed_url(issuer) or default_trust_issuer()

    jwk = _find_key(keys, kid)
    if jwk is None:
        url = jwks_url or key_set_url_for_issuer(expected_issuer)
        if not url:
            return _refuse("key_set", f"no key set can be fetched for issuer {expected_issuer}")
        fetch = fetch_json or _default_fetch_json
        try:
            jwk = _find_key(await _fetch_key_set(url, fetch, False), kid) or _find_key(await _fetch_key_set(url, fetch, True), kid)
        except Exception as error:  # noqa: BLE001 - the reason is reported, not raised
            return _refuse("key_set", str(error))
    if jwk is None:
        return _refuse("no_key", f"no key for kid {kid}")

    try:
        valid = _verify_rs256(jwk, f"{parts[0]}.{parts[1]}".encode("ascii"), _b64url_decode(parts[2]))
    except Exception:  # noqa: BLE001 - a key that cannot be read verifies nothing
        valid = False
    if not valid:
        return _refuse("signature", "signature does not verify")

    if not isinstance(payload.get("iss"), str) or _trimmed_url(payload["iss"]) != expected_issuer:
        return _refuse("issuer", f"issuer {payload.get('iss') or '(none)'} is not {expected_issuer}")
    if payload.get("aud") != TRUSTFLOW_RECORD_AUDIENCE:
        return _refuse("audience", f"aud {payload.get('aud') or '(none)'} is not a trust record")
    moment = now if now is not None else time.time()
    if not _number(payload.get("exp")) or payload["exp"] <= moment:
        return _refuse("expired", "the record has expired")
    if not _number(payload.get("iat")) or payload["iat"] > moment + TRUST_RECORD_LEEWAY_S:
        return _refuse("not_yet_valid", "the record is dated in the future")
    record = trust_record_shape(payload)
    if record is None:
        return _refuse("shape", "the claims are not a trust record")
    if subject is not None and not record_matches_subject(record, subject):
        return _refuse("subject", "the record is about another agent")
    return VerifyTrustRecordResult(ok=True, record=record, kid=kid)


# ---------------------------------------------------------------------------
# The card extension
# ---------------------------------------------------------------------------


def trust_record_extension(record: str) -> Dict[str, Any]:
    """The extension entry for `record`."""
    return {"uri": TRUSTFLOW_RECORD_EXTENSION_URI, "description": TRUSTFLOW_RECORD_EXTENSION_DESCRIPTION, "params": {"record": record}}


def with_trust_record_extension(card: Dict[str, Any], record: str) -> Dict[str, Any]:
    """`card` with its record set (replacing an earlier one); the card itself is not changed."""
    capabilities = dict(card.get("capabilities") or {})
    existing = capabilities.get("extensions")
    kept = [e for e in (existing if isinstance(existing, list) else []) if not (isinstance(e, dict) and e.get("uri") == TRUSTFLOW_RECORD_EXTENSION_URI)]
    capabilities["extensions"] = [*kept, trust_record_extension(record)]
    out = dict(card)
    out["capabilities"] = capabilities
    return out


def trust_record_from_card(card: Dict[str, Any]) -> Optional[str]:
    """The record a card carries, or None. Unverified: hand it to `verify_trust_record`."""
    capabilities = card.get("capabilities")
    extensions = capabilities.get("extensions") if isinstance(capabilities, dict) else None
    for entry in extensions if isinstance(extensions, list) else []:
        if isinstance(entry, dict) and entry.get("uri") == TRUSTFLOW_RECORD_EXTENSION_URI:
            params = entry.get("params")
            record = params.get("record") if isinstance(params, dict) else None
            return record if isinstance(record, str) and record else None
    return None
