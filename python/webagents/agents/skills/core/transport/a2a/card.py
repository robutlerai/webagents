"""
The A2A v1.0 agent card and its signature (A2A v1.0 sections 8 and 8.4), the
Python half of `a2a/card.ts`; both produce identical bytes for the same card,
pinned by `tests/fixtures/a2a/vectors.json`.

TWO CARDS, ON PURPOSE (plan item 1.3, 2026-09-26). The registration card at
`/.well-known/agent.json` (`server/core/registration.py`) is what the platform
reads once per principal and it refuses a card that does not name itself, so
it stays byte-compatible and is not touched. The v1.0 card served BESIDE it
at `/.well-known/agent-card.json` is for peers: OpenClaw and Hermes look there
first. The JSON-RPC interface is listed first, because Hermes takes the first
`JSONRPC` entry without checking its version.

THE SIGNATURE IS DETACHED JWS OVER THE JCS FORM OF THE PARSED CARD (section
8.4): drop `signatures`, drop every field left at its default, keep required
fields even when empty, keep `optional` (explicit-presence) fields even at
their default, canonicalise with RFC 8785, sign
`base64url(protected) + "." + base64url(payload)`, transmit the header and the
signature only.

WHERE a2a-python DIVERGES (src/a2a/utils/signing.py, released in 1.1.4 on
2026-09-08): its `_clean_empty` drops every `""`, `[]` and `{}`, REQUIRED ones
included, so for a card with an empty `description` or no `skills` the two
payloads differ, and only then; before 1.1.4 it signed
`json.dumps(sort_keys=True)` with ASCII escapes, which is not JCS for
non-ASCII text or numbers such as `1.0`. This implementation keeps required
fields, as the spec says, and the fixture's `signing_payload_cases` pin it.

Signing uses `cryptography`, which the SDK already depends on for its key
sets. EdDSA with the agent's Ed25519 key is the production path (`kid` = the
RFC 7638 thumbprint the key set publishes, `jku` = the agent's own
`jwks.json`); ES256 verifies for peers that sign that way (a2a-python's
sample does); HS256 exists only for the spec's byte-assembly vector.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Union
from urllib.parse import urlsplit

from .jcs import canonicalize

AGENT_CARD_WELL_KNOWN_SUFFIX = "/.well-known/agent-card.json"
LEGACY_CARD_WELL_KNOWN_SUFFIX = "/.well-known/agent.json"
KEY_SET_WELL_KNOWN_SUFFIX = "/.well-known/jwks.json"
#: Where the JSON-RPC and HTTP+JSON bindings are served, under the principal.
A2A_RPC_SUBPATH = "/a2a"

BEARER_SCHEME_DESCRIPTION = (
    "A Robutler platform token, an api key this agent accepts, or a peer token configured out of band."
)
HTTPSIG_SCHEME_DESCRIPTION = (
    "Web Bot Auth: RFC 9421 HTTP Message Signatures with Signature-Agent naming the caller's key set."
)


def build_a2a_agent_card(
    name: str,
    description: str,
    *,
    principal: str,
    skills: List[Dict[str, Any]],
    rpc_path: str = A2A_RPC_SUBPATH,
    version: str = "1.0.0",
    provider: Optional[Dict[str, str]] = None,
    documentation_url: Optional[str] = None,
    icon_url: Optional[str] = None,
    streaming: bool = True,
    input_modes: Optional[List[str]] = None,
    output_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The v1.0 card for an agent served at `principal`. Pure: no environment, no request."""
    base = principal.rstrip("/")
    rpc_url = f"{base}{rpc_path}"
    card: Dict[str, Any] = {
        "name": name,
        "description": description or "",
        "supportedInterfaces": [
            {"url": rpc_url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
            {"url": rpc_url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
        ],
    }
    if provider:
        card["provider"] = provider
    card["version"] = version
    if documentation_url:
        card["documentationUrl"] = documentation_url
    if icon_url:
        card["iconUrl"] = icon_url
    card["capabilities"] = {"streaming": streaming, "pushNotifications": False}
    card["securitySchemes"] = {
        "bearer": {
            "httpAuthSecurityScheme": {"scheme": "Bearer", "bearerFormat": "JWT", "description": BEARER_SCHEME_DESCRIPTION}
        },
        "httpsig": {"httpAuthSecurityScheme": {"scheme": "HTTPSig", "description": HTTPSIG_SCHEME_DESCRIPTION}},
    }
    card["securityRequirements"] = [{"schemes": {"bearer": {"list": []}}}, {"schemes": {"httpsig": {"list": []}}}]
    card["defaultInputModes"] = list(input_modes or ["text/plain", "application/json"])
    card["defaultOutputModes"] = list(output_modes or ["text/plain"])
    card["skills"] = skills
    return card


def skills_from_tools(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Card skills, one per tool, tagged `tool`, from either shape a tool
    listing has here: a `BaseAgent` registry entry (`get_tools_for_scopes`,
    whose `function` is the callable and whose `definition` is the OpenAI
    tool) or an OpenAI tool definition (`{"type": "function", "function":
    {...}}`), the shape the TypeScript `skillsFromTools` reads."""
    skills = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        definition = tool.get("definition")
        if isinstance(definition, dict) and isinstance(definition.get("function"), dict):
            fn: Any = definition["function"]
        elif isinstance(tool.get("function"), dict):
            fn = tool["function"]
        elif isinstance(tool.get("name"), str):
            fn = {"name": tool["name"], "description": tool.get("description")}
        else:
            continue
        if not fn.get("name"):
            continue
        skills.append({"id": fn["name"], "name": fn["name"], "description": fn.get("description") or "", "tags": ["tool"]})
    return skills


# ---------------------------------------------------------------------------
# The signing payload (section 8.4)
# ---------------------------------------------------------------------------

#: Required fields per message type, keyed by the path of the type: "" is the
#: card, "x[]" the items of the repeated field x. Always kept, even empty.
REQUIRED_FIELDS: Dict[str, tuple] = {
    "": ("name", "description", "supportedInterfaces", "version", "capabilities", "defaultInputModes", "defaultOutputModes", "skills"),
    "supportedInterfaces[]": ("url", "protocolBinding", "protocolVersion"),
    "skills[]": ("id", "name", "description", "tags"),
    "capabilities.extensions[]": ("uri",),
    "provider": ("url", "organization"),
}

#: Explicit-presence (`optional`) fields per type: kept when present, even at
#: their default, because a sender that wrote `"streaming": false` set it.
EXPLICIT_PRESENCE_FIELDS: Dict[str, tuple] = {
    "capabilities": ("streaming", "pushNotifications", "extendedAgentCard"),
    "supportedInterfaces[]": ("tenant",),
}


def _is_default(value: Any) -> bool:
    if value is None:
        return True
    if value is False or value == "" or (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0):
        return True
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _clean(value: Any, type_path: str) -> Any:
    if isinstance(value, list):
        return [_clean(item, f"{type_path}[]") for item in value]
    if isinstance(value, dict):
        required = REQUIRED_FIELDS.get(type_path, ())
        explicit = EXPLICIT_PRESENCE_FIELDS.get(type_path, ())
        out: Dict[str, Any] = {}
        for key, raw in value.items():
            if type_path == "" and key == "signatures":
                continue
            if raw is None:
                continue
            cleaned = _clean(raw, f"{type_path}.{key}" if type_path else key)
            if _is_default(cleaned) and key not in required and key not in explicit:
                continue
            out[key] = cleaned
        return out
    return value


def card_signing_payload(card: Dict[str, Any]) -> Dict[str, Any]:
    """The card as it is signed: no `signatures`, defaults removed per section 8.4."""
    return _clean(card, "")


def card_signing_bytes(card: Dict[str, Any]) -> bytes:
    """The canonical (JCS) bytes a card signature covers."""
    return canonicalize(card_signing_payload(card)).encode("utf-8")


# ---------------------------------------------------------------------------
# base64url
# ---------------------------------------------------------------------------


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


@dataclass
class CardSigner:
    """`alg`, the `kid` peers look up, an optional `jku`, and the raw signing function."""

    alg: str
    kid: str
    sign: Callable[[bytes], bytes]
    jku: Optional[str] = None


def protected_header(signer: CardSigner) -> Dict[str, str]:
    """The protected header, canonical so both SDKs emit identical bytes."""
    header = {"alg": signer.alg, "kid": signer.kid, "typ": "JOSE"}
    if signer.jku:
        header["jku"] = signer.jku
    return header


def signing_input(protected_b64: str, card: Dict[str, Any]) -> bytes:
    """The bytes a card signature is computed over: `protected "." base64url(JCS payload)`."""
    return f"{protected_b64}.{b64url(card_signing_bytes(card))}".encode("ascii")


def sign_agent_card(card: Dict[str, Any], signer: CardSigner) -> Dict[str, Any]:
    """`card` with one more detached signature. The card itself is not changed."""
    protected_b64 = b64url(canonicalize(protected_header(signer)).encode("utf-8"))
    signature = b64url(signer.sign(signing_input(protected_b64, card)))
    signed = dict(card)
    signed["signatures"] = list(card.get("signatures") or []) + [{"protected": protected_b64, "signature": signature}]
    return signed


def ed25519_card_signer(kid: str, private_key: Any, jku: Optional[str] = None) -> CardSigner:
    """An EdDSA signer for an Ed25519 private key (`cryptography`), `kid` its thumbprint."""
    return CardSigner(alg="EdDSA", kid=kid, jku=jku, sign=lambda data: private_key.sign(data))


def identity_card_signer(identity: Any) -> CardSigner:
    """The signer for the agent's own identity (`AgentSigningIdentity`): the
    current key, `kid` its thumbprint, `jku` the agent's key set."""
    keys = identity.held_keys()
    if not keys:
        raise ValueError("A2A card signer: the identity holds no key")
    current = keys[0]
    return ed25519_card_signer(current.thumbprint, current.private_key, jku=f"{identity.issuer}{KEY_SET_WELL_KNOWN_SUFFIX}")


def hmac_card_signer(kid: str, secret: bytes) -> CardSigner:
    """An HS256 signer, for the spec's byte-assembly vector and tests only."""
    return CardSigner(alg="HS256", kid=kid, sign=lambda data: hmac.new(secret, data, hashlib.sha256).digest())


# ---------------------------------------------------------------------------
# Verifying
# ---------------------------------------------------------------------------

KeyResolver = Callable[[Dict[str, Any]], Union[Optional[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]]


@dataclass
class VerifyCardResult:
    ok: bool
    checked: int
    kid: Optional[str] = None
    alg: Optional[str] = None
    reason: Optional[str] = None


def _blocked_host(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return hostname.lower() in ("localhost",)
    return not address.is_global


def jku_allowed(jku: str, card: Dict[str, Any], card_url: Optional[str] = None, allow_http: bool = False) -> bool:
    """A `jku` is fetched only from an origin the card itself is tied to."""
    parts = urlsplit(jku)
    if parts.scheme != "https" and not (allow_http and parts.scheme == "http"):
        return False
    if not parts.netloc:
        return False
    if not allow_http and _blocked_host(parts.hostname or ""):
        return False
    origins = set()
    for candidate in [card_url, *[i.get("url") for i in card.get("supportedInterfaces", []) if isinstance(i, dict)]]:
        if not candidate:
            continue
        c = urlsplit(candidate)
        if c.scheme and c.netloc:
            origins.add((c.scheme, c.netloc.lower()))
    return (parts.scheme, parts.netloc.lower()) in origins


async def verify_agent_card(
    card: Dict[str, Any],
    *,
    keys: Optional[List[Dict[str, Any]]] = None,
    resolve_key: Optional[KeyResolver] = None,
    secret: Optional[bytes] = None,
    allowed_algs: Optional[Iterable[str]] = None,
    card_url: Optional[str] = None,
    allow_http: bool = False,
    fetch_json: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
) -> VerifyCardResult:
    """Whether at least one of the card's signatures verifies (section 8.4).

    The key comes from `keys`, `resolve_key`, or the signature's `jku`, and a
    `jku` is fetched only from the origin the card was fetched from or one of
    its own interfaces, over https unless `allow_http`, never from a private
    address: a card must not be able to send a verifier to an arbitrary host.
    """
    signatures = card.get("signatures") or []
    if not signatures:
        return VerifyCardResult(ok=False, checked=0, reason="card carries no signatures")
    allowed = set(allowed_algs or (("EdDSA", "ES256", "HS256") if secret is not None else ("EdDSA", "ES256")))
    last_reason = "no signature verified"
    for entry in signatures:
        try:
            header = json.loads(b64url_decode(entry["protected"]).decode("utf-8"))
        except Exception:  # noqa: BLE001 - a malformed header is one reason among several
            last_reason = "protected header is not base64url JSON"
            continue
        alg = header.get("alg")
        kid = header.get("kid")
        if alg not in allowed:
            last_reason = f"alg {alg or '(none)'} is not accepted"
            continue
        if not kid:
            last_reason = "protected header has no kid"
            continue
        try:
            data = signing_input(entry["protected"], card)
            signature = b64url_decode(entry["signature"])
            outcome = await _verify_one(header, data, signature, card, keys, resolve_key, secret, card_url, allow_http, fetch_json)
        except Exception as error:  # noqa: BLE001 - the reason is reported, not raised
            last_reason = str(error)
            continue
        if outcome is True:
            return VerifyCardResult(ok=True, checked=len(signatures), kid=kid, alg=alg)
        last_reason = outcome
    return VerifyCardResult(ok=False, checked=len(signatures), reason=last_reason)


async def _verify_one(header, data, signature, card, keys, resolve_key, secret, card_url, allow_http, fetch_json) -> Union[bool, str]:
    alg = header["alg"]
    if alg == "HS256":
        if secret is None:
            return "HS256 needs a secret"
        expected = hmac.new(secret, data, hashlib.sha256).digest()
        return True if hmac.compare_digest(expected, signature) else "HS256 signature does not verify"
    jwk = await _resolve_jwk(header, card, keys, resolve_key, card_url, allow_http, fetch_json)
    if jwk is None:
        return f"no key for kid {header['kid']}"
    if alg == "EdDSA":
        if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
            return f"kid {header['kid']} is not an Ed25519 key"
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519

        public = ed25519.Ed25519PublicKey.from_public_bytes(b64url_decode(jwk["x"]))
        try:
            public.verify(signature, data)
            return True
        except InvalidSignature:
            return "EdDSA signature does not verify"
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        return f"kid {header['kid']} is not a P-256 key"
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    numbers = ec.EllipticCurvePublicNumbers(
        int.from_bytes(b64url_decode(jwk["x"]), "big"), int.from_bytes(b64url_decode(jwk["y"]), "big"), ec.SECP256R1()
    )
    public = numbers.public_key()
    if len(signature) != 64:
        return "ES256 signature is not r||s"
    der = encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
    try:
        public.verify(der, data, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return "ES256 signature does not verify"


async def _resolve_jwk(header, card, keys, resolve_key, card_url, allow_http, fetch_json) -> Optional[Dict[str, Any]]:
    kid = header["kid"]
    for key in keys or []:
        if key.get("kid") == kid:
            return key
    if resolve_key is not None:
        found = resolve_key(header)
        if hasattr(found, "__await__"):
            found = await found  # type: ignore[misc]
        return found  # type: ignore[return-value]
    jku = header.get("jku")
    if not jku:
        return None
    if not jku_allowed(jku, card, card_url=card_url, allow_http=allow_http):
        raise ValueError(f"jku {jku} is not an origin this card is served from")
    if fetch_json is None:
        fetch_json = _default_fetch_json
    body = await fetch_json(jku)
    for key in body.get("keys") or []:
        if key.get("kid") == kid:
            return key
    return None


async def _default_fetch_json(url: str) -> Dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        return response.json()
