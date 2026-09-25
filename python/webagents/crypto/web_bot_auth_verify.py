"""
Verifying a Web Bot Auth signature on a request an agent RECEIVES
(ADR-0045 section 3, 2026-09-25).

Both SDKs could sign what they send and neither could check what they were sent:
the only verifier was the portal's (`lib/auth/agent-auth.ts`,
`lib/auth/http-signatures/`, `lib/auth/web-bot-auth/`). This is that verifier
ported, step for step and refusal code for refusal code, minus what is the
platform's own business (agent registrations, owner keys, Redis). The
TypeScript twin is `typescript/src/crypto/web-bot-auth-verify.ts`; both run the
same cases (`tests/fixtures/web_bot_auth/verify-cases.json`) and answer with the
same codes and sentences.

THE STEPS, in the portal's order:
  1. `Signature-Input` and `Signature` parse, label for label.
  2. Only `tag="web-bot-auth"` labels count; none, or more than 2, refuses.
  3. Each label's parameters: integer `created` and `expires`, a window of at
     most 3600 s, 60 s of clock skew, `keyid` an RFC 7638 thumbprint, `alg`
     absent or `ed25519`, a 1 to 256 character `nonce`, 64 signature bytes.
  4. The request's `Host` is THIS agent's own authority (its public URL).
  5. Coverage: `@method @authority @path @query`, `content-digest` when there
     is a body, and exactly one covered `Signature-Agent` member.
  6. That member names where the keys are: a `jwks_uri` ending in
     `/.well-known/jwks.json` or a `directory` origin. Every label must name
     the same principal.
  7. The key set is fetched through the address guard (`webagents.net`): https
     only, public addresses only, no redirects, 64 KiB, 5 s, cached.
  8. The key whose thumbprint is `keyid` verifies Ed25519 over the base.
  9. `Content-Digest`, when covered, matches the body.
 10. The nonce is spent: a second use within the window is a replay.

PYTHON TYPE TRAPS the TypeScript side does not have: `bool` is an `int` and
`Token` is a `str`, so `created=?1` would pass an integer check and an
unquoted `tag=web-bot-auth` a string one. `_is_int` and `_is_string` refuse
both, as the TypeScript verifier does by type.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .http_signature import InnerList, Item, Token, request_target, serialize_inner_list, serialize_item
from .structured_fields import StructuredFieldParseError, param_of, parse_dictionary, parse_item

WEB_BOT_AUTH_TAG = "web-bot-auth"
CLOCK_TOLERANCE_S = 60
MAX_SIGNATURE_LIFETIME_S = 3600
NONCE_MAX_CHARS = 256
MAX_LABELS = 2
KEY_SET_MAX_KEYS = 16
KEY_SET_MAX_BYTES = 64 * 1024
KEY_SET_TIMEOUT_S = 5
KEY_SET_MIN_TTL_S = 300
KEY_SET_MAX_TTL_S = 86400
KEY_SET_FAILURE_TTL_S = 60
SIGNED_BODY_MAX_BYTES = 4 * 1024 * 1024
KEY_SET_WELL_KNOWN_SUFFIX = "/.well-known/jwks.json"
CARD_WELL_KNOWN_SUFFIX = "/.well-known/agent.json"
DIRECTORY_WELL_KNOWN_PATH = "/.well-known/http-message-signatures-directory"
DIRECTORY_MEDIA_TYPE = "application/http-message-signatures-directory+json"
_REQUIRED_COMPONENTS = ("@method", "@authority", "@path", "@query")
_REFUSED_DERIVED = frozenset({"@query-param", "@status", "@request-target"})
_REFUSED_PARAMS = frozenset({"sf", "bs", "tr", "name", "req"})
_THUMBPRINT_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_JWK_ALG_ACCEPTED = frozenset({"ed25519", "EdDSA", "Ed25519"})
#: RFC 9421 Appendix B.1 test keys, by thumbprint: a key set carrying one is a copy of an example.
TEST_KEY_THUMBPRINTS = frozenset(
    {
        "poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U",
        "oD0HwocPBSfpNy5W3bpJeyFGY_IQ_YpqxSjQ3Yd-CLA",
    }
)

MALFORMED = (
    "Signature-Input and Signature must be RFC 9651 dictionaries whose members match label for label, "
    "each Signature member a 64-byte Ed25519 signature."
)
_REQUIRED_SET_SENTENCE = '("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="<label>")'


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and not isinstance(value, Token)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Refusal:
    code: str
    description: str


@dataclass(frozen=True)
class VerifiedAgent:
    """A caller whose signature verified: `principal` is the `agent:` identity,
    `thumbprints` the `key:` identities, `identifier` where the keys were read."""

    principal: str
    thumbprints: Tuple[str, ...]
    identifier: str


@dataclass(frozen=True)
class VerifyOutcome:
    agent: Optional[VerifiedAgent] = None
    refusal: Optional[Refusal] = None

    @property
    def ok(self) -> bool:
        return self.agent is not None


def _refuse(code: str, description: str) -> VerifyOutcome:
    return VerifyOutcome(refusal=Refusal(code, description))


# -- replay -------------------------------------------------------------------------------------


class MemoryNonceStore:
    """Remembers spent nonces until their signature expires. Refuses when full
    rather than forgetting early."""

    def __init__(self, max_entries: int = 100_000):
        self._spent: Dict[str, int] = {}
        self._max = max_entries

    def spend(self, principal: str, nonce: str, until_s: int, now_s: int) -> bool:
        key = f"{principal}\n{nonce}"
        known = self._spent.get(key)
        if known is not None and known >= now_s:
            return False
        if len(self._spent) >= self._max:
            for k in [k for k, until in self._spent.items() if until < now_s]:
                del self._spent[k]
            if len(self._spent) >= self._max:
                return False
        self._spent[key] = until_s
        return True


# -- key sets -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredKey:
    thumbprint: str
    x: str


@dataclass(frozen=True)
class Discovery:
    type: str
    principal: str
    identifier: str
    fetch_url: str
    media_type: Optional[str] = None


@dataclass(frozen=True)
class KeySetOutcome:
    keys: Tuple[DiscoveredKey, ...] = ()
    ttl_s: int = 0
    code: Optional[str] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.code is None


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def jwk_thumbprint_any(entry: Mapping[str, Any]) -> Optional[str]:
    """RFC 7638 SHA-256 thumbprint of an OKP, RSA or EC public JWK; None otherwise."""
    kty = entry.get("kty")
    names = {"OKP": ("crv", "kty", "x"), "RSA": ("e", "kty", "n"), "EC": ("crv", "kty", "x", "y")}.get(kty)
    if names is None or any(not isinstance(entry.get(n), str) for n in names):
        return None
    members = {n: entry[n] for n in names}
    canonical = json.dumps(members, separators=(",", ":"), sort_keys=True)
    return _b64url(hashlib.sha256(canonical.encode("utf-8")).digest())


def parse_key_set(body: Any, *, well_known_directory: bool) -> Tuple[Optional[List[DiscoveredKey]], str]:
    """A JWK Set body to its usable Ed25519 keys (the portal's `parseKeySet` rules),
    or `(None, reason)`."""
    if not isinstance(body, dict) or not isinstance(body.get("keys"), list):
        return None, "the document has no keys array"
    entries = body["keys"]
    if len(entries) > KEY_SET_MAX_KEYS:
        return None, f"it carries more than {KEY_SET_MAX_KEYS} entries"
    keys: List[DiscoveredKey] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        thumbprint = jwk_thumbprint_any(entry)
        if thumbprint and thumbprint in TEST_KEY_THUMBPRINTS:
            return None, "it carries a known test key (RFC 9421 Appendix B.1)"
        if not thumbprint:
            continue
        x = entry.get("x")
        if entry.get("kty") != "OKP" or entry.get("crv") != "Ed25519" or not isinstance(x, str) or not _THUMBPRINT_RE.match(x):
            continue
        if "use" in entry and entry["use"] != "sig":
            continue
        if "key_ops" in entry and not (isinstance(entry["key_ops"], list) and "verify" in entry["key_ops"]):
            continue
        if "alg" in entry and not (isinstance(entry["alg"], str) and entry["alg"] in _JWK_ALG_ACCEPTED):
            continue
        if well_known_directory and "kid" in entry and entry["kid"] != thumbprint:
            return None, "a kid at the well-known directory must equal the key thumbprint"
        if thumbprint in seen:
            continue
        seen.add(thumbprint)
        keys.append(DiscoveredKey(thumbprint, x))
    if not keys:
        return None, "it carries no usable Ed25519 key"
    return keys, ""


def _cache_ttl(cache_control: Optional[str]) -> int:
    match = re.search(r"(?:^|,)\s*max-age\s*=\s*(\d+)", cache_control or "", re.I)
    asked = int(match.group(1)) if match else KEY_SET_MIN_TTL_S
    return min(KEY_SET_MAX_TTL_S, max(KEY_SET_MIN_TTL_S, asked))


class KeySetFetcher:
    """Where a verifier gets key sets: fetched through the address guard, cached."""

    def __init__(self, *, allow_private: bool = False, max_entries: int = 1000, now=None):
        self._cache: Dict[str, Tuple[int, KeySetOutcome]] = {}
        self._allow_private = allow_private
        self._max = max_entries
        self._now = now or (lambda: int(time.time()))
        self._in_flight = 0
        self._per_host: Dict[str, int] = {}

    async def get(self, discovery: Discovery) -> KeySetOutcome:
        now = self._now()
        hit = self._cache.get(discovery.fetch_url)
        if hit and hit[0] > now:
            return hit[1]
        outcome = await self._fetch(discovery)
        ttl = outcome.ttl_s if outcome.ok else KEY_SET_FAILURE_TTL_S
        if len(self._cache) >= self._max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[discovery.fetch_url] = (now + ttl, outcome)
        return outcome

    async def _fetch(self, discovery: Discovery) -> KeySetOutcome:
        from webagents.net.addresses import parse_allow_list
        from webagents.net.guarded_http import GuardError, exchange, resolve_allowed

        parts = urlsplit(discovery.fetch_url)
        host = (parts.hostname or "").lower()
        if self._in_flight >= 32 or self._per_host.get(host, 0) >= 2:
            return KeySetOutcome(code="key_set_unreachable", reason="too many key set fetches are in flight")
        self._in_flight += 1
        self._per_host[host] = self._per_host.get(host, 0) + 1
        try:
            allow = parse_allow_list(["0.0.0.0/0", "::/0"]) if self._allow_private else ()
            scheme = parts.scheme
            port = parts.port or (443 if scheme == "https" else 80)
            address = await resolve_allowed(host, port, allow)
            target = request_target("GET", discovery.fetch_url)
            host_header = parts.netloc.rpartition("@")[2]
            accept = f"{discovery.media_type}, application/json" if discovery.media_type else "application/json, application/jwk-set+json"
            answer = await exchange(
                method="GET",
                scheme=scheme,
                host=host,
                port=port,
                target=target.request_line or target.path,
                headers=[
                    ("Host", host_header),
                    ("Accept", accept),
                    ("Accept-Encoding", "identity"),
                    ("User-Agent", "WebAgents (+https://robutler.ai)"),
                ],
                body=b"",
                address=address,
                deadline=time.monotonic() + KEY_SET_TIMEOUT_S,
                max_bytes=KEY_SET_MAX_BYTES + 1,
            )
        except GuardError as e:
            reason = "no answer within 5 s" if e.code == "timeout" else e.message
            return KeySetOutcome(code="key_set_unreachable", reason=reason)
        except Exception:  # noqa: BLE001 - any other failure is an unreachable set, said once
            return KeySetOutcome(code="key_set_unreachable", reason="the fetch failed")
        finally:
            self._in_flight -= 1
            left = self._per_host.get(host, 1) - 1
            if left <= 0:
                self._per_host.pop(host, None)
            else:
                self._per_host[host] = left
        if 300 <= answer.status < 400:
            return KeySetOutcome(code="key_set_unreachable", reason="it answered with a redirect, which is not followed")
        if answer.status != 200:
            return KeySetOutcome(code="key_set_unreachable", reason=f"it answered {answer.status}")
        if answer.truncated or len(answer.body) > KEY_SET_MAX_BYTES:
            return KeySetOutcome(code="key_set_invalid", reason="it is larger than 64 KiB")
        media_type = (answer.header("content-type") or "").split(";", 1)[0].strip().lower()
        if discovery.media_type and media_type != discovery.media_type:
            return KeySetOutcome(code="key_set_invalid", reason=f"a directory must be served as {discovery.media_type}")
        try:
            body = json.loads(answer.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return KeySetOutcome(code="key_set_invalid", reason="it is not JSON")
        keys, reason = parse_key_set(body, well_known_directory=discovery.type == "directory")
        if keys is None:
            return KeySetOutcome(code="key_set_invalid", reason=reason)
        return KeySetOutcome(keys=tuple(keys), ttl_s=_cache_ttl(answer.header("cache-control")))


# -- authority, Signature-Agent, coverage ---------------------------------------------------------

_AUTHORITY_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)*\.?|\[[0-9a-f:.]+\])(?::[0-9]{1,5})?$"
)


def normalize_authority(host: Optional[str]) -> Optional[str]:
    """`host[:port]` lower-cased with the default ports stripped, or None."""
    if not isinstance(host, str):
        return None
    lower = host.strip().lower()
    if not lower or len(lower) > 255 or not _AUTHORITY_RE.match(lower):
        return None
    return re.sub(r":(?:443|80)$", "", lower)


@dataclass(frozen=True)
class _AgentMember:
    value: str
    type: Optional[str]
    invalid: Optional[str]


_DOT_SEGMENT_RE = re.compile(r"^(?:\.|%2e){1,2}$", re.I)


def _origin(parts) -> str:
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}".rstrip("/")


def _check_value(value: str, allow_http: bool):
    if not value or len(value) > 2048:
        return None, "the value is empty or too long"
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        parts.port  # noqa: B018 - raises for a bad port
    except ValueError:
        return None, "the value is not an absolute URL"
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None, "the value is not an absolute URL"
    if parts.scheme != "https" and not (allow_http and parts.scheme == "http"):
        return None, "the value must use https"
    if parts.username is not None or parts.password is not None:
        return None, "the value carries userinfo"
    if "#" in value:
        return None, "the value carries a fragment"
    if "?" in value:
        return None, "the value carries a query"
    if not hostname:
        return None, "the value has no host"
    path_start = value.find("/", value.find("//") + 2)
    raw_path = "" if path_start == -1 else value[path_start:]
    segments = raw_path.split("/")[1:]
    if any(_DOT_SEGMENT_RE.match(s) for s in segments):
        return None, "the value carries a dot segment"
    if any(len(s) == 0 for s in segments[:-1]):
        return None, "the value carries an empty path segment"
    try:
        normalized = request_target("GET", value)
    except Exception:  # noqa: BLE001 - not a URL the WHATWG rules accept
        return None, "the value is not an absolute URL"
    if (raw_path or "/") != normalized.path:
        return None, "the value path is not in canonical form"
    origin = normalized.url[: len(normalized.url) - len(normalized.request_line or normalized.path)]
    return (origin, normalized.path), ""


def resolve_discovery(member: _AgentMember, *, allow_http: bool, legacy: bool):
    """`(Discovery, None)` or `(None, reason)`: the `jwks_uri` and `directory` rows."""
    if member.invalid:
        return None, member.invalid
    checked, reason = _check_value(member.value, allow_http)
    if checked is None:
        return None, reason
    origin, path = checked
    is_origin = member.value in (origin, f"{origin}/")

    def directory():
        if not is_origin:
            return None, "a directory value must be an origin (scheme and host only, no path)"
        url = f"{origin}{DIRECTORY_WELL_KNOWN_PATH}"
        return Discovery("directory", origin, url, url, DIRECTORY_MEDIA_TYPE), None

    def jwks():
        if not path.endswith(KEY_SET_WELL_KNOWN_SUFFIX):
            return None, f"a key set URL must end in {KEY_SET_WELL_KNOWN_SUFFIX}"
        principal = f"{origin}{path[: -len(KEY_SET_WELL_KNOWN_SUFFIX)].rstrip('/')}"
        identifier = f"{origin}{path}"
        return Discovery("jwks_uri", principal, identifier, identifier), None

    cimd = "card-based key discovery (type=cimd) is not supported by this verifier"
    if legacy:
        return directory()
    if member.type == "jwks_uri":
        return jwks()
    if member.type == "directory":
        return directory()
    if member.type == "cimd":
        return None, cimd
    if member.type is None:
        if is_origin:
            return directory()
        if path.endswith(KEY_SET_WELL_KNOWN_SUFFIX):
            return jwks()
        if path.endswith(CARD_WELL_KNOWN_SUFFIX):
            return None, cimd
        return None, f"an untyped value must be an origin or a key set URL ending in {KEY_SET_WELL_KNOWN_SUFFIX}"
    return None, f"unsupported type {member.type}"


def _read_member(value: Any, params) -> _AgentMember:
    kind: Optional[str] = None
    invalid: Optional[str] = None
    if not _is_string(value):
        invalid = "the member value is not a string"
    for name, param in params or ():
        if name == "type":
            if not isinstance(param, Token):
                invalid = invalid or "the type parameter is not a token"
            else:
                kind = str(param)
            continue
        invalid = invalid or f"unknown member parameter {name}"
    return _AgentMember(value if _is_string(value) else "", kind, invalid)


def _parse_signature_agent(raw: str):
    """`(legacy, legacy_value, members)`, or None when the field does not parse."""
    trimmed = raw.strip()
    try:
        if trimmed.startswith('"'):
            item = parse_item(trimmed)
            if not _is_string(item.value) or item.params:
                return None
            return True, item.value, {}
        dictionary = parse_dictionary(trimmed)
    except StructuredFieldParseError:
        return None
    if not dictionary:
        return None
    members: Dict[str, _AgentMember] = {}
    for key, member in dictionary:
        members[key] = (
            _AgentMember("", None, "the member is an inner list")
            if isinstance(member, InnerList)
            else _read_member(member.value, member.params)
        )
    return False, None, members


# -- the signature base --------------------------------------------------------------------------


class _BaseError(Exception):
    pass


def _require_ascii(value: str) -> str:
    if not re.fullmatch(r"[\x20-\x7e]*", value):
        raise _BaseError("a component value is not printable ASCII")
    return value


def _component_value(ctx: Dict[str, Any], item: Item) -> str:
    if not _is_string(item.value):
        raise _BaseError("a component identifier is not a string")
    name = item.value
    key: Optional[str] = None
    for param, value in item.params or ():
        if param == "key":
            if not _is_string(value):
                raise _BaseError("a key parameter is not a string")
            if name.startswith("@"):
                raise _BaseError("a key parameter on a derived component")
            key = value
            continue
        raise _BaseError(f"the {param} parameter is refused" if param in _REFUSED_PARAMS else f"unknown parameter {param}")
    if name.startswith("@"):
        path = ctx["path"] or "/"
        if name == "@method":
            return _require_ascii(ctx["method"])
        if name == "@authority":
            return _require_ascii(ctx["authority"])
        if name == "@scheme":
            return ctx["scheme"]
        if name == "@path":
            return _require_ascii(path)
        if name == "@query":
            return _require_ascii(ctx["query"] or "?")
        if name == "@target-uri":
            return _require_ascii(f"{ctx['scheme']}://{ctx['authority']}{path}{ctx['query']}")
        raise _BaseError(f"{name} is refused" if name in _REFUSED_DERIVED else f"unknown derived component {name}")
    if name != name.lower():
        raise _BaseError("a field component name is not lower-case")
    raw = ctx["header"](name)
    if raw is None:
        raise _BaseError(f"the {name} field is absent")
    if key is None:
        return _require_ascii(raw.strip())
    try:
        dictionary = parse_dictionary(raw)
    except StructuredFieldParseError:
        raise _BaseError(f"the {name} field is not a dictionary") from None
    member = next((m for k, m in dictionary if k == key), None)
    if member is None:
        raise _BaseError(f"the {name} field has no member {key}")
    return _require_ascii(serialize_inner_list(member) if isinstance(member, InnerList) else serialize_item(member))


def _signature_base(ctx: Dict[str, Any], signature_input: InnerList) -> str:
    seen = set()
    lines = []
    for item in signature_input.items:
        identifier = serialize_item(item)
        if identifier in seen:
            raise _BaseError("a component is covered twice")
        seen.add(identifier)
        lines.append(f"{identifier}: {_component_value(ctx, item)}")
    lines.append(f'"@signature-params": {_require_ascii(serialize_inner_list(signature_input))}')
    return "\n".join(lines)


# -- verification --------------------------------------------------------------------------------


def _header_reader(headers: Union[Mapping[str, Any], Any]):
    if hasattr(headers, "getlist") or hasattr(headers, "get_list"):
        getter = getattr(headers, "getlist", None) or getattr(headers, "get_list")

        def read(name: str) -> Optional[str]:
            values = getter(name)
            return ", ".join(values) if values else None

        return read
    items = list(headers.items()) if hasattr(headers, "items") else []

    def read(name: str) -> Optional[str]:
        values = [v if isinstance(v, str) else ", ".join(v) for k, v in items if k.lower() == name and v is not None]
        return ", ".join(values) if values else None

    return read


def _verify_ed25519(x: str, base: str, signature: bytes) -> bool:
    if len(signature) != 64:
        return False
    try:
        raw = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
        Ed25519PublicKey.from_public_bytes(raw).verify(signature, base.encode("ascii"))
        return True
    except (InvalidSignature, ValueError):
        return False


def _digest_matches(header_value: str, body: bytes) -> bool:
    try:
        entries = parse_dictionary(header_value)
    except StructuredFieldParseError:
        return False
    if not entries:
        return False
    matched = False
    for alg, member in entries:
        if isinstance(member, InnerList) or not isinstance(member.value, (bytes, bytearray)):
            return False
        algorithm = {"sha-256": hashlib.sha256, "sha-512": hashlib.sha512}.get(alg)
        if algorithm is None:
            continue
        if not hmac.compare_digest(bytes(member.value), algorithm(body).digest()):
            return False
        matched = True
    return matched


@dataclass
class InboundRequest:
    method: str
    #: The request line's path and query as received.
    target: str
    headers: Any
    body: bytes = b""


async def verify_web_bot_auth(
    request: InboundRequest,
    *,
    authorities: Sequence[str],
    scheme: str,
    key_sets: Any,
    nonces: Any,
    allow_http: bool = False,
    now: Optional[int] = None,
) -> VerifyOutcome:
    """Verify the Web Bot Auth signature on an inbound request. Never raises for
    anything the caller sent."""
    header = _header_reader(request.headers)
    now_s = int(time.time()) if now is None else now

    # 1. The two fields, label for label.
    try:
        inputs = parse_dictionary(header("signature-input") or "")
        signatures = parse_dictionary(header("signature") or "")
    except StructuredFieldParseError:
        return _refuse("signature_malformed", MALFORMED)
    if not inputs:
        return _refuse("signature_malformed", MALFORMED)
    signature_map = dict(signatures)
    input_labels = {label for label, _ in inputs}
    if any(label not in input_labels for label, _ in signatures):
        return _refuse("signature_malformed", MALFORMED)
    parsed = []
    for label, member in inputs:
        sig = signature_map.get(label)
        if (
            not isinstance(member, InnerList)
            or sig is None
            or isinstance(sig, InnerList)
            or not isinstance(sig.value, (bytes, bytearray))
            or isinstance(sig.value, bool)
        ):
            return _refuse("signature_malformed", MALFORMED)
        if any(not _is_string(item.value) for item in member.items):
            return _refuse("signature_malformed", MALFORMED)
        parsed.append((label, member, bytes(sig.value)))

    # 2. Our tag, at most two.
    labels = [p for p in parsed if _is_string(param_of(p[1].params, "tag")) and param_of(p[1].params, "tag") == WEB_BOT_AUTH_TAG]
    if not labels:
        return _refuse(
            "signature_malformed",
            f'No signature carries tag="{WEB_BOT_AUTH_TAG}". Each Signature-Input member this agent verifies must set that tag.',
        )
    if len(labels) > MAX_LABELS:
        return _refuse(
            "signature_malformed",
            f'At most {MAX_LABELS} signatures tagged "{WEB_BOT_AUTH_TAG}" are verified per request; '
            "send one per key you hold while rotating and no more.",
        )

    # 3. Parameters.
    for _, signature_input, signature in labels:
        params = signature_input.params
        created, expires = param_of(params, "created"), param_of(params, "expires")
        keyid, alg, nonce = param_of(params, "keyid"), param_of(params, "alg"), param_of(params, "nonce")
        if not _is_int(created):
            return _refuse("signature_params_invalid", "The created parameter is required and must be an integer number of seconds since the epoch.")
        if not _is_int(expires):
            return _refuse("signature_params_invalid", "The expires parameter is required and must be an integer number of seconds since the epoch.")
        if not _is_string(keyid) or not _THUMBPRINT_RE.match(keyid):
            return _refuse(
                "signature_params_invalid",
                "The keyid parameter must be the RFC 7638 SHA-256 thumbprint of the signing key: base64url, no padding, 43 characters.",
            )
        if alg is not None and not (_is_string(alg) and alg == "ed25519"):
            return _refuse("signature_params_invalid", 'The alg parameter, when present, must be "ed25519"; this agent verifies Ed25519 signatures only.')
        if not _is_string(nonce) or not 0 < len(nonce) <= NONCE_MAX_CHARS:
            return _refuse(
                "signature_params_invalid",
                f"The nonce parameter is required: a string of 1 to {NONCE_MAX_CHARS} characters, random and used once "
                "(64 random bytes in base64 is the convention).",
            )
        if expires <= created:
            return _refuse("signature_params_invalid", "The expires parameter must be greater than created.")
        if expires - created > MAX_SIGNATURE_LIFETIME_S:
            return _refuse(
                "signature_params_invalid",
                f"The signature window (expires minus created) may be at most {MAX_SIGNATURE_LIFETIME_S} seconds; "
                "sign one request at a time with a short window.",
            )
        if len(signature) != 64:
            return _refuse("signature_malformed", MALFORMED)
        if created > now_s + CLOCK_TOLERANCE_S:
            return _refuse("signature_expired", f"The created parameter is more than {CLOCK_TOLERANCE_S} seconds in the future; check the signer's clock.")
        if expires < now_s - CLOCK_TOLERANCE_S:
            return _refuse("signature_expired", f"The signature expired more than {CLOCK_TOLERANCE_S} seconds ago; sign a fresh request.")

    # 4. Authority: this agent's own.
    accepted = [a for a in (normalize_authority(x) for x in authorities) if a]
    authority = normalize_authority(header("host"))
    if not authority or authority not in accepted:
        return _refuse(
            "signature_authority_mismatch",
            f"This agent verifies signatures made for its own address only; sign the request for "
            f"{accepted[0] if accepted else 'its public host'} and send it there.",
        )

    # 5. Body, coverage, and the Signature-Agent member each label covers.
    body = request.body or b""
    if len(body) > SIGNED_BODY_MAX_BYTES:
        return _refuse("signature_body_too_large", f"A signed request body may be at most {SIGNED_BODY_MAX_BYTES // (1024 * 1024)} MiB.")
    has_body = len(body) > 0
    agent_raw = header("signature-agent")
    if agent_raw is None:
        return _refuse("signature_agent_invalid", "The Signature-Agent header is absent.")
    agent_field = _parse_signature_agent(agent_raw)
    if agent_field is None:
        return _refuse("signature_malformed", "Signature-Agent must be an RFC 9651 dictionary of strings, or one string.")
    legacy, legacy_value, members = agent_field

    discovery: Optional[Discovery] = None
    covered = []
    for label, signature_input, signature in labels:
        names = [item.value for item in signature_input.items]
        for required in _REQUIRED_COMPONENTS:
            if required not in names:
                return _refuse(
                    "signature_coverage_insufficient",
                    f"The signature must cover {required}. Cover at least {_REQUIRED_SET_SENTENCE}, "
                    "omitting content-digest only when the request has no body.",
                )
        digest_covered = "content-digest" in names
        if has_body and not digest_covered:
            return _refuse(
                "signature_coverage_insufficient",
                'The request has a body, so it must send Content-Digest (sha-256 over the body bytes) and cover "content-digest" in the signature.',
            )
        agent_items = [item for item in signature_input.items if item.value == "signature-agent"]
        if len(agent_items) != 1:
            return _refuse(
                "signature_coverage_insufficient",
                'The signature must cover the Signature-Agent member for its label exactly once, as "signature-agent";key="<label>".',
            )
        key = param_of(agent_items[0].params, "key")
        if legacy:
            if key is not None:
                return _refuse(
                    "signature_coverage_insufficient",
                    'Signature-Agent is a bare string, so the covered component must be bare "signature-agent" with no key parameter.',
                )
            member = _AgentMember(legacy_value or "", None, None)
        else:
            if not _is_string(key) or not key:
                return _refuse(
                    "signature_coverage_insufficient",
                    'Signature-Agent is a dictionary, so the signature must cover one of its members as "signature-agent";key="<label>".',
                )
            member = members.get(key)
            if member is None:
                return _refuse("signature_agent_invalid", f"The Signature-Agent dictionary has no member keyed {key}.")
        resolved, reason = resolve_discovery(member, allow_http=allow_http, legacy=legacy)
        if resolved is None:
            return _refuse("signature_agent_invalid", f"The Signature-Agent member this signature covers is not usable: {reason}.")
        if discovery is not None and resolved.principal != discovery.principal:
            return _refuse("signature_agent_invalid", "Every signature on one request must name the same agent URL.")
        discovery = resolved
        covered.append(
            (
                signature_input,
                signature,
                param_of(signature_input.params, "keyid"),
                param_of(signature_input.params, "nonce"),
                param_of(signature_input.params, "expires"),
                digest_covered,
            )
        )
    if discovery is None:
        return _refuse("signature_malformed", MALFORMED)

    try:
        target = request_target(request.method, f"{scheme}://{authority}{request.target}")
    except Exception:  # noqa: BLE001 - a target the WHATWG rules refuse
        return _refuse("signature_malformed", MALFORMED)
    # WHATWG `url.search`: empty for no query and for a bare `?`, as the TypeScript verifier reads it.
    ctx = {
        "method": request.method,
        "authority": authority,
        "scheme": scheme,
        "path": target.path,
        "query": "" if target.query == "?" else target.query,
        "header": header,
    }

    # 7. The keys.
    key_set = await key_sets.get(discovery)
    if not key_set.ok:
        if key_set.code == "key_set_unreachable":
            return _refuse(key_set.code, f"The key set Signature-Agent names could not be fetched: {key_set.reason}.")
        return _refuse(key_set.code, f"The key set Signature-Agent names is not usable: {key_set.reason}.")

    # 8. Each label verifies.
    thumbprints: List[str] = []
    for signature_input, signature, keyid, _nonce, _expires, _digest in covered:
        found = next((k for k in key_set.keys if k.thumbprint == keyid), None)
        if found is None:
            return _refuse("signature_key_unknown", "No key in the published key set has the thumbprint this signature names as keyid.")
        try:
            base = _signature_base(ctx, signature_input)
        except _BaseError:
            return _refuse("signature_malformed", MALFORMED)
        if not _verify_ed25519(found.x, base, signature):
            return _refuse("signature_invalid", "The signature does not verify under the published key.")
        thumbprints.append(found.thumbprint)

    # 9. The body the signature vouches for.
    if any(c[5] for c in covered):
        digest_header = header("content-digest")
        if digest_header is None or not _digest_matches(digest_header, body):
            return _refuse("content_digest_mismatch", "Content-Digest does not match the request body.")

    # 10. Spend each nonce.
    for _, _, _, nonce, expires, _ in covered:
        if not nonces.spend(discovery.principal, nonce, expires + CLOCK_TOLERANCE_S, now_s):
            return _refuse("signature_replayed", "This signature's nonce was already used; sign each request afresh.")

    return VerifyOutcome(agent=VerifiedAgent(discovery.principal, tuple(thumbprints), discovery.identifier))

