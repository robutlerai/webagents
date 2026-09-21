"""
Web Bot Auth request signing: RFC 9421 HTTP Message Signatures under the
draft-ietf-webbotauth-httpsig-protocol-00 profile (ADR 0038 step 5, W2 design
sections 2.1 to 2.7 and 9.2, written 2026-09-17).

This is the credential a Python agent presents to the platform. Until this
date the SDK minted an RS256 bearer JWT and the platform verified it against
a PEM read off the agent card; that assertion is gone (no dual-stack window,
ADR operator decisions of 2026-09-17), and every authenticated request is now
signed with the agent's Ed25519 key over the request itself. What the
platform reads, in the order it reads it (design section 4.3):

  * `Signature-Input` and `Signature` (RFC 9421 sections 4.1 and 4.2): RFC 9651
    Dictionaries, one member per signature label. The covered components are
    fixed, in this order (design section 2.2):

        ("@method" "@authority" "@path" "@query" "content-digest"
         "signature-agent";key="<label>")

    `content-digest` is present exactly when the request has a body, and the
    `signature-agent` component is the Dictionary member keyed by the
    signature's own label, serialised WITH its parameters, so `type=jwks_uri`
    is inside the signature. The parameters are `created`, `expires`
    (`created + lifetime`, 60 seconds by default), `keyid` (the RFC 7638
    thumbprint of the signing key, which is also the `kid` the key set
    publishes), `alg="ed25519"`, a 64 byte random `nonce` in standard base64
    and `tag="web-bot-auth"`, in that order (design section 2.3).
  * `Signature-Agent` (P section 5.2.1): where the platform fetches the key
    set. The default form is the P-00 dictionary, one member per label,
    `sig1="<agent-url>/.well-known/jwks.json";type=jwks_uri`. The platform
    derives the principal (the agent URL, which IS `agent_registrations
    .agent_url`) by stripping the well-known suffix (design section 3.1).
    Two other forms exist as a setting so the next draft revision is a
    one-line default change (design section 2.4): `dictionary-untyped` is
    the editor's copy (the same member with no parameter) and
    `legacy-string` is the bare origin string older deployments send, whose
    covered component is then bare `"signature-agent"`.
  * `Content-Digest` (RFC 9530): `sha-256=:<base64>:` over the body bytes,
    present exactly when there is a body.

ROTATION (design section 2.5). An agent that holds more than one key signs
the request once per key, labels `sig1`, `sig2`, ..., each with its own
`Signature-Agent` member, nonce and `keyid`. That is what lets the platform's
continuity rule (admit a new key only when a key it already holds co-signed
the same request) be switched on with no SDK change; the single-key agent is
the common case and produces one label.

COVERED HEADERS (2026-09-18, machine-purchase design section 6.4, pass P9b).
A paid retry carries `Payment-Authorization` and `Robutler-Terms-Accepted`,
and the platform admits the payment and the assent only when BOTH are among
the signature's covered components, so that a proxy or a log between the
agent and the platform cannot attach a credential, or an acceptance of the
Terms, that the agent's own key never signed. `sign_request(covered_headers=
...)` and `WebBotAuth(covered_headers=...)` name plain header fields to
cover; they are placed after `content-digest` and before the
`signature-agent` member, so the fixed set keeps its shape and the member
stays last. The component name is the field name lowercased and bare (the
verifier refuses `;sf`, `;bs`, `;tr` and `;key` on anything but
`signature-agent`), and the value is the field value with leading and
trailing whitespace stripped, exactly what the verifier reads back off the
wire (RFC 9421 section 2.1). A header named but absent from the message is
refused before signing, because the verifier would answer 401
`signature_malformed` and the buyer would learn nothing from it. The
TypeScript signer places and values them identically, and
`tests/fixtures/web_bot_auth/vectors-covered-headers.json` pins those bytes
across both SDKs; with no `covered_headers` the output is byte-identical to
what this signer produced before, which the first vector file pins.

THE SERIALISER IS HAND-WRITTEN AND THERE IS NO PARSER. The signer emits a
small, fixed subset of RFC 9651 (Strings, Tokens, Integers, Byte Sequences,
Booleans as flag parameters, Inner Lists, Dictionaries) and never reads a
structured field back, so there is nothing here for a hostile input to
exercise and no dependency to pin. The signature base is built from the same
serialiser, which is why the bytes match the platform's verifier and the
TypeScript SDK byte for byte: `tests/fixtures/web_bot_auth/vectors.json`
pins that (design section 10.3).

`WebBotAuth` is the `httpx.Auth` that applies all of this to a request. It
sets `requires_request_body` so httpx reads the body before `auth_flow`
runs; without that a streaming body would be digested as empty and the
signature would not cover what is sent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


# --------------------------------------------------------------------------
# Profile constants (design sections 2.1 to 2.5)
# --------------------------------------------------------------------------

#: The three `Signature-Agent` forms the SDK can send (design section 2.4).
SIGNATURE_AGENT_FORMS = ("dictionary-typed", "dictionary-untyped", "legacy-string")
DEFAULT_SIGNATURE_AGENT_FORM = "dictionary-typed"
DEFAULT_LABEL = "sig1"
#: `expires - created`, in seconds. Matches Cloudflare's guidance and the
#: TypeScript SDK; the platform refuses anything over 3600.
SIGNATURE_LIFETIME_S = 60
#: The platform refuses a longer window (`AOAUTH_MAX_SIGNATURE_LIFETIME_S`), so the signer refuses to produce one.
SIGNATURE_MAX_LIFETIME_S = 3600
# The most keys a request is signed with: the current key and ONE previous
# key while rotating (design section 2.5). The platform verifies at most two
# `web-bot-auth` labels per request (`AOAUTH_MAX_LABELS`) and answers
# `signature_malformed` above that, so a third held key would break every
# signed request rather than ease a rotation (2026-09-18, W2 review).
# `JWKSManager.ensure_ed25519_key` loads at most one `.previous.pem`, so the
# SDK's own identities cannot exceed this; the cap is for a caller's own list.
MAX_HELD_KEYS = 2
SIGNATURE_TAG = "web-bot-auth"
SIGNATURE_ALG = "ed25519"
NONCE_BYTES = 64
KEY_SET_SUFFIX = "/.well-known/jwks.json"
CARD_SUFFIX = "/.well-known/agent.json"

#: The headers a signed request carries. `Content-Digest` only with a body.
SIGNATURE_HEADERS = ("Signature-Agent", "Signature-Input", "Signature", "Content-Digest")

#: Fields a signature may never list as a covered header: the body rule
#: covers the first, the other three are the signature itself (file
#: comment, "COVERED HEADERS").
COVERED_HEADERS_RESERVED = frozenset({"content-digest", "signature-agent", "signature-input", "signature"})

# RFC 9110 section 5.1 token, lowercased: the field-name alphabet.
_FIELD_NAME_RE = re.compile(r"^[a-z0-9!#$%&'*+\-.^_`|~]+$")


class SigningError(ValueError):
    """An input the signer refuses to sign. Never raised for a network reason."""


def normalize_covered_headers(names: Optional[Sequence[str]]) -> List[str]:
    """The covered header names as they are placed in the signature: stripped,
    lowercased, deduplicated in first-seen order, and refused when reserved,
    derived (`@`) or not a field name. Public so a caller composing a covered
    list (the MPP buyer adds its own names to the operator's) can see the
    list the signer will use. The TypeScript `normalizeCoveredHeaders`, rule
    for rule."""
    out: List[str] = []
    for raw in names or ():
        if not isinstance(raw, str):
            raise SigningError("covered_headers must be header field names")
        name = raw.strip().lower()
        if name.startswith("@"):
            raise SigningError(f"covered_headers names header fields; {raw!r} is a derived component")
        if not _FIELD_NAME_RE.match(name):
            raise SigningError(f"covered_headers: {raw!r} is not a header field name")
        if name in COVERED_HEADERS_RESERVED:
            suffix = " (it is covered whenever the message has a body)" if name == "content-digest" else ""
            raise SigningError(f"covered_headers: {name} is covered by the signer itself and cannot be listed{suffix}")
        if name not in out:
            out.append(name)
    return out


def _read_header(headers: Optional[Union[Mapping[str, str], httpx.Headers]], name: str) -> Optional[str]:
    """The value of `name` (already lowercased) in `headers`, or None. An
    `httpx.Headers` is read through its own case-insensitive `get`, which
    joins a repeated field with `, ` (RFC 9421 section 2.1 step 4); a plain
    mapping is matched case-insensitively here."""
    if headers is None:
        return None
    if isinstance(headers, httpx.Headers):
        return headers.get(name)
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name:
            return value
    return None


# --------------------------------------------------------------------------
# RFC 9651 serialiser subset (sections 4.1.1 to 4.1.11)
# --------------------------------------------------------------------------


class Token(str):
    """An RFC 9651 Token (section 3.3.4), e.g. `jwks_uri`. A plain `str` is a String."""


BareItem = Union[str, Token, int, bytes, bool]
#: Parameters are ORDERED (section 3.1.2); a `True` value serialises as the bare key.
Parameters = Sequence[Tuple[str, BareItem]]


@dataclass(frozen=True)
class Item:
    """A bare item with parameters (section 3.3)."""

    value: BareItem
    params: Parameters = ()


@dataclass(frozen=True)
class InnerList:
    """An Inner List with parameters (section 3.1.1)."""

    items: Sequence[Item]
    params: Parameters = ()


Member = Union[Item, InnerList]

_TOKEN_RE = re.compile(r"^[A-Za-z*][A-Za-z0-9:/!#$%&'*+\-.^_`|~]*$")
_KEY_RE = re.compile(r"^[a-z*][a-z0-9_\-.*]*$")
_PRINTABLE_ASCII_RE = re.compile(r"^[\x20-\x7e]*$")
_MAX_INTEGER = 10**15 - 1


def serialize_bare_item(value: BareItem) -> str:
    """Section 4.1.3.1. `bool` is checked before `int` and `Token` before `str`
    because each is a subclass of the other in Python."""
    if isinstance(value, bool):
        return "?1" if value else "?0"
    if isinstance(value, Token):
        if not _TOKEN_RE.match(value):
            raise SigningError(f"not a valid token: {value!r}")
        return str(value)
    if isinstance(value, str):
        if not _PRINTABLE_ASCII_RE.match(value):
            raise SigningError("a structured field string must be printable ASCII")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, int):
        if abs(value) > _MAX_INTEGER:
            raise SigningError("integer out of the structured field range")
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return ":" + base64.b64encode(bytes(value)).decode("ascii") + ":"
    raise SigningError(f"cannot serialise {type(value).__name__} as a bare item")


def _serialize_key(key: str) -> str:
    if not _KEY_RE.match(key):
        raise SigningError(f"not a valid dictionary or parameter key: {key!r}")
    return key


def serialize_parameters(params: Parameters) -> str:
    """Section 4.1.1.2: `;key` for a `True` value, `;key=value` otherwise."""
    out = []
    for key, value in params:
        out.append(";" + _serialize_key(key))
        if value is not True:
            out.append("=" + serialize_bare_item(value))
    return "".join(out)


def serialize_item(item: Item) -> str:
    """Section 4.1.3."""
    return serialize_bare_item(item.value) + serialize_parameters(item.params)


def serialize_inner_list(inner: InnerList) -> str:
    """Section 4.1.1.1."""
    return "(" + " ".join(serialize_item(item) for item in inner.items) + ")" + serialize_parameters(inner.params)


def serialize_member(member: Member) -> str:
    """A Dictionary member without its key: exactly the component value of
    `"field";key="k"` (RFC 9421 section 2.1.2)."""
    return serialize_inner_list(member) if isinstance(member, InnerList) else serialize_item(member)


def serialize_dictionary(members: Sequence[Tuple[str, Member]]) -> str:
    """Section 4.1.2. A member whose value is bare `True` is the key alone."""
    out = []
    for key, member in members:
        text = _serialize_key(key)
        if isinstance(member, Item) and member.value is True:
            text += serialize_parameters(member.params)
        else:
            text += "=" + serialize_member(member)
        out.append(text)
    return ", ".join(out)


# --------------------------------------------------------------------------
# Keys: the Ed25519 key, its thumbprint and its published JWK
# --------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def ed25519_public_jwk(public_key: ed25519.Ed25519PublicKey) -> Dict[str, str]:
    """The three thumbprint members of an Ed25519 public key (RFC 8037 section 2)."""
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"kty": "OKP", "crv": "Ed25519", "x": _b64url(raw)}


def jwk_thumbprint(jwk: Mapping[str, str]) -> str:
    """RFC 7638: SHA-256 over the required members (`crv`, `kty`, `x` for OKP)
    in lexicographic order with no whitespace, base64url without padding. 43
    characters, and the exact `keyid` the platform selects the key by."""
    required = {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"]}
    canonical = json.dumps(required, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _b64url(hashlib.sha256(canonical).digest())


@dataclass(frozen=True)
class SigningKey:
    """One key the signer holds: its thumbprint (the `keyid`) and the private key."""

    thumbprint: str
    private_key: ed25519.Ed25519PrivateKey

    @classmethod
    def from_private_key(cls, private_key: ed25519.Ed25519PrivateKey) -> "SigningKey":
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise SigningError("only Ed25519 keys sign requests (ADR 0038 operator decision 3)")
        return cls(jwk_thumbprint(ed25519_public_jwk(private_key.public_key())), private_key)

    def public_jwk(self) -> Dict[str, str]:
        """The entry the key set publishes: thumbprint members, `kid`, `use`, and
        no `alg` (the platform never reads one)."""
        jwk = ed25519_public_jwk(self.private_key.public_key())
        jwk["kid"] = self.thumbprint
        jwk["use"] = "sig"
        return jwk


# --------------------------------------------------------------------------
# Content-Digest (RFC 9530 section 2)
# --------------------------------------------------------------------------


def content_digest(body: bytes) -> str:
    """`sha-256=:<standard base64>:` over the body bytes. The platform also
    accepts `sha-512`; the SDK sends `sha-256` only (design section 2.2)."""
    return serialize_dictionary([("sha-256", Item(hashlib.sha256(body).digest()))])


# --------------------------------------------------------------------------
# The request target (RFC 9421 sections 2.2.1, 2.2.3, 2.2.6, 2.2.7)
#
# `@path` AND `@query` ARE THE WHATWG SPELLING, NOT THE CALLER'S (2026-09-19).
# The platform rebuilds both from `new URL(request.url)` (`pathname`, `search`,
# lib/auth/agent-auth.ts step 10), and the TypeScript signer reads them off
# the same parser, so that parser's normalisation is the contract: dot
# segments are resolved (`.`, `..` and their `%2e` spellings), a backslash is
# a slash, tabs and newlines vanish, and whatever is outside the path or query
# percent-encode set is percent-encoded as UTF-8, while an existing `%xx` is
# left exactly as written (never decoded, never re-cased).
#
# Until that day this function returned what `urlsplit` or `httpx`'s
# `raw_path` held. For `/a/../b` given as a string that signed `/a/../b`
# while httpx SENT `/b` (it resolves plain dot segments on its own) and the
# platform verified `/b`: `signature_invalid`, with a base that looked right.
# For `/a/%2e%2e/b` it signed and sent that spelling and the platform
# verified `/b`. The signer now signs the normalised target, and the two
# senders in this SDK (`WebBotAuth` and the MPP buyer) move the request onto
# the same spelling through `apply_signed_target`, so the URL SENT is the
# URL SIGNED. `tests/fixtures/web_bot_auth/vectors-request-target.json` pins
# the normalisation across both SDKs.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestTarget:
    """The derived component values of a request. `query` INCLUDES its leading
    `?`, or is `?` alone when the request has no query (section 2.2.7).
    `url` is the whole normalised URL (scheme, userinfo when there was one,
    authority, path, and the query when the input had a `?`) and
    `request_line` its path-and-query part, exactly what a sender must put on
    the request line (`httpx.URL.raw_path`) for the signature to verify."""

    method: str
    authority: str
    path: str
    query: str
    url: str = ""
    request_line: str = ""


_DEFAULT_PORTS = {"https": 443, "http": 80}

# WHATWG URL, "percent-encode sets". Every byte above 0x7E and every C0
# control is in all of them; these are the printable ASCII additions.
_QUERY_ENCODE = frozenset(b' "#<>')
_SPECIAL_QUERY_ENCODE = _QUERY_ENCODE | frozenset(b"'")
_PATH_ENCODE = _QUERY_ENCODE | frozenset(b"?^`{}")
_C0_OR_SPACE = "".join(chr(c) for c in range(0x21))
_TAB_OR_NEWLINE_RE = re.compile(r"[\t\n\r]")
_SINGLE_DOT = frozenset({".", "%2e"})
_DOUBLE_DOT = frozenset({"..", ".%2e", "%2e.", "%2e%2e"})
_ABSOLUTE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _percent_encode(text: str, encode_set: frozenset) -> str:
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise SigningError(f"the request URL is not encodable as UTF-8: {e}") from None
    return "".join(f"%{b:02X}" if (b < 0x20 or b > 0x7E or b in encode_set) else chr(b) for b in raw)


def _whatwg_path(raw_path: str) -> str:
    """The WHATWG path state for a special scheme, over a path whose
    backslashes are already slashes: a double-dot segment removes the segment
    before it, a single-dot segment is dropped, either one in LAST position
    leaves a trailing slash, empty segments are kept, and every other segment
    is percent-encoded with the path set."""
    segments = raw_path.split("/")[1:] if raw_path.startswith("/") else raw_path.split("/")
    out: List[str] = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        lowered = segment.lower()
        if lowered in _DOUBLE_DOT:
            if out:
                out.pop()
            if last:
                out.append("")
        elif lowered in _SINGLE_DOT:
            if last:
                out.append("")
        else:
            out.append(_percent_encode(segment, _PATH_ENCODE))
    return "/" + "/".join(out)


def _authority(scheme: str, host: str, port: Optional[int]) -> str:
    host = host.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # an IPv6 literal, as it appears in a URL
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        return f"{host}:{port}"
    return host


def request_target(method: str, url: Union[str, httpx.URL]) -> RequestTarget:
    """Split a request URL into the four derived components, `@path` and
    `@query` normalised as the WHATWG parser normalises them (section
    comment above). A string and the `httpx.URL` built from it read the same."""
    text = str(url)
    # WHATWG basic URL parser, the two preprocessing steps: strip leading and
    # trailing C0 control or space, then remove every ASCII tab or newline.
    text = _TAB_OR_NEWLINE_RE.sub("", text.strip(_C0_OR_SPACE))
    # The fragment is never sent; the query starts at the first `?` before it.
    before_fragment = text.split("#", 1)[0]
    head, has_query, raw_query = before_fragment.partition("?")
    # A backslash is a slash everywhere before the query, for http(s).
    head = head.replace("\\", "/")
    if not _ABSOLUTE_RE.match(head):
        raise SigningError("the request URL must be absolute http(s) with a host")
    parts = urlsplit(head)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise SigningError("the request URL must be absolute http(s) with a host")
    try:
        port = parts.port
    except ValueError as e:
        raise SigningError(f"the request URL has an invalid port: {e}") from e
    authority = _authority(scheme, parts.hostname, port)
    path = _whatwg_path(parts.path)
    query = _percent_encode(raw_query, _SPECIAL_QUERY_ENCODE)
    userinfo, at, _ = parts.netloc.rpartition("@")
    request_line = f"{path}{'?' + query if has_query else ''}"
    return RequestTarget(
        method=method.upper(),
        authority=authority,
        path=path,
        query=("?" + query) if query else "?",
        url=f"{scheme}://{userinfo + at}{authority}{request_line}",
        request_line=request_line,
    )


def apply_signed_target(request: httpx.Request, target: RequestTarget) -> None:
    """Move `request` onto the URL that was signed, so the URL SENT is the URL
    SIGNED (section comment above). A no-op when the request line httpx would
    write already equals the signed path and query, which is every ordinary
    URL. Raises `SigningError`, before anything is sent, if httpx would not
    put the signed spelling on the wire: a signature the platform is certain
    to refuse is not worth a request."""
    wanted = (target.request_line or target.path).encode("ascii")
    if request.url.raw_path == wanted:
        return
    moved = request.url.copy_with(raw_path=wanted)
    if moved.raw_path != wanted:
        raise SigningError(
            f"cannot send the signed request target {wanted.decode('ascii')!r}: the HTTP client would write "
            f"{moved.raw_path.decode('ascii', 'replace')!r}, and the platform verifies what it receives"
        )
    request.url = moved


# --------------------------------------------------------------------------
# Signature-Agent (design sections 2.4 and 3.1)
# --------------------------------------------------------------------------


def _check_form(form: str) -> None:
    if form not in SIGNATURE_AGENT_FORMS:
        raise SigningError(f"unknown Signature-Agent form {form!r}; one of {', '.join(SIGNATURE_AGENT_FORMS)}")


def canonical_agent_url(agent_url: str) -> str:
    """The one spelling of an agent URL: scheme and host lowercased, a default
    port dropped, an IPv6 host bracketed, the path kept, trailing slashes
    stripped. Anything that is not an absolute http(s) URL comes back with
    only the trailing slashes stripped, so a relative principal (`/agents/x`,
    the card's last resort) passes through.

    WHY (2026-09-18, W2 review): the platform derives the principal from
    `Signature-Agent` through a WHATWG URL parse, `origin + path`, which
    lowercases the host and drops `:443`, and then compares the card's `url`,
    `client_id` and `jwks_uri` to that by string equality. A `public_url`
    spelled `https://Agents.Example.com:443` therefore signed a request that
    verified and was refused `card_not_self_naming`, in both SDKs. Every
    consumer of the agent URL, the signer and the card alike, now reads it
    through this one function, so the two cannot disagree.
    """
    value = (agent_url or "").rstrip("/")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return value
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return value
    return f"{scheme}://{_authority(scheme, parts.hostname, port)}{parts.path.rstrip('/')}"


def normalize_agent_url(agent_url: str) -> str:
    """The principal as the platform will derive it: absolute http(s), no
    userinfo, query or fragment, no trailing slash (design section 3.1), in
    the canonical spelling of `canonical_agent_url`."""
    if not isinstance(agent_url, str) or not agent_url:
        raise SigningError("the agent URL is required: it names the key set the platform fetches")
    parts = urlsplit(agent_url)
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        raise SigningError("the agent URL must be absolute http(s) with a host")
    if parts.username is not None or parts.password is not None:
        raise SigningError("the agent URL must not carry userinfo")
    if "?" in agent_url or "#" in agent_url:
        raise SigningError("the agent URL must not carry a query or a fragment")
    if not _PRINTABLE_ASCII_RE.match(agent_url):
        raise SigningError("the agent URL must be printable ASCII")
    try:
        parts.port
    except ValueError:
        raise SigningError("the agent URL carries a port that is not a number in range") from None
    return canonical_agent_url(agent_url)


def _is_loopback_host(hostname: str) -> bool:
    return (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname in ("127.0.0.1", "::1")
    )


def assert_signable_agent_url(agent_url: str, *, allow_http: Optional[bool] = None) -> str:
    """The agent URL an identity may sign as, canonical, or a raised sentence
    saying why not. The TypeScript SDK's `assertSignableAgentUrl` word for
    word (2026-09-18, W2 review: the Python registration refused plain http
    unconditionally while TypeScript, like the platform, allows it under
    `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`, so a Python agent could not register
    on the local cluster at all). Loopback by name is refused always, because
    the PLATFORM refuses it by name before resolving anything; plain http is
    refused unless `allow_http` says otherwise, defaulting to that variable in
    this process's environment.
    """
    url = normalize_agent_url(agent_url)
    parts = urlsplit(url)
    if _is_loopback_host(parts.hostname or ""):
        raise SigningError(
            f"agent URL {agent_url} is a loopback address. The PLATFORM fetches the "
            "key set and the agent card from it, so set public_url / WEBAGENTS_PUBLIC_URL "
            "to an address reachable from the internet"
        )
    if allow_http is None:
        allow_http = os.getenv("ROBUTLER_AGENT_URL_ALLOW_PRIVATE") == "1"
    if parts.scheme != "https" and not (allow_http and parts.scheme == "http"):
        raise SigningError(
            f"agent URL {agent_url} is not https. The platform accepts a plaintext key set "
            "only where ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1 (its local overlay); set "
            "public_url / WEBAGENTS_PUBLIC_URL to an https address, or set "
            "ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1 for this process to sign for a plaintext one"
        )
    return url


def key_set_url(agent_url: str) -> str:
    """`{agent_url}/.well-known/jwks.json`: the `jwks_uri` member value and the card's `jwks_uri`."""
    return normalize_agent_url(agent_url) + KEY_SET_SUFFIX


def agent_card_url(agent_url: str) -> str:
    """`{agent_url}/.well-known/agent.json`: the card's `client_id` (design section 3.3)."""
    return normalize_agent_url(agent_url) + CARD_SUFFIX


def signature_agent_value(agent_url: str, form: str) -> str:
    """The String value a member carries for a form: the key-set URL for the
    two dictionary forms, the bare origin for the legacy string (which names
    an origin and nothing more, design section 2.4)."""
    _check_form(form)
    agent_url = normalize_agent_url(agent_url)
    if form == "legacy-string":
        parts = urlsplit(agent_url)
        return f"{parts.scheme}://{_authority(parts.scheme, parts.hostname or '', parts.port)}"
    return agent_url + KEY_SET_SUFFIX


def signature_agent_member(agent_url: str, form: str) -> Item:
    """One dictionary member: `"<url>";type=jwks_uri` or, untyped, `"<url>"`."""
    _check_form(form)
    value = signature_agent_value(agent_url, form)
    if form == "dictionary-typed":
        return Item(value, (("type", Token("jwks_uri")),))
    return Item(value)


def signature_agent_header(agent_url: str, form: str, labels: Sequence[str]) -> str:
    """The `Signature-Agent` field value: one member per label for the
    dictionary forms, the single String for the legacy form."""
    _check_form(form)
    if form == "legacy-string":
        return serialize_item(Item(signature_agent_value(agent_url, form)))
    member = signature_agent_member(agent_url, form)
    return serialize_dictionary([(label, member) for label in labels])


def signature_agent_component(form: str, label: str) -> Item:
    """The covered component identifier: `"signature-agent";key="<label>"`, or
    bare `"signature-agent"` for the legacy string (design section 2.6)."""
    _check_form(form)
    if form == "legacy-string":
        return Item("signature-agent")
    return Item("signature-agent", (("key", label),))


def signature_agent_component_value(agent_url: str, form: str) -> str:
    """The component value: the member serialised as an Item with its
    parameters (RFC 9421 section 2.1.2), or the legacy field value as sent."""
    _check_form(form)
    if form == "legacy-string":
        return serialize_item(Item(signature_agent_value(agent_url, form)))
    return serialize_member(signature_agent_member(agent_url, form))


# --------------------------------------------------------------------------
# The signature base (RFC 9421 section 2.5)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureParams:
    """The parameters of one signature, in the order they are serialised (design section 2.3)."""

    created: int
    expires: int
    keyid: str
    nonce: str
    alg: str = SIGNATURE_ALG
    tag: str = SIGNATURE_TAG

    def as_parameters(self) -> Parameters:
        return (
            ("created", self.created),
            ("expires", self.expires),
            ("keyid", self.keyid),
            ("alg", self.alg),
            ("nonce", self.nonce),
            ("tag", self.tag),
        )


def covered_components(
    form: str,
    label: str,
    has_body: bool,
    covered_headers: Sequence[str] = (),
) -> List[Item]:
    """Design section 2.2, in order. `content-digest` is covered iff there is
    a body; the covered headers (design 6.4, already normalised) sit between
    the body digest and the `signature-agent` member, which stays last."""
    components = [Item("@method"), Item("@authority"), Item("@path"), Item("@query")]
    if has_body:
        components.append(Item("content-digest"))
    components.extend(Item(name) for name in covered_headers)
    components.append(signature_agent_component(form, label))
    return components


def signature_params_inner_list(components: Sequence[Item], params: SignatureParams) -> InnerList:
    """The `Signature-Input` member value, which is also the `@signature-params` line."""
    return InnerList(tuple(components), params.as_parameters())


def build_signature_base(component_values: Sequence[Tuple[Item, str]], params: InnerList) -> str:
    """One line per covered component, `"identifier": value`, then the
    `@signature-params` line; `\\n` separated, no trailing newline (section
    2.5). Every value must be printable ASCII (section 2.5 step 4), and a
    CR or LF inside one would let a component forge a line, so it is refused."""
    lines = []
    seen = set()
    for component, value in component_values:
        identifier = serialize_item(component)
        if identifier in seen:
            raise SigningError(f"component covered twice: {identifier}")
        seen.add(identifier)
        if not _PRINTABLE_ASCII_RE.match(value):
            raise SigningError(f"component value of {identifier} is not printable ASCII")
        lines.append(f"{identifier}: {value}")
    lines.append(f'"@signature-params": {serialize_inner_list(params)}')
    return "\n".join(lines)


def component_values(
    target: RequestTarget,
    agent_url: str,
    form: str,
    label: str,
    digest: Optional[str],
    covered_values: Sequence[Tuple[str, str]] = (),
) -> List[Tuple[Item, str]]:
    """Pair every covered component with its value for this request.
    `covered_values` are `(name, value)` pairs for the covered headers, the
    name normalised and the value already stripped, in signing order."""
    values: List[Tuple[Item, str]] = [
        (Item("@method"), target.method),
        (Item("@authority"), target.authority),
        (Item("@path"), target.path),
        (Item("@query"), target.query),
    ]
    if digest is not None:
        values.append((Item("content-digest"), digest))
    values.extend((Item(name), value) for name, value in covered_values)
    values.append((signature_agent_component(form, label), signature_agent_component_value(agent_url, form)))
    return values


def resolve_covered_headers(
    headers: Optional[Union[Mapping[str, str], httpx.Headers]],
    covered_headers: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    """`(name, value)` for every covered header, the name normalised and the
    value stripped, refusing a name the message does not carry (file
    comment, "COVERED HEADERS"). Resolved once, before any key signs, so a
    missing field refuses the whole message rather than one label of it."""
    out: List[Tuple[str, str]] = []
    for name in normalize_covered_headers(covered_headers):
        raw = _read_header(headers, name)
        if raw is None:
            raise SigningError(
                f"cannot cover header {name}: the message does not carry it. A covered header must be "
                "set on the request before it is signed, since the verifier reads the value off the wire"
            )
        out.append((name, raw.strip()))
    return out


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------


def new_nonce() -> str:
    """64 random bytes as standard base64 (design section 2.3, P Appendix E)."""
    return base64.b64encode(os.urandom(NONCE_BYTES)).decode("ascii")


def signature_labels(label: str, count: int) -> List[str]:
    """`sig1`, `sig2`, ... for `count` held keys (design section 2.5): the given
    label names the first signature and the others take the same stem with
    their ordinal (`sig1` gives `sig2`, `agent` gives `agent2`)."""
    _serialize_key(label)
    stem = re.sub(r"\d+$", "", label) or label
    labels = [label] + [f"{stem}{i}" for i in range(2, count + 1)]
    if len(set(labels)) != len(labels):
        raise SigningError(f"label {label!r} cannot be extended to {count} distinct labels")
    return labels


@dataclass(frozen=True)
class SignatureRecord:
    """What one label signed, kept for tests and for logging a refusal."""

    label: str
    keyid: str
    created: int
    expires: int
    nonce: str
    base: str
    signature: bytes


@dataclass(frozen=True)
class SignedRequest:
    """The headers to add to the request, plus a record per signature, plus
    the `target` that was signed: `target.url` is the URL a sender must put
    on the wire, which differs from the URL passed in exactly when that one
    was not in the WHATWG spelling (`apply_signed_target`)."""

    headers: Dict[str, str]
    signatures: List[SignatureRecord]
    target: Optional[RequestTarget] = None


def check_lifetime(lifetime: object) -> int:
    """The signing window in whole seconds, 1 to 3600, or a `SigningError`.
    An INTEGER, as the TypeScript signer requires (2026-09-19): `0.5` passed
    the old `0 < lifetime <= 3600` test and `int()` then made
    `expires == created`, which the platform refuses as
    `signature_params_invalid`; `True` is an `int` in Python and is refused
    too."""
    if isinstance(lifetime, bool) or not isinstance(lifetime, int) or lifetime <= 0 or lifetime > SIGNATURE_MAX_LIFETIME_S:
        raise SigningError(
            f"the signature lifetime must be an integer between 1 and {SIGNATURE_MAX_LIFETIME_S} seconds"
        )
    return lifetime


def sign_request(
    keys: Sequence[SigningKey],
    agent_url: str,
    method: str,
    url: Union[str, httpx.URL],
    body: bytes = b"",
    *,
    form: str = DEFAULT_SIGNATURE_AGENT_FORM,
    label: str = DEFAULT_LABEL,
    lifetime: int = SIGNATURE_LIFETIME_S,
    created: Optional[int] = None,
    nonces: Optional[Sequence[str]] = None,
    allow_http: Optional[bool] = None,
    headers: Optional[Union[Mapping[str, str], httpx.Headers]] = None,
    covered_headers: Optional[Sequence[str]] = None,
) -> SignedRequest:
    """Sign one request with every key in `keys` (design sections 2.1 to 2.5).

    `keys` is the list of keys the agent holds, current key first; each one
    produces its own label, nonce and `Signature-Agent` member. `agent_url`
    is the principal (the URL the agent is mounted at, no trailing slash),
    from which the key-set URL on the wire is derived; it must be one the
    platform could fetch from (`assert_signable_agent_url`: not loopback, and
    https unless `allow_http`, which defaults to
    `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1` in the environment, the same switch
    the TypeScript signer and the platform's local overlay read). `body` is
    the exact bytes that will be sent; empty means no `Content-Digest` and no
    `content-digest` component. `covered_headers` names plain header fields
    to cover beside the fixed set, read off `headers` (the message's own
    fields, an `httpx.Headers` or a mapping matched case-insensitively); a
    named field the message does not carry is refused (file comment,
    "COVERED HEADERS"). `created` and `nonces` exist for deterministic tests
    and the cross-language vectors; production callers leave them unset and
    get the clock and 64 fresh random bytes per label.
    """
    _check_form(form)
    if not keys:
        raise SigningError("no signing key: call JWKSManager.ensure_ed25519_key first")
    if len(keys) > MAX_HELD_KEYS:
        raise SigningError(
            f"{len(keys)} keys held, but the platform verifies at most {MAX_HELD_KEYS} "
            "signatures per request: hold the current key and one previous key while "
            "rotating, and retire the older one first"
        )
    lifetime = check_lifetime(lifetime)
    agent_url = assert_signable_agent_url(agent_url, allow_http=allow_http)
    target = request_target(method, url)
    body = bytes(body or b"")
    digest = content_digest(body) if body else None
    # Covered headers (file comment): resolved once, before any key signs.
    covered = resolve_covered_headers(headers, covered_headers)
    covered_names = [name for name, _ in covered]
    labels = signature_labels(label, len(keys))
    if nonces is not None:
        if len(nonces) != len(keys):
            raise SigningError("one nonce per key is required when nonces are given")
        if any(not isinstance(n, str) or not n for n in nonces):
            raise SigningError("a nonce must be a non-empty string")
        # The platform spends each nonce once: two labels sharing one are
        # `signature_replayed` on the second (the TypeScript signer refuses
        # the same, 2026-09-19).
        if len(set(nonces)) != len(nonces):
            raise SigningError("every label needs its own nonce: the same nonce was given for two keys")
    if created is not None and (isinstance(created, bool) or not isinstance(created, int)):
        raise SigningError("created must be an integer")
    created_at = int(time.time()) if created is None else created
    expires_at = created_at + lifetime

    inputs: List[Tuple[str, Member]] = []
    signatures: List[Tuple[str, Member]] = []
    records: List[SignatureRecord] = []
    for index, (key, label_n) in enumerate(zip(keys, labels)):
        nonce = nonces[index] if nonces is not None else new_nonce()
        params = SignatureParams(created=created_at, expires=expires_at, keyid=key.thumbprint, nonce=nonce)
        components = covered_components(form, label_n, digest is not None, covered_names)
        params_list = signature_params_inner_list(components, params)
        base = build_signature_base(
            component_values(target, agent_url, form, label_n, digest, covered), params_list
        )
        signature = key.private_key.sign(base.encode("ascii"))
        inputs.append((label_n, params_list))
        signatures.append((label_n, Item(signature)))
        records.append(
            SignatureRecord(
                label=label_n,
                keyid=key.thumbprint,
                created=created_at,
                expires=expires_at,
                nonce=nonce,
                base=base,
                signature=signature,
            )
        )

    headers = {
        "Signature-Agent": signature_agent_header(agent_url, form, labels),
        "Signature-Input": serialize_dictionary(inputs),
        "Signature": serialize_dictionary(signatures),
    }
    if digest is not None:
        headers["Content-Digest"] = digest
    return SignedRequest(headers=headers, signatures=records, target=target)


class WebBotAuth(httpx.Auth):
    """`httpx.Auth` that signs every request with the agent's held keys.

    `requires_request_body` makes httpx read the body before `auth_flow` runs,
    so `Content-Digest` covers the bytes actually sent. Any signature header
    already on the request (a re-sent `Request` object) is replaced, never
    appended to: a signature is per send, with a fresh nonce.

    `covered_headers` names plain header fields to cover on every request
    this auth signs (file comment, "COVERED HEADERS"); they are read off
    each request's own headers, so a request that lacks one is refused with
    a `SigningError` out of the auth flow. The MPP buyer does not go through
    this class: its covered set changes per send (a credential and an assent
    ride only the paid retry), so it calls `sign_request` directly.
    """

    requires_request_body = True

    def __init__(
        self,
        keys: Sequence[SigningKey],
        agent_url: str,
        *,
        form: str = DEFAULT_SIGNATURE_AGENT_FORM,
        label: str = DEFAULT_LABEL,
        lifetime: int = SIGNATURE_LIFETIME_S,
        allow_http: Optional[bool] = None,
        covered_headers: Optional[Sequence[str]] = None,
    ) -> None:
        _check_form(form)
        if not keys:
            raise SigningError("no signing key: call JWKSManager.ensure_ed25519_key first")
        self.keys = list(keys)
        # Refused here, at construction, rather than on the first send, so a
        # loopback or plaintext agent URL is one clear sentence and not an
        # exception out of httpx's auth flow.
        self.agent_url = assert_signable_agent_url(agent_url, allow_http=allow_http)
        self.form = form
        self.label = label
        self.lifetime = check_lifetime(lifetime)
        self.allow_http = allow_http
        # Normalised at construction for the same reason: a reserved or
        # malformed name is one sentence here, not a failure on every send.
        self.covered_headers = normalize_covered_headers(covered_headers)

    def auth_flow(self, request: httpx.Request):
        signed = sign_request(
            self.keys,
            self.agent_url,
            request.method,
            request.url,
            request.content,
            form=self.form,
            label=self.label,
            lifetime=self.lifetime,
            allow_http=self.allow_http,
            headers=request.headers,
            covered_headers=self.covered_headers,
        )
        for name in SIGNATURE_HEADERS:
            request.headers.pop(name, None)
        request.headers.update(signed.headers)
        # The URL sent is the URL signed ("The request target" above).
        if signed.target is not None:
            apply_signed_target(request, signed.target)
        yield request


__all__ = [
    "CARD_SUFFIX",
    "DEFAULT_LABEL",
    "DEFAULT_SIGNATURE_AGENT_FORM",
    "InnerList",
    "Item",
    "KEY_SET_SUFFIX",
    "NONCE_BYTES",
    "RequestTarget",
    "SIGNATURE_AGENT_FORMS",
    "SIGNATURE_ALG",
    "SIGNATURE_HEADERS",
    "SIGNATURE_LIFETIME_S",
    "SIGNATURE_TAG",
    "SignatureParams",
    "SignatureRecord",
    "SignedRequest",
    "SigningError",
    "SigningKey",
    "Token",
    "WebBotAuth",
    "agent_card_url",
    "apply_signed_target",
    "check_lifetime",
    "SIGNATURE_MAX_LIFETIME_S",
    "assert_signable_agent_url",
    "canonical_agent_url",
    "MAX_HELD_KEYS",
    "build_signature_base",
    "component_values",
    "content_digest",
    "covered_components",
    "ed25519_public_jwk",
    "jwk_thumbprint",
    "key_set_url",
    "new_nonce",
    "normalize_agent_url",
    "request_target",
    "serialize_bare_item",
    "serialize_dictionary",
    "serialize_inner_list",
    "serialize_item",
    "serialize_member",
    "serialize_parameters",
    "sign_request",
    "signature_agent_component",
    "signature_agent_component_value",
    "signature_agent_header",
    "signature_agent_member",
    "signature_agent_value",
    "signature_labels",
    "signature_params_inner_list",
]
