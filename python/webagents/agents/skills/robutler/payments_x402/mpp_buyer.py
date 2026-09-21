"""
The MPP buyer: how an agent built on this SDK buys platform usage from
Robutler when a paid resource answers 402 (machine-purchase design, sections
2.1, 6.3 and 6.4; pass P9b, 2026-09-18). The Python twin of the TypeScript
SDK's `src/skills/payments/mpp-buyer.ts` (pass P9a), decision for decision:
what goes on the wire is the same bytes in the same order, and
`tests/fixtures/web_bot_auth/vectors-covered-headers.json` pins the
credential encoder and the covered signature across both languages.

WHAT IT DOES. `paying_fetch` sends a signed request; when the answer is a 402
carrying one `WWW-Authenticate: Payment` challenge whose `realm` is the host
it asked, and that host is on the policy's `realms` allowlist, it checks the
operator's policy, obtains a payment from a source the operator configured,
builds the credential, adds the Terms acceptance and re-sends THE SAME request
once with both headers among the signature's covered components. A fresh
challenge on a 402 that charged nothing (an expired challenge, a stale Terms
version) is paid again, a bounded number of times. Everything else is
returned to the caller as it came. `purchase` is the in-band variant: a UAMP
`payment.required` carries the challenge and a purchase URL, and the buyer
pays it there with the same loop, after which the caller resumes on the
socket with `payment.submit` scheme `balance`. `upgrade_headers` signs that
socket's upgrade, so the platform's socket door can tell who is asking and
send the in-band challenge at all.

The 2026-09-18 fix pass changed the following, identically in the TypeScript
twin, whose file comment carries the full reasoning:

- WHO IT PAYS (S-147). `policy.realms` is required (default: the host of
  `platform_url`; construction raises with neither). A challenge is paid only
  when its realm and the host it came from, or the purchase URL, are on it,
  checked before any source call and before any held credential is
  re-presented. "Realm equals the host asked" held for any host, so a
  model-chosen delegate URL or a UAMP peer could collect the payment.
- THE ANSWER AFTER A CREDENTIAL (the 503 contract shared with the portal,
  read by body CODE and not by the flag alone). SETTLED: a 2xx, any body with
  a `purchase` block (the door's 402 after a grant, 503 `funding_failed`, 409
  `purchase_already_granted` after a lost 200), or a 503 `funding_failed`; a
  settled answer with `retry.sameCredential: false` means "send the call
  again WITHOUT a credential", done once. RE-PRESENT: a 503 with
  `retry.sameCredential: true` (`sale_serve_pending` among them) and nothing
  else. SERVED ELSEWHERE: 409 `call_already_served` or `call_in_progress`:
  paid, terminal, never paid again and never re-presented. REFUNDED: 409
  `call_refunded`: the reservation is freed. UNKNOWN: any other 5xx, or an
  exception once the credential left. REFUSED: anything else, including a
  503 whose `sameCredential: false` names no payment (`mpp_not_configured`,
  `metered_not_ready`) and the velocity 429s: freed, reported through
  `on_refusal` as `platform_refused` for a 429 or 503, never retried.
- NEVER DISCARDING A CREDENTIAL (S-150). Re-presentation backs off
  (`Retry-After`, at least 5 s doubling, at most `max_retry_after_seconds`)
  until the challenge's expiry plus `settlement_grace_seconds`. Then, or on an
  unknown outcome, the credential is HELD: exposed in `on_refusal`, in
  `purchase`'s outcome and by `pending_credentials()`, kept counted, saved
  through `policy.persist`, and re-presented before this buyer pays again for
  the same resource (a pack credential also at a purchase URL). It used to be
  dropped after about 15 s, and the next call signed a second transfer.
- WHICH SELLER IT PAYS (S-154 residual). A card challenge is paid only when
  its `methodDetails.networkId` equals the pinned Robutler Stripe profile, a
  Tempo challenge only when its `recipient` equals the pinned deposit
  address, else `seller_not_pinned` before any source call: frames on an
  external agent's socket are relayed raw, so a challenge can read "realm =
  the platform" while naming an attacker's profile or wallet. The pin is
  `policy.stripe_profile_id` / `policy.tempo_deposit_address`, else read per
  allowlisted host from `https://<host>/openapi.json` (unsigned, https only,
  no redirects, 1 MB counted while it streams, 10 s, believed for an hour,
  see the 2026-09-19 list below), `payTo` on the stripe offer and
  `recipient` on the tempo offer. None, or two, pins nothing: fail closed. The
  platform's discovery offers carry both fields (lib/openapi/mpp-discovery.ts
  in the portal, since 2026-09-18), so no explicit pin is needed against it.
- ONE PURCHASE PER CALL (S-151). A 402 or 503 carrying a `purchase` block is a
  PAID pack: counted, reported through `on_purchase`. A fresh challenge after
  it is paid only while `policy.max_purchases_per_call` (default 1) allows.
- THE LEDGER is per process unless `policy.persist` is supplied. The daily cap
  is checked and reserved in one synchronous step before any await, so
  concurrent calls or a burst of UAMP events cannot all pass it. `persist`
  carries the ledger and the held credentials across restarts; it does not
  make replicas share one cap atomically.
- TIMEOUTS. The in-band purchase sends with an explicit
  `purchase_timeout_seconds` (default 60, the TypeScript value), and a client
  the buyer opens itself uses the same, never httpx's 5 s default, which
  raised mid Tempo redeem, after the broadcast.

The 2026-09-19 review changed the following, again identically in the
TypeScript twin:

- NO REDIRECT IS EVER FOLLOWED (S-180). Every send passes
  `follow_redirects=False` itself. httpx does not follow by default, but a
  client the OPERATOR built with `follow_redirects=True` did, and httpx strips
  only `Authorization` on a cross-origin redirect, so `Payment-Authorization`
  and `Robutler-Terms-Accepted` would have travelled to whatever the
  `Location` named and that host's 2xx been read as a settled purchase. A 3xx
  (301, 302, 303, 307, 308) is now a typed refusal, `redirect_refused`:
  `paying_fetch` raises `MppRedirectError` and `purchase` reports it as its
  outcome. It spends nothing: a fresh credential's reservation is released;
  one ALREADY in doubt stays held, since a redirect says nothing about
  whether it settled. It used to come back as an ordinary refused answer.
- THE CREDENTIAL HEADER IS CHECKED WHEN THE CHALLENGE IS READ. A challenge
  names the field its credential goes in (`header`). A name the signer
  refuses to cover (`content-digest`, `signature`, ...), one of the buyer's
  own signed fields, or a framing field the HTTP client owns
  (`content-length`, `connection`, ...) made the paid retry raise AFTER
  `_obtain` had the card source issue a token or the wallet sign, and the
  credential was then held against the daily cap although nothing was sent.
  Such a challenge is now unreadable (`credential_header_refusal`), so it is
  refused before any source call. And a retry that fails BEFORE it is sent
  (the signer refusing, for any reason) releases a fresh credential's
  reservation instead of holding it.
- A DISCOVERY PIN IS BELIEVED FOR AN HOUR (`DISCOVERY_PIN_TTL_MS`). A positive
  pin used to be cached for the life of the process, so a rotated deposit
  address was never re-read and every Tempo challenge was refused until
  restart. What the NEW read says is the whole answer: a re-read that fails,
  names no seller or names two pins nothing. The stale pin is never a
  fallback.
- THE URL SENT IS THE URL SIGNED. The signer signs the WHATWG spelling of the
  request target (`webagents/crypto/http_signature.py`, "The request
  target"), and `_prepare` moves the request onto it (`apply_signed_target`).
- A STREAMED ANSWER IS NEVER READ (finding sdk-1). Every send is
  `stream=True`, and whether a purchase settled is decided from what arrives
  with the HEAD: the status, and the `Payment-Receipt` header for the record.
  The buyer parses a body only when it is a small JSON document
  (`buyer_reads_body`: a JSON content type, never more than
  `BUYER_BODY_MAX_BYTES` declared or counted, and on a 2xx only with a
  declared length), which is every problem and purchase document the platform
  sends and nothing a caller streams. Until then every send read its whole
  answer before returning (httpx's non-streaming `send`), so a streamed
  completion after a paid retry reached the caller only once its last chunk
  had been sent, buffered whole, and an error body was read without bound.
  WHAT THE CALLER GETS BACK: a 2xx that declares no length (or more than the
  cap) is handed over UNREAD and open, as it arrived: iterate it
  (`aiter_lines`, `aiter_bytes`) or `aread()` it, then `aclose()` it. Every
  other answer is already read, up to the cap, so `.text` and `.json()` work
  on it as they always did; one past the cap is left unread with nothing
  lost. A client the buyer opened itself is closed when that response is.
- THE PURCHASE POINTER. A rail that has not verified its caller cannot mint a
  challenge for it: a token holder whose token ran dry on the `/llm` socket or
  on `POST /api/llm/chat/completions` is told where to buy instead, an `mpp`
  entry in `requirements` that carries `purchase_url` and NO `challenge` (the
  portal's lib/payments/purchase-pointer.ts). Both SDKs ignored such an
  entry. `purchase_at` acts on it: the buyer sends its own signed `POST` to
  the purchase URL, is answered 402 with the one challenge minted for ITS
  identity, and pays that under the whole policy (realm allowlist, seller
  pin, caps, Terms, held credentials first). Two things are checked BEFORE
  that first request leaves, because a pointer, unlike a challenge, carries
  no secret and anyone can write one: the purchase URL's host AND the host
  that named it (`source_url`: the resource that answered 402, or the socket)
  must both be on `policy.realms`. `paying_fetch` does the same on an HTTP
  402 whose JSON body carries `requirements` and whose `WWW-Authenticate`
  carries no challenge, then re-sends the original request once WITHOUT its
  payment token (`X-Payment-Token`, `X-PAYMENT`, `?payment_token=`): the
  purchase funded the signing identity's balance, and the platform's door
  serves from that balance only a signed request that names no token;
  re-sending the dry token could only answer 402 again, a pack bought and
  nothing served. A pointer purchase counts against `max_purchases_per_call`
  and the daily cap like any other, and `call` (any hashable the caller keeps
  for one call) carries the per-call count across separate `purchase` and
  `purchase_at` invocations.
- A POINTER IS FOLLOWED ONLY UNDER A DAILY CAP (the same day). The two host
  checks say where a pointer may come from and where it may lead; they do not
  bound HOW MUCH it can make this agent buy. A pointer carries no secret, so
  any peer reachable THROUGH an allowlisted host can write one: an external
  agent's UAMP frames are relayed raw by the platform, so they arrive from the
  platform's own host. Nobody but the pinned seller is paid and the usage
  lands on this agent's OWN balance, but it is the operator's card or wallet
  that is spent, on a peer's say-so. `max_purchases_per_call` bounds that per
  call and `daily_cap_cents` per day; with no daily cap configured the total
  across calls had no bound at all, because a peer starts a new call whenever
  it likes. So `purchase_at`, and `paying_fetch` for any purchase URL a 402
  BODY names, refuse with `pointer_needs_daily_cap` before anything is sent
  when `policy.daily_cap_cents` is None. An entry in a 402 body that does
  carry a challenge is held to the same rule: its purchase URL is still one
  the caller did not choose, and no platform rail sends that form over HTTP
  (a verified HTTP caller is challenged in `WWW-Authenticate`). An ordinary
  challenge, on the URL the caller itself chose to fetch, and an in-band
  `purchase`, are paid exactly as before, with or without a daily cap.

WHAT IT NEVER HOLDS. The SDK holds no card and no key it was not given: a
card pays through `CardPaymentSource.get_spt`, which returns a Stripe shared
payment token issued to Robutler's network profile for exactly the challenge
amount, and a stablecoin pays through
`StablecoinPaymentSource.sign_tempo_transfer`, which returns a signed,
UNBROADCAST transaction; the platform broadcasts it. Both are protocols the
operator implements (sync or async, either is awaited); nothing here reads an
environment variable for a secret.

WHAT THE POLICY DECIDES, and only the policy: `max_per_purchase_cents` (no
single challenge above it is paid), `daily_cap_cents` (a rolling 24 hour sum
of what was presented), `prefer_purchase` (`pack` or `exact`, sent as the
signed `Robutler-Purchase` hint the door reads, design D2), `methods` (the
order the buyer will pay in, sent as the signed `Robutler-Payment-Methods`
hint), and `accept_terms`: a pinned version, or a callback the operator sets
once. A challenge names the Terms version it is minted under and the paid
retry must carry `Robutler-Terms-Accepted` equal to it, signed; if the
operator's policy does not accept that version, nothing is paid and the 402
is returned. The platform records the assent against the agent row and the
signing key, so accepting is a legal act of the operator's, which is why it
is never defaulted here.

THE CHALLENGE PARSER is a port of the platform's pure `challenge.ts` by way of
the TypeScript buyer (`parseWwwAuthenticatePayment`, RFC 8785 JCS, strict
base64url). Where JavaScript and Python regular expressions differ (`\\s`,
`.`, `$`, `\\d`, case folding) the patterns below spell out the JavaScript
meaning, so a header parses to the same fields in both SDKs. The credential
is `Payment <base64url of JCS {challenge, source?, payload}>`, the challenge
echoed field for field as parsed, which is what the platform re-HMACs.

SIGNING. The buyer signs every send itself through `sign_request` rather than
through `WebBotAuth`, because its covered set changes per send: the hints
ride every request, the credential and the assent only the paid retry
(`webagents/crypto/http_signature.py`, "COVERED HEADERS"). The client's own
`auth` is disabled for these sends so nothing re-signs over the buyer.

Copy rule: every string here is machine-facing or operator-facing. The buyer
buys platform usage from Robutler; it never pays an agent or a creator, and
nothing here states a rate between usage and money.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import httpx

from webagents.crypto.http_signature import (
    DEFAULT_LABEL,
    DEFAULT_SIGNATURE_AGENT_FORM,
    SIGNATURE_HEADERS,
    SIGNATURE_LIFETIME_S,
    SigningKey,
    apply_signed_target,
    assert_signable_agent_url,
    check_lifetime,
    normalize_covered_headers,
    sign_request,
)

# --------------------------------------------------------------------------
# Wire constants (design sections 2.3 and 6.3)
# --------------------------------------------------------------------------

#: The core spec's alternate credential field, which every platform challenge names in `header`.
MPP_CREDENTIAL_HEADER = "Payment-Authorization"
MPP_RECEIPT_HEADER = "Payment-Receipt"
#: The paid retry's assent header; among the covered components on a signed request (design section 6.3).
TERMS_ACCEPTED_HEADER = "Robutler-Terms-Accepted"
#: The version every payment 402 names beside its `Link`.
TERMS_VERSION_HEADER = "Robutler-Terms-Version"
#: D2: the buyer's signed choice between a pack and an exact top-up.
PURCHASE_HINT_HEADER = "Robutler-Purchase"
#: Design 6.1: the buyer's signed method order, `tempo, stripe`, a comma and space between names.
PAYMENT_METHODS_HINT_HEADER = "Robutler-Payment-Methods"
MPP_PROBLEM_BASE = "https://paymentauth.org/problems/"
ROBUTLER_PROBLEM_BASE = "https://robutler.ai/problems/"
MPP_INTENT_CHARGE = "charge"

MPP_METHODS: Tuple[str, ...] = ("stripe", "tempo")
MPP_PURCHASE_KINDS: Tuple[str, ...] = ("pack", "exact")

#: The Tempo unit: cents times 10000 (design section 6.6).
TEMPO_UNITS_PER_CENT = 10_000

# Number.MAX_SAFE_INTEGER: the TypeScript buyer's `Number.isSafeInteger` bound.
_MAX_SAFE_INTEGER = 2**53 - 1

DEFAULT_MAX_CHALLENGES = 2
DEFAULT_MAX_RETRY_AFTER_SECONDS = 30
DEFAULT_RETRY_AFTER_SECONDS = 5
_DAY_SECONDS = 24 * 60 * 60

# --------------------------------------------------------------------------
# JavaScript regular-expression semantics, spelled out
# --------------------------------------------------------------------------

# ECMAScript `\s`: WhiteSpace plus LineTerminator. Python's `\s` differs at
# \x1c-\x1f, \x85 and \ufeff, so the parser uses this class instead.
_JS_WS = "\t\n\x0b\x0c\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
# ECMAScript `.` without the `s` flag: anything but a LineTerminator.
_JS_DOT = "[^\n\r\u2028\u2029]"

# `/(?:^|,)\s*Payment\s+/i`. ASCII-only case folding, as JavaScript's `/i`
# without `u` never folds a non-ASCII letter onto an ASCII one.
_PAYMENT_START_RE = re.compile(
    r"(?:^|,)[" + _JS_WS + r"]*(?i:payment)[" + _JS_WS + r"]+",
    re.ASCII,
)
# `/([A-Za-z0-9_-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))\s*(?:,\s*|$)/y`,
# applied with `match(text, pos)`, which anchors like the sticky flag. `\Z`
# is JavaScript's `$` without `m`: Python's `$` would also match before a
# trailing newline.
_PARAM_RE = re.compile(
    r"([A-Za-z0-9_-]+)[" + _JS_WS + r"]*=[" + _JS_WS + r"]*"
    r'(?:"((?:[^"\\]|\\' + _JS_DOT + r')*)"|([^' + _JS_WS + r',"]+))'
    r"[" + _JS_WS + r"]*(?:,[" + _JS_WS + r"]*|\Z)"
)
_UNESCAPE_RE = re.compile(r"\\(" + _JS_DOT + r")")
_BASE64URL_RE = re.compile(r"[A-Za-z0-9_-]+")
_ASCII_DIGITS_RE = re.compile(r"[0-9]+")
_SPT_RE = re.compile(r"spt_[A-Za-z0-9_]+")
_TERMS_LINK_RE = re.compile(r'<([^>]+)>[' + _JS_WS + r']*;[^,]*rel="?terms-of-service"?', re.IGNORECASE | re.ASCII)
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


# --------------------------------------------------------------------------
# RFC 8785 JCS and base64url
# --------------------------------------------------------------------------


def _js_number(value: float) -> str:
    """ECMAScript Number::toString(10), which `JSON.stringify` uses: the
    shortest digits that round-trip (Python's `repr` finds the same ones),
    laid out by the ECMA-262 rules, so `1e30` is `1e+30` and `5.0` is `5`."""
    if value == 0:
        return "0"
    if value < 0:
        return "-" + _js_number(-value)
    _, digit_tuple, raw_exponent = Decimal(repr(value)).as_tuple()
    all_digits = "".join(str(d) for d in digit_tuple)
    # repr may carry trailing zeros (`100.0`); move them into the exponent so
    # `digits` is the shortest significand ECMA-262 names `s`.
    digits = all_digits.rstrip("0") or "0"
    exponent = int(raw_exponent) + (len(all_digits) - len(digits))
    k = len(digits)
    n = exponent + k
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    exp = ("+" if e >= 0 else "-") + str(abs(e))
    if k == 1:
        return digits + "e" + exp
    return digits[0] + "." + digits[1:] + "e" + exp


def _js_string(value: str) -> str:
    """`JSON.stringify` of a string: Python's `json.dumps` escapes the same
    set in the same spelling (lowercase `\\u00xx`), except that JavaScript
    escapes a lone surrogate and Python would carry it raw."""
    out = json.dumps(value, ensure_ascii=False)
    return _LONE_SURROGATE_RE.sub(lambda m: "\\u%04x" % ord(m.group()), out)


def jcs_canonicalize(value: Any) -> str:
    """RFC 8785 canonical JSON: keys sorted by UTF-16 code units, no
    whitespace, ECMAScript number and string forms. Byte-identical to the
    platform's `canonicalize` and the TypeScript `jcsCanonicalize`, which is
    what makes the echoed challenge re-HMAC on the server. `None` is `null`
    (Python has no `undefined`, so no member is dropped); a tuple is an
    array; a non-string key or a non-finite number is refused."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _js_string(value)
    if isinstance(value, int):
        # A JavaScript number is a double: an integer past 2**53 is the
        # double it rounds to, and prints as that double does.
        if abs(value) <= _MAX_SAFE_INTEGER:
            return str(value)
        return _js_number(float(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JCS: non-finite numbers are not JSON")
        return _js_number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jcs_canonicalize(v) for v in value) + "]"
    if isinstance(value, Mapping):
        keys = list(value.keys())
        for key in keys:
            if not isinstance(key, str):
                raise ValueError(f"JCS: object keys must be strings, not {type(key).__name__}")
        keys.sort(key=lambda k: k.encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(_js_string(k) + ":" + jcs_canonicalize(value[k]) for k in keys) + "}"
    raise ValueError(f"JCS: cannot serialise a {type(value).__name__}")


def base64url_encode(data: Union[str, bytes]) -> str:
    """base64url with no padding; a string is encoded as UTF-8 first."""
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def base64url_decode(text: Any) -> Optional[bytes]:
    """Strict: the base64url alphabet only, no padding, else None."""
    if not isinstance(text, str) or not _BASE64URL_RE.fullmatch(text):
        return None
    if len(text) % 4 == 1:
        return None
    try:
        return base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))
    except (ValueError, TypeError):
        return None


def _reject_constant(name: str) -> Any:
    # JSON.parse refuses NaN and Infinity; json.loads would accept them.
    raise ValueError(f"not JSON: {name}")


def _decode_jcs_json(b64: Optional[str]) -> Any:
    if not b64:
        return None
    raw = base64url_decode(b64)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"), parse_constant=_reject_constant)
    except ValueError:
        return None


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_safe_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= _MAX_SAFE_INTEGER


# --------------------------------------------------------------------------
# The challenge, as the platform's pure module defines it
# --------------------------------------------------------------------------

_OPTIONAL_FIELDS = ("expires", "digest", "opaque", "header")


@dataclass(frozen=True)
class MppChallengeFields:
    """The auth-params of one challenge in their wire (base64url) forms; the
    object the credential echoes. An optional field the challenge did not
    carry is None and is never echoed."""

    id: str
    realm: str
    method: str
    intent: str
    request: str
    expires: Optional[str] = None
    digest: Optional[str] = None
    opaque: Optional[str] = None
    header: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """The echoed challenge: the five required fields, then each optional
        one only when present (the platform's HMAC input is rebuilt from the
        same slots)."""
        out = {"id": self.id, "realm": self.realm, "method": self.method, "intent": self.intent, "request": self.request}
        for name in _OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MppChallengeFields":
        return cls(**{k: data[k] for k in ("id", "realm", "method", "intent", "request", *_OPTIONAL_FIELDS) if k in data})


def parse_www_authenticate_payment(header: Optional[str]) -> Optional[MppChallengeFields]:
    """Parse the `Payment` challenge out of a `WWW-Authenticate` value. The
    value may list other schemes before or after it (httpx joins repeated
    fields with `, `), so the scan starts at the `Payment` scheme and stops at
    the first thing that is not an `auth-param`, which is where another scheme
    would begin. None when there is no `Payment` challenge or it lacks a
    required parameter (`id`, `realm`, `method`, `intent`, `request`). Values
    are unquoted and unescaped once, as the platform's parser does."""
    if not isinstance(header, str):
        return None
    start = _PAYMENT_START_RE.search(header)
    if not start:
        return None
    out: Dict[str, str] = {}
    pos = start.end()
    while pos < len(header):
        m = _PARAM_RE.match(header, pos)
        if not m:
            break
        out[m.group(1)] = _UNESCAPE_RE.sub(r"\1", m.group(2)) if m.group(2) is not None else m.group(3)
        pos = m.end()
    if not all(out.get(k) for k in ("id", "realm", "method", "intent", "request")):
        return None
    return MppChallengeFields(
        id=out["id"],
        realm=out["realm"],
        method=out["method"],
        intent=out["intent"],
        request=out["request"],
        expires=out.get("expires"),
        digest=out.get("digest"),
        opaque=out.get("opaque"),
        header=out.get("header"),
    )


@dataclass(frozen=True)
class MppChargeRequest:
    """The decoded `request` of a charge challenge. The Stripe method carries
    `methodDetails.networkId` and `paymentMethodTypes`; the Tempo method
    carries `recipient`, a token contract as `currency`, and `methodDetails`
    `chainId`, `memo` and `supportedModes` (design section 6.6). `amount` is a
    decimal integer string in the method's unit: cents for Stripe, cents
    times 10000 for Tempo."""

    amount: str
    currency: str
    method_details: Dict[str, Any]
    recipient: Optional[str] = None
    external_id: Optional[str] = None


def decode_challenge_request(request: Union[str, MppChallengeFields, None]) -> Optional[MppChargeRequest]:
    """The charge request a challenge's `request` parameter decodes to, or None."""
    b64 = request.request if isinstance(request, MppChallengeFields) else request
    decoded = _decode_jcs_json(b64)
    if not _is_record(decoded):
        return None
    amount = decoded.get("amount")
    if not isinstance(amount, str) or not _ASCII_DIGITS_RE.fullmatch(amount):
        return None
    currency = decoded.get("currency")
    if not isinstance(currency, str) or not currency:
        return None
    details = decoded.get("methodDetails")
    if not _is_record(details):
        return None
    recipient = decoded.get("recipient")
    external_id = decoded.get("externalId")
    return MppChargeRequest(
        amount=amount,
        currency=currency,
        method_details=details,
        recipient=recipient if isinstance(recipient, str) else None,
        external_id=external_id if isinstance(external_id, str) else None,
    )


def challenge_amount_cents(method: str, amount: Union[str, int]) -> Optional[int]:
    """A challenge's amount in whole cents whatever its method, or None when
    the unit does not divide or the amount is not a positive safe integer."""
    if isinstance(amount, str):
        if not _ASCII_DIGITS_RE.fullmatch(amount):
            return None
        units = int(amount)
    elif _is_safe_int(amount):
        units = int(amount)
    else:
        return None
    if units <= 0 or units > _MAX_SAFE_INTEGER:
        return None
    if method == "tempo":
        return units // TEMPO_UNITS_PER_CENT if units % TEMPO_UNITS_PER_CENT == 0 else None
    return units


def _parse_instant(value: Optional[str]) -> Optional[datetime]:
    """An ISO 8601 instant (the platform writes `2026-09-18T12:05:00.000Z`) or
    an HTTP date, as an aware UTC datetime; None when unreadable. A value with
    no offset is read as UTC. Written for Python 3.10, whose `fromisoformat`
    does not read `Z`."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    iso = text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text
    try:
        at = datetime.fromisoformat(iso)
    except ValueError:
        at = None
        # 3.10 reads only 3 or 6 fractional digits; normalise to 6.
        m = re.fullmatch(r"(.*T\d{2}:\d{2}:\d{2})\.(\d+)(.*)", iso)
        if m:
            try:
                at = datetime.fromisoformat(f"{m.group(1)}.{(m.group(2) + '000000')[:6]}{m.group(3)}")
            except ValueError:
                at = None
        if at is None:
            try:
                at = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
    if at is None:
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class MppChallenge:
    """One challenge, parsed and decoded, as the policy and the sources see it."""

    fields: MppChallengeFields
    request: MppChargeRequest
    #: Whole cents, whatever the method's own unit; None when unreadable.
    amount_cents: Optional[int]
    expires_at: Optional[datetime]
    #: The header the credential goes in: the challenge's `header` parameter, else `Authorization` (core spec).
    credential_header: str


# RFC 9110 section 5.1 token, lowercased: the field-name alphabet (the signer's own rule).
_CREDENTIAL_FIELD_NAME_RE = re.compile(r"[a-z0-9!#$%&'*+\-.^_`|~]+")

#: Field names a challenge may NOT put its credential in (file comment, "THE
#: CREDENTIAL HEADER IS CHECKED WHEN THE CHALLENGE IS READ"). Three groups,
#: identical in the TypeScript twin: the fields the signer refuses to cover
#: (the signature's own, and `content-digest`, which the body rule covers);
#: the buyer's own signed fields, where the credential would overwrite one or
#: be overwritten by it; and message framing and connection management, which
#: the HTTP client owns and refuses (`content-length`) or silently rewrites
#: (`host`). `authorization` is NOT here: it is the core spec's own default
#: when a challenge names no `header`, and the signer covers it like any
#: other field.
CREDENTIAL_HEADERS_REFUSED = frozenset(
    {
        "content-digest",
        "signature-agent",
        "signature-input",
        "signature",
        "robutler-terms-accepted",
        "robutler-purchase",
        "robutler-payment-methods",
        "host",
        "content-length",
        "content-type",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "te",
        "trailer",
        "expect",
        "cookie",
        "proxy-authorization",
        "proxy-connection",
    }
)


def credential_header_refusal(name: str) -> Optional[str]:
    """Why a challenge's credential header cannot be used, or None when it can."""
    lower = name.strip().lower()
    if not _CREDENTIAL_FIELD_NAME_RE.fullmatch(lower):
        return f"{name!r} is not a header field name"
    if lower in CREDENTIAL_HEADERS_REFUSED:
        return f"{lower} is a field the signer, the buyer or the HTTP client owns; a credential cannot be sent in it"
    return None


def read_mpp_challenge(header: Optional[str]) -> Optional[MppChallenge]:
    """Parse and decode a `WWW-Authenticate` value, or None when it carries no
    readable `Payment` challenge. A challenge whose `header` parameter names a
    field the credential cannot be sent in (`credential_header_refusal`) is
    not readable: it is refused HERE, before the policy reserves anything and
    before any payment source is asked."""
    fields = parse_www_authenticate_payment(header)
    if fields is None:
        return None
    request = decode_challenge_request(fields.request)
    if request is None:
        return None
    credential_header = fields.header.strip() if fields.header and fields.header.strip() else "Authorization"
    if credential_header_refusal(credential_header) is not None:
        return None
    return MppChallenge(
        fields=fields,
        request=request,
        amount_cents=challenge_amount_cents(fields.method, request.amount),
        expires_at=_parse_instant(fields.expires) if fields.expires else None,
        credential_header=credential_header,
    )


# --------------------------------------------------------------------------
# Credential and receipt
# --------------------------------------------------------------------------


def encode_mpp_credential(
    fields: Union[MppChallengeFields, Mapping[str, Any]],
    payload: Mapping[str, Any],
    source: Optional[str] = None,
) -> str:
    """`Payment <token68>`: base64url of JCS `{challenge, source?, payload}`,
    the challenge echoed exactly as parsed (absent optional fields stay
    absent, so the platform's HMAC input is rebuilt from the same slots)."""
    challenge = fields.to_dict() if isinstance(fields, MppChallengeFields) else MppChallengeFields.from_dict(fields).to_dict()
    body: Dict[str, Any] = {"challenge": challenge, "payload": dict(payload)}
    if source is not None:
        body["source"] = source
    return "Payment " + base64url_encode(jcs_canonicalize(body))


def tempo_credential_payload(serialized_transaction: str) -> Dict[str, Any]:
    """The Tempo credential payload around a signed, unbroadcast transaction.
    Checked against the platform's verifier on 2026-09-18
    (`verifyMppTempoCredential`, lib/payments/mpp/challenge.ts): pull mode is
    `type: "transaction"` and `signature` is the hex type 0x76 transaction,
    which `TEMPO_SIGNED_TRANSACTION_RE` mirrors."""
    return {"type": "transaction", "signature": serialized_transaction}


#: The platform's pull-mode shape: `0x76` (the Tempo transaction type) then whole bytes.
TEMPO_SIGNED_TRANSACTION_RE = re.compile(r"0x76(?:[0-9a-fA-F]{2})+")
_TEMPO_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
_TEMPO_PAYER_DID_RE = re.compile(r"did:pkh:eip155:([1-9][0-9]*):(0x[0-9a-fA-F]{40})")


def tempo_payer_did(chain_id: int, address: str) -> str:
    """The credential `source` for a Tempo payer:
    `did:pkh:eip155:<chainId>:<address>`, the only form the platform's
    `parseTempoPayerDid` accepts (lib/payments/mpp/tempo.ts). Both SDKs sent
    the bare address until 2026-09-18; the platform refused it as
    `malformed-credential` and answered a fresh challenge, so every stablecoin
    purchase from a wallet exposing an address failed after the wallet had
    signed twice. A DID passes through when it names the challenge's chain.
    Raises on anything else, and the caller asks this BEFORE the wallet
    signs."""
    did = _TEMPO_PAYER_DID_RE.fullmatch(address) if isinstance(address, str) else None
    if did:
        if int(did.group(1)) != chain_id:
            raise ValueError(f"the stablecoin source's DID names chain {did.group(1)} and the challenge names chain {chain_id}")
        return address
    if not isinstance(address, str) or not _TEMPO_ADDRESS_RE.fullmatch(address):
        raise ValueError("the stablecoin source address is neither a 0x address of 40 hex digits nor a did:pkh:eip155 DID")
    return f"did:pkh:eip155:{chain_id}:{address}"


@dataclass(frozen=True)
class MppReceipt:
    status: str
    method: str
    timestamp: str
    reference: str


def parse_payment_receipt(value: Optional[str]) -> Optional[MppReceipt]:
    """The `Payment-Receipt` value: base64url JCS `{status, method, timestamp, reference}`, or None."""
    if not value:
        return None
    decoded = _decode_jcs_json(value.strip())
    if not _is_record(decoded):
        return None
    parts = [decoded.get(k) for k in ("status", "method", "timestamp", "reference")]
    if not all(isinstance(p, str) for p in parts):
        return None
    return MppReceipt(status=parts[0], method=parts[1], timestamp=parts[2], reference=parts[3])


# --------------------------------------------------------------------------
# Response readers
# --------------------------------------------------------------------------


def retry_after_seconds(value: Optional[str], now: datetime, fallback: int = DEFAULT_RETRY_AFTER_SECONDS) -> int:
    """`Retry-After` as seconds: a delta, or an HTTP date relative to `now`; `fallback` when absent or unreadable."""
    if not value:
        return fallback
    text = value.strip()
    if _ASCII_DIGITS_RE.fullmatch(text):
        return int(text)
    at = _parse_instant(text)
    if at is None:
        return fallback
    return max(0, math.ceil((at - now).total_seconds()))


#: The most the buyer reads of any answer for itself: a problem or purchase document is a few kilobytes.
BUYER_BODY_MAX_BYTES = 64 * 1024
_JSON_MEDIA_TYPE_RE = re.compile(r"application/(?:[a-z0-9!#$&^_.+-]+\+)?json")
#: How a caller names a payment token; the machine door never serves a request that carries one (the portal's `presentedCredentials`).
_PAYMENT_TOKEN_HEADERS = ("x-payment-token", "x-payment")
_PAYMENT_TOKEN_QUERY = "payment_token"


def is_json_media_type(content_type: Optional[str]) -> bool:
    """`application/json` and every `application/*+json` (`application/problem+json`), parameters ignored."""
    media_type = (content_type or "").split(";")[0].strip().lower()
    return _JSON_MEDIA_TYPE_RE.fullmatch(media_type) is not None


def _declared_length(response: httpx.Response) -> Optional[int]:
    header = (response.headers.get("content-length") or "").strip()
    return int(header) if _ASCII_DIGITS_RE.fullmatch(header) else None


def _is_success(response: httpx.Response) -> bool:
    return 200 <= response.status_code < 300


def buyer_reads_body(response: httpx.Response, *, purchase_document: bool = False) -> bool:
    """Whether the buyer parses this answer's body for itself, decided from
    the HEAD alone (file comment, "A STREAMED ANSWER IS NEVER READ"; the
    TypeScript `buyerReadsBody`). Only a JSON document, never one that
    DECLARES more than `BUYER_BODY_MAX_BYTES`, and a 2xx only when it declares
    its length: a 2xx with no declared length is a stream the caller is
    waiting on, and it is handed over untouched. `purchase_document` lifts
    that last rule for the answer of a purchase URL the buyer itself called.
    An error answer needs no declared length (the platform's own 402s and
    503s are sent chunked); it is counted while it is read instead."""
    if not is_json_media_type(response.headers.get("content-type")):
        return False
    declared = _declared_length(response)
    if declared is not None and declared > BUYER_BODY_MAX_BYTES:
        return False
    if _is_success(response):
        return declared is not None or purchase_document
    return True


def _read_json(response: httpx.Response, *, purchase_document: bool = False) -> Optional[Dict[str, Any]]:
    """The JSON body of an answer, or None: only when `buyer_reads_body` says
    so, and only from what `_settle_body` already read (never a read of its
    own, so nothing here can wait on a stream)."""
    if not buyer_reads_body(response, purchase_document=purchase_document):
        return None
    try:
        parsed = json.loads(response.content.decode("utf-8", errors="replace"), parse_constant=_reject_constant)
    except (ValueError, httpx.ResponseNotRead, httpx.StreamError):
        return None
    return parsed if _is_record(parsed) else None


class _HandedStream(httpx.AsyncByteStream):
    """A response stream the buyer has looked into, handed on with nothing
    lost: the raw chunks it already took, then the rest of the original.
    Closing it closes the original, and `owner` when there is one: the client
    the buyer opened for this call, which must outlive a response the caller
    is still streaming and must not outlive it."""

    def __init__(
        self,
        inner: Any,
        prefix: Sequence[bytes] = (),
        rest: Optional[Any] = None,
        owner: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._inner = inner
        self._prefix = list(prefix)
        self._rest = rest
        self._owner = owner

    async def __aiter__(self):  # type: ignore[override]
        while self._prefix:
            yield self._prefix.pop(0)
        # `rest` None: nothing was taken, so the original is read from its start.
        async for chunk in self._inner if self._rest is None else self._rest:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            if self._owner is not None:
                owner, self._owner = self._owner, None
                await owner.aclose()


async def _settle_body(response: httpx.Response, timeout_s: float, *, purchase_document: bool = False) -> None:
    """Read what may be read of a streamed answer, and leave the rest as it
    arrived (file comment, "A STREAMED ANSWER IS NEVER READ"). Left UNREAD: a
    body that declares more than the cap, and a 2xx that declares no length
    (unless it is a purchase URL's own answer). Everything else is read raw,
    counted WHILE it streams and under `timeout_s`: within the cap it becomes
    the response's content, so `.text` and `.json()` work for the caller;
    past the cap the chunks taken are put back in front of the rest and the
    response stays unread; a body that stalls is closed, never waited on."""
    if response.is_closed or response.is_stream_consumed:
        return
    declared = _declared_length(response)
    if declared is not None and declared > BUYER_BODY_MAX_BYTES:
        return
    if _is_success(response) and declared is None and not purchase_document:
        return
    inner = response.stream
    rest = inner.__aiter__()  # type: ignore[union-attr]
    chunks: List[bytes] = []

    async def take() -> bool:
        size = 0
        async for chunk in rest:
            size += len(chunk)
            chunks.append(chunk)
            if size > BUYER_BODY_MAX_BYTES:
                return False
        return True

    try:
        whole = await asyncio.wait_for(take(), timeout=timeout_s)
    except (asyncio.TimeoutError, httpx.HTTPError):
        await response.aclose()
        return
    # `rest` is the iterator already advanced past `chunks`: exhausted when the whole body fitted.
    response.stream = _HandedStream(inner, chunks, rest)
    if whole:
        await response.aread()


async def _discard(response: Optional[httpx.Response]) -> None:
    """Close an answer the loop is moving past, so its connection goes back to the pool."""
    if response is not None and not response.is_closed:
        await response.aclose()


@dataclass(frozen=True)
class MppRequirementEntry:
    """The `mpp` entry of a `requirements` member (a UAMP `payment.required`,
    or the body of an HTTP 402 from a rail with no verified caller): the
    purchase URL, the challenge when the sender had verified its caller and
    minted one, and the Terms notice that came with it. `challenge` None is
    the platform's PURCHASE POINTER (file comment)."""

    purchase_url: str
    challenge: Optional[str] = None
    terms: Optional["MppTermsRequest"] = None


def mpp_requirement_of(requirements: Any) -> Optional[MppRequirementEntry]:
    """The first usable `mpp` entry of `requirements.schemes`, or None. An
    entry with NO `challenge` member is a pointer; one whose `challenge` is
    present and not a non-empty string is malformed and is never read as a
    pointer (the TypeScript `mppRequirementOf`)."""
    schemes = requirements.get("schemes") if _is_record(requirements) else None
    for entry in schemes if isinstance(schemes, list) else []:
        if not _is_record(entry) or entry.get("scheme") != "mpp":
            continue
        purchase_url = entry.get("purchase_url")
        if not isinstance(purchase_url, str) or not purchase_url.strip():
            continue
        challenge = entry.get("challenge")
        if challenge is not None and (not isinstance(challenge, str) or not challenge.strip()):
            continue
        terms = entry.get("terms")
        notice = None
        if _is_record(terms) and isinstance(terms.get("version"), str) and terms["version"].strip():
            notice = MppTermsRequest(version=terms["version"].strip(), url=terms["url"] if isinstance(terms.get("url"), str) else None)
        return MppRequirementEntry(purchase_url=purchase_url.strip(), challenge=challenge, terms=notice)
    return None


@dataclass(frozen=True)
class MppTermsRequest:
    """The Terms a challenge names: the version the paid retry accepts, and where to read them."""

    version: str
    url: Optional[str] = None


def read_terms_notice(response: httpx.Response, body: Optional[Mapping[str, Any]]) -> Optional[MppTermsRequest]:
    """The Terms the 402 names: the `Robutler-Terms-Version` header first,
    the problem body's `terms` object second; the url from the body, else
    from the `Link` with `rel="terms-of-service"`."""
    header_version = (response.headers.get(TERMS_VERSION_HEADER) or "").strip()
    terms = body.get("terms") if body is not None and _is_record(body.get("terms")) else None
    version = header_version or (terms["version"].strip() if terms and isinstance(terms.get("version"), str) else "")
    if not version:
        return None
    url = terms["url"] if terms and isinstance(terms.get("url"), str) else None
    if not url:
        link = response.headers.get("link")
        m = _TERMS_LINK_RE.search(link) if link else None
        if m:
            url = m.group(1)
    return MppTermsRequest(version=version, url=url)


def same_credential_of(body: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """`retry.sameCredential` of a problem body: True, False, or None when the body says neither."""
    if not body or not _is_record(body.get("retry")):
        return None
    same = body["retry"].get("sameCredential")
    return same if isinstance(same, bool) else None


def problem_code_of(body: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The problem body's `error` code, or None."""
    code = body.get("error") if body else None
    return code if isinstance(code, str) else None


def after_credential(status: int, body: Optional[Mapping[str, Any]]) -> str:
    """The file comment's readings of an answer to a request that carried a
    credential: `settled`, `retry_same`, `served_elsewhere`, `refunded`,
    `unknown` or `refused`. By CODE as well as flag (2026-09-18):
    `sameCredential: false` on a 503 is a paid purchase only for
    `funding_failed` (or a body with a `purchase` block);
    `mpp_not_configured` and `metered_not_ready` carry the same flag and
    charged nothing."""
    if 200 <= status < 300:
        return "settled"
    if body and _is_record(body.get("purchase")):
        return "settled"
    code = problem_code_of(body)
    same = same_credential_of(body)
    if status == 503 and same is True:
        return "retry_same"
    if status == 503 and code == "funding_failed":
        return "settled"
    if status == 409 and code in ("call_already_served", "call_in_progress"):
        return "served_elsewhere"
    if status == 409 and code == "call_refunded":
        return "refunded"
    if status == 503 and same is False:
        return "refused"
    if status >= 500:
        return "unknown"
    return "refused"


def receipt_of(response: httpx.Response, body: Optional[Mapping[str, Any]]) -> Optional[MppReceipt]:
    """The receipt of an answer: the `Payment-Receipt` header on a success,
    else the `receipt` a 409 carries in its body (an object, or the header
    form), since the platform never attaches the header to an error."""
    from_header = parse_payment_receipt(response.headers.get(MPP_RECEIPT_HEADER))
    if from_header is not None or not body:
        return from_header
    r = body.get("receipt")
    if isinstance(r, str):
        return parse_payment_receipt(r)
    if _is_record(r) and all(isinstance(r.get(k), str) for k in ("status", "method", "timestamp", "reference")):
        return MppReceipt(status=r["status"], method=r["method"], timestamp=r["timestamp"], reference=r["reference"])
    return None


# --------------------------------------------------------------------------
# Sources and policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SptRequest:
    #: Robutler's Stripe Business Network Profile id, the challenge's `methodDetails.networkId`.
    network_id: str
    amount_cents: int
    currency: str
    expires_at: Optional[datetime]
    challenge_id: str
    payment_method_types: List[str]


@runtime_checkable
class CardPaymentSource(Protocol):
    """A card source: returns a Stripe shared payment token (`spt_...`) issued
    to `network_id` for exactly `amount_cents`. In Stripe's sandbox that is
    `test_helpers/shared_payment/granted_tokens`; a Link Agent Wallet source
    has no documented API as of 2026-09-18 (design section 6.4, UNVERIFIED).
    May be sync or async."""

    def get_spt(self, request: SptRequest) -> Union[str, Awaitable[str]]: ...


@dataclass(frozen=True)
class TempoTransferRequest:
    chain_id: int
    #: The token contract the challenge names as `currency`.
    currency: str
    recipient: str
    #: Token units as the challenge states them (cents times 10000).
    amount: str
    memo: str
    valid_before: Optional[datetime]
    challenge_id: str


@runtime_checkable
class StablecoinPaymentSource(Protocol):
    """A stablecoin source: signs a Tempo `transferWithMemo` of exactly
    `amount` to `recipient` with the challenge's memo and returns the
    serialized, UNBROADCAST transaction (`0x76...`, the platform's pull-mode
    shape). The platform broadcasts it (pull mode, design section 6.6). An
    `address` attribute, when the source has one, is the payer: a `0x`
    address of 40 hex digits, or already its DID; the credential's `source`
    carries it as `did:pkh:eip155:<chainId>:<address>` (`tempo_payer_did`) so
    the platform can check it against the transaction's sender. May be sync
    or async."""

    def sign_tempo_transfer(self, request: TempoTransferRequest) -> Union[str, Awaitable[str]]: ...


AcceptTerms = Union[str, Callable[[MppTermsRequest], Union[bool, Awaitable[bool]]]]


@runtime_checkable
class MppBuyerPersistence(Protocol):
    """Where the ledger and the held credentials outlive the process (the
    TypeScript `MppBuyerPersistence`). `load` is asked once, before the first
    purchase decision, and merged into what the process already holds; `save`
    is handed the whole state after every change, in order, as the JSON-safe
    dict `{"version": 1, "ledger": [{"id", "at" (epoch ms), "cents"}],
    "pending": [MppPendingCredential.to_dict()]}`, the same shape the
    TypeScript buyer saves. A `save` that raises is logged and does not stop
    a purchase; a `load` that raises does. Either may be sync or async."""

    def load(self) -> Any: ...

    def save(self, state: Dict[str, Any]) -> Any: ...


@dataclass
class MppBuyerPolicy:
    """What the operator allows the buyer to do (file comment, "WHAT THE POLICY DECIDES")."""

    #: No single challenge above this is paid.
    max_per_purchase_cents: int
    #: The Terms acceptance, the operator's act: a pinned version (paid only
    #: when the challenge names exactly it) or a callback asked once per
    #: challenge, which returns True to accept. There is no default.
    accept_terms: AcceptTerms
    #: A rolling 24 hour cap on what is presented for payment. None means no
    #: daily cap, and then NO PURCHASE POINTER IS FOLLOWED
    #: (`pointer_needs_daily_cap`; file comment, "A POINTER IS FOLLOWED ONLY
    #: UNDER A DAILY CAP"): a challenge on a URL the caller chose is still
    #: paid. Per process unless `persist` is set.
    daily_cap_cents: Optional[int] = None
    #: D2: the signed `Robutler-Purchase` hint. None means the door's default
    #: (packs on cards, exact top-ups on stablecoin).
    prefer_purchase: Optional[str] = None
    #: The order the buyer pays in, sent as the signed
    #: `Robutler-Payment-Methods` hint. None: every method with a configured
    #: source, stablecoin first; the hint is then sent only when exactly one
    #: source is configured.
    methods: Optional[Sequence[str]] = None
    #: How many challenges one call may present before giving up. None: 2, the first and one fresh one.
    max_challenges: Optional[int] = None
    #: An optional hard cap on re-presentations within one call. None: the time budget alone bounds them.
    max_settlement_retries: Optional[int] = None
    #: The longest `Retry-After` (and backoff step) honoured, in seconds. None: 30.
    max_retry_after_seconds: Optional[int] = None
    #: S-147: the hosts this buyer pays, as `host`, `host:port` or a URL. A
    #: challenge is paid only when its `realm` and the host of the URL it came
    #: from (or the in-band purchase URL) are both here. None: the host of
    #: the buyer's `platform_url`; required when that is not set.
    realms: Optional[Sequence[str]] = None
    #: S-151: how many purchases that SETTLED one call may make. None: 1.
    max_purchases_per_call: Optional[int] = None
    #: S-150: how long past the challenge's expiry the buyer keeps
    #: re-presenting within one call, in seconds. None: 120.
    settlement_grace_seconds: Optional[int] = None
    #: Each request of an in-band `purchase`, in seconds. None: 60, the TypeScript value.
    purchase_timeout_seconds: Optional[int] = None
    #: Carries the ledger and the held credentials across restarts. Without it both are per process.
    persist: Optional[MppBuyerPersistence] = None
    #: S-154: Robutler's Stripe profile (`profile_...`); a card challenge naming another `networkId` is never paid. None: read from discovery.
    stripe_profile_id: Optional[str] = None
    #: S-154: Robutler's Tempo deposit address; a Tempo challenge naming another `recipient` is never paid. None: read from discovery.
    tempo_deposit_address: Optional[str] = None


MPP_BUYER_REFUSAL_REASONS: Tuple[str, ...] = (
    "no_challenge",
    "realm_not_allowed",
    "realm_mismatch",
    "unsupported_intent",
    "amount_unreadable",
    "challenge_expired",
    "method_unavailable",
    "over_max_per_purchase",
    "over_daily_cap",
    "terms_refused",
    "challenges_exhausted",
    "purchase_limit_per_call",
    # A purchase pointer is followed only when `policy.daily_cap_cents` is set: with none, nothing
    # bounds what a peer's pointers make this agent buy across calls. Nothing was sent.
    "pointer_needs_daily_cap",
    "settlement_retries_exhausted",
    "settlement_outcome_unknown",
    "pending_credential_refused",
    # A 429 or a no-payment 503 answered to a presented credential: nothing charged, not retried.
    "platform_refused",
    # S-154: the challenge's seller (card networkId, Tempo recipient) is not the pinned Robutler one, or none could be pinned.
    "seller_not_pinned",
    # S-180: the host answered a request with a redirect. Never followed; nothing spent.
    "redirect_refused",
)

#: The redirect statuses of the Fetch standard. A 304 or a 300 is an answer, not a redirect.
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Terminal answers to a presented credential that are not refusals
#: (2026-09-18, the portal's S-149): the per-call sale was served, is being
#: served, or was refunded, on an earlier presentation. Nothing is paid again.
MPP_TERMINAL_OUTCOMES: Tuple[str, ...] = ("call_already_served", "call_in_progress", "call_refunded")


def _iso(at: datetime) -> str:
    """An instant as JavaScript's `toISOString` writes it (`...T12:05:00.000Z`), so both SDKs store the same strings."""
    at = at.astimezone(timezone.utc)
    return at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{at.microsecond // 1000:03d}Z"


_PENDING_FIELDS = (
    ("id", "id"),
    ("realm", "realm"),
    ("http_method", "httpMethod"),
    ("url", "url"),
    ("header_name", "headerName"),
    ("value", "value"),
    ("payment_method", "paymentMethod"),
    ("kind", "kind"),
    ("amount_cents", "amountCents"),
    ("currency", "currency"),
    ("terms_version", "termsVersion"),
    ("expires_at", "expiresAt"),
    ("deadline", "deadline"),
    ("transaction_hash", "transactionHash"),
    ("held_at", "heldAt"),
    ("reason", "reason"),
)


@dataclass(frozen=True)
class MppPendingCredential:
    """A credential the buyer presented and cannot yet call settled or
    refused (S-150): the platform asked for it again past the settlement
    budget, or answered in a way that says nothing about whether money moved.
    Everything an operator needs to re-present it by hand is here: send any
    request with `http_method` to `url` (a door serves THAT request against
    the purchase), carrying `header_name: value` and, when `terms_version` is
    set, `Robutler-Terms-Accepted: terms_version`, both covered by the agent's
    signature. `to_dict` is the TypeScript `MppPendingCredential`, key for
    key, so one store serves either SDK."""

    id: str
    realm: str
    http_method: str
    url: str
    header_name: str
    value: str
    payment_method: str
    kind: Optional[str]
    amount_cents: int
    currency: str
    terms_version: Optional[str]
    expires_at: Optional[str]
    #: When the buyer's own re-presentation within a call stops: the expiry (or first presentation plus 300 s) plus the grace.
    deadline: str
    #: The Tempo transfer hash the platform named in its 503, when it did.
    transaction_hash: Optional[str]
    held_at: str
    #: `settlement_retries_exhausted`, `settlement_outcome_unknown`, or `refused` in a `pending_credential_refused` report.
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {wire: getattr(self, attr) for attr, wire in _PENDING_FIELDS}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["MppPendingCredential"]:
        """The record a store handed back, or None when it is not one."""
        if not isinstance(data, Mapping):
            return None
        for key in ("id", "realm", "httpMethod", "url", "headerName", "value", "paymentMethod", "currency", "deadline", "heldAt", "reason"):
            if not isinstance(data.get(key), str) or not data.get(key):
                return None
        for key in ("kind", "termsVersion", "expiresAt", "transactionHash"):
            if data.get(key) is not None and not isinstance(data.get(key), str):
                return None
        if not _is_safe_int(data.get("amountCents")) or data["amountCents"] <= 0 or _parse_instant(data["deadline"]) is None:
            return None
        return cls(**{attr: data.get(wire) for attr, wire in _PENDING_FIELDS})


class MppRedirectError(Exception):
    """A request the buyer sent was answered with a redirect (S-180; file
    comment, "NO REDIRECT IS EVER FOLLOWED"). Raised by `paying_fetch` and
    `paying_request`; `purchase` reports it as its outcome instead. `status`
    and `location` are the 3xx's own; `response` is that answer, for a caller
    that wants to look. `followed` is True only when the response says a
    redirect WAS followed (a transport that ignored `follow_redirects=False`):
    a credential on that request went to a host nobody named, so it is held
    as an unknown outcome rather than released. `pending_credential` is the
    credential that stays held because it was already in doubt (S-150)."""

    code = "redirect_refused"

    def __init__(
        self,
        url: str,
        *,
        status: Optional[int] = None,
        location: Optional[str] = None,
        followed: bool = False,
        response: Optional[httpx.Response] = None,
    ) -> None:
        if followed:
            message = (
                f"the request to {url} was redirected and the redirect was FOLLOWED by the configured transport, "
                "against follow_redirects=False; its answer is not read"
            )
        else:
            message = (
                f"{url} answered with a redirect{f' ({status})' if status else ''}{f' to {location}' if location else ''}; "
                "the buyer never follows one, because a followed request carries its payment credential to another host"
            )
        super().__init__(message)
        self.url = url
        self.status = status
        self.location = location
        self.followed = followed
        self.response = response
        self.pending_credential: Optional["MppPendingCredential"] = None


@dataclass(frozen=True)
class MppBuyerRefusal:
    reason: str
    url: str
    challenge_id: Optional[str]
    amount_cents: Optional[int]
    method: Optional[str]
    terms_version: Optional[str]
    detail: str
    #: The credential the refusal is about when its outcome is in doubt
    #: (S-150): held and re-presentable, or just refused after doubt.
    pending_credential: Optional[MppPendingCredential] = None


@dataclass(frozen=True)
class MppPurchaseRecord:
    url: str
    challenge_id: str
    method: str
    amount_cents: int
    currency: str
    terms_version: Optional[str]
    #: The `Payment-Receipt` of the answer, when one was attached.
    receipt: Optional[MppReceipt]
    #: The answer's status: 200 as a rule, 402 or 503 when the door granted
    #: the pack and still could not fund the call (S-151).
    status: int


@dataclass(frozen=True)
class MppPurchaseOutcome:
    """What `purchase` reports. `ok` with a `record`, or not `ok` with a
    `reason` (a refusal reason, `served_without_purchase` or
    `unexpected_status`), a `detail` and, when a credential's outcome is in
    doubt, that `pending_credential` for the operator to re-present.
    `redirect_refused` comes with `status` and `response` None: a redirect is
    never followed, so there is no answer."""

    ok: bool
    status: Optional[int]
    response: Optional[httpx.Response]
    record: Optional[MppPurchaseRecord] = None
    reason: Optional[str] = None
    detail: str = ""
    pending_credential: Optional[MppPendingCredential] = None
    #: The receipt a terminal 409 carries in its body (`call_already_served`, `call_in_progress`, `call_refunded`).
    receipt: Optional[MppReceipt] = None


# --------------------------------------------------------------------------
# The buyer
# --------------------------------------------------------------------------


@dataclass
class _Seed:
    method: str
    url: str
    headers: httpx.Headers
    body: bytes
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Credential:
    header_name: str
    value: str
    challenge_id: str
    method: str
    amount_cents: int
    currency: str
    terms_version: Optional[str]
    realm: str
    kind: Optional[str]
    expires_at: Optional[datetime]
    #: Epoch ms: this call's re-presentation stops here.
    deadline_ms: float
    transaction_hash: Optional[str] = None
    #: The platform asked for it again, or it came out of the held set: its outcome has been in doubt.
    in_doubt: bool = False


@dataclass
class _Pending:
    challenge: MppChallenge
    terms: Optional[MppTermsRequest]


@dataclass
class _LoopResult:
    response: httpx.Response
    #: The last credential sent, if any.
    presented: Optional[_Credential]
    #: The last credential that settled in this call, if any.
    settled: Optional[_Credential]
    refusal: Optional[str]
    detail: str
    pending: Optional[MppPendingCredential] = None
    receipt: Optional[MppReceipt] = None


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


DEFAULT_MAX_PURCHASES_PER_CALL = 1
DEFAULT_SETTLEMENT_GRACE_SECONDS = 120
#: A challenge with no `expires` (the platform always sends one) is given this long from its first presentation.
DEFAULT_SETTLEMENT_WINDOW_SECONDS = 300
DEFAULT_PURCHASE_TIMEOUT_SECONDS = 60
_DAY_MS = _DAY_SECONDS * 1000
#: S-154 discovery: the document path, its size and time bounds, and how long a document naming no seller is believed.
DISCOVERY_PATH = "/openapi.json"
_DISCOVERY_MAX_BYTES = 1024 * 1024
_DISCOVERY_TIMEOUT_S = 10.0
_DISCOVERY_NEGATIVE_TTL_MS = 300_000
#: How long a pin read from discovery is believed before it is read again: a rotated deposit address is picked up within the hour.
DISCOVERY_PIN_TTL_MS = 3_600_000


@dataclass(frozen=True)
class MppSellerPins:
    """The Robutler seller a discovery document names, per method; None where it names none (or two)."""

    stripe_profile_id: Optional[str] = None
    tempo_deposit_address: Optional[str] = None


def seller_pins_from_discovery(doc: Any) -> MppSellerPins:
    """The sellers a discovery document names (S-154, the TypeScript
    `sellerPinsFromDiscovery`): every operation's `x-payment-info.offers`,
    `payTo` on a `stripe` offer and `recipient` (or `extra.recipient`) on a
    `tempo` offer. Two different values for one method pin nothing."""
    stripe: set = set()
    tempo: set = set()
    paths = doc.get("paths") if _is_record(doc) and _is_record(doc.get("paths")) else {}
    for item in paths.values():
        if not _is_record(item):
            continue
        for op in item.values():
            info = op.get("x-payment-info") if _is_record(op) else None
            offers = info.get("offers") if _is_record(info) and isinstance(info.get("offers"), list) else []
            for offer in offers:
                if not _is_record(offer):
                    continue
                if offer.get("method") == "stripe" and isinstance(offer.get("payTo"), str) and offer["payTo"]:
                    stripe.add(offer["payTo"])
                if offer.get("method") == "tempo":
                    extra = offer.get("extra") if _is_record(offer.get("extra")) else {}
                    recipient = offer.get("recipient") if isinstance(offer.get("recipient"), str) else (
                        extra.get("recipient") if isinstance(extra.get("recipient"), str) else ""
                    )
                    if recipient:
                        tempo.add(recipient.lower())
    return MppSellerPins(
        stripe_profile_id=next(iter(stripe)) if len(stripe) == 1 else None,
        tempo_deposit_address=next(iter(tempo)) if len(tempo) == 1 else None,
    )
_REALM_ENTRY_FORBIDDEN_RE = re.compile(r"[/?#@\s]")
_log = logging.getLogger(__name__)


def parse_realm_entry(entry: str) -> Tuple[str, str]:
    """A `policy.realms` entry (`host`, `host:port` or a URL) as `(host,
    hostname)`, spelled as a WHATWG `URL` spells them (the TypeScript
    `parseRealmEntry`)."""
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("policy.realms entries must be non-empty hosts")
    raw = entry.strip()
    if "://" not in raw and _REALM_ENTRY_FORBIDDEN_RE.search(raw):
        raise ValueError(f"policy.realms entry {entry!r} is not a host")
    try:
        hostname, host = _whatwg_host_forms(raw if "://" in raw else f"https://{raw}")
    except (ValueError, httpx.InvalidURL) as e:
        raise ValueError(f"policy.realms entry {entry!r} is not a host") from e
    return host, hostname


def _resource_key(method: str, url: str) -> str:
    """`METHOD origin+path`: what one door is, whatever the query or body."""
    parsed = httpx.URL(url)
    _, host = _whatwg_host_forms(url)
    return f"{method.upper()} {parsed.scheme.lower()}://{host}{parsed.path}"


def _opaque_kind(fields: MppChallengeFields) -> Optional[str]:
    """The challenge opaque's `kind`; an opaque minted before kinds existed is a pack. None when unreadable."""
    decoded = _decode_jcs_json(fields.opaque)
    if not _is_record(decoded):
        return None
    kind = decoded.get("kind")
    return kind if isinstance(kind, str) else "pack"


def _now_ms(at: datetime) -> float:
    return at.timestamp() * 1000


class MppBuyer:
    """Buys platform usage from Robutler on a 402 (file comment).

    `keys` and `agent_url` are the agent's signing identity, exactly what
    `WebBotAuth` takes (`JWKSManager.held_ed25519_keys()` and the agent's
    public URL); the platform binds every challenge to that principal and
    key. `card` and `stablecoin` are the operator's payment sources, at least
    one. `platform_url` is the platform the buyer buys from; its host is the
    default `policy.realms`. `form`, `label`, `lifetime` and `allow_http`
    pass through to the signer; `covered_headers` are covered on every
    request beside the buyer's own. `client` is the `httpx.AsyncClient` to
    send with (its own `auth` is bypassed); without one a client is opened
    per call, with `purchase_timeout_seconds` as its timeout. `now` returns
    an aware datetime; `sleep` takes SECONDS (the TypeScript `sleep` takes
    milliseconds; the waits are the same)."""

    def __init__(
        self,
        *,
        keys: Sequence[SigningKey],
        agent_url: str,
        policy: Union[MppBuyerPolicy, Mapping[str, Any]],
        card: Optional[CardPaymentSource] = None,
        stablecoin: Optional[StablecoinPaymentSource] = None,
        platform_url: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        form: str = DEFAULT_SIGNATURE_AGENT_FORM,
        label: str = DEFAULT_LABEL,
        lifetime: int = SIGNATURE_LIFETIME_S,
        allow_http: Optional[bool] = None,
        covered_headers: Optional[Sequence[str]] = None,
        now: Optional[Callable[[], datetime]] = None,
        sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
        on_purchase: Optional[Callable[[MppPurchaseRecord], Any]] = None,
        on_refusal: Optional[Callable[[MppBuyerRefusal], Any]] = None,
    ) -> None:
        if not keys:
            raise ValueError("MppBuyer needs the agent identity it signs with (keys and agent_url)")
        if isinstance(policy, Mapping):
            policy = MppBuyerPolicy(**policy)
        if not isinstance(policy, MppBuyerPolicy):
            raise ValueError("MppBuyer needs a policy")
        if not _is_safe_int(policy.max_per_purchase_cents) or policy.max_per_purchase_cents <= 0:
            raise ValueError("policy.max_per_purchase_cents must be a positive whole number of cents")
        if policy.daily_cap_cents is not None and (not _is_safe_int(policy.daily_cap_cents) or policy.daily_cap_cents <= 0):
            raise ValueError("policy.daily_cap_cents must be a positive whole number of cents when set")
        if not isinstance(policy.accept_terms, str) and not callable(policy.accept_terms):
            raise ValueError(
                "policy.accept_terms must be a pinned Terms version or a callback; the SDK never accepts the Terms on its own"
            )
        if isinstance(policy.accept_terms, str) and not policy.accept_terms.strip():
            raise ValueError("policy.accept_terms cannot be an empty version")
        for method in policy.methods or ():
            if method not in MPP_METHODS:
                raise ValueError(f"policy.methods names an unknown method {method!r}")
        if policy.prefer_purchase is not None and policy.prefer_purchase not in MPP_PURCHASE_KINDS:
            raise ValueError(f"policy.prefer_purchase must be one of {', '.join(MPP_PURCHASE_KINDS)}")
        for name, zero_ok in (
            ("max_purchases_per_call", False),
            ("purchase_timeout_seconds", False),
            ("max_settlement_retries", True),
            ("settlement_grace_seconds", True),
        ):
            value = getattr(policy, name)
            if value is not None and (not _is_safe_int(value) or value < (0 if zero_ok else 1)):
                raise ValueError(f"policy.{name} must be a whole number{'' if zero_ok else ' above zero'} when set")
        if card is None and stablecoin is None:
            raise ValueError("MppBuyer needs at least one payment source (card or stablecoin)")
        self.policy = policy
        self.keys = list(keys)
        # Refused here rather than on the first send, as `WebBotAuth` does,
        # so a loopback or plaintext agent URL is one clear sentence.
        self.agent_url = assert_signable_agent_url(agent_url, allow_http=allow_http)
        entries = list(policy.realms) if policy.realms is not None else ([platform_url] if platform_url else [])
        if not entries:
            raise ValueError(
                "policy.realms must name the hosts this buyer pays (S-147: a challenge names its own seller), "
                "or set platform_url to default it to the platform host"
            )
        self._realms: List[Tuple[str, str]] = [parse_realm_entry(e) for e in entries]
        self.card = card
        self.stablecoin = stablecoin
        self.platform_url = platform_url
        self.client = client
        self.form = form
        self.label = label
        # Refused here, like the agent URL above: a whole number of seconds, 1 to 3600.
        self.lifetime = check_lifetime(lifetime)
        self.allow_http = allow_http
        self.covered_headers = normalize_covered_headers(covered_headers)
        self._now = now
        self._sleep = sleep
        self.on_purchase = on_purchase
        self.on_refusal = on_refusal
        self._ledger: List[Dict[str, Any]] = []
        self._held: Dict[str, MppPendingCredential] = {}
        #: Held ids a call is re-presenting right now, so a concurrent call does not present the same one.
        self._claimed: set = set()
        self._loaded = False
        #: Challenge ids `on_purchase` already fired for: a lost 200 answered later by a 409 is still ONE purchase.
        self._reported: Dict[str, None] = {}
        #: S-154: seller pins read from discovery, per allowlisted host: `(pins, at_ms)`; re-read after an hour, or after five minutes when the document names no seller for the method asked.
        self._discovered: Dict[str, Tuple[MppSellerPins, float]] = {}
        self._discovery_locks: Dict[str, asyncio.Lock] = {}
        self._load_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        #: Purchases that settled per `call` key, so `max_purchases_per_call` holds across separate `purchase` / `purchase_at` invocations of one call. Bounded, oldest out first.
        self._calls: Dict[Any, int] = {}

    # -- policy reads --------------------------------------------------------

    def method_order(self) -> List[str]:
        """The methods this buyer will pay with, in order: the policy's order,
        else stablecoin first, each only with a source."""
        wanted = self.policy.methods if self.policy.methods is not None else ("tempo", "stripe")
        return [m for m in wanted if self._has_source(m)]

    def allows_url(self, url: Union[str, httpx.URL]) -> bool:
        """True when `url`'s host is on the realm allowlist: the only hosts
        this buyer pays, or signs a socket upgrade for."""
        try:
            _, host = _whatwg_host_forms(str(url))
        except (ValueError, httpx.InvalidURL):
            return False
        return any(entry_host == host for entry_host, _ in self._realms)

    def _realm_allowed(self, realm: str) -> bool:
        r = realm.lower()
        return any(r == hostname or r == host for host, hostname in self._realms)

    def pending_credentials(self) -> List[MppPendingCredential]:
        """The credentials held because their outcome is in doubt (S-150), oldest first."""
        return sorted(self._held.values(), key=lambda p: p.held_at)

    def _has_source(self, method: str) -> bool:
        if method == "stripe":
            return self.card is not None
        if method == "tempo":
            return self.stablecoin is not None
        return False

    def now(self) -> datetime:
        at = self._now() if self._now else datetime.now(timezone.utc)
        return at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)

    def spent_today_cents(self) -> int:
        """Cents presented for payment in the last 24 hours, in this process (and what `persist` loaded)."""
        cutoff = _now_ms(self.now()) - _DAY_MS
        return sum(e["cents"] for e in self._ledger if e["at"] > cutoff)

    def _reserve(self, entry_id: str, cents: int) -> None:
        now = _now_ms(self.now())
        self._ledger = [e for e in self._ledger if e["at"] > now - _DAY_MS and e["id"] != entry_id]
        self._ledger.append({"id": entry_id, "at": now, "cents": cents})

    def _release(self, entry_id: str) -> None:
        self._ledger = [e for e in self._ledger if e["id"] != entry_id]

    # -- persistence ----------------------------------------------------------

    async def _ready(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            await self._load()
            self._loaded = True

    async def _load(self) -> None:
        persist = self.policy.persist
        if persist is None:
            return
        state = await _resolve(persist.load())
        if not isinstance(state, Mapping):
            return
        known = {e["id"] for e in self._ledger}
        ledger = state.get("ledger")
        for e in ledger if isinstance(ledger, list) else []:
            if (
                isinstance(e, Mapping)
                and isinstance(e.get("id"), str)
                and isinstance(e.get("at"), (int, float))
                and not isinstance(e.get("at"), bool)
                and _is_safe_int(e.get("cents"))
                and e["id"] not in known
            ):
                self._ledger.append({"id": e["id"], "at": e["at"], "cents": e["cents"]})
                known.add(e["id"])
        self._ledger.sort(key=lambda e: e["at"])
        pending = state.get("pending")
        for raw in pending if isinstance(pending, list) else []:
            record = MppPendingCredential.from_dict(raw)
            if record is not None and record.id not in self._held:
                self._held[record.id] = record

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "ledger": [dict(e) for e in self._ledger],
            "pending": [p.to_dict() for p in self.pending_credentials()],
        }

    async def _save(self) -> None:
        """Hand the state to `persist.save`, in order; the snapshot is taken now, before any await."""
        persist = self.policy.persist
        if persist is None:
            return
        state = self._snapshot()
        async with self._save_lock:
            try:
                await _resolve(persist.save(state))
            except Exception as e:  # noqa: BLE001 - a store failure must not stop a purchase
                _log.warning("[mpp-buyer] policy.persist.save failed: %s", e)

    def _pending_of(self, credential: _Credential, seed: _Seed, reason: str) -> MppPendingCredential:
        existing = self._held.get(credential.challenge_id)
        return MppPendingCredential(
            id=credential.challenge_id,
            realm=credential.realm,
            http_method=existing.http_method if existing else seed.method.upper(),
            url=existing.url if existing else seed.url,
            header_name=credential.header_name,
            value=credential.value,
            payment_method=credential.method,
            kind=credential.kind,
            amount_cents=credential.amount_cents,
            currency=credential.currency,
            terms_version=credential.terms_version,
            expires_at=_iso(credential.expires_at) if credential.expires_at else None,
            deadline=_iso(datetime.fromtimestamp(credential.deadline_ms / 1000, tz=timezone.utc)),
            transaction_hash=credential.transaction_hash,
            held_at=existing.held_at if existing else _iso(self.now()),
            reason=reason,
        )

    def _hold(self, credential: _Credential, seed: _Seed, reason: str) -> MppPendingCredential:
        record = self._pending_of(credential, seed, reason)
        self._held[record.id] = record
        return record

    def _held_for(self, seed: _Seed, realm: str, in_band: bool) -> Optional[MppPendingCredential]:
        """The oldest held credential this call may re-present on `seed`: the same resource, or a pack at a purchase URL."""
        key = _resource_key(seed.method, seed.url)
        r = realm.lower()
        for record in self.pending_credentials():
            if record.id in self._claimed or record.realm.lower() != r:
                continue
            if _resource_key(record.http_method, record.url) == key or (in_band and record.kind == "pack"):
                return record
        return None

    @staticmethod
    def _credential_of_held(record: MppPendingCredential) -> _Credential:
        deadline = _parse_instant(record.deadline)
        return _Credential(
            header_name=record.header_name,
            value=record.value,
            challenge_id=record.id,
            method=record.payment_method,
            amount_cents=record.amount_cents,
            currency=record.currency,
            terms_version=record.terms_version,
            realm=record.realm,
            kind=record.kind,
            expires_at=_parse_instant(record.expires_at) if record.expires_at else None,
            deadline_ms=_now_ms(deadline) if deadline else 0,
            transaction_hash=record.transaction_hash,
            in_doubt=True,
        )

    def _hint_headers(self) -> List[Tuple[str, str]]:
        """The signed hint headers, by name; covered on every request the
        buyer sends. The door mints ONE challenge, in the first method of its
        own order the hint allows (design 6.1), and that order is stablecoin
        first. A card-only buyer sending no hint met a Tempo challenge
        whenever stablecoin was on and refused it as `method_unavailable`
        (review, 2026-09-18), so a buyer that can pay only one way always says
        so; one that can pay both says nothing unless the policy orders
        them."""
        out: List[Tuple[str, str]] = []
        if self.policy.prefer_purchase:
            out.append((PURCHASE_HINT_HEADER, self.policy.prefer_purchase))
        order = self.method_order()
        if self.policy.methods is not None or len(order) == 1:
            out.append((PAYMENT_METHODS_HINT_HEADER, ", ".join(order)))
        return out

    async def upgrade_headers(
        self,
        url: str,
        *,
        created: Optional[int] = None,
        nonces: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """The headers that sign a UAMP socket's upgrade: a GET on the
        http(s) form of `url` (`wss:` is signed as `https:`, same host, path
        and query), which is exactly the request the platform's socket door
        rebuilds from the upgrade's Host header and path (`upgradeRequest`,
        lib/payments/machine-door-socket.ts) and verifies before it hands the
        socket off in-process. Nothing signed the upgrade until 2026-09-18,
        so the door never saw a signed agent and the in-band purchase could
        never start. The hints ride the upgrade, covered. Empty for a host
        off the realm allowlist. `created` and `nonces` are the signer's test
        seams, for the cross-language vectors only
        (`tests/fixtures/web_bot_auth/vectors-upgrade.json`)."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        mapped = {"wss": "https", "ws": "http", "https": "https", "http": "http"}.get(scheme)
        if mapped is None:
            raise ValueError(f"cannot sign an upgrade to a {scheme}: URL")
        target = urlunsplit(parts._replace(scheme=mapped))
        if not self.allows_url(target):
            return {}
        hints = self._hint_headers()
        signed = sign_request(
            self.keys,
            self.agent_url,
            "GET",
            target,
            b"",
            form=self.form,
            label=self.label,
            lifetime=self.lifetime,
            created=created,
            nonces=nonces,
            allow_http=self.allow_http,
            headers=dict(hints),
            covered_headers=[name for name, _ in hints],
        )
        out: Dict[str, str] = {name: value for name, value in hints}
        for name, value in signed.headers.items():
            out[name.lower()] = value
        return out

    # -- public entry points -------------------------------------------------

    async def paying_fetch(self, request: httpx.Request) -> httpx.Response:
        """Send `request`, signed, buying platform usage on a 402 and
        re-sending the same request once (file comment). The returned
        response is the last answer: the resource on success, or the 402,
        403 or 503 the loop stopped at, so a caller can read the problem body
        itself. A payment source that raises propagates: nothing is retried
        against a source that failed. A credential whose outcome is in doubt
        is reported through `on_refusal` and `pending_credentials()`.

        A 2xx that declares no length is returned UNREAD and open (file
        comment, "A STREAMED ANSWER IS NEVER READ"): iterate or `aread()` it,
        then `aclose()` it. Every other answer is already read."""
        body = await request.aread()
        seed = _Seed(
            method=request.method,
            url=str(request.url),
            headers=httpx.Headers(request.headers),
            body=bytes(body or b""),
            extensions=dict(request.extensions),
        )
        result = await self._with_client(seed, None, False, object())
        return result.response

    async def paying_request(
        self,
        method: str,
        url: Union[str, httpx.URL],
        *,
        headers: Optional[Mapping[str, str]] = None,
        content: Optional[Union[str, bytes]] = None,
        json: Any = None,
        params: Any = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """`paying_fetch` over a request built from these arguments, the way
        `httpx.AsyncClient.request` takes them."""
        extensions = {"timeout": httpx.Timeout(timeout).as_dict()} if timeout is not None else None
        request = httpx.Request(method, url, headers=headers, content=content, json=json, params=params, extensions=extensions)
        return await self.paying_fetch(request)

    async def purchase(
        self,
        url: str,
        challenge: str,
        terms: Optional[Union[MppTermsRequest, Mapping[str, Any]]] = None,
        *,
        method: str = "POST",
        headers: Optional[Mapping[str, str]] = None,
        call: Any = None,
    ) -> MppPurchaseOutcome:
        """Pay a challenge received out of band (a UAMP `payment.required`
        with scheme `mpp`) at `url`, the purchase URL the same message named,
        by `POST` with no body. The challenge's origin admits the purchase URL
        (design section 6.1), so the credential redeems there. The purchase
        URL's host must be on the realm allowlist (S-147), a held pack
        credential is re-presented here before a new one is paid (S-150), and
        every request carries an explicit timeout (never httpx's 5 s
        default, which raised mid Tempo redeem, after the broadcast). `call`
        is any hashable kept for ONE call: `max_purchases_per_call` then
        holds across every purchase made for it."""
        parsed = read_mpp_challenge(challenge)
        if parsed is None:
            return MppPurchaseOutcome(
                ok=False, status=None, response=None, reason="no_challenge",
                detail="the challenge is not a readable Payment challenge",
            )
        if isinstance(terms, MppTermsRequest):
            version, terms_url = terms.version, terms.url
        elif isinstance(terms, Mapping):
            version, terms_url = terms.get("version"), terms.get("url")
        else:
            version, terms_url = None, None
        version = version.strip() if isinstance(version, str) else ""
        notice = MppTermsRequest(version=version, url=terms_url if isinstance(terms_url, str) else None) if version else None
        seed = self._purchase_seed(url, method, headers)
        return await self._buy(None, seed, _Pending(challenge=parsed, terms=notice), object() if call is None else call)

    async def purchase_at(
        self,
        url: str,
        *,
        source_url: str,
        method: str = "POST",
        headers: Optional[Mapping[str, str]] = None,
        call: Any = None,
    ) -> MppPurchaseOutcome:
        """Act on a PURCHASE POINTER (file comment): an `mpp` requirement
        that names `url`, the purchase URL, and carries no challenge, because
        its sender had not verified who was asking. The buyer asks the
        purchase URL itself, by a signed `POST` with no body, is challenged
        there as its own identity, and pays that challenge under the same
        policy as any other. `source_url` is the URL of whatever named the
        pointer (the resource that answered 402, or the socket; `wss:` is
        read as `https:`). NOTHING is sent unless the hosts of BOTH `url` and
        `source_url` are on `policy.realms` (S-147): a pointer carries no
        secret, so any peer can write one, and an off-list peer must not be
        able to make this agent buy or aim its signed request at a host. And
        nothing is sent unless `policy.daily_cap_cents` is set
        (`pointer_needs_daily_cap`): a peer BEHIND an allowlisted host can
        write a pointer too, and without a daily cap the total it could make
        this agent buy across calls has no bound. The TypeScript
        `purchaseAt`, whose `from` is `source_url` here."""
        key = object() if call is None else call
        refusal = self._pointer_verdict(url, source_url, key)
        if refusal is not None:
            await self._refuse(url, None, None, *refusal)
            return MppPurchaseOutcome(ok=False, status=None, response=None, reason=refusal[0], detail=refusal[1])
        return await self._buy(None, self._purchase_seed(url, method, headers), None, key)

    def _purchase_seed(self, url: str, method: str = "POST", headers: Optional[Mapping[str, str]] = None) -> _Seed:
        timeout = self.policy.purchase_timeout_seconds or DEFAULT_PURCHASE_TIMEOUT_SECONDS
        return _Seed(
            method=method,
            url=url,
            headers=httpx.Headers(headers or {}),
            body=b"",
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )

    def _calls_made(self, call: Any) -> int:
        return self._calls.get(call, 0)

    def _count_call(self, call: Any, purchases: int) -> None:
        self._calls.pop(call, None)
        self._calls[call] = purchases
        if len(self._calls) > 1024:
            del self._calls[next(iter(self._calls))]

    def _pointer_verdict(self, url: str, source_url: str, call: Any) -> Optional[Tuple[str, str]]:
        """What is checked before a pointer is followed, all of it before
        any request leaves: both hosts on the allowlist (S-147); a configured
        daily cap (file comment, "A POINTER IS FOLLOWED ONLY UNDER A DAILY
        CAP"); and room left under `max_purchases_per_call`, so a call that
        already bought is not even asked for another challenge."""
        named: Optional[str] = None
        try:
            parts = urlsplit(source_url)
            mapped = {"wss": "https", "ws": "http", "https": "https", "http": "http"}.get(parts.scheme.lower())
            if mapped is not None and parts.netloc:
                named = urlunsplit(parts._replace(scheme=mapped))
        except ValueError:
            named = None
        if not self.allows_url(url) or named is None or not self.allows_url(named):
            return (
                "realm_not_allowed",
                f"the purchase pointer to {url}, named by {source_url}, is not followed: both hosts must be on "
                "policy.realms; nothing was sent",
            )
        policy = self.policy
        # The ceiling. `max_purchases_per_call` below bounds one call, and a
        # peer behind an allowlisted host starts as many calls as it likes:
        # only the daily cap bounds the total, so without one no pointer is
        # followed.
        if policy.daily_cap_cents is None:
            return (
                "pointer_needs_daily_cap",
                f"the purchase pointer to {url} is not followed: policy.daily_cap_cents is not set. A pointer carries no "
                "secret, so any peer reachable through an allowlisted host can write one, and without a daily cap nothing "
                "bounds what such pointers make this agent buy across calls. Set policy.daily_cap_cents to follow purchase "
                "pointers; a challenge on a URL the caller chose is paid as before. Nothing was sent",
            )
        max_purchases = DEFAULT_MAX_PURCHASES_PER_CALL if policy.max_purchases_per_call is None else policy.max_purchases_per_call
        made = self._calls_made(call)
        if made >= max_purchases:
            return (
                "purchase_limit_per_call",
                f"{made} purchase{'' if made == 1 else 's'} settled for this call and policy.max_purchases_per_call is "
                f"{max_purchases}; the purchase pointer would buy another",
            )
        return None

    async def _buy(
        self, client: Optional[httpx.AsyncClient], seed: _Seed, initial: Optional[_Pending], call: Any
    ) -> MppPurchaseOutcome:
        """One purchase at a purchase URL, as an outcome: the loop with a
        challenge already in hand (`initial`), or with none, in which case the
        first signed request is what asks for it (`purchase_at`). `client` is
        the client of a loop already running (the pointer inside
        `paying_fetch`), else one is resolved as for any call. The purchase
        URL's answer is never left open: nobody is streaming it."""
        try:
            if client is not None:
                result = await self._loop(client, seed, initial, True, call)
            else:
                result = await self._with_client(seed, initial, True, call)
        except MppRedirectError as err:
            # S-180: a redirect is a refusal with no answer. `on_refusal` has
            # already fired, and the ledger is already settled (`_present`).
            return MppPurchaseOutcome(
                ok=False, status=None, response=None, reason="redirect_refused", detail=str(err),
                pending_credential=err.pending_credential,
            )
        # Read whatever is still unread within the cap, then let go of the connection either way.
        await _settle_body(result.response, self._timeout_s(), purchase_document=True)
        await _discard(result.response)
        status = result.response.status_code
        if result.refusal:
            return MppPurchaseOutcome(
                ok=False, status=status, response=result.response, reason=result.refusal, detail=result.detail,
                pending_credential=result.pending, receipt=result.receipt,
            )
        if result.presented is None and result.settled is None:
            return MppPurchaseOutcome(
                ok=False, status=status, response=result.response, reason="served_without_purchase",
                detail="the purchase URL answered without a challenge and nothing was bought",
            )
        if result.settled is None or not result.response.is_success:
            return MppPurchaseOutcome(
                ok=False, status=status, response=result.response, reason="unexpected_status", detail=result.detail,
                pending_credential=result.pending,
            )
        return MppPurchaseOutcome(
            ok=True, status=status, response=result.response,
            record=self._record_of(
                seed.url, result.settled, result.response, _read_json(result.response, purchase_document=True)
            ),
        )

    def _timeout_s(self) -> float:
        """`policy.purchase_timeout_seconds`: the bound on every body the buyer reads for itself."""
        return float(self.policy.purchase_timeout_seconds or DEFAULT_PURCHASE_TIMEOUT_SECONDS)

    # -- the loop ------------------------------------------------------------

    def _record_of(
        self, url: str, credential: _Credential, response: httpx.Response, body: Optional[Mapping[str, Any]] = None
    ) -> MppPurchaseRecord:
        return MppPurchaseRecord(
            url=url,
            challenge_id=credential.challenge_id,
            method=credential.method,
            amount_cents=credential.amount_cents,
            currency=credential.currency,
            terms_version=credential.terms_version,
            receipt=receipt_of(response, body),
            status=response.status_code,
        )

    async def _report_purchase(
        self, url: str, credential: _Credential, response: httpx.Response, body: Optional[Mapping[str, Any]]
    ) -> None:
        """`on_purchase`, at most once per challenge id however many answers say the purchase settled."""
        if credential.challenge_id in self._reported:
            return
        self._reported[credential.challenge_id] = None
        if len(self._reported) > 1024:
            del self._reported[next(iter(self._reported))]
        if self.on_purchase is not None:
            await _resolve(self.on_purchase(self._record_of(url, credential, response, body)))

    async def _with_client(self, seed: _Seed, initial: Optional[_Pending], in_band: bool, call: Any) -> _LoopResult:
        if self.client is not None:
            return await self._loop(self.client, seed, initial, in_band, call)
        timeout = self.policy.purchase_timeout_seconds or DEFAULT_PURCHASE_TIMEOUT_SECONDS
        # Not `async with`: a streamed answer handed to the caller still needs
        # this client's connection, so the client is closed when THAT is
        # closed (`_HandedStream`), or here when nothing is left open.
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        try:
            result = await self._loop(client, seed, initial, in_band, call)
        except BaseException:
            await client.aclose()
            raise
        if result.response.is_closed:
            await client.aclose()
        else:
            result.response.stream = _HandedStream(result.response.stream, owner=client)
        return result

    async def _loop(
        self, client: httpx.AsyncClient, seed: _Seed, initial: Optional[_Pending], in_band: bool, call: Any
    ) -> _LoopResult:
        """The loop of the file comment, decision for decision the TypeScript
        `loop`. `initial` is a challenge already in hand (in-band purchase);
        None means send the request first and read the challenge off its
        402. `in_band` says `seed` is a purchase URL the buyer itself is
        calling, not a resource a caller asked for. `call` is where the count
        of settled purchases lives, so it outlasts this one loop (file
        comment, "THE PURCHASE POINTER")."""
        policy = self.policy
        max_challenges = DEFAULT_MAX_CHALLENGES if policy.max_challenges is None else policy.max_challenges
        max_purchases = DEFAULT_MAX_PURCHASES_PER_CALL if policy.max_purchases_per_call is None else policy.max_purchases_per_call
        max_retry_after = (
            DEFAULT_MAX_RETRY_AFTER_SECONDS if policy.max_retry_after_seconds is None else policy.max_retry_after_seconds
        )
        hints = self._hint_headers()

        credential: Optional[_Credential] = None
        presented: Optional[_Credential] = None
        settled: Optional[_Credential] = None
        challenges_paid = 0
        purchases = self._calls_made(call)
        settlement_retries = 0
        # The settlement budget is measured by the clock AND by what was
        # slept, so a frozen test clock still reaches the deadline.
        budget_from = 0.0
        waited_ms = 0.0
        held_checked = False
        retried_without = False
        pending = initial
        response: Optional[httpx.Response] = None

        def done(
            refusal: Optional[str],
            detail: str,
            record: Optional[MppPendingCredential] = None,
            receipt: Optional[MppReceipt] = None,
        ) -> _LoopResult:
            # Only the in-band path reaches a refusal with no response yet:
            # the caller of `purchase` reads the outcome, never this body.
            return _LoopResult(response or httpx.Response(402), presented, settled, refusal, detail, record, receipt)

        def body_of(answer: httpx.Response) -> Optional[Dict[str, Any]]:
            return _read_json(answer, purchase_document=in_band)

        if pending is None:
            response = await self._send_plain(client, seed, hints, in_band)

        try:
            while True:
                if pending is not None:
                    await self._ready()
                    realm_refusal = self._realm_verdict(seed.url, pending.challenge)
                    if realm_refusal is not None:
                        await self._refuse(seed.url, pending.challenge, pending.terms, *realm_refusal)
                        return done(*realm_refusal)
                    # S-150: a held credential for this resource (or a pack
                    # one, at a purchase URL) goes first. Paying the new
                    # challenge while it is in doubt is how the second
                    # transfer used to be signed.
                    if not held_checked:
                        held_checked = True
                        held = self._held_for(seed, pending.challenge.fields.realm, in_band)
                        if held is not None:
                            self._claimed.add(held.id)
                            again = self._credential_of_held(held)
                            credential = presented = again
                            settlement_retries = 0
                            budget_from, waited_ms = _now_ms(self.now()), 0.0
                            pending = None
                            await _discard(response)
                            response = await self._present(client, seed, hints, again, 0, in_band)
                            continue
                    if purchases >= max_purchases:
                        detail = (
                            f"{purchases} purchase{'' if purchases == 1 else 's'} settled for this call and "
                            f"policy.max_purchases_per_call is {max_purchases}; the fresh challenge would buy another"
                        )
                        await self._refuse(seed.url, pending.challenge, pending.terms, "purchase_limit_per_call", detail)
                        return done("purchase_limit_per_call", detail)
                    if challenges_paid >= max_challenges:
                        detail = f"{challenges_paid} challenges were paid for one request and the policy allows no more"
                        await self._refuse(seed.url, pending.challenge, pending.terms, "challenges_exhausted", detail)
                        return done("challenges_exhausted", detail)
                    # S-154: who is paid, before anything is reserved or
                    # issued (the pin may need one fetch, so it sits before
                    # the synchronous admit).
                    seller_refusal = await self._seller_verdict(client, pending.challenge)
                    if seller_refusal is not None:
                        await self._refuse(seed.url, pending.challenge, pending.terms, "seller_not_pinned", seller_refusal)
                        return done("seller_not_pinned", seller_refusal)
                    # Checked and reserved in one synchronous step (file comment, THE LEDGER).
                    verdict = self._admit(pending.challenge)
                    if isinstance(verdict, tuple):
                        await self._refuse(seed.url, pending.challenge, pending.terms, *verdict)
                        return done(*verdict)
                    entry_id = pending.challenge.fields.id
                    await self._save()
                    try:
                        if not await self._terms_accepted(pending.terms):
                            self._release(entry_id)
                            await self._save()
                            terms = pending.terms
                            assert terms is not None
                            where = f" ({terms.url})" if terms.url else ""
                            detail = f"the policy does not accept Terms version {terms.version}{where}"
                            await self._refuse(seed.url, pending.challenge, pending.terms, "terms_refused", detail)
                            return done("terms_refused", detail)
                        fresh = await self._obtain(pending.challenge, verdict, pending.terms)
                    except BaseException:
                        # Nothing was presented: a source that failed charged nothing.
                        self._release(entry_id)
                        await self._save()
                        raise
                    challenges_paid += 1
                    credential = presented = fresh
                    settlement_retries = 0
                    budget_from, waited_ms = _now_ms(self.now()), 0.0
                    pending = None
                    await _discard(response)
                    response = await self._present(client, seed, hints, fresh, 0, in_band)
                    continue

                current = response
                assert current is not None
                sent = credential
                if sent is None:
                    if current.status_code == 402:
                        challenge = read_mpp_challenge(current.headers.get("www-authenticate"))
                        if challenge is not None:
                            pending = _Pending(challenge=challenge, terms=read_terms_notice(current, body_of(current)))
                            continue
                        # THE PURCHASE POINTER (file comment): a 402 with no
                        # challenge whose body names where to buy. Never on a
                        # purchase URL's own answer: a pointer there would be
                        # a pointer to a pointer.
                        answered = None if in_band else body_of(current)
                        entry = mpp_requirement_of(answered.get("requirements")) if answered else None
                        if entry is not None:
                            refusal = self._pointer_verdict(entry.purchase_url, seed.url, call)
                            if refusal is not None:
                                await self._refuse(seed.url, None, None, *refusal)
                                return done(*refusal)
                            # An entry that does carry a challenge is paid as `purchase` pays one.
                            in_hand: Optional[_Pending] = None
                            if entry.challenge is not None:
                                parsed = read_mpp_challenge(entry.challenge)
                                if parsed is None:
                                    detail = "the 402 names a purchase URL with a challenge that is not a readable Payment challenge"
                                    await self._refuse(seed.url, None, None, "no_challenge", detail)
                                    return done("no_challenge", detail)
                                in_hand = _Pending(challenge=parsed, terms=entry.terms)
                            bought = await self._buy(client, self._purchase_seed(entry.purchase_url), in_hand, call)
                            if not bought.ok:
                                # `on_refusal` already fired inside; the caller gets the 402 it was answered.
                                reason = None if bought.reason in ("served_without_purchase", "unexpected_status") else bought.reason
                                return done(reason, bought.detail, bought.pending_credential, bought.receipt)
                            purchases = self._calls_made(call)
                            # The purchase funded THIS identity's balance, which
                            # the door serves only to a signed request naming
                            # no payment token.
                            seed = _without_payment_token(seed)
                            await _discard(response)
                            response = await self._send_plain(client, seed, hints, in_band)
                            continue
                    return done(None, f"answered {current.status_code}")

                body = body_of(current)
                reading = after_credential(current.status_code, body)

                if reading == "retry_same":
                    sent.in_doubt = True
                    tx_hash = body.get("transactionHash") if body else None
                    if isinstance(tx_hash, str) and tx_hash:
                        sent.transaction_hash = tx_hash
                    remaining = sent.deadline_ms - max(_now_ms(self.now()), budget_from + waited_ms)
                    capped = policy.max_settlement_retries is not None and settlement_retries >= policy.max_settlement_retries
                    if remaining <= 0 or capped:
                        record = self._hold(sent, seed, "settlement_retries_exhausted")
                        self._claimed.discard(sent.challenge_id)
                        credential = None
                        await self._save()
                        why = (
                            "policy.max_settlement_retries allows no more"
                            if capped
                            else "the settlement budget (the challenge expiry plus the grace) has run out"
                        )
                        detail = (
                            f"the platform asked {settlement_retries + 1} times to re-present the same credential and {why}; "
                            "the charge may have settled, so the credential is held and re-presented before this buyer "
                            f"pays again for {record.http_method} {record.url}"
                        )
                        await self._refuse(seed.url, None, None, "settlement_retries_exhausted", detail, record)
                        return done("settlement_retries_exhausted", detail, record)
                    settlement_retries += 1
                    backoff = DEFAULT_RETRY_AFTER_SECONDS * 2 ** (settlement_retries - 1)
                    seconds = min(max(retry_after_seconds(current.headers.get("retry-after"), self.now()), backoff), max_retry_after)
                    ms = max(0.0, min(seconds * 1000.0, float(math.ceil(remaining))))
                    waited_ms += ms
                    await _discard(response)
                    response = await self._present(client, seed, hints, sent, ms / 1000.0, in_band)
                    continue

                if reading == "settled":
                    self._held.pop(sent.challenge_id, None)
                    self._claimed.discard(sent.challenge_id)
                    credential = None
                    settled = sent
                    purchases += 1
                    self._count_call(call, purchases)
                    await self._save()
                    await self._report_purchase(seed.url, sent, current, body)
                    if current.status_code == 402:
                        # S-151: the pack was granted and the door still could
                        # not fund the call. It is paid and counted; the fresh
                        # challenge is a SECOND purchase, which the per-call
                        # limit decides.
                        challenge = read_mpp_challenge(current.headers.get("www-authenticate"))
                        if challenge is not None:
                            pending = _Pending(challenge=challenge, terms=read_terms_notice(current, body))
                            continue
                    elif current.status_code in (503, 409) and same_credential_of(body) is False and not retried_without:
                        # The contract: the purchase is done (503
                        # `funding_failed`, or 409 `purchase_already_granted`
                        # after a lost 200), so the call goes again WITHOUT
                        # the credential, once, after Retry-After (a 409 names
                        # none and goes at once). Never paid again.
                        retried_without = True
                        fallback = 0 if current.status_code == 409 else DEFAULT_RETRY_AFTER_SECONDS
                        seconds = min(retry_after_seconds(current.headers.get("retry-after"), self.now(), fallback), max_retry_after)
                        if seconds > 0:
                            await (self._sleep or asyncio.sleep)(seconds)
                        await _discard(response)
                        response = await self._send_plain(client, seed, hints, in_band)
                        continue
                    return done(None, f"answered {current.status_code}")

                code = problem_code_of(body)
                if reading == "served_elsewhere":
                    # A per-call sale served, or being served, on an earlier
                    # presentation (the portal's S-149). It is paid; the door
                    # never serves it twice, so re-presenting can only answer
                    # 409 again and is not done, and nothing is paid again.
                    self._held.pop(sent.challenge_id, None)
                    self._claimed.discard(sent.challenge_id)
                    credential = None
                    settled = sent
                    await self._save()
                    await self._report_purchase(seed.url, sent, current, body)
                    detail = (
                        "the call this credential paid for is being served on an earlier presentation; it is not served "
                        "again and nothing is paid again"
                        if code == "call_in_progress"
                        else "the call this credential paid for was already served on an earlier presentation whose "
                        "response was lost; nothing is paid again"
                    )
                    return done(code, detail, None, receipt_of(current, body))

                if reading == "refunded":
                    # Refunded: nothing is owed, so the reservation is freed.
                    # A new purchase is a new call under the normal policy.
                    self._held.pop(sent.challenge_id, None)
                    self._claimed.discard(sent.challenge_id)
                    credential = None
                    self._release(sent.challenge_id)
                    await self._save()
                    return done(
                        "call_refunded",
                        "the per-call purchase this credential paid for was refunded; its reservation is freed and "
                        "nothing is paid again",
                        None,
                        receipt_of(current, body),
                    )

                if reading == "unknown":
                    record = self._hold(sent, seed, "settlement_outcome_unknown")
                    self._claimed.discard(sent.challenge_id)
                    credential = None
                    await self._save()
                    detail = (
                        f"the platform answered {current.status_code} to the request carrying the credential without "
                        "saying whether it settled; the credential is held and re-presented before this buyer pays "
                        f"again for {record.http_method} {record.url}"
                    )
                    await self._refuse(seed.url, None, None, "settlement_outcome_unknown", detail, record)
                    return done("settlement_outcome_unknown", detail, record)

                # Refused: nothing was charged by this presentation.
                self._claimed.discard(sent.challenge_id)
                credential = None
                if sent.in_doubt:
                    # It had been in doubt and the platform now refuses it: it
                    # will never complete it. Cleared, left counted (money may
                    # have moved before the doubt began), reported with the
                    # credential for the operator, and nothing new is paid in
                    # this call.
                    record = self._pending_of(sent, seed, "refused")
                    self._held.pop(sent.challenge_id, None)
                    await self._save()
                    detail = (
                        "the platform refused a credential whose outcome had been in doubt "
                        f"(answered {current.status_code}); it is no longer held and is reported here for reconciliation"
                    )
                    await self._refuse(seed.url, None, None, "pending_credential_refused", detail, record)
                    return done("pending_credential_refused", detail, record)
                self._release(sent.challenge_id)
                await self._save()
                if current.status_code == 402:
                    challenge = read_mpp_challenge(current.headers.get("www-authenticate"))
                    if challenge is None:
                        detail = "the 402 carries no readable Payment challenge"
                        await self._refuse(seed.url, None, None, "no_challenge", detail)
                        return done("no_challenge", detail)
                    pending = _Pending(challenge=challenge, terms=read_terms_notice(current, body))
                    continue
                if current.status_code in (429, 503):
                    # A velocity cap or a sale the deployment cannot make:
                    # nothing was charged, the reservation is freed, and the
                    # buyer does not retry.
                    detail = (
                        f"the platform refused the purchase ({current.status_code}{' ' + code if code else ''}); "
                        "nothing was charged and it is not retried"
                    )
                    await self._refuse(seed.url, None, None, "platform_refused", detail, None, sent)
                    return done("platform_refused", detail)
                return done(None, f"answered {current.status_code}")
        finally:
            if credential is not None:
                self._claimed.discard(credential.challenge_id)

    async def _refuse(
        self,
        url: str,
        challenge: Optional[MppChallenge],
        terms: Optional[MppTermsRequest],
        reason: str,
        detail: str,
        pending_credential: Optional[MppPendingCredential] = None,
        presented: Optional[_Credential] = None,
    ) -> None:
        if self.on_refusal is None:
            return
        p = pending_credential
        c = presented
        await _resolve(
            self.on_refusal(
                MppBuyerRefusal(
                    reason=reason,
                    url=url,
                    challenge_id=challenge.fields.id if challenge else (p.id if p else (c.challenge_id if c else None)),
                    amount_cents=challenge.amount_cents if challenge else (p.amount_cents if p else (c.amount_cents if c else None)),
                    method=challenge.fields.method if challenge else (p.payment_method if p else (c.method if c else None)),
                    terms_version=terms.version if terms else (p.terms_version if p else (c.terms_version if c else None)),
                    detail=detail,
                    pending_credential=p,
                )
            )
        )

    async def _seller_verdict(self, client: httpx.AsyncClient, challenge: MppChallenge) -> Optional[str]:
        """S-154: None when the challenge's seller is the pinned Robutler one,
        else why not. A method other than stripe or tempo is left to `_admit`,
        which refuses it as `method_unavailable`."""
        method = challenge.fields.method
        if method not in ("stripe", "tempo"):
            return None
        details = challenge.request.method_details
        if method == "stripe":
            named = details.get("networkId") if isinstance(details.get("networkId"), str) else ""
            pinned = self.policy.stripe_profile_id
            source = "policy.stripe_profile_id"
        else:
            named = challenge.request.recipient or ""
            pinned = self.policy.tempo_deposit_address
            source = "policy.tempo_deposit_address"
        if not pinned:
            pins = await self._discovered_pins(client, challenge.fields.realm, method)
            pinned = (pins.stripe_profile_id if method == "stripe" else pins.tempo_deposit_address) if pins else None
            source = "the platform discovery document"
        what = "Stripe profile" if method == "stripe" else "deposit address"
        if not pinned:
            setting = "policy.stripe_profile_id" if method == "stripe" else "policy.tempo_deposit_address"
            return (
                f"no Robutler {what} is pinned for {challenge.fields.realm}: set {setting}, the discovery document "
                "names none; nothing is paid to an unpinned seller"
            )
        same = named == pinned if method == "stripe" else named.lower() == pinned.lower()
        if same:
            return None
        label = "Stripe profile" if method == "stripe" else "recipient"
        return (
            f"the challenge names {label} {named or '(none)'} and the Robutler seller pinned by {source} is {pinned}; "
            "nothing is paid to another seller"
        )

    async def _discovered_pins(self, client: httpx.AsyncClient, realm: str, method: str) -> Optional[MppSellerPins]:
        """The pins discovery names for the allowlisted host `realm` resolves
        to. A document that names the seller for `method` is believed for
        `DISCOVERY_PIN_TTL_MS`, one that does not for five minutes; then it
        is read again, and what the NEW read says is the whole answer: a
        re-read that fails, names no seller or names two pins nothing, and
        nothing is paid. The stale pin is never a fallback."""
        r = realm.lower()
        entry = next((e for e in self._realms if e[1] == r or e[0] == r), None)
        if entry is None:
            return None
        host = entry[0]
        lock = self._discovery_locks.setdefault(host, asyncio.Lock())
        async with lock:
            cached = self._discovered.get(host)
            now = _now_ms(self.now())
            if cached is not None:
                pins, at = cached
                named = pins.stripe_profile_id if method == "stripe" else pins.tempo_deposit_address
                if now - at < (DISCOVERY_PIN_TTL_MS if named else _DISCOVERY_NEGATIVE_TTL_MS):
                    return pins
            pins = await self._fetch_pins(client, f"https://{host}{DISCOVERY_PATH}")
            self._discovered[host] = (pins, _now_ms(self.now()))
            return pins

    async def _fetch_pins(self, client: httpx.AsyncClient, url: str) -> MppSellerPins:
        none = MppSellerPins()
        if not self.allows_url(url) or not url.startswith("https://"):
            return none
        try:
            # Unsigned (`auth=None`): discovery is public, and nothing here
            # identifies the agent. httpx follows no redirect by default.
            async with client.stream(
                "GET", url, headers={"accept": "application/json"}, timeout=_DISCOVERY_TIMEOUT_S,
                auth=None, follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    return none
                chunks: List[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > _DISCOVERY_MAX_BYTES:
                        return none
                    chunks.append(chunk)
            return seller_pins_from_discovery(json.loads(b"".join(chunks).decode("utf-8")))
        except Exception:  # noqa: BLE001 - no pin is the fail-closed answer
            return none

    def _realm_verdict(self, url: str, challenge: MppChallenge) -> Optional[Tuple[str, str]]:
        """S-147 first, then the realm-is-the-host rule. Runs before any held
        credential is re-presented and before any source call."""
        hostname, host = _whatwg_host_forms(url)
        realm = challenge.fields.realm.lower()
        if not self._realm_allowed(realm) or not self.allows_url(url):
            return (
                "realm_not_allowed",
                f"the challenge realm {challenge.fields.realm} (asked at {host}) is not on policy.realms; "
                "nothing is paid to a host the operator did not name",
            )
        if realm != hostname and realm != host:
            return (
                "realm_mismatch",
                f"the challenge realm {challenge.fields.realm} is not the host asked ({host}); "
                "nothing is paid to a third party",
            )
        return None

    def _admit(self, challenge: MppChallenge) -> Union[str, Tuple[str, str]]:
        """The rest of the policy, in the order a refusal is cheapest, and on
        success the reservation, all synchronous: no await may sit between
        the daily-cap check and the reservation (file comment, THE LEDGER).
        The method name on success, else `(reason, detail)`."""
        policy = self.policy
        if challenge.fields.intent != MPP_INTENT_CHARGE:
            return ("unsupported_intent", f"intent {challenge.fields.intent} is not charge")
        if challenge.amount_cents is None:
            return ("amount_unreadable", "the challenge amount is not a whole number of cents in its method unit")
        if challenge.expires_at is not None and challenge.expires_at <= self.now():
            return ("challenge_expired", "the challenge had expired before it could be paid")
        method = challenge.fields.method
        if method not in self.method_order():
            return ("method_unavailable", f"the challenge asks for method {method} and the policy has no source for it")
        if challenge.amount_cents > policy.max_per_purchase_cents:
            return (
                "over_max_per_purchase",
                f"{challenge.amount_cents} cents is above policy.max_per_purchase_cents ({policy.max_per_purchase_cents})",
            )
        spent = self.spent_today_cents()
        if policy.daily_cap_cents is not None and spent + challenge.amount_cents > policy.daily_cap_cents:
            return (
                "over_daily_cap",
                f"{spent} cents presented in the last 24 hours plus {challenge.amount_cents} would "
                f"exceed policy.daily_cap_cents ({policy.daily_cap_cents})",
            )
        self._reserve(challenge.fields.id, challenge.amount_cents)
        return method

    async def _terms_accepted(self, terms: Optional[MppTermsRequest]) -> bool:
        """The Terms, last: the operator's callback runs only once everything cheaper has passed."""
        if terms is None:
            return True
        accept = self.policy.accept_terms
        if isinstance(accept, str):
            return accept == terms.version
        return bool(await _resolve(accept(MppTermsRequest(version=terms.version, url=terms.url))))

    async def _obtain(self, challenge: MppChallenge, method: str, terms: Optional[MppTermsRequest]) -> _Credential:
        """Obtain the payment from the source for the method and build the
        credential. A source error propagates: nothing is retried against a
        source that failed."""
        amount_cents = int(challenge.amount_cents or 0)
        details = challenge.request.method_details
        source: Optional[str] = None
        if method == "stripe":
            network_id = details.get("networkId") if isinstance(details.get("networkId"), str) else ""
            if not network_id:
                raise ValueError("the Stripe challenge names no networkId")
            raw_types = details.get("paymentMethodTypes")
            types = [t for t in raw_types if isinstance(t, str)] if isinstance(raw_types, list) else ["card"]
            spt = await _resolve(
                self.card.get_spt(  # type: ignore[union-attr]
                    SptRequest(
                        network_id=network_id,
                        amount_cents=amount_cents,
                        currency=challenge.request.currency,
                        expires_at=challenge.expires_at,
                        challenge_id=challenge.fields.id,
                        payment_method_types=types,
                    )
                )
            )
            if not isinstance(spt, str) or not _SPT_RE.fullmatch(spt):
                raise ValueError("the card source did not return a shared payment token id (spt_...)")
            payload: Dict[str, Any] = {"spt": spt}
        else:
            chain_id = _chain_id(details.get("chainId"))
            memo = details.get("memo") if isinstance(details.get("memo"), str) else ""
            recipient = challenge.request.recipient or ""
            if chain_id is None or not memo or not recipient:
                raise ValueError("the Tempo challenge lacks chainId, memo or recipient")
            wallet = self.stablecoin
            # The DID is built before the wallet signs: an address the
            # platform would refuse must not cost a signature (tempo_payer_did).
            address = getattr(wallet, "address", None)
            source = tempo_payer_did(chain_id, address) if address is not None else None
            serialized = await _resolve(
                wallet.sign_tempo_transfer(  # type: ignore[union-attr]
                    TempoTransferRequest(
                        chain_id=chain_id,
                        currency=challenge.request.currency,
                        recipient=recipient,
                        amount=challenge.request.amount,
                        memo=memo,
                        valid_before=challenge.expires_at,
                        challenge_id=challenge.fields.id,
                    )
                )
            )
            if not isinstance(serialized, str) or not TEMPO_SIGNED_TRANSACTION_RE.fullmatch(serialized):
                raise ValueError("the stablecoin source did not return a serialized Tempo transaction (0x76 followed by hex bytes)")
            payload = tempo_credential_payload(serialized)
        grace = self.policy.settlement_grace_seconds
        grace_ms = (DEFAULT_SETTLEMENT_GRACE_SECONDS if grace is None else grace) * 1000
        base = (
            _now_ms(challenge.expires_at)
            if challenge.expires_at is not None
            else _now_ms(self.now()) + DEFAULT_SETTLEMENT_WINDOW_SECONDS * 1000
        )
        return _Credential(
            header_name=challenge.credential_header,
            value=encode_mpp_credential(challenge.fields, payload, source),
            challenge_id=challenge.fields.id,
            method=method,
            amount_cents=amount_cents,
            currency=challenge.request.currency,
            terms_version=terms.version if terms else None,
            realm=challenge.fields.realm,
            kind=_opaque_kind(challenge.fields),
            expires_at=challenge.expires_at,
            deadline_ms=base + grace_ms,
        )

    async def _present(
        self,
        client: httpx.AsyncClient,
        seed: _Seed,
        hints: Sequence[Tuple[str, str]],
        credential: _Credential,
        delay_s: float,
        in_band: bool = False,
    ) -> httpx.Response:
        """Send the request carrying `credential`, after `delay_s`. Three ways
        it can fail, read differently (file comment):

        * BEFORE ANYTHING WAS SENT (the wait was cancelled, or the signer
          refused the request): a fresh credential never left this process,
          so its reservation is released, exactly as when a payment source
          fails. One already in doubt stays held.
        * A REDIRECT (S-180): never followed. Nothing is spent: a fresh
          credential's reservation is released; one already in doubt stays
          held, since a redirect says nothing about whether it settled.
          Reported as `redirect_refused`, then raised.
        * ANYTHING ELSE once the request was handed to the client (a network
          error, a timeout, a cancellation): the outcome is unknown, so the
          credential is HELD and reported before the error propagates
          (S-150): the caller's retry, or the operator, can then complete it
          instead of paying again."""
        dispatched = False
        try:
            if delay_s > 0:
                await (self._sleep or asyncio.sleep)(delay_s)
            request = self._prepare(seed, hints, credential)
            dispatched = True
            return await self._dispatch(client, seed, request, in_band)
        except BaseException as err:
            self._claimed.discard(credential.challenge_id)
            redirect = err if isinstance(err, MppRedirectError) and not err.followed else None
            if (not dispatched or redirect is not None) and not credential.in_doubt:
                self._release(credential.challenge_id)
                await self._save()
                if redirect is not None:
                    await self._refuse(
                        seed.url, None, None, "redirect_refused",
                        f"{redirect}; nothing was spent and the reservation is released", None, credential,
                    )
                raise
            record = self._hold(credential, seed, "settlement_outcome_unknown")
            await self._save()
            if isinstance(err, MppRedirectError):
                err.pending_credential = record
            if redirect is not None:
                detail = (
                    f"{redirect}; the credential was already in doubt, so it stays held and is re-presented before "
                    f"this buyer pays again for {record.http_method} {record.url}"
                )
                await self._refuse(seed.url, None, None, "redirect_refused", detail, record)
                raise
            detail = (
                f"the request carrying the credential failed ({err!r}); the credential is held and re-presented "
                f"before this buyer pays again for {record.http_method} {record.url}"
            )
            await self._refuse(seed.url, None, None, "settlement_outcome_unknown", detail, record)
            raise

    async def _send_plain(
        self, client: httpx.AsyncClient, seed: _Seed, hints: Sequence[Tuple[str, str]], in_band: bool = False
    ) -> httpx.Response:
        """A send that carries no credential: the first request, and the retry
        after a settled purchase. A redirect is reported, then raised (S-180)."""
        try:
            return await self._dispatch(client, seed, self._prepare(seed, hints, None), in_band)
        except MppRedirectError as err:
            await self._refuse(seed.url, None, None, "redirect_refused", str(err))
            raise

    def _prepare(
        self,
        seed: _Seed,
        hints: Sequence[Tuple[str, str]],
        credential: Optional[_Credential],
    ) -> httpx.Request:
        """Build and sign one send: a fresh request from the seed, the hints
        and, on a paid retry, the credential and the assent, all covered,
        freshly signed (a new nonce every time; the platform spends each nonce
        once). Nothing is sent here, so a raise means nothing left this
        process. The request is moved onto the URL that was signed
        (`apply_signed_target`), so the URL sent is the URL signed."""
        headers = httpx.Headers(seed.headers)
        for name in SIGNATURE_HEADERS:
            headers.pop(name, None)
        covered: List[str] = list(self.covered_headers)
        for name, value in hints:
            headers[name] = value
            covered.append(name)
        if credential is not None:
            headers[credential.header_name] = credential.value
            covered.append(credential.header_name)
            if credential.terms_version:
                headers[TERMS_ACCEPTED_HEADER] = credential.terms_version
                covered.append(TERMS_ACCEPTED_HEADER)
        else:
            headers.pop(TERMS_ACCEPTED_HEADER, None)
        request = httpx.Request(
            seed.method,
            seed.url,
            headers=headers,
            content=seed.body if seed.body else None,
            extensions=dict(seed.extensions) or None,
        )
        signed = sign_request(
            self.keys,
            self.agent_url,
            request.method,
            request.url,
            seed.body,
            form=self.form,
            label=self.label,
            lifetime=self.lifetime,
            allow_http=self.allow_http,
            headers=request.headers,
            covered_headers=covered,
        )
        request.headers.update(signed.headers)
        if signed.target is not None:
            apply_signed_target(request, signed.target)
        return request

    async def _dispatch(
        self, client: httpx.AsyncClient, seed: _Seed, request: httpx.Request, in_band: bool = False
    ) -> httpx.Response:
        """Hand a prepared request to the client and refuse a redirect (S-180).
        `follow_redirects=False` is passed HERE, on every send: httpx's own
        default is not to follow, but the client is the operator's, and one
        built with `follow_redirects=True` would carry the credential to the
        `Location`. `auth=None` bypasses the client's own auth: nothing may
        re-sign over the buyer's signature (file comment, "SIGNING").

        `stream=True` (finding sdk-1): the answer comes back as soon as its
        HEAD has, and `_settle_body` then reads only what the head says may be
        read. A plain `send` reads the whole body first, which is how a
        streamed completion used to reach the caller only after its last
        chunk."""
        response = await client.send(request, auth=None, follow_redirects=False, stream=True)
        if response.history:
            await _discard(response)
            raise MppRedirectError(seed.url, followed=True, response=response)
        if response.status_code in REDIRECT_STATUSES:
            await _discard(response)
            raise MppRedirectError(
                seed.url, status=response.status_code, location=response.headers.get("location"), response=response
            )
        await _settle_body(response, self._timeout_s(), purchase_document=in_band)
        return response


def _without_payment_token(seed: _Seed) -> _Seed:
    """`seed` with every way of naming a payment token removed (file comment,
    "THE PURCHASE POINTER"): the token is what ran dry, and the platform's
    door serves the balance a purchase funded only to a signed request that
    names none. Everything else, the caller's other headers and the body, is
    kept, and a URL that named no token is left byte-identical."""
    headers = httpx.Headers(seed.headers)
    for name in _PAYMENT_TOKEN_HEADERS:
        headers.pop(name, None)
    url = seed.url
    parsed = httpx.URL(url)
    if _PAYMENT_TOKEN_QUERY in parsed.params:
        url = str(parsed.copy_remove_param(_PAYMENT_TOKEN_QUERY))
    return _Seed(method=seed.method, url=url, headers=headers, body=seed.body, extensions=dict(seed.extensions))


def _whatwg_host_forms(url: str) -> Tuple[str, str]:
    """`(hostname, host)` as a WHATWG `URL` spells them, which is what the
    TypeScript buyer compares the realm with: lowercased, IDNA as punycode,
    an IPv6 literal bracketed, and `host` adding a non-default port.
    `httpx.URL` already drops a default port and keeps the ASCII host in
    `raw_host`. A URL with no host raises, as `new URL` would."""
    parsed = httpx.URL(url)
    raw = parsed.raw_host.decode("ascii").lower()
    if not raw:
        raise ValueError(f"not an absolute URL with a host: {url}")
    hostname = f"[{raw}]" if ":" in raw else raw
    return hostname, hostname if parsed.port is None else f"{hostname}:{parsed.port}"


def _chain_id(value: Any) -> Optional[int]:
    """`methodDetails.chainId` as a safe integer: a JSON number, or a string
    of ASCII digits; anything else is None."""
    if _is_safe_int(value):
        return int(value)
    if isinstance(value, float) and value.is_integer() and abs(value) <= _MAX_SAFE_INTEGER:
        return int(value)
    if isinstance(value, str) and _ASCII_DIGITS_RE.fullmatch(value.strip()):
        parsed = int(value.strip())
        return parsed if parsed <= _MAX_SAFE_INTEGER else None
    return None


__all__ = [
    "AcceptTerms",
    "BUYER_BODY_MAX_BYTES",
    "MppRequirementEntry",
    "buyer_reads_body",
    "is_json_media_type",
    "mpp_requirement_of",
    "CREDENTIAL_HEADERS_REFUSED",
    "CardPaymentSource",
    "DISCOVERY_PIN_TTL_MS",
    "MppRedirectError",
    "REDIRECT_STATUSES",
    "credential_header_refusal",
    "DEFAULT_PURCHASE_TIMEOUT_SECONDS",
    "DISCOVERY_PATH",
    "MPP_BUYER_REFUSAL_REASONS",
    "MPP_CREDENTIAL_HEADER",
    "MPP_INTENT_CHARGE",
    "MPP_METHODS",
    "MPP_PROBLEM_BASE",
    "MPP_PURCHASE_KINDS",
    "MPP_RECEIPT_HEADER",
    "MPP_TERMINAL_OUTCOMES",
    "MppBuyer",
    "MppBuyerPersistence",
    "MppBuyerPolicy",
    "MppBuyerRefusal",
    "MppChallenge",
    "MppChallengeFields",
    "MppChargeRequest",
    "MppPendingCredential",
    "MppSellerPins",
    "MppPurchaseOutcome",
    "MppPurchaseRecord",
    "MppReceipt",
    "MppTermsRequest",
    "PAYMENT_METHODS_HINT_HEADER",
    "PURCHASE_HINT_HEADER",
    "ROBUTLER_PROBLEM_BASE",
    "SptRequest",
    "StablecoinPaymentSource",
    "TEMPO_SIGNED_TRANSACTION_RE",
    "TEMPO_UNITS_PER_CENT",
    "TERMS_ACCEPTED_HEADER",
    "TERMS_VERSION_HEADER",
    "TempoTransferRequest",
    "after_credential",
    "base64url_decode",
    "base64url_encode",
    "challenge_amount_cents",
    "decode_challenge_request",
    "encode_mpp_credential",
    "jcs_canonicalize",
    "parse_payment_receipt",
    "parse_realm_entry",
    "problem_code_of",
    "receipt_of",
    "parse_www_authenticate_payment",
    "read_mpp_challenge",
    "read_terms_notice",
    "retry_after_seconds",
    "same_credential_of",
    "seller_pins_from_discovery",
    "tempo_credential_payload",
    "tempo_payer_did",
]
