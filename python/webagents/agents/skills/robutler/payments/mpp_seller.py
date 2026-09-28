"""
MPP alongside x402, on the same priced endpoints (webagents gap-closure plan
2.6, spec pack section 2; 2026-09-26), the twin of `skills/payments/mpp-seller.ts`.

The SELLER half of MPP; the BUYER (`payments_x402/mpp_buyer.py`) lends its
pure helpers (RFC 8785 JCS, base64url without padding) so the two halves
cannot disagree on bytes. What a server does:

  - THE CHALLENGE (2.1): one `WWW-Authenticate: Payment …` per method on the
    same 402 that carries x402's `PAYMENT-REQUIRED`, `Cache-Control:
    no-store`; `request` is base64url-nopad of the JCS form of the method's
    request object; `opaque` binds the resource URL.
  - STATELESS IDS: `id = base64url-nopad(HMAC-SHA256(secret, slots))` in the
    CURRENT slot order (mpp-specs #362): `realm|method|intent|request|
    expires|digest|opaque`, with the `header` slot BEFORE `opaque` when a
    challenge names one (this seller never does).
  - THE CREDENTIAL (2.2): `Authorization: Payment <base64url-nopad(JSON)>`
    (or `Payment-Authorization`), `{challenge, source?, payload}`; the echoed
    challenge must re-HMAC to its id, be ours, unexpired and single use.
  - METHODS: `stripe` (a Shared Payment Token, settled as a PaymentIntent
    with `shared_payment_granted_token`, `confirm: true`, redirects never,
    idempotency key `<challenge.id>_<spt>`; 200 only when `succeeded`), and
    `robutler` (the credits scheme's token, settled through the paywall's
    credits scheme with the challenge's own nonce). "Servers MUST NOT
    require user accounts for payment" (11.7): `robutler` is only offered
    beside an account-less method (Stripe, or a chain scheme on x402).
  - THE RECEIPT (2.3): `Payment-Receipt: base64url-nopad(JSON)` on 2xx only,
    `Cache-Control: private`.
  - PROBLEMS: RFC 9457 `application/problem+json` under
    `https://paymentauth.org/problems/`; `method-unsupported` is 400, the
    rest 402 with a fresh challenge.

No credential or receipt is ever logged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from ..payments_x402.mpp_buyer import base64url_decode, base64url_encode, jcs_canonicalize
from .x402_credits import UsedNonces
from .x402_wire import CREDITS_DECIMALS, credits_to_nanocredits

MPP_AUTHORIZATION_HEADER = "Authorization"
MPP_ALT_CREDENTIAL_HEADER = "Payment-Authorization"
MPP_RECEIPT_HEADER = "Payment-Receipt"
MPP_INTENT_CHARGE = "charge"
MPP_METHOD_STRIPE = "stripe"
MPP_METHOD_ROBUTLER = "robutler"
MPP_PROBLEM_BASE = "https://paymentauth.org/problems/"
MPP_CHALLENGE_TTL_SECONDS = 300
#: A method id is lowercase letters only (2.8).
MPP_METHOD_RE = re.compile(r"^[a-z]+$")

_CHALLENGE_KEYS = ("id", "realm", "method", "intent", "request", "expires", "digest", "opaque", "header", "description")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rfc3339(at: datetime) -> str:
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── HMAC ids, the current slot order ─────────────────────────────────────────


def challenge_hmac_input(fields: Mapping[str, Any]) -> str:
    """Seven positional slots, empty strings for absent ones; the header BEFORE opaque when present."""
    slots = [fields["realm"], fields["method"], fields["intent"], fields["request"], fields.get("expires") or "", fields.get("digest") or ""]
    if fields.get("header"):
        slots.append(fields["header"])
    slots.append(fields.get("opaque") or "")
    return "|".join(slots)


def compute_challenge_id(secret: str, fields: Mapping[str, Any]) -> str:
    """`base64url-nopad(HMAC-SHA256(secret, slots))`: the stateless challenge id."""
    return base64url_encode(hmac.new(secret.encode("utf-8"), challenge_hmac_input(fields).encode("utf-8"), hashlib.sha256).digest())


def encode_jcs_param(value: Any) -> str:
    return base64url_encode(jcs_canonicalize(value))


def decode_jcs_param(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text:
        return None
    raw = base64url_decode(text)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def format_challenge(fields: Mapping[str, Any]) -> str:
    """One challenge as a `WWW-Authenticate` value, params quoted, in the order mppx writes them."""
    params: List[Tuple[str, str]] = [("id", fields["id"]), ("realm", fields["realm"]), ("method", fields["method"]), ("intent", fields["intent"])]
    if fields.get("expires"):
        params.append(("expires", fields["expires"]))
    params.append(("request", fields["request"]))
    for key in ("digest", "opaque", "header"):
        if fields.get(key):
            params.append((key, fields[key]))
    quoted = ", ".join(f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in params)
    return f"Payment {quoted}"


def parse_www_authenticate_payment(header: Optional[str]) -> Optional[Dict[str, Any]]:
    """The `Payment` challenge out of a `WWW-Authenticate` value, or None."""
    if not isinstance(header, str):
        return None
    start = re.search(r"(?:^|,)\s*Payment\s+", header, re.IGNORECASE)
    if not start:
        return None
    out: Dict[str, str] = {}
    pattern = re.compile(r'([A-Za-z0-9_-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))\s*(?:,\s*|$)')
    pos = start.end()
    while pos < len(header):
        match = pattern.match(header, pos)
        if not match:
            break
        out[match.group(1)] = re.sub(r"\\(.)", r"\1", match.group(2)) if match.group(2) is not None else match.group(3)
        pos = match.end()
    if not all(out.get(k) for k in ("id", "realm", "method", "intent", "request")):
        return None
    return {k: out[k] for k in _CHALLENGE_KEYS if k in out}


# ── Problems ─────────────────────────────────────────────────────────────────


def problem_body(problem: str, status: int, detail: str) -> Tuple[str, str]:
    """`(body, content_type)` for an RFC 9457 problem."""
    return json.dumps({"type": f"{MPP_PROBLEM_BASE}{problem}", "title": problem.replace("-", " "), "status": status, "detail": detail}), "application/problem+json"


# ── Credentials ──────────────────────────────────────────────────────────────


def read_mpp_credential(headers: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The credential a request carries, decoded without trusting it.

    None when there is none; `{"ok": False, "detail": ...}` when malformed;
    `{"ok": True, "credential": {"challenge", "source", "payload"}}` otherwise.
    """
    auth = headers.get(MPP_AUTHORIZATION_HEADER)
    alt = headers.get(MPP_ALT_CREDENTIAL_HEADER)
    value = alt if alt else (auth if auth and re.match(r"^\s*Payment\s", auth, re.IGNORECASE) else None)
    if not value:
        return None
    match = re.match(r"^\s*Payment\s+([A-Za-z0-9_-]+)\s*$", value)
    if not match:
        return {"ok": False, "detail": 'the credential is not "Payment <base64url>"'}
    raw = base64url_decode(match.group(1))
    if not raw:
        return {"ok": False, "detail": "the credential is not base64url"}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return {"ok": False, "detail": "the credential does not decode to JSON"}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("challenge"), dict) or not isinstance(parsed.get("payload"), dict):
        return {"ok": False, "detail": "the credential lacks a challenge or a payload object"}
    challenge = parsed["challenge"]
    for key in ("id", "realm", "method", "intent", "request"):
        if not isinstance(challenge.get(key), str) or not challenge[key]:
            return {"ok": False, "detail": f"challenge.{key} is missing"}
    for key in ("expires", "digest", "opaque", "header", "description"):
        if key in challenge and challenge[key] is not None and not isinstance(challenge[key], str):
            return {"ok": False, "detail": f"challenge.{key} is not a string"}
    fields = {k: challenge[k] for k in _CHALLENGE_KEYS if challenge.get(k) is not None}
    source = parsed.get("source") if isinstance(parsed.get("source"), str) else None
    return {"ok": True, "credential": {"challenge": fields, "source": source, "payload": parsed["payload"]}}


# ── Stripe ───────────────────────────────────────────────────────────────────


class StripeSptClient(Protocol):
    async def create_payment_intent(self, params: Mapping[str, Any], idempotency_key: str) -> Mapping[str, Any]: ...


# ── The seller ───────────────────────────────────────────────────────────────


class MppSeller:
    def __init__(
        self,
        realm: str,
        secret: str,
        stripe: Optional[Mapping[str, Any]] = None,
        credits: bool = True,
        challenge_ttl_seconds: Optional[int] = None,
        now: Optional[Callable[[], datetime]] = None,
        claim_challenge: Optional[Callable[[Dict[str, Any]], Awaitable[bool]]] = None,
    ) -> None:
        """`stripe`: `{"profile_id", "client", "currency"?, "payment_method_types"?, "units_per_credit"?}`.

        `claim_challenge` (S-297, 2026-09-26) claims a `stripe` challenge id
        in a store EVERY replica shares, before the handler runs: True the
        first time, False when it was claimed before. The per-process set is
        all a single process needs; a fleet's Stripe idempotency key would
        otherwise answer a replayed credential the original PaymentIntent and
        the handler would run again. The `robutler` method's single use is the
        credits scheme's (`claim_nonce`).
        """
        if not realm:
            raise ValueError("mpp: a realm is required")
        if not secret:
            raise ValueError("mpp: a challenge secret is required")
        self.realm = realm
        self.secret = secret
        self.stripe = dict(stripe) if stripe else None
        self.credits = credits
        self.ttl = challenge_ttl_seconds or MPP_CHALLENGE_TTL_SECONDS
        self.now = now or _now
        self.claim_challenge = claim_challenge
        self._used = UsedNonces()

    def credits_offered(self, chain_offered: bool) -> bool:
        return self.credits and (self.stripe is not None or chain_offered)

    def _mint(self, method: str, request: Any, url: str) -> Dict[str, Any]:
        unsigned = {
            "realm": self.realm,
            "method": method,
            "intent": MPP_INTENT_CHARGE,
            "request": encode_jcs_param(request),
            "expires": _rfc3339(self.now() + timedelta(seconds=self.ttl)),
            "opaque": encode_jcs_param({"resource": url}),
        }
        return {"id": compute_challenge_id(self.secret, unsigned), **unsigned}

    def challenges(self, url: str, max_credits: float, chain_offered: bool, credits_nonce: Optional[str] = None, platform_url: Optional[str] = None, description: Optional[str] = None) -> List[str]:
        out: List[str] = []
        if self.stripe:
            amount = str(int(round(float(max_credits) * float(self.stripe.get("units_per_credit", 100)))))
            request: Dict[str, Any] = {"amount": amount, "currency": self.stripe.get("currency", "usd")}
            if description:
                request["description"] = description
            request["methodDetails"] = {"networkId": self.stripe["profile_id"], "paymentMethodTypes": list(self.stripe.get("payment_method_types", ["card", "link"]))}
            out.append(format_challenge(self._mint(MPP_METHOD_STRIPE, request, url)))
        if self.credits_offered(chain_offered) and credits_nonce:
            details: Dict[str, Any] = {"nonce": credits_nonce, "decimals": CREDITS_DECIMALS}
            if platform_url:
                details["platform"] = platform_url
            out.append(format_challenge(self._mint(MPP_METHOD_ROBUTLER, {"amount": credits_to_nanocredits(max_credits), "currency": "credits", "methodDetails": details}, url)))
        return out

    async def admit(self, credential: Mapping[str, Any], url: str, max_credits: float) -> Dict[str, Any]:
        """Verify a credential for `url`. `{"ok": True, ...}` with what the paywall settles, or a refusal."""
        ch = credential["challenge"]

        def refuse(status: int, problem: str, detail: str) -> Dict[str, Any]:
            return {"ok": False, "status": status, "problem": problem, "detail": detail}

        if ch.get("intent") != MPP_INTENT_CHARGE:
            return refuse(400, "method-unsupported", f"intent {ch.get('intent')} is not served")
        if not MPP_METHOD_RE.match(ch.get("method") or ""):
            return refuse(400, "method-unsupported", "method ids are lowercase letters")
        if ch["method"] not in (MPP_METHOD_STRIPE, MPP_METHOD_ROBUTLER):
            return refuse(400, "method-unsupported", f"method {ch['method']} is not served")
        if ch["method"] == MPP_METHOD_STRIPE and not self.stripe:
            return refuse(400, "method-unsupported", "stripe is not configured")
        if ch.get("realm") != self.realm:
            return refuse(402, "invalid-challenge", "the challenge is for another realm")
        unsigned = {k: v for k, v in ch.items() if k != "id"}
        if compute_challenge_id(self.secret, unsigned) != ch["id"]:
            return refuse(402, "invalid-challenge", "the challenge id does not verify")
        expires_at: Optional[datetime] = None
        if ch.get("expires"):
            try:
                expires_at = datetime.strptime(ch["expires"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                return refuse(402, "invalid-challenge", "expires is not RFC 3339")
            if self.now() > expires_at:
                return refuse(402, "payment-expired", "the challenge has expired")
        opaque = decode_jcs_param(ch.get("opaque"))
        if not opaque or opaque.get("resource") != url:
            return refuse(402, "invalid-challenge", "the challenge is for another resource")
        request = decode_jcs_param(ch.get("request"))
        if not request or not isinstance(request.get("amount"), str):
            return refuse(402, "invalid-challenge", "the request is not readable")

        if ch["method"] == MPP_METHOD_STRIPE:
            stripe = self.stripe or {}
            spt = credential["payload"].get("spt")
            if not isinstance(spt, str) or not re.match(r"^spt_[A-Za-z0-9]+$", spt):
                return refuse(402, "malformed-credential", "payload.spt is not a shared payment token")
            try:
                amount = int(request["amount"])
            except ValueError:
                return refuse(402, "invalid-challenge", "the request amount is not an integer")
            expected = int(round(float(max_credits) * float(stripe.get("units_per_credit", 100))))
            if amount < expected:
                return refuse(402, "payment-insufficient", "the challenge amount is below the price")
            expires_epoch = int((expires_at or (self.now() + timedelta(seconds=self.ttl))).timestamp())
            if not self._used.claim(ch["id"], expires_epoch, self.now()):
                return refuse(402, "invalid-challenge", "the challenge was already used")
            if self.claim_challenge is not None and not await self.claim_challenge({"challenge_id": ch["id"], "expires": expires_epoch, "resource": url}):
                return refuse(402, "invalid-challenge", "the challenge was already used")
            external_id = credential["payload"].get("externalId") if isinstance(credential["payload"].get("externalId"), str) else None

            async def settle_stripe() -> Dict[str, Any]:
                try:
                    intent = await stripe["client"].create_payment_intent(
                        {
                            "amount": amount,
                            "currency": str(request.get("currency") or stripe.get("currency", "usd")),
                            "shared_payment_granted_token": spt,
                            "confirm": True,
                            "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                            "metadata": {"challenge_id": ch["id"]},
                        },
                        idempotency_key=f"{ch['id']}_{spt}",
                    )
                except Exception as error:
                    return {"ok": False, "problem": "verification-failed", "detail": f"the payment could not be taken: {error}"}
                if intent.get("status") != "succeeded":
                    return {"ok": False, "problem": "verification-failed", "detail": f"the payment did not succeed ({intent.get('status')})"}
                receipt: Dict[str, Any] = {"status": "success", "method": MPP_METHOD_STRIPE, "timestamp": _rfc3339(self.now()), "reference": intent["id"]}
                if external_id:
                    receipt["externalId"] = external_id
                return {"ok": True, "receipt": receipt}

            return {"ok": True, "method": MPP_METHOD_STRIPE, "challenge_id": ch["id"], "settle_stripe": settle_stripe}

        token = credential["payload"].get("token")
        if not isinstance(token, str) or not token.strip():
            return refuse(402, "malformed-credential", "payload.token is missing")
        details = request.get("methodDetails") if isinstance(request.get("methodDetails"), dict) else {}
        if not isinstance(details.get("nonce"), str):
            return refuse(402, "invalid-challenge", "the challenge carries no nonce")
        try:
            if int(request["amount"]) < int(credits_to_nanocredits(max_credits)):
                return refuse(402, "payment-insufficient", "the challenge amount is below the price")
        except ValueError:
            return refuse(402, "invalid-challenge", "the request amount is not an integer")
        return {"ok": True, "method": MPP_METHOD_ROBUTLER, "challenge_id": ch["id"], "credits_token": token.strip(), "credits_nonce": details["nonce"]}

    @staticmethod
    def receipt_header(receipt: Mapping[str, Any]) -> Tuple[str, str]:
        return MPP_RECEIPT_HEADER, base64url_encode(json.dumps(dict(receipt), separators=(",", ":"), ensure_ascii=False))

    def credits_receipt(self, reference: str) -> Dict[str, Any]:
        return {"status": "success", "method": MPP_METHOD_ROBUTLER, "timestamp": _rfc3339(self.now()), "reference": reference}


__all__ = [
    "MPP_AUTHORIZATION_HEADER",
    "MPP_ALT_CREDENTIAL_HEADER",
    "MPP_RECEIPT_HEADER",
    "MPP_INTENT_CHARGE",
    "MPP_METHOD_STRIPE",
    "MPP_METHOD_ROBUTLER",
    "MPP_PROBLEM_BASE",
    "MPP_CHALLENGE_TTL_SECONDS",
    "MPP_METHOD_RE",
    "challenge_hmac_input",
    "compute_challenge_id",
    "encode_jcs_param",
    "decode_jcs_param",
    "format_challenge",
    "parse_www_authenticate_payment",
    "problem_body",
    "read_mpp_credential",
    "StripeSptClient",
    "MppSeller",
]
