"""
The Robutler credits scheme under x402 (webagents gap-closure plan 2.6,
2026-09-26), the twin of `skills/payments/x402-credits.ts`: `robutler-credits`
on `robutler:1`, spec pack section 1.9.

WHAT IT SELLS. Platform usage, priced in credits, paid from a Robutler
payment token the caller already holds. Robutler is the seller of record
(`payTo: robutler`); the serving agent is credited through the existing
settle, which is where Creator Rewards come from. Nothing here moves credits
from one user to another.

BOUND TO THE REQUEST. The 402 carries a server nonce in `extra.nonce`,
`<uuid>.<expires>.<hmac>`, the HMAC (SHA-256, a per-server secret) over the
resource URL, the amount, the uuid, the expiry and, since S-297
(2026-09-26), the request binding the paywall computes (`request_binding`
in paywall.py: the method, the path, the query string and a SHA-256 of the
body), so a retry that changes the inputs of the request the 402 answered is
`nonce_not_ours`. The paywall verifies the HMAC, the expiry and single use,
so a stateless server verifies a challenge it does not remember and a
captured token buys nothing elsewhere. The uuid is the settle's
Idempotency-Key (`settle:x402:<uuid>`).

SINGLE USE ACROSS PROCESSES (S-297). The used-nonce set is per process, so
a paid request replayed to another replica, or after a restart, used to run
the handler again and reach the platform as a REPEAT of the same settle,
which replayed instead of charging: a second run for free. A client that
has a shared store claims the nonce there BEFORE the handler runs (an async
`claim_nonce(claim)` on the client, duck-typed; the platform's in-process
client claims in its database), and a client that has none still answers
nothing to a replay: the paywall treats a settle the platform answered
`replayed` as "already used" and withholds the answer.

VERIFY, THEN SETTLE AFTER THE HANDLER: the token is verified before the
handler runs (locally when a verifier is configured, else
`POST /api/payments/verify`), the charge is made after the handler answered
below 400 by `POST /api/payments/settle` by token. A handler error settles
nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

import httpx

from .idempotency import IDEMPOTENCY_KEY_BODY_FIELD, idempotency_headers
from .settle_result import read_settle_result
from .x402_wire import (
    CREDITS_ASSET,
    CREDITS_DECIMALS,
    CREDITS_MAX_TIMEOUT_SECONDS,
    CREDITS_NETWORK,
    CREDITS_PAY_TO,
    CREDITS_SCHEME,
)


#: A payment the platform could not be reached to verify (B12, 2026-09-28):
#: the reason the verify answers, which the paywall answers 503 with
#: `PLATFORM_UNREACHABLE_MESSAGE` rather than letting `httpx.ConnectError`
#: escape as a 500 (the e2e run's paid retry against a platform that was not
#: there). The TypeScript `PLATFORM_UNREACHABLE` is the same; fixture
#: `payments/final_sdk_x402_platform.json`.
PLATFORM_UNREACHABLE = "platform_unreachable"
PLATFORM_UNREACHABLE_MESSAGE = "The payment platform could not be reached to verify this payment. Try again in a moment."


class CreditsClient(Protocol):
    async def verify_token(self, token: str, expected_audience: Any = None) -> Dict[str, Any]: ...

    async def settle_token(self, token: str, amount_credits: float, idempotency_key: str, description: Optional[str] = None, resource: Optional[str] = None) -> Dict[str, Any]: ...


class PlatformCreditsClient:
    """The platform over HTTP: `/api/payments/verify` and `/api/payments/settle`.

    `verify_locally`, when given, is tried first (the JWKS verification the
    x402 skill carries); it answers `{"isValid": True, "balance": ...}` or None.
    """

    def __init__(
        self,
        platform_url: str,
        api_key: Optional[str] = None,
        verify_locally: Optional[Callable[[str, Any], Awaitable[Optional[Dict[str, Any]]]]] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 20.0,
    ) -> None:
        self.base = platform_url.rstrip("/")
        self.api_key = api_key
        self.verify_locally = verify_locally
        self._client = client
        self.timeout = timeout

    def _auth(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _post(self, path: str, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(f"{self.base}{path}", json=body, headers={**self._auth(), **(headers or {})})
        except httpx.TransportError as error:
            # Not there (refused, timed out, no route): an answer, never the
            # exception (B12). A verify reads it as unreachable; a settle as
            # failed, whose answer the paywall withholds.
            import logging

            logging.getLogger("webagents.payments.x402").warning(f"x402 credits {path}: {self.base} could not be reached: {error}")
            return {"_status": 0, "_unreachable": True, "error": PLATFORM_UNREACHABLE}
        finally:
            if self._client is None:
                await client.aclose()
        try:
            data = response.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("_status", response.status_code)
        return data

    async def verify_token(self, token: str, expected_audience: Any = None) -> Dict[str, Any]:
        if self.verify_locally is not None:
            try:
                local = await self.verify_locally(token, expected_audience)
            except Exception:
                local = None
            if local and local.get("isValid"):
                return {"valid": True, "balance": local.get("balance")}
        body: Dict[str, Any] = {"token": token}
        if expected_audience:
            body["expectedAudience"] = expected_audience
        data = await self._post("/api/payments/verify", body)
        if data.get("_unreachable"):
            return {"valid": False, "invalidReason": PLATFORM_UNREACHABLE}
        if data.get("valid") is not True:
            return {"valid": False, "invalidReason": data.get("error") or f"verify answered {data.get('_status')}"}
        balance = data.get("balanceCredits", data.get("balanceDollars"))
        return {"valid": True, "balance": float(balance) if balance is not None else None}

    async def settle_token(self, token: str, amount_credits: float, idempotency_key: str, description: Optional[str] = None, resource: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"token": token, "amount": amount_credits, IDEMPOTENCY_KEY_BODY_FIELD: idempotency_key}
        if description:
            body["description"] = description
        if resource:
            body["resource"] = resource
        data = await self._post("/api/payments/settle", body, idempotency_headers(idempotency_key))
        data.pop("_status", None)
        data.pop("_unreachable", None)
        return read_settle_result(data, "x402 credits settle")


# ── The nonce ────────────────────────────────────────────────────────────────

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_NONCE_RE = re.compile(r"^([0-9a-f-]{36})\.(\d{1,12})\.([0-9a-f]{32})$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _epoch(at: datetime) -> int:
    return int(at.timestamp())


def nonce_message(resource: str, amount: str, id: str, expires: int, binding: str = "") -> str:
    """The HMAC input, `|`-joined, with the request binding as a fifth part when the nonce is bound (S-297)."""
    base = f"{resource}|{amount}|{id}|{expires}"
    return f"{base}|{binding}" if binding else base


def _hmac_hex(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def mint_credits_nonce(secret: str, resource: str, amount: str, ttl_seconds: Optional[int] = None, now: Optional[datetime] = None, id: Optional[str] = None, binding: str = "") -> str:
    """`<uuid>.<expires>.<hmac16>`: a nonce bound to one resource, amount and (when given) request, verifiable without state."""
    nonce_id = id or str(_uuid.uuid4())
    expires = _epoch(now or _now()) + (ttl_seconds or CREDITS_MAX_TIMEOUT_SECONDS)
    mac = _hmac_hex(secret, nonce_message(resource, amount, nonce_id, expires, binding))[:32]
    return f"{nonce_id}.{expires}.{mac}"


def verify_credits_nonce(secret: str, nonce: Any, resource: str, amount: str, now: Optional[datetime] = None, binding: str = "") -> Dict[str, Any]:
    """Verify a nonce's HMAC and expiry against THIS resource, amount and request binding. Single use is the caller's set."""
    if not isinstance(nonce, str):
        return {"ok": False, "reason": "nonce_missing"}
    match = _NONCE_RE.match(nonce)
    if not match or not _UUID_RE.match(match.group(1)):
        return {"ok": False, "reason": "nonce_malformed"}
    nonce_id, expires_text, mac = match.groups()
    expires = int(expires_text)
    expected = _hmac_hex(secret, nonce_message(resource, amount, nonce_id, expires, binding))[:32]
    if not hmac.compare_digest(mac, expected):
        return {"ok": False, "reason": "nonce_not_ours"}
    if _epoch(now or _now()) > expires:
        return {"ok": False, "reason": "nonce_expired"}
    return {"ok": True, "id": nonce_id, "expires": expires}


class UsedNonces:
    """The nonces this process has settled, bounded: expired ids are swept as it grows."""

    def __init__(self, cap: int = 20_000) -> None:
        self._used: Dict[str, int] = {}
        self._cap = cap

    def claim(self, id: str, expires: int, now: Optional[datetime] = None) -> bool:
        if id in self._used:
            return False
        if len(self._used) >= 1_000:
            cutoff = _epoch(now or _now())
            for key in [k for k, v in self._used.items() if v < cutoff]:
                del self._used[key]
        if len(self._used) >= self._cap:
            del self._used[next(iter(self._used))]
        self._used[id] = expires
        return True

    def __len__(self) -> int:
        return len(self._used)


# ── The scheme ───────────────────────────────────────────────────────────────


class CreditsScheme:
    """The credits scheme: mints the entry, verifies a payment, settles it."""

    scheme = CREDITS_SCHEME
    network = CREDITS_NETWORK

    def __init__(
        self,
        client: CreditsClient,
        nonce_secret: Optional[str] = None,
        nonce_ttl_seconds: Optional[int] = None,
        platform_url: Optional[str] = None,
        expected_audience: Any = None,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.client = client
        self.secret = nonce_secret or secrets.token_hex(32)
        self.ttl = nonce_ttl_seconds
        self.platform_url = platform_url
        self.expected_audience = expected_audience
        self.now = now or _now
        self._used = UsedNonces()

    def requirement(self, resource: str, amount_nano: str, binding: str = "") -> Dict[str, Any]:
        """The `accepts[]` entry, nonce included; `binding` ties the nonce to the request (S-297)."""
        nonce = mint_credits_nonce(self.secret, resource, amount_nano, ttl_seconds=self.ttl, now=self.now(), binding=binding)
        extra: Dict[str, Any] = {"nonce": nonce, "decimals": CREDITS_DECIMALS, "tokenType": "jwt"}
        if self.platform_url:
            extra["platform"] = self.platform_url
        return {
            "scheme": CREDITS_SCHEME,
            "network": CREDITS_NETWORK,
            "amount": amount_nano,
            "asset": CREDITS_ASSET,
            "payTo": CREDITS_PAY_TO,
            "maxTimeoutSeconds": self.ttl or CREDITS_MAX_TIMEOUT_SECONDS,
            "extra": extra,
        }

    async def verify(self, payload: Mapping[str, Any], nonce: Any, resource: str, amount_nano: str, binding: str = "") -> Dict[str, Any]:
        """Verify a credits payment: the nonce (HMAC over the resource, the
        amount and the request binding; expiry; single use here and, when the
        client has one, in its shared store) and the token. The shared claim
        comes LAST, after the token verified, so an invalid payment does not
        burn the nonce."""
        token = payload.get("token")
        if not isinstance(token, str) or not token.strip():
            return {"ok": False, "reason": "token_missing"}
        check = verify_credits_nonce(self.secret, nonce, resource, amount_nano, self.now(), binding=binding)
        if not check["ok"]:
            return check
        verified = await self.client.verify_token(token.strip(), self.expected_audience)
        if not verified.get("valid"):
            return {"ok": False, "reason": verified.get("invalidReason") or "token_invalid"}
        balance = verified.get("balance")
        amount_credits = int(amount_nano) / (10 ** CREDITS_DECIMALS)
        if isinstance(balance, (int, float)) and not isinstance(balance, bool) and balance < amount_credits:
            return {"ok": False, "reason": "insufficient_funds"}
        if not self._used.claim(check["id"], check["expires"], self.now()):
            return {"ok": False, "reason": "nonce_used"}
        claim_nonce = getattr(self.client, "claim_nonce", None)
        if claim_nonce is not None and not await claim_nonce({"nonce_id": check["id"], "expires": check["expires"], "resource": resource, "binding": binding}):
            return {"ok": False, "reason": "nonce_used"}
        return {"ok": True, "token": token.strip(), "nonce_id": check["id"], "balance": balance}

    @staticmethod
    def settle_key(nonce_id: str) -> str:
        """The Idempotency-Key of the settle for a verified payment, in the fixture's fresh shape."""
        return f"settle:x402:{nonce_id}"

    async def settle(self, verified: Mapping[str, Any], amount_nano: str, description: Optional[str] = None, resource: Optional[str] = None) -> Dict[str, Any]:
        amount_credits = int(amount_nano) / (10 ** CREDITS_DECIMALS)
        return await self.client.settle_token(verified["token"], amount_credits, self.settle_key(verified["nonce_id"]), description=description, resource=resource)


__all__ = [
    "PLATFORM_UNREACHABLE",
    "PLATFORM_UNREACHABLE_MESSAGE",
    "CreditsClient",
    "PlatformCreditsClient",
    "nonce_message",
    "mint_credits_nonce",
    "verify_credits_nonce",
    "UsedNonces",
    "CreditsScheme",
]
