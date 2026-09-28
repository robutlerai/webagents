"""
The x402 facilitator client (webagents gap-closure plan 2.6, 2026-09-26),
the twin of `skills/payments/x402-facilitator.ts`.

A chain scheme (`exact`, `upto`) is verified and settled by a facilitator
(spec pack section 1.6): `POST /verify` is read-only, `POST /settle` moves
the value, `GET /supported` lists the kinds it serves. The public one at
`https://x402.org/facilitator` takes no credentials and serves testnets
only; Coinbase's CDP facilitator wants a per-request JWT minted from a CDP
API key pair, bound to the method, host and path (`cdp_authorizer`). A URL,
optional fixed headers and an optional per-request authorizer are the whole
configuration; `facilitator_from_config` reads them from config or the
environment (`X402_FACILITATOR_URL`, `CDP_API_KEY_ID`, `CDP_API_KEY_SECRET`).

`FacilitatorClient` is a Protocol so a test settles against a local fake.
`settlement_pending` is an answer, not an error: it is handed back with its
`transaction` hash, and the paywall never settles the same payment twice on
the strength of it.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol
from urllib.parse import urlsplit

import httpx

X402_ORG_FACILITATOR_URL = "https://x402.org/facilitator"
CDP_FACILITATOR_URL = "https://api.cdp.coinbase.com/platform/v2/x402"

Authorizer = Callable[[str, str], Awaitable[Dict[str, str]]]


class FacilitatorClient(Protocol):
    async def verify(self, payload: Mapping[str, Any], requirement: Mapping[str, Any]) -> Dict[str, Any]: ...

    async def settle(self, payload: Mapping[str, Any], requirement: Mapping[str, Any]) -> Dict[str, Any]: ...


class HttpFacilitatorClient:
    """A facilitator over HTTP. Its own errors (a 5xx, a body that is not JSON)
    come back as a failed verify or settle naming the status, never as a raise."""

    def __init__(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        authorize: Optional[Authorizer] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.headers = dict(headers or {})
        self.authorize = authorize
        self._client = client
        self.timeout = timeout

    async def _call(self, method: str, path: str, body: Any = None) -> tuple[int, Optional[Dict[str, Any]]]:
        url = f"{self.url}{path}"
        headers: Dict[str, str] = {"Accept": "application/json", **self.headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.authorize is not None:
            headers.update(await self.authorize(method, url))
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.request(method, url, headers=headers, content=json.dumps(body).encode("utf-8") if body is not None else None)
        finally:
            if self._client is None:
                await client.aclose()
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        return response.status_code, parsed if isinstance(parsed, dict) else None

    async def verify(self, payload: Mapping[str, Any], requirement: Mapping[str, Any]) -> Dict[str, Any]:
        status, data = await self._call("POST", "/verify", {"x402Version": payload.get("x402Version"), "paymentPayload": dict(payload), "paymentRequirements": dict(requirement)})
        if data is None or not isinstance(data.get("isValid"), bool):
            return {"isValid": False, "invalidReason": f"facilitator_error:{status}"}
        return data

    async def settle(self, payload: Mapping[str, Any], requirement: Mapping[str, Any]) -> Dict[str, Any]:
        status, data = await self._call("POST", "/settle", {"x402Version": payload.get("x402Version"), "paymentPayload": dict(payload), "paymentRequirements": dict(requirement)})
        if data is None or not isinstance(data.get("success"), bool):
            return {"success": False, "errorReason": f"facilitator_error:{status}", "transaction": "", "network": requirement.get("network")}
        if not isinstance(data.get("transaction"), str):
            data["transaction"] = ""
        if not isinstance(data.get("network"), str):
            data["network"] = requirement.get("network")
        return data

    async def supported(self) -> Dict[str, Any]:
        status, data = await self._call("GET", "/supported")
        if data is None or not isinstance(data.get("kinds"), list):
            raise RuntimeError(f"x402 facilitator /supported answered {status}")
        return data


# ── CDP ──────────────────────────────────────────────────────────────────────


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def cdp_authorizer(key_id: str, key_secret: str) -> Authorizer:
    """A per-request CDP JWT: `ES256` for an EC key pair (PEM secret), `EdDSA`
    for an Ed25519 pair (base64 seed plus public key), two-minute window,
    `uris` naming exactly the call it authorizes. `cryptography` is imported
    lazily so a server that never configures CDP never loads it."""
    if not key_id or not key_secret:
        raise ValueError("x402: a CDP key id and secret are both required")

    async def authorize(method: str, url: str) -> Dict[str, str]:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

        parts = urlsplit(url)
        now = int(time.time())
        is_pem = "-----BEGIN" in key_secret
        header = {"alg": "ES256" if is_pem else "EdDSA", "kid": key_id, "typ": "JWT", "nonce": secrets.token_hex(16)}
        claims = {"sub": key_id, "iss": "cdp", "aud": ["cdp_service"], "nbf": now, "exp": now + 120, "uris": [f"{method.upper()} {parts.netloc}{parts.path}"]}
        signing_input = f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}.{_b64url(json.dumps(claims, separators=(',', ':')).encode())}"
        if is_pem:
            key = serialization.load_pem_private_key(key_secret.encode("utf-8"), password=None)
            der = key.sign(signing_input.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
            r, s = decode_dss_signature(der)
            signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        else:
            raw = base64.b64decode(key_secret)
            if len(raw) not in (32, 64):
                raise ValueError("x402: a CDP Ed25519 secret is 32 or 64 base64 bytes")
            key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
            signature = key.sign(signing_input.encode("utf-8"))
        return {"Authorization": f"Bearer {signing_input}.{_b64url(signature)}"}

    return authorize


def facilitator_from_config(config: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None) -> HttpFacilitatorClient:
    """A client from config, the environment the fallback for each part."""
    config = dict(config or {})
    env = dict(env if env is not None else os.environ)
    cdp = config.get("cdp") or {}
    key_id = cdp.get("key_id") or cdp.get("keyId") or env.get("CDP_API_KEY_ID")
    key_secret = cdp.get("key_secret") or cdp.get("keySecret") or env.get("CDP_API_KEY_SECRET")
    has_cdp = bool(key_id and key_secret)
    url = config.get("url") or env.get("X402_FACILITATOR_URL") or (CDP_FACILITATOR_URL if has_cdp else X402_ORG_FACILITATOR_URL)
    return HttpFacilitatorClient(
        url,
        headers=config.get("headers"),
        authorize=cdp_authorizer(key_id, key_secret) if has_cdp else None,
        client=config.get("client"),
        timeout=float(config.get("timeout", 30.0)),
    )


__all__ = [
    "X402_ORG_FACILITATOR_URL",
    "CDP_FACILITATOR_URL",
    "FacilitatorClient",
    "HttpFacilitatorClient",
    "cdp_authorizer",
    "facilitator_from_config",
]
