"""
The paywall: standard x402 (v2 and v1) and MPP on a priced `@http` endpoint
(webagents gap-closure plan 2.6, 2026-09-26), the twin of
`skills/payments/paywall.ts`.

One object, built by the payment skill from its configuration and asked by
the server (`server/core/app.py`) when a request reaches a handler that
carries `_webagents_pricing` (a `@pricing` stacked on `@http`). In the spec's
order (spec pack section 1.5):

  1. No payment: 402 with `PAYMENT-REQUIRED` (v2, header) AND a v1 JSON
     body, one `WWW-Authenticate: Payment` per MPP method, `no-store`. The
     offers are the credits scheme and, with a chain seller configured, the
     chain scheme (`exact`, `upto` for a metered endpoint). Every entry is
     well-formed for a strict parser.
  2. A payment: read it, match it to an offer, VERIFY it (credits locally
     and against the platform, a chain scheme through the facilitator, MPP
     by the seller's rules), only then run the handler.
  3. The handler answered below 400: SETTLE once, answer with
     `PAYMENT-RESPONSE` / `X-PAYMENT-RESPONSE` (or `Payment-Receipt` for
     MPP) and `Cache-Control: private`. A metered endpoint's actual is read
     from `Settlement-Overrides`, which is stripped. 400 or above: nothing
     settled, the answer goes out as it is.
  4. The settle failed: 402 with the failed settle response and `{}`.
     `settlement_pending` is delivered, recorded, never settled twice. A
     settle the platform answered `replayed` (S-297, 2026-09-26) IS a
     failure here: an earlier request spent the payment, so this one is a
     replay and gets `{}`, never the handler's answer.

BOUND TO THE REQUEST (S-297). The credits nonce a 402 mints covers, beside
the resource and the amount, `request_binding(request)`: the method, the
path, the query string and a SHA-256 of the body. The retry must be the
request the 402 answered; one that changes the inputs is `nonce_not_ours`.
Starlette caches the body, so the handler still reads it.

WHO RECEIVES A CHAIN PAYMENT is configuration (`chain["pay_to"]`): a
self-hosted agent names its own address in its own software; the platform's
dispatcher keeps its own chain path off until counsel clears it. The credits
scheme needs no gate: Robutler is its seller of record by construction.

Requests are Starlette requests (headers, url, method); `run()` answers a
Starlette `Response`, and so does `handle()`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from starlette.responses import Response

from .mpp_seller import MppSeller, problem_body, read_mpp_credential
from .x402_credits import PLATFORM_UNREACHABLE, PLATFORM_UNREACHABLE_MESSAGE, CreditsScheme
from .x402_wire import (
    X402_HEADERS,
    build_payment_required,
    credits_to_asset_units,
    credits_to_nanocredits,
    encode_base64_json,
    is_amount_string,
    read_payment_payload,
    read_settlement_overrides,
    requirement_matches,
    settle_response_header,
    v1_network_to_caip2,
)

logger = logging.getLogger("webagents.payments.paywall")

Runner = Callable[[], Awaitable[Response]]


@dataclass
class PricedEndpoint:
    """What the paywall needs to know about the endpoint it guards."""

    path: str
    method: str
    #: The `@pricing` metadata: `credits_per_call`, `lock`, `reason`.
    pricing: Dict[str, Any]
    description: Optional[str] = None
    #: Bazaar discovery metadata: `{"input": ..., "output"?: ..., "schema"?: ...}`.
    discovery: Optional[Dict[str, Any]] = None


def endpoint_price(pricing: Mapping[str, Any]) -> Tuple[float, float, bool]:
    """`(credits, max_credits, metered)`: a `lock` above the per-call price makes the endpoint metered."""
    per_call = pricing.get("credits_per_call")
    per_call = float(per_call) if isinstance(per_call, (int, float)) and not isinstance(per_call, bool) else 0.0
    lock = pricing.get("lock")
    lock = float(lock) if isinstance(lock, (int, float)) and not isinstance(lock, bool) else None
    max_credits = lock if lock is not None and lock > per_call else per_call
    if not max_credits > 0:
        raise ValueError("x402: a priced endpoint needs credits_per_call or lock above zero")
    return (per_call if per_call > 0 else max_credits), max_credits, (lock is not None and lock > per_call)


def resource_url(request: Any) -> str:
    """The resource URL a challenge names and a nonce is bound to: the request URL without its query."""
    url = request.url
    return f"{url.scheme}://{url.netloc}{url.path}"


async def request_binding(request: Any) -> str:
    """The request binding a credits nonce covers (S-297): `METHOD|path|query|
    sha256(body)`, the query without its `?` and empty when there is none, the
    digest in lowercase hex (the empty body's for GET and HEAD). Starlette
    caches `body()`, so the handler still reads it. The fixture
    `tests/fixtures/payments/paywall_x402_mpp.json` (`credits_nonce.
    bound_vectors`) pins the format for both SDKs."""
    method = str(request.method).upper()
    body = b"" if method in ("GET", "HEAD") else await request.body()
    return f"{method}|{request.url.path}|{request.url.query}|{hashlib.sha256(body).hexdigest()}"


def _platform_unreachable() -> Response:
    """503 when the platform could not verify a payment (B12; the TypeScript
    `platformUnreachable`, fixture `payments/final_sdk_x402_platform.json`)."""
    return _json_response(
        {"error": {"code": "payment_platform_unreachable", "message": PLATFORM_UNREACHABLE_MESSAGE}},
        503,
        {"Retry-After": "5"},
    )


def _json_response(body: Any, status: int, headers: Optional[Mapping[str, str]] = None) -> Response:
    return Response(json.dumps(body), status_code=status, media_type="application/json", headers={"Cache-Control": "no-store", **(headers or {})})


@dataclass
class Paywall:
    credits: Optional[CreditsScheme] = None
    #: `{"pay_to", "network", "asset", "decimals", "units_per_credit"?, "schemes"?, "max_timeout_seconds"?, "extra"?, "facilitator"}`.
    chain: Optional[Dict[str, Any]] = None
    mpp: Optional[MppSeller] = None
    #: What the 402 says about the service: `service_name`, `description`, `mime_type`, `tags`, `icon_url`.
    resource: Dict[str, Any] = field(default_factory=dict)
    log: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def __post_init__(self) -> None:
        if self.credits is None and not self.chain:
            raise ValueError("x402: a paywall needs the credits scheme or a chain seller")

    def _log(self, event: str, fields: Dict[str, Any]) -> None:
        if self.log:
            self.log(event, fields)
        else:
            logger.info("%s %s", event, {k: v for k, v in fields.items()})

    # ── Offers ────────────────────────────────────────────────────────────

    def offers(self, endpoint: PricedEndpoint, url: str, binding: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """The offers; `binding` ties the credits nonce to the request (S-297)."""
        _, max_credits, metered = endpoint_price(endpoint.pricing)
        accepts: List[Dict[str, Any]] = []
        if self.credits is not None:
            accepts.append(self.credits.requirement(url, credits_to_nanocredits(max_credits), binding))
        if self.chain:
            chain = self.chain
            schemes = list(chain.get("schemes") or ["exact"])
            entry: Dict[str, Any] = {
                "scheme": "upto" if metered and "upto" in schemes else "exact",
                "network": chain["network"],
                "amount": credits_to_asset_units(max_credits, int(chain["decimals"]), chain.get("units_per_credit", 1)),
                "asset": chain["asset"],
                "payTo": chain["pay_to"],
                "maxTimeoutSeconds": chain.get("max_timeout_seconds", 60),
            }
            if chain.get("extra"):
                entry["extra"] = dict(chain["extra"])
            accepts.append(entry)
        resource: Dict[str, Any] = {
            "url": url,
            "description": endpoint.description or self.resource.get("description") or f"{endpoint.method} {endpoint.path}",
            "mimeType": self.resource.get("mime_type", "application/json"),
        }
        for src, dst in (("service_name", "serviceName"), ("tags", "tags"), ("icon_url", "iconUrl")):
            if self.resource.get(src):
                resource[dst] = self.resource[src]
        return accepts, resource, max_credits

    async def challenge(self, endpoint: PricedEndpoint, request: Any, error: Optional[str] = None, problem: Optional[Tuple[str, str]] = None) -> Response:
        """The 402: the v2 header, the v1 body (or problem details), the MPP challenges, `no-store`."""
        url = resource_url(request)
        accepts, resource, max_credits = self.offers(endpoint, url, await request_binding(request))
        v2, v1 = build_payment_required(resource, accepts, error=error, bazaar=endpoint.discovery)
        headers: List[Tuple[str, str]] = [("Cache-Control", "no-store"), (X402_HEADERS["required"], encode_base64_json(v2))]
        if self.mpp is not None:
            credits_entry = next((r for r in accepts if self.credits is not None and r["scheme"] == self.credits.scheme), None)
            nonce = credits_entry["extra"].get("nonce") if credits_entry else None
            for value in self.mpp.challenges(url, max_credits, chain_offered=bool(self.chain), credits_nonce=nonce, platform_url=self.credits.platform_url if self.credits else None, description=resource.get("description")):
                headers.append(("WWW-Authenticate", value))
        if problem is not None:
            body, content_type = problem_body(problem[0], 402, problem[1])
            response = Response(body, status_code=402, media_type=content_type)
        else:
            response = Response(json.dumps(v1), status_code=402, media_type="application/json")
        for name, value in headers:
            response.headers.append(name, value)
        return response

    # ── Serve ─────────────────────────────────────────────────────────────

    async def handle(self, endpoint: PricedEndpoint, request: Any, run: Runner) -> Response:
        read = read_payment_payload(request.headers)
        mpp_read = read_mpp_credential(request.headers) if self.mpp is not None else None
        if mpp_read is not None and read is not None:
            return _json_response({"error": {"code": "invalid_payment", "message": "one payment per request: an MPP credential and an x402 payment were both sent"}}, 400)
        if mpp_read is not None:
            return await self._handle_mpp(endpoint, request, mpp_read, run)
        if read is None:
            return await self.challenge(endpoint, request)
        if read["version"] == 0:
            return _json_response({"error": {"code": "invalid_payment", "message": read["error"]}}, 400)

        url = resource_url(request)
        binding = await request_binding(request)
        admitted = await self._admit(endpoint, url, read, binding)
        if "refusal" in admitted:
            self._log("x402.refused", {"path": endpoint.path, "reason": admitted["refusal"]})
            # The platform was not there to verify the payment (B12): not the
            # payer's fault, and not a crash. 503 with a sentence, nothing run.
            if admitted["refusal"] == PLATFORM_UNREACHABLE:
                return _platform_unreachable()
            return await self.challenge(endpoint, request, error=admitted["refusal"])

        response = await run()
        if response.status_code >= 400:
            self._log("x402.unsettled", {"path": endpoint.path, "status": response.status_code})
            return response

        overrides = read_settlement_overrides(response.headers)
        settled = await self._settle(admitted, endpoint, url, overrides)
        name, value = settle_response_header(admitted["version"], settled)
        if settled.get("success") is not True and settled.get("errorReason") != "settlement_pending":
            # A replayed settle is a replayed request (S-297): the answer is withheld.
            self._log("x402.replayed" if settled.get("errorReason") == "replayed" else "x402.settle_failed", {"path": endpoint.path, "reason": settled.get("errorReason")})
            return _json_response({}, 402, {name: value})
        if X402_HEADERS["settlement_overrides"] in response.headers:
            del response.headers[X402_HEADERS["settlement_overrides"]]
        response.headers[name] = value
        response.headers["Cache-Control"] = "private"
        if settled.get("errorReason") == "settlement_pending":
            self._log("x402.settlement_pending", {"path": endpoint.path, "transaction": settled.get("transaction"), "network": settled.get("network")})
        else:
            self._log("x402.settled", {"path": endpoint.path, "scheme": admitted["requirement"]["scheme"], "network": settled.get("network"), "amount": settled.get("amount")})
        return response

    # ── MPP ───────────────────────────────────────────────────────────────

    async def _handle_mpp(self, endpoint: PricedEndpoint, request: Any, mpp_read: Dict[str, Any], run: Runner) -> Response:
        mpp = self.mpp
        assert mpp is not None
        if not mpp_read.get("ok"):
            return await self.challenge(endpoint, request, problem=("malformed-credential", mpp_read["detail"]))
        url = resource_url(request)
        per_call, max_credits, metered = endpoint_price(endpoint.pricing)
        admitted = await mpp.admit(mpp_read["credential"], url, max_credits)
        if not admitted["ok"]:
            self._log("mpp.refused", {"path": endpoint.path, "problem": admitted["problem"]})
            if admitted["status"] == 400:
                body, content_type = problem_body(admitted["problem"], 400, admitted["detail"])
                return Response(body, status_code=400, media_type=content_type, headers={"Cache-Control": "no-store"})
            return await self.challenge(endpoint, request, problem=(admitted["problem"], admitted["detail"]))

        credits_verified: Optional[Dict[str, Any]] = None
        if admitted["method"] == "robutler":
            if self.credits is None:
                return await self.challenge(endpoint, request, problem=("method-unsupported", "credits are not taken here"))
            credits_verified = await self.credits.verify({"token": admitted["credits_token"]}, admitted["credits_nonce"], url, credits_to_nanocredits(max_credits), await request_binding(request))
            if not credits_verified["ok"] and credits_verified["reason"] == PLATFORM_UNREACHABLE:
                return _platform_unreachable()
            if not credits_verified["ok"]:
                reason = credits_verified["reason"]
                problem = "invalid-challenge" if reason == "nonce_used" else "payment-expired" if reason == "nonce_expired" else "payment-insufficient" if reason == "insufficient_funds" else "verification-failed"
                return await self.challenge(endpoint, request, problem=(problem, reason))

        response = await run()
        if response.status_code >= 400:
            self._log("mpp.unsettled", {"path": endpoint.path, "status": response.status_code})
            return response

        overrides = read_settlement_overrides(response.headers)
        if X402_HEADERS["settlement_overrides"] in response.headers:
            del response.headers[X402_HEADERS["settlement_overrides"]]
        response.headers["Cache-Control"] = "private"

        if admitted["method"] == "stripe":
            settled = await admitted["settle_stripe"]()
            if not settled["ok"]:
                self._log("mpp.settle_failed", {"path": endpoint.path, "method": "stripe", "problem": settled["problem"]})
                return await self.challenge(endpoint, request, problem=(settled["problem"], settled["detail"]))
            name, value = MppSeller.receipt_header(settled["receipt"])
            response.headers[name] = value
            self._log("mpp.settled", {"path": endpoint.path, "method": "stripe"})
            return response

        if credits_verified is not None and self.credits is not None:
            actual_nano = self._actual_nano(per_call, max_credits, metered, overrides)
            result = await self.credits.settle(credits_verified, actual_nano, description=f"mpp {endpoint.method} {endpoint.path}", resource=url)
            if result.get("success") and result.get("replayed"):
                # The payment was spent by an earlier request (S-297): a replay, answered nothing.
                self._log("mpp.replayed", {"path": endpoint.path, "method": "robutler"})
                return await self.challenge(endpoint, request, problem=("invalid-challenge", "the challenge was already used"))
            if not result.get("success"):
                self._log("mpp.settle_failed", {"path": endpoint.path, "method": "robutler", "reason": result.get("error")})
                return await self.challenge(endpoint, request, problem=("verification-failed", result.get("error") or "the settle failed"))
            name, value = MppSeller.receipt_header(mpp.credits_receipt(CreditsScheme.settle_key(credits_verified["nonce_id"])))
            response.headers[name] = value
            self._log("mpp.settled", {"path": endpoint.path, "method": "robutler", "amount": actual_nano})
            return response

        return await self.challenge(endpoint, request, problem=("method-unsupported", "no way to settle this method"))

    # ── Verify ────────────────────────────────────────────────────────────

    async def _admit(self, endpoint: PricedEndpoint, url: str, read: Dict[str, Any], binding: str = "") -> Dict[str, Any]:
        _, max_credits, _ = endpoint_price(endpoint.pricing)
        amount_nano = credits_to_nanocredits(max_credits)
        payload = read["payload"]

        if read["version"] == 2:
            accepted = payload["accepted"]
            if self.credits is not None and accepted.get("scheme") == self.credits.scheme and accepted.get("network") == self.credits.network:
                offered = self.credits.requirement(url, amount_nano, binding)
                echoed_nonce = (accepted.get("extra") or {}).get("nonce") if isinstance(accepted.get("extra"), dict) else None
                offered_with_nonce = {**offered, "extra": {**offered["extra"], "nonce": echoed_nonce}}
                if not requirement_matches(offered_with_nonce, accepted):
                    return {"refusal": "accepted requirement does not match an offer"}
                verified = await self.credits.verify(payload["payload"], echoed_nonce, url, amount_nano, binding)
                if not verified["ok"]:
                    return {"refusal": verified["reason"]}
                return {"version": 2, "scheme": "credits", "requirement": accepted, "payload": payload, "credits": verified}
            if self.chain and accepted.get("network") == self.chain["network"]:
                offered = next((r for r in self.offers(endpoint, url)[0] if r["network"] == self.chain["network"]), None)
                if offered is None or not requirement_matches(offered, accepted):
                    return {"refusal": "accepted requirement does not match an offer"}
                verify = await self.chain["facilitator"].verify(payload, accepted)
                if not verify.get("isValid"):
                    return {"refusal": verify.get("invalidReason") or "verification failed"}
                return {"version": 2, "scheme": "chain", "requirement": accepted, "payload": payload}
            return {"refusal": "unsupported scheme or network"}

        network = v1_network_to_caip2(payload["network"])
        if self.credits is not None and payload["scheme"] == self.credits.scheme and network == self.credits.network:
            verified = await self.credits.verify(payload["payload"], payload["payload"].get("nonce"), url, amount_nano, binding)
            if not verified["ok"]:
                return {"refusal": verified["reason"]}
            return {"version": 1, "scheme": "credits", "requirement": self.credits.requirement(url, amount_nano, binding), "payload": payload, "credits": verified}
        if self.chain and network == self.chain["network"]:
            requirement = next((r for r in self.offers(endpoint, url)[0] if r["network"] == self.chain["network"]), None)
            if requirement is None or requirement["scheme"] != payload["scheme"]:
                return {"refusal": "unsupported scheme for this network"}
            verify = await self.chain["facilitator"].verify({**payload, "network": network}, requirement)
            if not verify.get("isValid"):
                return {"refusal": verify.get("invalidReason") or "verification failed"}
            return {"version": 1, "scheme": "chain", "requirement": requirement, "payload": payload}
        return {"refusal": "unsupported scheme or network"}

    # ── Settle ────────────────────────────────────────────────────────────

    @staticmethod
    def _actual_nano(per_call: float, max_credits: float, metered: bool, overrides: Optional[Dict[str, str]]) -> str:
        actual = credits_to_nanocredits(max_credits if metered else per_call)
        if metered and overrides:
            if "credits" in overrides:
                actual = credits_to_nanocredits(overrides["credits"])
            elif "amount" in overrides:
                actual = overrides["amount"]
            if int(actual) > int(credits_to_nanocredits(max_credits)):
                actual = credits_to_nanocredits(max_credits)
        return actual

    async def _settle(self, admitted: Dict[str, Any], endpoint: PricedEndpoint, url: str, overrides: Optional[Dict[str, str]]) -> Dict[str, Any]:
        per_call, max_credits, metered = endpoint_price(endpoint.pricing)
        description = f"x402 {endpoint.method} {endpoint.path}"
        if admitted["scheme"] == "credits" and self.credits is not None:
            actual_nano = self._actual_nano(per_call, max_credits, metered, overrides)
            result = await self.credits.settle(admitted["credits"], actual_nano, description=description, resource=url)
            # A settle the platform REPLAYED charged nothing now: an earlier
            # request spent this payment, so this one is a replay and fails (S-297).
            success = bool(result.get("success")) and not result.get("replayed")
            out: Dict[str, Any] = {"success": success, "transaction": "", "network": admitted["requirement"]["network"], "amount": actual_nano}
            if not success:
                out["errorReason"] = "replayed" if result.get("replayed") else (result.get("error") or "settle_failed")
            return out
        chain = self.chain or {}
        requirement = dict(admitted["requirement"])
        if requirement.get("scheme") == "upto":
            actual = requirement["amount"]
            if overrides and "credits" in overrides:
                actual = credits_to_asset_units(overrides["credits"], int(chain["decimals"]), chain.get("units_per_credit", 1))
            elif overrides and is_amount_string(overrides.get("amount")):
                actual = overrides["amount"]
            if int(actual) > int(requirement["amount"]):
                actual = requirement["amount"]
            requirement["amount"] = actual
        payload = {**admitted["payload"], "network": chain["network"]} if admitted["version"] == 1 else admitted["payload"]
        return await chain["facilitator"].settle(payload, requirement)


# ── The servers' seam ────────────────────────────────────────────────────────


def to_http_response(result: Any) -> Response:
    """What a handler returned, as the Response the paywall settles on: a
    Response as it is, a dict or list as JSON, text as text. A streaming
    handler (an async generator) cannot be priced through this seam: the
    settle needs the handler's final status, which a stream has not decided.

    The `@pricing` wrapper answers `(result, {"pricing": {...}})` for a fixed
    or a dynamic price (it was written for tools). The tuple is unwrapped
    here, and a dynamic price (`pricing.credits`) becomes the metered
    override the paywall settles, exactly as a `Settlement-Overrides` header
    would; a fixed price says nothing the endpoint's own price did not.
    """
    override: Optional[str] = None
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) and isinstance(result[1].get("pricing"), dict):
        credits = result[1]["pricing"].get("credits")
        if isinstance(credits, (int, float)) and not isinstance(credits, bool) and credits >= 0:
            override = json.dumps({"credits": str(credits)})
        result = result[0]
    response = _plain_response(result)
    if override is not None and X402_HEADERS["settlement_overrides"] not in response.headers:
        response.headers[X402_HEADERS["settlement_overrides"]] = override
    return response


def _plain_response(result: Any) -> Response:
    if isinstance(result, Response):
        return result
    if hasattr(result, "__anext__") or hasattr(result, "__aiter__"):
        return _json_response({"error": {"code": "priced_stream_unsupported", "message": "A priced endpoint cannot stream; answer a Response."}}, 500)
    if isinstance(result, (dict, list)):
        return Response(json.dumps(result), status_code=200, media_type="application/json")
    if isinstance(result, (str, bytes)):
        return Response(result, status_code=200, media_type="text/plain")
    if result is None:
        return Response("", status_code=204)
    return Response(json.dumps(result, default=str), status_code=200, media_type="application/json")


def priced_endpoint_of(handler_config: Mapping[str, Any]) -> Optional[PricedEndpoint]:
    """A `PricedEndpoint` for a registered handler that carries `@pricing`, else None."""
    func = handler_config.get("function")
    pricing = getattr(func, "_webagents_pricing", None)
    if not isinstance(pricing, dict):
        return None
    return PricedEndpoint(
        path=str(handler_config.get("subpath") or "/"),
        method=str(handler_config.get("method") or "get").upper(),
        pricing=pricing,
        description=(getattr(func, "__doc__", None) or "").strip().splitlines()[0] if (getattr(func, "__doc__", None) or "").strip() else None,
        discovery=getattr(func, "_http_discovery", None),
    )


#: The refusal for a priced endpoint on an agent with no paywall (fixture
#: `paywall_x402_mpp.json`, `not_configured.message`; the TypeScript
#: `PAYMENT_NOT_CONFIGURED_MESSAGE`). Python has no second sentence here: its
#: `PaymentSkillX402` is the seller, where TypeScript's `PaymentX402Skill`
#: names `PaymentSkill` instead (2026-09-27).
PAYMENT_NOT_CONFIGURED_MESSAGE = "This endpoint is priced, and the agent has no payment skill to take a payment with."


def resolve_paywall(agent: Any) -> Optional[Paywall]:
    """The paywall an agent's payment skill carries, or None."""
    skills = getattr(agent, "skills", None) or {}
    for skill in (skills.values() if isinstance(skills, dict) else skills):
        paywall = getattr(skill, "paywall", None)
        if isinstance(paywall, Paywall):
            return paywall
    return None


async def serve_through_paywall(agent: Any, handler_config: Mapping[str, Any], request: Any, run: Runner) -> Response:
    """Serve a handler through the agent's paywall when it is priced; a priced
    handler on an agent with no paywall is refused (503), never served free."""
    priced = priced_endpoint_of(handler_config)
    if priced is None:
        return await run()
    paywall = resolve_paywall(agent)
    if paywall is None:
        return _json_response({"error": {"code": "payment_not_configured", "message": PAYMENT_NOT_CONFIGURED_MESSAGE}}, 503)
    return await paywall.handle(priced, request, run)


__all__ = ["PricedEndpoint", "PAYMENT_NOT_CONFIGURED_MESSAGE", "endpoint_price", "resource_url", "request_binding", "Paywall", "to_http_response", "priced_endpoint_of", "resolve_paywall", "serve_through_paywall"]
