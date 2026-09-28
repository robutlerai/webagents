"""
Standard x402 on a priced endpoint (webagents gap-closure plan 2.6,
2026-09-26), the twin of the TypeScript `paywall-x402-w2pay.test.ts`: the
wire codec against the spec pack's section 1.10 vectors, the well-formedness
rule, the credits nonce, and the paywall's flow with a fake platform
(credits) and a local fake facilitator (chain). The fixture
`fixtures/payments/paywall_x402_mpp.json` is read by both suites.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from starlette.requests import Request
from starlette.responses import Response

from webagents.agents.skills.robutler.payments.paywall import Paywall, PricedEndpoint
from webagents.agents.skills.robutler.payments.x402_credits import CreditsScheme, mint_credits_nonce, verify_credits_nonce
from webagents.agents.skills.robutler.payments.x402_wire import (
    CREDITS_ASSET,
    CREDITS_DECIMALS,
    CREDITS_NETWORK,
    CREDITS_PAY_TO,
    CREDITS_SCHEME,
    V1_NETWORK_NAMES,
    X402_CORS_ALLOW_HEADERS,
    X402_CORS_EXPOSE_HEADERS,
    X402_HEADERS,
    credits_to_nanocredits,
    decode_base64_json,
    encode_base64_json,
    is_well_formed_requirement,
    requirement_matches,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "paywall_x402_mpp.json").read_text())
X = FIXTURE["x402"]
URL_ = "https://agent.example.com/agents/mini/quote"


# ── The names ────────────────────────────────────────────────────────────────


def test_the_fixture_pins_the_names():
    h = X["headers"]
    assert X402_HEADERS == {"required": h["required"], "signature": h["signature"], "response": h["response"], "v1_payment": h["v1_payment"], "v1_response": h["v1_response"], "settlement_overrides": h["settlement_overrides"]}
    assert list(X402_CORS_EXPOSE_HEADERS) == X["cors"]["expose"]
    assert list(X402_CORS_ALLOW_HEADERS) == X["cors"]["allow"]
    c = X["credits_scheme"]
    assert (CREDITS_SCHEME, CREDITS_NETWORK, CREDITS_ASSET, CREDITS_PAY_TO, CREDITS_DECIMALS) == (c["scheme"], c["network"], c["asset"], c["pay_to"], c["decimals"])
    for caip2, name in X["v1_network_names"].items():
        assert V1_NETWORK_NAMES[caip2] == name


# ── Section 1.10 vectors ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["payment_required_v2", "payment_signature_v2", "settle_response_success", "settle_response_failure"])
def test_the_vectors_decode_and_re_encode_byte_for_byte(name):
    v = X["x402_vectors"][name]
    assert decode_base64_json(v["base64"]) == v["json"]
    assert encode_base64_json(v["json"]) == v["base64"]


def test_payment_signature_is_1088_characters_and_bad_base64_is_refused():
    assert len(X["x402_vectors"]["payment_signature_v2"]["base64"]) == 1088
    for bad in X["x402_vectors"]["not_base64_standard"]:
        assert decode_base64_json(bad) is None


# ── Section 1.9 ──────────────────────────────────────────────────────────────


def test_well_formedness_and_matching():
    for ok in X["well_formed"]["accepted"]:
        assert is_well_formed_requirement(ok)
    for bad in X["well_formed"]["rejected"]:
        assert not is_well_formed_requirement(bad)
    offered = X["well_formed"]["accepted"][0]
    assert requirement_matches(offered, dict(offered))
    assert requirement_matches(offered, {**offered, "extra": {"added": 1}})
    assert not requirement_matches(offered, {**offered, "amount": "9999"})
    assert not requirement_matches(offered, {**offered, "maxTimeoutSeconds": "60"})
    with_extra = {**offered, "extra": {"name": "USDC", "version": "2"}}
    assert requirement_matches(with_extra, {**with_extra, "extra": {"name": "USDC", "version": "2", "more": True}})
    assert not requirement_matches(with_extra, {**with_extra, "extra": {"name": "USDC"}})


# ── The credits nonce ────────────────────────────────────────────────────────


def test_the_nonce_vectors_and_the_credit_amounts():
    n = X["credits_nonce"]
    for v in n["vectors"]:
        now = datetime.fromtimestamp(v["expires"] - 300, tz=timezone.utc)
        assert mint_credits_nonce(n["secret"], v["resource"], v["amount"], now=now, id=v["id"]) == v["nonce"]
    for c in n["credits_to_nanocredits"]:
        assert credits_to_nanocredits(c["credits"]) == c["nanocredits"]


def test_the_nonce_verifies_its_own_and_refuses_the_rest():
    n = X["credits_nonce"]
    a, b = n["vectors"]
    now = datetime.fromtimestamp(a["expires"] - 10, tz=timezone.utc)
    assert verify_credits_nonce(n["secret"], a["nonce"], a["resource"], a["amount"], now)["id"] == a["id"]
    assert verify_credits_nonce(n["secret"], a["nonce"], a["resource"], b["amount"], now)["reason"] == "nonce_not_ours"
    assert verify_credits_nonce(n["secret"], a["nonce"], a["resource"] + "/other", a["amount"], now)["reason"] == "nonce_not_ours"
    assert verify_credits_nonce(n["secret"], a["nonce"][:-1] + "0", a["resource"], a["amount"], now)["reason"] == "nonce_not_ours"
    late = datetime.fromtimestamp(a["expires"] + 1, tz=timezone.utc)
    assert verify_credits_nonce(n["secret"], a["nonce"], a["resource"], a["amount"], late)["reason"] == "nonce_expired"
    assert verify_credits_nonce(n["secret"], "garbage", a["resource"], a["amount"], now)["reason"] == "nonce_malformed"


# ── The paywall ──────────────────────────────────────────────────────────────


class FakeCredits:
    def __init__(self, settle_result: Optional[Dict[str, Any]] = None):
        self.settles: List[Dict[str, Any]] = []
        self.verifies: List[Dict[str, Any]] = []
        self.settle_result = settle_result

    async def verify_token(self, token, expected_audience=None):
        self.verifies.append({"token": token})
        return {"valid": True, "balance": 5.0} if token == "tok_valid" else {"valid": False, "invalidReason": "token_invalid"}

    async def settle_token(self, token, amount_credits, idempotency_key, description=None, resource=None):
        self.settles.append({"token": token, "amount_credits": amount_credits, "idempotency_key": idempotency_key, "resource": resource})
        return self.settle_result or {"success": True, "charged": str(int(round(amount_credits * 1e9)))}


class FakeFacilitator:
    def __init__(self, verify=None, settle=None):
        self.calls: List[str] = []
        self._verify = verify
        self._settle = settle
        self.seen: List[Dict[str, Any]] = []

    async def verify(self, payload, requirement):
        self.calls.append("verify")
        self.seen.append(dict(payload))
        return self._verify or {"isValid": True, "payer": "0x857b06519E91e3A54538791bDbb0E22373e36b66"}

    async def settle(self, payload, requirement):
        self.calls.append(f"settle:{requirement['amount']}")
        return self._settle or {"success": True, "transaction": "0xabc", "network": requirement["network"], "payer": "0x857b"}


CHAIN = {"pay_to": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C", "network": "eip155:84532", "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e", "decimals": 6, "extra": {"name": "USDC", "version": "2"}}


def paywall(credits: Optional[FakeCredits] = None, facilitator: Optional[FakeFacilitator] = None, schemes=None, events: Optional[List[str]] = None) -> Paywall:
    return Paywall(
        credits=CreditsScheme(credits, nonce_secret="s3cret", platform_url="https://robutler.ai") if credits else None,
        chain={**CHAIN, "schemes": schemes, "facilitator": facilitator} if facilitator else None,
        resource={"service_name": "Mini", "description": "A quote"},
        log=(lambda e, f: events.append(e)) if events is not None else None,
    )


ENDPOINT = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 0.01}, description="A quote")
METERED = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 0.001, "lock": 0.05}, description="A quote")


def request(headers: Optional[Dict[str, str]] = None, path: str = "/agents/mini/quote") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "https",
        "server": ("agent.example.com", 443),
        "path": path,
        "query_string": b"q=1",
        "headers": [(k.lower().encode(), v.encode()) for k, v in {"host": "agent.example.com", **(headers or {})}.items()],
    }
    # A receive channel with an empty body: the paywall reads the body for
    # the nonce's request binding (S-297), and a scope-only Request has none.
    return Request(scope, receive=_empty_receive)


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def challenge(p: Paywall, endpoint: PricedEndpoint = ENDPOINT):
    async def never():
        raise AssertionError("the handler must not run")

    res = await p.handle(endpoint, request(), never)
    v2 = decode_base64_json(res.headers[X402_HEADERS["required"]])
    v1 = json.loads(res.body)
    return res, v2, v1


def v2_payment(accepted: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, str]:
    return {X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": accepted, "payload": payload})}


async def ok_handler(body: Any = None, status: int = 200, headers: Optional[Dict[str, str]] = None) -> Response:
    return Response(json.dumps(body if body is not None else {"price": 42}), status_code=status, media_type="application/json", headers=headers or {})


async def test_the_402_carries_the_v2_header_the_v1_body_and_only_well_formed_entries():
    res, v2, v1 = await challenge(paywall(FakeCredits(), FakeFacilitator()))
    assert res.status_code == 402
    assert res.headers["cache-control"] == "no-store"
    assert v2["x402Version"] == 2 and v2["resource"]["url"] == URL_ and v2["resource"]["serviceName"] == "Mini"
    assert [r["scheme"] for r in v2["accepts"]] == [CREDITS_SCHEME, "exact"]
    for entry in v2["accepts"]:
        assert is_well_formed_requirement(entry)
    credits = v2["accepts"][0]
    assert credits["network"] == CREDITS_NETWORK and credits["amount"] == "10000000" and credits["asset"] == CREDITS_ASSET and credits["payTo"] == CREDITS_PAY_TO
    assert isinstance(credits["extra"]["nonce"], str)
    chain = v2["accepts"][1]
    assert chain["network"] == "eip155:84532" and chain["amount"] == "10000" and chain["payTo"] == CHAIN["pay_to"] and chain["extra"] == CHAIN["extra"]
    assert v1["x402Version"] == 1 and v1["error"]
    assert v1["accepts"][1]["network"] == "base-sepolia" and v1["accepts"][1]["maxAmountRequired"] == "10000" and v1["accepts"][1]["resource"] == URL_
    assert v1["accepts"][0]["maxAmountRequired"] == "10000000"


async def test_offers_follow_the_configuration_and_the_metering():
    _, v2, _ = await challenge(paywall(FakeCredits()))
    assert [r["scheme"] for r in v2["accepts"]] == [CREDITS_SCHEME]
    _, upto, _ = await challenge(paywall(FakeCredits(), FakeFacilitator(), schemes=["exact", "upto"]), METERED)
    assert [r["scheme"] for r in upto["accepts"]] == [CREDITS_SCHEME, "upto"]
    assert upto["accepts"][0]["amount"] == "50000000" and upto["accepts"][1]["amount"] == "50000"
    _, exact_only, _ = await challenge(paywall(FakeCredits(), FakeFacilitator()), METERED)
    assert exact_only["accepts"][1]["scheme"] == "exact"


async def test_the_bazaar_declaration_rides_in_v2_extensions_and_v1_output_schema():
    discovered = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 0.01}, discovery={"input": {"type": "http", "method": "GET"}, "output": {"type": "json", "example": {"price": 1}}})
    _, v2, v1 = await challenge(paywall(FakeCredits()), discovered)
    assert v2["extensions"]["bazaar"] == {"info": {"input": {"type": "http", "method": "GET"}, "output": {"type": "json", "example": {"price": 1}}}}
    assert v1["accepts"][0]["outputSchema"] == {"type": "json", "example": {"price": 1}}


async def test_a_credits_payment_verifies_before_the_handler_and_settles_once_after_it():
    credits = FakeCredits()
    events: List[str] = []
    p = paywall(credits, events=events)
    _, v2, _ = await challenge(p)
    entry = v2["accepts"][0]
    ran: List[str] = []

    async def handler():
        ran.append("handler")
        assert len(credits.verifies) == 1 and len(credits.settles) == 0
        return await ok_handler()

    res = await p.handle(ENDPOINT, request(v2_payment(entry, {"token": "tok_valid"})), handler)
    assert ran == ["handler"] and res.status_code == 200
    assert json.loads(res.body) == {"price": 42}
    assert res.headers["cache-control"] == "private"
    assert len(credits.settles) == 1
    settle = credits.settles[0]
    assert settle["token"] == "tok_valid" and abs(settle["amount_credits"] - 0.01) < 1e-12 and settle["resource"] == URL_
    assert settle["idempotency_key"] == f"settle:x402:{entry['extra']['nonce'].split('.')[0]}"
    assert decode_base64_json(res.headers[X402_HEADERS["response"]]) == {"success": True, "transaction": "", "network": CREDITS_NETWORK, "amount": "10000000"}
    assert "x402.settled" in events

    again = await p.handle(ENDPOINT, request(v2_payment(entry, {"token": "tok_valid"})), ok_handler)
    assert again.status_code == 402
    assert decode_base64_json(again.headers[X402_HEADERS["required"]])["error"] == "nonce_used"
    assert len(credits.settles) == 1


async def test_refusals_before_the_handler_and_nothing_settled_on_a_handler_error():
    credits = FakeCredits()
    p = paywall(credits)
    _, v2, _ = await challenge(p)
    entry = v2["accepts"][0]

    async def never():
        raise AssertionError("must not run")

    bad = await p.handle(ENDPOINT, request(v2_payment(entry, {"token": "tok_bad"})), never)
    assert bad.status_code == 402 and decode_base64_json(bad.headers[X402_HEADERS["required"]])["error"] == "token_invalid"
    assert (await p.handle(ENDPOINT, request(v2_payment({**entry, "amount": "1"}, {"token": "tok_valid"})), never)).status_code == 402
    elsewhere = await p.handle(PricedEndpoint(path="/other", method="GET", pricing={"credits_per_call": 0.01}), request(v2_payment(entry, {"token": "tok_valid"}), path="/agents/mini/other"), never)
    assert elsewhere.status_code == 402
    assert credits.settles == []

    failed = await p.handle(ENDPOINT, request(v2_payment(entry, {"token": "tok_valid"})), lambda: ok_handler({"e": 1}, 500))
    assert failed.status_code == 500 and X402_HEADERS["response"] not in failed.headers
    assert credits.settles == []


async def test_a_failed_settle_answers_402_with_the_failed_response_and_an_empty_object():
    credits = FakeCredits(settle_result={"success": False, "error": "lock refused"})
    p = paywall(credits)
    _, v2, _ = await challenge(p)
    res = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0], {"token": "tok_valid"})), ok_handler)
    assert res.status_code == 402 and res.body == b"{}"
    assert decode_base64_json(res.headers[X402_HEADERS["response"]])["errorReason"] == "lock refused"


async def test_a_metered_endpoint_settles_the_override_clamped_and_strips_the_header():
    credits = FakeCredits()
    p = paywall(credits)
    _, v2, _ = await challenge(p, METERED)
    res = await p.handle(METERED, request(v2_payment(v2["accepts"][0], {"token": "tok_valid"})), lambda: ok_handler(headers={X402_HEADERS["settlement_overrides"]: json.dumps({"credits": "0.002"})}))
    assert res.status_code == 200 and X402_HEADERS["settlement_overrides"] not in res.headers
    assert abs(credits.settles[0]["amount_credits"] - 0.002) < 1e-12
    assert decode_base64_json(res.headers[X402_HEADERS["response"]])["amount"] == "2000000"
    _, again, _ = await challenge(p, METERED)
    await p.handle(METERED, request(v2_payment(again["accepts"][0], {"token": "tok_valid"})), lambda: ok_handler(headers={X402_HEADERS["settlement_overrides"]: json.dumps({"credits": "9"})}))
    assert abs(credits.settles[1]["amount_credits"] - 0.05) < 1e-12


async def test_v1_names_the_scheme_carries_the_nonce_and_answers_x_payment_response():
    credits = FakeCredits()
    p = paywall(credits)
    _, _, v1 = await challenge(p)
    entry = v1["accepts"][0]
    headers = {X402_HEADERS["v1_payment"]: encode_base64_json({"x402Version": 1, "scheme": entry["scheme"], "network": entry["network"], "payload": {"token": "tok_valid", "nonce": entry["extra"]["nonce"]}})}
    res = await p.handle(ENDPOINT, request(headers), ok_handler)
    assert res.status_code == 200 and X402_HEADERS["response"] not in res.headers
    assert decode_base64_json(res.headers[X402_HEADERS["v1_response"]])["network"] == CREDITS_NETWORK
    assert len(credits.settles) == 1


async def test_a_malformed_payment_header_is_a_400():
    res = await paywall(FakeCredits()).handle(ENDPOINT, request({X402_HEADERS["signature"]: "not-base64!"}), ok_handler)
    assert res.status_code == 400 and json.loads(res.body)["error"]["code"] == "invalid_payment"


async def test_a_chain_payment_through_a_local_facilitator_settles_after_the_handler():
    facilitator = FakeFacilitator()
    p = paywall(FakeCredits(), facilitator)
    _, v2, _ = await challenge(p)
    entry = next(r for r in v2["accepts"] if r["scheme"] == "exact")
    payload = X["x402_vectors"]["payment_signature_v2"]["json"]["payload"]

    async def handler():
        assert facilitator.calls == ["verify"]
        return await ok_handler()

    res = await p.handle(ENDPOINT, request(v2_payment(entry, payload)), handler)
    assert res.status_code == 200 and facilitator.calls == ["verify", "settle:10000"]
    assert decode_base64_json(res.headers[X402_HEADERS["response"]]) == {"success": True, "transaction": "0xabc", "network": "eip155:84532", "payer": "0x857b"}


async def test_chain_refusals_pending_and_upto():
    facilitator = FakeFacilitator()
    p = paywall(facilitator=facilitator)
    _, v2, _ = await challenge(p)
    failed = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0], {"signature": "0x"})), lambda: ok_handler({}, 503))
    assert failed.status_code == 503 and facilitator.calls == ["verify"]

    refusing = FakeFacilitator(verify={"isValid": False, "invalidReason": "insufficient_funds"})
    q = paywall(facilitator=refusing)
    _, offers, _ = await challenge(q)

    async def never():
        raise AssertionError("must not run")

    res = await q.handle(ENDPOINT, request(v2_payment(offers["accepts"][0], {"signature": "0x"})), never)
    assert res.status_code == 402 and decode_base64_json(res.headers[X402_HEADERS["required"]])["error"] == "insufficient_funds"

    pending = FakeFacilitator(settle={"success": False, "errorReason": "settlement_pending", "transaction": "0xpending", "network": "eip155:84532"})
    events: List[str] = []
    r = paywall(facilitator=pending, events=events)
    _, offers, _ = await challenge(r)
    res = await r.handle(ENDPOINT, request(v2_payment(offers["accepts"][0], {"signature": "0x"})), ok_handler)
    assert res.status_code == 200
    assert decode_base64_json(res.headers[X402_HEADERS["response"]])["transaction"] == "0xpending"
    assert [c for c in pending.calls if c.startswith("settle")] == ["settle:10000"] and "x402.settlement_pending" in events

    metered = FakeFacilitator()
    s = paywall(facilitator=metered, schemes=["exact", "upto"])
    _, offers, _ = await challenge(s, METERED)
    assert offers["accepts"][0]["scheme"] == "upto"
    await s.handle(METERED, request(v2_payment(offers["accepts"][0], {"signature": "0x"})), lambda: ok_handler(headers={X402_HEADERS["settlement_overrides"]: json.dumps({"amount": "500"})}))
    assert metered.calls == ["verify", "settle:500"]
    _, offers, _ = await challenge(s, METERED)
    await s.handle(METERED, request(v2_payment(offers["accepts"][0], {"signature": "0x"})), lambda: ok_handler(headers={X402_HEADERS["settlement_overrides"]: json.dumps({"amount": "999999"})}))
    assert metered.calls[3] == "settle:50000"


async def test_v1_chain_payload_is_matched_by_the_v1_network_name():
    facilitator = FakeFacilitator()
    p = paywall(facilitator=facilitator)
    _, _, v1 = await challenge(p)
    assert v1["accepts"][0]["network"] == "base-sepolia"
    res = await p.handle(ENDPOINT, request({X402_HEADERS["v1_payment"]: encode_base64_json({"x402Version": 1, "scheme": "exact", "network": "base-sepolia", "payload": {"signature": "0x"}})}), ok_handler)
    assert res.status_code == 200 and facilitator.seen[0]["network"] == "eip155:84532"
    assert decode_base64_json(res.headers[X402_HEADERS["v1_response"]])["success"] is True


def test_settle_key_is_the_fixtures_fresh_x402_shape():
    assert CreditsScheme.settle_key("3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c") == "settle:x402:3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c"
