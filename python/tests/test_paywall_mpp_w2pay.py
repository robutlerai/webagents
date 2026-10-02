"""
MPP alongside x402 on a priced endpoint (webagents gap-closure plan 2.6,
spec pack section 2; 2026-09-26), the twin of the TypeScript
`paywall-mpp-w2pay.test.ts`: the stateless ids against the section 2.7
vectors (current slot order), the challenge on the same 402 as the x402
offers, binding, expiry and single use, the `stripe` method against a Stripe
double, the `robutler` method through the credits scheme, and the account
rule (11.7).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import Response

from webagents.agents.skills.robutler.payments.mpp_seller import (
    MPP_INTENT_CHARGE,
    MPP_METHOD_RE,
    MPP_METHOD_ROBUTLER,
    MPP_METHOD_STRIPE,
    MPP_PROBLEM_BASE,
    MPP_RECEIPT_HEADER,
    MppSeller,
    challenge_hmac_input,
    compute_challenge_id,
    format_challenge,
    parse_www_authenticate_payment,
    read_mpp_credential,
)
from webagents.agents.skills.robutler.payments.paywall import Paywall, PricedEndpoint
from webagents.agents.skills.robutler.payments.x402_credits import CreditsScheme
from webagents.agents.skills.robutler.payments.x402_wire import X402_HEADERS, decode_base64_json, encode_base64_json
from webagents.agents.skills.robutler.payments_x402.mpp_buyer import base64url_decode, base64url_encode

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "paywall_x402_mpp.json").read_text())
M = FIXTURE["mpp"]
URL_ = "https://agent.example.com/agents/mini/quote"
NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)


def test_the_fixture_pins_the_names():
    assert MPP_RECEIPT_HEADER == M["headers"]["receipt"]
    assert MPP_METHOD_STRIPE == M["methods"]["stripe"] and MPP_METHOD_ROBUTLER == M["methods"]["credits"]
    assert MPP_INTENT_CHARGE == M["intent"] and MPP_PROBLEM_BASE == M["problem_base"]
    assert MPP_METHOD_RE.pattern == M["method_id_pattern"]
    for bad in M["invalid_method_ids"]:
        assert not MPP_METHOD_RE.match(bad)


def test_section_2_7_ids_in_the_current_slot_order():
    v = M["id_vectors"]
    base = {"realm": v["realm"], "method": v["method"], "intent": v["intent"], "request": v["request"]}
    assert base64url_encode(json.dumps(v["request_json"], separators=(",", ":"))) == v["request"]
    assert base64url_encode(json.dumps(v["opaque_json"], separators=(",", ":"))) == v["opaque"]
    assert challenge_hmac_input(base) == v["hmac_input_plain"]
    assert challenge_hmac_input({**base, "header": "Payment-Authorization"}) == v["hmac_input_with_header"]
    assert challenge_hmac_input({**base, "header": "Payment-Authorization", "opaque": v["opaque"]}) == v["hmac_input_with_header_and_opaque"]
    assert compute_challenge_id(v["secret"], base) == v["plain"]
    assert compute_challenge_id(v["secret"], {**base, "header": "Payment-Authorization"}) == v["with_header"]
    both = compute_challenge_id(v["secret"], {**base, "header": "Payment-Authorization", "opaque": v["opaque"]})
    assert both == v["with_header_and_opaque"] and both != v["ietf_text_order_with_header_and_opaque_NOT_ours"]


def test_a_formatted_challenge_round_trips_and_the_stripe_receipt_matches_the_fixture():
    v = M["id_vectors"]
    fields = {"id": v["plain"], "realm": v["realm"], "method": v["method"], "intent": v["intent"], "request": v["request"], "expires": "2026-09-26T12:00:00Z", "opaque": v["opaque"]}
    assert parse_www_authenticate_payment(format_challenge(fields)) == fields
    s = M["stripe_e2e"]
    assert base64url_encode(json.dumps(s["receipt_json"], separators=(",", ":"))) == s["receipt_base64url"]
    assert json.loads(base64url_decode(s["receipt_base64url"]).decode()) == s["receipt_json"]


# ── The paywall with MPP ─────────────────────────────────────────────────────


class StripeDouble:
    def __init__(self, status: str = "succeeded"):
        self.calls: List[Dict[str, Any]] = []
        self.status = status

    async def create_payment_intent(self, params, idempotency_key):
        self.calls.append({"params": dict(params), "idempotency_key": idempotency_key})
        return {"id": "pi_1N4Zv32eZvKYlo2CPhVPkJlW", "status": self.status}


class CreditsDouble:
    def __init__(self):
        self.settles: List[Dict[str, Any]] = []

    async def verify_token(self, token, expected_audience=None):
        return {"valid": True, "balance": 90.0} if token == "tok_valid" else {"valid": False, "invalidReason": "token_invalid"}

    async def settle_token(self, token, amount_credits, idempotency_key, description=None, resource=None):
        self.settles.append({"token": token, "amount_credits": amount_credits, "idempotency_key": idempotency_key})
        return {"success": True, "charged": "1"}


class Facilitator:
    async def verify(self, payload, requirement):
        return {"isValid": True}

    async def settle(self, payload, requirement):
        return {"success": True, "transaction": "0x", "network": requirement["network"]}


#: Priced at 50 credits: 5000 cents on the Stripe side, the spec pack's own e2e amount.
ENDPOINT = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 50}, description="AI generation")


class Clock:
    def __init__(self, at: datetime = NOW):
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def build(stripe: Optional[StripeDouble] = None, credits: Optional[CreditsDouble] = None, chain: bool = False, clock: Optional[Clock] = None, mpp_credits: bool = True) -> Paywall:
    clock = clock or Clock()
    mpp = MppSeller(realm="agent.example.com", secret="mpp-secret", stripe={"profile_id": "profile_test_123", "client": stripe} if stripe else None, credits=mpp_credits, now=clock)
    return Paywall(
        credits=CreditsScheme(credits, nonce_secret="s", platform_url="https://robutler.ai", now=clock) if credits else None,
        chain={"pay_to": "0x1", "network": "eip155:84532", "asset": "0x2", "decimals": 6, "facilitator": Facilitator()} if chain else None,
        mpp=mpp,
    )


def request(headers: Optional[Dict[str, str]] = None) -> Request:
    scope = {"type": "http", "method": "GET", "scheme": "https", "server": ("agent.example.com", 443), "path": "/agents/mini/quote", "query_string": b"x=1", "headers": [(k.lower().encode(), v.encode()) for k, v in {"host": "agent.example.com", **(headers or {})}.items()]}
    # A receive channel with an empty body: the paywall reads the body for
    # the nonce's request binding (S-297), and a scope-only Request has none.
    return Request(scope, receive=_empty_receive)


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def never():
    raise AssertionError("the handler must not run")


async def ok():
    return Response("image", status_code=200)


async def challenges(p: Paywall):
    res = await p.handle(ENDPOINT, request(), never)
    by_method: Dict[str, Dict[str, Any]] = {}
    for value in res.headers.getlist("www-authenticate"):
        parsed = parse_www_authenticate_payment(value)
        if parsed:
            by_method[parsed["method"]] = parsed
    return res, by_method


def credential(challenge: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, str]:
    return {"Authorization": f"Payment {base64url_encode(json.dumps({'challenge': challenge, 'payload': payload}, separators=(',', ':')))}"}


def receipt_of(res: Response) -> Dict[str, Any]:
    return json.loads(base64url_decode(res.headers[MPP_RECEIPT_HEADER]).decode())


def problem_of(res: Response) -> str:
    return json.loads(res.body)["type"]


async def test_the_402_carries_a_challenge_per_method_beside_the_x402_offers():
    res, by_method = await challenges(build(StripeDouble(), CreditsDouble()))
    assert res.status_code == 402 and res.headers["cache-control"] == "no-store" and res.headers.get(X402_HEADERS["required"])
    assert sorted(by_method) == ["robutler", "stripe"]
    stripe = by_method["stripe"]
    assert stripe["realm"] == "agent.example.com" and stripe["intent"] == "charge" and stripe["expires"] == "2026-09-26T12:05:00Z"
    assert json.loads(base64url_decode(stripe["request"]).decode()) == {"amount": "5000", "currency": "usd", "description": "AI generation", "methodDetails": {"networkId": "profile_test_123", "paymentMethodTypes": ["card", "link"]}}
    assert json.loads(base64url_decode(stripe["opaque"]).decode()) == {"resource": URL_}
    v2 = decode_base64_json(res.headers[X402_HEADERS["required"]])
    robutler_request = json.loads(base64url_decode(by_method["robutler"]["request"]).decode())
    assert robutler_request["methodDetails"]["nonce"] == v2["accepts"][0]["extra"]["nonce"] and robutler_request["currency"] == "credits"


async def test_the_account_backed_method_is_never_offered_alone():
    assert (await challenges(build(credits=CreditsDouble())))[1] == {}
    assert list((await challenges(build(credits=CreditsDouble(), chain=True)))[1]) == ["robutler"]


async def test_stripe_runs_the_handler_then_creates_the_intent_under_the_challenge_key_and_answers_a_receipt():
    stripe = StripeDouble()
    p = build(stripe, CreditsDouble())
    _, by_method = await challenges(p)
    challenge = by_method["stripe"]

    async def handler():
        assert stripe.calls == []
        return await ok()

    res = await p.handle(ENDPOINT, request(credential(challenge, {"spt": "spt_1N4Zv32eZvKYlo2CPhVPkJlW"})), handler)
    assert res.status_code == 200 and res.headers["cache-control"] == "private"
    assert stripe.calls == [{
        "params": {"amount": 5000, "currency": "usd", "shared_payment_granted_token": "spt_1N4Zv32eZvKYlo2CPhVPkJlW", "confirm": True, "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"}, "metadata": {"challenge_id": challenge["id"]}},
        "idempotency_key": f"{challenge['id']}_spt_1N4Zv32eZvKYlo2CPhVPkJlW",
    }]
    assert receipt_of(res) == {"status": "success", "method": "stripe", "timestamp": "2026-09-26T12:00:00Z", "reference": "pi_1N4Zv32eZvKYlo2CPhVPkJlW"}


async def test_single_use_tampering_realm_and_expiry():
    stripe = StripeDouble()
    clock = Clock()
    p = build(stripe, CreditsDouble(), clock=clock)
    _, by_method = await challenges(p)
    challenge = by_method["stripe"]
    headers = credential(challenge, {"spt": "spt_1"})
    assert (await p.handle(ENDPOINT, request(headers), ok)).status_code == 200
    replay = await p.handle(ENDPOINT, request(headers), ok)
    assert replay.status_code == 402 and replay.headers["content-type"].startswith("application/problem+json")
    assert problem_of(replay) == f"{MPP_PROBLEM_BASE}invalid-challenge" and "Payment " in replay.headers["www-authenticate"]
    assert len(stripe.calls) == 1
    tampered = await p.handle(ENDPOINT, request(credential({**challenge, "id": "AAAA"}, {"spt": "spt_2"})), ok)
    assert problem_of(tampered) == f"{MPP_PROBLEM_BASE}invalid-challenge"
    elsewhere = await p.handle(ENDPOINT, request(credential({**challenge, "realm": "other.example"}, {"spt": "spt_2"})), ok)
    assert problem_of(elsewhere) == f"{MPP_PROBLEM_BASE}invalid-challenge"
    _, fresh = await challenges(p)
    clock.at = datetime(2026, 9, 26, 12, 6, 0, tzinfo=timezone.utc)
    expired = await p.handle(ENDPOINT, request(credential(fresh["stripe"], {"spt": "spt_3"})), ok)
    assert problem_of(expired) == f"{MPP_PROBLEM_BASE}payment-expired" and len(stripe.calls) == 1


async def test_an_intent_that_did_not_succeed_answers_verification_failed_without_a_receipt():
    p = build(StripeDouble("requires_action"), CreditsDouble())
    _, by_method = await challenges(p)
    res = await p.handle(ENDPOINT, request(credential(by_method["stripe"], {"spt": "spt_9"})), ok)
    assert res.status_code == 402 and problem_of(res) == f"{MPP_PROBLEM_BASE}verification-failed" and MPP_RECEIPT_HEADER not in res.headers


async def test_malformed_unknown_method_two_payments_and_a_bearer():
    p = build(StripeDouble(), CreditsDouble())
    malformed = await p.handle(ENDPOINT, request({"Authorization": "Payment not-base64!"}), ok)
    assert malformed.status_code == 402 and problem_of(malformed) == f"{MPP_PROBLEM_BASE}malformed-credential"
    _, by_method = await challenges(p)
    unknown = await p.handle(ENDPOINT, request(credential({**by_method["stripe"], "method": "tempo"}, {"type": "hash", "hash": "0x"})), ok)
    assert unknown.status_code == 400 and problem_of(unknown) == f"{MPP_PROBLEM_BASE}method-unsupported"
    both = await p.handle(ENDPOINT, request({**credential(by_method["stripe"], {"spt": "spt_1"}), X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": {}, "payload": {}})}), ok)
    assert both.status_code == 400
    bearer = await p.handle(ENDPOINT, request({"Authorization": "Bearer abc"}), ok)
    assert bearer.status_code == 402 and bearer.headers["content-type"].startswith("application/json")


async def test_the_robutler_method_settles_once_through_the_credits_scheme():
    credits = CreditsDouble()
    p = build(StripeDouble(), credits)
    _, by_method = await challenges(p)
    challenge = by_method["robutler"]
    res = await p.handle(ENDPOINT, request(credential(challenge, {"token": "tok_valid"})), ok)
    assert res.status_code == 200 and len(credits.settles) == 1
    settle = credits.settles[0]
    assert abs(settle["amount_credits"] - 50) < 1e-9
    receipt = receipt_of(res)
    assert receipt["method"] == "robutler" and receipt["reference"] == settle["idempotency_key"] and receipt["status"] == "success"
    replay = await p.handle(ENDPOINT, request(credential(challenge, {"token": "tok_valid"})), ok)
    assert replay.status_code == 402 and problem_of(replay) == f"{MPP_PROBLEM_BASE}invalid-challenge" and len(credits.settles) == 1
    _, fresh = await challenges(p)
    bad = await p.handle(ENDPOINT, request(credential(fresh["robutler"], {"token": "tok_bad"})), never)
    assert bad.status_code == 402 and problem_of(bad) == f"{MPP_PROBLEM_BASE}verification-failed"


def test_read_mpp_credential_reads_the_two_headers_and_nothing_else():
    cred = base64url_encode(json.dumps({"challenge": {"id": "i", "realm": "r", "method": "stripe", "intent": "charge", "request": "e30"}, "payload": {"spt": "spt_1"}}))
    assert read_mpp_credential({"Authorization": f"Payment {cred}"})["ok"] is True
    assert read_mpp_credential({"Payment-Authorization": f"Payment {cred}"})["ok"] is True
    assert read_mpp_credential({"Authorization": "Bearer x"}) is None
    assert read_mpp_credential({}) is None
    # Built outside the f-string: a backslash inside one is Python 3.12 syntax, and CI runs 3.10.
    no_challenge = base64url_encode('{"payload":{}}')
    assert read_mpp_credential({"Authorization": f"Payment {no_challenge}"})["ok"] is False
