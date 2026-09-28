"""
S-297 (2026-09-26): a replayed x402 or MPP payment never runs the handler
again for free, and a nonce is bound to the request the 402 answered. The
twin of the TypeScript `paywall-replay-w2fix.test.ts`; the fixture
`fixtures/payments/paywall_x402_mpp.json` pins the binding format and the
bound nonce vectors for both SDKs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import Response

from webagents.agents.skills.robutler.payments.mpp_seller import MppSeller, parse_www_authenticate_payment
from webagents.agents.skills.robutler.payments.paywall import Paywall, PricedEndpoint, request_binding
from webagents.agents.skills.robutler.payments.x402_credits import CreditsScheme, mint_credits_nonce, nonce_message, verify_credits_nonce
from webagents.agents.skills.robutler.payments.x402_wire import X402_HEADERS, decode_base64_json, encode_base64_json
from webagents.agents.skills.robutler.payments_x402.mpp_buyer import base64url_encode

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "paywall_x402_mpp.json").read_text())
N = FIXTURE["x402"]["credits_nonce"]
URL_ = "https://agent.example.com/agents/mini/quote"
ENDPOINT = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 0.01}, description="A quote")
POST_ENDPOINT = PricedEndpoint(path="/quote", method="POST", pricing={"credits_per_call": 0.01}, description="A quote")


def request(headers: Optional[Dict[str, str]] = None, method: str = "GET", query: bytes = b"q=1", body: bytes = b"", path: str = "/agents/mini/quote") -> Request:
    scope = {
        "type": "http",
        "method": method,
        "scheme": "https",
        "server": ("agent.example.com", 443),
        "path": path,
        "query_string": query,
        "headers": [(k.lower().encode(), v.encode()) for k, v in {"host": "agent.example.com", **(headers or {})}.items()],
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=receive)


# ── The binding and the bound nonce ─────────────────────────────────────────


async def test_the_request_binding_matches_the_fixture_and_leaves_the_body_readable():
    r = N["request_binding"]
    posted = request(method=r["method"], query=r["query"].encode(), body=r["body"].encode(), path=r["path"], headers={"content-type": "application/json"})
    assert await request_binding(posted) == r["binding"]
    assert (await posted.body()).decode() == r["body"]
    assert await request_binding(request(query=r["query"].encode(), path=r["path"])) == r["get_binding"]
    assert await request_binding(request(query=b"", path=r["path"])) == f"GET|{r['path']}||{r['get_binding'].split('|')[3]}"


def test_the_bound_vectors_mint_and_verify_and_bound_and_unbound_do_not_cross():
    for v in N["bound_vectors"]:
        assert nonce_message(v["resource"], v["amount"], v["id"], v["expires"], v["binding"]) == v["message"]
        now = datetime.fromtimestamp(v["expires"] - 300, tz=timezone.utc)
        assert mint_credits_nonce(N["secret"], v["resource"], v["amount"], now=now, id=v["id"], binding=v["binding"]) == v["nonce"]
        at = datetime.fromtimestamp(v["expires"] - 10, tz=timezone.utc)
        assert verify_credits_nonce(N["secret"], v["nonce"], v["resource"], v["amount"], at, binding=v["binding"])["id"] == v["id"]
        assert verify_credits_nonce(N["secret"], v["nonce"], v["resource"], v["amount"], at)["reason"] == "nonce_not_ours"
        assert verify_credits_nonce(N["secret"], v["nonce"], v["resource"], v["amount"], at, binding=v["binding"] + "x")["reason"] == "nonce_not_ours"
    a = N["vectors"][0]
    at = datetime.fromtimestamp(a["expires"] - 10, tz=timezone.utc)
    assert verify_credits_nonce(N["secret"], a["nonce"], a["resource"], a["amount"], at, binding=N["bound_vectors"][1]["binding"])["reason"] == "nonce_not_ours"


# ── The paywall ──────────────────────────────────────────────────────────────


class SharedStore:
    """The claim table two 'processes' share."""

    def __init__(self) -> None:
        self.claimed: set = set()
        self.claims: List[Dict[str, Any]] = []


class FakeCredits:
    def __init__(self, store: Optional[SharedStore] = None, settle_result: Optional[Dict[str, Any]] = None):
        self.settles: List[Dict[str, Any]] = []
        self.settle_result = settle_result
        if store is not None:
            self._store = store
            self.claim_nonce = self._claim  # type: ignore[assignment]

    async def verify_token(self, token, expected_audience=None):
        # Enough for the 50-credit MPP endpoint below as well as the 0.01-credit one.
        return {"valid": True, "balance": 90.0} if token == "tok_valid" else {"valid": False, "invalidReason": "token_invalid"}

    async def settle_token(self, token, amount_credits, idempotency_key, description=None, resource=None):
        self.settles.append({"token": token, "idempotency_key": idempotency_key})
        return self.settle_result or {"success": True, "charged": str(int(round(amount_credits * 1e9)))}

    async def _claim(self, claim: Dict[str, Any]) -> bool:
        self._store.claims.append(claim)
        if claim["nonce_id"] in self._store.claimed:
            return False
        self._store.claimed.add(claim["nonce_id"])
        return True


def paywall(credits: FakeCredits, events: Optional[List[str]] = None) -> Paywall:
    return Paywall(credits=CreditsScheme(credits, nonce_secret="fleet", platform_url="https://robutler.ai"), resource={"service_name": "Mini"}, log=(lambda e, f: events.append(e)) if events is not None else None)


async def never():
    raise AssertionError("the handler must not run")


async def challenge(p: Paywall, req: Request, endpoint: PricedEndpoint = ENDPOINT) -> Dict[str, Any]:
    res = await p.handle(endpoint, req, never)
    assert res.status_code == 402
    return decode_base64_json(res.headers[X402_HEADERS["required"]])


def v2_payment(accepted: Dict[str, Any], token: str = "tok_valid") -> Dict[str, str]:
    return {X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": accepted, "payload": {"token": token}})}


def refusal(res: Response) -> str:
    return decode_base64_json(res.headers[X402_HEADERS["required"]])["error"]


class Ran:
    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> Response:
        self.count += 1
        return Response("ok", status_code=200)


async def test_a_shared_store_claims_before_the_handler_and_a_second_process_refuses_the_replay():
    store = SharedStore()
    pod_a, pod_b = paywall(FakeCredits(store)), paywall(FakeCredits(store))
    v2 = await challenge(pod_a, request())
    ran = Ran()
    first = await pod_a.handle(ENDPOINT, request(v2_payment(v2["accepts"][0])), ran)
    assert first.status_code == 200 and ran.count == 1
    nonce_id = v2["accepts"][0]["extra"]["nonce"].split(".")[0]
    assert store.claims[0]["nonce_id"] == nonce_id and store.claims[0]["resource"] == URL_
    assert store.claims[0]["binding"].startswith("GET|/agents/mini/quote|q=1|")

    replay = await pod_b.handle(ENDPOINT, request(v2_payment(v2["accepts"][0])), ran)
    assert replay.status_code == 402 and refusal(replay) == "nonce_used"
    assert ran.count == 1


async def test_an_invalid_token_burns_nothing():
    store = SharedStore()
    p = paywall(FakeCredits(store))
    v2 = await challenge(p, request())
    bad = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0], "tok_bad")), never)
    assert bad.status_code == 402 and store.claims == []
    good = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0])), Ran())
    assert good.status_code == 200


async def test_a_replayed_settle_is_withheld():
    events: List[str] = []
    p = paywall(FakeCredits(settle_result={"success": True, "replayed": True, "charged": "10000000"}), events)
    v2 = await challenge(p, request())

    async def answer():
        return Response("the answer", status_code=200)

    res = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0])), answer)
    assert res.status_code == 402 and res.body == b"{}"
    settled = decode_base64_json(res.headers[X402_HEADERS["response"]])
    assert settled["success"] is False and settled["errorReason"] == "replayed"
    assert "x402.replayed" in events and "x402.settled" not in events


async def test_a_retry_with_another_query_or_body_is_not_ours():
    p = paywall(FakeCredits())
    v2 = await challenge(p, request())
    ran = Ran()
    other = await p.handle(ENDPOINT, request(v2_payment(v2["accepts"][0]), query=b"q=2"), ran)
    assert other.status_code == 402 and refusal(other) == "nonce_not_ours" and ran.count == 0

    body = json.dumps({"q": "how much"}).encode()
    v2 = await challenge(p, request(method="POST", body=body), POST_ENDPOINT)
    changed = await p.handle(POST_ENDPOINT, request(v2_payment(v2["accepts"][0]), method="POST", body=json.dumps({"q": "everything"}).encode()), ran)
    assert changed.status_code == 402 and refusal(changed) == "nonce_not_ours" and ran.count == 0
    same_request = request(v2_payment(v2["accepts"][0]), method="POST", body=body)

    async def reads_body():
        assert await same_request.body() == body
        return Response("ok", status_code=200)

    assert (await p.handle(POST_ENDPOINT, same_request, reads_body)).status_code == 200


async def test_a_v1_payment_is_bound_the_same_way():
    p = paywall(FakeCredits())
    v2 = await challenge(p, request())
    entry = v2["accepts"][0]
    v1 = {X402_HEADERS["v1_payment"]: encode_base64_json({"x402Version": 1, "scheme": entry["scheme"], "network": entry["network"], "payload": {"token": "tok_valid", "nonce": entry["extra"]["nonce"]}})}
    ran = Ran()
    assert refusal(await p.handle(ENDPOINT, request(v1, query=b"q=other"), ran)) == "nonce_not_ours"
    assert (await p.handle(ENDPOINT, request(v1), ran)).status_code == 200 and ran.count == 1


# ── MPP ──────────────────────────────────────────────────────────────────────

NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
MPP_ENDPOINT = PricedEndpoint(path="/quote", method="GET", pricing={"credits_per_call": 50}, description="AI generation")


class StripeDouble:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def create_payment_intent(self, params, idempotency_key):
        self.calls.append(idempotency_key)
        return {"id": "pi_1", "status": "succeeded"}


def build(credits: FakeCredits, stripe: Optional[StripeDouble] = None, claim_challenge=None) -> Paywall:
    mpp = MppSeller(realm="agent.example.com", secret="mpp-secret", stripe={"profile_id": "profile_test_123", "client": stripe} if stripe else None, now=lambda: NOW, claim_challenge=claim_challenge)
    return Paywall(credits=CreditsScheme(credits, nonce_secret="s", platform_url="https://robutler.ai", now=lambda: NOW), mpp=mpp)


async def challenges(p: Paywall) -> Dict[str, Dict[str, Any]]:
    res = await p.handle(MPP_ENDPOINT, request(), never)
    by_method: Dict[str, Dict[str, Any]] = {}
    for value in res.headers.getlist("www-authenticate"):
        parsed = parse_www_authenticate_payment(value)
        if parsed:
            by_method[parsed["method"]] = parsed
    return by_method


def credential(challenge_fields: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, str]:
    return {"Authorization": f"Payment {base64url_encode(json.dumps({'challenge': challenge_fields, 'payload': payload}, separators=(',', ':')))}"}


async def test_mpp_robutler_replayed_settle_is_withheld():
    p = build(FakeCredits(settle_result={"success": True, "replayed": True, "charged": "1"}), StripeDouble())
    by_method = await challenges(p)

    async def answer():
        return Response("the answer", status_code=200)

    res = await p.handle(MPP_ENDPOINT, request(credential(by_method["robutler"], {"token": "tok_valid"})), answer)
    assert res.status_code == 402 and res.body != b"the answer"
    assert "invalid-challenge" in res.body.decode()


async def test_mpp_robutler_shares_the_credits_claim_across_processes():
    store = SharedStore()
    pod_a, pod_b = build(FakeCredits(store), StripeDouble()), build(FakeCredits(store), StripeDouble())
    by_method = await challenges(pod_a)
    ran = Ran()
    assert (await pod_a.handle(MPP_ENDPOINT, request(credential(by_method["robutler"], {"token": "tok_valid"})), ran)).status_code == 200
    assert (await pod_b.handle(MPP_ENDPOINT, request(credential(by_method["robutler"], {"token": "tok_valid"})), ran)).status_code == 402
    assert ran.count == 1


async def test_mpp_stripe_claim_challenge_hook_refuses_a_second_process():
    claimed: set = set()

    async def claim_challenge(claim: Dict[str, Any]) -> bool:
        if claim["challenge_id"] in claimed:
            return False
        claimed.add(claim["challenge_id"])
        return True

    stripe_a, stripe_b = StripeDouble(), StripeDouble()
    pod_a = build(FakeCredits(), stripe_a, claim_challenge)
    pod_b = build(FakeCredits(), stripe_b, claim_challenge)
    by_method = await challenges(pod_a)
    ran = Ran()
    assert (await pod_a.handle(MPP_ENDPOINT, request(credential(by_method["stripe"], {"spt": "spt_1N4Zv32eZvKYlo2CPhVPkJlW"})), ran)).status_code == 200
    assert len(stripe_a.calls) == 1
    assert (await pod_b.handle(MPP_ENDPOINT, request(credential(by_method["stripe"], {"spt": "spt_1N4Zv32eZvKYlo2CPhVPkJlW"})), ran)).status_code == 402
    assert stripe_b.calls == [] and ran.count == 1
    alone = build(FakeCredits(), StripeDouble())
    ch = await challenges(alone)
    assert (await alone.handle(MPP_ENDPOINT, request(credential(ch["stripe"], {"spt": "spt_1N4Zv32eZvKYlo2CPhVPkJlW"})), Ran())).status_code == 200
    assert (await alone.handle(MPP_ENDPOINT, request(credential(ch["stripe"], {"spt": "spt_1N4Zv32eZvKYlo2CPhVPkJlW"})), Ran())).status_code == 402
