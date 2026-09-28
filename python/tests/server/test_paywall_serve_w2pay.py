"""
A priced `@http` endpoint served end to end (webagents gap-closure plan 2.6,
2026-09-26), the twin of the TypeScript `paywall-serve-w2pay.test.ts`:
`@pricing` stacked on `@http` reaches the server as a priced handler, and
both routes (the static mount and the dynamic catch-all) answer an unpaid
request with a standard x402 402 (v2 header, v1 body), verify a credits
payment against the platform before the handler, settle once after it,
expose the payment headers over CORS, and refuse a priced endpoint on an
agent with no payment skill instead of serving it free. The platform client
is pinned on the wire with an httpx mock transport.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.payments import pricing
from webagents.agents.skills.robutler.payments.paywall import Paywall
from webagents.agents.skills.robutler.payments.x402_credits import CreditsScheme, PlatformCreditsClient
from webagents.agents.skills.robutler.payments.x402_wire import CREDITS_SCHEME, X402_HEADERS, decode_base64_json, encode_base64_json
from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
from webagents.agents.tools.decorators import http
from webagents.server.core.app import WebAgentsServer

PLATFORM = "https://platform.test"


class Quotes(Skill):
    served: List[str] = []

    @http("/quote", method="get")
    @pricing(credits_per_call=0.01, reason="A quote")
    async def quote(self) -> dict:
        """A quote"""
        Quotes.served.append("quote")
        return {"price": 42}

    @http("/metered", method="get")
    @pricing(lock=0.05)
    async def metered(self) -> tuple:
        """A metered quote: the handler names the actual price."""
        Quotes.served.append("metered")
        from webagents.agents.skills.robutler.payments import PricingInfo

        return {"price": 7}, PricingInfo(credits=0.002, reason="two thousandths")

    @http("/free", method="get")
    async def free(self) -> dict:
        Quotes.served.append("free")
        return {"free": True}


class Platform:
    """The platform as the credits scheme calls it, on an httpx mock transport."""

    def __init__(self) -> None:
        self.verifies = 0
        self.settles: List[Dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/payments/verify":
            self.verifies += 1
            return httpx.Response(200, json={"valid": True, "balanceCredits": 3} if body.get("token") == "tok_valid" else {"valid": False, "error": "invalid token"})
        if request.url.path == "/api/payments/settle":
            self.settles.append({"body": body, "headers": dict(request.headers)})
            return httpx.Response(200, json={"success": True, "charged": "12500000", "chargedCredits": 0.0125})
        return httpx.Response(404)


def _agent(platform: Platform | None, payments: bool = True) -> BaseAgent:
    skills: Dict[str, Skill] = {"quotes": Quotes()}
    if payments:
        skill = PaymentSkillX402({"webagents_api_url": PLATFORM, "robutler_api_key": "rok_agent", "x402": {"nonce_secret": "fleet-secret"}})
        assert platform is not None
        client = PlatformCreditsClient(PLATFORM, api_key="rok_agent", client=httpx.AsyncClient(transport=httpx.MockTransport(platform.handler)))
        skill._paywall = Paywall(credits=CreditsScheme(client, nonce_secret="fleet-secret", platform_url=PLATFORM))
        skills["payments"] = skill
    agent = BaseAgent(name="priced", instructions="Priced.", skills=skills)
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _static(agent: BaseAgent) -> TestClient:
    return TestClient(WebAgentsServer(agents=[agent], quiet=True).app)


def _dynamic(agent: BaseAgent) -> TestClient:
    def resolve(name, **_):
        return agent if name == "priced" else None

    return TestClient(WebAgentsServer(agents=[], dynamic_agents=resolve, quiet=True).app)


@pytest.fixture(autouse=True)
def _reset():
    Quotes.served.clear()


@pytest.fixture(params=["static", "dynamic"])
def serve(request):
    return _static if request.param == "static" else _dynamic


def _v2(res) -> Dict[str, Any]:
    return decode_base64_json(res.headers[X402_HEADERS["required"]])


def test_free_is_served_and_priced_answers_402_with_both_documents(serve):
    platform = Platform()
    client = serve(_agent(platform))
    assert client.get("/priced/free").status_code == 200
    assert Quotes.served == ["free"]

    res = client.get("/priced/quote")
    assert res.status_code == 402 and Quotes.served == ["free"]
    assert res.headers["cache-control"] == "no-store"
    v2 = _v2(res)
    assert v2["x402Version"] == 2 and v2["resource"]["url"].endswith("/priced/quote") and v2["resource"]["description"] == "A quote"
    assert [r["scheme"] for r in v2["accepts"]] == [CREDITS_SCHEME] and v2["accepts"][0]["amount"] == "10000000"
    v1 = res.json()
    assert v1["x402Version"] == 1 and v1["accepts"][0]["maxAmountRequired"] == "10000000"
    assert platform.verifies == 0


def test_a_credits_payment_is_verified_before_the_handler_and_settled_once_after_it(serve):
    platform = Platform()
    client = serve(_agent(platform))
    entry = _v2(client.get("/priced/quote"))["accepts"][0]
    headers = {X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": entry, "payload": {"token": "tok_valid"}})}
    paid = client.get("/priced/quote", headers=headers)
    assert paid.status_code == 200 and paid.json() == {"price": 42}
    assert Quotes.served == ["quote"] and platform.verifies == 1 and len(platform.settles) == 1
    nonce_id = entry["extra"]["nonce"].split(".")[0]
    settle = platform.settles[0]
    assert settle["body"]["token"] == "tok_valid" and settle["body"]["amount"] == 0.01 and settle["body"]["idempotencyKey"] == f"settle:x402:{nonce_id}"
    assert settle["headers"]["idempotency-key"] == f"settle:x402:{nonce_id}" and settle["headers"]["authorization"] == "Bearer rok_agent"
    assert decode_base64_json(paid.headers[X402_HEADERS["response"]])["amount"] == "10000000"
    assert paid.headers["cache-control"] == "private"

    again = client.get("/priced/quote", headers=headers)
    assert again.status_code == 402 and Quotes.served == ["quote"] and len(platform.settles) == 1


def test_a_metered_handler_names_its_actual_price_through_the_pricing_tuple(serve):
    platform = Platform()
    client = serve(_agent(platform))
    entry = _v2(client.get("/priced/metered"))["accepts"][0]
    assert entry["amount"] == "50000000"
    paid = client.get("/priced/metered", headers={X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": entry, "payload": {"token": "tok_valid"}})})
    assert paid.status_code == 200 and paid.json() == {"price": 7}
    assert X402_HEADERS["settlement_overrides"] not in paid.headers
    assert abs(platform.settles[0]["body"]["amount"] - 0.002) < 1e-12
    assert decode_base64_json(paid.headers[X402_HEADERS["response"]])["amount"] == "2000000"


def test_an_invalid_token_never_reaches_the_handler(serve):
    platform = Platform()
    client = serve(_agent(platform))
    entry = _v2(client.get("/priced/quote"))["accepts"][0]
    res = client.get("/priced/quote", headers={X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": entry, "payload": {"token": "tok_bad"}})})
    assert res.status_code == 402 and Quotes.served == [] and platform.settles == []


def test_a_priced_endpoint_with_no_payment_skill_is_refused_never_served_free(serve):
    client = serve(_agent(None, payments=False))
    res = client.get("/priced/quote")
    assert res.status_code == 503 and res.json()["error"]["code"] == "payment_not_configured" and Quotes.served == []


def test_the_payment_headers_are_exposed_over_cors():
    client = _static(_agent(Platform()))
    res = client.get("/priced/quote", headers={"Origin": "http://localhost:5173"})
    exposed = res.headers.get("access-control-expose-headers", "")
    assert "PAYMENT-REQUIRED" in exposed and "PAYMENT-RESPONSE" in exposed and "WWW-Authenticate" in exposed


async def test_the_platform_client_speaks_the_settle_and_verify_wire():
    platform = Platform()
    client = PlatformCreditsClient(PLATFORM, api_key="rok_agent", client=httpx.AsyncClient(transport=httpx.MockTransport(platform.handler)))
    assert await client.verify_token("tok_valid") == {"valid": True, "balance": 3.0}
    assert (await client.verify_token("tok_bad"))["valid"] is False
    result = await client.settle_token("tok_valid", 0.01, "settle:x402:3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c", description="d", resource="r")
    assert result["success"] is True and result["chargedCredits"] == 0.0125
    sent = platform.settles[0]
    assert sent["body"] == {"token": "tok_valid", "amount": 0.01, "idempotencyKey": "settle:x402:3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c", "description": "d", "resource": "r"}
    assert sent["headers"]["idempotency-key"] == "settle:x402:3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c"
