"""
A priced `@http` endpoint on an agent with no paywall is refused with the
sentence the shared fixture pins (`tests/fixtures/payments/paywall_x402_mpp.json`,
`not_configured`), never served free (2026-09-27, the final e2e re-run). The
TypeScript side of the same fixture also pins the sentence its
`PaymentX402Skill` answers, which names `PaymentSkill`; Python has no twin of
that sentence because `PaymentSkillX402` extends `PaymentSkill` and is the
seller, which this file proves too: with it on the agent the same request is
the 402 challenge.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.payments import pricing
from webagents.agents.skills.robutler.payments.paywall import PAYMENT_NOT_CONFIGURED_MESSAGE, resolve_paywall
from webagents.agents.skills.robutler.payments.x402_wire import X402_HEADERS
from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
from webagents.agents.tools.decorators import http
from webagents.server.core.app import WebAgentsServer

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "payments" / "paywall_x402_mpp.json").read_text())
NOT_CONFIGURED = FIXTURE["x402"]["not_configured"]


class Quotes(Skill):
    served = []

    @http("/quote", method="get")
    @pricing(credits_per_call=0.01, reason="A quote")
    async def quote(self) -> dict:
        """A quote"""
        Quotes.served.append("quote")
        return {"price": 42}


def _agent(seller: bool) -> BaseAgent:
    skills = {"quotes": Quotes()}
    if seller:
        skills["payments"] = PaymentSkillX402({"robutler_api_key": "rok_agent", "x402": {"nonce_secret": "fleet-secret"}})
    return BaseAgent(name="priced", instructions="Priced.", skills=skills)


def _client(agent: BaseAgent) -> TestClient:
    return TestClient(WebAgentsServer(agents=[agent], enable_monitoring=False).app)


def test_the_fixture_is_the_contract():
    assert PAYMENT_NOT_CONFIGURED_MESSAGE == NOT_CONFIGURED["message"]
    assert NOT_CONFIGURED["code"] == "payment_not_configured" and NOT_CONFIGURED["status"] == 503
    assert "PaymentSkill" in NOT_CONFIGURED["typescript_x402_skill"]["message"]


def test_a_priced_endpoint_with_no_payment_skill_answers_the_fixtures_sentence():
    Quotes.served.clear()
    res = _client(_agent(seller=False)).get("/priced/quote")
    assert res.status_code == NOT_CONFIGURED["status"]
    assert res.json() == {"error": {"code": NOT_CONFIGURED["code"], "message": NOT_CONFIGURED["message"]}}
    assert Quotes.served == []


def test_payment_skill_x402_is_the_seller_in_python():
    Quotes.served.clear()
    agent = _agent(seller=True)
    assert resolve_paywall(agent) is not None
    res = _client(agent).get("/priced/quote")
    assert res.status_code == 402, res.text
    assert res.headers.get(X402_HEADERS["required"])
    assert Quotes.served == []
