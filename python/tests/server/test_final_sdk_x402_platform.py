"""
Where a priced endpoint says to pay, and what it answers when the platform
cannot be reached (B12, 2026-09-28), against the shared fixture
`payments/final_sdk_x402_platform.json` the TypeScript suite reads too.

The e2e run's priced agent, with no `ROBUTLER_API_URL`, published
`"platform": "http://localhost:3000"` in its 402, and a paid retry then
answered 500: the verify's `httpx.ConnectError` escaped the paywall.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.payments import pricing
from webagents.agents.skills.robutler.payments.paywall import Paywall
from webagents.agents.skills.robutler.payments.x402_credits import CreditsScheme, PlatformCreditsClient
from webagents.agents.skills.robutler.payments.x402_wire import X402_HEADERS, decode_base64_json, encode_base64_json
from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
from webagents.agents.tools.decorators import http
from webagents.server.core.app import WebAgentsServer

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "payments" / "final_sdk_x402_platform.json").read_text()
)


@pytest.fixture(autouse=True)
def _nothing_names_a_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("ROBUTLER_API_URL", "ROBUTLER_INTERNAL_API_URL", "ROBUTLER_PLATFORM_URL", "WEBAGENTS_PROFILE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


class Quotes(Skill):
    served = []

    @http("/quote", method="get")
    @pricing(credits_per_call=0.01, reason="A quote")
    async def quote(self) -> dict:
        """A quote"""
        Quotes.served.append("quote")
        return {"price": 42}


def test_the_platform_a_402_names_is_never_localhost_by_default():
    skill = PaymentSkillX402({"x402": {"nonce_secret": "fleet-secret"}})
    assert skill.webagents_api_url == FIXTURE["default_platform"]
    agent = BaseAgent(name="priced", instructions="Priced.", skills={"quotes": Quotes(), "payments": skill})
    asyncio.run(agent._ensure_skills_initialized())
    res = TestClient(WebAgentsServer(agents=[agent], quiet=True).app).get("/priced/quote")
    assert res.status_code == 402
    entry = decode_base64_json(res.headers[X402_HEADERS["required"]])["accepts"][0]
    assert entry["extra"]["platform"] == FIXTURE["default_platform"]
    assert "localhost" not in res.text


def test_the_unreachable_platform_is_an_answer_not_an_exception():
    client = PlatformCreditsClient(FIXTURE["unreachable_platform"], timeout=2.0)
    verified = asyncio.run(client.verify_token("tok"))
    assert verified == {"valid": False, "invalidReason": FIXTURE["unreachable"]["reason"]}
    settled = asyncio.run(client.settle_token("tok", 0.01, "settle:x402:x"))
    assert settled.get("success") is False and settled.get("error") == FIXTURE["unreachable"]["reason"]


def test_a_paid_retry_against_an_unreachable_platform_is_503_and_runs_nothing():
    Quotes.served.clear()
    skill = PaymentSkillX402({"webagents_api_url": FIXTURE["unreachable_platform"], "x402": {"nonce_secret": "fleet-secret"}})
    client = PlatformCreditsClient(FIXTURE["unreachable_platform"], timeout=2.0)
    skill._paywall = Paywall(credits=CreditsScheme(client, nonce_secret="fleet-secret", platform_url=FIXTURE["unreachable_platform"]))
    agent = BaseAgent(name="priced", instructions="Priced.", skills={"quotes": Quotes(), "payments": skill})
    asyncio.run(agent._ensure_skills_initialized())
    http_client = TestClient(WebAgentsServer(agents=[agent], quiet=True).app)
    entry = decode_base64_json(http_client.get("/priced/quote").headers[X402_HEADERS["required"]])["accepts"][0]
    headers: Dict[str, str] = {
        X402_HEADERS["signature"]: encode_base64_json({"x402Version": 2, "accepted": entry, "payload": {"token": "tok_valid"}})
    }
    paid = http_client.get("/priced/quote", headers=headers)
    expected = FIXTURE["unreachable"]
    assert paid.status_code == expected["status"]
    assert paid.json() == expected["body"]
    assert paid.headers["retry-after"] == expected["retry_after"]
    assert Quotes.served == []
