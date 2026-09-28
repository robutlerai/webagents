"""
Per-hop budgets in the open SDK (webagents gap-closure plan 2.3,
2026-09-26), the twin of the TypeScript `nli-delegate-budget-w2pay.test.ts`:
`nli_tool` takes a `budget` (the fixture's parameter), a hop is funded by a
child token for EXACTLY that budget through the platform's delegate route
(never the parent's whole balance), a refusal fails the hop closed instead of
forwarding the parent, and each hop returns a receipt in the fixture's shape.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from webagents.agents.skills.robutler.nli.budget import (
    DELEGATE_BUDGET_PARAMETER,
    format_delegate_receipt,
    mint_child_token,
    read_child_receipt,
    resolve_delegate_budget,
    token_fingerprint,
    token_id_of,
)
from webagents.agents.skills.robutler.nli.skill import NLISkill

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "delegate_budget.json").read_text())
CHILD_ID = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _b64url(value: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")


CHILD_JWT = f"{_b64url({'alg': 'RS256'})}.{_b64url({'jti': CHILD_ID, 'payment': {'balance': 0.1}})}.sig"
PARENT_JWT = f"{_b64url({'alg': 'RS256'})}.{_b64url({'jti': 'parent-1', 'payment': {'balance': 3, 'max_depth': 3}})}.sig"


def test_the_fixture_pins_the_parameter_and_the_receipt():
    p = FIXTURE["parameter"]
    assert DELEGATE_BUDGET_PARAMETER["name"] == p["name"]
    assert DELEGATE_BUDGET_PARAMETER["description"] == p["description"]
    assert DELEGATE_BUDGET_PARAMETER["default"] == p["default"] and DELEGATE_BUDGET_PARAMETER["max"] == p["max"]
    assert resolve_delegate_budget(None) == {"ok": True, "budget": p["default"]}
    assert resolve_delegate_budget(0.5) == {"ok": True, "budget": 0.5}
    assert resolve_delegate_budget(p["max"] + 1)["ok"] is False
    assert resolve_delegate_budget(0)["ok"] is False
    assert resolve_delegate_budget("nope")["ok"] is False
    for v in FIXTURE["receipt"]["vectors"]:
        # The receipt names the child by a fingerprint of its id, never the id (S-303).
        assert token_fingerprint(v["tokenId"]) == v["token"]
        assert format_delegate_receipt({"budget": v["budget"], "spent": v["spent"], "remaining": v["remaining"], "token": v["token"]}) == v["text"]
        assert v["tokenId"] not in v["text"]
    assert token_id_of(CHILD_JWT) == CHILD_ID and token_id_of("nope") is None


class Platform:
    """The delegate and verify routes, on an httpx mock transport; records what was sent."""

    def __init__(self) -> None:
        self.delegations: List[Dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/payments/delegate":
            self.delegations.append({"body": body, "auth": request.headers.get("authorization")})
            if body.get("amount") == 4.5:
                return httpx.Response(400, json={"error": "Delegation refused: max_depth exhausted"})
            return httpx.Response(200, json={"token": CHILD_JWT, "tokenId": CHILD_ID, "amountCredits": body["amount"]})
        if request.url.path == "/api/payments/verify":
            return httpx.Response(200, json={"valid": True, "balanceCredits": 0.0877})
        return httpx.Response(404)


async def test_the_helpers_speak_the_routes():
    platform = Platform()
    client = httpx.AsyncClient(transport=httpx.MockTransport(platform.handler))
    minted = await mint_child_token("https://platform.test", "rok", PARENT_JWT, "@x", 0.25, client=client)
    assert minted == {"ok": True, "token": CHILD_JWT, "tokenId": CHILD_ID, "amountCredits": 0.25}
    assert platform.delegations[0]["body"] == {"parentToken": PARENT_JWT, "delegateTo": "x", "amount": 0.25}
    assert sorted(platform.delegations[0]["body"]) == sorted(FIXTURE["delegate_route"]["body"])
    assert platform.delegations[0]["auth"] == "Bearer rok"
    refused = await mint_child_token("https://platform.test", "rok", PARENT_JWT, "@x", 4.5, client=client)
    assert refused == {"ok": False, "error": "Delegation refused: max_depth exhausted"}
    receipt = await read_child_receipt("https://platform.test", CHILD_JWT, 0.1, client=client)
    assert receipt == {"budget": 0.1, "spent": 0.0123, "remaining": 0.0877, "token": token_fingerprint(CHILD_ID)}
    assert format_delegate_receipt(receipt) == FIXTURE["receipt"]["vectors"][0]["text"]


def _skill(platform: Platform, monkeypatch: pytest.MonkeyPatch) -> NLISkill:
    monkeypatch.setenv("ROBUTLER_API_URL", "https://platform.test")
    skill = NLISkill({"transport": "http"})
    skill.agent = type("Agent", (), {"name": "planner", "api_key": "rok_agent", "config": {}})()
    skill.logger = __import__("logging").getLogger("test.nli")
    skill._auth_token = "rok_agent"
    skill.http_client = httpx.AsyncClient(transport=httpx.MockTransport(platform.handler))
    return skill


async def test_the_delegation_mints_exactly_the_budget_never_the_parents_balance(monkeypatch):
    platform = Platform()
    skill = _skill(platform, monkeypatch)
    transport = httpx.MockTransport(platform.handler)
    real_client = httpx.AsyncClient

    class Client(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    child = await skill._delegate_payment(PARENT_JWT, "callee", 0.1, "@callee")
    assert child == CHILD_JWT
    # The parent JWT says balance 3; the hop asked for 0.1 and got exactly that.
    assert platform.delegations[-1]["body"]["amount"] == 0.1
    refused = await skill._delegate_payment(PARENT_JWT, "callee", 4.5, "@callee")
    assert refused is None


async def test_a_refused_derivation_fails_the_hop_closed_and_never_forwards_the_parent(monkeypatch):
    platform = Platform()
    skill = _skill(platform, monkeypatch)
    sent: List[Dict[str, str]] = []

    async def post(url, json=None, headers=None, timeout=None):
        sent.append(dict(headers or {}))
        return httpx.Response(200, json={"choices": [{"message": {"content": "never"}}]})

    skill.http_client.post = post  # type: ignore[assignment]

    async def refuse(*args, **kwargs):
        return None

    monkeypatch.setattr(skill, "_delegate_payment", refuse)
    monkeypatch.setattr(skill, "_resolve_agent_to_url", lambda ident: "https://platform.test/agents/callee/chat/completions")
    monkeypatch.setattr(skill, "_resolve_agent_id", _async_value("callee-id"))
    from webagents.server.context import context_vars

    ctx = type("Ctx", (), {"auth": None, "payment_token": PARENT_JWT, "payments": None, "request": None, "get": lambda self, k, d=None: d})()
    monkeypatch.setattr(context_vars, "get_context", lambda: ctx)
    result = await skill.nli_tool("@callee", "hi", budget=0.1)
    assert "refused" in result and "budget could not be derived" in result
    assert sent == []


async def test_a_budget_above_the_maximum_is_refused_before_anything_runs(monkeypatch):
    platform = Platform()
    skill = _skill(platform, monkeypatch)
    result = await skill.nli_tool("@callee", "hi", budget=99)
    assert "exceeds maximum allowed" in result
    assert platform.delegations == []


async def test_a_hop_returns_its_receipt(monkeypatch):
    platform = Platform()
    skill = _skill(platform, monkeypatch)
    transport = httpx.MockTransport(platform.handler)
    real_client = httpx.AsyncClient

    class Client(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    text = await skill._with_receipt("answer", CHILD_JWT, 0.1, CHILD_ID)
    assert text == "answer\n" + FIXTURE["receipt"]["vectors"][0]["text"]
    assert await skill._with_receipt("answer", None, 0.1, None) == "answer"


def _async_value(value):
    async def inner(*args, **kwargs):
        return value

    return inner
