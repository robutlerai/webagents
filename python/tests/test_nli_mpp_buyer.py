"""
The NLI skill's buyer wiring (machine-purchase design section 6.4; pass P9b,
2026-09-18), the Python twin of
`webagents/typescript/tests/unit/skills/nli/mpp-buyer-wiring.test.ts` and
`tests/unit/uamp/client-mpp.test.ts`. With `mpp_buyer` in the config, the
HTTP delegate goes through the buyer's `paying_request` instead of the bare
post, a UAMP `payment.required` carrying an `mpp` scheme is paid in band and
the run resumes with `payment.submit` scheme `balance`, and the model-facing
failure-mode prompt stops telling the model to suggest a top-up for a 402.
Without a buyer every one of those is exactly what it was.

The buyer here is a stub with the two methods the skill duck-types; the
buyer's own suite is `tests/test_mpp_buyer.py`.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from webagents.agents.skills.robutler.nli import NLISkill
from webagents.agents.skills.robutler.nli.skill import _find_mpp_scheme

CHALLENGE = 'Payment id="c1", realm="robutler.ai", method="stripe", intent="charge", request="cmVx"'
PURCHASE_URL = "https://robutler.ai/api/mpp/credits"
TERMS = {"url": "https://robutler.ai/doc/terms-of-service", "version": "2026-07-31"}


def sse(content: str) -> bytes:
    return (f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\ndata: [DONE]\n\n').encode()


class StubBuyer:
    def __init__(self, response: Any = None, purchase_ok: bool = True) -> None:
        self.requests: List[dict] = []
        self.purchases: List[tuple] = []
        self.response = response
        self.purchase_ok = purchase_ok

    async def paying_request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        if self.response is not None:
            return self.response
        return httpx.Response(200, content=sse("bought"), request=httpx.Request(method, url))

    async def purchase(self, url, challenge, terms=None):
        self.purchases.append((url, challenge, terms))
        if self.purchase_ok:
            return SimpleNamespace(ok=True)
        return SimpleNamespace(ok=False, reason="terms_refused", detail="the policy does not accept Terms version x")


class Agent:
    name = "test-agent"
    api_key = "test_api_key"


def make_skill(**config: Any) -> NLISkill:
    # A fixed base for routing; the default (the platform lookup) is test_nli_skill's.
    skill = NLISkill({"timeout": 5.0, "max_retries": 0, "default_authorization": 0.05, "max_authorization": 1.0, "agent_base_url": "http://localhost:2224", **config})
    skill.agent = Agent()
    skill.logger = MagicMock()
    skill._auth_token = Agent.api_key
    skill.http_client = AsyncMock()
    skill._resolve_agent_id = AsyncMock(return_value=None)
    skill._mint_owner_assertion = AsyncMock(return_value=None)
    return skill


class FakeSocket:
    """Records frames sent, replays scripted server events (the shape of
    `MockUAMPWebSocket` in tests/test_nli_skill.py). Every script ends with
    `response.done`."""

    def __init__(self, events: List[dict]) -> None:
        self.sent: List[dict] = []
        self._events = [json.dumps(e) for e in events]

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        if not self._events:
            raise AssertionError("script exhausted")
        return self._events.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def payment_required(schemes: List[dict], amount: str = "0.50") -> dict:
    return {"type": "payment.required", "requirements": {"amount": amount, "currency": "USD", "schemes": schemes}}


MPP_SCHEMES = [
    {"scheme": "token", "network": "robutler"},
    {"scheme": "mpp", "challenge": CHALLENGE, "purchase_url": PURCHASE_URL, "terms": TERMS},
]


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------


def test_the_failure_mode_prompt_suggests_a_top_up_only_without_a_buyer():
    plain = make_skill().nli_general_prompt()
    assert "suggest /topup" in plain
    assert "purchase policy" not in plain

    bought = make_skill(mpp_buyer=StubBuyer()).nli_general_prompt()
    assert "suggest /topup" not in bought
    assert "payment_required / 402" in bought
    assert "purchase policy" in bought
    assert "re-sent the request once" in bought
    # The 402 line still ends in an instruction not to retry from the model's side.
    assert "do NOT retry, the operator must change the policy or the funding" in bought
    # Copy rule: usage is bought from Robutler; no agent or creator is paid.
    assert not re.search(r"pay (the|an) (agent|creator)", bought, re.IGNORECASE)


# --------------------------------------------------------------------------
# The HTTP delegate
# --------------------------------------------------------------------------


async def test_the_http_delegate_goes_through_the_buyer_with_the_same_request():
    buyer = StubBuyer()
    skill = make_skill(transport="http", mpp_buyer=buyer)
    with patch("webagents.server.context.context_vars.get_context", return_value=None):
        result = await skill.nli_tool(agent="@acme", message="hi")
    assert result == "bought"
    skill.http_client.post.assert_not_called()
    [call] = buyer.requests
    assert call["method"] == "POST"
    assert call["url"] == "http://localhost:2224/agents/acme/chat/completions"
    assert call["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert call["json"]["stream"] is True
    assert call["headers"]["Authorization"] == "Bearer test_api_key"
    assert call["timeout"] == 5.0


async def test_without_a_buyer_the_http_delegate_is_the_post_it_always_was():
    skill = make_skill(transport="http")
    skill.http_client.post.return_value = httpx.Response(200, content=sse("plain"))
    with patch("webagents.server.context.context_vars.get_context", return_value=None):
        result = await skill.nli_tool(agent="@acme", message="hi")
    assert result == "plain"
    skill.http_client.post.assert_called_once()


async def test_a_402_the_buyer_could_not_fund_still_fails_the_delegate_closed():
    refused = httpx.Response(402, content=b'{"status":402}', request=httpx.Request("POST", "http://x"))
    skill = make_skill(transport="http", mpp_buyer=StubBuyer(response=refused))
    with patch("webagents.server.context.context_vars.get_context", return_value=None):
        result = await skill.nli_tool(agent="@acme", message="hi")
    assert result.startswith("\u274c Failed to communicate with @acme")
    assert "Payment required" in result


async def test_a_platform_agent_is_posted_at_v1_chat_completions_the_route_the_portal_door_prices():
    # 2026-09-18: the review found the buyer posting to
    # `/agents/{name}/chat/completions`, which the portal serves with no
    # handler (a redirect to the profile page), while its agent HTTP door
    # prices only `/v1/chat/completions`.
    class OnList(StubBuyer):
        def allows_url(self, url):
            return httpx.URL(url).host == "robutler.ai"

    buyer = OnList()
    skill = make_skill(transport="http", mpp_buyer=buyer, agent_base_url="https://robutler.ai")
    with patch("webagents.server.context.context_vars.get_context", return_value=None):
        assert await skill.nli_tool(agent="@acme", message="hi") == "bought"
        [c async for c in skill.stream_message("https://robutler.ai/agents/acme", [{"role": "user", "content": "hi"}])]
        [c async for c in skill.stream_message("https://other.example/agents/acme", [{"role": "user", "content": "hi"}])]
    assert [r["url"] for r in buyer.requests] == [
        "https://robutler.ai/agents/acme/v1/chat/completions",
        "https://robutler.ai/agents/acme/v1/chat/completions",
        # A host off the buyer's list keeps the route the Python SDK server serves.
        "https://other.example/agents/acme/chat/completions",
    ]


async def test_the_handoff_stream_goes_through_the_buyer():
    buyer = StubBuyer()
    skill = make_skill(mpp_buyer=buyer)
    with patch("webagents.server.context.context_vars.get_context", return_value=None):
        chunks = [c async for c in skill.stream_message("https://portal.example.com/agents/acme", [{"role": "user", "content": "hi"}])]
    assert chunks == [{"choices": [{"delta": {"content": "bought"}}]}]
    assert buyer.requests[0]["url"] == "https://portal.example.com/agents/acme/chat/completions"


# --------------------------------------------------------------------------
# The in-band purchase on UAMP
# --------------------------------------------------------------------------


async def _run_uamp(skill: NLISkill, events: List[dict], payment_token: Any = None, connects: Any = None):
    ws = FakeSocket([{"type": "session.created", "session_id": "s1"}, *events])

    def connect(url, **kwargs):
        if connects is not None:
            connects.append((url, kwargs))
        return ws

    with patch("webagents.agents.skills.robutler.nli.skill.websockets.connect", side_effect=connect):
        result = await skill._send_via_uamp("@acme", "hi", {}, payment_token, 5.0)
    return result, ws


async def test_the_upgrade_is_signed_through_the_buyer_and_never_beside_a_payment_token():
    # 2026-09-18: nothing signed the upgrade, so the platform's socket door
    # (which acts only on `Signature-Input` and no payment token) never sent
    # this client an in-band `mpp` entry.
    class Signing(StubBuyer):
        def __init__(self):
            super().__init__()
            self.signed: List[str] = []

        async def upgrade_headers(self, url):
            self.signed.append(url)
            return {"signature-input": 'sig1=("@method")', "signature": "sig1=:AA==:"}

    buyer = Signing()
    skill = make_skill(mpp_buyer=buyer)
    skill._auth_token = None
    connects: list = []
    done = [{"type": "response.delta", "delta": {"text": "served"}}, {"type": "response.done"}]
    assert (await _run_uamp(skill, done, connects=connects))[0] == "served"
    assert buyer.signed == ["ws://localhost:2224/agents/acme/uamp"]
    [(url, kwargs)] = connects
    assert url == "ws://localhost:2224/agents/acme/uamp"
    [(name, headers)] = kwargs.items()
    assert name in ("additional_headers", "extra_headers")
    assert headers == {"signature-input": 'sig1=("@method")', "signature": "sig1=:AA==:"}

    connects.clear()
    assert (await _run_uamp(skill, list(done), payment_token="pt_1", connects=connects))[0] == "served"
    assert buyer.signed == ["ws://localhost:2224/agents/acme/uamp"]
    assert connects[0][1] == {}


async def test_hands_the_mpp_entry_to_the_buyer_and_resumes_with_scheme_balance():
    buyer = StubBuyer()
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [
        payment_required(MPP_SCHEMES),
        {"type": "response.delta", "delta": {"text": "served"}},
        {"type": "response.done"},
    ])
    assert result == "served"
    assert buyer.purchases == [(PURCHASE_URL, CHALLENGE, TERMS)]
    submit = ws.sent[-1]
    assert submit["type"] == "payment.submit"
    assert submit["payment"] == {"scheme": "balance", "amount": "0.50"}


async def test_a_refusing_buyer_falls_through_to_the_token_scheme():
    buyer = StubBuyer(purchase_ok=False)
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [
        payment_required(MPP_SCHEMES),
        {"type": "response.delta", "delta": {"text": "served"}},
        {"type": "response.done"},
    ], payment_token="pt_1")
    assert result == "served"
    assert len(buyer.purchases) == 1
    submit = ws.sent[-1]
    assert submit["payment"]["scheme"] == "token"
    assert submit["payment"]["token"] == "pt_1"


async def test_a_refusing_buyer_with_no_token_fails_as_before():
    skill = make_skill(mpp_buyer=StubBuyer(purchase_ok=False))
    result, ws = await _run_uamp(skill, [payment_required(MPP_SCHEMES)])
    assert result is None
    assert not any(frame["type"] == "payment.submit" for frame in ws.sent)


@pytest.mark.parametrize(
    "schemes",
    [
        [{"scheme": "token", "network": "robutler"}],
        [{"scheme": "token"}, {"scheme": "mpp", "challenge": CHALLENGE}],
        [{"scheme": "token"}, {"scheme": "mpp", "challenge": " ", "purchase_url": PURCHASE_URL}],
    ],
)
async def test_without_a_usable_mpp_entry_the_buyer_is_never_asked(schemes):
    buyer = StubBuyer()
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [
        payment_required(schemes),
        {"type": "response.delta", "delta": {"text": "served"}},
        {"type": "response.done"},
    ], payment_token="pt_1")
    assert result == "served"
    assert buyer.purchases == []
    assert ws.sent[-1]["payment"]["scheme"] == "token"


async def test_without_a_buyer_an_mpp_entry_changes_nothing():
    skill = make_skill()
    result, ws = await _run_uamp(skill, [
        payment_required(MPP_SCHEMES),
        {"type": "response.delta", "delta": {"text": "served"}},
        {"type": "response.done"},
    ], payment_token="pt_1")
    assert result == "served"
    assert ws.sent[-1]["payment"]["scheme"] == "token"


# --------------------------------------------------------------------------
# The purchase pointer (2026-09-19): an `mpp` entry with a purchase URL and NO
# challenge, which the platform's `/llm` socket sends a token holder whose
# token ran dry (nobody on that rail was verified, so nothing could be minted
# for them). The skill ignored it.
# --------------------------------------------------------------------------

POINTER_SCHEMES = [{"scheme": "token"}, {"scheme": "mpp", "purchase_url": PURCHASE_URL}]


class PointerBuyer(StubBuyer):
    def __init__(self, pointer_ok: bool = True) -> None:
        super().__init__()
        self.pointers: List[dict] = []
        self.pointer_ok = pointer_ok

    async def purchase_at(self, url, *, source_url, call=None):
        self.pointers.append({"url": url, "source_url": source_url, "call": call})
        if self.pointer_ok:
            return SimpleNamespace(ok=True)
        return SimpleNamespace(ok=False, reason="realm_not_allowed", detail="off the allowlist")


SERVED = [{"type": "response.delta", "delta": {"text": "served"}}, {"type": "response.done"}]


async def test_a_pointer_is_handed_to_purchase_at_with_this_socket_as_its_source_and_no_token_resumes_with_balance():
    buyer = PointerBuyer()
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [payment_required(POINTER_SCHEMES), *SERVED])
    assert result == "served"
    assert buyer.purchases == []
    [pointer] = buyer.pointers
    assert (pointer["url"], pointer["source_url"]) == (PURCHASE_URL, "ws://localhost:2224/agents/acme/uamp")
    assert pointer["call"] is not None
    assert ws.sent[-1]["payment"] == {"scheme": "balance", "amount": "0.50"}


async def test_a_delegate_that_pays_by_token_resumes_on_its_token_once_the_pointer_purchase_is_made():
    buyer = PointerBuyer()
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [payment_required(POINTER_SCHEMES), *SERVED], payment_token="pt_1")
    assert result == "served"
    assert len(buyer.pointers) == 1
    submits = [f for f in ws.sent if f["type"] == "payment.submit"]
    # One submit, the token: `balance` is not a scheme the socket that sends a pointer resumes on.
    assert [f["payment"]["scheme"] for f in submits] == ["token"]
    assert submits[0]["payment"]["token"] == "pt_1"


async def test_a_refused_pointer_falls_through_to_the_token_or_fails_as_before():
    buyer = PointerBuyer(pointer_ok=False)
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [payment_required(POINTER_SCHEMES), *SERVED], payment_token="pt_1")
    assert result == "served"
    assert ws.sent[-1]["payment"]["scheme"] == "token"

    result, ws = await _run_uamp(make_skill(mpp_buyer=PointerBuyer(pointer_ok=False)), [payment_required(POINTER_SCHEMES)])
    assert result is None
    assert not any(frame["type"] == "payment.submit" for frame in ws.sent)


async def test_a_buyer_that_cannot_follow_a_pointer_is_never_handed_one():
    # `StubBuyer` has `purchase` and no `purchase_at`: today's behaviour exactly.
    buyer = StubBuyer()
    skill = make_skill(mpp_buyer=buyer)
    result, ws = await _run_uamp(skill, [payment_required(POINTER_SCHEMES), *SERVED], payment_token="pt_1")
    assert result == "served"
    assert buyer.purchases == []
    assert ws.sent[-1]["payment"]["scheme"] == "token"


async def test_every_purchase_of_one_turn_carries_the_same_call_key_and_a_buyer_without_the_keyword_is_not_handed_it():
    buyer = PointerBuyer()
    skill = make_skill(mpp_buyer=buyer)
    await _run_uamp(skill, [payment_required(POINTER_SCHEMES), payment_required(POINTER_SCHEMES), *SERVED])
    await _run_uamp(skill, [payment_required(POINTER_SCHEMES), *SERVED])
    first, second, third = [p["call"] for p in buyer.pointers]
    assert first is second
    assert third is not first

    class Keyed(StubBuyer):
        async def purchase(self, url, challenge, terms=None, *, call=None):
            self.purchases.append((url, challenge, terms, call))
            return SimpleNamespace(ok=True)

    keyed = Keyed()
    await _run_uamp(make_skill(mpp_buyer=keyed), [payment_required(MPP_SCHEMES), *SERVED])
    assert keyed.purchases[0][3] is not None
    # `StubBuyer.purchase` takes no `call`: handing it one would be a TypeError, not a purchase.
    legacy = StubBuyer()
    assert (await _run_uamp(make_skill(mpp_buyer=legacy), [payment_required(MPP_SCHEMES), *SERVED]))[0] == "served"
    assert legacy.purchases == [(PURCHASE_URL, CHALLENGE, TERMS)]


def test_find_mpp_scheme_reads_a_pointer_as_an_entry_with_no_challenge():
    assert _find_mpp_scheme(POINTER_SCHEMES) == {"challenge": None, "purchase_url": PURCHASE_URL}
    # Malformed is not a pointer: a challenge that is present must be a non-empty string.
    assert _find_mpp_scheme([{"scheme": "mpp", "challenge": " ", "purchase_url": PURCHASE_URL}]) is None
    assert _find_mpp_scheme([{"scheme": "mpp"}]) is None
    assert _find_mpp_scheme([{"scheme": "mpp", "purchase_url": " "}]) is None


def test_find_mpp_scheme_reads_only_a_complete_entry_and_keeps_terms():
    assert _find_mpp_scheme(MPP_SCHEMES) == {"challenge": CHALLENGE, "purchase_url": PURCHASE_URL, "terms": TERMS}
    assert _find_mpp_scheme([{"scheme": "mpp", "challenge": CHALLENGE, "purchase_url": PURCHASE_URL}]) == {
        "challenge": CHALLENGE, "purchase_url": PURCHASE_URL,
    }
    assert _find_mpp_scheme(None) is None
    assert _find_mpp_scheme([{"scheme": "mpp", "challenge": 1, "purchase_url": PURCHASE_URL}]) is None
