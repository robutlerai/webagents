"""
Every settle carries an Idempotency-Key (2026-09-26), pinned against the
fixture both SDKs and the portal read (`fixtures/payments/settle_idempotency.json`):
the header and body names, the stable derivation `settle:<lock>:<purpose>` for
the settles the skill's lifecycle names, a fresh `settle:<scope>:<uuid>` for
the ones nothing names, the client's retry policy (a POST is sent again after
a 5xx or a dropped connection ONLY when it carries the key, with the same key),
and the reader keeping `replayed`.

Also pinned here, because the Python suite had no explicit pin for two of the
HEAD billing fixes (`CHANGELOG.md`, Unreleased): a tool's fee is charged once,
as a usage record in the one settle at finalize, never per tool; and usage
reaches the wire in the snake_case the portal's `usageRecordSchema` reads.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from webagents.agents.skills.robutler.api import RobutlerClient
from webagents.agents.skills.robutler.api.types import ApiResponse
from webagents.agents.skills.robutler.payments import PaymentContext, PaymentSkill
from webagents.agents.skills.robutler.payments.idempotency import (
    IDEMPOTENCY_KEY_BODY_FIELD,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENT_REPLAYED_HEADER,
    fresh_settle_idempotency_key,
    settle_idempotency_key,
)
from webagents.agents.skills.robutler.payments.settle_result import read_settle_result

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "settle_idempotency.json").read_text())
VALID = re.compile(FIXTURE["pattern"])
FRESH = re.compile(FIXTURE["fresh"]["pattern"])


class Ctx:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        if key in self._data:
            return self._data[key]
        if hasattr(self, key) and key != "_data":
            return getattr(self, key)
        return default

    def set(self, key, value):
        self._data[key] = value


def _skill(client=None) -> PaymentSkill:
    """A skill over a Mock client (settle answers success), or over the REAL client passed in."""
    if client is None:
        client = Mock()
        client.tokens = Mock()
        client.tokens.settle = AsyncMock(return_value={"success": True, "chargedDollars": 0.003})
    with patch("webagents.agents.skills.robutler.payments.skill.RobutlerClient") as cls:
        cls.return_value = client
        skill = PaymentSkill({"enable_billing": True, "webagents_api_url": "http://test.localhost", "robutler_api_key": "k"})
    skill.logger = Mock()
    skill.client = client
    skill.agent = Mock(name="agent")
    return skill


# ---------------------------------------------------------------------------
# The fixture pins the names and the derivation
# ---------------------------------------------------------------------------


def test_the_names_match_the_fixture():
    assert IDEMPOTENCY_KEY_HEADER == FIXTURE["header"]
    assert IDEMPOTENCY_KEY_BODY_FIELD == FIXTURE["body_field"]
    assert IDEMPOTENT_REPLAYED_HEADER == FIXTURE["replayed_header"]
    assert IDEMPOTENCY_KEY_MAX_LENGTH == FIXTURE["max_length"]


def test_every_stable_vector_derives_and_nothing_else_does():
    for v in FIXTURE["stable"]["vectors"]:
        assert settle_idempotency_key(v["lock_id"], v["purpose"]) == v["key"]
        assert VALID.match(v["key"])
    with pytest.raises(ValueError, match="lock id"):
        settle_idempotency_key("", "usage")
    with pytest.raises(ValueError, match="purpose"):
        settle_idempotency_key("lock-1", "has space")


def test_a_fresh_key_is_minted_per_call_in_the_fixture_shape():
    for scope in FIXTURE["fresh"]["scopes"]:
        a, b = fresh_settle_idempotency_key(scope), fresh_settle_idempotency_key(scope)
        assert FRESH.match(a) and VALID.match(a)
        assert a.startswith(f"settle:{scope}:")
        assert a != b


def test_the_fixtures_invalid_keys_fail_the_platform_pattern():
    for bad in FIXTURE["invalid_keys"]:
        assert not VALID.match(bad)


# ---------------------------------------------------------------------------
# The skill derives a stable key for every settle it issues
# ---------------------------------------------------------------------------


async def test_finalize_settles_usage_and_releases_with_stable_keys_and_a_repeat_reuses_them():
    skill = _skill()
    ctx = Ctx()
    ctx.payments = PaymentContext(payment_token="pt", lock_id="lock_fin", locked_amount_dollars=0.01)
    ctx.usage = [{"type": "llm", "model": "m", "prompt_tokens": 3, "completion_tokens": 2}]

    await skill.finalize_payment(ctx)
    first = [c.kwargs.get("idempotency_key") for c in skill.client.tokens.settle.call_args_list]
    assert first == ["settle:lock_fin:usage", "settle:lock_fin:release"]
    for k in first:
        assert VALID.match(k)

    # The run loop can finalize the same context twice on its error paths
    # (base_agent.py): the same keys, so the platform replays instead of charging.
    skill.client.tokens.settle.reset_mock()
    await skill.finalize_payment(ctx)
    second = [c.kwargs.get("idempotency_key") for c in skill.client.tokens.settle.call_args_list]
    assert second == first


async def test_two_locks_never_share_a_key():
    skill = _skill()
    for lock in ("lock_a", "lock_b"):
        ctx = Ctx()
        ctx.payments = PaymentContext(payment_token="pt", lock_id=lock)
        ctx.usage = [{"type": "llm", "model": "m", "prompt_tokens": 1, "completion_tokens": 1}]
        await skill.finalize_payment(ctx)
    keys = [c.kwargs.get("idempotency_key") for c in skill.client.tokens.settle.call_args_list]
    assert len(set(keys)) == len(keys) == 4


async def test_the_derivation_table_and_the_explicit_override():
    skill = _skill()
    await skill._settle_payment("lock_x", amount=0.02, charge_type="agent_fee")
    await skill._settle_payment("lock_x", amount=0, release=True)
    await skill._settle_payment("lock_x", usage=[{"type": "tool", "pricing": {"credits": 0.01}}])
    await skill._settle_payment("lock_x", amount=0.02, idempotency_key="settle:lock_x:agent_fee")
    await skill._settle_payment("lock_x", amount=0.02)
    calls = skill.client.tokens.settle.call_args_list
    assert calls[0].kwargs["idempotency_key"] == "settle:lock_x:agent_fee"
    assert calls[1].kwargs["idempotency_key"] == "settle:lock_x:release"
    assert calls[2].kwargs["idempotency_key"] == "settle:lock_x:usage"
    assert calls[3].kwargs["idempotency_key"] == "settle:lock_x:agent_fee"
    # An amount with nothing to name it: the skill passes no key and the
    # client mints one for the call (pinned below), so two calls are two settles.
    assert "idempotency_key" not in calls[4].kwargs


# ---------------------------------------------------------------------------
# HEAD billing fixes the Python suite had no explicit pin for
# ---------------------------------------------------------------------------


async def test_a_tool_fee_is_charged_once_as_a_usage_record_at_finalize_never_per_tool():
    skill = _skill()
    ctx = Ctx()
    ctx.payments = PaymentContext(payment_token="pt", lock_id="lock_tool", locked_amount_dollars=0.5)
    ctx.set("tool_result", "Weather for Paris: sunny")
    await skill.handle_tool_completion(ctx)
    skill.client.tokens.settle.assert_not_called()

    ctx.usage = [{"type": "tool", "tool_name": "get_weather", "pricing": {"credits": 0.05, "reason": "Weather lookup"}}]
    await skill.finalize_payment(ctx)
    charging = [c for c in skill.client.tokens.settle.call_args_list if not c.kwargs.get("release")]
    assert len(charging) == 1
    assert charging[0].kwargs["usage"] == ctx.usage
    assert "amount" not in charging[0].kwargs
    assert charging[0].kwargs.get("charge_type") is None


async def test_usage_reaches_the_wire_in_snake_case_with_the_key_in_header_and_body():
    client = RobutlerClient(api_key="rok_test", base_url="https://platform.example")
    client._make_request = AsyncMock(return_value=ApiResponse(success=True, data={"success": True, "charged": "7", "chargedDollars": 0.000000007}))
    skill = _skill(client)
    ctx = Ctx()
    ctx.payments = PaymentContext(payment_token="pt", lock_id="lock_wire")
    ctx.usage = [{"type": "llm", "model": "xai/grok-3", "prompt_tokens": 200, "completion_tokens": 80, "cached_read_tokens": 10}]

    await skill.finalize_payment(ctx)

    settle_call = client._make_request.call_args_list[0]
    assert settle_call.args[:2] == ("POST", "/payments/settle")
    body = settle_call.kwargs["data"]
    assert body["usage"] == ctx.usage
    assert set(body["usage"][0]) >= {"prompt_tokens", "completion_tokens", "cached_read_tokens"}
    assert body[IDEMPOTENCY_KEY_BODY_FIELD] == "settle:lock_wire:usage"
    assert settle_call.kwargs["headers"] == {IDEMPOTENCY_KEY_HEADER: "settle:lock_wire:usage"}


# ---------------------------------------------------------------------------
# The client: every settle carries a key, and only a keyed POST is retried
# ---------------------------------------------------------------------------


def _client(**kwargs) -> RobutlerClient:
    return RobutlerClient(api_key="rok_test", base_url="https://platform.example", **kwargs)


async def test_the_client_mints_a_fresh_key_when_the_caller_names_none():
    client = _client()
    client._make_request = AsyncMock(return_value=ApiResponse(success=True, data={"success": True}))
    await client.tokens.settle("L1", amount=0.003)
    await client.tokens.settle("L1", amount=0.003)
    keys = [c.kwargs["data"][IDEMPOTENCY_KEY_BODY_FIELD] for c in client._make_request.call_args_list]
    headers = [c.kwargs["headers"][IDEMPOTENCY_KEY_HEADER] for c in client._make_request.call_args_list]
    assert keys == headers
    assert all(FRESH.match(k) and k.startswith("settle:client:") for k in keys)
    assert keys[0] != keys[1]


async def test_the_client_sends_the_callers_key_and_reads_replayed():
    client = _client()
    client._make_request = AsyncMock(return_value=ApiResponse(success=True, data={"success": True, "charged": "5", "replayed": True}))
    result = await client.tokens.settle("L1", usage=[{"type": "llm", "model": "m", "prompt_tokens": 1}], idempotency_key="settle:L1:usage")
    call = client._make_request.call_args
    assert call.kwargs["data"][IDEMPOTENCY_KEY_BODY_FIELD] == "settle:L1:usage"
    assert call.kwargs["headers"] == {IDEMPOTENCY_KEY_HEADER: "settle:L1:usage"}
    assert result["replayed"] is True


async def test_redeem_mints_a_fresh_key_per_call():
    client = _client()
    client._make_request = AsyncMock(return_value=ApiResponse(success=True, data={"success": True}))
    await client.tokens.redeem("a.b.c", 0.01)
    await client.tokens.redeem("a.b.c", 0.01)
    keys = [c.kwargs["data"][IDEMPOTENCY_KEY_BODY_FIELD] for c in client._make_request.call_args_list]
    assert all(FRESH.match(k) and k.startswith("settle:redeem:") for k in keys)
    assert keys[0] != keys[1]
    assert [c.kwargs["headers"][IDEMPOTENCY_KEY_HEADER] for c in client._make_request.call_args_list] == keys


class _Session:
    """Answers each request with the next outcome: a status, or an exception; records what was sent."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, dict(kwargs.get("headers") or {}), kwargs.get("json")))
        outcome = self.outcomes.pop(0)

        class _Exchange:
            async def __aenter__(self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return SimpleNamespace(status=outcome, text=AsyncMock(return_value='{"success": true}'))

            async def __aexit__(self, *exc):
                return False

        return _Exchange()


@pytest.fixture
def exchanges(monkeypatch):
    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    def attach(*outcomes):
        session = _Session(*outcomes)
        client = _client()
        client._get_session = AsyncMock(return_value=session)
        return client, session

    return attach


KEYED = {IDEMPOTENCY_KEY_HEADER: "settle:L1:usage"}


async def test_a_keyed_settle_is_sent_again_after_a_5xx_with_the_same_key(exchanges):
    client, session = exchanges(502, 200)
    response = await client._make_request("POST", "/payments/settle", data={"lockId": "L1"}, headers=KEYED)
    assert response.success is True
    assert [m for m, _, _ in session.calls] == ["POST", "POST"]
    assert all(h[IDEMPOTENCY_KEY_HEADER] == "settle:L1:usage" for _, h, _ in session.calls)


async def test_a_keyed_settle_is_sent_again_after_a_dropped_connection_with_the_same_key(exchanges):
    client, session = exchanges(aiohttp.ServerDisconnectedError(), 200)
    response = await client._make_request("POST", "/payments/settle", data={"lockId": "L1"}, headers=KEYED)
    assert response.success is True
    assert [m for m, _, _ in session.calls] == ["POST", "POST"]
    assert session.calls[0][1][IDEMPOTENCY_KEY_HEADER] == session.calls[1][1][IDEMPOTENCY_KEY_HEADER]


async def test_a_keyless_post_is_still_sent_once_after_a_5xx_or_a_dropped_connection(exchanges):
    client, session = exchanges(502, 200)
    assert (await client._make_request("POST", "/payments/lock", data={"amount": 1})).success is False
    assert [m for m, _, _ in session.calls] == ["POST"]
    client, session = exchanges(aiohttp.ServerDisconnectedError(), 200)
    assert (await client._make_request("POST", "/payments/settle", data={"lockId": "L1"})).success is False
    assert [m for m, _, _ in session.calls] == ["POST"]


async def test_a_keyed_patch_is_not_a_keyed_post(exchanges):
    client, session = exchanges(502, 200)
    response = await client._make_request("PATCH", "/payments/lock/L1", data={"additionalAmount": 1}, headers=KEYED)
    assert response.success is False
    assert [m for m, _, _ in session.calls] == ["PATCH"]


async def test_the_settle_resource_never_sends_a_post_without_the_key(exchanges):
    client, session = exchanges(200)
    await client.tokens.settle("L1", amount=0.003)
    method, headers, body = session.calls[0]
    assert method == "POST"
    assert headers[IDEMPOTENCY_KEY_HEADER] == body[IDEMPOTENCY_KEY_BODY_FIELD]
    assert VALID.match(headers[IDEMPOTENCY_KEY_HEADER])


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def test_the_reader_keeps_replayed_on_a_success_and_never_on_a_failure():
    assert read_settle_result({"success": True, "charged": "5", "replayed": True}, "t", Mock())["replayed"] is True
    assert "replayed" not in read_settle_result({"success": True, "charged": "5"}, "t", Mock())
    assert read_settle_result({"success": False, "replayed": True, "error": "x"}, "t", Mock())["replayed"] is False
