"""
Per-call settle keys (wave-0 review finding 5, 2026-09-26): a settle made for
one tool call carries that call's id in its key, so two per-call settles for
one purpose on one lock never derive the same key. Pinned against the shared
fixture's `per_call` section (`fixtures/payments/settle_idempotency.json`),
which the TypeScript suite and the portal read too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from webagents.agents.skills.robutler.payments import PaymentSkill
from webagents.agents.skills.robutler.payments.idempotency import settle_idempotency_key

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "payments" / "settle_idempotency.json").read_text())
VALID = re.compile(FIXTURE["pattern"])
PER_CALL = FIXTURE["per_call"]


def _skill() -> PaymentSkill:
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


def test_every_per_call_vector_derives_and_passes_the_platform_pattern():
    assert PER_CALL["vectors"]
    for v in PER_CALL["vectors"]:
        assert settle_idempotency_key(v["lock_id"], v["purpose"], v["call_id"]) == v["key"]
        assert VALID.match(v["key"])


def test_two_calls_on_one_lock_are_two_keys_and_no_call_id_is_the_lifecycle_key():
    a, b = PER_CALL["vectors"][:2]
    assert a["lock_id"] == b["lock_id"] and a["purpose"] == b["purpose"]
    assert a["key"] != b["key"]
    assert settle_idempotency_key(a["lock_id"], a["purpose"]) == f"settle:{a['lock_id']}:{a['purpose']}"


def test_the_fixtures_invalid_call_ids_are_refused():
    for bad in PER_CALL["invalid_call_ids"]:
        with pytest.raises(ValueError, match="call id"):
            settle_idempotency_key("lock-1", "agent_fee", bad)


async def test_the_skill_derives_a_per_call_key_when_a_settle_names_its_tool_call():
    skill = _skill()
    v = PER_CALL["vectors"][0]
    await skill._settle_payment(v["lock_id"], amount=0.02, charge_type=v["purpose"], call_id=v["call_id"])
    await skill._settle_payment(v["lock_id"], amount=0.02, charge_type=v["purpose"])
    await skill._settle_payment("lock_r", amount=0, release=True, call_id="call_1")
    calls = skill.client.tokens.settle.call_args_list
    assert calls[0].kwargs["idempotency_key"] == v["key"]
    assert calls[1].kwargs["idempotency_key"] == f"settle:{v['lock_id']}:{v['purpose']}"
    # A release never carries a call id: one release per lock.
    assert calls[2].kwargs["idempotency_key"] == "settle:lock_r:release"
