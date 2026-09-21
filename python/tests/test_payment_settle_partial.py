"""
Partial settles (2026-09-18), the Python twin of
`typescript/tests/unit/skills/payments/settle-result.test.ts`. Since the
portal's S-148 fix a short lock is charged what it holds, and
`POST /api/payments/settle` answers `success: true` with `partial`, and, when
partial, `unbilled` and `requested`. The SDK read only `success`, so a partial
settle was reported as charged in full. These cases pin the reader and its two
callers: the payment skill's finalize and the x402 skill's settle.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("robutler")

from webagents.agents.skills.robutler.payments import PaymentContext, PaymentSkill
from webagents.agents.skills.robutler.payments.settle_result import read_settle_result

FULL = {"success": True, "partial": False, "charged": "5000000", "chargedDollars": 0.005}
PARTIAL = {
    "success": True,
    "partial": True,
    "charged": "3000000",
    "chargedDollars": 0.003,
    "unbilled": "2000000",
    "unbilledDollars": 0.002,
    "requested": "5000000",
    "requestedDollars": 0.005,
}


class Ctx:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, getattr(self, key, default) if key != "_data" else default)

    def set(self, key, value):
        self._data[key] = value


def test_the_reader_keeps_partial_charged_and_unbilled_and_warns_on_a_partial_only():
    logger = Mock()
    assert read_settle_result(FULL, "t", logger)["partial"] is False
    logger.warning.assert_not_called()
    r = read_settle_result(PARTIAL, "t", logger)
    assert (r["success"], r["partial"], r["unbilled"], r["unbilledDollars"], r["requested"]) == (True, True, "2000000", 0.002, "5000000")
    logger.warning.assert_called_once()
    assert "PARTIAL settle, charged 0.003 of 0.005 (unbilled 0.002)" in logger.warning.call_args.args[0]


def test_an_older_platform_reads_as_not_partial_and_a_failure_is_never_partial():
    assert read_settle_result({"success": True, "charged": "1"}, "t", Mock())["partial"] is False
    failed = read_settle_result({"success": False, "partial": True, "error": "x"}, "t", Mock())
    assert (failed["success"], failed["partial"], failed["error"]) == (False, False, "x")
    assert read_settle_result(None, "t", Mock()) == {"success": False, "partial": False}


def _skill(settle_answers):
    client = Mock()
    client.tokens = Mock()
    client.tokens.settle = AsyncMock(side_effect=list(settle_answers))
    with patch("webagents.agents.skills.robutler.payments.skill.RobutlerClient") as cls:
        cls.return_value = client
        skill = PaymentSkill({"enable_billing": True, "webagents_api_url": "http://test.localhost", "robutler_api_key": "k"})
    skill.logger = Mock()
    skill.client = client
    skill.agent = Mock(name="agent")
    return skill


@pytest.mark.asyncio
async def test_finalize_marks_the_run_partial_with_what_went_unbilled():
    skill = _skill([PARTIAL, FULL])
    ctx = Ctx()
    ctx.payments = PaymentContext(payment_token="pt", lock_id="lock_1", locked_amount_dollars=0.005)
    ctx.usage = [{"type": "llm", "model": "m", "prompt_tokens": 1, "completion_tokens": 1}]
    await skill.finalize_payment(ctx)
    assert ctx.payments.settle_partial is True
    assert ctx.payments.unbilled_dollars == pytest.approx(0.002)
    assert any("PARTIAL" in str(c.args[0]) for c in skill.logger.warning.call_args_list)

    full = _skill([FULL, FULL])
    ctx2 = Ctx()
    ctx2.payments = PaymentContext(payment_token="pt", lock_id="lock_2", locked_amount_dollars=0.005)
    ctx2.usage = [{"type": "llm", "model": "m"}]
    await full.finalize_payment(ctx2)
    assert ctx2.payments.settle_partial is False


@pytest.mark.asyncio
async def test_the_x402_settle_reports_a_partial_and_never_as_the_full_amount():
    from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402

    skill = PaymentSkillX402.__new__(PaymentSkillX402)
    skill.logger = Mock()
    skill._jwks_manager = None
    skill.agent = Mock(id="agent-1")
    skill.client = Mock()
    skill.client.facilitator = Mock()
    skill.client.facilitator.verify = AsyncMock(return_value={"isValid": True, "balance": 10})
    skill.client.facilitator.settle = AsyncMock(return_value=PARTIAL)
    endpoint = Mock()
    endpoint._webagents_pricing = {"credits_per_call": 0.005, "reason": "api"}
    with patch("webagents.agents.skills.robutler.payments_x402.skill.decode_payment_header", return_value={"scheme": "token", "network": "robutler"}), patch(
        "webagents.agents.skills.robutler.payments_x402.skill.extract_token_from_payment", return_value="tok"
    ):
        result = await skill._process_x402_payment("hdr", Ctx(), endpoint)
    assert (result["partial"], result["unbilledDollars"]) == (True, 0.002)
    assert any("PARTIALLY" in str(c.args[0]) for c in skill.logger.warning.call_args_list)
    skill.logger.info.assert_not_called()
