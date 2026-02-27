"""
Tests for simplified PaymentSkill finalize flow.

Verifies that finalize_payment uses a single settle call for non-BYOK usage
(commission distribution handled server-side) vs separate byok_llm settle for BYOK.
"""

import pytest

try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

from unittest.mock import Mock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from webagents.agents.skills.robutler.payments.skill import PaymentSkill, PaymentContext
from robutler.api import RobutlerClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal context for testing finalize_payment."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)


def _make_payment_skill(mock_client: Mock) -> PaymentSkill:
    """Instantiate PaymentSkill with billing enabled and a mocked client."""
    config = {
        'enable_billing': True,
        'minimum_balance': 0.01,
        'webagents_api_url': 'http://test.localhost',
        'robutler_api_key': 'rok_testapikey',
    }
    skill = PaymentSkill(config)
    skill.logger = Mock()
    skill.client = mock_client
    skill.agent = Mock(name='test-agent')
    return skill


def _make_client() -> Mock:
    """Create a mock RobutlerClient with tokens resource."""
    client = Mock(spec=RobutlerClient)
    client.tokens = Mock()
    client.tokens.settle = AsyncMock(return_value={'success': True})
    client.tokens.lock = AsyncMock(return_value={'lockId': 'lock-1', 'lockedAmountDollars': 0.005})
    client.tokens.extend_lock = AsyncMock(return_value={'success': True})
    return client


# ---------------------------------------------------------------------------
# Tests: Simplified single-settle for non-BYOK
# ---------------------------------------------------------------------------

class TestFinalizePaymentSimplified:
    """Verify the simplified finalize flow makes a single settle call."""

    @pytest.mark.asyncio
    async def test_non_byok_single_settle(self):
        """Non-BYOK: one settle call for combined LLM + tool costs."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-abc',
            locked_amount_dollars=0.10,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 200, 'completion_tokens': 80},
                {'type': 'tool', 'tool_name': 'weather', 'pricing': {'credits': 0.001, 'reason': 'API call'}},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000030, 0.000048)  # prompt_cost, completion_cost

            await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        # Non-release settle calls (amount > 0) should be exactly 1
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]
        assert len(non_release) == 1, (
            f"Expected 1 non-release settle call for non-BYOK, got {len(non_release)}: {non_release}"
        )

        call_kwargs = non_release[0].kwargs
        assert call_kwargs['lock_id'] == 'lock-abc'
        charged = call_kwargs['amount']
        # LLM: 0.000030 + 0.000048 = 0.000078
        # Tool: 0.001
        # Total: 0.001078
        assert abs(charged - 0.001078) < 0.0001, f"Charged amount {charged} != ~0.001078"
        assert call_kwargs.get('charge_type') is None, "Non-BYOK settle should not set charge_type"

    @pytest.mark.asyncio
    async def test_non_byok_release_after_settle(self):
        """Non-BYOK: a release call follows the settle."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-abc',
            locked_amount_dollars=0.10,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000015, 0.000030)

            await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        release_calls = [c for c in settle_calls if c.kwargs.get('release', False)]
        assert len(release_calls) >= 1, "Expected at least one release call after settle"

    @pytest.mark.asyncio
    async def test_non_byok_zero_cost_releases_lock(self):
        """Non-BYOK: zero cost skips settle but still releases the lock."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-abc',
            locked_amount_dollars=0.005,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[],
        )

        await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]
        assert len(non_release) == 0, "Zero cost should not call non-release settle"
        release_calls = [c for c in settle_calls if c.kwargs.get('release', False)]
        assert len(release_calls) == 1, "Should release the lock even when cost is 0"


# ---------------------------------------------------------------------------
# Tests: BYOK flow still uses separate byok_llm settle
# ---------------------------------------------------------------------------

class TestFinalizeBYOK:
    """Verify BYOK path makes a separate byok_llm settle."""

    @pytest.mark.asyncio
    async def test_byok_separate_settle(self):
        """BYOK: LLM cost settles as byok_llm, then tool costs settle separately."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-byok',
            locked_amount_dollars=0.10,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=True,
            byok_provider_key_id='pk-123',
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 300, 'completion_tokens': 100},
                {'type': 'tool', 'tool_name': 'weather', 'pricing': {'credits': 0.002, 'reason': 'API'}},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000045, 0.000060)

            await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]

        # Expect 2 non-release settles: byok_llm + tool costs
        assert len(non_release) == 2, (
            f"Expected 2 non-release settle calls for BYOK, got {len(non_release)}"
        )

        byok_call = non_release[0].kwargs
        assert byok_call['charge_type'] == 'byok_llm'
        assert byok_call.get('provider_key_id') == 'pk-123'
        llm_cost = byok_call['amount']
        assert abs(llm_cost - 0.000105) < 0.0001

        tool_call = non_release[1].kwargs
        assert tool_call.get('charge_type') is None
        assert abs(tool_call['amount'] - 0.002) < 0.0001

    @pytest.mark.asyncio
    async def test_byok_llm_only_no_tool_costs(self):
        """BYOK with LLM only: single byok_llm settle, no tool settle."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-byok-2',
            locked_amount_dollars=0.05,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=True,
            byok_provider_key_id='pk-456',
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000015, 0.000030)

            await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]

        assert len(non_release) == 1
        assert non_release[0].kwargs['charge_type'] == 'byok_llm'

    @pytest.mark.asyncio
    async def test_byok_tools_only_no_llm(self):
        """BYOK with tools only (no LLM usage): single non-typed settle."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-byok-3',
            locked_amount_dollars=0.05,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=True,
            usage=[
                {'type': 'tool', 'tool_name': 'search', 'pricing': {'credits': 0.005, 'reason': 'search'}},
            ],
        )

        await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]

        # is_byok but llm_cost=0 → goes to else branch → single settle
        assert len(non_release) == 1
        assert non_release[0].kwargs.get('charge_type') is None


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestFinalizeEdgeCases:
    """Edge cases for finalize_payment."""

    @pytest.mark.asyncio
    async def test_no_payment_context_returns_early(self):
        """No payment context → returns context unchanged."""
        client = _make_client()
        skill = _make_payment_skill(client)

        context = FakeContext(usage=[])

        result = await skill.finalize_payment(context)

        assert result is context
        client.tokens.settle.assert_not_called()

    @pytest.mark.asyncio
    async def test_billing_disabled_skips_settlement(self):
        """Billing disabled → no settle calls."""
        client = _make_client()
        config = {
            'enable_billing': False,
            'webagents_api_url': 'http://test.localhost',
            'robutler_api_key': 'rok_testapikey',
        }
        skill = PaymentSkill(config)
        skill.logger = Mock()
        skill.client = client
        skill.agent = Mock(name='test-agent')

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-abc',
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            ],
        )

        await skill.finalize_payment(context)

        client.tokens.settle.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_lock_id_raises(self):
        """Non-zero cost but no lock → error logged, does not crash."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id=None,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000015, 0.000030)

            # Should not crash even though no lock_id — error is caught
            result = await skill.finalize_payment(context)

        assert result is context

    @pytest.mark.asyncio
    async def test_multiple_llm_records_aggregated(self):
        """Multiple LLM usage records are summed into one settle call."""
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-multi',
            locked_amount_dollars=0.50,
        )
        context = FakeContext(
            payments=payment_ctx,
            is_byok=False,
            usage=[
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
                {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 200, 'completion_tokens': 100},
                {'type': 'tool', 'tool_name': 'search', 'pricing': {'credits': 0.003, 'reason': 'search'}},
            ],
        )

        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000020, 0.000040)

            await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]

        assert len(non_release) == 1, (
            f"Multiple LLM records should aggregate into one settle, got {len(non_release)}"
        )

        charged = non_release[0].kwargs['amount']
        # 2 LLM calls: 2 * (0.000020 + 0.000040) = 0.000120
        # 1 tool: 0.003
        # Total: 0.003120
        assert abs(charged - 0.003120) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
