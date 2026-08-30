# PARTIALLY REMOVED (M4 suite triage): the cases that pinned the retired
# litellm/byok settle architecture (LITELLM_AVAILABLE flag,
# PaymentContext.byok_providers, is_byok/byok_llm settles) are gone; the
# current payment flow is covered by test_payment_transport_agnostic.py,
# test_completions_payment_preflight.py and the tests in this file.
#
# The triage over-reached and has been walked back: cases that were NOT about
# the retired architecture are restored below, each marked RESTORED with what
# (if anything) had to be adapted to the current code.
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

    # [removed: test_non_byok_single_settle — see M4 triage note at top]
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

    # [removed: test_byok_separate_settle — see M4 triage note at top]
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
        """Non-zero usage but no lock → error handled, no settle, no crash.

        RESTORED (this is non-BYOK finalize coverage of a live code path, not
        the retired settle architecture). Adapted in one respect: the original
        patched `cost_per_token`, which no longer exists — cost is computed
        server-side from the forwarded usage records now — so the patch is
        dropped and the assertion is on the observable behaviour instead.
        """
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

        # Should not crash even though there is no lock_id — the error is caught.
        result = await skill.finalize_payment(context)

        assert result is context
        client.tokens.settle.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_llm_records_aggregated(self):
        """Multiple usage records go out in ONE settle call, none dropped.

        RESTORED. Adapted: the original asserted the CLIENT-side sum of
        `cost_per_token`; the skill now forwards the raw records and the
        server prices them, so the invariant that still matters — one settle
        carrying every record — is what is asserted.
        """
        client = _make_client()
        skill = _make_payment_skill(client)

        payment_ctx = PaymentContext(
            payment_token='jwt.token.here',
            lock_id='lock-multi',
            locked_amount_dollars=0.50,
        )
        usage = [
            {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 200, 'completion_tokens': 100},
            {'type': 'tool', 'tool_name': 'search', 'pricing': {'credits': 0.003, 'reason': 'search'}},
        ]
        context = FakeContext(payments=payment_ctx, is_byok=False, usage=list(usage))

        await skill.finalize_payment(context)

        settle_calls = client.tokens.settle.call_args_list
        non_release = [c for c in settle_calls if not c.kwargs.get('release', False)]
        assert len(non_release) == 1, (
            f"Multiple usage records should go out in one settle, got {len(non_release)}"
        )
        forwarded = non_release[0].kwargs['usage']
        assert forwarded == usage, "every usage record must reach the settle call"

        release_calls = [c for c in settle_calls if c.kwargs.get('release', False)]
        assert len(release_calls) == 1, "the remaining lock balance must be released"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
