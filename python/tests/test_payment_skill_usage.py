"""
Tests for PaymentSkill usage-forwarding behavior.

Validates that PaymentSkill forwards raw usage records to _settle_payment
instead of computing costs locally. Server-side cost computation via
MODEL_PRICING is now canonical.
"""

import pytest
try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

import inspect
from unittest.mock import Mock, AsyncMock, patch

from webagents.agents.skills.robutler.payments import PaymentSkill, PaymentContext
from robutler.api import RobutlerClient


class MockContext:
    """Minimal mock context for PaymentSkill tests."""
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        if key in self._data:
            return self._data[key]
        if hasattr(self, key) and key != '_data':
            return getattr(self, key)
        return default

    def set(self, key, value):
        self._data[key] = value


@pytest.fixture
def mock_webagents_client():
    client = Mock(spec=RobutlerClient)
    client._make_request = AsyncMock()
    client.health_check = AsyncMock()
    client.close = AsyncMock()
    client.tokens = Mock()
    client.tokens.validate = AsyncMock(return_value=True)
    client.tokens.validate_with_balance = AsyncMock(return_value={'valid': True, 'balance': 10.0})
    client.tokens.redeem = AsyncMock(return_value=True)
    client.tokens.lock = AsyncMock(return_value={'lockId': 'lock_test_123', 'lockedAmountDollars': 0.005})
    client.tokens.settle = AsyncMock(return_value={'success': True, 'chargedDollars': 0.003})
    client.tokens.extend_lock = AsyncMock(return_value={'success': True})
    return client


@pytest.fixture
def payment_skill(mock_webagents_client):
    config = {
        'enable_billing': True,
        'minimum_balance': 5.0,
        'per_message_lock': 0.01,
        'default_tool_lock': 0.25,
        'webagents_api_url': 'http://test.localhost',
        'robutler_api_key': 'test_api_key',
    }
    with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_cls:
        mock_cls.return_value = mock_webagents_client
        skill = PaymentSkill(config)
        skill.logger = Mock()
        skill.client = mock_webagents_client
        skill.agent = Mock(name='test-agent')
        return skill


class TestFinalizeForwardsUsageRecords:
    """finalize_payment must send raw usage records to _settle_payment."""

    @pytest.mark.asyncio
    async def test_finalize_sends_usage_array_for_llm(self, payment_skill, mock_webagents_client):
        """Usage records with type='llm' are forwarded as-is to settle."""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='pt_test', lock_id='lock_1', locked_amount_dollars=0.05,
        )
        context.usage = [
            {'type': 'llm', 'model': 'xai/grok-3', 'prompt_tokens': 200, 'completion_tokens': 80},
        ]

        await payment_skill.finalize_payment(context)

        settle_calls = mock_webagents_client.tokens.settle.call_args_list
        usage_sent = False
        for call in settle_calls:
            if call.kwargs.get('usage') or (call.args and isinstance(call.args[0], list)):
                usage_sent = True
                usage_arg = call.kwargs.get('usage', call.args[0] if call.args else None)
                assert any(r['model'] == 'xai/grok-3' for r in usage_arg)
                break
        assert usage_sent, "Expected settle to be called with usage= parameter"

    @pytest.mark.asyncio
    async def test_finalize_sends_usage_array_for_byok(self, payment_skill, mock_webagents_client):
        """BYOK scenario still forwards raw usage records."""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='pt_test', lock_id='lock_1', locked_amount_dollars=0.05,
        )
        context.usage = [
            {'type': 'llm', 'model': 'xai/grok-3', 'prompt_tokens': 100, 'completion_tokens': 50},
        ]
        context.is_byok = True
        context.byok_provider_key_id = 'key_abc'

        await payment_skill.finalize_payment(context)

        settle_calls = mock_webagents_client.tokens.settle.call_args_list
        byok_call = [c for c in settle_calls if c.kwargs.get('charge_type') == 'byok_llm']
        assert len(byok_call) >= 1, "Expected at least one settle with charge_type='byok_llm'"
        assert byok_call[0].kwargs.get('usage') is not None

    @pytest.mark.asyncio
    async def test_finalize_payment_successful_flag(self, payment_skill, mock_webagents_client):
        """payment_successful is set to True after settle succeeds."""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='pt_test', lock_id='lock_1', locked_amount_dollars=0.05,
        )
        context.usage = [
            {'type': 'llm', 'model': 'fireworks/deepseek-v3p2', 'prompt_tokens': 500, 'completion_tokens': 200},
        ]

        await payment_skill.finalize_payment(context)

        assert context.payments.payment_successful is True


class TestSettlePaymentParameters:
    """_settle_payment passes usage or amount correctly to the API client."""

    @pytest.mark.asyncio
    async def test_settle_passes_usage_kwarg(self, payment_skill, mock_webagents_client):
        """When usage is provided, it is forwarded as 'usage' to client.tokens.settle."""
        usage_records = [
            {'type': 'llm', 'model': 'xai/grok-3', 'prompt_tokens': 100, 'completion_tokens': 50},
        ]

        await payment_skill._settle_payment('lock_abc', usage=usage_records, description='test')

        mock_webagents_client.tokens.settle.assert_called_once()
        call_kwargs = mock_webagents_client.tokens.settle.call_args.kwargs
        assert 'usage' in call_kwargs
        assert call_kwargs['usage'] == usage_records
        assert 'amount' not in call_kwargs

    @pytest.mark.asyncio
    async def test_settle_passes_amount_kwarg(self, payment_skill, mock_webagents_client):
        """When only amount is provided (no usage), it is passed as 'amount'."""
        await payment_skill._settle_payment('lock_abc', amount=0.05, description='flat rate')

        call_kwargs = mock_webagents_client.tokens.settle.call_args.kwargs
        assert call_kwargs['amount'] == 0.05
        assert 'usage' not in call_kwargs

    @pytest.mark.asyncio
    async def test_settle_passes_both_when_given(self, payment_skill, mock_webagents_client):
        """When both usage and amount are provided, both are forwarded."""
        usage_records = [
            {'type': 'llm', 'model': 'xai/grok-3', 'prompt_tokens': 50, 'completion_tokens': 25},
        ]

        await payment_skill._settle_payment('lock_abc', amount=0.01, usage=usage_records, description='both')

        call_kwargs = mock_webagents_client.tokens.settle.call_args.kwargs
        assert call_kwargs['amount'] == 0.01
        assert call_kwargs['usage'] == usage_records

    @pytest.mark.asyncio
    async def test_settle_release_lock(self, payment_skill, mock_webagents_client):
        """release=True is forwarded to the settle call."""
        await payment_skill._settle_payment('lock_abc', amount=0, release=True)

        call_kwargs = mock_webagents_client.tokens.settle.call_args.kwargs
        assert call_kwargs['release'] is True


class TestNoLiteLLMImport:
    """PaymentSkill must NOT import or reference litellm / cost_per_token."""

    def test_no_litellm_in_source(self):
        """The skill module source should not reference litellm or cost_per_token."""
        import webagents.agents.skills.robutler.payments.skill as skill_mod
        source = inspect.getsource(skill_mod)
        assert 'import litellm' not in source
        assert 'from litellm' not in source
        assert 'cost_per_token' not in source

    def test_no_litellm_available_flag(self):
        """LITELLM_AVAILABLE should not be defined in the skill module."""
        import webagents.agents.skills.robutler.payments.skill as skill_mod
        assert not hasattr(skill_mod, 'LITELLM_AVAILABLE')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
