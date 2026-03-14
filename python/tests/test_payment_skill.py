"""
Unit and Integration Tests for PaymentSkill

Tests payment token extraction, payment context setup (verify + lock),
settlement, 402 error handling, and @pricing decorator integration.

Updated for V2.0 lock/settle API.
"""

import json
import pytest
try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal

# Import PaymentSkill and related classes
from webagents.agents.skills.robutler.payments import (
    PaymentSkill,
    PaymentContext,
    PaymentValidationError,
    PaymentChargingError,
    InsufficientBalanceError,
    PaymentRequiredError,
    pricing,
    PricingInfo,
)
from webagents.agents.skills.robutler.payments.exceptions import (
    PaymentError,
    PaymentTokenRequiredError,
    PaymentTokenInvalidError,
    PaymentPlatformUnavailableError,
)
from webagents.agents.core.base_agent import BaseAgent
from robutler.api import RobutlerClient
from webagents.agents.tools.decorators import tool


class MockContext:
    """Mock context for testing PaymentSkill.

    Supports both attribute-style access (context.payments) and
    dict-style access (context.get/set), matching the real context.
    """
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


class MockResponse:
    """Mock LLM response for cost calculation testing"""
    def __init__(self, model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50, tool_calls=None):
        self.model = model
        self.usage = Mock()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens
        self.usage.total_tokens = prompt_tokens + completion_tokens
        self.tool_calls = tool_calls or []


class MockSkillWithPricingTools:
    """Mock skill with @pricing decorated tools for testing"""

    def __init__(self):
        self.agent = None

    @tool
    @pricing(credits_per_call=1000, reason="Weather lookup service")
    async def get_weather(self, location: str) -> str:
        """Get weather for a location - fixed pricing"""
        return f"Weather in {location}: Sunny, 25°C"

    @tool
    @pricing()  # Dynamic pricing
    async def analyze_text(self, text: str) -> tuple:
        """Analyze text - dynamic pricing based on length"""
        complexity = len(text)
        result = f"Analysis of {complexity} characters: {text[:50]}..."

        # Return tuple with pricing info for dynamic pricing
        pricing_info = PricingInfo(
            credits=complexity * 0.5,
            reason=f"Text analysis of {complexity} characters",
            metadata={"character_count": complexity, "complexity_factor": 0.5}
        )
        return result, pricing_info

    @tool  # No pricing decorator
    async def free_tool(self, data: str) -> str:
        """Free tool with no pricing"""
        return f"Processed: {data}"


class MockAgentWithPricingSkills:
    """Mock agent for testing pricing functionality"""

    def __init__(self):
        self.name = "test_agent"
        self.skills = {}
        self._tools = []


class MockToolCall:
    """Mock tool call for testing"""

    def __init__(self, tool_name: str, arguments: dict = None):
        self.function = type('MockFunction', (), {})()
        self.function.name = tool_name
        self.function.arguments = arguments or {}


@pytest.fixture
def mock_webagents_client():
    """Mock RobutlerClient with lock/settle API"""
    client = Mock(spec=RobutlerClient)
    client._make_request = AsyncMock()
    client.health_check = AsyncMock()
    client.close = AsyncMock()

    client.tokens = Mock()
    client.tokens.validate = AsyncMock(return_value=True)
    client.tokens.validate_with_balance = AsyncMock(return_value={'valid': True, 'balance': 10.0})
    client.tokens.redeem = AsyncMock(return_value=True)
    client.tokens.get_balance = AsyncMock(return_value=10.0)
    client.tokens.lock = AsyncMock(return_value={'lockId': 'lock_test_123', 'lockedAmountDollars': 0.005})
    client.tokens.settle = AsyncMock(return_value={'success': True})
    client.tokens.extend_lock = AsyncMock(return_value={'success': True})

    return client


@pytest.fixture
def payment_config():
    """Payment skill configuration for testing"""
    return {
        'enable_billing': True,
        'minimum_balance': 5.0,
        'per_message_lock': 0.01,
        'default_tool_lock': 0.25,
        'webagents_api_url': 'http://test.localhost',
        'robutler_api_key': 'test_api_key'
    }


@pytest.fixture
def payment_skill(payment_config, mock_webagents_client):
    """PaymentSkill instance for testing"""
    with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
        mock_client_class.return_value = mock_webagents_client

        mock_webagents_client.health_check.return_value = Mock(success=True)

        skill = PaymentSkill(payment_config)
        skill.logger = Mock()
        skill.client = mock_webagents_client
        skill.agent = Mock(name='test-agent')

        return skill


@pytest.fixture
def mock_context_with_payment_token():
    """Mock context with payment token set as attribute"""
    context = MockContext()
    context.payment_token = 'pt_test_valid_token_12345'
    auth = SimpleNamespace(user_id='user_123', agent_id='agent_456')
    context.auth = auth
    return context


@pytest.fixture
def mock_context_no_payment_token():
    """Mock context without payment token"""
    context = MockContext()
    return context


# ===== UNIT TESTS =====

class TestPaymentSkillInitialization:
    """Test PaymentSkill initialization and configuration"""

    def test_payment_skill_creation(self, payment_config):
        """Test PaymentSkill creation with configuration"""
        skill = PaymentSkill(payment_config)

        assert skill.enable_billing == True
        assert skill.minimum_balance == 5.0
        assert skill.per_message_lock == 0.01
        assert skill.default_tool_lock == 0.25
        assert skill.webagents_api_url == 'http://test.localhost'
        assert skill.robutler_api_key == 'test_api_key'

    def test_payment_skill_default_config(self):
        """Test PaymentSkill creation with default configuration"""
        skill = PaymentSkill()

        assert skill.enable_billing == True
        assert skill.minimum_balance == 0.01
        assert skill.per_message_lock == 0.005
        assert skill.default_tool_lock == 0.20

    @patch.dict('os.environ', {'MINIMUM_BALANCE': '10.0', 'PER_MESSAGE_LOCK': '0.02'})
    def test_payment_skill_env_vars(self):
        """Test PaymentSkill respects environment variables"""
        skill = PaymentSkill()

        assert skill.minimum_balance == 10.0
        assert skill.per_message_lock == 0.02


class TestPaymentTokenExtraction:
    """Test _extract_payment_token with various context shapes"""

    def test_extract_token_from_context_attribute(self, payment_skill):
        """Test extracting token from context.payment_token attribute"""
        context = MockContext()
        context.payment_token = 'pt_attr_token_12345'

        token = payment_skill._extract_payment_token(context)
        assert token == 'pt_attr_token_12345'

    def test_extract_token_from_request_headers(self, payment_skill):
        """Test extracting token from context.request.headers"""
        context = MockContext()
        context.request = SimpleNamespace(
            headers={'X-Payment-Token': 'pt_header_token_12345'},
            query_params={}
        )

        token = payment_skill._extract_payment_token(context)
        assert token == 'pt_header_token_12345'

    def test_extract_token_from_query_params(self, payment_skill):
        """Test extracting token from context.request.query_params"""
        context = MockContext()
        context.request = SimpleNamespace(
            headers={},
            query_params={'payment_token': 'pt_query_token_12345'}
        )

        token = payment_skill._extract_payment_token(context)
        assert token == 'pt_query_token_12345'

    def test_extract_token_returns_none_when_missing(self, payment_skill):
        """Test returns None when no token is available"""
        context = MockContext()

        token = payment_skill._extract_payment_token(context)
        assert token is None

    def test_extract_token_prefers_context_attribute(self, payment_skill):
        """Test context.payment_token takes precedence over request headers"""
        context = MockContext()
        context.payment_token = 'pt_attr_token'
        context.request = SimpleNamespace(
            headers={'X-Payment-Token': 'pt_header_token'},
            query_params={}
        )

        token = payment_skill._extract_payment_token(context)
        assert token == 'pt_attr_token'


class TestPaymentContextSetup:
    """Test payment context setup with verify + lock flow"""

    @pytest.mark.asyncio
    async def test_setup_with_valid_token_and_lock(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test payment context setup with valid token, verify, and lock"""
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True,
            'balance': 10.0
        }
        mock_webagents_client.tokens.lock.return_value = {
            'lockId': 'lock_abc_123',
            'lockedAmountDollars': 0.01
        }

        result_context = await payment_skill.setup_payment_context(mock_context_with_payment_token)

        payment_ctx = getattr(result_context, 'payments', None)
        assert payment_ctx is not None
        assert payment_ctx.payment_token == 'pt_test_valid_token_12345'
        assert payment_ctx.lock_id == 'lock_abc_123'
        assert payment_ctx.locked_amount_dollars == 0.01
        assert payment_ctx.user_id == 'user_123'
        assert payment_ctx.agent_id == 'agent_456'

        mock_webagents_client.tokens.validate_with_balance.assert_called_once_with('pt_test_valid_token_12345')
        mock_webagents_client.tokens.lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_insufficient_balance(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test payment context setup with insufficient balance raises error"""
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True,
            'balance': 0.0005  # Below min_usable of 0.001
        }

        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)

        assert "Insufficient balance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_setup_no_token_billing_enabled(self, payment_skill, mock_context_no_payment_token):
        """Test missing payment token raises PaymentTokenRequiredError"""
        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_no_payment_token)

        assert "PAYMENT_TOKEN_REQUIRED" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_setup_invalid_token(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test invalid token raises PaymentTokenInvalidError"""
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': False,
            'error': 'Token expired'
        }

        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)

        assert "PAYMENT_TOKEN_INVALID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_setup_billing_disabled(self, payment_config, mock_context_no_payment_token):
        """Test payment context setup when billing is disabled"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()

        result_context = await skill.setup_payment_context(mock_context_no_payment_token)
        assert result_context == mock_context_no_payment_token

    @pytest.mark.asyncio
    async def test_setup_billing_disabled_passthrough_token(self, payment_config):
        """Test billing disabled still stores token for passthrough (e.g. NLI)"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()

        context = MockContext()
        context.payment_token = 'pt_passthrough_token'

        result_context = await skill.setup_payment_context(context)
        payment_ctx = getattr(result_context, 'payments', None)
        assert payment_ctx is not None
        assert payment_ctx.payment_token == 'pt_passthrough_token'


class TestPaymentCharging:
    """Test payment token charging and settlement"""

    @pytest.mark.asyncio
    async def test_charge_payment_token_success(self, payment_skill, mock_webagents_client):
        """Test successful legacy payment token charging"""
        mock_webagents_client.tokens.redeem.return_value = True

        result = await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')

        assert result == True
        mock_webagents_client.tokens.redeem.assert_called_once_with(
            'test_token', 0.5, description='Test charge'
        )

    @pytest.mark.asyncio
    async def test_charge_payment_token_failure(self, payment_skill, mock_webagents_client):
        """Test failed payment token charging wraps error"""
        mock_webagents_client.tokens.redeem.side_effect = Exception("Failed to redeem token")

        with pytest.raises(PaymentChargingError):
            await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')

    @pytest.mark.asyncio
    async def test_charge_payment_token_no_client(self, payment_skill):
        """Test charging with no client raises platform unavailable"""
        payment_skill.client = None

        with pytest.raises(PaymentPlatformUnavailableError):
            await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')

    @pytest.mark.asyncio
    async def test_settle_payment_success(self, payment_skill, mock_webagents_client):
        """Test successful payment settlement against a lock"""
        mock_webagents_client.tokens.settle.return_value = {'success': True}

        result = await payment_skill._settle_payment('lock_123', 0.05, description='Test settle')

        assert result['success'] == True
        mock_webagents_client.tokens.settle.assert_called_once_with(
            lock_id='lock_123', amount=0.05, description='Test settle',
            charge_type=None, release=False
        )

    @pytest.mark.asyncio
    async def test_settle_payment_failure(self, payment_skill, mock_webagents_client):
        """Test failed settlement wraps error"""
        mock_webagents_client.tokens.settle.side_effect = Exception("Settle failed")

        with pytest.raises(PaymentChargingError):
            await payment_skill._settle_payment('lock_123', 0.05, description='Test settle')

    @pytest.mark.asyncio
    async def test_settle_payment_no_client(self, payment_skill):
        """Test settlement with no client raises platform unavailable"""
        payment_skill.client = None

        with pytest.raises(PaymentPlatformUnavailableError):
            await payment_skill._settle_payment('lock_123', 0.05, description='Test settle')

    @pytest.mark.asyncio
    async def test_extend_lock_success(self, payment_skill, mock_webagents_client):
        """Test successful lock extension"""
        mock_webagents_client.tokens.extend_lock.return_value = {'success': True}

        result = await payment_skill._extend_lock('lock_123', 0.10)

        assert result['success'] == True
        mock_webagents_client.tokens.extend_lock.assert_called_once_with('lock_123', 0.10)

    @pytest.mark.asyncio
    async def test_extend_lock_failure(self, payment_skill, mock_webagents_client):
        """Test failed lock extension returns error dict"""
        mock_webagents_client.tokens.extend_lock.side_effect = Exception("Extend failed")

        result = await payment_skill._extend_lock('lock_123', 0.10)

        assert result['success'] == False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_extend_lock_no_client(self, payment_skill):
        """Test lock extension with no client returns error dict"""
        payment_skill.client = None

        result = await payment_skill._extend_lock('lock_123', 0.10)

        assert result['success'] == False


class TestFinalizePayment:
    """Test payment finalization (cost summing + settlement)"""

    @pytest.mark.asyncio
    async def test_finalize_with_llm_costs(self, payment_skill, mock_webagents_client):
        """Test finalization calculates LLM costs and settles"""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123',
            locked_amount_dollars=0.05
        )
        context.usage = [
            {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50}
        ]

        with patch('webagents.agents.skills.robutler.payments.skill.LITELLM_AVAILABLE', True), \
             patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000030, 0.000015)

            await payment_skill.finalize_payment(context)

        assert context.payments.payment_successful == True
        mock_webagents_client.tokens.settle.assert_called()

    @pytest.mark.asyncio
    async def test_finalize_with_tool_costs(self, payment_skill, mock_webagents_client):
        """Test finalization processes tool pricing records"""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123',
            locked_amount_dollars=0.50
        )
        context.usage = [
            {'type': 'tool', 'tool_name': 'get_weather', 'pricing': {'credits': 0.05, 'reason': 'Weather lookup'}}
        ]

        await payment_skill.finalize_payment(context)

        assert context.payments.payment_successful == True
        mock_webagents_client.tokens.settle.assert_called()

    @pytest.mark.asyncio
    async def test_finalize_with_mixed_costs(self, payment_skill, mock_webagents_client):
        """Test finalization sums both LLM and tool costs"""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123',
            locked_amount_dollars=1.0
        )
        context.usage = [
            {'type': 'llm', 'model': 'gpt-4o-mini', 'prompt_tokens': 100, 'completion_tokens': 50},
            {'type': 'tool', 'tool_name': 'weather', 'pricing': {'credits': 0.10, 'reason': 'Weather'}},
        ]

        with patch('webagents.agents.skills.robutler.payments.skill.LITELLM_AVAILABLE', True), \
             patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cpt:
            mock_cpt.return_value = (0.000030, 0.000015)

            await payment_skill.finalize_payment(context)

        assert context.payments.payment_successful == True

    @pytest.mark.asyncio
    async def test_finalize_with_zero_cost_releases_lock(self, payment_skill, mock_webagents_client):
        """Test finalization with no usage releases the lock"""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123',
            locked_amount_dollars=0.01
        )
        context.usage = []

        await payment_skill.finalize_payment(context)

        mock_webagents_client.tokens.settle.assert_called_once_with(
            lock_id='lock_123', amount=0, description='',
            charge_type=None, release=True
        )

    @pytest.mark.asyncio
    async def test_finalize_without_lock_does_not_succeed(self, payment_skill, mock_webagents_client):
        """Test finalization with cost but no lock logs error"""
        context = MockContext()
        context.payments = PaymentContext(
            payment_token='test_token',
        )
        context.usage = [
            {'type': 'tool', 'tool_name': 'test', 'pricing': {'credits': 0.10}}
        ]

        await payment_skill.finalize_payment(context)

        assert context.payments.payment_successful == False

    @pytest.mark.asyncio
    async def test_finalize_billing_disabled(self, payment_config):
        """Test finalization is a no-op when billing is disabled"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()

        context = MockContext()
        context.payments = PaymentContext(payment_token='test_token')

        result = await skill.finalize_payment(context)
        assert result == context

    @pytest.mark.asyncio
    async def test_finalize_no_payment_context(self, payment_skill):
        """Test finalization returns context unchanged when no payment context"""
        context = MockContext()

        result = await payment_skill.finalize_payment(context)
        assert result == context


class TestErrorHandling:
    """Test error handling and 402 Payment Required responses"""

    @pytest.mark.asyncio
    async def test_402_error_insufficient_balance(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test that insufficient balance raises appropriate error"""
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True,
            'balance': 0.0005  # Below min_usable of 0.001
        }

        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)

        assert "Insufficient balance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_402_error_no_payment_token(self, payment_skill, mock_context_no_payment_token):
        """Test that missing payment token raises appropriate error"""
        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_no_payment_token)

        assert "PAYMENT_TOKEN_REQUIRED" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_402_error_invalid_payment_token(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test that invalid payment token raises appropriate error"""
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': False,
            'error': 'Invalid token',
            'balance': 0.0
        }

        with pytest.raises(PaymentError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)

        assert "PAYMENT_TOKEN_INVALID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_payment_charging_failure_handling(self, payment_skill, mock_webagents_client):
        """Test that charging failure wraps in PaymentChargingError"""
        mock_webagents_client.tokens.redeem.side_effect = Exception("Redeem failed")

        with pytest.raises(PaymentChargingError):
            await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')


# ===== PRICING DECORATOR TESTS =====

class TestPricingDecoratorIntegration:
    """Test @pricing decorator integration with PaymentSkill"""

    @pytest.fixture
    def payment_skill_with_pricing_tools(self, payment_config, mock_webagents_client):
        """PaymentSkill with mock agent containing @pricing decorated tools"""
        with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
            mock_client_class.return_value = mock_webagents_client
            mock_webagents_client.health_check.return_value = Mock(success=True)

            skill = PaymentSkill(payment_config)
            skill.logger = Mock()
            skill.client = mock_webagents_client

            agent = MockAgentWithPricingSkills()
            pricing_skill = MockSkillWithPricingTools()
            pricing_skill.agent = agent
            agent.skills['test'] = pricing_skill

            agent._tools = [
                pricing_skill.get_weather,
                pricing_skill.analyze_text,
                pricing_skill.free_tool,
            ]

            skill.agent = agent

            return skill

    @pytest.mark.asyncio
    async def test_pricing_info_class(self):
        """Test PricingInfo dataclass functionality"""
        pricing_info = PricingInfo(
            credits=250.5,
            reason="Complex analysis",
            metadata={"complexity": "high", "tokens": 1000},
            on_success=lambda: print("Success"),
            on_fail=lambda: print("Failed")
        )

        assert pricing_info.credits == 250.5
        assert pricing_info.reason == "Complex analysis"
        assert pricing_info.metadata["complexity"] == "high"
        assert pricing_info.metadata["tokens"] == 1000
        assert pricing_info.on_success is not None
        assert pricing_info.on_fail is not None

    def test_pricing_decorator_attaches_metadata(self):
        """Test @pricing stores metadata on the function"""
        @pricing(credits_per_call=0.05, reason="Test tool")
        def test_func():
            return "result"

        assert hasattr(test_func, '_webagents_pricing')
        meta = test_func._webagents_pricing
        assert meta['credits_per_call'] == 0.05
        assert meta['reason'] == "Test tool"
        assert meta['supports_dynamic'] == False

    def test_pricing_decorator_dynamic_mode(self):
        """Test @pricing() with no args enables dynamic pricing"""
        @pricing()
        def test_func():
            return "result"

        meta = test_func._webagents_pricing
        assert meta['credits_per_call'] is None
        assert meta['supports_dynamic'] == True

    def test_pricing_decorator_fixed_adds_usage_tuple(self):
        """Test fixed pricing wraps result with usage tuple"""
        @pricing(credits_per_call=0.10, reason="Fixed cost tool")
        def test_func():
            return "plain result"

        result = test_func()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "plain result"
        assert result[1]['pricing']['credits'] == 0.10
        assert result[1]['pricing']['reason'] == "Fixed cost tool"

    @pytest.mark.asyncio
    async def test_pricing_decorator_dynamic_with_pricing_info(self):
        """Test dynamic pricing returns converted usage tuple"""
        @pricing()
        async def test_func():
            info = PricingInfo(credits=5.0, reason="Dynamic cost")
            return "dynamic result", info

        result = await test_func()
        assert isinstance(result, tuple)
        assert result[0] == "dynamic result"
        assert result[1]['pricing']['credits'] == 5.0
        assert result[1]['pricing']['reason'] == "Dynamic cost"

    def test_get_pricing_for_tool_found(self, payment_skill_with_pricing_tools):
        """Test _get_pricing_for_tool finds pricing metadata"""
        skill = payment_skill_with_pricing_tools

        tool_call = MockToolCall("get_weather")
        pricing_meta = skill._get_pricing_for_tool(tool_call)

        assert pricing_meta is not None
        assert pricing_meta['credits_per_call'] == 1000
        assert pricing_meta['reason'] == "Weather lookup service"

    def test_get_pricing_for_tool_dynamic(self, payment_skill_with_pricing_tools):
        """Test _get_pricing_for_tool finds dynamic pricing metadata"""
        skill = payment_skill_with_pricing_tools

        tool_call = MockToolCall("analyze_text")
        pricing_meta = skill._get_pricing_for_tool(tool_call)

        assert pricing_meta is not None
        assert pricing_meta['credits_per_call'] is None
        assert pricing_meta['supports_dynamic'] is True

    def test_get_pricing_for_tool_not_found(self, payment_skill_with_pricing_tools):
        """Test _get_pricing_for_tool returns None for unknown tools"""
        skill = payment_skill_with_pricing_tools

        tool_call = MockToolCall("nonexistent_tool")
        pricing_meta = skill._get_pricing_for_tool(tool_call)

        assert pricing_meta is None

    def test_get_pricing_for_free_tool(self, payment_skill_with_pricing_tools):
        """Test _get_pricing_for_tool returns None for tools without @pricing"""
        skill = payment_skill_with_pricing_tools

        tool_call = MockToolCall("free_tool")
        pricing_meta = skill._get_pricing_for_tool(tool_call)

        assert pricing_meta is None


class TestPreauthToolLock:
    """Test preauth_tool_lock hook"""

    @pytest.fixture
    def skill_with_lock(self, payment_config, mock_webagents_client):
        with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
            mock_client_class.return_value = mock_webagents_client

            skill = PaymentSkill(payment_config)
            skill.logger = Mock()
            skill.client = mock_webagents_client
            skill.agent = Mock(name='test-agent', _tools=[])

            return skill

    @pytest.mark.asyncio
    async def test_preauth_billing_disabled(self, payment_config, mock_webagents_client):
        """Test preauth is no-op when billing is disabled"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()
        skill.client = mock_webagents_client
        skill.agent = Mock(name='test-agent', _tools=[])

        context = MockContext()
        context._data['tool_call'] = MockToolCall("some_tool")

        result = await skill.preauth_tool_lock(context)
        assert result == context
        mock_webagents_client.tokens.extend_lock.assert_not_called()

    @pytest.mark.asyncio
    async def test_preauth_extends_existing_lock(self, skill_with_lock, mock_webagents_client):
        """Test preauth extends existing lock for tool execution"""
        skill = skill_with_lock

        context = MockContext()
        context._data['tool_call'] = MockToolCall("some_tool")
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123',
            locked_amount_dollars=0.01
        )

        mock_webagents_client.tokens.extend_lock.return_value = {'success': True}

        result = await skill.preauth_tool_lock(context)

        mock_webagents_client.tokens.extend_lock.assert_called_once()
        assert context.payments.locked_amount_dollars == 0.01 + 0.25  # default_tool_lock

    @pytest.mark.asyncio
    async def test_preauth_no_lock_creates_fresh_lock(self, skill_with_lock, mock_webagents_client):
        """Test preauth creates a fresh lock when none exists"""
        skill = skill_with_lock

        context = MockContext()
        context._data['tool_call'] = MockToolCall("some_tool")
        context.payments = PaymentContext(
            payment_token='test_token',
        )

        mock_webagents_client.tokens.lock.return_value = {
            'lockId': 'fresh_lock_123',
            'lockedAmountDollars': 0.25
        }

        result = await skill.preauth_tool_lock(context)

        mock_webagents_client.tokens.lock.assert_called_once()
        assert context.payments.lock_id == 'fresh_lock_123'

    @pytest.mark.asyncio
    async def test_preauth_no_tool_call(self, skill_with_lock):
        """Test preauth returns context unchanged when no tool_call"""
        context = MockContext()

        result = await skill_with_lock.preauth_tool_lock(context)
        assert result == context


class TestAccumulateLLMCosts:
    """Test accumulate_llm_costs hook (BYOK key fetching)"""

    @pytest.mark.asyncio
    async def test_noop_without_byok(self, payment_skill):
        """Test hook is a no-op without BYOK providers"""
        context = MockContext()

        result = await payment_skill.accumulate_llm_costs(context)
        assert result == context

    @pytest.mark.asyncio
    async def test_fetches_byok_keys_when_providers_present(self, payment_skill):
        """Test hook fetches BYOK keys when byok_providers is set"""
        context = MockContext()
        context.byok_providers = ['openai']
        context.byok_user_id = 'user_123'

        with patch.object(payment_skill, '_fetch_byok_keys', new_callable=AsyncMock) as mock_fetch:
            await payment_skill.accumulate_llm_costs(context)
            mock_fetch.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_skips_byok_fetch_if_keys_cached(self, payment_skill):
        """Test hook skips fetching when byok_keys already cached"""
        context = MockContext()
        context.byok_providers = ['openai']
        context.byok_keys = {'openai': {'key': 'sk-test'}}

        with patch.object(payment_skill, '_fetch_byok_keys', new_callable=AsyncMock) as mock_fetch:
            await payment_skill.accumulate_llm_costs(context)
            mock_fetch.assert_not_called()


class TestHandleToolCompletion:
    """Test handle_tool_completion hook"""

    @pytest.mark.asyncio
    async def test_noop_billing_disabled(self, payment_config):
        """Test hook is no-op when billing disabled"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()

        context = MockContext()
        result = await skill.handle_tool_completion(context)
        assert result == context

    @pytest.mark.asyncio
    async def test_logs_on_tool_error(self, payment_skill):
        """Test hook logs when tool result indicates error"""
        context = MockContext()
        context._data['tool_result'] = "Error: something went wrong"
        context.payments = PaymentContext(
            payment_token='test_token',
            lock_id='lock_123'
        )

        result = await payment_skill.handle_tool_completion(context)
        assert result == context
        payment_skill.logger.info.assert_called()


# ===== INTEGRATION TESTS =====

class TestPaymentSkillIntegration:
    """Integration tests for PaymentSkill with BaseAgent"""

    @pytest.fixture
    def agent_with_payment_skill(self, payment_config, mock_webagents_client):
        """Create BaseAgent with PaymentSkill for integration testing"""
        with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
            mock_client_class.return_value = mock_webagents_client
            mock_webagents_client.health_check.return_value = Mock(success=True)

            payment_skill = PaymentSkill(payment_config)
            payment_skill.logger = Mock()
            payment_skill.client = mock_webagents_client

            agent = BaseAgent(
                name="test-payment-agent",
                instructions="Test agent with payment processing",
                skills={"payments": payment_skill}
            )

            return agent

    @pytest.mark.asyncio
    async def test_agent_initialization_with_payment_skill(self, agent_with_payment_skill):
        """Test agent initialization with PaymentSkill"""
        agent = agent_with_payment_skill

        assert "payments" in agent.skills
        assert isinstance(agent.skills["payments"], PaymentSkill)

        payment_skill = agent.skills["payments"]
        assert hasattr(payment_skill, 'setup_payment_context')
        assert hasattr(payment_skill, 'finalize_payment')
        assert hasattr(payment_skill, '_charge_payment_token')
        assert hasattr(payment_skill, '_settle_payment')


class TestPaymentContextDataclass:
    """Test PaymentContext dataclass"""

    def test_payment_context_defaults(self):
        """Test PaymentContext default values"""
        ctx = PaymentContext()
        assert ctx.payment_token is None
        assert ctx.user_id is None
        assert ctx.agent_id is None
        assert ctx.lock_id is None
        assert ctx.locked_amount_dollars == 0.0
        assert ctx.payment_successful == False
        assert ctx.byok_providers == []
        assert ctx.byok_user_id is None

    def test_payment_context_with_values(self):
        """Test PaymentContext with explicit values"""
        ctx = PaymentContext(
            payment_token='pt_test',
            user_id='user_1',
            agent_id='agent_1',
            lock_id='lock_1',
            locked_amount_dollars=0.05
        )
        assert ctx.payment_token == 'pt_test'
        assert ctx.user_id == 'user_1'
        assert ctx.agent_id == 'agent_1'
        assert ctx.lock_id == 'lock_1'
        assert ctx.locked_amount_dollars == 0.05


class TestHelperMethods:
    """Test internal helper methods"""

    def test_get_tool_name_from_dict(self, payment_skill):
        """Test extracting tool name from dict-style tool_call"""
        tool_call = {'function': {'name': 'test_tool'}}
        name = payment_skill._get_tool_name(tool_call)
        assert name == 'test_tool'

    def test_get_tool_name_from_object(self, payment_skill):
        """Test extracting tool name from object-style tool_call"""
        tool_call = MockToolCall("my_tool")
        name = payment_skill._get_tool_name(tool_call)
        assert name == 'my_tool'

    def test_extract_tool_args_from_dict(self, payment_skill):
        """Test extracting tool args from dict-style tool_call"""
        tool_call = {'function': {'name': 'test', 'arguments': json.dumps({'key': 'value'})}}
        args = payment_skill._extract_tool_args(tool_call)
        assert args == {'key': 'value'}

    def test_extract_tool_args_from_object(self, payment_skill):
        """Test extracting tool args from object-style tool_call"""
        tool_call = MockToolCall("test")
        tool_call.function.arguments = json.dumps({'param': 42})
        args = payment_skill._extract_tool_args(tool_call)
        assert args == {'param': 42}

    def test_extract_tool_args_empty(self, payment_skill):
        """Test extracting tool args when none provided"""
        tool_call = {'function': {'name': 'test'}}
        args = payment_skill._extract_tool_args(tool_call)
        assert args == {}

    def test_extract_tool_args_dict_passthrough(self, payment_skill):
        """Test extracting tool args when arguments is already a dict"""
        tool_call = MockToolCall("test", arguments={'direct': True})
        args = payment_skill._extract_tool_args(tool_call)
        assert args == {'direct': True}


class TestDynamicPricingBilling:
    """Test dynamic pricing via PricingInfo and tool usage record settlement."""

    @pytest.mark.asyncio
    async def test_pricing_info_creates_tool_usage_record(self, payment_skill, mock_webagents_client):
        """When @pricing returns PricingInfo, verify it produces a tool usage record."""
        mock_skill = MockSkillWithPricingTools()

        result, pricing_info = await mock_skill.analyze_text("hello world test")

        assert isinstance(pricing_info, dict), "pricing wrapper should convert PricingInfo to dict"
        assert 'pricing' in pricing_info
        assert pricing_info['pricing']['credits'] == pytest.approx(16 * 0.5)  # 16 chars * 0.5

    @pytest.mark.asyncio
    async def test_fixed_pricing_returns_clean_result(self, payment_skill, mock_webagents_client):
        """When @pricing has credits_per_call, verify the wrapper adds pricing tuple."""
        mock_skill = MockSkillWithPricingTools()

        result = await mock_skill.get_weather("NYC")

        # @pricing wrapper wraps the result as (result, usage_dict)
        assert isinstance(result, tuple), "@pricing should wrap return value"
        actual_result, usage = result
        assert "Sunny" in actual_result
        assert usage['pricing']['credits'] == 1000

    @pytest.mark.asyncio
    async def test_finalize_settles_tool_records(self, payment_skill, mock_webagents_client):
        """Verify finalize_payment settles accumulated tool usage records."""
        context = MockContext()
        payment_ctx = PaymentContext()
        payment_ctx.lock_id = 'lock_dynamic_123'
        payment_ctx.locked_amount_dollars = 1.0
        payment_ctx.payment_token = 'tok_test'
        context.payments = payment_ctx

        # Simulate accumulated usage records (as the payment skill would build them)
        context.usage = [
            {'type': 'tool', 'tool': 'analyze_text', 'pricing': {'credits': 8.0, 'reason': 'Text analysis of 16 characters'}},
            {'type': 'tool', 'tool': 'get_weather', 'pricing': {'credits': 1000, 'reason': 'Weather lookup service'}},
        ]

        mock_webagents_client.tokens.settle = AsyncMock(return_value={'success': True, 'chargedDollars': 1008.0})

        await payment_skill.finalize_payment(context)

        mock_webagents_client.tokens.settle.assert_called()
        # The settle call may use positional or keyword args depending on _settle_payment
        settle_calls = mock_webagents_client.tokens.settle.call_args_list
        # First settle should be the usage settle, second is the release
        usage_settle = settle_calls[0]
        assert usage_settle.kwargs.get('lock_id') == 'lock_dynamic_123'
        usage_sent = usage_settle.kwargs.get('usage')
        assert usage_sent is not None, f"Expected usage kwarg, got: {usage_settle}"
        assert len(usage_sent) == 2
        assert payment_ctx.payment_successful is True

    @pytest.mark.asyncio
    async def test_finalize_releases_lock_on_no_usage(self, payment_skill, mock_webagents_client):
        """Verify lock is released with amount=0 when there are no usage records."""
        context = MockContext()
        payment_ctx = PaymentContext()
        payment_ctx.lock_id = 'lock_empty_456'
        payment_ctx.locked_amount_dollars = 0.5
        payment_ctx.payment_token = 'tok_test'
        context.payments = payment_ctx
        context.usage = []

        mock_webagents_client.tokens.settle = AsyncMock(return_value={'success': True})

        await payment_skill.finalize_payment(context)

        mock_webagents_client.tokens.settle.assert_called_once_with(
            lock_id='lock_empty_456',
            amount=0,
            description='',
            charge_type=None,
            release=True,
        )

    @pytest.mark.asyncio
    async def test_preauth_extends_lock_for_priced_tool(self, payment_skill, mock_webagents_client):
        """Verify before_toolcall extends the lock for a @pricing decorated tool."""
        mock_skill = MockSkillWithPricingTools()

        # Wire up skill -> agent -> payment_skill
        mock_agent = MockAgentWithPricingSkills()
        mock_skill.agent = mock_agent
        mock_agent._tools = [mock_skill.get_weather, mock_skill.analyze_text, mock_skill.free_tool]
        mock_agent.skills = {'mock_skill': mock_skill}
        payment_skill.agent = mock_agent

        context = MockContext()
        payment_ctx = PaymentContext()
        payment_ctx.lock_id = 'lock_preauth_789'
        payment_ctx.locked_amount_dollars = 0.05
        payment_ctx.payment_token = 'tok_test'
        context.payments = payment_ctx

        tool_call = MockToolCall("get_weather", arguments={'location': 'NYC'})
        context.set("tool_call", tool_call)

        mock_webagents_client.tokens.extend_lock = AsyncMock(return_value={'success': True})

        await payment_skill.preauth_tool_lock(context)

        mock_webagents_client.tokens.extend_lock.assert_called_once()
        extend_args = mock_webagents_client.tokens.extend_lock.call_args
        assert extend_args[0][0] == 'lock_preauth_789'
        assert extend_args[0][1] == 1000  # credits_per_call from @pricing


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
