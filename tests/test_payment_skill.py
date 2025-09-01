"""
Unit and Integration Tests for PaymentSkill

Tests payment token validation, balance checking, usage tracking,
402 error handling, and @pricing decorator integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal

# Import PaymentSkill and related classes
from webagents.agents.skills.robutler.payments import (
    PaymentSkill, 
    PaymentContext, 
    PaymentValidationError,
    PaymentChargingError,
    InsufficientBalanceError,
    PaymentRequiredError
)
from webagents.agents.core.base_agent import BaseAgent
from robutler.api import RobutlerClient
from webagents.agents.tools.decorators import tool, pricing, PricingInfo


class MockContext:
    """Mock context for testing PaymentSkill"""
    def __init__(self):
        self._data = {}
        self.headers = {}
        self.query_params = {}
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
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
            credits=complexity * 0.5,  # 0.5 credits per character
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


class MockToolCall:
    """Mock tool call for testing"""
    
    def __init__(self, tool_name: str, arguments: dict = None):
        self.function = type('MockFunction', (), {})()
        self.function.name = tool_name
        self.function.arguments = arguments or {}


@pytest.fixture
def mock_webagents_client():
    """Mock RobutlerClient for testing"""
    client = Mock(spec=RobutlerClient)
    client._make_request = AsyncMock()
    client.health_check = AsyncMock()
    client.close = AsyncMock()
    
    # Mock the tokens resource for object-oriented API calls
    client.tokens = Mock()
    client.tokens.validate = AsyncMock(return_value=True)
    client.tokens.validate_with_balance = AsyncMock(return_value={'valid': True, 'balance': 10.0})
    client.tokens.redeem = AsyncMock(return_value=True)
    client.tokens.get_balance = AsyncMock(return_value=10.0)
    
    return client


@pytest.fixture
def payment_config():
    """Payment skill configuration for testing"""
    return {
        'enable_billing': True,
        'agent_pricing_percent': 1.5,  # 50% markup
        'minimum_balance': 5.0,  # $5 minimum
        'webagents_api_url': 'http://test.localhost',
        'robutler_api_key': 'test_api_key'
    }


@pytest.fixture
def payment_skill(payment_config, mock_webagents_client):
    """PaymentSkill instance for testing"""
    with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
        mock_client_class.return_value = mock_webagents_client
        
        # Mock health check to succeed
        mock_webagents_client.health_check.return_value = Mock(success=True)
        
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()  # Mock logger to avoid setup issues
        
        # Mock initialize without full agent
        skill.client = mock_webagents_client
        
        return skill


@pytest.fixture
def mock_context_with_payment_token():
    """Mock context with payment token and identity headers"""
    context = MockContext()
    context._data.update({
        'headers': {
            'X-Payment-Token': 'pt_test_valid_token_12345',
            'X-Origin-User-ID': 'user_origin_123',
            'X-Peer-User-ID': 'user_peer_456',
            'X-Agent-Owner-User-ID': 'agent_owner_789'
        },
        'query_params': {}
    })
    return context


@pytest.fixture
def mock_context_no_payment_token():
    """Mock context without payment token"""
    context = MockContext()
    context._data.update({
        'headers': {
            'X-Origin-User-ID': 'user_origin_123',
            'X-Peer-User-ID': 'user_peer_456',
        },
        'query_params': {}
    })
    return context


# ===== UNIT TESTS =====

class TestPaymentSkillInitialization:
    """Test PaymentSkill initialization and configuration"""
    
    def test_payment_skill_creation(self, payment_config):
        """Test PaymentSkill creation with configuration"""
        skill = PaymentSkill(payment_config)
        
        assert skill.enable_billing == True
        assert skill.agent_pricing_percent == 1.5
        assert skill.minimum_balance == 5.0
        assert skill.webagents_api_url == 'http://test.localhost'
        assert skill.robutler_api_key == 'test_api_key'
    
    def test_payment_skill_default_config(self):
        """Test PaymentSkill creation with default configuration"""
        skill = PaymentSkill()
        
        assert skill.enable_billing == True
        assert skill.agent_pricing_percent == 1.2  # Default 20% markup
        assert skill.minimum_balance == 1.0  # Default minimum
        assert skill.robutler_api_key == 'rok_testapikey'  # Default test key
    
    @patch.dict('os.environ', {'AGENT_PRICING_PERCENT': '2.0', 'MINIMUM_BALANCE': '10.0'})
    def test_payment_skill_env_vars(self):
        """Test PaymentSkill respects environment variables"""
        skill = PaymentSkill()
        
        assert skill.agent_pricing_percent == 2.0
        assert skill.minimum_balance == 10.0


class TestPaymentTokenValidation:
    """Test payment token validation logic"""
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_valid(self, payment_skill, mock_webagents_client):
        """Test successful payment token validation"""
        # Mock successful validation response
        mock_webagents_client.tokens.validate.return_value = True
        
        result = await payment_skill._validate_payment_token('valid_token')
        
        assert result == True
        mock_webagents_client.tokens.validate.assert_called_once_with('valid_token')
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_invalid(self, payment_skill, mock_webagents_client):
        """Test failed payment token validation"""
        # Mock failed validation response
        mock_webagents_client.tokens.validate.return_value = False
        
        result = await payment_skill._validate_payment_token('invalid_token')
        
        assert result == False
        mock_webagents_client.tokens.validate.assert_called_once_with('invalid_token')
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_with_balance_sufficient(self, payment_skill, mock_webagents_client):
        """Test payment token validation with sufficient balance"""
        # Mock successful balance check
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True, 
            'balance': 10.0
        }
        
        result = await payment_skill._validate_payment_token_with_balance('token_with_balance')
        
        assert result['valid'] == True
        assert result['balance'] == 10.0
        mock_webagents_client.tokens.validate_with_balance.assert_called_once_with('token_with_balance')
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_with_balance_insufficient(self, payment_skill, mock_webagents_client):
        """Test payment token validation with insufficient balance"""
        # Mock low balance response
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True, 
            'balance': 1.0
        }
        
        result = await payment_skill._validate_payment_token_with_balance('token_low_balance')
        
        assert result['valid'] == True
        assert result['balance'] == 1.0
        mock_webagents_client.tokens.validate_with_balance.assert_called_once_with('token_low_balance')
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_with_balance_api_error(self, payment_skill, mock_webagents_client):
        """Test payment token validation when API returns error"""
        # Mock API error
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': False, 
            'error': 'API error', 
            'balance': 0.0
        }
        
        result = await payment_skill._validate_payment_token_with_balance('token_api_error')
        
        assert result['valid'] == False
        assert result['balance'] == 0.0
        mock_webagents_client.tokens.validate_with_balance.assert_called_once_with('token_api_error')


class TestPaymentContextSetup:
    """Test payment context setup and identity extraction"""
    
    @pytest.mark.asyncio
    async def test_setup_payment_context_with_valid_token(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test payment context setup with valid token and sufficient balance"""
        # Mock successful validation with sufficient balance
        mock_webagents_client._make_request.return_value = Mock(
            success=True,
            data={'balance': 10.0}  # Above minimum_balance of 5.0
        )
        
        result_context = await payment_skill.setup_payment_context(mock_context_with_payment_token)
        
        # Verify payment context was created
        payment_context = result_context.get('payment_context')
        assert payment_context is not None
        assert payment_context.payment_token == 'pt_test_valid_token_12345'
        assert payment_context.origin_user_id == 'user_origin_123'
        assert payment_context.peer_user_id == 'user_peer_456'
        assert payment_context.agent_owner_user_id == 'agent_owner_789'
        
        # Verify context variables were set
        assert result_context.get('payment_token') == 'pt_test_valid_token_12345'
        assert result_context.get('origin_user_id') == 'user_origin_123'
        assert result_context.get('peer_user_id') == 'user_peer_456'
        assert result_context.get('agent_owner_user_id') == 'agent_owner_789'
    
    @pytest.mark.asyncio
    async def test_setup_payment_context_insufficient_balance(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test payment context setup with insufficient balance should raise InsufficientBalanceError"""
        # Mock validation with insufficient balance
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True,
            'balance': 2.0  # Below minimum_balance of 5.0
        }
        
        from webagents.agents.skills.robutler.payments.exceptions import InsufficientBalanceError
        with pytest.raises(InsufficientBalanceError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)
        
        assert "Insufficient balance" in str(exc_info.value)
        assert "2.0" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_setup_payment_context_no_token_billing_enabled(self, payment_skill, mock_context_no_payment_token):
        """Test payment context setup without token when billing is enabled should raise PaymentTokenRequiredError"""
        from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
        with pytest.raises(PaymentTokenRequiredError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_no_payment_token)
        
        assert "Payment token required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_setup_payment_context_billing_disabled(self, payment_config, mock_context_no_payment_token):
        """Test payment context setup when billing is disabled"""
        payment_config['enable_billing'] = False
        skill = PaymentSkill(payment_config)
        skill.logger = Mock()  # Add logger to avoid setup issues
        
        result_context = await skill.setup_payment_context(mock_context_no_payment_token)
        
        # Should pass through without validation when billing is disabled
        assert result_context == mock_context_no_payment_token


class TestCostCalculation:
    """Test LiteLLM cost calculation and markup application"""
    
    @pytest.mark.asyncio
    async def test_calculate_llm_cost_with_completion_cost(self, payment_skill):
        """Test LLM cost calculation using completion_cost"""
        mock_response = MockResponse("gpt-4o-mini", 100, 50)
        
        with patch('webagents.agents.skills.robutler.payments.skill.completion_cost') as mock_completion_cost:
            mock_completion_cost.return_value = 0.000045  # Mock cost
            
            cost = await payment_skill._calculate_llm_cost(mock_response, {})
            
            assert cost == 0.000045
            mock_completion_cost.assert_called_once_with(completion_response=mock_response)
    
    @pytest.mark.asyncio
    async def test_calculate_llm_cost_fallback_to_cost_per_token(self, payment_skill):
        """Test LLM cost calculation fallback to cost_per_token"""
        mock_response = MockResponse("gpt-4o-mini", 100, 50)
        
        with patch('webagents.agents.skills.robutler.payments.skill.completion_cost') as mock_completion_cost, \
             patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cost_per_token:
            
            # Make completion_cost fail
            mock_completion_cost.side_effect = Exception("completion_cost failed")
            
            # Mock cost_per_token
            mock_cost_per_token.return_value = (0.000030, 0.000015)  # (prompt_cost, completion_cost)
            
            cost = await payment_skill._calculate_llm_cost(mock_response, {})
            
            assert cost == 0.000045  # 0.000030 + 0.000015
            mock_cost_per_token.assert_called_once_with(
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50
            )
    
    @pytest.mark.asyncio
    async def test_calculate_cost_from_tokens(self, payment_skill):
        """Test direct cost calculation from tokens"""
        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cost_per_token:
            mock_cost_per_token.return_value = (0.000020, 0.000010)
            
            cost = await payment_skill._calculate_cost_from_tokens("gpt-4o-mini", 200, 100)
            
            assert abs(cost - 0.000030) < 1e-8  # Use tolerance for floating-point comparison
            mock_cost_per_token.assert_called_once_with(
                model="gpt-4o-mini",
                prompt_tokens=200,
                completion_tokens=100
            )
    
    @pytest.mark.asyncio
    async def test_accumulate_costs_with_agent_pricing_percent(self, payment_skill):
        """Test cost accumulation with agent pricing percent application"""
        # Setup mock context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Mock response
        mock_response = MockResponse("gpt-4o-mini", 150, 75)
        context.set('response', mock_response)
        
        # Mock cost calculation
        with patch.object(payment_skill, '_calculate_llm_cost', return_value=0.000060):
            result_context = await payment_skill.accumulate_costs(context)
            
            updated_payment_context = result_context.get('payment_context')
            assert abs(updated_payment_context.accumulated_cost_usd - 0.000090) < 1e-8  # 0.000060 * 1.5
            assert len(updated_payment_context.operations) == 1
            
            operation = updated_payment_context.operations[0]
            assert operation['type'] == 'llm_request'
            assert operation['cost_usd'] == 0.000060
            assert abs(operation['final_cost_usd'] - 0.000090) < 1e-8
            assert operation['model'] == 'gpt-4o-mini'


class TestPaymentCharging:
    """Test payment token charging and transaction creation"""
    
    @pytest.mark.asyncio
    async def test_charge_payment_token_success(self, payment_skill, mock_webagents_client):
        """Test successful payment token charging"""
        # Mock successful redemption
        mock_webagents_client.tokens.redeem.return_value = True
        
        result = await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')
        
        assert result == True
        mock_webagents_client.tokens.redeem.assert_called_once_with('test_token', 0.5)
    
    @pytest.mark.asyncio
    async def test_charge_payment_token_failure(self, payment_skill, mock_webagents_client):
        """Test failed payment token charging"""
        # Mock failed redemption by raising an exception
        from robutler.api.client import WebAgentsAPIError
        mock_webagents_client.tokens.redeem.side_effect = WebAgentsAPIError("Failed to redeem token", 400, {})
        
        # Should raise PaymentChargingError
        from webagents.agents.skills.robutler.payments.exceptions import PaymentChargingError
        with pytest.raises(PaymentChargingError):
            await payment_skill._charge_payment_token('test_token', 0.50, 'Test charge')
    
    @pytest.mark.asyncio
    async def test_finalize_payment_with_token_charging(self, payment_skill, mock_webagents_client):
        """Test payment finalization with token charging"""
        # Setup context with payment context that has accumulated costs
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token_12345",
            origin_user_id="user_123",
            agent_owner_user_id="agent_456",
            accumulated_cost_usd=1.25
        )
        payment_context.operations = [
            {'type': 'llm_request', 'cost_usd': 0.75, 'final_cost_usd': 1.125},
            {'type': 'llm_request', 'cost_usd': 0.08, 'final_cost_usd': 0.125}
        ]
        context.set('payment_context', payment_context)
        
        # Mock successful charging
        mock_webagents_client.tokens.redeem.return_value = True
        
        result_context = await payment_skill.finalize_payment(context)
        
        # Verify payment token was charged
        mock_webagents_client.tokens.redeem.assert_called_once_with(
            "test_token_12345", 1.25
        )


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
        
        # Check that payment skill has the required methods
        payment_skill = agent.skills["payments"]
        assert hasattr(payment_skill, 'setup_payment_context')
        assert hasattr(payment_skill, 'finalize_payment')
        assert hasattr(payment_skill, '_validate_payment_token')
        assert hasattr(payment_skill, '_charge_payment_token')
    
    @pytest.mark.asyncio
    async def test_payment_tools_execution(self, agent_with_payment_skill):
        """Test execution of payment tools"""
        agent = agent_with_payment_skill
        payment_skill = agent.skills["payments"]
        
        # Test estimate_llm_cost tool
        with patch('webagents.agents.skills.robutler.payments.skill.cost_per_token') as mock_cost_per_token:
            mock_cost_per_token.return_value = (0.000020, 0.000010)
            
            result = await payment_skill.estimate_llm_cost(
                model="gpt-4o-mini",
                prompt_tokens=100,
                completion_tokens=50,
                context=None
            )
            
            assert result['success'] == True
            estimate = result['estimate']
            assert estimate['model'] == "gpt-4o-mini"
            assert abs(estimate['base_cost_usd'] - 0.000030) < 1e-8  # Use tolerance for floating-point comparison
            assert abs(estimate['final_cost_usd'] - 0.000045) < 1e-8  # 1.5x markup
            assert estimate['agent_pricing_percent'] == 1.5


class TestStreamingVsNonStreamingUsageTracking:
    """Test usage tracking accuracy in streaming vs non-streaming scenarios"""
    
    @pytest.fixture
    def mock_streaming_response_chunks(self):
        """Mock streaming response chunks"""
        chunks = [
            {"id": "chatcmpl-123", "choices": [{"delta": {"role": "assistant"}}]},
            {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "chatcmpl-123", "choices": [{"delta": {"content": " world"}}]},
            {
                "id": "chatcmpl-123", 
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
        ]
        return chunks
    
    @pytest.fixture
    def mock_non_streaming_response(self):
        """Mock non-streaming response"""
        return {
            "id": "chatcmpl-456",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
    
    @pytest.mark.asyncio
    async def test_non_streaming_usage_tracking(self, payment_skill, mock_non_streaming_response):
        """Test usage tracking in non-streaming scenario"""
        # Setup context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Mock response as LLM response object
        mock_response = Mock()
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        context.set('response', mock_response)
        
        # Mock cost calculation
        with patch.object(payment_skill, '_calculate_llm_cost', return_value=0.000050) as mock_calc_cost:
            result_context = await payment_skill.accumulate_costs(context)
            
            # Verify cost was calculated and accumulated
            mock_calc_cost.assert_called_once_with(mock_response, context)
            
            updated_payment_context = result_context.get('payment_context')
            assert abs(updated_payment_context.accumulated_cost_usd - 0.000075) < 1e-8  # 0.000050 * 1.5
            assert len(updated_payment_context.operations) == 1
            
            operation = updated_payment_context.operations[0]
            assert operation['type'] == 'llm_request'
            assert operation['cost_usd'] == 0.000050
            assert abs(operation['final_cost_usd'] - 0.000075) < 1e-8
    
    @pytest.mark.asyncio
    async def test_streaming_usage_tracking_multiple_chunks(self, payment_skill):
        """Test usage tracking with multiple streaming chunks"""
        # Setup context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Simulate multiple streaming chunks with different costs
        chunk_responses = [
            Mock(model="gpt-4o-mini", usage=Mock(prompt_tokens=10, completion_tokens=2)),
            Mock(model="gpt-4o-mini", usage=Mock(prompt_tokens=12, completion_tokens=3)),
            Mock(model="gpt-4o-mini", usage=Mock(prompt_tokens=15, completion_tokens=5))
        ]
        
        expected_total_cost = 0.0
        
        with patch.object(payment_skill, '_calculate_llm_cost') as mock_calc_cost:
            # Mock different costs for each chunk
            mock_calc_cost.side_effect = [0.000020, 0.000030, 0.000040]
            
            # Process each chunk
            for chunk_response in chunk_responses:
                context.set('response', chunk_response)
                context = await payment_skill.accumulate_costs(context)
                expected_total_cost += mock_calc_cost.return_value * payment_skill.agent_pricing_percent
            
            # Verify all costs were accumulated
            updated_payment_context = context.get('payment_context')
            total_accumulated = sum(op['final_cost_usd'] for op in updated_payment_context.operations)
            
            assert len(updated_payment_context.operations) == 3
            assert abs(total_accumulated - (0.000135)) < 0.000001  # (0.000020 + 0.000030 + 0.000040) * 1.5
    
    @pytest.mark.asyncio
    async def test_end_to_end_payment_flow_non_streaming(self, payment_skill, mock_webagents_client):
        """Test complete payment flow for non-streaming request"""
        # Setup initial context with valid payment token
        context = MockContext()
        context._data.update({
            'headers': {
                'X-Payment-Token': 'pt_valid_token_12345',
                'X-Origin-User-ID': 'user_origin_123',
                'X-Agent-Owner-User-ID': 'agent_owner_456'
            }
        })
        
        # Mock successful token validation with sufficient balance
        mock_webagents_client._make_request.side_effect = [
            Mock(success=True, data={'balance': 10.0}),  # Token balance check
            Mock(success=True),  # Token redemption
        ]
        
        # Step 1: Setup payment context
        context = await payment_skill.setup_payment_context(context)
        payment_context = context.get('payment_context')
        assert payment_context is not None
        assert payment_context.payment_token == 'pt_valid_token_12345'
        
        # Step 2: Process response and accumulate costs
        mock_response = MockResponse("gpt-4o-mini", 100, 50)
        context.set('response', mock_response)
        
        with patch.object(payment_skill, '_calculate_llm_cost', return_value=0.000080):
            context = await payment_skill.accumulate_costs(context)
            
            updated_payment_context = context.get('payment_context')
            assert abs(updated_payment_context.accumulated_cost_usd - 0.000120) < 1e-8  # 0.000080 * 1.5
        
        # Step 3: Finalize payment
        with patch.object(payment_skill, '_charge_payment_token', return_value=True) as mock_charge:
            await payment_skill.finalize_payment(context)
            
            # Verify payment was charged (with tolerance for floating-point precision)
            actual_call = mock_charge.call_args[0]
            assert actual_call[0] == 'pt_valid_token_12345'
            assert abs(actual_call[1] - 0.000120) < 1e-8  # Use tolerance for amount
            assert actual_call[2] == 'Agent usage: 1 operations'
    
    @pytest.mark.asyncio
    async def test_end_to_end_payment_flow_streaming(self, payment_skill, mock_webagents_client):
        """Test complete payment flow for streaming request"""
        # Setup initial context
        context = MockContext()
        context._data.update({
            'headers': {
                'X-Payment-Token': 'pt_streaming_token_12345',
                'X-Origin-User-ID': 'user_stream_123',
                'X-Agent-Owner-User-ID': 'agent_owner_789'
            }
        })
        
        # Mock token validation
        mock_webagents_client._make_request.side_effect = [
            Mock(success=True, data={'balance': 15.0}),  # Token balance check
            Mock(success=True),  # Token redemption
        ]
        
        # Step 1: Setup payment context
        context = await payment_skill.setup_payment_context(context)
        
        # Step 2: Simulate streaming chunks
        streaming_chunks = [
            MockResponse("gpt-4o-mini", 20, 5),
            MockResponse("gpt-4o-mini", 25, 8),
            MockResponse("gpt-4o-mini", 30, 12)
        ]
        
        total_expected_cost = 0.0
        
        with patch.object(payment_skill, '_calculate_llm_cost') as mock_calc_cost:
            mock_calc_cost.side_effect = [0.000025, 0.000035, 0.000045]  # Different costs per chunk
            
            for chunk in streaming_chunks:
                context.set('response', chunk)
                context = await payment_skill.accumulate_costs(context)
            
            updated_payment_context = context.get('payment_context')
            total_cost = sum(op['final_cost_usd'] for op in updated_payment_context.operations)
            
            assert len(updated_payment_context.operations) == 3
            assert abs(total_cost - 0.0001575) < 0.0000001  # (0.000025 + 0.000035 + 0.000045) * 1.5
        
        # Step 3: Finalize payment
        with patch.object(payment_skill, '_charge_payment_token', return_value=True) as mock_charge:
            await payment_skill.finalize_payment(context)
            
            # Verify payment was charged (with tolerance for floating-point precision)
            actual_call = mock_charge.call_args[0]
            assert actual_call[0] == 'pt_streaming_token_12345'
            assert abs(actual_call[1] - 0.0001575) < 1e-8  # Use tolerance for amount
            assert actual_call[2] == 'Agent usage: 3 operations'


class TestErrorHandling:
    """Test error handling and 402 Payment Required responses"""
    
    @pytest.mark.asyncio
    async def test_402_error_insufficient_balance(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test that insufficient balance raises InsufficientBalanceError (402)"""
        # Mock token validation with insufficient balance
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': True,
            'balance': 2.0  # Below minimum_balance of 5.0
        }
        
        from webagents.agents.skills.robutler.payments.exceptions import InsufficientBalanceError
        with pytest.raises(InsufficientBalanceError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)
        
        error_message = str(exc_info.value)
        assert "Insufficient balance" in error_message
        assert "2.0" in error_message
    
    @pytest.mark.asyncio
    async def test_402_error_no_payment_token(self, payment_skill, mock_context_no_payment_token):
        """Test that missing payment token raises PaymentTokenRequiredError (402)"""
        from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
        with pytest.raises(PaymentTokenRequiredError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_no_payment_token)
        
        assert "Payment token required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_402_error_invalid_payment_token(self, payment_skill, mock_context_with_payment_token, mock_webagents_client):
        """Test that invalid payment token raises PaymentTokenInvalidError (402)"""
        # Mock invalid token response
        mock_webagents_client.tokens.validate_with_balance.return_value = {
            'valid': False,
            'error': 'Invalid token',
            'balance': 0.0
        }
        
        from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenInvalidError
        with pytest.raises(PaymentTokenInvalidError) as exc_info:
            await payment_skill.setup_payment_context(mock_context_with_payment_token)
        
        assert "Invalid token" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_payment_charging_failure_handling(self, payment_skill, mock_webagents_client):
        """Test handling of payment charging failures"""
        # Setup context with accumulated costs
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="failing_token",
            accumulated_cost_usd=5.0
        )
        context.set('payment_context', payment_context)
        
        # Mock failed payment charging
        with patch.object(payment_skill, '_charge_payment_token', return_value=False):
            # Should not raise exception, but should log error
            result_context = await payment_skill.finalize_payment(context)
            
            # Verify error was logged (would be checked in real implementation)
            assert result_context == context  # Context should be returned unchanged
    
    @pytest.mark.asyncio
    async def test_cost_calculation_error_handling(self, payment_skill):
        """Test handling of cost calculation errors"""
        context = MockContext()
        payment_context = PaymentContext(payment_token="test_token")
        context.set('payment_context', payment_context)
        
        # Mock response that will cause calculation error
        mock_response = Mock()
        mock_response.model = "unknown-model"
        mock_response.usage = None  # This should cause an error
        context.set('response', mock_response)
        
        # Should handle error gracefully and not accumulate cost
        result_context = await payment_skill.accumulate_costs(context)
        
        updated_payment_context = result_context.get('payment_context')
        assert updated_payment_context.accumulated_cost_usd == 0.0
        assert len(updated_payment_context.operations) == 0


# ===== PRICING DECORATOR TESTS =====

class TestPricingDecoratorIntegration:
    """Test @pricing decorator integration with PaymentSkill"""
    
    @pytest.fixture
    def payment_skill_with_pricing_tools(self, payment_config, mock_webagents_client):
        """PaymentSkill with mock agent containing @pricing decorated tools"""
        with patch('webagents.agents.skills.robutler.payments.skill.RobutlerClient') as mock_client_class:
            mock_client_class.return_value = mock_webagents_client
            mock_webagents_client.health_check.return_value = Mock(success=True)
            
            # Create payment skill
            skill = PaymentSkill(payment_config)
            skill.logger = Mock()
            skill.client = mock_webagents_client
            
            # Create mock agent with pricing tools
            agent = MockAgentWithPricingSkills()
            pricing_skill = MockSkillWithPricingTools()
            pricing_skill.agent = agent
            agent.skills['test'] = pricing_skill
            
            # Assign agent to payment skill
            skill.agent = agent
            
            return skill
    
    @pytest.mark.asyncio
    async def test_find_tool_function_success(self, payment_skill_with_pricing_tools):
        """Test _find_tool_function finds tools correctly"""
        skill = payment_skill_with_pricing_tools
        
        # Test finding existing tools
        weather_tool = await skill._find_tool_function("get_weather")
        analyze_tool = await skill._find_tool_function("analyze_text")
        free_tool = await skill._find_tool_function("free_tool")
        
        assert weather_tool is not None
        assert analyze_tool is not None
        assert free_tool is not None
        
        # Verify pricing metadata
        weather_pricing = getattr(weather_tool, '_webagents_pricing', None)
        analyze_pricing = getattr(analyze_tool, '_webagents_pricing', None)
        free_pricing = getattr(free_tool, '_webagents_pricing', None)
        
        assert weather_pricing is not None
        assert weather_pricing['credits_per_call'] == 1000
        assert weather_pricing['reason'] == "Weather lookup service"
        
        assert analyze_pricing is not None
        assert analyze_pricing['credits_per_call'] is None  # Dynamic pricing
        assert analyze_pricing['supports_dynamic'] is True
        
        assert free_pricing is None  # No pricing decorator
    
    @pytest.mark.asyncio
    async def test_find_tool_function_not_found(self, payment_skill_with_pricing_tools):
        """Test _find_tool_function returns None for non-existent tools"""
        skill = payment_skill_with_pricing_tools
        
        nonexistent_tool = await skill._find_tool_function("nonexistent_tool")
        assert nonexistent_tool is None
    
    @pytest.mark.asyncio
    async def test_calculate_tool_cost_fixed_pricing(self, payment_skill_with_pricing_tools):
        """Test _calculate_tool_cost for fixed pricing tools"""
        skill = payment_skill_with_pricing_tools
        
        # Mock tool function with fixed pricing
        tool_func = Mock()
        tool_call = MockToolCall("get_weather", {"location": "New York"})
        pricing_info = {
            'credits_per_call': 1000,
            'reason': 'Weather lookup',
            'supports_dynamic': False
        }
        
        cost = await skill._calculate_tool_cost(tool_func, tool_call, pricing_info, {})
        
        # 1000 credits * $0.001/credit = $1.00
        expected_cost = 1000 * 0.001
        assert abs(cost - expected_cost) < 0.001
    
    @pytest.mark.asyncio
    async def test_calculate_tool_cost_dynamic_pricing(self, payment_skill_with_pricing_tools):
        """Test _calculate_tool_cost for dynamic pricing tools"""
        skill = payment_skill_with_pricing_tools
        
        # Mock tool function with dynamic pricing
        tool_func = Mock()
        tool_call = MockToolCall("analyze_text", {"text": "test"})
        pricing_info = {
            'credits_per_call': None,
            'reason': 'Text analysis',
            'supports_dynamic': True
        }
        
        cost = await skill._calculate_tool_cost(tool_func, tool_call, pricing_info, {})
        
        # Should use default tool cost for dynamic pricing
        expected_cost = 0.01  # default_tool_cost_usd
        assert abs(cost - expected_cost) < 0.001
    
    @pytest.mark.asyncio
    async def test_process_tool_pricing_with_fixed_pricing(self, payment_skill_with_pricing_tools):
        """Test _process_tool_pricing with fixed pricing tools"""
        skill = payment_skill_with_pricing_tools
        
        # Create mock response with tool call
        tool_calls = [MockToolCall("get_weather", {"location": "Paris"})]
        response = MockResponse(tool_calls=tool_calls)
        
        # Create context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Process tool pricing
        total_cost = await skill._process_tool_pricing(response, context)
        
        # Should return cost for weather tool (1000 credits * $0.001)
        expected_cost = 1.0
        assert abs(total_cost - expected_cost) < 0.001
        
        # Check that operation was recorded
        operations = payment_context.operations
        assert len(operations) == 1
        assert operations[0]['type'] == 'tool_execution'
        assert operations[0]['tool_name'] == 'get_weather'
        assert operations[0]['pricing_type'] == 'fixed'
    
    @pytest.mark.asyncio
    async def test_process_tool_pricing_with_dynamic_pricing(self, payment_skill_with_pricing_tools):
        """Test _process_tool_pricing with dynamic pricing tools"""
        skill = payment_skill_with_pricing_tools
        
        # Create mock response with dynamic pricing tool call
        tool_calls = [MockToolCall("analyze_text", {"text": "test text"})]
        response = MockResponse(tool_calls=tool_calls)
        
        # Create context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Process tool pricing
        total_cost = await skill._process_tool_pricing(response, context)
        
        # Should use default cost for dynamic pricing
        expected_cost = 0.01
        assert abs(total_cost - expected_cost) < 0.001
        
        # Check that operation was recorded
        operations = payment_context.operations
        assert len(operations) == 1
        assert operations[0]['type'] == 'tool_execution'
        assert operations[0]['tool_name'] == 'analyze_text'
        assert operations[0]['pricing_type'] == 'dynamic'
    
    @pytest.mark.asyncio
    async def test_process_tool_pricing_with_free_tools(self, payment_skill_with_pricing_tools):
        """Test _process_tool_pricing ignores tools without @pricing decorator"""
        skill = payment_skill_with_pricing_tools
        
        # Create mock response with free tool call
        tool_calls = [MockToolCall("free_tool", {"data": "test"})]
        response = MockResponse(tool_calls=tool_calls)
        
        # Create context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Process tool pricing
        total_cost = await skill._process_tool_pricing(response, context)
        
        # Should be zero cost for free tools
        assert total_cost == 0.0
        
        # Check that no operations were recorded
        operations = payment_context.operations
        assert len(operations) == 0
    
    @pytest.mark.asyncio
    async def test_process_tool_pricing_with_mixed_tools(self, payment_skill_with_pricing_tools):
        """Test _process_tool_pricing with mixed tool calls"""
        skill = payment_skill_with_pricing_tools
        
        # Create mock response with multiple tool calls
        tool_calls = [
            MockToolCall("get_weather", {"location": "Tokyo"}),
            MockToolCall("analyze_text", {"text": "analysis"}),
            MockToolCall("free_tool", {"data": "free"}),
            MockToolCall("nonexistent_tool", {"param": "value"})
        ]
        response = MockResponse(tool_calls=tool_calls)
        
        # Create context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        
        # Process tool pricing
        total_cost = await skill._process_tool_pricing(response, context)
        
        # Should be cost for weather (1.0) + analyze (0.01) = 1.01
        expected_cost = 1.0 + 0.01
        assert abs(total_cost - expected_cost) < 0.001
        
        # Check that only 2 operations were recorded (not free or nonexistent)
        operations = payment_context.operations
        assert len(operations) == 2
        
        # Check operation details
        tool_names = [op['tool_name'] for op in operations]
        assert 'get_weather' in tool_names
        assert 'analyze_text' in tool_names
        assert 'free_tool' not in tool_names
        assert 'nonexistent_tool' not in tool_names
    
    @pytest.mark.asyncio
    async def test_accumulate_costs_with_llm_and_tools(self, payment_skill_with_pricing_tools):
        """Test accumulate_costs combines LLM and tool costs correctly"""
        skill = payment_skill_with_pricing_tools
        
        # Create mock response with both LLM usage and tool calls
        tool_calls = [
            MockToolCall("get_weather", {"location": "Berlin"}),
            MockToolCall("analyze_text", {"text": "test"})
        ]
        response = MockResponse(
            model="gpt-4o-mini",
            prompt_tokens=500,
            completion_tokens=200,
            tool_calls=tool_calls
        )
        
        # Create context with payment context
        context = MockContext()
        payment_context = PaymentContext(
            payment_token="test_token",
            origin_user_id="user_123"
        )
        context.set('payment_context', payment_context)
        context.set('response', response)
        
        # Mock LLM cost calculation
        with patch.object(skill, '_calculate_llm_cost', return_value=0.05) as mock_llm_cost:
            await skill.accumulate_costs(context)
        
        # Check accumulated costs
        updated_context = context.get('payment_context')
        
        # Should have LLM cost + tool costs
        # LLM: $0.05 * 1.5 (agent pricing percent) = $0.075
        # Tools: $1.0 (weather) + $0.01 (analyze) = $1.01
        # Total: $0.075 + $1.01 = $1.085
        expected_total = (0.05 * 1.5) + 1.0 + 0.01
        assert abs(updated_context.accumulated_cost_usd - expected_total) < 0.001
        
        # Check operations count: 1 LLM + 2 tools = 3 operations
        operations = updated_context.operations
        assert len(operations) == 3
        
        # Check operation types
        operation_types = [op['type'] for op in operations]
        assert 'llm_request' in operation_types
        assert operation_types.count('tool_execution') == 2
    
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
    
    @pytest.mark.asyncio
    async def test_tool_cost_conversion_configuration(self, payment_skill_with_pricing_tools):
        """Test that cost_per_credit configuration affects tool pricing"""
        skill = payment_skill_with_pricing_tools
        
        # Set custom cost per credit
        skill.cost_per_credit = 0.002  # $0.002 per credit instead of default $0.001
        
        # Test cost calculation with custom rate
        tool_func = Mock()
        tool_call = MockToolCall("get_weather")
        pricing_info = {
            'credits_per_call': 500,
            'reason': 'Weather lookup',
            'supports_dynamic': False
        }
        
        cost = await skill._calculate_tool_cost(tool_func, tool_call, pricing_info, {})
        
        # 500 credits * $0.002/credit = $1.00
        expected_cost = 500 * 0.002
        assert abs(cost - expected_cost) < 0.001


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 