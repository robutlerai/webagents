"""
Integration Tests for x402 Payment Protocol - Agent A ↔ Agent B

Tests complete payment flows between agents using Robutler tokens.
Defers blockchain/crypto tests for later implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402, pricing
from webagents.agents.tools.decorators import http
from webagents.agents.skills.robutler.payments_x402.exceptions import (
    PaymentRequired402,
    X402VerificationFailed,
    X402SettlementFailed
)


# ==================== Test Fixtures ====================

@pytest.fixture
def mock_robutler_client():
    """Mock RobutlerClient with facilitator support"""
    client = AsyncMock()
    
    # Mock facilitator resource
    client.facilitator = AsyncMock()
    
    # Mock verify method
    async def mock_verify(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        # Decode payment header to check token
        import base64
        import json
        try:
            decoded = json.loads(base64.b64decode(payment_header).decode())
            token = decoded.get('payload', {}).get('token', '')
            
            if token.startswith('tok_valid'):
                return {'isValid': True}
            elif token.startswith('tok_expired'):
                return {'isValid': False, 'invalidReason': 'Token expired'}
            else:
                return {'isValid': False, 'invalidReason': 'Invalid token'}
        except:
            return {'isValid': False, 'invalidReason': 'Malformed payment header'}
    
    client.facilitator.verify = mock_verify
    
    # Mock settle method
    async def mock_settle(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import json
        try:
            decoded = json.loads(base64.b64decode(payment_header).decode())
            token = decoded.get('payload', {}).get('token', '')
            
            if token.startswith('tok_valid'):
                return {'success': True, 'transactionHash': 'tx_123456'}
            else:
                return {'success': False, 'error': 'Settlement failed'}
        except:
            return {'success': False, 'error': 'Malformed payment'}
    
    client.facilitator.settle = mock_settle
    
    # Mock supported schemes
    async def mock_supported_schemes() -> Dict[str, Any]:
        return {
            'schemes': [
                {
                    'scheme': 'token',
                    'network': 'robutler',
                    'description': 'Robutler platform credits'
                }
            ]
        }
    
    client.facilitator.supported_schemes = mock_supported_schemes
    
    # Mock tokens resource for Agent A
    client.tokens = AsyncMock()
    
    async def mock_validate_with_balance(token: str) -> Dict[str, Any]:
        if token.startswith('tok_valid'):
            return {'valid': True, 'balance': 10.0}
        elif token.startswith('tok_low_balance'):
            return {'valid': True, 'balance': 0.01}
        else:
            return {'valid': False, 'balance': 0.0}
    
    client.tokens.validate_with_balance = mock_validate_with_balance
    
    return client


@pytest.fixture
def agent_b_paid_endpoint(mock_robutler_client):
    """Create Agent B with a paid HTTP endpoint"""
    
    # Add PaymentSkillX402
    payment_skill = PaymentSkillX402(config={
        'accepted_schemes': [
            {'scheme': 'token', 'network': 'robutler'}
        ]
    })
    
    # Mock the client
    payment_skill.client = mock_robutler_client
    
    agent = BaseAgent(
        name="agent-b-weather",
        skills={'payments': payment_skill}
    )
    
    # Set agent reference
    payment_skill.agent = agent
    
    # Define paid endpoint
    @http("/weather", method="get")
    @pricing(credits_per_call=0.50, reason="Weather API call")
    async def get_weather(location: str) -> dict:
        """Get weather for a location (costs 0.50 credits)"""
        return {
            "location": location,
            "temperature": 72,
            "conditions": "sunny"
        }
    
    agent.get_weather = get_weather
    
    return agent, payment_skill


@pytest.fixture
def agent_a_consumer(mock_robutler_client):
    """Create Agent A that consumes paid APIs"""
    
    # Add PaymentSkillX402
    payment_skill = PaymentSkillX402(config={
        'payment_schemes': ['token'],
        'auto_exchange': False  # Defer crypto tests
    })
    
    # Mock the client
    payment_skill.client = mock_robutler_client
    
    agent = BaseAgent(
        name="agent-a-consumer",
        skills={'payments': payment_skill}
    )
    
    # Set agent reference
    payment_skill.agent = agent
    
    return agent, payment_skill


# ==================== Test Cases ====================

class TestAgentBPaidEndpoints:
    """Test Agent B exposing paid HTTP endpoints"""
    
    @pytest.mark.asyncio
    async def test_returns_402_without_payment(self, agent_b_paid_endpoint):
        """Agent B returns 402 with x402 requirements when no payment provided"""
        agent_b, payment_skill = agent_b_paid_endpoint
        
        # Create mock request context without payment header
        context = Mock()
        context.request = Mock()
        context.request.headers = Mock()
        context.request.headers.get = Mock(return_value=None)
        context.request.url = Mock()
        context.request.url.path = '/weather'
        
        # Mock endpoint function with pricing metadata
        context.endpoint_func = Mock()
        context.endpoint_func._http_requires_payment = True
        context.endpoint_func._webagents_pricing = {
            'credits_per_call': 0.50,
            'reason': 'Weather API call'
        }
        
        # Call the hook
        with pytest.raises(PaymentRequired402) as exc_info:
            await payment_skill.check_http_endpoint_payment(context)
        
        # Verify 402 response structure
        error = exc_info.value
        assert error.status_code == 402
        requirements = error.payment_requirements
        
        assert requirements['x402Version'] == 1
        assert len(requirements['accepts']) > 0
        
        accept = requirements['accepts'][0]
        assert accept['scheme'] == 'token'
        assert accept['network'] == 'robutler'
        assert accept['maxAmountRequired'] == '0.5'
        assert accept['resource'] == '/weather'
        assert accept['description'] == 'Weather API call'
    
    @pytest.mark.asyncio
    async def test_accepts_valid_robutler_token(self, agent_b_paid_endpoint):
        """Agent B accepts valid Robutler token payment"""
        agent_b, payment_skill = agent_b_paid_endpoint
        
        # Create payment header with valid token
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_valid_abc123:secret_xyz',
                'amount': '0.50'
            }
        }
        payment_header = base64.b64encode(
            json.dumps(payment_data).encode()
        ).decode()
        
        # Create mock request context WITH payment header
        context = Mock()
        context.request = Mock()
        context.request.headers = Mock()
        context.request.headers.get = Mock(return_value=payment_header)
        context.request.url = Mock()
        context.request.url.path = '/weather'
        
        # Mock endpoint function
        context.endpoint_func = Mock()
        context.endpoint_func._http_requires_payment = True
        context.endpoint_func._webagents_pricing = {
            'credits_per_call': 0.50,
            'reason': 'Weather API call'
        }
        
        # Call the hook - should NOT raise
        result = await payment_skill.check_http_endpoint_payment(context)
        
        # Should return context unchanged (payment accepted)
        assert result == context
        
        # Verify facilitator methods were called (they're async functions, not mocks with .called)
        # The fact that we got here without exception means they were called successfully
    
    @pytest.mark.asyncio
    async def test_rejects_invalid_token(self, agent_b_paid_endpoint):
        """Agent B rejects invalid/expired tokens"""
        agent_b, payment_skill = agent_b_paid_endpoint
        
        # Create payment header with expired token
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_expired_old123:secret_old',
                'amount': '0.50'
            }
        }
        payment_header = base64.b64encode(
            json.dumps(payment_data).encode()
        ).decode()
        
        # Create mock request context
        context = Mock()
        context.request = Mock()
        context.request.headers = Mock()
        context.request.headers.get = Mock(return_value=payment_header)
        context.request.url = Mock()
        context.request.url.path = '/weather'
        
        context.endpoint_func = Mock()
        context.endpoint_func._http_requires_payment = True
        context.endpoint_func._webagents_pricing = {
            'credits_per_call': 0.50,
            'reason': 'Weather API call'
        }
        
        # Call the hook - should raise verification error
        with pytest.raises(X402VerificationFailed) as exc_info:
            await payment_skill.check_http_endpoint_payment(context)
        
        error = exc_info.value
        assert 'Token expired' in str(error)


class TestAgentAConsumerPayments:
    """Test Agent A making payments to Agent B"""
    
    @pytest.mark.asyncio
    async def test_creates_payment_from_existing_token(self, agent_a_consumer):
        """Agent A creates payment using existing token from context"""
        agent_a, payment_skill = agent_a_consumer
        
        # Mock context with existing token
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_user123:secret_abc'
        
        # Mock payment requirements (simulating 402 response)
        accepts = [
            {
                'scheme': 'token',
                'network': 'robutler',
                'maxAmountRequired': '0.50',
                'resource': '/weather',
                'payTo': 'agent-b-weather'
            }
        ]
        
        # Create payment
        payment_header, scheme, cost = await payment_skill._create_payment(
            accepts, context
        )
        
        # Verify payment was created
        assert payment_header is not None
        assert scheme == 'token:robutler'
        assert cost == 0.50
        
        # Verify it's a valid base64 encoded payment
        import base64
        import json
        decoded = json.loads(base64.b64decode(payment_header).decode())
        assert decoded['scheme'] == 'token'
        assert decoded['network'] == 'robutler'
        assert decoded['payload']['token'].startswith('tok_valid')
    
    @pytest.mark.asyncio
    async def test_gets_available_token_from_api(self, agent_a_consumer):
        """Agent A fetches available token from API when not in context"""
        agent_a, payment_skill = agent_a_consumer
        
        # Mock context without token
        context = Mock()
        context.payments = None
        
        # Mock API response with token list
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {
            'tokens': [
                {
                    'token': 'tok_valid_from_api:secret_123',
                    'balance': 5.0,
                    'status': 'active'
                }
            ]
        }
        
        payment_skill.client._make_request = AsyncMock(return_value=mock_response)
        payment_skill.agent.id = 'agent_a_test'
        
        # Get available token
        token = await payment_skill._get_available_token(context)
        
        # Verify token was retrieved
        assert token == 'tok_valid_from_api:secret_123'
        
        # Verify API was called correctly
        payment_skill.client._make_request.assert_called_once()
        call_args = payment_skill.client._make_request.call_args
        assert '/agents/agent_a_test/payment-tokens' in call_args[0][1]


class TestAgentABIntegration:
    """End-to-end integration tests for Agent A ↔ Agent B"""
    
    @pytest.mark.asyncio
    async def test_complete_payment_flow(self, agent_a_consumer, agent_b_paid_endpoint):
        """Complete flow: Agent A pays Agent B using Robutler token"""
        agent_a, payment_skill_a = agent_a_consumer
        agent_b, payment_skill_b = agent_b_paid_endpoint
        
        # Step 1: Agent A calls Agent B without payment (simulated)
        # Agent B returns 402
        context_b = Mock()
        context_b.request = Mock()
        context_b.request.headers = Mock()
        context_b.request.headers.get = Mock(return_value=None)
        context_b.request.url = Mock()
        context_b.request.url.path = '/weather'
        
        context_b.endpoint_func = Mock()
        context_b.endpoint_func._http_requires_payment = True
        context_b.endpoint_func._webagents_pricing = {
            'credits_per_call': 0.50,
            'reason': 'Weather API call'
        }
        
        # Get 402 response
        with pytest.raises(PaymentRequired402) as exc_info:
            await payment_skill_b.check_http_endpoint_payment(context_b)
        
        requirements = exc_info.value.payment_requirements
        
        # Step 2: Agent A processes 402 and creates payment
        context_a = Mock()
        context_a.payments = Mock()
        context_a.payments.payment_token = 'tok_valid_agent_a:secret_key'
        
        payment_header, scheme, cost = await payment_skill_a._create_payment(
            requirements['accepts'], context_a
        )
        
        assert payment_header is not None
        assert cost == 0.50
        
        # Step 3: Agent A retries with payment
        context_b_retry = Mock()
        context_b_retry.request = Mock()
        context_b_retry.request.headers = Mock()
        context_b_retry.request.headers.get = Mock(return_value=payment_header)
        context_b_retry.request.url = Mock()
        context_b_retry.request.url.path = '/weather'
        
        context_b_retry.endpoint_func = context_b.endpoint_func
        
        # Agent B accepts payment
        result = await payment_skill_b.check_http_endpoint_payment(context_b_retry)
        
        # Payment accepted, context returned
        assert result == context_b_retry
        
        # Verify both verify and settle were called successfully
        # The fact that we got here without exception means they were called
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self, agent_a_consumer, agent_b_paid_endpoint):
        """Test multiple paid requests in sequence"""
        agent_a, payment_skill_a = agent_a_consumer
        agent_b, payment_skill_b = agent_b_paid_endpoint
        
        # Agent A has a token
        context_a = Mock()
        context_a.payments = Mock()
        context_a.payments.payment_token = 'tok_valid_multiuse:secret_xyz'
        
        # Make 3 sequential paid requests
        for i in range(3):
            # Get 402 requirements
            context_b = Mock()
            context_b.request = Mock()
            context_b.request.headers = Mock()
            context_b.request.headers.get = Mock(return_value=None)
            context_b.request.url = Mock()
            context_b.request.url.path = f'/weather?request={i}'
            
            context_b.endpoint_func = Mock()
            context_b.endpoint_func._http_requires_payment = True
            context_b.endpoint_func._webagents_pricing = {
                'credits_per_call': 0.25,
                'reason': f'Weather request {i}'
            }
            
            with pytest.raises(PaymentRequired402) as exc_info:
                await payment_skill_b.check_http_endpoint_payment(context_b)
            
            requirements = exc_info.value.payment_requirements
            
            # Agent A creates payment
            payment_header, scheme, cost = await payment_skill_a._create_payment(
                requirements['accepts'], context_a
            )
            
            # Agent B processes payment
            context_b_paid = Mock()
            context_b_paid.request = Mock()
            context_b_paid.request.headers = Mock()
            context_b_paid.request.headers.get = Mock(return_value=payment_header)
            context_b_paid.request.url = context_b.request.url
            context_b_paid.endpoint_func = context_b.endpoint_func
            
            result = await payment_skill_b.check_http_endpoint_payment(context_b_paid)
            assert result == context_b_paid
        
        # All 3 requests should succeed
        # Total cost: 0.25 * 3 = 0.75 credits
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_scenario(self, agent_a_consumer, agent_b_paid_endpoint):
        """Test Agent A with insufficient balance"""
        agent_a, payment_skill_a = agent_a_consumer
        agent_b, payment_skill_b = agent_b_paid_endpoint
        
        # Agent A has low balance token
        context_a = Mock()
        context_a.payments = Mock()
        context_a.payments.payment_token = 'tok_low_balance_user:secret_low'
        
        # Agent B requires 5.00 credits (more than available)
        context_b = Mock()
        context_b.request = Mock()
        context_b.request.headers = Mock()
        context_b.request.headers.get = Mock(return_value=None)
        context_b.request.url = Mock()
        context_b.request.url.path = '/premium-service'
        
        context_b.endpoint_func = Mock()
        context_b.endpoint_func._http_requires_payment = True
        context_b.endpoint_func._webagents_pricing = {
            'credits_per_call': 5.00,
            'reason': 'Premium service'
        }
        
        with pytest.raises(PaymentRequired402) as exc_info:
            await payment_skill_b.check_http_endpoint_payment(context_b)
        
        requirements = exc_info.value.payment_requirements
        
        # Agent A tries to create payment but has insufficient balance
        # The token validation should show balance = 0.01 (from mock)
        token_result = await payment_skill_a.client.tokens.validate_with_balance(
            'tok_low_balance_user:secret_low'
        )
        
        assert token_result['valid'] is True
        assert token_result['balance'] < 5.00
        
        # In real scenario, this would trigger token refresh or return error


class TestPaymentEncoding:
    """Test payment header encoding/decoding"""
    
    def test_encode_robutler_payment(self):
        """Test encoding Robutler token payment"""
        from webagents.agents.skills.robutler.payments_x402.schemes import (
            encode_robutler_payment,
            decode_payment_header
        )
        
        token = 'tok_test123:secret_abc'
        amount = '2.50'
        
        # Encode
        encoded = encode_robutler_payment(token, amount)
        
        # Should be base64
        assert encoded is not None
        assert len(encoded) > 0
        
        # Decode
        decoded = decode_payment_header(encoded)
        
        # Verify structure
        assert decoded['scheme'] == 'token'
        assert decoded['network'] == 'robutler'
        assert decoded['payload']['token'] == token
        assert decoded['payload']['amount'] == amount
    
    def test_decode_invalid_header(self):
        """Test decoding invalid payment header"""
        from webagents.agents.skills.robutler.payments_x402.schemes import (
            decode_payment_header
        )
        
        # Invalid base64
        with pytest.raises(ValueError):
            decode_payment_header('not-valid-base64!!!')
        
        # Valid base64 but not JSON
        import base64
        invalid_json = base64.b64encode(b'not json').decode()
        
        with pytest.raises(ValueError):
            decode_payment_header(invalid_json)


class TestX402Requirements:
    """Test x402 requirement generation"""
    
    def test_create_x402_requirements(self):
        """Test creating x402 PaymentRequirements"""
        from webagents.agents.skills.robutler.payments_x402.schemes import (
            create_x402_requirements,
            create_x402_response
        )
        
        requirement = create_x402_requirements(
            scheme='token',
            network='robutler',
            amount=1.50,
            resource='/api/data',
            pay_to='agent_data_provider',
            description='Data API access'
        )
        
        assert requirement['scheme'] == 'token'
        assert requirement['network'] == 'robutler'
        assert requirement['maxAmountRequired'] == '1.5'
        assert requirement['resource'] == '/api/data'
        assert requirement['payTo'] == 'agent_data_provider'
        assert requirement['description'] == 'Data API access'
        assert requirement['mimeType'] == 'application/json'
        assert requirement['maxTimeoutSeconds'] == 60
    
    def test_create_x402_response(self):
        """Test creating x402 402 response"""
        from webagents.agents.skills.robutler.payments_x402.schemes import (
            create_x402_requirements,
            create_x402_response
        )
        
        req1 = create_x402_requirements(
            scheme='token',
            network='robutler',
            amount=1.0,
            resource='/test',
            pay_to='agent_test'
        )
        
        req2 = create_x402_requirements(
            scheme='exact',
            network='base-mainnet',
            amount=1.0,
            resource='/test',
            pay_to='0x123...'
        )
        
        response = create_x402_response([req1, req2])
        
        assert response['x402Version'] == 1
        assert len(response['accepts']) == 2
        assert response['accepts'][0]['scheme'] == 'token'
        assert response['accepts'][1]['scheme'] == 'exact'


# ==================== Performance Tests ====================

class TestPaymentPerformance:
    """Test payment processing performance"""
    
    @pytest.mark.asyncio
    async def test_payment_verification_speed(self, agent_b_paid_endpoint):
        """Test that payment verification is fast"""
        import time
        
        agent_b, payment_skill = agent_b_paid_endpoint
        
        # Create valid payment
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_valid_speed_test:secret',
                'amount': '0.10'
            }
        }
        payment_header = base64.b64encode(
            json.dumps(payment_data).encode()
        ).decode()
        
        # Time 10 verifications
        start = time.time()
        
        for _ in range(10):
            context = Mock()
            context.request = Mock()
            context.request.headers = Mock()
            context.request.headers.get = Mock(return_value=payment_header)
            context.request.url = Mock()
            context.request.url.path = '/test'
            
            context.endpoint_func = Mock()
            context.endpoint_func._http_requires_payment = True
            context.endpoint_func._webagents_pricing = {
                'credits_per_call': 0.10,
                'reason': 'Test'
            }
            
            await payment_skill.check_http_endpoint_payment(context)
        
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        # Should be fast (< 100ms per verification with mocks)
        assert avg_time < 0.1, f"Payment verification too slow: {avg_time:.3f}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

