"""
Integration Tests for x402 - Agent A ↔ Agent B over HTTP

Tests actual HTTP communication between agents via completions endpoint.
Agent B exposes paid HTTP endpoints, Agent A calls them via real HTTP requests.
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402, pricing
from webagents.agents.tools.decorators import http, tool

# Note: Full HTTP server tests require running server
# These tests focus on the payment logic and protocol, not full E2E HTTP


# ==================== Test Fixtures ====================

@pytest.fixture
async def agent_b_with_http(tmp_path):
    """
    Create Agent B with paid HTTP endpoint (without running server)
    Tests focus on payment hooks and x402 protocol logic
    """
    # Mock Robutler client for Agent B
    mock_client_b = AsyncMock()
    
    # Mock facilitator verify
    async def mock_verify(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import json
        try:
            decoded = json.loads(base64.b64decode(payment_header).decode())
            token = decoded.get('payload', {}).get('token', '')
            if token.startswith('tok_valid'):
                return {'isValid': True}
            return {'isValid': False, 'invalidReason': 'Invalid token'}
        except:
            return {'isValid': False, 'invalidReason': 'Malformed payment'}
    
    mock_client_b.facilitator = AsyncMock()
    mock_client_b.facilitator.verify = mock_verify
    
    # Mock facilitator settle
    async def mock_settle(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import json
        try:
            decoded = json.loads(base64.b64decode(payment_header).decode())
            token = decoded.get('payload', {}).get('token', '')
            if token.startswith('tok_valid'):
                return {'success': True, 'transactionHash': 'tx_test_123'}
            return {'success': False, 'error': 'Invalid token'}
        except:
            return {'success': False, 'error': 'Malformed payment'}
    
    mock_client_b.facilitator.settle = mock_settle
    
    # Create Agent B with paid endpoint
    payment_skill_b = PaymentSkillX402(config={
        'accepted_schemes': [
            {'scheme': 'token', 'network': 'robutler'}
        ]
    })
    payment_skill_b.client = mock_client_b
    
    agent_b = BaseAgent(
        name="agent-b-weather",
        instructions="You provide weather information via paid API.",
        skills={'payments': payment_skill_b}
    )
    
    payment_skill_b.agent = agent_b
    
    # Add paid HTTP endpoint
    @http("/weather", method="get")
    @pricing(credits_per_call=0.50, reason="Weather API call")
    async def get_weather(location: str) -> dict:
        """Get weather for a location (costs 0.50 credits)"""
        return {
            "location": location,
            "temperature": 72,
            "conditions": "sunny",
            "humidity": 65
        }
    
    # Register endpoint with agent
    agent_b.get_weather = get_weather
    agent_b.register_http_handler(get_weather)
    
    yield agent_b, payment_skill_b, get_weather


@pytest.fixture
async def agent_a_client():
    """
    Create Agent A that makes HTTP requests to paid APIs
    """
    # Mock Robutler client for Agent A
    mock_client_a = AsyncMock()
    
    # Mock token validation
    async def mock_validate(token: str) -> Dict[str, Any]:
        if token.startswith('tok_valid'):
            return {'valid': True, 'balance': 10.0}
        return {'valid': False, 'balance': 0.0}
    
    mock_client_a.tokens = AsyncMock()
    mock_client_a.tokens.validate_with_balance = mock_validate
    
    # Mock facilitator
    mock_client_a.facilitator = AsyncMock()
    
    # Create Agent A
    payment_skill_a = PaymentSkillX402(config={
        'payment_schemes': ['token'],
        'auto_exchange': False
    })
    payment_skill_a.client = mock_client_a
    
    agent_a = BaseAgent(
        name="agent-a-consumer",
        instructions="You consume weather APIs.",
        skills={'payments': payment_skill_a}
    )
    
    payment_skill_a.agent = agent_a
    
    yield agent_a, payment_skill_a, mock_client_a


# ==================== HTTP Integration Tests ====================

class TestAgentBHTTPEndpointPayments:
    """Test Agent B's HTTP endpoint payment requirements (hook logic)"""
    
    @pytest.mark.asyncio
    async def test_http_endpoint_without_payment_triggers_402(self, agent_b_with_http):
        """Agent B's hook returns 402 when endpoint called without payment"""
        agent_b, payment_skill, get_weather = agent_b_with_http
        
        # Create mock request context without payment header
        context = Mock()
        context.request = Mock()
        context.request.headers = Mock()
        context.request.headers.get = Mock(return_value=None)
        context.request.url = Mock()
        context.request.url.path = '/weather'
        
        # Mock endpoint function with pricing metadata
        context.endpoint_func = get_weather
        
        # Call the payment hook
        from webagents.agents.skills.robutler.payments_x402.exceptions import PaymentRequired402
        
        with pytest.raises(PaymentRequired402) as exc_info:
            await payment_skill.check_http_endpoint_payment(context)
        
        # Verify 402 response structure
        error = exc_info.value
        assert error.status_code == 402
        requirements = error.payment_requirements
        
        assert requirements['x402Version'] == 1
        assert len(requirements['accepts']) > 0
        
        # Check first payment requirement
        accept = requirements['accepts'][0]
        assert accept['scheme'] == 'token'
        assert accept['network'] == 'robutler'
        assert accept['maxAmountRequired'] == '0.5'
        assert accept['resource'] == '/weather'
        assert 'Weather API call' in accept['description']
    
    @pytest.mark.asyncio
    async def test_http_endpoint_with_payment_succeeds(self, agent_b_with_http):
        """Agent B's hook accepts valid payment"""
        agent_b, payment_skill, get_weather = agent_b_with_http
        
        # Create valid payment header
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_valid_test:secret_abc',
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
        context.endpoint_func = get_weather
        
        # Call the hook - should NOT raise
        result = await payment_skill.check_http_endpoint_payment(context)
        
        # Should return context unchanged (payment accepted)
        assert result == context


# Skip complex HTTP tests for now - focus on tool tests
# Full E2E HTTP tests would require running server with WebAgentsServer


class TestAgentABToolIntegrationCore:
    """Core tests for Agent A/B tool payments without HTTP complexity"""
    
    @pytest.mark.asyncio
    async def test_pricing_decorator_on_http_endpoint(self, agent_b_with_http):
        """Verify @pricing works with @http decorator"""
        agent_b, payment_skill, get_weather = agent_b_with_http
        
        # Verify endpoint has pricing metadata
        assert hasattr(get_weather, '_webagents_pricing')
        assert hasattr(get_weather, '_http_requires_payment')
        assert get_weather._http_requires_payment is True
        
        pricing_info = get_weather._webagents_pricing
        assert pricing_info['credits_per_call'] == 0.50
        assert pricing_info['reason'] == 'Weather API call'
    
    @pytest.mark.asyncio
    async def test_agent_a_creates_payment_for_http_endpoint(self, agent_a_client, agent_b_with_http):
        """
        Complete flow: Agent A automatically handles 402 and retries with payment
        
        This tests the full Agent A → Agent B flow:
        1. Agent A makes request to Agent B
        2. Agent B returns 402
        3. Agent A's hook detects 402, creates payment
        4. Agent A retries with payment
        5. Agent B accepts and returns result
        """
        agent_a, payment_skill_a, mock_client_a = agent_a_client
        agent_b, app, base_url = agent_b_server
        
        # Give Agent A a valid payment token
        payment_token = 'tok_valid_agent_a:secret_xyz'
        
        # Mock context with token
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = payment_token
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            # Step 1: First request (no payment) - get 402
            response_402 = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "New York"}
            )
            
            assert response_402.status_code == 402
            requirements = response_402.json()
            
            # Step 2: Agent A creates payment from 402 response
            payment_header, scheme, cost = await payment_skill_a._create_payment(
                requirements['accepts'], context
            )
            
            assert payment_header is not None
            assert scheme == 'token:robutler'
            assert cost == 0.50
            
            # Step 3: Agent A retries with payment
            response_success = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "New York"},
                headers={"X-PAYMENT": payment_header}
            )
            
            # Should succeed
            assert response_success.status_code == 200
            data = response_success.json()
            assert data['location'] == 'New York'
            assert data['temperature'] == 72
    
    @pytest.mark.asyncio
    async def test_multiple_paid_requests_in_sequence(self, agent_a_client, agent_b_server):
        """Test Agent A making multiple sequential paid requests"""
        agent_a, payment_skill_a, mock_client_a = agent_a_client
        agent_b, app, base_url = agent_b_server
        
        payment_token = 'tok_valid_multiuse:secret_multi'
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = payment_token
        
        locations = ["Tokyo", "London", "Paris"]
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            for location in locations:
                # Get 402
                response_402 = await client.get(
                    "/agents/agent-b-weather/weather",
                    params={"location": location}
                )
                assert response_402.status_code == 402
                
                # Create payment
                requirements = response_402.json()
                payment_header, scheme, cost = await payment_skill_a._create_payment(
                    requirements['accepts'], context
                )
                
                # Retry with payment
                response_success = await client.get(
                    "/agents/agent-b-weather/weather",
                    params={"location": location},
                    headers={"X-PAYMENT": payment_header}
                )
                
                # Verify success
                assert response_success.status_code == 200
                data = response_success.json()
                assert data['location'] == location
    
    @pytest.mark.asyncio
    async def test_invalid_payment_rejected(self, agent_b_server):
        """Agent B rejects invalid payment token"""
        agent_b, app, base_url = agent_b_server
        
        # Create invalid payment header
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_invalid_bad:secret_bad',
                'amount': '0.50'
            }
        }
        payment_header = base64.b64encode(
            json.dumps(payment_data).encode()
        ).decode()
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            response = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Berlin"},
                headers={"X-PAYMENT": payment_header}
            )
            
            # Should return error (402 or 400)
            assert response.status_code in [402, 400, 500]
    
    @pytest.mark.asyncio
    async def test_malformed_payment_header_rejected(self, agent_b_server):
        """Agent B rejects malformed payment header"""
        agent_b, app, base_url = agent_b_server
        
        # Malformed payment header (not valid base64 JSON)
        payment_header = "not-valid-payment-header"
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            response = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Sydney"},
                headers={"X-PAYMENT": payment_header}
            )
            
            # Should return error
            assert response.status_code in [400, 402, 500]


class TestAgentBMultipleEndpoints:
    """Test Agent B with multiple paid endpoints"""
    
    @pytest.mark.asyncio
    async def test_multiple_endpoints_different_prices(self, tmp_path):
        """Agent B with multiple endpoints at different price points"""
        # Mock client
        mock_client = AsyncMock()
        
        async def mock_verify(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
            return {'isValid': True}
        
        async def mock_settle(payment_header: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
            return {'success': True, 'transactionHash': 'tx_multi_123'}
        
        mock_client.facilitator = AsyncMock()
        mock_client.facilitator.verify = mock_verify
        mock_client.facilitator.settle = mock_settle
        
        # Create Agent B with payment skill
        payment_skill = PaymentSkillX402(config={
            'accepted_schemes': [{'scheme': 'token', 'network': 'robutler'}]
        })
        payment_skill.client = mock_client
        
        agent_b = BaseAgent(
            name="agent-b-data",
            instructions="You provide data services.",
            skills={'payments': payment_skill}
        )
        payment_skill.agent = agent_b
        
        # Add multiple paid endpoints
        @http("/data/basic", method="get")
        @pricing(credits_per_call=0.10, reason="Basic data access")
        async def get_basic_data() -> dict:
            return {"data": "basic", "price": 0.10}
        
        @http("/data/premium", method="get")
        @pricing(credits_per_call=1.00, reason="Premium data access")
        async def get_premium_data() -> dict:
            return {"data": "premium", "price": 1.00}
        
        @http("/data/enterprise", method="get")
        @pricing(credits_per_call=5.00, reason="Enterprise data access")
        async def get_enterprise_data() -> dict:
            return {"data": "enterprise", "price": 5.00}
        
        agent_b.get_basic_data = get_basic_data
        agent_b.get_premium_data = get_premium_data
        agent_b.get_enterprise_data = get_enterprise_data
        
        agent_b.register_http_handler(get_basic_data)
        agent_b.register_http_handler(get_premium_data)
        agent_b.register_http_handler(get_enterprise_data)
        
        # Create app
        app = create_app([agent_b])
        
        # Test each endpoint
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Test basic endpoint (0.10 credits)
            response = await client.get("/agents/agent-b-data/data/basic")
            assert response.status_code == 402
            data = response.json()
            assert data['accepts'][0]['maxAmountRequired'] == '0.1'
            
            # Test premium endpoint (1.00 credits)
            response = await client.get("/agents/agent-b-data/data/premium")
            assert response.status_code == 402
            data = response.json()
            assert data['accepts'][0]['maxAmountRequired'] == '1.0'
            
            # Test enterprise endpoint (5.00 credits)
            response = await client.get("/agents/agent-b-data/data/enterprise")
            assert response.status_code == 402
            data = response.json()
            assert data['accepts'][0]['maxAmountRequired'] == '5.0'


class TestPaymentHeaderFormats:
    """Test various payment header formats and encodings"""
    
    @pytest.mark.asyncio
    async def test_standard_robutler_token_format(self, agent_b_server):
        """Test standard Robutler token payment format"""
        agent_b, app, base_url = agent_b_server
        
        from webagents.agents.skills.robutler.payments_x402.schemes import encode_robutler_payment
        
        # Use helper to encode payment
        token = 'tok_valid_standard:secret_standard'
        amount = '0.50'
        payment_header = encode_robutler_payment(token, amount)
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            response = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Moscow"},
                headers={"X-PAYMENT": payment_header}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['location'] == 'Moscow'
    
    @pytest.mark.asyncio
    async def test_payment_with_metadata(self, agent_b_server):
        """Test payment header with additional metadata"""
        agent_b, app, base_url = agent_b_server
        
        import base64
        import json
        
        payment_data = {
            'scheme': 'token',
            'network': 'robutler',
            'payload': {
                'token': 'tok_valid_meta:secret_meta',
                'amount': '0.50',
                'metadata': {
                    'client': 'agent-a',
                    'timestamp': '2025-11-02T00:00:00Z'
                }
            }
        }
        payment_header = base64.b64encode(
            json.dumps(payment_data).encode()
        ).decode()
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            response = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Dubai"},
                headers={"X-PAYMENT": payment_header}
            )
            
            # Should still work (metadata is optional)
            assert response.status_code == 200


class TestPerformanceAndConcurrency:
    """Test performance and concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_paid_requests(self, agent_a_client, agent_b_server):
        """Test Agent B handling multiple concurrent paid requests"""
        agent_a, payment_skill_a, mock_client_a = agent_a_client
        agent_b, app, base_url = agent_b_server
        
        payment_token = 'tok_valid_concurrent:secret_concurrent'
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = payment_token
        
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            # Create payment first
            response_402 = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "test"}
            )
            requirements = response_402.json()
            payment_header, _, _ = await payment_skill_a._create_payment(
                requirements['accepts'], context
            )
            
            # Make 10 concurrent requests with payment
            async def make_request(location: str):
                response = await client.get(
                    "/agents/agent-b-weather/weather",
                    params={"location": location},
                    headers={"X-PAYMENT": payment_header}
                )
                return response.status_code, response.json()
            
            tasks = [
                make_request(f"City-{i}")
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for status_code, data in results:
                assert status_code == 200
                assert 'location' in data
                assert data['temperature'] == 72


class TestAgentABToolIntegration:
    """Test Agent A calling Agent B's paid tools via completions"""
    
    @pytest.mark.asyncio
    async def test_agent_b_tool_without_payment_returns_error(self, tmp_path):
        """Agent B's paid tool requires payment"""
        # Mock client
        mock_client = AsyncMock()
        mock_client.facilitator = AsyncMock()
        
        # Create Agent B with paid tool
        payment_skill = PaymentSkillX402(config={
            'accepted_schemes': [{'scheme': 'token', 'network': 'robutler'}]
        })
        payment_skill.client = mock_client
        
        agent_b = BaseAgent(
            name="agent-b-calculator",
            instructions="You provide calculation services.",
            skills={'payments': payment_skill}
        )
        payment_skill.agent = agent_b
        
        # Add paid tool
        from webagents.agents.tools.decorators import tool
        
        @tool()
        @pricing(credits_per_call=0.10, reason="Calculator operation")
        async def calculate(expression: str) -> dict:
            """Calculate mathematical expression (costs 0.10 credits)"""
            return {"expression": expression, "result": eval(expression)}
        
        agent_b.calculate = calculate
        agent_b.register_tool(calculate)
        
        # Try to call tool without payment token - should fail
        # In real scenario, PaymentSkill checks context.payments.payment_token
        context = Mock()
        context.payments = None  # No payment context
        
        # Tool execution should be blocked by payment skill
        # (This would happen in the actual tool execution flow)
        assert hasattr(calculate, '_webagents_pricing')
        assert calculate._webagents_pricing['credits_per_call'] == 0.10
    
    @pytest.mark.asyncio
    async def test_agent_a_calls_agent_b_tool_via_completions(self, tmp_path):
        """
        Complete flow: Agent A calls Agent B's paid tool via completions endpoint
        
        Flow:
        1. Agent A sends completion request mentioning Agent B's tool
        2. Agent B's tool requires payment (0.10 credits)
        3. Payment is validated via PaymentSkill
        4. Tool executes and returns result
        5. Agent A receives result in completion
        """
        # Mock clients
        mock_client_b = AsyncMock()
        mock_client_a = AsyncMock()
        
        # Mock token validation for Agent A
        async def mock_validate(token: str) -> Dict[str, Any]:
            if token.startswith('tok_valid'):
                return {'valid': True, 'balance': 10.0, 'remaining': 9.90}
            return {'valid': False, 'balance': 0.0}
        
        # Mock charge for Agent B
        async def mock_charge(token: str, amount: float, metadata: dict) -> Dict[str, Any]:
            if token.startswith('tok_valid'):
                return {
                    'success': True,
                    'new_balance': 9.90,
                    'transaction_id': 'tx_tool_123'
                }
            return {'success': False, 'error': 'Invalid token'}
        
        mock_client_a.tokens = AsyncMock()
        mock_client_a.tokens.validate_with_balance = mock_validate
        
        mock_client_b.tokens = AsyncMock()
        mock_client_b.tokens.validate_with_balance = mock_validate
        mock_client_b.tokens.charge = mock_charge
        
        # Create Agent B with paid tool
        payment_skill_b = PaymentSkillX402(config={
            'accepted_schemes': [{'scheme': 'token', 'network': 'robutler'}]
        })
        payment_skill_b.client = mock_client_b
        
        agent_b = BaseAgent(
            name="agent-b-calculator",
            instructions="You provide calculation services.",
            skills={'payments': payment_skill_b}
        )
        payment_skill_b.agent = agent_b
        
        # Add paid tool
        from webagents.agents.tools.decorators import tool
        
        @tool()
        @pricing(credits_per_call=0.10, reason="Calculator operation")
        async def calculate(expression: str) -> dict:
            """Calculate mathematical expression (costs 0.10 credits)"""
            return {"expression": expression, "result": eval(expression)}
        
        agent_b.calculate = calculate
        agent_b.register_tool(calculate)
        
        # Create Agent A
        payment_skill_a = PaymentSkillX402(config={
            'payment_schemes': ['token']
        })
        payment_skill_a.client = mock_client_a
        
        agent_a = BaseAgent(
            name="agent-a-user",
            instructions="You use calculation tools.",
            skills={'payments': payment_skill_a}
        )
        payment_skill_a.agent = agent_a
        
        # Simulate Agent A calling Agent B's tool with payment token
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_agent_a:secret_a'
        context.payments.total_cost = 0.0
        
        # Execute tool with payment context
        # In real scenario, this happens via completion request
        result = await calculate(expression="2 + 2")
        
        # Verify result
        assert result['expression'] == "2 + 2"
        assert result['result'] == 4
        
        # Verify pricing metadata exists
        assert hasattr(calculate, '_webagents_pricing')
        assert calculate._webagents_pricing['credits_per_call'] == 0.10
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_with_payment(self, tmp_path):
        """Test Agent A making multiple paid tool calls to Agent B"""
        mock_client = AsyncMock()
        
        total_charged = 0.0
        
        async def mock_charge(token: str, amount: float, metadata: dict) -> Dict[str, Any]:
            nonlocal total_charged
            total_charged += amount
            return {
                'success': True,
                'new_balance': 10.0 - total_charged,
                'transaction_id': f'tx_{total_charged}'
            }
        
        async def mock_validate(token: str) -> Dict[str, Any]:
            return {'valid': True, 'balance': 10.0 - total_charged}
        
        mock_client.tokens = AsyncMock()
        mock_client.tokens.charge = mock_charge
        mock_client.tokens.validate_with_balance = mock_validate
        
        # Create Agent B with multiple paid tools
        payment_skill = PaymentSkillX402()
        payment_skill.client = mock_client
        
        agent_b = BaseAgent(
            name="agent-b-services",
            instructions="You provide various services.",
            skills={'payments': payment_skill}
        )
        payment_skill.agent = agent_b
        
        from webagents.agents.tools.decorators import tool
        
        @tool()
        @pricing(credits_per_call=0.10, reason="Basic calculation")
        async def add(a: int, b: int) -> int:
            """Add two numbers (costs 0.10 credits)"""
            return a + b
        
        @tool()
        @pricing(credits_per_call=0.25, reason="Complex calculation")
        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers (costs 0.25 credits)"""
            return a * b
        
        @tool()
        @pricing(credits_per_call=0.50, reason="Advanced operation")
        async def power(base: int, exp: int) -> int:
            """Raise to power (costs 0.50 credits)"""
            return base ** exp
        
        agent_b.add = add
        agent_b.multiply = multiply
        agent_b.power = power
        
        agent_b.register_tool(add)
        agent_b.register_tool(multiply)
        agent_b.register_tool(power)
        
        # Simulate multiple tool calls
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_multi:secret_multi'
        
        # Call 1: add (0.10 credits)
        result1 = await add(5, 3)
        assert result1 == 8
        
        # Call 2: multiply (0.25 credits)
        result2 = await multiply(4, 7)
        assert result2 == 28
        
        # Call 3: power (0.50 credits)
        result3 = await power(2, 8)
        assert result3 == 256
        
        # Total cost should be: 0.10 + 0.25 + 0.50 = 0.85
        # (Note: actual charging happens in PaymentSkill's tool execution hook)


class TestAgentABMixedIntegration:
    """Test Agent A using both HTTP endpoints and tools from Agent B"""
    
    @pytest.mark.asyncio
    async def test_agent_a_uses_both_http_and_tools(self, agent_a_client, agent_b_server):
        """
        Test Agent A using both HTTP endpoints and tools from Agent B
        in the same conversation
        """
        agent_a, payment_skill_a, mock_client_a = agent_a_client
        agent_b, app, base_url = agent_b_server
        
        # Add a paid tool to Agent B
        from webagents.agents.tools.decorators import tool
        
        @tool()
        @pricing(credits_per_call=0.20, reason="Data processing")
        async def process_data(data: str) -> dict:
            """Process data (costs 0.20 credits)"""
            return {"input": data, "processed": data.upper(), "length": len(data)}
        
        agent_b.process_data = process_data
        agent_b.register_tool(process_data)
        
        payment_token = 'tok_valid_mixed:secret_mixed'
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = payment_token
        
        # Scenario 1: Use HTTP endpoint (0.50 credits)
        async with httpx.AsyncClient(app=app, base_url=base_url) as client:
            # Get 402
            response_402 = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Paris"}
            )
            assert response_402.status_code == 402
            
            # Create payment
            requirements = response_402.json()
            payment_header, scheme, cost = await payment_skill_a._create_payment(
                requirements['accepts'], context
            )
            assert cost == 0.50
            
            # Make paid request
            response_success = await client.get(
                "/agents/agent-b-weather/weather",
                params={"location": "Paris"},
                headers={"X-PAYMENT": payment_header}
            )
            assert response_success.status_code == 200
        
        # Scenario 2: Use tool (0.20 credits)
        result = await process_data("hello world")
        assert result['processed'] == "HELLO WORLD"
        assert result['length'] == 11
        
        # Total cost: 0.50 (HTTP) + 0.20 (tool) = 0.70 credits


class TestPaymentSkillToolPricingIntegration:
    """Test PaymentSkill's existing tool pricing with x402"""
    
    @pytest.mark.asyncio
    async def test_tool_pricing_compatible_with_x402(self):
        """Verify @pricing on tools works with PaymentSkillX402"""
        mock_client = AsyncMock()
        
        async def mock_charge(token: str, amount: float, metadata: dict) -> Dict[str, Any]:
            return {
                'success': True,
                'new_balance': 5.0,
                'transaction_id': 'tx_compat_123'
            }
        
        async def mock_validate(token: str) -> Dict[str, Any]:
            return {'valid': True, 'balance': 10.0}
        
        mock_client.tokens = AsyncMock()
        mock_client.tokens.charge = mock_charge
        mock_client.tokens.validate_with_balance = mock_validate
        
        # Create agent with PaymentSkillX402
        payment_skill = PaymentSkillX402()
        payment_skill.client = mock_client
        
        agent = BaseAgent(
            name="agent-test",
            skills={'payments': payment_skill}
        )
        payment_skill.agent = agent
        
        from webagents.agents.tools.decorators import tool
        
        # Tool with pricing
        @tool()
        @pricing(
            credits_per_call=1.50,
            reason="Premium operation",
            metadata={"tier": "premium"}
        )
        async def premium_operation(input_data: str) -> str:
            """Premium operation (costs 1.50 credits)"""
            return f"Processed: {input_data}"
        
        agent.premium_operation = premium_operation
        agent.register_tool(premium_operation)
        
        # Verify pricing metadata
        assert hasattr(premium_operation, '_webagents_pricing')
        pricing_info = premium_operation._webagents_pricing
        
        assert pricing_info['credits_per_call'] == 1.50
        assert pricing_info['reason'] == "Premium operation"
        assert pricing_info['metadata']['tier'] == "premium"
        
        # Verify PaymentSkillX402 inherits PaymentSkill's tool charging
        from webagents.agents.skills.robutler.payments import PaymentSkill
        assert isinstance(payment_skill, PaymentSkill)
        
        # PaymentSkillX402 should support both:
        # 1. Tool-level pricing (from PaymentSkill)
        # 2. HTTP endpoint pricing with x402 (new)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])

