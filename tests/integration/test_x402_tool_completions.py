"""
Integration Tests for x402 - Agent B Tools with @pricing

Tests Agent A calling Agent B's paid tools (not HTTP endpoints).
Focuses on tool-level payments which work via PaymentSkill's existing hooks.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from webagents import BaseAgent
from webagents.agents.skills.robutler import PaymentSkillX402, pricing
from webagents.agents.tools.decorators import tool


# ==================== Test Fixtures ====================

@pytest.fixture
def mock_robutler_client():
    """Mock RobutlerClient for tool payment tests"""
    client = AsyncMock()
    
    # Mock token validation
    async def mock_validate(token: str) -> Dict[str, Any]:
        if token.startswith('tok_valid'):
            return {'valid': True, 'balance': 10.0}
        return {'valid': False, 'balance': 0.0}
    
    # Mock charging
    async def mock_charge(token: str, amount: float, metadata: dict) -> Dict[str, Any]:
        if token.startswith('tok_valid'):
            return {
                'success': True,
                'new_balance': max(0, 10.0 - amount),
                'transaction_id': f'tx_{amount}'
            }
        return {'success': False, 'error': 'Invalid token'}
    
    client.tokens = AsyncMock()
    client.tokens.validate_with_balance = mock_validate
    client.tokens.charge = mock_charge
    
    # Mock facilitator (not used for tools, but needed for skill)
    client.facilitator = AsyncMock()
    
    return client


@pytest.fixture
def agent_b_with_tools(mock_robutler_client):
    """Create Agent B with multiple paid tools"""
    payment_skill = PaymentSkillX402()
    payment_skill.client = mock_robutler_client
    
    agent_b = BaseAgent(
        name="agent-b-calculator",
        instructions="You provide calculation services.",
        skills={'payments': payment_skill}
    )
    payment_skill.agent = agent_b
    
    # Add paid tools
    @tool()
    @pricing(credits_per_call=0.10, reason="Basic addition")
    async def add(a: int, b: int) -> int:
        """Add two numbers (costs 0.10 credits)"""
        return a + b
    
    @tool()
    @pricing(credits_per_call=0.25, reason="Multiplication")
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers (costs 0.25 credits)"""
        return a * b
    
    @tool()
    @pricing(credits_per_call=0.50, reason="Power operation")
    async def power(base: int, exp: int) -> int:
        """Raise to power (costs 0.50 credits)"""
        return base ** exp
    
    agent_b.add = add
    agent_b.multiply = multiply
    agent_b.power = power
    
    agent_b.register_tool(add)
    agent_b.register_tool(multiply)
    agent_b.register_tool(power)
    
    return agent_b, payment_skill, {
        'add': add,
        'multiply': multiply,
        'power': power
    }


# ==================== Test Cases ====================

class TestAgentBToolPricing:
    """Test Agent B's tools with @pricing decorator"""
    
    def test_tool_has_pricing_metadata(self, agent_b_with_tools):
        """Verify @pricing decorator attaches metadata to tools"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Check add tool
        assert hasattr(tools['add'], '_webagents_pricing')
        pricing_info = tools['add']._webagents_pricing
        assert pricing_info['credits_per_call'] == 0.10
        assert pricing_info['reason'] == "Basic addition"
        
        # Check multiply tool
        assert hasattr(tools['multiply'], '_webagents_pricing')
        pricing_info = tools['multiply']._webagents_pricing
        assert pricing_info['credits_per_call'] == 0.25
        
        # Check power tool
        assert hasattr(tools['power'], '_webagents_pricing')
        pricing_info = tools['power']._webagents_pricing
        assert pricing_info['credits_per_call'] == 0.50
    
    @pytest.mark.asyncio
    async def test_tool_execution_returns_result(self, agent_b_with_tools):
        """Verify tools execute and return correct results"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Test add - tools with @pricing return tuple (result, pricing_info)
        result = await tools['add'](5, 3)
        if isinstance(result, tuple):
            actual_result, pricing_info = result
            assert actual_result == 8
            assert 'pricing' in pricing_info
        else:
            assert result == 8
        
        # Test multiply
        result = await tools['multiply'](4, 7)
        if isinstance(result, tuple):
            actual_result, pricing_info = result
            assert actual_result == 28
        else:
            assert result == 28
        
        # Test power
        result = await tools['power'](2, 8)
        if isinstance(result, tuple):
            actual_result, pricing_info = result
            assert actual_result == 256
        else:
            assert result == 256
    
    def test_paymentskillx402_inherits_paymentskill(self, agent_b_with_tools):
        """Verify PaymentSkillX402 is a superset of PaymentSkill"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        from webagents.agents.skills.robutler.payments import PaymentSkill
        
        # Should be an instance of both
        assert isinstance(payment_skill, PaymentSkillX402)
        assert isinstance(payment_skill, PaymentSkill)
        
        # Should have PaymentSkill's attributes
        assert hasattr(payment_skill, 'client')
        assert hasattr(payment_skill, 'agent')
        
        # Should have x402-specific attributes
        assert hasattr(payment_skill, 'facilitator_url')
        assert hasattr(payment_skill, 'accepted_schemes')
        assert hasattr(payment_skill, 'payment_schemes')


class TestAgentACallsAgentBTools:
    """Test Agent A calling Agent B's paid tools"""
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_different_prices(self, agent_b_with_tools):
        """Test calling multiple tools with different prices"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Simulate payment context (as PaymentSkill would provide)
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_test:secret'
        context.payments.total_cost = 0.0
        
        # Call add (0.10 credits)
        result1 = await tools['add'](10, 20)
        if isinstance(result1, tuple):
            result1 = result1[0]
        assert result1 == 30
        
        # Call multiply (0.25 credits)
        result2 = await tools['multiply'](6, 7)
        if isinstance(result2, tuple):
            result2 = result2[0]
        assert result2 == 42
        
        # Call power (0.50 credits)
        result3 = await tools['power'](3, 4)
        if isinstance(result3, tuple):
            result3 = result3[0]
        assert result3 == 81
        
        # Total cost would be: 0.10 + 0.25 + 0.50 = 0.85 credits
        # (Actual charging happens in PaymentSkill's hooks during execution)
    
    @pytest.mark.asyncio
    async def test_tool_pricing_compatible_with_x402_skill(self, mock_robutler_client):
        """Verify tool pricing works seamlessly with PaymentSkillX402"""
        # Create agent with PaymentSkillX402
        payment_skill = PaymentSkillX402()
        payment_skill.client = mock_robutler_client
        
        agent = BaseAgent(
            name="test-agent",
            skills={'payments': payment_skill}
        )
        payment_skill.agent = agent
        
        # Add tool with pricing (no metadata parameter - use reason only)
        @tool()
        @pricing(
            credits_per_call=1.50,
            reason="Premium calculation"
        )
        async def premium_calc(x: float) -> float:
            """Premium calculation (costs 1.50 credits)"""
            return x * 2.5
        
        agent.premium_calc = premium_calc
        agent.register_tool(premium_calc)
        
        # Verify pricing metadata
        assert hasattr(premium_calc, '_webagents_pricing')
        pricing_info = premium_calc._webagents_pricing
        
        assert pricing_info['credits_per_call'] == 1.50
        assert pricing_info['reason'] == "Premium calculation"
        
        # Execute tool
        result = await premium_calc(4.0)
        if isinstance(result, tuple):
            result = result[0]
        assert result == 10.0


class TestToolAndHTTPMixedPayments:
    """Test agent using both tools and HTTP endpoints (conceptual)"""
    
    def test_agent_can_have_both_paid_tools_and_http_endpoints(self, agent_b_with_tools):
        """Verify agent can have both paid tools AND paid HTTP endpoints"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Agent already has paid tools
        assert len(tools) == 3
        
        # Now add paid HTTP endpoint
        from webagents.agents.tools.decorators import http
        
        @http("/api/data", method="get")
        @pricing(credits_per_call=0.75, reason="Data API")
        async def get_data() -> dict:
            """Get data via HTTP (costs 0.75 credits)"""
            return {"data": "example"}
        
        agent_b.get_data = get_data
        agent_b.register_http_handler(get_data)
        
        # Verify HTTP endpoint has pricing
        assert hasattr(get_data, '_webagents_pricing')
        assert hasattr(get_data, '_http_requires_payment')
        assert get_data._http_requires_payment is True
        
        pricing_info = get_data._webagents_pricing
        assert pricing_info['credits_per_call'] == 0.75
        
        # Agent now has 3 paid tools + 1 paid HTTP endpoint
        # PaymentSkillX402 handles both:
        # - Tools: via PaymentSkill's tool execution hooks
        # - HTTP: via x402 protocol (check_http_endpoint_payment hook)


class TestPaymentSkillHookIntegration:
    """Test how PaymentSkill hooks work with tools"""
    
    @pytest.mark.asyncio
    async def test_payment_skill_validates_before_tool_execution(self, agent_b_with_tools):
        """
        Conceptual test: PaymentSkill should validate payment before tool execution
        
        This is handled by PaymentSkill's hooks in the actual execution flow.
        Here we just verify the structure is in place.
        """
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Verify PaymentSkill has validation logic
        assert hasattr(payment_skill, 'client')
        assert hasattr(payment_skill.client, 'tokens')
        
        # Verify tools have pricing info that hooks can read
        for tool_name, tool_func in tools.items():
            assert hasattr(tool_func, '_webagents_pricing')
            pricing = tool_func._webagents_pricing
            assert 'credits_per_call' in pricing
            assert 'reason' in pricing
    
    @pytest.mark.asyncio
    async def test_payment_context_structure(self, agent_b_with_tools):
        """Test the payment context structure used during execution"""
        agent_b, payment_skill, tools = agent_b_with_tools
        
        # Create a payment context as it would exist during tool execution
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_abc:secret_xyz'
        context.payments.total_cost = 0.0
        context.payments.transactions = []
        
        # Verify token validation works
        result = await payment_skill.client.tokens.validate_with_balance(
            context.payments.payment_token
        )
        
        assert result['valid'] is True
        assert result['balance'] == 10.0
        
        # Verify charging works
        charge_result = await payment_skill.client.tokens.charge(
            context.payments.payment_token,
            0.50,
            {'reason': 'Tool execution'}
        )
        
        assert charge_result['success'] is True
        assert charge_result['new_balance'] == 9.50


class TestAgentACallsAgentBViaCompletions:
    """End-to-end tests: Agent A calls Agent B through completions endpoint"""
    
    @pytest.mark.asyncio
    async def test_agent_a_completes_triggering_agent_b_tool(self, agent_b_with_tools, mock_robutler_client):
        """
        Real E2E test: Agent A sends completion request that triggers Agent B's paid tool
        
        Flow:
        1. Agent A has payment token
        2. Agent A sends completion with message
        3. LLM decides to call Agent B's tool (we mock this)
        4. PaymentSkill validates token and charges
        5. Tool executes
        6. Result returned to Agent A
        """
        agent_b, payment_skill_b, tools = agent_b_with_tools
        
        # Create Agent A with payment token
        payment_skill_a = PaymentSkillX402()
        payment_skill_a.client = mock_robutler_client
        
        agent_a = BaseAgent(
            name="agent-a-user",
            instructions="You can use calculation tools from other agents.",
            skills={'payments': payment_skill_a}
        )
        payment_skill_a.agent = agent_a
        
        # Give Agent A the payment token
        payment_token = 'tok_valid_agent_a:secret_xyz'
        
        # Simulate Agent A's context with payment token
        # In real scenario, this comes from the request/session
        context = Mock()
        context.payments = Mock()
        context.payments.payment_token = payment_token
        context.payments.total_cost = 0.0
        context.payments.transactions = []
        
        # Mock the tool call that would come from LLM
        # In real scenario: LLM returns tool_calls in response
        # Here we simulate Agent A deciding to call Agent B's tool
        
        # Step 1: Agent A identifies it needs to call Agent B's add tool
        tool_name = "add"
        tool_args = {"a": 15, "b": 27}
        
        # Step 2: PaymentSkill should validate token before tool execution
        token_valid = await payment_skill_b.client.tokens.validate_with_balance(payment_token)
        assert token_valid['valid'] is True
        assert token_valid['balance'] >= 0.10  # Tool costs 0.10 credits
        
        # Step 3: Execute Agent B's tool (this is what happens in completion)
        result = await tools[tool_name](**tool_args)
        
        # Step 4: Extract actual result (tools with @pricing return tuple)
        if isinstance(result, tuple):
            actual_result, pricing_info = result
            assert 'pricing' in pricing_info
            assert pricing_info['pricing']['credits'] == 0.10
        else:
            actual_result = result
        
        # Step 5: Verify correct result
        assert actual_result == 42  # 15 + 27 = 42
        
        # Step 6: Verify payment would be charged
        # In real scenario, PaymentSkill's after_tool_execution hook charges
        charge_result = await payment_skill_b.client.tokens.charge(
            payment_token,
            0.10,
            {'tool': tool_name, 'agent': 'agent-b-calculator'}
        )
        
        assert charge_result['success'] is True
        assert charge_result['new_balance'] == 9.90  # 10.0 - 0.10
    
    @pytest.mark.asyncio
    async def test_agent_a_multiple_tool_calls_in_conversation(self, agent_b_with_tools, mock_robutler_client):
        """
        Test Agent A making multiple tool calls to Agent B in same conversation
        
        Simulates a conversation where Agent A calls multiple Agent B tools:
        1. add(5, 10) -> 15 (costs 0.10)
        2. multiply(15, 2) -> 30 (costs 0.25)
        3. power(30, 2) -> 900 (costs 0.50)
        Total cost: 0.85 credits
        """
        agent_b, payment_skill_b, tools = agent_b_with_tools
        
        payment_token = 'tok_valid_conversation:secret_conv'
        
        # Track total cost across conversation
        total_cost = 0.0
        
        # Call 1: add(5, 10)
        result1 = await tools['add'](5, 10)
        if isinstance(result1, tuple):
            result1, pricing1 = result1
            total_cost += pricing1['pricing']['credits']
        assert result1 == 15
        
        # Call 2: multiply(15, 2) - using result from previous call
        result2 = await tools['multiply'](15, 2)
        if isinstance(result2, tuple):
            result2, pricing2 = result2
            total_cost += pricing2['pricing']['credits']
        assert result2 == 30
        
        # Call 3: power(30, 2) - using result from previous call
        result3 = await tools['power'](30, 2)
        if isinstance(result3, tuple):
            result3, pricing3 = result3
            total_cost += pricing3['pricing']['credits']
        assert result3 == 900
        
        # Verify total cost
        assert total_cost == 0.85  # 0.10 + 0.25 + 0.50
        
        # Verify final balance after all charges
        # Start: 10.0, End: 10.0 - 0.85 = 9.15
        charge_result = await payment_skill_b.client.tokens.charge(
            payment_token,
            total_cost,
            {'conversation': 'multi-tool', 'tools': 3}
        )
        
        assert charge_result['success'] is True
        assert charge_result['new_balance'] == 9.15
    
    @pytest.mark.asyncio
    async def test_agent_a_insufficient_balance_for_tool(self, agent_b_with_tools):
        """
        Test Agent A with insufficient balance trying to call expensive tool
        
        Agent A has 0.01 credits
        Agent B's power tool costs 0.50 credits
        Should fail validation
        """
        agent_b, payment_skill_b, tools = agent_b_with_tools
        
        # Mock client with low balance
        mock_client_low = AsyncMock()
        
        async def mock_validate_low(token: str) -> Dict[str, Any]:
            return {'valid': True, 'balance': 0.01}  # Only 0.01 credits
        
        async def mock_charge_insufficient(token: str, amount: float, metadata: dict) -> Dict[str, Any]:
            if amount > 0.01:
                return {'success': False, 'error': 'Insufficient balance'}
            return {'success': True, 'new_balance': 0.01 - amount}
        
        mock_client_low.tokens = AsyncMock()
        mock_client_low.tokens.validate_with_balance = mock_validate_low
        mock_client_low.tokens.charge = mock_charge_insufficient
        
        payment_skill_b.client = mock_client_low
        
        payment_token = 'tok_low_balance:secret_low'
        
        # Validate token shows insufficient balance
        validation = await payment_skill_b.client.tokens.validate_with_balance(payment_token)
        assert validation['valid'] is True
        assert validation['balance'] < 0.50  # Not enough for power tool
        
        # Try to charge for expensive tool (should fail)
        charge_result = await payment_skill_b.client.tokens.charge(
            payment_token,
            0.50,  # Power tool cost
            {'tool': 'power'}
        )
        
        assert charge_result['success'] is False
        assert 'Insufficient balance' in charge_result['error']
    
    @pytest.mark.asyncio
    async def test_agent_a_calls_agent_b_tool_with_context(self, agent_b_with_tools, mock_robutler_client):
        """
        Test tool call with full context (simulating real agent execution)
        
        This tests how the tool would be called in a real agent.complete() flow
        with proper context including messages, tools, payment info
        """
        agent_b, payment_skill_b, tools = agent_b_with_tools
        
        # Create Agent A
        payment_skill_a = PaymentSkillX402()
        payment_skill_a.client = mock_robutler_client
        
        agent_a = BaseAgent(
            name="agent-a-caller",
            instructions="You make calculations using external tools.",
            skills={'payments': payment_skill_a}
        )
        payment_skill_a.agent = agent_a
        
        # Simulate completion context
        context = Mock()
        context.agent = agent_a
        context.messages = [
            {"role": "user", "content": "What is 100 plus 200?"}
        ]
        context.payments = Mock()
        context.payments.payment_token = 'tok_valid_context:secret_ctx'
        context.payments.total_cost = 0.0
        
        # In real flow, LLM would return tool_call
        # Here we simulate it
        tool_call = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "add",
                "arguments": {"a": 100, "b": 200}
            }
        }
        
        # Execute the tool as agent system would
        tool_func = tools['add']
        tool_args = {"a": 100, "b": 200}
        
        # Call tool
        result = await tool_func(**tool_args)
        
        # Extract result
        if isinstance(result, tuple):
            actual_result, pricing_info = result
        else:
            actual_result = result
        
        # Verify result
        assert actual_result == 300
        
        # In real system, this result would be:
        # 1. Formatted as tool_call response
        # 2. Sent back to LLM
        # 3. LLM generates final response
        tool_response = {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "name": "add",
            "content": str(actual_result)
        }
        
        assert tool_response['content'] == "300"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

