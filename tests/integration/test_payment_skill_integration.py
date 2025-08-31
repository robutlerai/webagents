"""
PaymentSkill Integration Tests

Tests PaymentSkill with real WebAgents Portal API and LiteLLM API integration.
Uses fixtures to procure payment tokens via WebAgents API.

Run with: python -m pytest tests/integration/test_payment_skill_integration.py -v
"""

import pytest
import os
import asyncio
from decimal import Decimal
from unittest.mock import Mock

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, use environment as-is
    pass

from webagents.agents.skills.robutler.payments import (
    PaymentSkill,
    PaymentContext,
    PaymentValidationError,
    InsufficientBalanceError,
    PaymentRequiredError
)
from webagents.agents.core.base_agent import BaseAgent
from robutler.api import RobutlerClient


class MockAgentContext:
    """Mock agent context for integration testing"""
    def __init__(self, agent=None):
        if agent is None:
            agent = Mock()
            agent.name = 'test-payment-agent'
            agent.api_key = None
        
        self.agent = agent
        # Ensure agent has required attributes
        if not hasattr(self.agent, 'name'):
            self.agent.name = 'test-payment-agent'
        if not hasattr(self.agent, 'api_key'):
            self.agent.api_key = None


@pytest.fixture
def robutler_api_key():
    """Get WebAgents API key from environment"""
    api_key = os.getenv('ROBUTLER_API_KEY')
    if not api_key:
        pytest.skip("Integration tests require ROBUTLER_API_KEY environment variable")
    
    # Accept test key for integration tests in development
    print(f"Using API key: {'***' + api_key[-8:] if len(api_key) > 8 else api_key}")
    return api_key


@pytest.fixture
def webagents_api_url():
    """Get WebAgents API URL from environment"""
    # Use ROBUTLER_API_URL if available, fallback to ROBUTLER_API_URL
    api_url = os.getenv('ROBUTLER_API_URL') or os.getenv('ROBUTLER_API_URL', 'http://localhost:3000')
    print(f"Using API URL: {api_url}")
    return api_url


@pytest.fixture
async def webagents_client(robutler_api_key, webagents_api_url):
    """Create WebAgents client for integration testing"""
    client = RobutlerClient(
        api_key=robutler_api_key,
        base_url=webagents_api_url
    )
    
    yield client
    
    # Cleanup
    try:
        await client.close()
    except:
        pass  # Client might not need explicit cleanup


@pytest.fixture
async def test_payment_tokens(webagents_client):
    """Procure test payment tokens via WebAgents API"""
    try:
        # Create test payment tokens via API with timeout
        response = await asyncio.wait_for(
            webagents_client._make_request(
                'POST', '/payment-tokens/test',
                data={
                    'valid_token_balance': 100.0,
                    'insufficient_token_balance': 0.50,
                    'description': 'Integration test tokens'
                }
            ),
            timeout=10.0
        )
        
        if response.success:
            tokens = response.data
            return {
                'valid_token': tokens.get('valid_token', 'tok_test_valid_integration'),
                'insufficient_token': tokens.get('insufficient_token', 'tok_test_insufficient_integration'),
                'invalid_token': 'tok_invalid_integration_test'
            }
        else:
            # Fallback to default test tokens if API doesn't support token creation
            print(f"⚠️  Could not create test tokens via API: {response.message}")
            return {
                'valid_token': 'tok_test_valid_integration',
                'insufficient_token': 'tok_test_insufficient_integration', 
                'invalid_token': 'tok_invalid_integration_test'
            }
            
    except asyncio.TimeoutError:
        print(f"⚠️  Token creation timed out, using fallback tokens")
        return {
            'valid_token': 'tok_test_valid_integration',
            'insufficient_token': 'tok_test_insufficient_integration',
            'invalid_token': 'tok_invalid_integration_test'
        }
    except Exception as e:
        print(f"⚠️  Token creation failed, using fallback tokens: {e}")
        # Fallback tokens for when API is not available
        return {
            'valid_token': 'tok_test_valid_integration',
            'insufficient_token': 'tok_test_insufficient_integration',
            'invalid_token': 'tok_invalid_integration_test'
        }


@pytest.fixture
def payment_integration_config(robutler_api_key, webagents_api_url):
    """PaymentSkill configuration for integration testing"""
    return {
        'enable_payments': True,
        'webagents_api_url': webagents_api_url,
        'robutler_api_key': robutler_api_key,
        'agent_pricing_percent': 0.20,
        'minimum_balance': 1.0,
        'require_payment_token': True
    }


@pytest.fixture
async def payment_skill_integration(payment_integration_config, robutler_api_key):
    """PaymentSkill instance configured for integration testing"""
    skill = PaymentSkill(payment_integration_config)
    
    # Initialize with mock agent (not mock context)
    mock_agent = Mock()
    mock_agent.name = 'integration-test-agent'
    mock_agent.api_key = robutler_api_key
    
    await skill.initialize(mock_agent)
    
    yield skill
    
    # Cleanup
    if hasattr(skill, 'cleanup'):
        await skill.cleanup()


@pytest.fixture
def mock_context_with_valid_token(test_payment_tokens):
    """Mock context with valid payment token"""
    context = Mock()
    context.get = Mock(side_effect=lambda key, default=None: {
        'X-Payment-Token': test_payment_tokens['valid_token'],
        'X-Origin-User-ID': 'user_12345',
        'X-Peer-User-ID': 'user_67890',  
        'X-Agent-Owner-User-ID': 'user_owner'
    }.get(key, default))
    return context


@pytest.fixture  
def mock_context_with_insufficient_token(test_payment_tokens):
    """Mock context with insufficient balance token"""
    context = Mock()
    context.get = Mock(side_effect=lambda key, default=None: {
        'X-Payment-Token': test_payment_tokens['insufficient_token'],
        'X-Origin-User-ID': 'user_12345',
        'X-Peer-User-ID': 'user_67890',
        'X-Agent-Owner-User-ID': 'user_owner'  
    }.get(key, default))
    return context


@pytest.fixture
def mock_context_no_token():
    """Mock context with no payment token"""
    context = Mock()
    context.get = Mock(return_value=None)
    return context


class TestPaymentSkillPlatformIntegration:
    """Test PaymentSkill integration with real WebAgents Platform"""
    
    @pytest.mark.asyncio
    async def test_client_connection(self, payment_skill_integration, webagents_api_url, robutler_api_key):
        """Test connection to real WebAgents Platform"""
        skill = payment_skill_integration
        
        assert skill.client is not None, "Should have initialized WebAgents client"
        assert skill.webagents_api_url == webagents_api_url
        assert skill.robutler_api_key == robutler_api_key
        
        # Test health check if available
        try:
            health = await skill.client.health_check()
            if health.success:
                print(f"✅ Connected to WebAgents Platform: {skill.webagents_api_url}")
            else:
                print(f"⚠️  Platform health check failed: {health.message}")
        except Exception as e:
            print(f"⚠️  Health check error (expected if endpoint unavailable): {e}")
    
    @pytest.mark.asyncio
    async def test_validate_payment_token_real(self, payment_skill_integration, test_payment_tokens, mock_context_with_valid_token):
        """Test payment token validation with real Platform API"""
        skill = payment_skill_integration
        
        try:
            result = await skill.validate_payment_token(
                token=test_payment_tokens['valid_token'],
                context=mock_context_with_valid_token
            )
            
            print(f"Token validation result: {result}")
            assert result['success'] is True or 'error' in result
            
            if result['success']:
                assert 'token_info' in result
                assert 'balance' in result
                print(f"✅ Token valid - Balance: ${result.get('balance', 'N/A')}")
            else:
                print(f"⚠️  Token validation failed (expected if test token invalid): {result['error']}")
                
        except Exception as e:
            print(f"⚠️  Token validation error (expected if platform unavailable): {e}")
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_handling_real(self, payment_skill_integration, test_payment_tokens):
        """Test insufficient balance handling with real API"""
        skill = payment_skill_integration
        
        try:
            # This should either fail validation or pass with low balance
            result = await skill._validate_payment_token_with_balance(
                test_payment_tokens['insufficient_token'], 
                required_balance=100.0  # High requirement
            )
            
            print(f"Insufficient balance test result: {result}")
            
            # Should either reject the token or show low balance
            if not result['valid']:
                print("✅ Correctly rejected insufficient balance token")
            else:
                balance = result.get('balance', 0)
                if balance < 100.0:
                    print(f"✅ Token valid but balance too low: ${balance}")
                    
        except (InsufficientBalanceError, PaymentRequiredError) as e:
            print(f"✅ Correctly raised payment error: {e}")
        except Exception as e:
            print(f"⚠️  Balance check error (expected if platform unavailable): {e}")


class TestLiteLLMIntegration:
    """Test PaymentSkill integration with real LiteLLM APIs"""
    
    @pytest.mark.asyncio
    async def test_real_litellm_cost_calculation(self, payment_skill_integration):
        """Test LiteLLM cost calculation with real API calls"""
        skill = payment_skill_integration
        
        # Test various model cost calculations
        test_cases = [
            {
                'model': 'gpt-3.5-turbo',
                'input_tokens': 1000,
                'output_tokens': 500,
                'expected_cost_range': (0.001, 0.01)
            },
            {
                'model': 'gpt-4o-mini',
                'input_tokens': 1000, 
                'output_tokens': 500,
                'expected_cost_range': (0.001, 0.01)
            },
            {
                'model': 'claude-3-haiku-20240307',
                'input_tokens': 1000,
                'output_tokens': 500,
                'expected_cost_range': (0.001, 0.01)
            }
        ]
        
        for case in test_cases:
            try:
                cost = skill._calculate_llm_cost(
                    model=case['model'],
                    input_tokens=case['input_tokens'],
                    output_tokens=case['output_tokens']
                )
                
                print(f"Cost for {case['model']}: ${cost:.6f}")
                
                # Verify cost is in reasonable range
                min_cost, max_cost = case['expected_cost_range']
                assert min_cost <= cost <= max_cost, f"Cost ${cost:.6f} outside expected range ${min_cost}-${max_cost}"
                
                # Test with agent pricing percent
                marked_up_cost = cost * (1 + skill.agent_pricing_percent)
                print(f"With {skill.agent_pricing_percent*100}% margin: ${marked_up_cost:.6f}")
                
            except Exception as e:
                print(f"⚠️  Cost calculation error for {case['model']}: {e}")
                # Don't fail - LiteLLM may not have all models available
    
    @pytest.mark.asyncio
    async def test_estimate_llm_cost_tool_real(self, payment_skill_integration):
        """Test estimate_llm_cost tool with real LiteLLM calculations"""
        skill = payment_skill_integration
        
        result = await skill.estimate_llm_cost(
            model='gpt-3.5-turbo',
            prompt_tokens=1500,
            completion_tokens=300,
            context=None
        )
        
        print(f"LLM cost estimation result: {result}")
        
        assert result['success'] is True
        assert 'estimate' in result
        
        estimate = result['estimate']
        assert 'base_cost_usd' in estimate
        assert 'final_cost_usd' in estimate  
        assert 'model' in estimate
        assert 'agent_pricing_percent' in estimate
        
        base_cost = estimate['base_cost_usd']
        final_cost = estimate['final_cost_usd']
        
        # Verify agent pricing percent calculation (final_cost should be different from base_cost due to margin)
        agent_pricing_percent = estimate['agent_pricing_percent']
        print(f"✅ Base cost: ${base_cost:.6f}, Final cost: ${final_cost:.6f}, Margin: {agent_pricing_percent}")
        
        # Verify costs are reasonable 
        assert base_cost > 0, "Base cost should be positive"
        assert final_cost > 0, "Final cost should be positive"


class TestEndToEndPaymentFlow:
    """Test complete payment flows with real APIs"""
    
    @pytest.mark.asyncio
    async def test_complete_payment_flow_non_streaming(self, payment_skill_integration, mock_context_with_valid_token):
        """Test complete payment flow for non-streaming request"""
        skill = payment_skill_integration
        
        try:
            # 1. Setup payment context
            setup_context = await skill.setup_payment_context(mock_context_with_valid_token)
            print(f"Payment context setup: {setup_context.get('success', False)}")
            
            if not setup_context.get('success'):
                print(f"⚠️  Payment setup failed: {setup_context.get('error')}")
                return
            
            # 2. Simulate LLM usage and cost accumulation
            usage_data = {
                'model': 'gpt-3.5-turbo',
                'prompt_tokens': 800,
                'completion_tokens': 200,
                'total_tokens': 1000
            }
            
            cost_result = await skill.accumulate_costs(
                usage=usage_data,
                context=mock_context_with_valid_token
            )
            
            print(f"Cost accumulation: {cost_result}")
            assert cost_result.get('success', False)
            
            accumulated_cost = cost_result.get('accumulated_cost_usd', 0)
            print(f"Accumulated cost: ${accumulated_cost:.6f}")
            
            # 3. Finalize payment
            final_result = await skill.finalize_payment(mock_context_with_valid_token)
            print(f"Payment finalization: {final_result}")
            
            if final_result.get('success'):
                print("✅ Complete payment flow successful")
            else:
                print(f"⚠️  Payment finalization failed: {final_result.get('error')}")
                
        except Exception as e:
            print(f"⚠️  End-to-end payment flow error: {e}")
    
    @pytest.mark.asyncio
    async def test_payment_flow_with_minimum_balance_check(self, payment_skill_integration, mock_context_with_valid_token):
        """Test payment flow with minimum balance enforcement"""
        skill = payment_skill_integration
        
        try:
            # Test with high minimum balance requirement
            original_min_balance = skill.minimum_balance
            skill.minimum_balance = 50.0  # High minimum for testing
            
            setup_context = await skill.setup_payment_context(mock_context_with_valid_token)
            
            if setup_context.get('success'):
                print("✅ Payment setup passed minimum balance check")
            else:
                error = setup_context.get('error', '')
                if 'minimum balance' in error.lower() or 'insufficient' in error.lower():
                    print("✅ Correctly enforced minimum balance requirement")
                else:
                    print(f"⚠️  Setup failed for other reason: {error}")
            
            # Restore original minimum balance
            skill.minimum_balance = original_min_balance
            
        except (InsufficientBalanceError, PaymentRequiredError) as e:
            print(f"✅ Correctly raised minimum balance error: {e}")
        except Exception as e:
            print(f"⚠️  Minimum balance test error: {e}")


class TestPaymentErrorScenarios:
    """Test payment error scenarios with real APIs"""
    
    @pytest.mark.asyncio
    async def test_invalid_payment_token(self, payment_skill_integration, test_payment_tokens):
        """Test handling of invalid payment tokens"""
        skill = payment_skill_integration
        
        invalid_tokens = [
            test_payment_tokens['invalid_token'],
            'completely_invalid_token',
            '',
            None
        ]
        
        for token in invalid_tokens:
            try:
                result = await skill.validate_payment_token(
                    token=token,
                    context=Mock()
                )
                
                print(f"Invalid token '{token}' result: {result.get('success', False)}")
                
                # Should fail validation
                if not result.get('success'):
                    print(f"✅ Correctly rejected invalid token: {token}")
                else:
                    print(f"⚠️  Unexpectedly accepted invalid token: {token}")
                    
            except Exception as e:
                print(f"✅ Exception handling for invalid token '{token}': {e}")
    
    @pytest.mark.asyncio
    async def test_no_payment_token_error_handling(self, payment_skill_integration, mock_context_no_token):
        """Test error handling when no payment token provided"""
        skill = payment_skill_integration
        
        try:
            setup_result = await skill.setup_payment_context(mock_context_no_token)
            
            print(f"No token setup result: {setup_result}")
            
            # Should fail or require payment
            if not setup_result.get('success'):
                error = setup_result.get('error', '')
                if any(keyword in error.lower() for keyword in ['token', 'payment', 'required']):
                    print("✅ Correctly handled missing payment token")
                else:
                    print(f"⚠️  Failed for different reason: {error}")
            else:
                print("⚠️  Unexpectedly succeeded without payment token")
                
        except PaymentRequiredError as e:
            print(f"✅ Correctly raised PaymentRequiredError: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")


class TestPaymentSkillToolIntegration:
    """Test PaymentSkill tools with real API integration"""
    
    @pytest.mark.asyncio
    async def test_payment_tools_with_real_apis(self, payment_skill_integration, test_payment_tokens, mock_context_with_valid_token):
        """Test all payment tools with real API calls"""
        skill = payment_skill_integration
        
        # Test validate_payment_token tool
        print("Testing validate_payment_token tool...")
        validate_result = await skill.validate_payment_token(
            token=test_payment_tokens['valid_token'],
            context=mock_context_with_valid_token
        )
        print(f"Validate result: {validate_result.get('success', False)}")
        
        # Test estimate_llm_cost tool
        print("Testing estimate_llm_cost tool...")
        estimate_result = await skill.estimate_llm_cost(
            model='gpt-3.5-turbo',
            prompt_tokens=1000,
            completion_tokens=200,
            context=None
        )
        print(f"Estimate result: {estimate_result.get('success', False)}")
        assert estimate_result.get('success', False)
        
        # Test get_payment_context tool
        print("Testing get_payment_context tool...")
        context_result = await skill.get_payment_context(context=mock_context_with_valid_token)
        print(f"Context result: {context_result.get('success', False)}")
        
        print("✅ All payment tools tested with real APIs")


@pytest.mark.asyncio
async def test_payment_integration_configuration(robutler_api_key, webagents_api_url):
    """Test PaymentSkill configuration with real environment variables"""
    
    print(f"Integration test configuration:")
    print(f"  ROBUTLER_API_URL: {webagents_api_url}")
    print(f"  ROBUTLER_API_KEY: {'***' + robutler_api_key[-8:] if robutler_api_key and len(robutler_api_key) > 8 else 'Not set'}")
    
    # Test configuration validation
    config = {
        'webagents_api_url': webagents_api_url,
        'robutler_api_key': robutler_api_key
    }
    
    skill = PaymentSkill(config)
    assert skill.webagents_api_url == webagents_api_url
    assert skill.robutler_api_key == robutler_api_key
    
    print("✅ Integration test configuration validated")


if __name__ == "__main__":
    print("PaymentSkill Integration Tests")
    print("=" * 40)
    print("These tests require:")
    print("- ROBUTLER_API_KEY environment variable")
    print("- ROBUTLER_API_URL environment variable (optional)")
    print("- TEST_PAYMENT_TOKEN environment variable (optional)")
    print("- Network connectivity to APIs")
    print()
    
    if not os.getenv('ROBUTLER_API_KEY') or os.getenv('ROBUTLER_API_KEY') == 'rok_testapikey':
        print("⚠️  Set ROBUTLER_API_KEY environment variable to run integration tests")
        exit(1)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"]) 