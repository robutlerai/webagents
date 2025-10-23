"""
Tests for WebAgents Tool and Pricing Decorators

Tests the @tool, @pricing, and other decorators functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Tuple

from webagents.agents.tools.decorators import tool, pricing, PricingInfo, hook, prompt, handoff, http


class TestPricingDecorator:
    """Test @pricing decorator functionality"""
    
    def test_pricing_info_dataclass(self):
        """Test PricingInfo dataclass creation and attributes"""
        # Test with all parameters
        pricing_info = PricingInfo(
            credits=150.5,
            reason="Data processing task",
            metadata={"complexity": "medium", "data_size": 1000},
            on_success=lambda: "success",
            on_fail=lambda: "failed"
        )
        
        assert pricing_info.credits == 150.5
        assert pricing_info.reason == "Data processing task"
        assert pricing_info.metadata["complexity"] == "medium"
        assert pricing_info.metadata["data_size"] == 1000
        assert pricing_info.on_success() == "success"
        assert pricing_info.on_fail() == "failed"
        
        # Test with minimal parameters
        minimal_pricing = PricingInfo(credits=100, reason="Simple task")
        assert minimal_pricing.credits == 100
        assert minimal_pricing.reason == "Simple task"
        assert minimal_pricing.metadata is None
        assert minimal_pricing.on_success is None
        assert minimal_pricing.on_fail is None
    
    def test_pricing_decorator_fixed_pricing(self):
        """Test @pricing decorator with fixed pricing"""
        @pricing(credits_per_call=500, reason="Fixed price service")
        def fixed_pricing_tool(param: str) -> str:
            return f"Result: {param}"
        
        # Check pricing metadata was attached
        assert hasattr(fixed_pricing_tool, '_webagents_pricing')
        pricing_meta = fixed_pricing_tool._webagents_pricing
        
        assert pricing_meta['credits_per_call'] == 500
        assert pricing_meta['reason'] == "Fixed price service"
        assert pricing_meta['supports_dynamic'] is False
        assert pricing_meta['on_success'] is None
        assert pricing_meta['on_fail'] is None
        
        # Test function still works
        result = fixed_pricing_tool("test")
        assert result == "Result: test"
    
    def test_pricing_decorator_dynamic_pricing(self):
        """Test @pricing decorator with dynamic pricing"""
        @pricing()  # No fixed pricing
        def dynamic_pricing_tool(data: str) -> Tuple[str, PricingInfo]:
            length = len(data)
            result = f"Processed {length} characters"
            pricing_info = PricingInfo(
                credits=length * 0.1,
                reason=f"Processing {length} characters"
            )
            return result, pricing_info
        
        # Check pricing metadata
        pricing_meta = dynamic_pricing_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] is None
        assert pricing_meta['supports_dynamic'] is True
        assert "Tool 'dynamic_pricing_tool' execution" in pricing_meta['reason']
        
        # Test function works and returns tuple
        result, pricing_info = dynamic_pricing_tool("hello world")
        assert result == "Processed 11 characters"
        assert isinstance(pricing_info, PricingInfo)
        assert pricing_info.credits == 1.1  # 11 * 0.1
        assert pricing_info.reason == "Processing 11 characters"
    
    def test_pricing_decorator_with_callbacks(self):
        """Test @pricing decorator with success/failure callbacks"""
        success_called = False
        fail_called = False
        
        def on_success():
            nonlocal success_called
            success_called = True
        
        def on_fail():
            nonlocal fail_called
            fail_called = True
        
        @pricing(credits_per_call=200, on_success=on_success, on_fail=on_fail)
        def callback_tool() -> str:
            return "Done"
        
        # Check callback metadata
        pricing_meta = callback_tool._webagents_pricing
        assert pricing_meta['on_success'] is on_success
        assert pricing_meta['on_fail'] is on_fail
        
        # Function should still work
        result = callback_tool()
        assert result == "Done"
    
    @pytest.mark.asyncio
    async def test_pricing_decorator_async_function(self):
        """Test @pricing decorator with async functions"""
        @pricing(credits_per_call=300, reason="Async processing")
        async def async_pricing_tool(value: int) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return value * 2
        
        # Check metadata
        pricing_meta = async_pricing_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] == 300
        assert pricing_meta['reason'] == "Async processing"
        
        # Test async function works
        result = await async_pricing_tool(21)
        assert result == 42
    
    def test_pricing_decorator_without_parameters(self):
        """Test @pricing() decorator without parameters defaults to dynamic pricing"""
        @pricing()
        def no_params_tool() -> str:
            return "No params"
        
        pricing_meta = no_params_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] is None
        assert pricing_meta['supports_dynamic'] is True
        assert "Tool 'no_params_tool' execution" in pricing_meta['reason']


class TestToolAndPricingIntegration:
    """Test integration between @tool and @pricing decorators"""
    
    def test_tool_with_fixed_pricing(self):
        """Test @tool combined with fixed @pricing"""
        @tool(name="weather_service", description="Get weather data")
        @pricing(credits_per_call=1000, reason="Weather API call")
        def weather_tool(location: str) -> str:
            return f"Weather for {location}"
        
        # Check tool metadata
        assert hasattr(weather_tool, '_webagents_is_tool')
        assert weather_tool._webagents_is_tool is True
        assert weather_tool._tool_name == "weather_service"
        assert weather_tool._tool_description == "Get weather data"
        
        # Check pricing metadata
        assert hasattr(weather_tool, '_webagents_pricing')
        pricing_meta = weather_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] == 1000
        assert pricing_meta['reason'] == "Weather API call"
        
        # Function should work
        result = weather_tool("New York")
        assert result == "Weather for New York"
    
    def test_tool_with_dynamic_pricing(self):
        """Test @tool combined with dynamic @pricing"""
        @tool(description="Analyze text content")
        @pricing()
        def analyze_tool(text: str) -> Tuple[str, PricingInfo]:
            word_count = len(text.split())
            result = f"Analysis: {word_count} words"
            pricing_info = PricingInfo(
                credits=word_count * 2,
                reason=f"Word analysis ({word_count} words)"
            )
            return result, pricing_info
        
        # Check both decorators applied
        assert hasattr(analyze_tool, '_webagents_is_tool')
        assert hasattr(analyze_tool, '_webagents_pricing')
        
        # Test function
        result, pricing_info = analyze_tool("hello world test")
        assert result == "Analysis: 3 words"
        assert pricing_info.credits == 6  # 3 words * 2
    
    def test_tool_without_pricing(self):
        """Test @tool without @pricing decorator"""
        @tool(name="free_service")
        def free_tool(data: str) -> str:
            return f"Free: {data}"
        
        # Should have tool metadata but no pricing metadata
        assert hasattr(free_tool, '_webagents_is_tool')
        assert not hasattr(free_tool, '_webagents_pricing')
        
        # Function should work
        result = free_tool("test")
        assert result == "Free: test"


class TestOtherDecorators:
    """Test other decorators for completeness"""
    
    def test_hook_decorator(self):
        """Test @hook decorator"""
        @hook("on_message", priority=10, scope="owner")
        def message_hook(context):
            return context
        
        assert hasattr(message_hook, '_webagents_is_hook')
        assert message_hook._hook_event_type == "on_message"
        assert message_hook._hook_priority == 10
        assert message_hook._hook_scope == "owner"
    
    def test_prompt_decorator(self):
        """Test @prompt decorator"""
        @prompt(priority=20, scope="all")
        def system_prompt(context):
            return "System prompt text"
        
        assert hasattr(system_prompt, '_webagents_is_prompt')
        assert system_prompt._prompt_priority == 20
        assert system_prompt._prompt_scope == "all"
    
    def test_handoff_decorator(self):
        """Test @handoff decorator"""
        @handoff(name="escalate", handoff_type="agent", scope="admin")
        def escalation_handoff(issue: str):
            return f"Escalated: {issue}"
        
        assert hasattr(escalation_handoff, '_webagents_is_handoff')
        assert escalation_handoff._handoff_name == "escalate"
        assert escalation_handoff._handoff_type == "agent"
        assert escalation_handoff._handoff_scope == "admin"
    
    def test_http_decorator(self):
        """Test @http decorator"""
        @http("/api/data", method="post", scope="owner")
        def http_endpoint(data: dict):
            return {"received": data}
        
        assert hasattr(http_endpoint, '_webagents_is_http')
        assert http_endpoint._http_subpath == "/api/data"
        assert http_endpoint._http_method == "post"
        assert http_endpoint._http_scope == "owner"


class TestDecoratorEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_multiple_pricing_decorators(self):
        """Test that multiple @pricing decorators work (first one wins)"""
        @pricing(credits_per_call=100)
        @pricing(credits_per_call=200, reason="Override")
        def multi_pricing_tool():
            return "test"
        
        # First decorator should win (innermost application)
        pricing_meta = multi_pricing_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] == 100
        # The first decorator doesn't override the reason, so it uses default
        assert "Tool 'multi_pricing_tool' execution" in pricing_meta['reason']
    
    def test_pricing_with_zero_credits(self):
        """Test @pricing with zero credits"""
        @pricing(credits_per_call=0, reason="Free tier")
        def zero_cost_tool():
            return "free"
        
        pricing_meta = zero_cost_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] == 0
        assert pricing_meta['reason'] == "Free tier"
    
    def test_pricing_with_negative_credits(self):
        """Test @pricing with negative credits (edge case)"""
        @pricing(credits_per_call=-50, reason="Refund operation")
        def refund_tool():
            return "refunded"
        
        pricing_meta = refund_tool._webagents_pricing
        assert pricing_meta['credits_per_call'] == -50
        assert pricing_meta['reason'] == "Refund operation"
    
    def test_pricing_decorator_preserves_function_attributes(self):
        """Test that @pricing preserves original function attributes"""
        def original_function():
            """Original docstring"""
            return "original"
        
        original_function.custom_attr = "custom_value"
        
        decorated = pricing(credits_per_call=100)(original_function)
        
        # Should preserve name and docstring
        assert decorated.__name__ == original_function.__name__
        assert decorated.__doc__ == original_function.__doc__
        
        # Should have pricing metadata
        assert hasattr(decorated, '_webagents_pricing')


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 