"""
Test BaseAgent HTTP handlers and capabilities system
"""

import pytest
from unittest.mock import Mock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill, Handoff, HandoffResult
from webagents.agents.tools.decorators import tool, hook, handoff, http
from webagents.server.context.context_vars import Context


# Mock skill for testing
class MockSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.agent = None
        self.initialized = False
    
    async def initialize(self, agent_context):
        self.agent_context = agent_context
        self.agent = agent_context
        self.initialized = True


# Test HTTP handler functions
@http("/weather", method="get", scope="owner")
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location"""
    return {
        "location": location,
        "temperature": 25,
        "units": units,
        "status": "sunny"
    }


@http("/data", method="post")
async def post_data(data: dict) -> dict:
    """Post data endpoint"""
    return {
        "received": data,
        "status": "success",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@http("/admin/stats", method="get", scope="admin")
def get_admin_stats() -> dict:
    """Admin-only statistics endpoint"""
    return {
        "users": 100,
        "requests": 1000,
        "uptime": "24h"
    }


# Test other decorated functions for capabilities
@tool(scope="owner")
def example_tool(message: str) -> str:
    """Example tool for capabilities testing"""
    return f"Tool says: {message}"


@hook("on_request", priority=10)
async def example_hook(context):
    """Example hook for capabilities testing"""
    context.set("hook_executed", True)
    return context


@handoff(handoff_type="agent")
async def example_handoff(target: str) -> HandoffResult:
    """Example handoff for capabilities testing"""
    return HandoffResult(
        result=f"Handoff to {target}",
        handoff_type="agent",
        success=True
    )


class TestHTTPDecorator:
    """Test @http decorator functionality"""
    
    def test_http_decorator_basic(self):
        """Test basic @http decorator functionality"""
        @http("/test")
        def test_handler():
            return {"test": "success"}
        
        assert hasattr(test_handler, '_webagents_is_http')
        assert test_handler._webagents_is_http is True
        assert test_handler._http_subpath == "/test"
        assert test_handler._http_method == "get"  # default
        assert test_handler._http_scope == "all"   # default
    
    def test_http_decorator_with_params(self):
        """Test @http decorator with custom parameters"""
        assert get_weather._webagents_is_http is True
        assert get_weather._http_subpath == "/weather"
        assert get_weather._http_method == "get"
        assert get_weather._http_scope == "owner"
        assert "Get weather for a location" in get_weather._http_description
    
    def test_http_decorator_post_method(self):
        """Test @http decorator with POST method"""
        assert post_data._webagents_is_http is True
        assert post_data._http_subpath == "/data"
        assert post_data._http_method == "post"
        assert post_data._http_scope == "all"
    
    def test_http_decorator_invalid_method(self):
        """Test @http decorator with invalid HTTP method"""
        with pytest.raises(ValueError, match="Invalid HTTP method 'INVALID'"):
            @http("/test", method="INVALID")
            def invalid_handler():
                pass
    
    def test_http_decorator_subpath_normalization(self):
        """Test that subpaths are normalized to start with /"""
        @http("test")  # No leading slash
        def test_handler():
            return {}
        
        assert test_handler._http_subpath == "/test"


class TestBaseAgentHTTP:
    """Test BaseAgent HTTP handler functionality"""
    
    def test_http_handler_registration(self):
        """Test HTTP handler registration via __init__"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[get_weather, post_data]
        )
        
        handlers = agent.get_all_http_handlers()
        assert len(handlers) == 2
        
        # Check handler details
        handler_paths = {h['subpath'] for h in handlers}
        assert "/weather" in handler_paths
        assert "/data" in handler_paths
        
        handler_methods = {h['method'] for h in handlers}
        assert "get" in handler_methods
        assert "post" in handler_methods
    
    def test_http_handler_conflict_detection(self):
        """Test HTTP handler conflict detection"""
        @http("/weather")
        def duplicate_weather():
            return {}
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[get_weather]
        )
        
        # Should raise error when trying to register conflicting handler
        with pytest.raises(ValueError, match="HTTP handler conflict"):
            agent.register_http_handler(duplicate_weather)
    
    def test_http_handler_core_path_conflict(self):
        """Test that HTTP handlers can't use core paths"""
        @http("/chat/completions")
        def conflicting_handler():
            return {}
        
        agent = BaseAgent(name="test-agent", instructions="Test agent")
        
        with pytest.raises(ValueError, match="conflicts with core handler"):
            agent.register_http_handler(conflicting_handler)
    
    def test_http_handler_scope_filtering(self):
        """Test HTTP handler scope filtering"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[get_weather, get_admin_stats, post_data]
        )
        
        # Test different scope access
        all_handlers = agent.get_http_handlers_for_scope("all")
        owner_handlers = agent.get_http_handlers_for_scope("owner")
        admin_handlers = agent.get_http_handlers_for_scope("admin")
        
        # All scope should only see public handlers
        all_paths = {h['subpath'] for h in all_handlers}
        assert "/data" in all_paths  # scope="all"
        assert "/weather" not in all_paths  # scope="owner"
        assert "/admin/stats" not in all_paths  # scope="admin"
        
        # Owner scope should see owner + public handlers
        owner_paths = {h['subpath'] for h in owner_handlers}
        assert "/data" in owner_paths
        assert "/weather" in owner_paths
        assert "/admin/stats" not in owner_paths
        
        # Admin scope should see all handlers
        admin_paths = {h['subpath'] for h in admin_handlers}
        assert "/data" in admin_paths
        assert "/weather" in admin_paths
        assert "/admin/stats" in admin_paths
    
    def test_http_handler_registration_without_decorator(self):
        """Test that undecorated functions can't be registered as HTTP handlers"""
        def plain_function():
            return {}
        
        agent = BaseAgent(name="test-agent", instructions="Test agent")
        
        with pytest.raises(ValueError, match="not decorated with @http"):
            agent.register_http_handler(plain_function)


class TestCapabilitiesSystem:
    """Test the capabilities auto-registration system"""
    
    def test_capabilities_auto_registration(self):
        """Test that capabilities are auto-registered based on decorator type"""
        capabilities = [
            example_tool,      # @tool
            example_hook,      # @hook
            example_handoff,   # @handoff
            get_weather       # @http
        ]
        
        agent = BaseAgent(
            name="capabilities-agent",
            instructions="Test capabilities",
            capabilities=capabilities
        )
        
        # Check that each capability was registered in the correct registry
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_hooks) == 1
        assert len(agent._registered_handoffs) == 1
        assert len(agent._registered_http_handlers) == 1
        
        # Verify correct registration
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "example_tool" in tool_names
        
        hook_events = set(agent._registered_hooks.keys())
        assert "on_request" in hook_events
        
        handoff_targets = {h['config'].target for h in agent._registered_handoffs}
        assert "example_handoff" in handoff_targets
        
        http_paths = {h['subpath'] for h in agent._registered_http_handlers}
        assert "/weather" in http_paths
    
    def test_capabilities_mixed_with_explicit_params(self):
        """Test capabilities combined with explicit parameter registration"""
        @tool
        def capabilities_tool():
            return "capabilities"
        
        @tool
        def explicit_tool():
            return "explicit"
        
        agent = BaseAgent(
            name="mixed-agent",
            instructions="Mixed registration test",
            tools=[explicit_tool],  # Explicit registration
            capabilities=[capabilities_tool]  # Auto-registration
        )
        
        # Both should be registered
        assert len(agent._registered_tools) == 2
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "explicit_tool" in tool_names
        assert "capabilities_tool" in tool_names
    
    def test_capabilities_with_undecorated_function(self):
        """Test that undecorated functions in capabilities are ignored"""
        def plain_function():
            return "plain"
        
        agent = BaseAgent(
            name="plain-agent",
            instructions="Plain function test",
            capabilities=[example_tool, plain_function]  # Mixed decorated and plain
        )
        
        # Only decorated function should be registered
        assert len(agent._registered_tools) == 1
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "example_tool" in tool_names


class TestDirectRegistration:
    """Test direct registration methods (@agent.tool, @agent.http, etc.)"""
    
    def test_agent_tool_decorator(self):
        """Test @agent.tool direct registration"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.tool
        def direct_tool(message: str) -> str:
            return f"Direct: {message}"
        
        # Should be registered
        assert len(agent._registered_tools) == 1
        tool = agent._registered_tools[0]
        assert tool['name'] == "direct_tool"
        assert tool['source'] == "agent"
        
        # Should be callable
        result = direct_tool("test")
        assert result == "Direct: test"
    
    def test_agent_tool_decorator_with_params(self):
        """Test @agent.tool with custom parameters"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.tool(name="custom_tool", scope="owner")
        def direct_tool_custom(value: int) -> int:
            return value * 10
        
        tool = agent._registered_tools[0]
        assert tool['name'] == "custom_tool"
        assert tool['scope'] == "owner"
    
    def test_agent_http_decorator(self):
        """Test @agent.http direct registration"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.http("/direct")
        def direct_handler(param: str) -> dict:
            return {"param": param, "source": "direct"}
        
        # Should be registered
        assert len(agent._registered_http_handlers) == 1
        handler = agent._registered_http_handlers[0]
        assert handler['subpath'] == "/direct"
        assert handler['method'] == "get"
        assert handler['source'] == "agent"
    
    def test_agent_http_decorator_with_params(self):
        """Test @agent.http with custom parameters"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.http("/custom", method="post", scope="admin")
        async def direct_handler_custom(data: dict) -> dict:
            return {"received": data}
        
        handler = agent._registered_http_handlers[0]
        assert handler['subpath'] == "/custom"
        assert handler['method'] == "post"
        assert handler['scope'] == "admin"
    
    def test_agent_hook_decorator(self):
        """Test @agent.hook direct registration"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.hook("on_response", priority=5)
        async def direct_hook(context):
            context.set("direct_hook", True)
            return context
        
        # Should be registered
        assert "on_response" in agent._registered_hooks
        assert len(agent._registered_hooks["on_response"]) == 1
        hook = agent._registered_hooks["on_response"][0]
        assert hook['priority'] == 5
        assert hook['source'] == "agent"
    
    def test_agent_handoff_decorator(self):
        """Test @agent.handoff direct registration"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.handoff(handoff_type="llm")
        async def direct_handoff(model: str) -> HandoffResult:
            return HandoffResult(
                result=f"Switched to {model}",
                handoff_type="llm",
                success=True
            )
        
        # Should be registered
        assert len(agent._registered_handoffs) == 1
        handoff = agent._registered_handoffs[0]
        assert handoff['config'].handoff_type == "llm"
        assert handoff['source'] == "agent"
    
    def test_direct_registration_conflict_detection(self):
        """Test that direct registration also detects conflicts"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.http("/conflict")
        def first_handler():
            return {"first": True}
        
        # Should raise conflict error
        with pytest.raises(ValueError, match="HTTP handler conflict"):
            @agent.http("/conflict")
            def second_handler():
                return {"second": True}


class TestHTTPIntegration:
    """Test HTTP handler integration scenarios"""
    
    def test_complex_http_scenario(self):
        """Test complex scenario with multiple HTTP handlers and scopes"""
        agent = BaseAgent(
            name="api-agent",
            instructions="API agent with HTTP handlers",
            scopes=["owner", "admin"],
            http_handlers=[get_weather, post_data, get_admin_stats]
        )
        
        # Test comprehensive capabilities
        all_handlers = agent.get_all_http_handlers()
        assert len(all_handlers) == 3
        
        # Test scope-based access
        owner_handlers = agent.get_http_handlers_for_scope("owner")
        admin_handlers = agent.get_http_handlers_for_scope("admin")
        
        owner_paths = {h['subpath'] for h in owner_handlers}
        admin_paths = {h['subpath'] for h in admin_handlers}
        
        # Owner should see owner + public handlers
        assert "/weather" in owner_paths  # owner scope
        assert "/data" in owner_paths     # public scope
        assert "/admin/stats" not in owner_paths  # admin only
        
        # Admin should see all handlers
        assert len(admin_paths) == 3
    
    def test_http_handlers_with_skills(self):
        """Test HTTP handlers work with skills system"""
        class HTTPSkill(MockSkill):
            def __init__(self):
                super().__init__()
            
            @http("/skill-api")
            def skill_handler(self):
                return {"source": "skill", "skill": self.__class__.__name__}
        
        agent = BaseAgent(
            name="skill-http-agent",
            instructions="Agent with HTTP skill",
            skills={"http_skill": HTTPSkill()},
            http_handlers=[get_weather]
        )
        
        # Should have handlers from both skill and direct registration
        handlers = agent.get_all_http_handlers()
        paths = {h['subpath'] for h in handlers}
        
        # Note: Skill HTTP handlers would be discovered by _auto_register_skill_decorators
        # For now, we just test direct registration
        assert "/weather" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 