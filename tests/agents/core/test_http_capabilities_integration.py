"""
Test HTTP Capabilities Integration

Tests the complete HTTP capabilities system including decorators, 
auto-registration, direct registration, and server integration.
"""

import pytest
import asyncio
from typing import Dict, Any

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.tools.decorators import tool, http, hook, handoff
from webagents.agents.skills.base import HandoffResult


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
        @http("/weather", method="get", scope="owner")
        def get_weather(location: str, units: str = "celsius") -> dict:
            """Get weather for a location"""
            return {
                "location": location,
                "temperature": 25,
                "units": units,
                "condition": "sunny"
            }
        
        assert get_weather._webagents_is_http is True
        assert get_weather._http_subpath == "/weather"
        assert get_weather._http_method == "get"
        assert get_weather._http_scope == "owner"
        assert "Get weather for a location" in get_weather._http_description

    def test_http_decorator_post_method(self):
        """Test @http decorator with POST method"""
        @http("/data", method="post")
        async def post_data(data: dict) -> dict:
            """Post data endpoint"""
            return {
                "received": data,
                "status": "success",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
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

    def test_http_decorator_all_methods(self):
        """Test all supported HTTP methods"""
        methods = ["get", "post", "put", "delete", "patch", "head", "options"]
        
        for method in methods:
            @http(f"/{method}", method=method)
            def handler():
                return {"method": method}
            
            assert handler._http_method == method

    def test_http_decorator_scope_types(self):
        """Test different scope types"""
        @http("/all", scope="all")
        def all_handler():
            return {}
        
        @http("/owner", scope="owner")
        def owner_handler():
            return {}
        
        @http("/admin", scope="admin")
        def admin_handler():
            return {}
        
        @http("/multi", scope=["owner", "admin"])
        def multi_handler():
            return {}
        
        assert all_handler._http_scope == "all"
        assert owner_handler._http_scope == "owner"
        assert admin_handler._http_scope == "admin"
        assert multi_handler._http_scope == ["owner", "admin"]


class TestBaseAgentHTTPIntegration:
    """Test BaseAgent HTTP handler integration"""

    def test_http_handler_registration_via_init(self):
        """Test HTTP handler registration via __init__"""
        @http("/weather")
        def get_weather():
            return {"weather": "sunny"}
        
        @http("/data", method="post")
        def post_data():
            return {"status": "received"}
        
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
        def duplicate_weather1():
            return {"handler": 1}
        
        @http("/weather")
        def duplicate_weather2():
            return {"handler": 2}
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[duplicate_weather1]
        )
        
        # Should raise error when trying to register conflicting handler
        with pytest.raises(ValueError, match="HTTP handler conflict"):
            agent.register_http_handler(duplicate_weather2)

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
        @http("/weather", scope="owner")
        def weather_handler():
            return {}
        
        @http("/admin", scope="admin") 
        def admin_handler():
            return {}
        
        @http("/public", scope="all")
        def public_handler():
            return {}
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[weather_handler, admin_handler, public_handler]
        )
        
        # Test different scope access
        all_handlers = agent.get_http_handlers_for_scope("all")
        owner_handlers = agent.get_http_handlers_for_scope("owner")
        admin_handlers = agent.get_http_handlers_for_scope("admin")
        
        # All scope should only see public handlers
        all_paths = {h['subpath'] for h in all_handlers}
        assert "/public" in all_paths
        assert "/weather" not in all_paths
        assert "/admin" not in all_paths
        
        # Owner scope should see owner + public handlers
        owner_paths = {h['subpath'] for h in owner_handlers}
        assert "/public" in owner_paths
        assert "/weather" in owner_paths
        assert "/admin" not in owner_paths
        
        # Admin scope should see all handlers
        admin_paths = {h['subpath'] for h in admin_handlers}
        assert "/public" in admin_paths
        assert "/weather" in admin_paths
        assert "/admin" in admin_paths

    def test_http_handler_registration_without_decorator(self):
        """Test that undecorated functions can't be registered as HTTP handlers"""
        def plain_function():
            return {}
        
        agent = BaseAgent(name="test-agent", instructions="Test agent")
        
        with pytest.raises(ValueError, match="not decorated with @http"):
            agent.register_http_handler(plain_function)


class TestCapabilitiesAutoRegistration:
    """Test the capabilities auto-registration system"""

    def test_capabilities_auto_registration(self):
        """Test that capabilities are auto-registered based on decorator type"""
        @tool(scope="owner")
        def example_tool(message: str) -> str:
            """Example tool for capabilities testing"""
            return f"Tool says: {message}"

        @http("/weather")
        def example_http(location: str = "NYC") -> dict:
            """Example HTTP handler for capabilities testing"""
            return {"location": location, "temp": 25}

        @hook("on_request", priority=10)
        async def example_hook(context):
            """Example hook for capabilities testing"""
            return context

        @handoff(handoff_type="agent")
        async def example_handoff(target: str) -> HandoffResult:
            """Example handoff for capabilities testing"""
            return HandoffResult(
                result=f"Handoff to {target}",
                handoff_type="agent",
                success=True
            )

        capabilities = [example_tool, example_http, example_hook, example_handoff]
        
        agent = BaseAgent(
            name="capabilities-agent",
            instructions="Test capabilities",
            capabilities=capabilities
        )
        
        # Check that each capability was registered in the correct registry
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_http_handlers) == 1
        assert len(agent._registered_hooks) == 1
        assert len(agent._registered_handoffs) == 1
        
        # Verify correct registration
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "example_tool" in tool_names
        
        http_paths = {h['subpath'] for h in agent._registered_http_handlers}
        assert "/weather" in http_paths
        
        hook_events = set(agent._registered_hooks.keys())
        assert "on_request" in hook_events
        
        handoff_targets = {h['config'].target for h in agent._registered_handoffs}
        assert "example_handoff" in handoff_targets

    def test_capabilities_mixed_with_explicit_params(self):
        """Test capabilities combined with explicit parameter registration"""
        @tool
        def capabilities_tool():
            return "capabilities"
        
        @tool
        def explicit_tool():
            return "explicit"
        
        @http("/cap-test")
        def capabilities_http():
            return {"source": "capabilities"}
        
        @http("/explicit")
        def explicit_http():
            return {"source": "explicit"}
        
        agent = BaseAgent(
            name="mixed-agent",
            instructions="Mixed registration test",
            tools=[explicit_tool],  # Explicit registration
            http_handlers=[explicit_http],  # Explicit registration
            capabilities=[capabilities_tool, capabilities_http]  # Auto-registration
        )
        
        # Both explicit and capabilities should be registered
        assert len(agent._registered_tools) == 2
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "explicit_tool" in tool_names
        assert "capabilities_tool" in tool_names
        
        assert len(agent._registered_http_handlers) == 2
        http_paths = {h['subpath'] for h in agent._registered_http_handlers}
        assert "/explicit" in http_paths
        assert "/cap-test" in http_paths

    def test_capabilities_with_undecorated_function(self):
        """Test that undecorated functions in capabilities are ignored"""
        def plain_function():
            return "plain"
        
        @tool
        def decorated_tool():
            return "decorated"
        
        agent = BaseAgent(
            name="plain-agent",
            instructions="Plain function test",
            capabilities=[decorated_tool, plain_function]  # Mixed decorated and plain
        )
        
        # Only decorated function should be registered
        assert len(agent._registered_tools) == 1
        tool_names = {t['name'] for t in agent._registered_tools}
        assert "decorated_tool" in tool_names


class TestDirectRegistration:
    """Test direct registration methods (@agent.tool, @agent.http, etc.)"""

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
        
        # Should be callable
        result = direct_handler("test")
        assert result == {"param": "test", "source": "direct"}

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

    def test_agent_hook_decorator(self):
        """Test @agent.hook direct registration"""
        agent = BaseAgent(name="direct-agent", instructions="Direct registration test")
        
        @agent.hook("on_response", priority=5)
        async def direct_hook(context):
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


class TestHTTPFunctionExecution:
    """Test that HTTP handler functions execute correctly"""

    def test_simple_http_function(self):
        """Test simple HTTP function execution"""
        @http("/status")
        def get_status() -> dict:
            return {
                "status": "healthy",
                "version": "2.0.0",
                "uptime": "5h 23m"
            }
        
        result = get_status()
        assert result["status"] == "healthy"
        assert result["version"] == "2.0.0"
        assert result["uptime"] == "5h 23m"

    def test_http_function_with_params(self):
        """Test HTTP function with parameters"""
        @http("/search")
        def search_data(query: str, limit: int = 10) -> dict:
            return {
                "query": query,
                "limit": limit,
                "results": [f"result_{i}" for i in range(min(limit, 3))]
            }
        
        result = search_data("python", 5)
        assert result["query"] == "python"
        assert result["limit"] == 5
        assert len(result["results"]) == 3

    def test_async_http_function(self):
        """Test async HTTP function execution"""
        @http("/async", method="post")
        async def async_process(data: dict) -> dict:
            # Simulate async processing
            await asyncio.sleep(0.01)
            return {
                "processed": data,
                "async": True,
                "status": "complete"
            }
        
        async def run_test():
            result = await async_process({"message": "test"})
            assert result["async"] is True
            assert result["processed"]["message"] == "test"
            assert result["status"] == "complete"
        
        asyncio.run(run_test())

    def test_http_function_error_handling(self):
        """Test HTTP function error handling"""
        @http("/divide")
        def divide_numbers(a: float, b: float) -> dict:
            if b == 0:
                raise ValueError("Division by zero")
            return {"result": a / b}
        
        # Normal operation
        result = divide_numbers(10, 2)
        assert result["result"] == 5.0
        
        # Error case
        with pytest.raises(ValueError, match="Division by zero"):
            divide_numbers(10, 0)


class TestComplexIntegrationScenarios:
    """Test complex integration scenarios"""

    def test_http_endpoint_using_agent_tools(self):
        """Test HTTP endpoint that uses agent tools"""
        @tool(scope="owner")
        def calculate_score(data: str) -> float:
            """Internal tool for score calculation"""
            # Mock calculation
            return 0.95 if "good" in data.lower() else 0.3

        @http("/score", method="post")
        def score_api(data: dict) -> dict:
            """HTTP endpoint that uses internal tool"""
            data_str = str(data)
            score = calculate_score(data_str)
            
            return {
                "data": data,
                "score": score,
                "grade": "A" if score > 0.9 else "B" if score > 0.7 else "C"
            }

        agent = BaseAgent(
            name="scoring-agent",
            instructions="Scoring agent",
            tools=[calculate_score],
            http_handlers=[score_api]
        )
        
        # Verify registration
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_http_handlers) == 1
        
        # Test integration
        test_data = {"message": "This is good data", "quality": "high"}
        result = score_api(test_data)
        
        assert result["data"] == test_data
        assert result["score"] == 0.95
        assert result["grade"] == "A"

    def test_comprehensive_agent_with_all_capabilities(self):
        """Test agent with tools, HTTP handlers, hooks, and handoffs"""
        @tool(scope="owner")
        def process_data(data: str) -> str:
            return f"Processed: {data}"

        @http("/upload", method="post")
        def upload_file(file_data: dict) -> dict:
            processed = process_data(str(file_data))
            return {"uploaded": True, "processed": processed}

        @http("/status")
        def get_status() -> dict:
            return {"status": "operational", "capabilities": "full"}

        @hook("on_request", priority=5)
        def log_requests(context):
            return context

        @handoff(handoff_type="agent")
        def escalate_to_human(issue: str) -> HandoffResult:
            return HandoffResult(
                result=f"Escalated: {issue}",
                handoff_type="agent",
                success=True
            )

        agent = BaseAgent(
            name="comprehensive-agent",
            instructions="Agent with all capability types",
            scopes=["owner", "admin"],
            capabilities=[process_data, upload_file, get_status, log_requests, escalate_to_human]
        )
        
        # Verify all capabilities registered
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_http_handlers) == 2
        assert len(agent._registered_hooks) == 1
        assert len(agent._registered_handoffs) == 1
        
        # Test HTTP endpoint using tool
        test_file = {"name": "test.txt", "content": "hello world"}
        result = upload_file(test_file)
        
        assert result["uploaded"] is True
        assert "Processed:" in result["processed"]
        assert str(test_file) in result["processed"]

    def test_multiple_agents_with_different_scopes(self):
        """Test multiple agents with different HTTP endpoint scopes"""
        # Public agent
        @http("/public/info")
        def public_info():
            return {"info": "public", "access": "all"}

        public_agent = BaseAgent(
            name="public-agent",
            instructions="Public API agent",
            scopes=["all"],
            http_handlers=[public_info]
        )
        
        # Admin agent
        @http("/admin/config", scope="admin")
        def admin_config():
            return {"config": "sensitive", "access": "admin"}

        admin_agent = BaseAgent(
            name="admin-agent", 
            instructions="Admin API agent",
            scopes=["admin"],
            http_handlers=[admin_config]
        )
        
        # Verify different scope configurations
        public_handlers = public_agent.get_http_handlers_for_scope("all")
        assert len(public_handlers) == 1
        
        admin_handlers_all = admin_agent.get_http_handlers_for_scope("all")
        admin_handlers_admin = admin_agent.get_http_handlers_for_scope("admin")
        assert len(admin_handlers_all) == 0  # Admin endpoint not visible to "all"
        assert len(admin_handlers_admin) == 1  # Admin endpoint visible to "admin"


@pytest.mark.integration
class TestHTTPCapabilitiesIntegration:
    """Integration tests for the complete HTTP capabilities system"""

    def test_end_to_end_http_capabilities(self):
        """Test complete end-to-end HTTP capabilities flow"""
        # Create a realistic agent with mixed capabilities
        @tool(scope="owner")
        def analyze_sentiment(text: str) -> dict:
            """Analyze sentiment of text"""
            return {
                "text": text,
                "sentiment": "positive" if "good" in text.lower() else "negative",
                "confidence": 0.85
            }

        @http("/api/sentiment", method="post")
        def sentiment_api(request_data: dict) -> dict:
            """HTTP API for sentiment analysis"""
            text = request_data.get("text", "")
            if not text:
                raise ValueError("Text is required")
            
            # Use internal tool
            analysis = analyze_sentiment(text)
            
            return {
                "request_id": "req_123",
                "analysis": analysis,
                "status": "success",
                "api_version": "v1"
            }

        @http("/api/health")
        def health_check() -> dict:
            """Health check endpoint"""
            return {
                "status": "healthy",
                "services": ["sentiment_analysis"],
                "uptime": "99.9%"
            }

        @hook("on_request", priority=10)
        def validate_requests(context):
            """Validate incoming requests"""
            # Mock validation
            return context

        agent = BaseAgent(
            name="sentiment-api",
            instructions="Sentiment analysis API service",
            scopes=["owner", "admin"],
            capabilities=[analyze_sentiment, sentiment_api, health_check, validate_requests]
        )
        
        # Verify complete setup
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_http_handlers) == 2
        assert len(agent._registered_hooks) == 1
        
        # Test the complete flow
        test_request = {"text": "This is a good day!"}
        result = sentiment_api(test_request)
        
        # Verify the flow worked
        assert result["status"] == "success"
        assert result["analysis"]["text"] == "This is a good day!"
        assert result["analysis"]["sentiment"] == "positive"
        assert result["analysis"]["confidence"] == 0.85
        assert result["api_version"] == "v1"
        
        # Test health check
        health = health_check()
        assert health["status"] == "healthy"
        assert "sentiment_analysis" in health["services"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 