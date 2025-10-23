#!/usr/bin/env python3
"""
HTTP Test Runner

Simple test runner to verify HTTP functionality works correctly.
Run this to validate the HTTP decorator system.
"""

import sys
import traceback


def test_http_decorator():
    """Test basic HTTP decorator functionality"""
    print("🧪 Testing @http decorator...")
    
    try:
        from webagents.agents.tools.decorators import http
        
        @http("/test", method="get", scope="owner")
        def test_handler(param: str = "default") -> dict:
            return {"param": param, "test": True}
        
        # Verify decorator metadata
        assert hasattr(test_handler, '_webagents_is_http')
        assert test_handler._webagents_is_http is True
        assert test_handler._http_subpath == "/test"
        assert test_handler._http_method == "get"
        assert test_handler._http_scope == "owner"
        
        # Test function execution
        result = test_handler("hello")
        assert result["param"] == "hello"
        assert result["test"] is True
        
        print("   ✅ @http decorator working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ @http decorator failed: {e}")
        traceback.print_exc()
        return False


def test_base_agent_http():
    """Test BaseAgent HTTP handler registration"""
    print("🤖 Testing BaseAgent HTTP integration...")
    
    try:
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import http
        
        @http("/weather")
        def get_weather(location: str = "NYC") -> dict:
            return {"location": location, "temp": 25}
        
        @http("/data", method="post")
        def post_data(data: dict) -> dict:
            return {"received": data, "status": "ok"}
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            http_handlers=[get_weather, post_data]
        )
        
        # Verify registration
        handlers = agent.get_all_http_handlers()
        assert len(handlers) == 2
        
        paths = {h['subpath'] for h in handlers}
        assert "/weather" in paths
        assert "/data" in paths
        
        methods = {h['method'] for h in handlers}
        assert "get" in methods
        assert "post" in methods
        
        print("   ✅ BaseAgent HTTP registration working")
        return True
        
    except Exception as e:
        print(f"   ❌ BaseAgent HTTP integration failed: {e}")
        traceback.print_exc()
        return False


def test_capabilities_registration():
    """Test capabilities auto-registration"""
    print("📦 Testing capabilities auto-registration...")
    
    try:
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import tool, http, hook, handoff
        from webagents.agents.skills.base import HandoffResult
        
        @tool(scope="owner")
        def my_tool(msg: str) -> str:
            return f"Tool: {msg}"
        
        @http("/api")
        def my_api(data: dict) -> dict:
            return {"api_response": data}
        
        @hook("on_request", priority=5)
        def my_hook(context):
            return context
        
        @handoff(handoff_type="agent")
        def my_handoff(target: str) -> HandoffResult:
            return HandoffResult(result=f"Handoff: {target}", handoff_type="agent")
        
        agent = BaseAgent(
            name="capabilities-agent",
            instructions="Test capabilities",
            capabilities=[my_tool, my_api, my_hook, my_handoff]
        )
        
        # Verify auto-registration
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_http_handlers) == 1
        assert len(agent._registered_hooks) == 1
        assert len(agent._registered_handoffs) == 1
        
        print("   ✅ Capabilities auto-registration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Capabilities auto-registration failed: {e}")
        traceback.print_exc()
        return False


def test_direct_registration():
    """Test direct registration methods"""
    print("🎯 Testing direct registration (@agent.http)...")
    
    try:
        from webagents.agents.core.base_agent import BaseAgent
        
        agent = BaseAgent(
            name="direct-agent",
            instructions="Direct registration test"
        )
        
        @agent.http("/status")
        def status_endpoint() -> dict:
            return {"status": "healthy", "agent": agent.name}
        
        @agent.tool(scope="all")
        def calc_tool(expr: str) -> str:
            return f"Result: {eval(expr)}"
        
        # Verify registration
        assert len(agent._registered_http_handlers) == 1
        assert len(agent._registered_tools) == 1
        
        # Test function execution
        status = status_endpoint()
        assert status["status"] == "healthy"
        assert status["agent"] == "direct-agent"
        
        calc = calc_tool("2 + 3")
        assert calc == "Result: 5"
        
        print("   ✅ Direct registration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Direct registration failed: {e}")
        traceback.print_exc()
        return False


def test_scope_filtering():
    """Test scope-based filtering"""
    print("🔒 Testing scope filtering...")
    
    try:
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import http
        
        @http("/public", scope="all")
        def public_handler():
            return {"access": "public"}
        
        @http("/owner", scope="owner")
        def owner_handler():
            return {"access": "owner"}
        
        @http("/admin", scope="admin")
        def admin_handler():
            return {"access": "admin"}
        
        agent = BaseAgent(
            name="scope-agent",
            instructions="Scope test",
            http_handlers=[public_handler, owner_handler, admin_handler]
        )
        
        # Test scope filtering
        all_handlers = agent.get_http_handlers_for_scope("all")
        owner_handlers = agent.get_http_handlers_for_scope("owner")
        admin_handlers = agent.get_http_handlers_for_scope("admin")
        
        all_paths = {h['subpath'] for h in all_handlers}
        owner_paths = {h['subpath'] for h in owner_handlers}
        admin_paths = {h['subpath'] for h in admin_handlers}
        
        # Verify scope hierarchy
        assert "/public" in all_paths
        assert "/owner" not in all_paths
        assert "/admin" not in all_paths
        
        assert "/public" in owner_paths
        assert "/owner" in owner_paths
        assert "/admin" not in owner_paths
        
        assert "/public" in admin_paths
        assert "/owner" in admin_paths
        assert "/admin" in admin_paths
        
        print("   ✅ Scope filtering working")
        return True
        
    except Exception as e:
        print(f"   ❌ Scope filtering failed: {e}")
        traceback.print_exc()
        return False


def test_conflict_detection():
    """Test conflict detection"""
    print("⚠️  Testing conflict detection...")
    
    try:
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import http
        
        @http("/test")
        def handler1():
            return {"handler": 1}
        
        @http("/test")  # Same path
        def handler2():
            return {"handler": 2}
        
        agent = BaseAgent(
            name="conflict-agent",
            instructions="Conflict test",
            http_handlers=[handler1]
        )
        
        # Should raise conflict error
        try:
            agent.register_http_handler(handler2)
            print("   ❌ Conflict detection failed - no error raised")
            return False
        except ValueError as e:
            if "conflict" in str(e).lower():
                print("   ✅ Conflict detection working")
                return True
            else:
                print(f"   ❌ Wrong error type: {e}")
                return False
        
    except Exception as e:
        print(f"   ❌ Conflict detection test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all HTTP tests"""
    print("🚀 HTTP Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        test_http_decorator,
        test_base_agent_http,
        test_capabilities_registration,
        test_direct_registration,
        test_scope_filtering,
        test_conflict_detection
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed}/{passed + failed} ({100 * passed / (passed + failed):.1f}%)")
    
    if failed == 0:
        print("\n🎉 All HTTP tests passed! The implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 