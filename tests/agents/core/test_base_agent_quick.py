"""
Quick tests for BaseAgent initialization with tools, hooks, and handoffs
"""

import pytest
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Handoff
from webagents.agents.tools.decorators import tool, hook, handoff


def simple_tool(message: str) -> str:
    """Simple test tool"""
    return f"Tool says: {message}"


@tool(name="fancy_tool", scope="owner")  
def decorated_tool(value: int) -> int:
    """Fancy decorated tool"""
    return value * 10


def simple_hook(context):
    """Simple test hook"""
    if hasattr(context, 'set'):
        context.set("hook_called", True)
    return context


@handoff(handoff_type="agent")
def simple_handoff(target: str):
    """Simple test handoff"""
    from webagents.agents.skills.base import HandoffResult
    return HandoffResult(result=f"Handoff to {target}", handoff_type="agent")


class TestBaseAgentQuick:
    """Quick smoke tests for BaseAgent initialization"""
    
    def test_basic_creation(self):
        """Test basic agent creation works"""
        agent = BaseAgent(name="test", instructions="Test agent")
        assert agent.name == "test"
        assert agent.instructions == "Test agent"
        assert agent.scopes == ["all"]  # Check default scopes
    
    def test_tools_registration(self):
        """Test tools are registered properly"""
        tools = [simple_tool, decorated_tool]
        agent = BaseAgent(
            name="test", 
            instructions="Test", 
            tools=tools
        )
        
        # Check tools were registered
        assert len(agent._registered_tools) == 2
        
        # Check tool names
        tool_names = [t['name'] for t in agent._registered_tools]
        assert "simple_tool" in tool_names
        assert "fancy_tool" in tool_names  # Uses decorator name
    
    def test_hooks_registration(self):
        """Test hooks are registered properly"""
        hooks = {
            "on_request": [simple_hook],
            "on_response": [simple_hook]
        }
        agent = BaseAgent(
            name="test",
            instructions="Test", 
            hooks=hooks
        )
        
        # Check hooks were registered
        assert "on_request" in agent._registered_hooks
        assert "on_response" in agent._registered_hooks
        assert len(agent._registered_hooks["on_request"]) == 1
        assert len(agent._registered_hooks["on_response"]) == 1
    
    def test_handoffs_registration(self):
        """Test handoffs are registered properly"""
        config = Handoff(
            target="other-agent",
            handoff_type="agent",
            description="Test handoff"
        )
        
        handoffs = [config, simple_handoff]
        agent = BaseAgent(
            name="test",
            instructions="Test",
            handoffs=handoffs
        )
        
        # Check handoffs were registered
        assert len(agent._registered_handoffs) == 2
        
        # Check targets
        targets = [h['config'].target for h in agent._registered_handoffs]
        assert "other-agent" in targets
        assert "simple_handoff" in targets  # Function name used as target
    
    def test_all_together(self):
        """Test all capabilities together"""
        agent = BaseAgent(
            name="full-test",
            instructions="Full test agent",
            tools=[simple_tool],
            hooks={"on_request": [simple_hook]},
            handoffs=[simple_handoff]
        )
        
        assert len(agent._registered_tools) == 1
        assert len(agent._registered_hooks) == 1
        assert len(agent._registered_handoffs) == 1
        assert agent.name == "full-test"
    
    def test_tool_lookup(self):
        """Test tool lookup functionality"""
        agent = BaseAgent(
            name="lookup-test",
            instructions="Lookup test",
            tools=[decorated_tool]
        )
        
        # Test finding tool by name
        tool_func = agent._get_tool_function_by_name("fancy_tool")
        assert tool_func is not None
        assert callable(tool_func)
        
        # Test tool execution
        result = tool_func(5)
        assert result == 50  # 5 * 10
    
    def test_scope_filtering(self):
        """Test scope-based tool filtering"""
        @tool(scope="admin")
        def admin_tool():
            return "admin"
            
        @tool(scope="all")
        def public_tool():
            return "public"
        
        agent = BaseAgent(
            name="scope-test",
            instructions="Scope test",
            tools=[admin_tool, public_tool]
        )
        
        # Get tools for different scopes
        all_tools = agent.get_tools_for_scope("all")
        admin_tools = agent.get_tools_for_scope("admin")
        
        # All scope should only see public tools
        assert len(all_tools) == 1
        
        # Admin scope should see both (admin inherits from all)
        assert len(admin_tools) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 