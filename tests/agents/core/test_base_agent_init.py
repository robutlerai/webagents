"""
Test BaseAgent initialization with tools, hooks, and handoffs
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill, Handoff, HandoffResult
from webagents.agents.tools.decorators import tool, hook, handoff
from webagents.server.context.context_vars import Context


# Mock skills for testing
class MockSkill(Skill):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.agent = None
        self.initialized = False
    
    async def initialize(self, agent_context):
        self.agent_context = agent_context
        self.agent = agent_context
        self.initialized = True


class MockLLMSkill(MockSkill):
    async def chat_completion(self, messages, tools=None, stream=False):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Mock response"}}],
            "usage": {"total_tokens": 10}
        }
    
    async def chat_completion_stream(self, messages, tools=None):
        yield {"choices": [{"delta": {"content": "Mock"}}]}


# Test tool functions
def plain_tool_function(param: str) -> str:
    """A plain function without decorator"""
    return f"Plain tool result: {param}"


@tool(name="decorated_tool", description="A decorated tool", scope="owner")
def decorated_tool_function(param: str) -> str:
    """A tool with decorator"""
    return f"Decorated tool result: {param}"


@tool
def simple_decorated_tool(value: int) -> int:
    """Simple decorated tool"""
    return value * 2


# Test hook functions
async def simple_hook_function(context: Context) -> Context:
    """Simple hook function"""
    context.set("hook_executed", True)
    return context


@hook("on_request", priority=10, scope="admin")
async def decorated_hook_function(context: Context) -> Context:
    """Decorated hook function"""
    context.set("decorated_hook_executed", True)
    return context


def sync_hook_function(context: Context) -> Context:
    """Synchronous hook function"""
    context.set("sync_hook_executed", True)
    return context


# Test handoff functions
@handoff(name="test_handoff", handoff_type="agent", description="Test handoff", scope="owner")
async def decorated_handoff_function(target: str, context: Context = None) -> HandoffResult:
    """Decorated handoff function"""
    return HandoffResult(
        result=f"Handoff to {target}",
        handoff_type="agent",
        success=True
    )


async def plain_handoff_function(target: str) -> HandoffResult:
    """Plain handoff function"""
    return HandoffResult(
        result=f"Plain handoff to {target}",
        handoff_type="agent",
        success=True
    )


class TestBaseAgentInit:
    """Test BaseAgent initialization with new parameters"""
    
    def test_basic_initialization(self):
        """Test basic agent initialization without extra parameters"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test instructions"
        )
        
        assert agent.name == "test-agent"
        assert agent.instructions == "Test instructions"
        assert agent.scopes == ["all"]  # Updated to use scopes
        assert len(agent._registered_tools) == 0
        assert len(agent._registered_hooks) == 0
        assert len(agent._registered_handoffs) == 0
    
    def test_initialization_with_tools(self):
        """Test agent initialization with tools parameter"""
        tools = [plain_tool_function, decorated_tool_function, simple_decorated_tool]
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test instructions",
            tools=tools
        )
        
        # Check that tools were registered
        assert len(agent._registered_tools) == 3
        
        # Check tool registration details
        tool_names = [tool['name'] for tool in agent._registered_tools]
        assert "plain_tool_function" in tool_names
        assert "decorated_tool" in tool_names  # Uses custom name from decorator
        assert "simple_decorated_tool" in tool_names
        
        # Check tool sources
        tool_sources = [tool['source'] for tool in agent._registered_tools]
        assert all(source == "agent" for source in tool_sources)
        
        # Check tool scopes
        for tool in agent._registered_tools:
            if tool['name'] == "decorated_tool":
                assert tool['scope'] == "owner"  # Explicit scope from decorator
            elif tool['name'] == "simple_decorated_tool":
                assert tool['scope'] == "all"  # Default scope from @tool decorator
            elif tool['name'] == "plain_tool_function":
                assert tool['scope'] == ["all"]  # Undecorated tool inherits agent scopes
    
    def test_initialization_with_hooks(self):
        """Test agent initialization with hooks parameter"""
        hooks = {
            "on_request": [simple_hook_function, decorated_hook_function],
            "on_chunk": [
                sync_hook_function,
                {"handler": simple_hook_function, "priority": 5, "scope": "admin"}
            ]
        }
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test instructions",
            hooks=hooks
        )
        
        # Check that hooks were registered
        assert len(agent._registered_hooks) == 2
        assert "on_request" in agent._registered_hooks
        assert "on_chunk" in agent._registered_hooks
        
        # Check on_request hooks
        on_request_hooks = agent._registered_hooks["on_request"]
        assert len(on_request_hooks) == 2
        
        # Check priority sorting (higher priority first for hooks)
        priorities = [hook['priority'] for hook in on_request_hooks]
        assert priorities == sorted(priorities, reverse=True)
        
        # Check on_chunk hooks
        on_chunk_hooks = agent._registered_hooks["on_chunk"]
        assert len(on_chunk_hooks) == 2
        
        # Find the hook with custom configuration
        custom_hook = next(hook for hook in on_chunk_hooks if hook['priority'] == 5)
        assert custom_hook['scope'] == "admin"
        assert custom_hook['source'] == "agent"
    
    def test_initialization_with_handoffs(self):
        """Test agent initialization with handoffs parameter"""
        # Create Handoff object
        config_handoff = Handoff(
            target="external-agent",
            handoff_type="agent",
            description="External agent handoff",
            scope="admin"
        )
        
        handoffs = [config_handoff, decorated_handoff_function]
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test instructions",
            handoffs=handoffs
        )
        
        # Check that handoffs were registered
        assert len(agent._registered_handoffs) == 2
        
        # Check sources
        sources = [handoff['source'] for handoff in agent._registered_handoffs]
        assert all(source == "agent" for source in sources)
        
        # Check that Handoff was registered directly
        config_handoff_found = False
        for handoff in agent._registered_handoffs:
            if handoff['config'].target == "external-agent":
                config_handoff_found = True
                assert handoff['config'].handoff_type == "agent"
                assert handoff['config'].scope == "admin"
        assert config_handoff_found
        
        # Check that decorated function was converted to Handoff
        decorated_handoff_found = False
        for handoff in agent._registered_handoffs:
            if handoff['config'].target == "test_handoff":  # From decorator name
                decorated_handoff_found = True
                assert handoff['config'].handoff_type == "agent"
                assert handoff['config'].scope == "owner"
                assert 'function' in handoff['config'].metadata
        assert decorated_handoff_found
    
    def test_initialization_with_all_parameters(self):
        """Test agent initialization with all new parameters"""
        # Prepare test data
        tools = [plain_tool_function, decorated_tool_function]
        hooks = {
            "on_request": [simple_hook_function],
            "on_response": [sync_hook_function]
        }
        handoffs = [decorated_handoff_function]
        skills = {"mock_llm": MockLLMSkill(), "primary_llm": MockLLMSkill()}
        
        agent = BaseAgent(
            name="full-test-agent",
            instructions="Full test instructions",
            skills=skills,
            scopes=["owner"],
            tools=tools,
            hooks=hooks,
            handoffs=handoffs
        )
        
        # Verify all registrations
        assert len(agent._registered_tools) >= 2  # At least our tools
        assert len(agent._registered_hooks) >= 2  # At least our hooks
        assert len(agent._registered_handoffs) >= 1  # At least our handoffs
        assert len(agent.skills) >= 2  # mock_llm + primary_llm from model
        
        # Verify scope inheritance
        assert agent.scopes == ["owner"]
        
        # Check that skills were initialized
        assert "mock_llm" in agent.skills
        assert "primary_llm" in agent.skills  # Created from model parameter
    
    def test_scope_inheritance(self):
        """Test that agent scope is inherited by capabilities"""
        from webagents.agents.tools.decorators import tool
        
        def tool_without_decorator(param: str) -> str:
            """Undecorated tool that should inherit agent scopes"""
            return param
        
        async def hook_without_scope(context: Context) -> Context:
            return context
        
        agent = BaseAgent(
            name="scope-test-agent",
            instructions="Scope test",
            scopes=["admin"],
            tools=[tool_without_decorator],
            hooks={"on_request": [hook_without_scope]}
        )
        
        # Check tool scope inheritance - undecorated tool should inherit agent scopes
        tool = agent._registered_tools[0]
        assert tool['scope'] == ["admin"]
        
        # Check hook scope inheritance
        hook = agent._registered_hooks["on_request"][0]
        assert hook['scope'] == ["admin"]
    
    def test_empty_parameters(self):
        """Test initialization with empty lists/dicts"""
        agent = BaseAgent(
            name="empty-test-agent",
            instructions="Empty test",
            tools=[],
            hooks={},
            handoffs=[]
        )
        
        assert len(agent._registered_tools) == 0
        assert len(agent._registered_hooks) == 0
        assert len(agent._registered_handoffs) == 0
    
    def test_none_parameters(self):
        """Test initialization with None parameters (should be handled gracefully)"""
        agent = BaseAgent(
            name="none-test-agent",
            instructions="None test",
            tools=None,
            hooks=None,
            handoffs=None
        )
        
        assert len(agent._registered_tools) == 0
        assert len(agent._registered_hooks) == 0
        assert len(agent._registered_handoffs) == 0
    
    def test_invalid_hook_configuration(self):
        """Test that invalid hook configurations are handled gracefully"""
        hooks = {
            "on_request": [
                simple_hook_function,
                {"priority": 10},  # Missing handler
                {"handler": "not_a_function", "priority": 5}  # Invalid handler
            ]
        }
        
        agent = BaseAgent(
            name="invalid-hook-agent",
            instructions="Invalid hook test",
            hooks=hooks
        )
        
        # Only the valid hook should be registered
        assert len(agent._registered_hooks["on_request"]) == 1
        assert agent._registered_hooks["on_request"][0]['handler'] == simple_hook_function
    
    def test_mixed_handoff_types(self):
        """Test registration of mixed handoff types"""
        config_handoff = Handoff(
            target="config-target",
            handoff_type="agent"
        )
        
        handoffs = [
            config_handoff,
            decorated_handoff_function,
            plain_handoff_function  # This should be ignored (no decorator)
        ]
        
        agent = BaseAgent(
            name="mixed-handoff-agent",
            instructions="Mixed handoff test",
            handoffs=handoffs
        )
        
        # Only config and decorated handoffs should be registered
        # plain_handoff_function lacks @handoff decorator so should be ignored
        assert len(agent._registered_handoffs) == 2
        
        targets = [handoff['config'].target for handoff in agent._registered_handoffs]
        assert "config-target" in targets
        assert "test_handoff" in targets  # From decorated function
    
    def test_tool_registration_preserves_metadata(self):
        """Test that tool registration preserves decorator metadata"""
        tools = [decorated_tool_function]
        
        agent = BaseAgent(
            name="metadata-test-agent",
            instructions="Metadata test",
            tools=tools
        )
        
        tool = agent._registered_tools[0]
        assert tool['name'] == "decorated_tool"  # Custom name from decorator
        assert tool['description'] == "A decorated tool"  # Custom description
        assert tool['scope'] == "owner"  # Custom scope
        assert 'definition' in tool  # OpenAI tool definition should be preserved


@pytest.mark.asyncio
class TestBaseAgentIntegration:
    """Integration tests for BaseAgent with registered capabilities"""
    
    async def test_tool_execution_integration(self):
        """Test that registered tools can be found and used"""
        tools = [decorated_tool_function, simple_decorated_tool]
        
        agent = BaseAgent(
            name="integration-agent",
            instructions="Integration test",
            tools=tools
        )
        
        # Test tool lookup
        tool_func = agent._get_tool_function_by_name("decorated_tool")
        assert tool_func is not None
        assert callable(tool_func)
        
        # Test tool execution
        result = tool_func("test")
        assert result == "Decorated tool result: test"
        
        simple_tool_func = agent._get_tool_function_by_name("simple_decorated_tool")
        assert simple_tool_func is not None
        result = simple_tool_func(5)
        assert result == 10
    
    async def test_hook_execution_integration(self):
        """Test that registered hooks can be executed"""
        hooks = {
            "on_request": [simple_hook_function, sync_hook_function]
        }
        
        agent = BaseAgent(
            name="hook-integration-agent",
            instructions="Hook integration test",
            hooks=hooks
        )
        
        # Create mock context
        context = Mock()
        context.set = Mock()
        
        # Get hooks and verify they exist
        request_hooks = agent.get_all_hooks("on_request")
        assert len(request_hooks) == 2
        
        # Execute hooks
        for hook_config in request_hooks:
            handler = hook_config['handler']
            if asyncio.iscoroutinefunction(handler):
                result = await handler(context)
            else:
                result = handler(context)
            assert result == context
        
        # Verify both hooks were called
        assert context.set.call_count >= 2
    
    async def test_scope_filtering_integration(self):
        """Test that scope filtering works correctly"""
        @tool(scope="admin")
        def admin_tool(param: str) -> str:
            return f"Admin: {param}"
        
        @tool(scope="owner")
        def owner_tool(param: str) -> str:
            return f"Owner: {param}"
        
        @tool  # Default scope="all"
        def public_tool(param: str) -> str:
            return f"Public: {param}"
        
        tools = [admin_tool, owner_tool, public_tool]
        
        agent = BaseAgent(
            name="scope-integration-agent",
            instructions="Scope integration test",
            tools=tools
        )
        
        # Test scope filtering
        all_tools = agent.get_tools_for_scope("all")
        assert len(all_tools) == 1  # Only public_tool
        
        owner_tools = agent.get_tools_for_scope("owner")
        assert len(owner_tools) == 2  # owner_tool + public_tool
        
        admin_tools = agent.get_tools_for_scope("admin")
        assert len(admin_tools) == 3  # All tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 