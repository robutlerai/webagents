"""
Test BaseAgent scopes system - multiple scopes support and management
"""

import pytest
from unittest.mock import Mock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill, Handoff, HandoffResult
from webagents.agents.tools.decorators import tool, hook, handoff
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


class MockLLMSkill(MockSkill):
    async def chat_completion(self, messages, tools=None, stream=False):
        return {
            "choices": [{"message": {"role": "assistant", "content": "Mock response"}}],
            "usage": {"total_tokens": 10}
        }


# Test tools with different scopes
@tool(scope="owner")
def owner_tool(param: str) -> str:
    """Owner-only tool"""
    return f"Owner: {param}"


@tool(scope=["admin", "owner"])
def multi_scope_tool(param: str) -> str:
    """Tool accessible to multiple scopes"""
    return f"Multi: {param}"


@tool(scope="admin")
def admin_tool(param: str) -> str:
    """Admin-only tool"""
    return f"Admin: {param}"


@tool  # Default scope "all"
def public_tool(param: str) -> str:
    """Public tool"""
    return f"Public: {param}"


# Test hooks with different scopes
@hook("on_request", scope="owner")
async def owner_hook(context: Context) -> Context:
    """Owner-only hook"""
    context.set("owner_hook_executed", True)
    return context


@hook("on_request", scope=["admin", "owner"])
async def multi_scope_hook(context: Context) -> Context:
    """Multi-scope hook"""
    context.set("multi_scope_hook_executed", True)
    return context


# Test handoffs with different scopes
@handoff(scope="admin")
async def admin_handoff(target: str) -> HandoffResult:
    """Admin-only handoff"""
    return HandoffResult(
        result=f"Admin handoff to {target}",
        handoff_type="agent",
        success=True
    )


@handoff(scope=["owner", "admin"])
async def multi_scope_handoff(target: str) -> HandoffResult:
    """Multi-scope handoff"""
    return HandoffResult(
        result=f"Multi handoff to {target}",
        handoff_type="agent",
        success=True
    )


class TestBaseAgentScopes:
    """Test BaseAgent scopes system"""
    
    def test_default_scopes_initialization(self):
        """Test that agent gets default scopes when none specified"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent"
        )
        
        assert agent.scopes == ["all"]
        assert agent.get_scopes() == ["all"]
        assert agent.has_scope("all")
        assert not agent.has_scope("owner")
    
    def test_custom_scopes_initialization(self):
        """Test agent initialization with custom scopes"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["owner", "admin"]
        )
        
        assert agent.scopes == ["owner", "admin"]
        assert agent.get_scopes() == ["owner", "admin"]
        assert agent.has_scope("owner")
        assert agent.has_scope("admin")
        assert not agent.has_scope("all")
    
    def test_scope_management_methods(self):
        """Test scope management methods"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["all"]
        )
        
        # Test add_scope
        agent.add_scope("owner")
        assert "owner" in agent.scopes
        assert agent.has_scope("owner")
        
        # Test add_scope duplicate (should not add)
        agent.add_scope("owner")
        assert agent.scopes.count("owner") == 1
        
        # Test remove_scope
        agent.remove_scope("owner")
        assert "owner" not in agent.scopes
        assert not agent.has_scope("owner")
        
        # Test remove_scope non-existent (should not error)
        agent.remove_scope("nonexistent")
        
        # Test set_scopes
        agent.set_scopes(["admin", "owner"])
        assert agent.scopes == ["admin", "owner"]
        
        # Test clear_scopes
        agent.clear_scopes()
        assert agent.scopes == []
        assert not agent.has_scope("all")
    
    def test_tools_scope_inheritance(self):
        """Test that tools inherit agent scopes"""
        def plain_tool(param: str) -> str:
            return param
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["owner", "admin"],
            tools=[plain_tool]
        )
        
        # Plain tools should inherit agent scopes
        tool_config = agent._registered_tools[0]
        assert tool_config['scope'] == ["owner", "admin"]
    
    def test_decorated_tools_scope_override(self):
        """Test that decorated tools can override inherited scopes"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["owner", "admin"],
            tools=[owner_tool, multi_scope_tool, public_tool]
        )
        
        # Find tools by name and check their scopes
        tools_by_name = {tool['name']: tool for tool in agent._registered_tools}
        
        assert tools_by_name['owner_tool']['scope'] == "owner"
        assert tools_by_name['multi_scope_tool']['scope'] == ["admin", "owner"]
        assert tools_by_name['public_tool']['scope'] == "all"
    
    def test_hooks_scope_inheritance(self):
        """Test that hooks inherit agent scopes"""
        def plain_hook(context):
            return context
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["owner"],
            hooks={"on_request": [plain_hook]}
        )
        
        # Plain hooks should inherit agent scopes
        hook_config = agent._registered_hooks["on_request"][0]
        assert hook_config['scope'] == ["owner"]
    
    def test_handoffs_scope_inheritance(self):
        """Test that handoffs inherit agent scopes"""
        async def plain_handoff(target: str) -> HandoffResult:
            return HandoffResult(result="test", handoff_type="agent")
        
        # Mark as handoff
        plain_handoff._webagents_is_handoff = True
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["admin"],
            handoffs=[plain_handoff]
        )
        
        # Plain handoffs should inherit agent scopes
        handoff_config = agent._registered_handoffs[0]
        assert handoff_config['config'].scope == ["admin"]
    
    def test_scope_filtering_single_scope(self):
        """Test filtering tools by single scope"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            tools=[owner_tool, admin_tool, public_tool, multi_scope_tool]
        )
        
        # Test filtering by different scopes
        all_tools = agent.get_tools_for_scope("all")
        owner_tools = agent.get_tools_for_scope("owner")
        admin_tools = agent.get_tools_for_scope("admin")
        
        # Extract tool names for easier testing
        all_names = {tool['name'] for tool in all_tools}
        owner_names = {tool['name'] for tool in owner_tools}
        admin_names = {tool['name'] for tool in admin_tools}
        
        # "all" scope should only see public tools
        assert "public_tool" in all_names
        assert len(all_names) == 1
        
        # "owner" scope should see owner + public + multi-scope tools
        assert "owner_tool" in owner_names
        assert "public_tool" in owner_names
        assert "multi_scope_tool" in owner_names
        assert "admin_tool" not in owner_names
        
        # "admin" scope should see all tools (admin has highest hierarchy)
        assert "admin_tool" in admin_names
        assert "owner_tool" in admin_names
        assert "public_tool" in admin_names
        assert "multi_scope_tool" in admin_names
    
    def test_scope_filtering_multiple_scopes(self):
        """Test filtering tools by multiple scopes"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            tools=[owner_tool, admin_tool, public_tool, multi_scope_tool]
        )
        
        # Test filtering by multiple scopes
        mixed_tools = agent.get_tools_for_scopes(["all", "owner"])
        admin_owner_tools = agent.get_tools_for_scopes(["admin", "owner"])
        
        mixed_names = {tool['name'] for tool in mixed_tools}
        admin_owner_names = {tool['name'] for tool in admin_owner_tools}
        
        # ["all", "owner"] should see public + owner + multi-scope tools
        assert "public_tool" in mixed_names
        assert "owner_tool" in mixed_names
        assert "multi_scope_tool" in mixed_names
        assert "admin_tool" not in mixed_names
        
        # ["admin", "owner"] should see all tools (admin has highest hierarchy)
        assert len(admin_owner_names) == 4  # All tools
    
    def test_scope_inheritance_with_configuration(self):
        """Test scope inheritance with hook configurations"""
        def plain_hook(context):
            return context
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["admin"],
            hooks={
                "on_request": [
                    plain_hook,  # Should inherit ["admin"]
                    {"handler": plain_hook, "priority": 10},  # Should inherit ["admin"]
                    {"handler": plain_hook, "scope": "owner", "priority": 5}  # Should use "owner"
                ]
            }
        )
        
        hooks = agent._registered_hooks["on_request"]
        
        # Check that scopes were inherited/set correctly
        scopes = [hook['scope'] for hook in hooks]
        
        # First two should inherit agent scopes, third should use explicit scope
        assert scopes[0] == ["admin"]  # First hook inherits
        assert scopes[1] == ["admin"]  # Second hook inherits  
        assert scopes[2] == "owner"    # Third hook has explicit scope
    
    def test_handoff_config_objects_with_scopes(self):
        """Test Handoff objects with different scopes"""
        config1 = Handoff(
            target="agent-1",
            handoff_type="agent",
            scope="owner"
        )
        
        config2 = Handoff(
            target="agent-2",
            handoff_type="agent",
            scope=["admin", "owner"]
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["admin"],
            handoffs=[config1, config2]
        )
        
        handoffs = agent._registered_handoffs
        
        # Check that Handoff scopes were preserved
        assert handoffs[0]['config'].scope == "owner"
        assert handoffs[1]['config'].scope == ["admin", "owner"]
    
    def test_scope_modification_after_initialization(self):
        """Test modifying scopes after agent initialization"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=["all"]
        )
        
        # Start with basic scopes
        assert agent.get_scopes() == ["all"]
        
        # Add more scopes
        agent.add_scope("owner")
        agent.add_scope("admin")
        
        assert "owner" in agent.scopes
        assert "admin" in agent.scopes
        assert len(agent.scopes) == 3
        
        # Remove a scope
        agent.remove_scope("all")
        assert "all" not in agent.scopes
        assert len(agent.scopes) == 2
        
        # Set completely new scopes
        agent.set_scopes(["custom", "special"])
        assert agent.scopes == ["custom", "special"]
        assert not agent.has_scope("owner")
        assert not agent.has_scope("admin")
    
    def test_empty_scopes_handling(self):
        """Test handling of empty scopes"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test agent",
            scopes=[]
        )
        
        assert agent.scopes == []
        assert not agent.has_scope("all")
        
        # Tools filtering should still work
        tools = agent.get_tools_for_scopes([])
        assert isinstance(tools, list)
        
        # Can still add scopes
        agent.add_scope("owner")
        assert agent.has_scope("owner")


class TestScopesIntegration:
    """Integration tests for scopes system"""
    
    def test_complex_scope_scenario(self):
        """Test complex scenario with multiple scope types"""
        # Create agent with multiple scopes
        agent = BaseAgent(
            name="complex-agent",
            instructions="Complex test agent",
            scopes=["owner", "admin"],
            tools=[owner_tool, admin_tool, multi_scope_tool, public_tool],
            hooks={
                "on_request": [owner_hook, multi_scope_hook]
            },
            handoffs=[admin_handoff, multi_scope_handoff]
        )
        
        # Verify agent scopes
        assert agent.has_scope("owner")
        assert agent.has_scope("admin")
        
        # Test tool access for different user scopes
        user_all_tools = agent.get_tools_for_scope("all")
        user_owner_tools = agent.get_tools_for_scope("owner")
        user_admin_tools = agent.get_tools_for_scope("admin")
        
        # Verify different users see appropriate tools
        assert len(user_all_tools) == 1  # Only public
        assert len(user_owner_tools) == 3  # owner + multi + public
        assert len(user_admin_tools) == 4  # All tools
        
        # Test multiple user scopes
        multi_user_tools = agent.get_tools_for_scopes(["all", "owner"])
        assert len(multi_user_tools) == 3  # owner + multi + public
    
    def test_scope_inheritance_chain(self):
        """Test scope inheritance through the registration chain"""
        def plain_tool_1():
            return "tool1"
            
        def plain_tool_2():
            return "tool2"
        
        agent = BaseAgent(
            name="inheritance-agent",
            instructions="Test inheritance",
            scopes=["custom", "special"],
            tools=[plain_tool_1, owner_tool, plain_tool_2]
        )
        
        # Check that plain tools inherited agent scopes
        # but decorated tool kept its own scope
        for tool in agent._registered_tools:
            if tool['name'] in ['plain_tool_1', 'plain_tool_2']:
                assert tool['scope'] == ["custom", "special"]
            elif tool['name'] == 'owner_tool':
                assert tool['scope'] == "owner"  # Kept explicit scope


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 