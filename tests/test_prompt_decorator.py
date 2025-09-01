"""
Prompt Decorator Test - WebAgents

Test prompt decorator functionality, auto-registration, and system prompt enhancement.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook, prompt
from webagents.server.context.context_vars import Context, create_context


class PromptTestSkill(Skill):
    """Test skill with prompt decorators for testing automatic registration"""
    
    async def initialize(self, agent):
        self.agent = agent
        
    @prompt(priority=10, scope="all")
    def system_info_prompt(self, context: Context) -> str:
        return "System Status: Online"
    
    @prompt(priority=20, scope="owner")
    async def user_specific_prompt(self, context: Context) -> str:
        user_id = getattr(context, 'peer_user_id', 'anonymous')
        return f"Current User: {user_id}"
    
    @prompt(priority=5)
    def high_priority_prompt(self, context: Context) -> str:
        return "HIGH PRIORITY: Test Mode Active"
    
    @prompt(priority=30, scope=["admin"])
    def admin_only_prompt(self, context: Context) -> str:
        return "ADMIN MODE: Full system access"


@pytest.fixture
def agent_with_prompts():
    """Create agent with prompt test skill"""
    agent = BaseAgent(
        name="test-prompt-agent",
        instructions="You are a test agent.",
        model="openai/gpt-4o-mini",
        skills={
            "llm": OpenAISkill({"model": "gpt-4o-mini"}),
            "prompt_test": PromptTestSkill()
        }
    )
    return agent


class TestPromptDecorator:
    """Test prompt decorator functionality"""
    
    def test_prompt_decorator_registration(self, agent_with_prompts):
        """Test that @prompt decorated methods are automatically registered"""
        # Check that prompts were registered
        all_prompts = agent_with_prompts._registered_prompts
        assert len(all_prompts) == 4
        
        # Check prompt names
        prompt_names = [p['name'] for p in all_prompts]
        assert 'system_info_prompt' in prompt_names
        assert 'user_specific_prompt' in prompt_names
        assert 'high_priority_prompt' in prompt_names
        assert 'admin_only_prompt' in prompt_names
    
    def test_prompt_priority_ordering(self, agent_with_prompts):
        """Test that prompts are ordered by priority (lower numbers first)"""
        all_prompts = agent_with_prompts._registered_prompts
        priorities = [p['priority'] for p in all_prompts]
        
        # Should be sorted by priority: [5, 10, 20, 30]
        assert priorities == [5, 10, 20, 30]
        
        # Check the order of prompt names
        prompt_names = [p['name'] for p in all_prompts]
        expected_order = ['high_priority_prompt', 'system_info_prompt', 'user_specific_prompt', 'admin_only_prompt']
        assert prompt_names == expected_order
    
    def test_prompt_scope_filtering(self, agent_with_prompts):
        """Test that prompts are filtered by user scope"""
        # Test 'all' scope - should get prompts with 'all' and 'owner' scope
        all_scope_prompts = agent_with_prompts.get_prompts_for_scope('all')
        assert len(all_scope_prompts) == 2  # system_info_prompt, high_priority_prompt
        
        # Test 'owner' scope - should get all except admin-only
        owner_scope_prompts = agent_with_prompts.get_prompts_for_scope('owner')
        assert len(owner_scope_prompts) == 3  # all except admin_only_prompt
        
        # Test 'admin' scope - should get all prompts
        admin_scope_prompts = agent_with_prompts.get_prompts_for_scope('admin')
        assert len(admin_scope_prompts) == 4  # all prompts
    
    @pytest.mark.asyncio
    async def test_prompt_execution(self, agent_with_prompts):
        """Test that prompt functions execute correctly"""
        context = create_context(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
            peer_user_id="test_user_123",
            agent_owner_user_id="owner_123"  # Different from peer_user_id, so scope is "all"
        )
        context.user_id = "test_user_123"  # Set for the prompt function to access
        
        # Execute prompts
        prompt_content = await agent_with_prompts._execute_prompts(context)
        
        # Should contain content from all-accessible prompts in priority order
        # Since peer_user_id != agent_owner_user_id, scope is "all"
        assert "HIGH PRIORITY: Test Mode Active" in prompt_content
        assert "System Status: Online" in prompt_content
        # user_specific_prompt has scope="owner", so not accessible to "all" scope
        assert "Current User: test_user_123" not in prompt_content
        assert "ADMIN MODE" not in prompt_content  # Admin-only, not accessible
        
        # Check ordering (high priority first)
        lines = prompt_content.split('\n\n')
        assert lines[0] == "HIGH PRIORITY: Test Mode Active"
        assert lines[1] == "System Status: Online"
    
    @pytest.mark.asyncio
    async def test_prompt_execution_owner_scope(self, agent_with_prompts):
        """Test prompt execution with owner scope"""
        context = create_context(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
            peer_user_id="test_user_123",
            agent_owner_user_id="test_user_123"  # Same as peer_user_id, so scope is "owner"
        )
        context.user_id = "test_user_123"  # Set for the prompt function to access
        
        # Set the context in the context variable so decorators can access it
        from webagents.server.context.context_vars import set_context
        set_context(context)
        
        try:
            # Execute prompts
            prompt_content = await agent_with_prompts._execute_prompts(context)
            
            # Should contain content from owner-accessible prompts in priority order
            assert "HIGH PRIORITY: Test Mode Active" in prompt_content
            assert "System Status: Online" in prompt_content
            assert "Current User: test_user_123" in prompt_content  # Owner scope
            assert "ADMIN MODE" not in prompt_content  # Admin-only, not accessible
            
            # Check ordering (high priority first)
            lines = prompt_content.split('\n\n')
            assert lines[0] == "HIGH PRIORITY: Test Mode Active"
            assert lines[1] == "System Status: Online"
            assert lines[2] == "Current User: test_user_123"
        finally:
            # Clean up context
            set_context(None)
    
    @pytest.mark.asyncio
    async def test_prompt_context_injection(self, agent_with_prompts):
        """Test that context is properly injected into prompt functions"""
        context = create_context(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
            peer_user_id="injected_user",
            agent_owner_user_id="owner_123"  # Different from peer_user_id, so scope is "all"
        )
        context.user_id = "injected_user"
        
        # Mock the context
        with patch('webagents.server.context.context_vars.get_context', return_value=context):
            prompt_content = await agent_with_prompts._execute_prompts(context)
            
            # Should include system info (accessible to 'all' scope)
            assert "System Status: Online" in prompt_content
            assert "HIGH PRIORITY: Test Mode Active" in prompt_content
    
    @pytest.mark.asyncio
    async def test_message_enhancement_with_prompts(self, agent_with_prompts):
        """Test that messages are enhanced with dynamic prompts"""
        context = create_context(
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            stream=False,
            peer_user_id="test_user",
            agent_owner_user_id="owner_123"  # Different from peer_user_id, so scope is "all"
        )
        
        original_messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        # Enhance messages
        enhanced_messages = await agent_with_prompts._enhance_messages_with_prompts(original_messages, context)
        
        # Should have added a system message at the beginning
        assert len(enhanced_messages) == 2
        assert enhanced_messages[0]["role"] == "system"
        assert enhanced_messages[1]["role"] == "user"
        
        # System message should contain agent instructions + dynamic prompts
        system_content = enhanced_messages[0]["content"]
        assert "You are a test agent." in system_content  # Original instructions
        assert "HIGH PRIORITY: Test Mode Active" in system_content
        assert "System Status: Online" in system_content
    
    @pytest.mark.asyncio 
    async def test_message_enhancement_with_existing_system_message(self, agent_with_prompts):
        """Test that existing system messages are enhanced with dynamic prompts"""
        context = create_context(
            messages=[
                {"role": "system", "content": "Existing system message"},
                {"role": "user", "content": "Hello"}
            ],
            stream=False,
            peer_user_id="test_user",
            agent_owner_user_id="owner_123"  # Different from peer_user_id, so scope is "all"
        )
        
        original_messages = [
            {"role": "system", "content": "Existing system message"},
            {"role": "user", "content": "Hello"}
        ]
        
        # Enhance messages
        enhanced_messages = await agent_with_prompts._enhance_messages_with_prompts(original_messages, context)
        
        # Should still have 2 messages
        assert len(enhanced_messages) == 2
        assert enhanced_messages[0]["role"] == "system"
        assert enhanced_messages[1]["role"] == "user"
        
        # System message should contain original + dynamic prompts
        system_content = enhanced_messages[0]["content"]
        assert "Existing system message" in system_content
        assert "HIGH PRIORITY: Test Mode Active" in system_content
        assert "System Status: Online" in system_content
    
    @pytest.mark.asyncio
    async def test_no_prompts_returns_original_messages(self):
        """Test that messages are unchanged when no prompts are registered"""
        # Create agent without prompt skills
        agent = BaseAgent(
            name="no-prompts-agent",
            instructions="You are a simple agent.",
            model="openai/gpt-4o-mini",
            skills={"llm": OpenAISkill({"model": "gpt-4o-mini"})}
        )
        
        context = create_context(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )
        
        original_messages = [{"role": "user", "content": "Hello"}]
        
        # Enhance messages
        enhanced_messages = await agent._enhance_messages_with_prompts(original_messages, context)
        
        # Should be identical to original messages
        assert enhanced_messages == original_messages
    
    def test_prompt_error_handling(self, agent_with_prompts):
        """Test that prompt execution errors don't break the system"""
        # Create a skill with a failing prompt
        class FailingPromptSkill(Skill):
            @prompt(priority=1)
            def failing_prompt(self, context: Context) -> str:
                raise Exception("Prompt execution failed")
        
        # Add the failing skill
        failing_skill = FailingPromptSkill()
        agent_with_prompts._auto_register_skill_decorators(failing_skill, "failing")
        
        # Execute prompts - should handle the error gracefully
        context = create_context(
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
            peer_user_id="test_user",
            agent_owner_user_id="owner_123"  # Different from peer_user_id, so scope is "all"
        )
        
        async def test_error_handling():
            with patch('builtins.print') as mock_print:
                prompt_content = await agent_with_prompts._execute_prompts(context)
                # Should still have content from other prompts
                assert "System Status: Online" in prompt_content
                # Should have logged the error
                mock_print.assert_called()
        
        asyncio.run(test_error_handling())


if __name__ == "__main__":
    pytest.main([__file__]) 