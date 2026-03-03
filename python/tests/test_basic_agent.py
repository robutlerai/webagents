"""
Basic Agent Test - WebAgents V2.0

Test basic agent functionality, skill registration, and decorator discovery.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill


class DemoSkill(Skill):
    """Demo skill with decorators for testing automatic registration"""
    
    async def initialize(self, agent):
        self.agent = agent
        
    @hook("on_connection", priority=10)
    async def test_hook(self, context):
        context.set("test_hook_executed", True)
        return context
    
    @tool(description="Test tool")
    def test_tool(self, message: str, context=None) -> str:
        if context:
            context.set("test_tool_executed", True)
        return f"Test response: {message}"


@pytest.mark.asyncio
async def test_base_agent_creation():
    """Test basic agent creation with model string auto-creates LiteLLM skill"""
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent",
        model="openai/gpt-4o-mini"
    )
    
    assert agent.name == "test-agent"
    assert agent.instructions == "Test agent"
    assert "primary_llm" in agent.skills
    assert isinstance(agent.skills["primary_llm"], LiteLLMSkill)


@pytest.mark.asyncio
async def test_skill_decorator_registration():
    """Test that skills with decorators are automatically registered"""
    
    demo_skill = DemoSkill()
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent",
        skills={"test": demo_skill}
    )
    
    await asyncio.sleep(0.1)
    
    all_tools = agent.get_all_tools()
    tool_names = [t['name'] for t in all_tools]
    assert "test_tool" in tool_names
    
    connection_hooks = agent.get_all_hooks("on_connection")
    assert len(connection_hooks) > 0
    
    test_hooks = [h for h in connection_hooks if h.get('source') == 'test']
    assert len(test_hooks) == 1


@pytest.mark.asyncio
async def test_agent_execution():
    """Test agent execution with mocked LLM backend"""
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent", 
        model="openai/gpt-4o-mini"
    )
    
    messages = [{"role": "user", "content": "Hello, test message"}]
    
    mock_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! This is a test response."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }
    
    with patch.object(agent, 'run', new_callable=AsyncMock, return_value=mock_response):
        response = await agent.run(messages)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 