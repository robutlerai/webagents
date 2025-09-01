"""
Basic Agent Test - WebAgents V2.0

Test basic agent functionality, skill registration, and decorator discovery.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook


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
    """Test basic agent creation with OpenAI skill"""
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent",
        model="openai/gpt-4o-mini"
    )
    
    assert agent.name == "test-agent"
    assert agent.instructions == "Test agent"
    assert "primary_llm" in agent.skills
    assert isinstance(agent.skills["primary_llm"], OpenAISkill)


@pytest.mark.asyncio
async def test_skill_decorator_registration():
    """Test that skills with decorators are automatically registered"""
    
    demo_skill = DemoSkill()
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent",
        skills={"test": demo_skill}
    )
    
    # Give the async skill initialization time to complete
    await asyncio.sleep(0.1)
    
    # Check that tools were registered
    all_tools = agent.get_all_tools()
    tool_names = [tool['name'] for tool in all_tools]
    print(f"Registered tools: {tool_names}")
    assert "test_tool" in tool_names
    
    # Check that hooks were registered
    connection_hooks = agent.get_all_hooks("on_connection")
    print(f"Connection hooks: {[h.get('source') for h in connection_hooks]}")
    assert len(connection_hooks) > 0
    
    # Check that the hook is from our test skill
    test_hooks = [h for h in connection_hooks if h.get('source') == 'test']
    assert len(test_hooks) == 1


@pytest.mark.asyncio
async def test_agent_execution():
    """Test agent execution and context flow"""
    
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent", 
        model="openai/gpt-4o-mini"
    )
    
    messages = [{"role": "user", "content": "Hello, test message"}]
    
    # Test non-streaming execution
    response = await agent.run(messages)
    
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert "content" in response["choices"][0]["message"]
    print(f"Non-streaming response: {response['choices'][0]['message']['content']}")
    
    # Test streaming execution
    chunks = []
    async for chunk in agent.run_streaming(messages):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    print(f"Streaming chunks received: {len(chunks)}")
    
    # Verify final chunk has finish_reason
    final_chunk = chunks[-1]
    assert final_chunk.get("choices", [{}])[0].get("finish_reason") == "stop"


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"]) 