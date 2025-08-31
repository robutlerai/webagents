"""
LiteLLM Skill Tests - WebAgents V2.0

Tests for the LiteLLMSkill cross-provider LLM routing functionality.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.server.context.context_vars import create_context, set_context


@pytest.mark.asyncio
async def test_litellm_skill_initialization():
    """Test LiteLLMSkill initialization and configuration"""
    
    # Test with custom config
    config = {
        'model': 'gpt-4o',
        'temperature': 0.8,
        'max_tokens': 2000,
        'fallback_models': ['gpt-4o-mini', 'claude-3-haiku-20240307']
    }
    
    litellm_skill = LiteLLMSkill(config)
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    # Give skills time to initialize
    await asyncio.sleep(0.1)
    
    # Verify skill configuration
    assert litellm_skill.model == 'gpt-4o'
    assert litellm_skill.temperature == 0.8
    assert litellm_skill.max_tokens == 2000
    assert litellm_skill.fallback_models == ['gpt-4o-mini', 'claude-3-haiku-20240307']
    assert litellm_skill.current_model == 'gpt-4o'
    assert litellm_skill.agent == agent
    
    # Verify API keys are loaded
    assert 'openai' in litellm_skill.api_keys
    assert 'anthropic' in litellm_skill.api_keys


@pytest.mark.asyncio
async def test_litellm_tools_registration():
    """Test that LiteLLM tools are automatically registered"""
    
    litellm_skill = LiteLLMSkill()
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Check that LiteLLM tools were registered
    all_tools = agent.get_all_tools()
    tool_names = [tool['name'] for tool in all_tools]
    
    expected_tools = [
        'list_available_models',
        'switch_model',
        'get_usage_stats'
    ]
    
    for tool_name in expected_tools:
        assert tool_name in tool_names, f"Tool {tool_name} not registered"
    
    print(f"LiteLLM tools registered: {[t for t in tool_names if any(keyword in t for keyword in ['model', 'usage', 'litellm'])]}")


@pytest.mark.asyncio
async def test_model_configurations():
    """Test model configuration and capabilities"""
    
    litellm_skill = LiteLLMSkill()
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test default model configurations
    assert 'gpt-4o-mini' in litellm_skill.model_configs
    assert 'claude-3-5-sonnet-20241022' in litellm_skill.model_configs
    assert 'grok-beta' in litellm_skill.model_configs
    
    # Test model config properties
    gpt4o_mini = litellm_skill.model_configs['gpt-4o-mini']
    assert gpt4o_mini.provider == 'openai'
    assert gpt4o_mini.supports_tools == True
    assert gpt4o_mini.supports_streaming == True
    assert gpt4o_mini.max_tokens == 16384
    
    claude_config = litellm_skill.model_configs['claude-3-5-sonnet-20241022']
    assert claude_config.provider == 'anthropic'
    assert claude_config.supports_tools == True


@pytest.mark.asyncio
async def test_list_available_models_tool():
    """Test the list_available_models tool functionality"""
    
    litellm_skill = LiteLLMSkill()
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Create context
    context = create_context(request_id="test-models")
    set_context(context)
    
    # Find the tool
    all_tools = agent.get_all_tools()
    list_models_tool = next(t['function'] for t in all_tools if t['name'] == 'list_available_models')
    
    # Test listing all models
    models = await list_models_tool()
    
    assert len(models) > 0
    assert any(m['name'] == 'gpt-4o-mini' for m in models)
    assert any(m['name'] == 'claude-3-5-sonnet-20241022' for m in models)
    
    # Check model structure
    gpt_model = next(m for m in models if m['name'] == 'gpt-4o-mini')
    assert 'provider' in gpt_model
    assert 'max_tokens' in gpt_model
    assert 'supports_tools' in gpt_model
    assert 'supports_streaming' in gpt_model
    assert 'available' in gpt_model
    
    print(f"Found {len(models)} available models")
    
    # Test filtering by provider
    openai_models = await list_models_tool(provider="openai")
    assert all(m['provider'] == 'openai' for m in openai_models)
    
    anthropic_models = await list_models_tool(provider="anthropic")
    assert all(m['provider'] == 'anthropic' for m in anthropic_models)


@pytest.mark.asyncio
async def test_switch_model_tool():
    """Test model switching functionality"""
    
    litellm_skill = LiteLLMSkill({'model': 'gpt-4o-mini'})
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Create context
    context = create_context(request_id="test-switch")
    set_context(context)
    
    # Find the tool
    all_tools = agent.get_all_tools()
    switch_model_tool = next(t['function'] for t in all_tools if t['name'] == 'switch_model')
    
    # Test initial model
    assert litellm_skill.current_model == 'gpt-4o-mini'
    
    # Test switching to a valid model
    result = await switch_model_tool("gpt-4o")
    assert "Switched to model: gpt-4o" in result
    assert litellm_skill.current_model == "gpt-4o"
    
    # Test switching to Claude
    result = await switch_model_tool("claude-3-5-sonnet-20241022")
    assert "Switched to model: claude-3-5-sonnet-20241022" in result
    assert litellm_skill.current_model == "claude-3-5-sonnet-20241022"
    
    # Test switching to invalid model
    result = await switch_model_tool("invalid-model-name")
    assert "not available" in result
    assert litellm_skill.current_model == "claude-3-5-sonnet-20241022"  # Should stay the same
    
    print(f"Model switching test completed: {litellm_skill.current_model}")


@pytest.mark.asyncio
async def test_usage_stats_tool():
    """Test usage statistics tracking"""
    
    litellm_skill = LiteLLMSkill()
    
    agent = BaseAgent(
        name="litellm-test-agent",
        instructions="Test agent for LiteLLM",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Create context
    context = create_context(request_id="test-stats")
    set_context(context)
    
    # Find the tool
    all_tools = agent.get_all_tools()
    get_stats_tool = next(t['function'] for t in all_tools if t['name'] == 'get_usage_stats')
    
    # Test initial stats
    stats = await get_stats_tool()
    
    assert 'current_model' in stats
    assert 'total_calls' in stats
    assert 'model_usage' in stats
    assert 'error_counts' in stats
    assert 'available_providers' in stats
    
    assert stats['current_model'] == litellm_skill.current_model
    assert stats['total_calls'] == 0  # No calls made yet
    assert isinstance(stats['available_providers'], list)
    assert 'openai' in stats['available_providers']
    
    print(f"Usage stats: {stats}")


@pytest.mark.asyncio
async def test_litellm_skill_with_agent():
    """Test LiteLLMSkill integration with BaseAgent"""
    
    # Test agent using LiteLLM as primary LLM
    agent = BaseAgent(
        name="litellm-agent",
        instructions="You are a helpful assistant using LiteLLM",
        model="litellm/gpt-4o-mini"  # Use LiteLLM routing
    )
    
    await asyncio.sleep(0.1)
    
    # Verify LiteLLM skill was created and configured
    assert "primary_llm" in agent.skills
    litellm_skill = agent.skills["primary_llm"]
    assert litellm_skill is not None
    assert isinstance(litellm_skill, LiteLLMSkill)
    
    # Verify model configuration
    assert litellm_skill.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling and fallback mechanisms"""
    
    config = {
        'model': 'gpt-4o',
        'fallback_models': ['gpt-4o-mini', 'claude-3-haiku-20240307']
    }
    
    litellm_skill = LiteLLMSkill(config)
    
    agent = BaseAgent(
        name="error-test-agent",
        instructions="Test agent for error handling",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test error tracking
    litellm_skill._track_error("gpt-4o")
    litellm_skill._track_error("gpt-4o")
    
    assert litellm_skill.error_counts["gpt-4o"] == 2
    
    # Test usage tracking
    litellm_skill._track_usage("gpt-4o-mini")
    litellm_skill._track_usage("gpt-4o-mini")
    litellm_skill._track_usage("claude-3-haiku-20240307")
    
    assert litellm_skill.usage_stats["gpt-4o-mini"] == 2
    assert litellm_skill.usage_stats["claude-3-haiku-20240307"] == 1


@pytest.mark.asyncio
async def test_compatibility_methods():
    """Test compatibility methods for BaseAgent integration"""
    
    litellm_skill = LiteLLMSkill()
    
    agent = BaseAgent(
        name="compat-test-agent",
        instructions="Test agent for compatibility",
        skills={"litellm": litellm_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test get_dependencies
    deps = litellm_skill.get_dependencies()
    assert isinstance(deps, list)
    assert len(deps) == 0  # LiteLLM is self-contained
    
    # Test embedding placeholder (V2.1 feature)
    embedding = await litellm_skill.generate_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # Standard embedding size


@pytest.mark.asyncio 
async def test_model_provider_mapping():
    """Test that model names map to correct providers"""
    
    litellm_skill = LiteLLMSkill()
    
    # Test OpenAI models
    assert litellm_skill.model_configs['gpt-4o'].provider == 'openai'
    assert litellm_skill.model_configs['gpt-4o-mini'].provider == 'openai' 
    assert litellm_skill.model_configs['gpt-3.5-turbo'].provider == 'openai'
    
    # Test Anthropic models
    assert litellm_skill.model_configs['claude-3-5-sonnet-20241022'].provider == 'anthropic'
    assert litellm_skill.model_configs['claude-3-haiku-20240307'].provider == 'anthropic'
    
    # Test XAI models  
    assert litellm_skill.model_configs['grok-beta'].provider == 'xai'
    
    # Test Google models
    assert litellm_skill.model_configs['gemini-1.5-pro'].provider == 'google'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 