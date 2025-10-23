"""
Comprehensive test suite for WebAgents handoff system

Tests cover:
- Handoff registration and configuration
- Handoff decorator functionality
- Handoff execution (streaming and non-streaming)
- Priority-based handoff selection
- Generator vs non-generator handoffs
- LocalAgentHandoff system
- Scope-based access control
- Edge cases and error handling
"""

import pytest
import asyncio
from typing import Dict, Any, List, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill, Handoff, HandoffResult
from webagents.agents.tools.decorators import handoff
from webagents.agents.core.handoffs import LocalAgentHandoff, create_local_handoff_system
from webagents.server.context.context_vars import create_context, set_context, get_context


# ===== Test Fixtures =====

class MockLLMSkill(Skill):
    """Mock LLM skill for testing handoffs"""
    
    def __init__(self):
        super().__init__({})
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
    
    async def initialize(self, agent):
        """Initialize and register as handoff"""
        self.agent = agent
        # Register as default completion handoff
        handoff_config = Handoff(
            target="mock_llm",
            description="Mock LLM completion",
            scope="all",
            metadata={
                'function': self.chat_completion,
                'priority': 50,
                'is_generator': False
            }
        )
        agent.register_handoff(handoff_config, source="mock_llm_skill")
    
    async def chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, stream: bool = False):
        """Mock chat completion"""
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        
        return {
            "id": "chatcmpl-test",
            "created": 1234567890,
            "model": "mock-model",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response #{self.call_count}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }


@handoff(name="custom_handoff", prompt="Custom handoff for testing", scope="all")
async def decorated_handoff(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, **kwargs):
    """Test handoff decorated with @handoff"""
    return {
        "id": "chatcmpl-custom",
        "created": 1234567890,
        "model": "custom-model",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Custom handoff response"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }


@handoff(name="streaming_handoff", prompt="Streaming handoff", scope="all")
async def streaming_handoff_gen(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
    """Test streaming handoff (async generator)"""
    # Yield streaming chunks
    yield {
        "id": "chatcmpl-stream",
        "created": 1234567890,
        "model": "stream-model",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "Streaming "},
            "finish_reason": None
        }]
    }
    yield {
        "id": "chatcmpl-stream",
        "created": 1234567890,
        "model": "stream-model",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": "response"},
            "finish_reason": None
        }]
    }
    yield {
        "id": "chatcmpl-stream",
        "created": 1234567890,
        "model": "stream-model",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }


# ===== Test Classes =====

class TestHandoffRegistration:
    """Test handoff registration and configuration"""
    
    @pytest.mark.asyncio
    async def test_handoff_registration_via_skill(self):
        """Test handoff registration through skill initialization"""
        mock_skill = MockLLMSkill()
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            skills={"llm": mock_skill}
        )
        
        # Initialize skills to trigger handoff registration
        await agent._ensure_skills_initialized()
        
        # Check handoff was registered
        assert len(agent._registered_handoffs) == 1
        assert agent._registered_handoffs[0]['config'].target == "mock_llm"
        assert agent.active_handoff is not None
        assert agent.active_handoff.target == "mock_llm"
    
    def test_handoff_registration_via_decorator(self):
        """Test handoff registration via @handoff decorator"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[decorated_handoff]
        )
        
        # Check handoff was registered
        assert len(agent._registered_handoffs) == 1
        assert agent._registered_handoffs[0]['config'].target == "custom_handoff"
        assert agent.active_handoff.target == "custom_handoff"
    
    def test_handoff_registration_via_config(self):
        """Test handoff registration via Handoff config object"""
        
        async def handoff_func(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        
        config = Handoff(
            target="config_handoff",
            description="Test handoff via config",
            scope="owner",
            metadata={
                'function': handoff_func,
                'priority': 30,
                'is_generator': False
            }
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[config]
        )
        
        # Check handoff was registered
        assert len(agent._registered_handoffs) == 1
        assert agent._registered_handoffs[0]['config'].target == "config_handoff"
        assert agent._registered_handoffs[0]['config'].scope == "owner"
        assert agent.active_handoff.target == "config_handoff"
    
    def test_multiple_handoff_registration(self):
        """Test registering multiple handoffs"""
        async def handoff1(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "handoff1"}}]}
        
        async def handoff2(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "handoff2"}}]}
        
        config1 = Handoff(
            target="handoff1",
            description="First handoff",
            metadata={'function': handoff1, 'priority': 50}
        )
        config2 = Handoff(
            target="handoff2",
            description="Second handoff",
            metadata={'function': handoff2, 'priority': 30}
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[config1, config2]
        )
        
        # Check both handoffs were registered
        assert len(agent._registered_handoffs) == 2
        
        # Check priority-based default selection (lower priority = higher precedence)
        assert agent.active_handoff.target == "handoff2"  # Priority 30 (higher precedence)


class TestHandoffPriority:
    """Test priority-based handoff selection"""
    
    def test_priority_based_default_selection(self):
        """Test that lowest priority handoff becomes default"""
        async def low_priority(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "low"}}]}
        
        async def high_priority(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "high"}}]}
        
        config_low = Handoff(
            target="low_priority",
            metadata={'function': low_priority, 'priority': 10}  # Lower = higher precedence
        )
        config_high = Handoff(
            target="high_priority",
            metadata={'function': high_priority, 'priority': 90}
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[config_high, config_low]  # Register in wrong order
        )
        
        # Active handoff should be the one with lowest priority number
        assert agent.active_handoff.target == "low_priority"
    
    def test_priority_update_on_new_registration(self):
        """Test that active handoff updates when higher priority handoff is registered"""
        async def first_handoff(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "first"}}]}
        
        async def better_handoff(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "better"}}]}
        
        agent = BaseAgent(name="test-agent", instructions="Test")
        
        # Register first handoff
        config1 = Handoff(
            target="first",
            metadata={'function': first_handoff, 'priority': 50}
        )
        agent.register_handoff(config1, source="test")
        assert agent.active_handoff.target == "first"
        
        # Register better handoff (lower priority number)
        config2 = Handoff(
            target="better",
            metadata={'function': better_handoff, 'priority': 20}
        )
        agent.register_handoff(config2, source="test")
        assert agent.active_handoff.target == "better"


class TestHandoffExecution:
    """Test handoff execution (_execute_handoff method)"""
    
    @pytest.mark.asyncio
    async def test_execute_handoff_non_streaming(self):
        """Test executing non-streaming handoff"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[decorated_handoff]
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Execute handoff (non-streaming)
        result_coro = agent._execute_handoff(
            agent.active_handoff,
            messages,
            tools=[],
            stream=False
        )
        
        # Should return coroutine (await it)
        response = await result_coro
        
        # Check response
        assert response["choices"][0]["message"]["content"] == "Custom handoff response"
    
    @pytest.mark.asyncio
    async def test_execute_handoff_streaming(self):
        """Test executing streaming handoff"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[streaming_handoff_gen]
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Execute handoff (streaming) - returns async generator directly
        stream_gen = agent._execute_handoff(
            agent.active_handoff,
            messages,
            tools=[],
            stream=True
        )
        
        # Should be async generator (NO await!)
        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)
        
        # Check we got streaming chunks
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["content"] == "Streaming "
        assert chunks[1]["choices"][0]["delta"]["content"] == "response"
        assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_execute_non_generator_as_streaming(self):
        """Test executing non-generator handoff in streaming mode (should adapt)"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[decorated_handoff]
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Execute non-generator in streaming mode
        stream_gen = agent._execute_handoff(
            agent.active_handoff,
            messages,
            tools=[],
            stream=True
        )
        
        # Should adapt to streaming (yield single chunk)
        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)
        
        # Should have one chunk with full response
        assert len(chunks) == 1
        assert "delta" in chunks[0]["choices"][0]
    
    @pytest.mark.asyncio
    async def test_execute_generator_as_non_streaming(self):
        """Test executing generator handoff in non-streaming mode (should consume)"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[streaming_handoff_gen]
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Execute generator in non-streaming mode
        result_coro = agent._execute_handoff(
            agent.active_handoff,
            messages,
            tools=[],
            stream=False
        )
        
        # Should consume all chunks and return final response
        response = await result_coro
        
        # Check reconstructed response
        assert "choices" in response
        assert response["choices"][0]["finish_reason"] == "stop"


class TestHandoffIntegration:
    """Test handoff integration with agent run methods"""
    
    @pytest.mark.asyncio
    async def test_agent_run_with_handoff(self):
        """Test agent.run() uses handoff for completion"""
        mock_skill = MockLLMSkill()
        agent = BaseAgent(
            name="test-agent",
            instructions="Test instructions",
            skills={"llm": mock_skill}
        )
        
        # Create context
        messages = [{"role": "user", "content": "Hello"}]
        context = create_context(messages=messages, stream=False, agent=agent)
        set_context(context)
        
        # Run agent
        response = await agent.run(messages=messages, stream=False)
        
        # Check handoff was called
        assert mock_skill.call_count == 1
        assert response["choices"][0]["message"]["content"] == "Mock response #1"
    
    @pytest.mark.asyncio
    async def test_agent_run_streaming_with_handoff(self):
        """Test agent.run_streaming() uses handoff for completion"""
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[streaming_handoff_gen]
        )
        
        # Create context
        messages = [{"role": "user", "content": "Hello"}]
        context = create_context(messages=messages, stream=True, agent=agent)
        set_context(context)
        
        # Run agent in streaming mode
        chunks = []
        async for chunk in agent.run_streaming(messages=messages):
            chunks.append(chunk)
        
        # Should get streaming chunks
        assert len(chunks) >= 3


class TestHandoffScopes:
    """Test scope-based handoff access control"""
    
    def test_handoff_scope_inheritance(self):
        """Test handoffs inherit agent scopes when not explicitly set"""
        async def handoff_func(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        
        @handoff(name="explicit_scope", scope="admin")
        async def scoped_handoff(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "admin"}}]}
        
        # Agent with owner scope
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            scopes=["owner"],
            handoffs=[scoped_handoff]
        )
        
        # Check handoff has correct scope
        assert agent._registered_handoffs[0]['config'].scope == "admin"


class TestLocalAgentHandoff:
    """Test LocalAgentHandoff system for agent-to-agent transfers"""
    
    @pytest.mark.asyncio
    async def test_local_handoff_creation(self):
        """Test creating LocalAgentHandoff system"""
        agent1 = BaseAgent(name="agent1", instructions="Agent 1")
        agent2 = BaseAgent(name="agent2", instructions="Agent 2")
        
        handoff_system = create_local_handoff_system({
            "agent1": agent1,
            "agent2": agent2
        })
        
        assert len(handoff_system.agents) == 2
        assert "agent1" in handoff_system.agents
        assert "agent2" in handoff_system.agents
    
    @pytest.mark.asyncio
    async def test_local_handoff_agent_registration(self):
        """Test registering agents with LocalAgentHandoff"""
        handoff_system = LocalAgentHandoff({})
        
        agent = BaseAgent(name="test-agent", instructions="Test")
        handoff_system.register_agent("test-agent", agent)
        
        assert "test-agent" in handoff_system.agents
        assert handoff_system.agents["test-agent"] == agent
    
    @pytest.mark.asyncio
    async def test_local_handoff_agent_unregistration(self):
        """Test unregistering agents from LocalAgentHandoff"""
        agent = BaseAgent(name="test-agent", instructions="Test")
        handoff_system = LocalAgentHandoff({"test-agent": agent})
        
        handoff_system.unregister_agent("test-agent")
        
        assert "test-agent" not in handoff_system.agents
    
    @pytest.mark.asyncio
    async def test_local_handoff_get_available_agents(self):
        """Test getting list of available agents"""
        agent1 = BaseAgent(name="agent1", instructions="Agent 1")
        agent2 = BaseAgent(name="agent2", instructions="Agent 2")
        
        handoff_system = LocalAgentHandoff({
            "agent1": agent1,
            "agent2": agent2
        })
        
        available = handoff_system.get_available_agents()
        assert len(available) == 2
        assert "agent1" in available
        assert "agent2" in available
    
    @pytest.mark.asyncio
    async def test_local_handoff_execution(self):
        """Test executing handoff between agents"""
        # Create agents with mock LLM skills
        mock_skill1 = MockLLMSkill()
        mock_skill2 = MockLLMSkill()
        
        agent1 = BaseAgent(name="agent1", instructions="Agent 1", skills={"llm": mock_skill1})
        agent2 = BaseAgent(name="agent2", instructions="Agent 2", skills={"llm": mock_skill2})
        
        # Initialize skills
        await agent1._ensure_skills_initialized()
        await agent2._ensure_skills_initialized()
        
        handoff_system = LocalAgentHandoff({
            "agent1": agent1,
            "agent2": agent2
        })
        
        # Create context for handoff
        messages = [{"role": "user", "content": "Hello"}]
        context = create_context(
            messages=messages,
            stream=False,
            agent=agent1
        )
        set_context(context)
        
        # Execute handoff
        result = await handoff_system.execute_handoff(
            source_agent="agent1",
            target_agent="agent2",
            handoff_data={"reason": "User requested transfer"}
        )
        
        # Check result
        assert result.success
        assert result.handoff_type == "local_agent"
        assert "agent2" in result.metadata["target_agent"]
    
    @pytest.mark.asyncio
    async def test_local_handoff_invalid_source(self):
        """Test handoff with invalid source agent"""
        agent = BaseAgent(name="agent1", instructions="Test")
        handoff_system = LocalAgentHandoff({"agent1": agent})
        
        # Create context
        context = create_context(
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            agent=agent
        )
        set_context(context)
        
        # Try handoff with invalid source
        result = await handoff_system.execute_handoff(
            source_agent="nonexistent",
            target_agent="agent1"
        )
        
        assert not result.success
        assert "not found" in result.metadata.get("error", "").lower()
    
    @pytest.mark.asyncio
    async def test_local_handoff_statistics(self):
        """Test handoff statistics tracking"""
        mock_skill1 = MockLLMSkill()
        mock_skill2 = MockLLMSkill()
        
        agent1 = BaseAgent(name="agent1", instructions="Agent 1", skills={"llm": mock_skill1})
        agent2 = BaseAgent(name="agent2", instructions="Agent 2", skills={"llm": mock_skill2})
        
        await agent1._ensure_skills_initialized()
        await agent2._ensure_skills_initialized()
        
        handoff_system = LocalAgentHandoff({
            "agent1": agent1,
            "agent2": agent2
        })
        
        # Create context
        context = create_context(
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            agent=agent1
        )
        set_context(context)
        
        # Execute a successful handoff
        result = await handoff_system.execute_handoff(
            source_agent="agent1",
            target_agent="agent2"
        )
        
        # Get statistics
        stats = handoff_system.get_handoff_stats()
        
        assert stats["total_handoffs"] == 1
        assert stats["successful_handoffs"] == 1
        assert stats["failed_handoffs"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["most_common_source"] == "agent1"
        assert stats["most_common_target"] == "agent2"


class TestHandoffEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_agent_without_handoff_raises_error(self):
        """Test that agent without handoff raises error on run()"""
        agent = BaseAgent(name="test-agent", instructions="Test")
        
        # Create context
        context = create_context(
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            agent=agent
        )
        set_context(context)
        
        # Should raise error about missing handoff
        with pytest.raises(ValueError, match="No handoff registered"):
            asyncio.run(agent.run(messages=[{"role": "user", "content": "test"}]))
    
    def test_handoff_without_function_raises_error(self):
        """Test that handoff without function raises error"""
        config = Handoff(
            target="broken_handoff",
            metadata={'is_generator': False}  # No function!
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[config]
        )
        
        messages = [{"role": "user", "content": "test"}]
        
        # Should raise error about missing function
        with pytest.raises(ValueError, match="No function for handoff"):
            asyncio.run(agent._execute_handoff(
                agent.active_handoff,
                messages,
                tools=[],
                stream=False
            ))
    
    @pytest.mark.asyncio
    async def test_handoff_prompt_registration(self):
        """Test that handoff description is registered as prompt"""
        async def test_handoff(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        
        config = Handoff(
            target="test_handoff",
            description="This is a test handoff prompt that should be registered",
            metadata={'function': test_handoff, 'priority': 50}
        )
        
        agent = BaseAgent(
            name="test-agent",
            instructions="Test",
            handoffs=[config]
        )
        
        # Check that handoff prompt was registered
        prompts = agent.get_prompts_for_scope("all")
        
        # Should have at least the handoff prompt
        prompt_sources = [p['source'] for p in prompts]
        assert any('handoff_prompt' in source for source in prompt_sources)


class TestHandoffDecorator:
    """Test @handoff decorator functionality"""
    
    def test_handoff_decorator_attributes(self):
        """Test that @handoff decorator sets correct attributes"""
        @handoff(name="test_handoff", prompt="Test handoff prompt", scope="owner")
        async def test_func(messages, tools=None, **kwargs):
            return {"result": "test"}
        
        assert hasattr(test_func, '_webagents_is_handoff')
        assert test_func._webagents_is_handoff is True
        assert test_func._handoff_name == "test_handoff"
        assert test_func._handoff_prompt == "Test handoff prompt"
        assert test_func._handoff_scope == "owner"
    
    def test_handoff_decorator_defaults(self):
        """Test @handoff decorator default values"""
        @handoff()
        async def default_handoff(messages, tools=None, **kwargs):
            """Default handoff docstring"""
            return {"result": "test"}
        
        assert default_handoff._handoff_name == "default_handoff"  # Uses function name
        assert default_handoff._handoff_prompt == "Default handoff docstring"
        assert default_handoff._handoff_scope == "all"
    
    def test_agent_handoff_decorator_direct_registration(self):
        """Test @agent.handoff direct registration"""
        agent = BaseAgent(name="test-agent", instructions="Test")
        
        @agent.handoff(name="direct_handoff")
        async def direct_handoff_func(messages, tools=None, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "direct"}}]}
        
        # Check handoff was registered
        assert len(agent._registered_handoffs) == 1
        assert agent._registered_handoffs[0]['config'].target == "direct_handoff"


# ===== Run Tests =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

