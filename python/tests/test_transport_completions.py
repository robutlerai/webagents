"""
Comprehensive tests for CompletionsTransportSkill

Tests OpenAI Chat Completions API compatibility:
- Request parameters
- Response format
- Streaming
- Tool calling
- Error handling
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from webagents.agents.skills.core.transport.completions.skill import CompletionsTransportSkill


# ============================================================================
# Mock Fixtures
# ============================================================================

class MockAgent:
    """Mock agent for testing"""
    def __init__(self):
        self.name = "test-agent"
        self.skills = {}
        self._registered_handoffs = []
        self.active_handoff = None
    
    async def run_streaming(self, messages, tools=None, **kwargs):
        """Mock streaming response"""
        yield {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]}
        yield {"id": "chatcmpl-123", "choices": [{"delta": {"content": " World"}}]}
        yield {"id": "chatcmpl-123", "choices": [{"delta": {}, "finish_reason": "stop"}]}


class MockContext:
    """Mock context for testing"""
    def __init__(self, agent=None):
        self.agent = agent
        self.messages = []
        self.stream = True
        self.auth = None


@pytest.fixture
def skill():
    return CompletionsTransportSkill()


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_context(mock_agent):
    return MockContext(mock_agent)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestCompletionsInitialization:
    """Test skill initialization"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, skill, mock_agent):
        """Test basic initialization"""
        await skill.initialize(mock_agent)
        assert skill.agent == mock_agent
    
    def test_default_scope(self, skill):
        """Test default scope is 'all'"""
        assert skill.scope == "all"
    
    def test_http_endpoint_registered(self, skill):
        """Test HTTP endpoint is properly decorated"""
        assert hasattr(skill.chat_completions, '_webagents_is_http')
        assert skill.chat_completions._webagents_is_http is True
        assert skill.chat_completions._http_subpath == '/chat/completions'
        assert skill.chat_completions._http_method == 'post'


# ============================================================================
# Request Parameter Tests
# ============================================================================

class TestCompletionsRequestParameters:
    """Test all OpenAI request parameters are supported"""
    
    @pytest.mark.asyncio
    async def test_messages_parameter(self, skill, mock_agent, mock_context):
        """Test messages are passed correctly"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "Hi"}}]}
                mock_handoff.return_value = mock_stream()
                
                messages = [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
                
                async for _ in skill.chat_completions(messages=messages):
                    pass
                
                call_args = mock_handoff.call_args
                assert call_args[0][0] == messages
    
    @pytest.mark.asyncio
    async def test_temperature_parameter(self, skill, mock_agent, mock_context):
        """Test temperature is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    temperature=0.7
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('temperature') == 0.7
    
    @pytest.mark.asyncio
    async def test_max_tokens_parameter(self, skill, mock_agent, mock_context):
        """Test max_tokens is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=500
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('max_tokens') == 500
    
    @pytest.mark.asyncio
    async def test_top_p_parameter(self, skill, mock_agent, mock_context):
        """Test top_p (nucleus sampling) is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    top_p=0.9
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('top_p') == 0.9
    
    @pytest.mark.asyncio
    async def test_frequency_penalty_parameter(self, skill, mock_agent, mock_context):
        """Test frequency_penalty is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    frequency_penalty=0.5
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('frequency_penalty') == 0.5
    
    @pytest.mark.asyncio
    async def test_presence_penalty_parameter(self, skill, mock_agent, mock_context):
        """Test presence_penalty is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    presence_penalty=0.3
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('presence_penalty') == 0.3
    
    @pytest.mark.asyncio
    async def test_stop_sequences_parameter(self, skill, mock_agent, mock_context):
        """Test stop sequences are forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    stop=["END", "STOP"]
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('stop') == ["END", "STOP"]
    
    @pytest.mark.asyncio
    async def test_n_parameter(self, skill, mock_agent, mock_context):
        """Test n (number of completions) is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    n=3
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('n') == 3
    
    @pytest.mark.asyncio
    async def test_response_format_parameter(self, skill, mock_agent, mock_context):
        """Test response_format is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": '{"key": "value"}'}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    response_format={"type": "json_object"}
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('response_format') == {"type": "json_object"}
    
    @pytest.mark.asyncio
    async def test_seed_parameter(self, skill, mock_agent, mock_context):
        """Test seed (deterministic outputs) is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    seed=12345
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('seed') == 12345
    
    @pytest.mark.asyncio
    async def test_user_parameter(self, skill, mock_agent, mock_context):
        """Test user (tracking) is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    user="user-123"
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('user') == "user-123"
    
    @pytest.mark.asyncio
    async def test_logprobs_parameters(self, skill, mock_agent, mock_context):
        """Test logprobs and top_logprobs are forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    logprobs=True,
                    top_logprobs=5
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('logprobs') is True
                assert mock_handoff.call_args.kwargs.get('top_logprobs') == 5


# ============================================================================
# Tool Calling Tests
# ============================================================================

class TestCompletionsToolCalling:
    """Test tool calling functionality"""
    
    @pytest.mark.asyncio
    async def test_tools_parameter(self, skill, mock_agent, mock_context):
        """Test tools are passed correctly"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }]
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Weather?"}],
                    tools=tools
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('tools') == tools
    
    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, skill, mock_agent, mock_context):
        """Test tool_choice='auto' is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    tool_choice="auto"
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('tool_choice') == "auto"
    
    @pytest.mark.asyncio
    async def test_tool_choice_required(self, skill, mock_agent, mock_context):
        """Test tool_choice='required' is forwarded"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    tool_choice="required"
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('tool_choice') == "required"
    
    @pytest.mark.asyncio
    async def test_tool_choice_specific_function(self, skill, mock_agent, mock_context):
        """Test tool_choice with specific function"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                tool_choice = {"type": "function", "function": {"name": "get_weather"}}
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    tool_choice=tool_choice
                ):
                    pass
                
                assert mock_handoff.call_args.kwargs.get('tool_choice') == tool_choice


# ============================================================================
# Streaming Tests
# ============================================================================

class TestCompletionsStreaming:
    """Test SSE streaming functionality"""
    
    @pytest.mark.asyncio
    async def test_streaming_is_off_by_default(self, skill, mock_agent, mock_context):
        """`stream` omitted means a JSON completion, per the OpenAI API.

        This was `test_streaming_enabled_by_default` and asserted SSE for an
        omitted `stream`, i.e. it pinned the bug (2026-09-24): the official
        SDKs omit the field on a plain call and could not parse the answer.
        """
        await skill.initialize(mock_agent)

        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "Hello"}}]}
                    yield {"choices": [{"delta": {"content": " World"}}]}
                mock_handoff.return_value = mock_stream()

                chunks = [c async for c in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}]
                )]

                from starlette.responses import JSONResponse

                assert len(chunks) == 1 and isinstance(chunks[0], JSONResponse)
                assert json.loads(chunks[0].body)["choices"][0]["message"]["content"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_sse_format(self, skill, mock_agent, mock_context):
        """Test SSE format is correct"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                chunks = []
                async for chunk in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}], stream=True
                ):
                    chunks.append(chunk)
                
                # Each chunk should start with "data: " and end with "\n\n"
                for chunk in chunks:
                    assert chunk.startswith("data: ")
                    assert chunk.endswith("\n\n")
    
    @pytest.mark.asyncio
    async def test_non_streaming_mode(self, skill, mock_agent, mock_context):
        """Test non-streaming mode collects all chunks"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"id": "123", "choices": [{"delta": {"content": "Hello"}}]}
                    yield {"id": "123", "choices": [{"delta": {"content": " World"}}]}
                    yield {"id": "123", "choices": [{"delta": {}, "finish_reason": "stop"}]}
                mock_handoff.return_value = mock_stream()
                
                chunks = []
                async for chunk in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=False
                ):
                    chunks.append(chunk)

                # ONE JSON response (2026-09-24). This asserted "merged
                # response + [DONE]", i.e. it pinned the bug: a stream=false
                # request answered with SSE framing that no OpenAI client
                # parses. The server returns a Response yielded first as the
                # whole answer.
                from starlette.responses import JSONResponse

                assert len(chunks) == 1
                assert isinstance(chunks[0], JSONResponse)
                assert chunks[0].media_type == "application/json"
                body = json.loads(chunks[0].body)
                assert body["object"] == "chat.completion"
                assert body["choices"][0]["message"]["content"] == "Hello World"
                assert body["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_non_streaming_with_no_output_is_still_a_completion(self, skill, mock_agent, mock_context):
        """Not `{}`: a client must still be able to read `choices[0]`."""
        await skill.initialize(mock_agent)

        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    return
                    yield  # pragma: no cover - makes this an async generator

                mock_handoff.return_value = mock_stream()

                chunks = [c async for c in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}], stream=False
                )]

                body = json.loads(chunks[0].body)
                assert body["choices"][0]["message"]["content"] == ""


# ============================================================================
# Response Format Tests
# ============================================================================

class TestCompletionsResponseFormat:
    """Test response format and chunk merging"""
    
    def test_merge_empty_chunks(self, skill):
        """Test merging empty chunks"""
        result = skill._merge_streaming_chunks([])
        assert result == {}
    
    def test_merge_single_chunk(self, skill):
        """Test merging single chunk"""
        chunks = [{
            "id": "chatcmpl-123",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{"delta": {"content": "Hello"}}]
        }]
        
        result = skill._merge_streaming_chunks(chunks)
        
        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello"
    
    def test_merge_multiple_chunks(self, skill):
        """Test merging multiple chunks"""
        chunks = [
            {"id": "123", "created": 1234567890, "model": "gpt-4", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "123", "created": 1234567890, "model": "gpt-4", "choices": [{"delta": {"content": " World"}}]},
            {"id": "123", "created": 1234567890, "model": "gpt-4", "choices": [{"delta": {}, "finish_reason": "stop"}]}
        ]
        
        result = skill._merge_streaming_chunks(chunks)
        
        assert result["choices"][0]["message"]["content"] == "Hello World"
        assert result["choices"][0]["finish_reason"] == "stop"
    
    def test_merge_with_tool_calls(self, skill):
        """Test merging chunks with tool calls"""
        chunks = [
            {"id": "123", "choices": [{"delta": {"tool_calls": [{"id": "call_1", "function": {"name": "test"}}]}}]},
            {"id": "123", "choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"a":1}'}}]}}]}
        ]
        
        result = skill._merge_streaming_chunks(chunks)
        
        assert "tool_calls" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["tool_calls"]) == 2
    
    def test_merge_real_sdk_chunks_with_null_tool_calls(self, skill):
        """The shape the OpenAI SDK actually produces (2026-09-24).

        Every plain-text delta carries `"tool_calls": null` (and
        `function_call`, `refusal`). The hand-written chunks above never
        included the null keys, which is how `extend(None)` survived: the
        daemon answered 500 to every non-streaming chat with a file agent.
        Copied from a live daemon's streamed output.
        """
        sdk_delta = {"function_call": None, "refusal": None, "role": "assistant", "tool_calls": None}
        chunks = [
            {"id": "c", "choices": [{"index": 0, "delta": {**sdk_delta, "content": "Hello"}, "finish_reason": None}]},
            {"id": "c", "choices": [{"index": 0, "delta": {**sdk_delta, "content": " world"}, "finish_reason": None}]},
            {"id": "c", "choices": [{"index": 0, "delta": {**sdk_delta, "content": None}, "finish_reason": "stop"}]},
        ]

        result = skill._merge_streaming_chunks(chunks)

        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in result["choices"][0]["message"]

    def test_merge_with_usage(self, skill):
        """Test merging captures usage from last chunk"""
        chunks = [
            {"id": "123", "choices": [{"delta": {"content": "Hi"}}]},
            {"id": "123", "choices": [{"delta": {}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        ]
        
        result = skill._merge_streaming_chunks(chunks)
        
        assert result["usage"]["total_tokens"] == 15


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestCompletionsEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_messages(self, skill, mock_agent, mock_context):
        """Test handling empty messages"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                chunks = []
                async for chunk in skill.chat_completions(messages=[]):
                    chunks.append(chunk)
                
                # Should still work with empty messages
                assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_none_messages_defaulted(self, skill, mock_agent, mock_context):
        """Test None messages are defaulted to empty list"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                chunks = []
                async for chunk in skill.chat_completions(messages=None):
                    chunks.append(chunk)
                
                # Should use empty list
                assert mock_handoff.call_args[0][0] == []
    
    @pytest.mark.asyncio
    async def test_all_parameters_combined(self, skill, mock_agent, mock_context):
        """Test all parameters can be combined"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "test"}}]}
                mock_handoff.return_value = mock_stream()
                
                async for _ in skill.chat_completions(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="gpt-4",
                    temperature=0.5,
                    max_tokens=100,
                    top_p=0.9,
                    frequency_penalty=0.5,
                    presence_penalty=0.5,
                    stop=["END"],
                    n=1,
                    response_format={"type": "json_object"},
                    seed=42,
                    user="test-user",
                    logprobs=True,
                    top_logprobs=3,
                    tools=[{"type": "function", "function": {"name": "test"}}],
                    tool_choice="auto"
                ):
                    pass
                
                kwargs = mock_handoff.call_args.kwargs
                assert kwargs['temperature'] == 0.5
                assert kwargs['max_tokens'] == 100
                assert kwargs['top_p'] == 0.9
                assert kwargs['seed'] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
