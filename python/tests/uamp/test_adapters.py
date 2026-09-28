"""
Tests for UAMP Protocol Adapters.

Covers the transport adapters that still exist:
- CompletionsUAMPAdapter (OpenAI Chat Completions)
- RealtimeUAMPAdapter (OpenAI Realtime)

The A2A and ACP adapters were removed on 2026-09-26 with the transports they
served (gap-closure lanes w1-a2a and w1-protocols): A2A v1.0 maps parts in
`transport/a2a/protocol.py` (tests/test_transport_a2a_v1.py), and ACP now runs
over stdio (`transport/acp/protocol.py`, tests/test_acp_protocol.py and
tests/test_acp_stdio.py). Their adapter tests, including the ACP-to-Completions
round trip, went with them.
"""

import pytest
import json

from webagents.agents.skills.core.transport.completions.uamp_adapter import CompletionsUAMPAdapter
from webagents.agents.skills.core.transport.realtime.uamp_adapter import RealtimeUAMPAdapter
from webagents.uamp import (
    SessionCreateEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
    InputTextEvent,
    InputAudioEvent,
    InputImageEvent,
    InputFileEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ToolCallEvent,
    ToolResultEvent,
    ProgressEvent,
    AudioDeltaEvent,
    TranscriptDeltaEvent,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
    SessionConfig,
    Session,
    VoiceConfig,
)


# =============================================================================
# Completions Adapter Tests
# =============================================================================

class TestCompletionsUAMPAdapter:
    """Tests for the OpenAI Chat Completions adapter."""
    
    @pytest.fixture
    def adapter(self):
        return CompletionsUAMPAdapter()
    
    def test_simple_message_to_uamp(self, adapter):
        """Simple message should convert to UAMP events."""
        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
        
        events = adapter.to_uamp(request)
        
        # Should have: session.create, input.text, response.create
        assert len(events) == 3
        assert isinstance(events[0], SessionCreateEvent)
        assert isinstance(events[1], InputTextEvent)
        assert isinstance(events[2], ResponseCreateEvent)
        
        # Check input text
        assert events[1].text == "Hello!"
        assert events[1].role == "user"
    
    def test_system_message_to_uamp(self, adapter):
        """System message should be preserved."""
        request = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ]
        }
        
        events = adapter.to_uamp(request)
        
        # Find system message
        system_events = [e for e in events if isinstance(e, InputTextEvent) and e.role == "system"]
        assert len(system_events) == 1
        assert system_events[0].text == "You are helpful"
    
    def test_tool_result_to_uamp(self, adapter):
        """Tool result message should convert to ToolResultEvent."""
        request = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": 72}'}
            ]
        }
        
        events = adapter.to_uamp(request)
        
        # Find tool result
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].call_id == "call_123"
        assert tool_results[0].result == '{"temp": 72}'
    
    def test_tools_in_session(self, adapter):
        """Tools should be passed to session config."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"}
                }
            }
        ]
        request = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": tools
        }
        
        events = adapter.to_uamp(request)
        session_event = events[0]
        
        assert isinstance(session_event, SessionCreateEvent)
        assert session_event.session.tools == tools
    
    def test_simple_response_from_uamp(self, adapter):
        """Simple response should convert from UAMP events."""
        events = [
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text="Hello")
            ),
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text=" there!")
            ),
            ResponseDoneEvent(
                response_id="resp_123",
                response=ResponseOutput(
                    id="resp_123",
                    status="completed",
                    output=[],
                    usage=UsageStats(input_tokens=5, output_tokens=3, total_tokens=8)
                )
            )
        ]
        
        response = adapter.from_uamp(events)
        
        assert response["id"] == "resp_123"
        assert response["choices"][0]["message"]["content"] == "Hello there!"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"]["total_tokens"] == 8
    
    def test_tool_call_response_from_uamp(self, adapter):
        """Tool call response should convert from UAMP events."""
        events = [
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(
                    type="tool_call",
                    tool_call={
                        "id": "call_abc",
                        "name": "get_weather",
                        "arguments": '{"loc":'
                    }
                )
            ),
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(
                    type="tool_call",
                    tool_call={
                        "id": "call_abc",
                        "name": "get_weather",
                        "arguments": '"NYC"}'
                    }
                )
            ),
            ResponseDoneEvent(
                response_id="resp_123",
                response=ResponseOutput(id="resp_123", status="completed", output=[])
            )
        ]
        
        response = adapter.from_uamp(events)
        
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"loc":"NYC"}'
    
    def test_streaming_text_chunk(self, adapter):
        """Streaming text should produce SSE format."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="Hi")
        )
        
        chunk = adapter.from_uamp_streaming(event)
        
        assert chunk.startswith("data: ")
        data = json.loads(chunk.replace("data: ", "").strip())
        assert data["choices"][0]["delta"]["content"] == "Hi"
    
    def test_streaming_done_chunk(self, adapter):
        """Done event should produce [DONE] marker."""
        event = ResponseDoneEvent(
            response_id="resp_123",
            response=ResponseOutput(id="resp_123", status="completed", output=[])
        )
        
        chunk = adapter.from_uamp_streaming(event)
        
        assert "data: [DONE]" in chunk
    
    def test_messages_to_uamp_convenience(self):
        """messages_to_uamp should convert messages only."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        events = CompletionsUAMPAdapter.messages_to_uamp(messages)
        
        # Should have session + message + response.create
        assert len(events) == 3
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello"


# =============================================================================
# Realtime Adapter Tests
# =============================================================================

class TestRealtimeUAMPAdapter:
    """Tests for the OpenAI Realtime API adapter."""
    
    @pytest.fixture
    def adapter(self):
        return RealtimeUAMPAdapter()
    
    def test_session_update_to_uamp(self, adapter):
        """session.update should convert to SessionUpdateEvent."""
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "nova",
                "instructions": "Be helpful"
            }
        }
        
        result = adapter.to_uamp(event)
        
        assert isinstance(result, SessionUpdateEvent)
        assert result.session.modalities == ["text", "audio"]
        assert result.session.voice.name == "nova"
        assert result.session.instructions == "Be helpful"
    
    def test_audio_append_to_uamp(self, adapter):
        """input_audio_buffer.append should convert to InputAudioEvent."""
        event = {
            "type": "input_audio_buffer.append",
            "audio": "base64audiodata"
        }
        
        result = adapter.to_uamp(event)
        
        assert isinstance(result, InputAudioEvent)
        assert result.audio == "base64audiodata"
        assert result.format == "pcm16"
    
    def test_conversation_item_text_to_uamp(self, adapter):
        """Text conversation item should convert to InputTextEvent."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello realtime!"}]
            }
        }
        
        result = adapter.to_uamp(event)
        
        assert isinstance(result, InputTextEvent)
        assert result.text == "Hello realtime!"
        assert result.role == "user"
    
    def test_response_create_to_uamp(self, adapter):
        """response.create should convert to ResponseCreateEvent."""
        event = {"type": "response.create"}
        
        result = adapter.to_uamp(event)
        
        assert isinstance(result, ResponseCreateEvent)
    
    def test_response_cancel_to_uamp(self, adapter):
        """response.cancel should convert to ResponseCancelEvent."""
        event = {"type": "response.cancel"}
        
        result = adapter.to_uamp(event)
        
        assert isinstance(result, ResponseCancelEvent)
    
    def test_session_created_from_uamp(self, adapter):
        """SessionCreatedEvent should convert to session.created."""
        event = SessionCreatedEvent(
            session=Session(
                id="sess_123",
                config=SessionConfig(
                    modalities=["text", "audio"],
                    voice=VoiceConfig(name="alloy"),
                    instructions="Be helpful"
                )
            )
        )
        
        result = adapter.from_uamp(event)
        
        assert result["type"] == "session.created"
        assert result["session"]["id"] == "sess_123"
        assert result["session"]["modalities"] == ["text", "audio"]
        assert result["session"]["voice"] == "alloy"
    
    def test_text_delta_from_uamp(self, adapter):
        """Text delta should convert to response.text.delta."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="Hello")
        )
        
        result = adapter.from_uamp(event)
        
        assert result["type"] == "response.text.delta"
        assert result["delta"] == "Hello"
    
    def test_audio_delta_from_uamp(self, adapter):
        """AudioDeltaEvent should convert to response.audio.delta."""
        event = AudioDeltaEvent(
            response_id="resp_123",
            audio="base64audio"
        )
        
        result = adapter.from_uamp(event)
        
        assert result["type"] == "response.audio.delta"
        assert result["delta"] == "base64audio"
    
    def test_response_done_from_uamp(self, adapter):
        """ResponseDoneEvent should convert to response.done."""
        event = ResponseDoneEvent(
            response_id="resp_123",
            response=ResponseOutput(
                id="resp_123",
                status="completed",
                output=[ContentItem(type="text", text="Done!")],
                usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
            )
        )
        
        result = adapter.from_uamp(event)
        
        assert result["type"] == "response.done"
        assert result["response"]["status"] == "completed"
        assert result["response"]["usage"]["total_tokens"] == 15
