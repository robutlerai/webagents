"""
Tests for UAMP Protocol Adapters.

Covers all transport adapters:
- CompletionsUAMPAdapter (OpenAI Chat Completions)
- A2AUAMPAdapter (Google Agent2Agent)
- RealtimeUAMPAdapter (OpenAI Realtime)
- ACPUAMPAdapter (Agent Client Protocol)
"""

import pytest
import json

from webagents.agents.skills.core.transport.completions.uamp_adapter import CompletionsUAMPAdapter
from webagents.agents.skills.core.transport.a2a.uamp_adapter import A2AUAMPAdapter
from webagents.agents.skills.core.transport.realtime.uamp_adapter import RealtimeUAMPAdapter
from webagents.agents.skills.core.transport.acp.uamp_adapter import ACPUAMPAdapter
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
# A2A Adapter Tests
# =============================================================================

class TestA2AUAMPAdapter:
    """Tests for the Google A2A Protocol adapter."""
    
    @pytest.fixture
    def adapter(self):
        return A2AUAMPAdapter()
    
    def test_simple_text_message_to_uamp(self, adapter):
        """Simple text message should convert to UAMP events."""
        request = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello A2A!"}]
            }
        }
        
        events = adapter.to_uamp(request)
        
        assert len(events) == 3  # session, text, response.create
        assert isinstance(events[0], SessionCreateEvent)
        assert isinstance(events[1], InputTextEvent)
        assert isinstance(events[2], ResponseCreateEvent)
        
        assert events[1].text == "Hello A2A!"
        assert events[1].role == "user"
    
    def test_agent_role_converts_to_assistant(self, adapter):
        """A2A 'agent' role should convert to 'assistant'."""
        request = {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "Hi"}]},
                {"role": "agent", "parts": [{"type": "text", "text": "Hello!"}]}
            ]
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert text_events[0].role == "user"
        assert text_events[1].role == "assistant"
    
    def test_image_file_to_uamp(self, adapter):
        """Image file part should convert to InputImageEvent."""
        request = {
            "message": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "file",
                        "file": {
                            "name": "photo.png",
                            "mimeType": "image/png",
                            "data": "base64encodeddata"
                        }
                    }
                ]
            }
        }
        
        events = adapter.to_uamp(request)
        
        image_events = [e for e in events if isinstance(e, InputImageEvent)]
        assert len(image_events) == 1
        assert "data:image/png;base64,base64encodeddata" in image_events[0].image
    
    def test_generic_file_to_uamp(self, adapter):
        """Non-image file should convert to InputFileEvent."""
        request = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "file",
                        "file": {
                            "name": "document.pdf",
                            "mimeType": "application/pdf",
                            "data": "base64data"
                        }
                    }
                ]
            }
        }
        
        events = adapter.to_uamp(request)
        
        file_events = [e for e in events if isinstance(e, InputFileEvent)]
        assert len(file_events) == 1
        assert file_events[0].filename == "document.pdf"
        assert file_events[0].mime_type == "application/pdf"
    
    def test_data_part_to_uamp(self, adapter):
        """Data part should convert to text with JSON."""
        request = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "data",
                        "data": {"key": "value", "count": 42},
                        "mimeType": "application/json"
                    }
                ]
            }
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert "application/json" in text_events[0].text
        assert '"key": "value"' in text_events[0].text
    
    def test_response_from_uamp(self, adapter):
        """UAMP events should convert to A2A task result."""
        events = [
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text="Hello from ")
            ),
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text="A2A!")
            ),
            ResponseDoneEvent(
                response_id="resp_123",
                response=ResponseOutput(id="resp_123", status="completed", output=[])
            )
        ]
        
        result = adapter.from_uamp(events)
        
        assert result["status"] == "completed"
        assert result["message"]["role"] == "agent"
        assert len(result["message"]["parts"]) == 1
        assert result["message"]["parts"][0]["text"] == "Hello from A2A!"
    
    def test_streaming_text_event(self, adapter):
        """Streaming text delta should produce task.message event."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="chunk")
        )
        
        result = adapter.from_uamp_streaming(event)
        
        assert result["event"] == "task.message"
        assert result["data"]["role"] == "agent"
        assert result["data"]["parts"][0]["text"] == "chunk"
    
    def test_streaming_progress_event(self, adapter):
        """Progress event should produce task.progress event."""
        event = ProgressEvent(
            stage="processing",
            message="Analyzing input...",
            percent=50
        )
        
        result = adapter.from_uamp_streaming(event)
        
        assert result["event"] == "task.progress"
        assert result["data"]["stage"] == "processing"
        assert result["data"]["percent"] == 50


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


# =============================================================================
# ACP Adapter Tests
# =============================================================================

class TestACPUAMPAdapter:
    """Tests for the Agent Client Protocol adapter."""
    
    @pytest.fixture
    def adapter(self):
        return ACPUAMPAdapter()
    
    def test_prompt_submit_to_uamp(self, adapter):
        """prompt/submit should convert to UAMP events."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompt/submit",
            "params": {
                "messages": [
                    {"role": "user", "content": "Hello ACP!"}
                ]
            }
        }
        
        events = adapter.to_uamp(request)
        
        assert len(events) == 3  # session, text, response.create
        assert isinstance(events[0], SessionCreateEvent)
        assert isinstance(events[1], InputTextEvent)
        assert isinstance(events[2], ResponseCreateEvent)
        
        assert events[1].text == "Hello ACP!"
    
    def test_chat_submit_to_uamp(self, adapter):
        """chat/submit should also convert to UAMP events."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "chat/submit",
            "params": {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"}
                ]
            }
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 2
        assert text_events[0].role == "system"
        assert text_events[1].role == "user"
    
    def test_tool_call_to_uamp(self, adapter):
        """tools/call should convert to ToolResultEvent."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_weather",
                "arguments": {"location": "NYC"}
            }
        }
        
        events = adapter.to_uamp(request)
        
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert "get_weather" in events[0].call_id
    
    def test_response_from_uamp(self, adapter):
        """UAMP events should convert to ACP JSON-RPC response."""
        events = [
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text="Hello ")
            ),
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(type="text", text="ACP!")
            ),
            ResponseDoneEvent(
                response_id="resp_123",
                response=ResponseOutput(id="resp_123", status="completed", output=[])
            )
        ]
        
        result = adapter.from_uamp(events, request_id=1)
        
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"]["status"] == "complete"
        assert result["result"]["content"] == "Hello ACP!"
    
    def test_tool_calls_from_uamp(self, adapter):
        """Tool call deltas should accumulate in response."""
        events = [
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(
                    type="tool_call",
                    tool_call={"id": "call_1", "name": "search", "arguments": '{"q":'}
                )
            ),
            ResponseDeltaEvent(
                response_id="resp_123",
                delta=ContentDelta(
                    type="tool_call",
                    tool_call={"id": "call_1", "name": "search", "arguments": '"test"}'}
                )
            ),
            ResponseDoneEvent(
                response_id="resp_123",
                response=ResponseOutput(id="resp_123", status="completed", output=[])
            )
        ]
        
        result = adapter.from_uamp(events, request_id=1)
        
        assert len(result["result"]["tool_calls"]) == 1
        assert result["result"]["tool_calls"][0]["name"] == "search"
        assert result["result"]["tool_calls"][0]["arguments"] == '{"q":"test"}'
    
    def test_streaming_text_notification(self, adapter):
        """Text delta should produce prompt/progress notification."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="chunk")
        )
        
        result = adapter.from_uamp_streaming(event, request_id=1)
        
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "prompt/progress"
        assert result["params"]["content"] == "chunk"
        assert result["params"]["requestId"] == "1"
    
    def test_streaming_done_notification(self, adapter):
        """ResponseDone should produce prompt/done notification."""
        event = ResponseDoneEvent(
            response_id="resp_123",
            response=ResponseOutput(id="resp_123", status="completed", output=[])
        )
        
        result = adapter.from_uamp_streaming(event, request_id=1)
        
        assert result["method"] == "prompt/done"
        assert result["params"]["status"] == "complete"
    
    def test_helper_methods(self, adapter):
        """Helper methods should create valid JSON-RPC structures."""
        notification = adapter.make_notification("test/event", {"key": "value"})
        assert notification["jsonrpc"] == "2.0"
        assert notification["method"] == "test/event"
        assert "id" not in notification
        
        response = adapter.make_response(42, {"data": "test"})
        assert response["id"] == 42
        assert response["result"]["data"] == "test"
        
        error = adapter.make_error(1, -32600, "Invalid request")
        assert error["error"]["code"] == -32600
        assert error["error"]["message"] == "Invalid request"


# =============================================================================
# Cross-Adapter Compatibility Tests
# =============================================================================

class TestCrossAdapterCompatibility:
    """Tests to ensure adapters can interoperate through UAMP."""
    
    def test_completions_to_a2a_round_trip(self):
        """Message can flow: Completions -> UAMP -> A2A format."""
        completions_adapter = CompletionsUAMPAdapter()
        a2a_adapter = A2AUAMPAdapter()
        
        # Convert completions request to UAMP
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        uamp_events = completions_adapter.to_uamp(request)
        
        # Verify UAMP events are valid
        assert any(isinstance(e, SessionCreateEvent) for e in uamp_events)
        assert any(isinstance(e, InputTextEvent) for e in uamp_events)
        
        # Simulate response through UAMP
        response_events = [
            ResponseDeltaEvent(
                response_id="resp_1",
                delta=ContentDelta(type="text", text="Hi there!")
            ),
            ResponseDoneEvent(
                response_id="resp_1",
                response=ResponseOutput(id="resp_1", status="completed", output=[])
            )
        ]
        
        # Convert UAMP response to A2A format
        a2a_result = a2a_adapter.from_uamp(response_events)
        
        assert a2a_result["message"]["parts"][0]["text"] == "Hi there!"
        assert a2a_result["message"]["role"] == "agent"
    
    def test_acp_to_completions_round_trip(self):
        """Message can flow: ACP -> UAMP -> Completions format."""
        acp_adapter = ACPUAMPAdapter()
        completions_adapter = CompletionsUAMPAdapter()
        
        # Convert ACP request to UAMP
        request = {
            "method": "prompt/submit",
            "params": {"messages": [{"role": "user", "content": "Test"}]}
        }
        uamp_events = acp_adapter.to_uamp(request)
        
        # Verify structure
        assert any(isinstance(e, InputTextEvent) for e in uamp_events)
        
        # Simulate response
        response_events = [
            ResponseDeltaEvent(
                response_id="resp_1",
                delta=ContentDelta(type="text", text="Response")
            ),
            ResponseDoneEvent(
                response_id="resp_1",
                response=ResponseOutput(
                    id="resp_1",
                    status="completed",
                    output=[],
                    usage=UsageStats(input_tokens=5, output_tokens=1, total_tokens=6)
                )
            )
        ]
        
        # Convert to completions format
        completions_response = completions_adapter.from_uamp(response_events)
        
        assert completions_response["choices"][0]["message"]["content"] == "Response"
        assert completions_response["usage"]["total_tokens"] == 6
