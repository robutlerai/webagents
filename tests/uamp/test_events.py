"""
Tests for UAMP Event definitions.
"""

import pytest
import json

from webagents.uamp import (
    BaseEvent,
    SessionCreateEvent,
    SessionCreatedEvent,
    InputTextEvent,
    InputAudioEvent,
    InputImageEvent,
    ResponseCreateEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ToolCallEvent,
    ToolResultEvent,
    PingEvent,
    PongEvent,
    SessionConfig,
    Session,
    ContentDelta,
    ContentItem,
    UsageStats,
    ResponseOutput,
)
from webagents.uamp.events import generate_event_id


class TestEventIdGeneration:
    """Tests for event ID generation."""
    
    def test_event_id_format(self):
        """Event IDs should have proper format."""
        event_id = generate_event_id()
        assert event_id.startswith("evt_")
        assert len(event_id) == 16  # evt_ + 12 hex chars
    
    def test_event_ids_unique(self):
        """Each event ID should be unique."""
        ids = [generate_event_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestSessionEvents:
    """Tests for session events."""
    
    def test_session_create_event(self):
        """SessionCreateEvent should serialize correctly."""
        config = SessionConfig(
            modalities=["text", "audio"],
            instructions="You are a helpful assistant",
        )
        event = SessionCreateEvent(
            uamp_version="1.0",
            session=config
        )
        
        d = event.to_dict()
        assert d["type"] == "session.create"
        assert d["uamp_version"] == "1.0"
        assert d["session"]["modalities"] == ["text", "audio"]
        assert d["session"]["instructions"] == "You are a helpful assistant"
        assert "event_id" in d
    
    def test_session_created_event(self):
        """SessionCreatedEvent should serialize correctly."""
        session = Session(
            id="sess_123",
            created_at=1700000000000,
            status="active"
        )
        event = SessionCreatedEvent(
            uamp_version="1.0",
            session=session
        )
        
        d = event.to_dict()
        assert d["type"] == "session.created"
        assert d["uamp_version"] == "1.0"
        assert d["session"]["id"] == "sess_123"
        assert d["session"]["status"] == "active"


class TestInputEvents:
    """Tests for input events."""
    
    def test_input_text_event(self):
        """InputTextEvent should serialize correctly."""
        event = InputTextEvent(
            text="Hello, world!",
            role="user"
        )
        
        d = event.to_dict()
        assert d["type"] == "input.text"
        assert d["text"] == "Hello, world!"
        assert d["role"] == "user"
    
    def test_input_text_system_role(self):
        """InputTextEvent should support system role."""
        event = InputTextEvent(
            text="You are a helpful assistant",
            role="system"
        )
        
        d = event.to_dict()
        assert d["role"] == "system"
    
    def test_input_audio_event(self):
        """InputAudioEvent should serialize correctly."""
        event = InputAudioEvent(
            audio="base64encodedaudio==",
            format="pcm16",
            is_final=True
        )
        
        d = event.to_dict()
        assert d["type"] == "input.audio"
        assert d["audio"] == "base64encodedaudio=="
        assert d["format"] == "pcm16"
        assert d["is_final"] is True
    
    def test_input_image_event_base64(self):
        """InputImageEvent should support base64."""
        event = InputImageEvent(
            image="base64data==",
            format="png",
            detail="high"
        )
        
        d = event.to_dict()
        assert d["type"] == "input.image"
        assert d["image"] == "base64data=="
        assert d["format"] == "png"
        assert d["detail"] == "high"
    
    def test_input_image_event_url(self):
        """InputImageEvent should support URL."""
        event = InputImageEvent(
            image={"url": "https://example.com/image.png"}
        )
        
        d = event.to_dict()
        assert d["image"]["url"] == "https://example.com/image.png"


class TestResponseEvents:
    """Tests for response events."""
    
    def test_response_create_event(self):
        """ResponseCreateEvent should serialize correctly."""
        event = ResponseCreateEvent()
        
        d = event.to_dict()
        assert d["type"] == "response.create"
    
    def test_response_created_event(self):
        """ResponseCreatedEvent should serialize correctly."""
        event = ResponseCreatedEvent(response_id="resp_123")
        
        d = event.to_dict()
        assert d["type"] == "response.created"
        assert d["response_id"] == "resp_123"
    
    def test_response_delta_text(self):
        """ResponseDeltaEvent should serialize text delta."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="Hello")
        )
        
        d = event.to_dict()
        assert d["type"] == "response.delta"
        assert d["response_id"] == "resp_123"
        assert d["delta"]["type"] == "text"
        assert d["delta"]["text"] == "Hello"
    
    def test_response_delta_tool_call(self):
        """ResponseDeltaEvent should serialize tool call delta."""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(
                type="tool_call",
                tool_call={
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location":'
                }
            )
        )
        
        d = event.to_dict()
        assert d["delta"]["type"] == "tool_call"
        assert d["delta"]["tool_call"]["name"] == "get_weather"
    
    def test_response_done_event(self):
        """ResponseDoneEvent should serialize correctly."""
        event = ResponseDoneEvent(
            response_id="resp_123",
            response=ResponseOutput(
                id="resp_123",
                status="completed",
                output=[ContentItem(type="text", text="Hello!")],
                usage=UsageStats(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15
                )
            )
        )
        
        d = event.to_dict()
        assert d["type"] == "response.done"
        assert d["response"]["status"] == "completed"
        assert d["response"]["output"][0]["text"] == "Hello!"
        assert d["response"]["usage"]["total_tokens"] == 15


class TestToolEvents:
    """Tests for tool events."""
    
    def test_tool_call_event(self):
        """ToolCallEvent should serialize correctly."""
        event = ToolCallEvent(
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "NYC"}'
        )
        
        d = event.to_dict()
        assert d["type"] == "tool.call"
        assert d["call_id"] == "call_123"
        assert d["name"] == "get_weather"
        assert d["arguments"] == '{"location": "NYC"}'
    
    def test_tool_result_event(self):
        """ToolResultEvent should serialize correctly."""
        event = ToolResultEvent(
            call_id="call_123",
            result='{"temperature": 72}',
            is_error=False
        )
        
        d = event.to_dict()
        assert d["type"] == "tool.result"
        assert d["call_id"] == "call_123"
        assert d["result"] == '{"temperature": 72}'
        assert "is_error" not in d  # Only included when True
    
    def test_tool_result_error(self):
        """ToolResultEvent should mark errors."""
        event = ToolResultEvent(
            call_id="call_123",
            result='{"error": "Not found"}',
            is_error=True
        )
        
        d = event.to_dict()
        assert d["is_error"] is True


class TestUtilityEvents:
    """Tests for utility events."""
    
    def test_ping_event(self):
        """PingEvent should serialize correctly."""
        event = PingEvent()
        
        d = event.to_dict()
        assert d["type"] == "ping"
        assert "event_id" in d
    
    def test_pong_event(self):
        """PongEvent should serialize correctly."""
        event = PongEvent()
        
        d = event.to_dict()
        assert d["type"] == "pong"


class TestEventTimestamps:
    """Tests for event timestamps."""
    
    def test_auto_timestamp(self):
        """Events should have auto-generated timestamps."""
        event = InputTextEvent(text="Hello")
        assert event.timestamp is not None
        assert event.timestamp > 0
    
    def test_timestamp_in_dict(self):
        """Timestamp should be in serialized dict."""
        event = InputTextEvent(text="Hello")
        d = event.to_dict()
        assert "timestamp" in d
        assert isinstance(d["timestamp"], int)
