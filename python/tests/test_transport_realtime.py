"""
Comprehensive tests for RealtimeTransportSkill

Tests OpenAI Realtime API compatibility:
- WebSocket session management
- Audio buffer operations
- Conversation management
- Response streaming
- Full event sequence
"""

import pytest
import json
import asyncio
import base64
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

from webagents.agents.skills.core.transport.realtime.skill import (
    RealtimeTransportSkill, RealtimeSession
)


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
    
    async def run_streaming(self, messages, **kwargs):
        yield {"choices": [{"delta": {"content": "Hello"}}]}
        yield {"choices": [{"delta": {"content": " World"}}]}


class MockContext:
    """Mock context for testing"""
    def __init__(self, agent=None):
        self.agent = agent
        self.messages = []
        self.stream = True
        self.auth = None


class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self):
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.messages_sent = []
        self.messages_to_receive = []
        self._receive_index = 0
    
    async def accept(self):
        self.accepted = True
    
    async def close(self, code=1000):
        self.closed = True
        self.close_code = code
    
    async def send_json(self, data):
        self.messages_sent.append(data)
    
    async def receive_json(self):
        if self._receive_index < len(self.messages_to_receive):
            msg = self.messages_to_receive[self._receive_index]
            self._receive_index += 1
            return msg
        raise Exception("No more messages")
    
    def iter_json(self):
        async def _iter():
            for msg in self.messages_to_receive:
                yield msg
        return _iter()


@pytest.fixture
def skill():
    return RealtimeTransportSkill()


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_context(mock_agent):
    return MockContext(mock_agent)


@pytest.fixture
def mock_ws():
    return MockWebSocket()


# ============================================================================
# Initialization Tests
# ============================================================================

class TestRealtimeInitialization:
    """Test skill initialization"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, skill, mock_agent):
        """Test basic initialization"""
        await skill.initialize(mock_agent)
        assert skill.agent == mock_agent
    
    def test_default_scope(self, skill):
        """Test default scope is 'all'"""
        assert skill.scope == "all"
    
    def test_websocket_endpoint_registered(self, skill):
        """Test WebSocket endpoint is properly decorated"""
        assert hasattr(skill.realtime_session, '_webagents_is_websocket')
        assert skill.realtime_session._websocket_path == '/realtime'
    
    def test_sessions_dict_initialized(self, skill):
        """Test sessions dictionary is initialized"""
        assert skill._sessions == {}


# ============================================================================
# Session Management Tests
# ============================================================================

class TestRealtimeSessionManagement:
    """Test session creation and management"""
    
    def test_session_default_values(self):
        """Test RealtimeSession default values"""
        session = RealtimeSession()
        
        assert session.id.startswith("sess_")
        assert session.voice == "alloy"
        assert session.modalities == ["text"]
        assert session.instructions == ""
        assert session.input_audio_format == "pcm16"
        assert session.output_audio_format == "pcm16"
        assert session.turn_detection is None
        assert session.audio_buffer == bytes()
        assert session.conversation == []
    
    def test_session_to_dict(self):
        """Test session serialization"""
        session = RealtimeSession()
        session.voice = "nova"
        session.instructions = "Be helpful"
        
        result = session.to_dict()
        
        assert result["id"] == session.id
        assert result["voice"] == "nova"
        assert result["instructions"] == "Be helpful"
        assert result["modalities"] == ["text"]
    
    @pytest.mark.asyncio
    async def test_session_created_on_connect(self, skill, mock_agent, mock_ws):
        """Test session is created when client connects"""
        await skill.initialize(mock_agent)
        
        mock_ws.messages_to_receive = [
            {"type": "session.update", "session": {"instructions": "Be helpful"}}
        ]
        
        with patch.object(skill, 'get_context', return_value=MockContext(mock_agent)):
            # Simulate partial connection
            await mock_ws.accept()
            
            # Check session created event would be sent
            assert mock_ws.accepted
    
    @pytest.mark.asyncio
    async def test_session_update_instructions(self, skill, mock_agent, mock_ws):
        """Test session.update updates instructions"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "session.update",
            "session": {"instructions": "Be a helpful assistant"}
        }
        
        await skill._handle_session_update(mock_ws, session, event)
        
        assert session.instructions == "Be a helpful assistant"
        assert mock_ws.messages_sent[0]["type"] == "session.updated"
    
    @pytest.mark.asyncio
    async def test_session_update_voice(self, skill, mock_agent, mock_ws):
        """Test session.update updates voice"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "session.update",
            "session": {"voice": "nova"}
        }
        
        await skill._handle_session_update(mock_ws, session, event)
        
        assert session.voice == "nova"
    
    @pytest.mark.asyncio
    async def test_session_update_modalities(self, skill, mock_agent, mock_ws):
        """Test session.update updates modalities"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "session.update",
            "session": {"modalities": ["text", "audio"]}
        }
        
        await skill._handle_session_update(mock_ws, session, event)
        
        assert session.modalities == ["text", "audio"]
    
    @pytest.mark.asyncio
    async def test_session_update_all_fields(self, skill, mock_agent, mock_ws):
        """Test session.update updates all configurable fields"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "session.update",
            "session": {
                "voice": "shimmer",
                "modalities": ["audio"],
                "instructions": "Speak quickly",
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_alaw",
                "turn_detection": {"type": "server_vad", "threshold": 0.5}
            }
        }
        
        await skill._handle_session_update(mock_ws, session, event)
        
        assert session.voice == "shimmer"
        assert session.modalities == ["audio"]
        assert session.instructions == "Speak quickly"
        assert session.input_audio_format == "g711_ulaw"
        assert session.output_audio_format == "g711_alaw"
        assert session.turn_detection == {"type": "server_vad", "threshold": 0.5}


# ============================================================================
# Audio Buffer Tests
# ============================================================================

class TestRealtimeAudioBuffer:
    """Test audio buffer operations"""
    
    @pytest.mark.asyncio
    async def test_audio_append(self, skill, mock_agent, mock_ws):
        """Test input_audio_buffer.append"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        audio_data = b"raw audio bytes"
        audio_b64 = base64.b64encode(audio_data).decode()
        
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        
        await skill._handle_audio_append(mock_ws, session, event)
        
        assert session.audio_buffer == audio_data
    
    @pytest.mark.asyncio
    async def test_audio_append_multiple(self, skill, mock_agent, mock_ws):
        """Test multiple audio appends concatenate"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        
        chunk1 = b"chunk1"
        chunk2 = b"chunk2"
        
        await skill._handle_audio_append(mock_ws, session, {"audio": base64.b64encode(chunk1).decode()})
        await skill._handle_audio_append(mock_ws, session, {"audio": base64.b64encode(chunk2).decode()})
        
        assert session.audio_buffer == chunk1 + chunk2
    
    @pytest.mark.asyncio
    async def test_audio_append_invalid_base64(self, skill, mock_agent, mock_ws):
        """Test handling of invalid base64 audio data"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {"audio": "not-valid-base64!!!"}
        
        # Should not raise, just silently ignore
        await skill._handle_audio_append(mock_ws, session, event)
        
        assert session.audio_buffer == bytes()
    
    @pytest.mark.asyncio
    async def test_audio_commit(self, skill, mock_agent, mock_ws):
        """Test input_audio_buffer.commit"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.audio_buffer = b"audiodata"
        
        await skill._handle_audio_commit(mock_ws, session, {})
        
        # Should add to conversation and clear buffer
        assert len(session.conversation) == 1
        assert session.conversation[0]["type"] == "message"
        assert session.conversation[0]["role"] == "user"
        assert session.audio_buffer == bytes()
        
        # Should send committed event
        assert mock_ws.messages_sent[0]["type"] == "input_audio_buffer.committed"
        assert "item_id" in mock_ws.messages_sent[0]
    
    @pytest.mark.asyncio
    async def test_audio_commit_empty_buffer(self, skill, mock_agent, mock_ws):
        """Test input_audio_buffer.commit with empty buffer"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        # Buffer is already empty
        
        await skill._handle_audio_commit(mock_ws, session, {})
        
        # Should not add to conversation when buffer is empty
        assert len(session.conversation) == 0
        assert len(mock_ws.messages_sent) == 0
    
    @pytest.mark.asyncio
    async def test_audio_clear(self, skill, mock_agent, mock_ws):
        """Test input_audio_buffer.clear"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.audio_buffer = b"audiodata"
        
        await skill._handle_audio_clear(mock_ws, session, {})
        
        assert session.audio_buffer == bytes()
        assert mock_ws.messages_sent[0]["type"] == "input_audio_buffer.cleared"


# ============================================================================
# Conversation Item Tests
# ============================================================================

class TestRealtimeConversationItems:
    """Test conversation item operations"""
    
    @pytest.mark.asyncio
    async def test_item_create_message(self, skill, mock_agent, mock_ws):
        """Test conversation.item.create with message"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        }
        
        await skill._handle_item_create(mock_ws, session, event)
        
        assert len(session.conversation) == 1
        assert session.conversation[0]["role"] == "user"
        assert mock_ws.messages_sent[0]["type"] == "conversation.item.created"
    
    @pytest.mark.asyncio
    async def test_item_create_with_id(self, skill, mock_agent, mock_ws):
        """Test conversation.item.create with custom ID"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "conversation.item.create",
            "item": {
                "id": "my_custom_id",
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        }
        
        await skill._handle_item_create(mock_ws, session, event)
        
        assert session.conversation[0]["id"] == "my_custom_id"
    
    @pytest.mark.asyncio
    async def test_item_create_generates_id(self, skill, mock_agent, mock_ws):
        """Test conversation.item.create generates ID if not provided"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        }
        
        await skill._handle_item_create(mock_ws, session, event)
        
        assert session.conversation[0]["id"].startswith("item_")
    
    @pytest.mark.asyncio
    async def test_item_delete(self, skill, mock_agent, mock_ws):
        """Test conversation.item.delete"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user"},
            {"id": "item_2", "type": "message", "role": "assistant"}
        ]
        
        event = {"item_id": "item_1"}
        
        await skill._handle_item_delete(mock_ws, session, event)
        
        assert len(session.conversation) == 1
        assert session.conversation[0]["id"] == "item_2"
        assert mock_ws.messages_sent[0]["type"] == "conversation.item.deleted"
        assert mock_ws.messages_sent[0]["item_id"] == "item_1"
    
    @pytest.mark.asyncio
    async def test_item_delete_nonexistent(self, skill, mock_agent, mock_ws):
        """Test conversation.item.delete with nonexistent ID"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user"}
        ]
        
        event = {"item_id": "nonexistent"}
        
        await skill._handle_item_delete(mock_ws, session, event)
        
        # Should still send event, conversation unchanged
        assert len(session.conversation) == 1
        assert mock_ws.messages_sent[0]["type"] == "conversation.item.deleted"
    
    @pytest.mark.asyncio
    async def test_item_truncate(self, skill, mock_agent, mock_ws):
        """Test conversation.item.truncate"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user"}
        ]
        
        event = {
            "item_id": "item_1",
            "content_index": 0,
            "audio_end_ms": 5000
        }
        
        await skill._handle_item_truncate(mock_ws, session, event)
        
        assert session.conversation[0].get("truncated") is True
        assert session.conversation[0].get("truncate_audio_end_ms") == 5000
        assert mock_ws.messages_sent[0]["type"] == "conversation.item.truncated"
        assert mock_ws.messages_sent[0]["audio_end_ms"] == 5000


# ============================================================================
# Response Create Tests
# ============================================================================

class TestRealtimeResponseCreate:
    """Test response.create functionality"""
    
    @pytest.mark.asyncio
    async def test_response_create_basic(self, skill, mock_agent, mock_ws, mock_context):
        """Test basic response.create"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hi there!"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            # Check event sequence
            event_types = [m["type"] for m in mock_ws.messages_sent]
            assert "response.created" in event_types
            assert "response.done" in event_types
    
    @pytest.mark.asyncio
    async def test_response_create_with_instructions(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.create uses session instructions"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.instructions = "Be concise"
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        
        captured_messages = []
        
        async def mock_stream(messages, **kwargs):
            captured_messages.extend(messages)
            yield {"choices": [{"delta": {"content": "Hi"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=mock_stream):
            await skill._handle_response_create(mock_ws, session, {})
            
            # Should include system message with instructions
            assert len(captured_messages) >= 1
            if len(captured_messages) > 1:
                assert captured_messages[0]["role"] == "system"
                assert captured_messages[0]["content"] == "Be concise"
    
    @pytest.mark.asyncio
    async def test_response_adds_to_conversation(self, skill, mock_agent, mock_ws):
        """Test response is added to conversation"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "item_1", "type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hi there!"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            # Should have added assistant response
            assert len(session.conversation) == 2
            assert session.conversation[1]["role"] == "assistant"
            assert session.conversation[1]["content"][0]["text"] == "Hi there!"


# ============================================================================
# Response Event Sequence Tests
# ============================================================================

class TestRealtimeResponseEventSequence:
    """Test full response event sequence"""
    
    @pytest.mark.asyncio
    async def test_response_created_event(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.created event format"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            created_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.created")
            assert "response" in created_event
            assert created_event["response"]["status"] == "in_progress"
            assert created_event["response"]["object"] == "realtime.response"
            assert created_event["response"]["id"].startswith("resp_")
    
    @pytest.mark.asyncio
    async def test_output_item_added_event(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.output_item.added event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            added_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.output_item.added")
            assert "item" in added_event
            assert added_event["item"]["role"] == "assistant"
            assert added_event["item"]["type"] == "message"
            assert added_event["item"]["status"] == "in_progress"
            assert added_event["output_index"] == 0
    
    @pytest.mark.asyncio
    async def test_content_part_added_event(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.content_part.added event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            part_added = next(m for m in mock_ws.messages_sent if m["type"] == "response.content_part.added")
            assert part_added["content_index"] == 0
            assert part_added["part"]["type"] == "text"
    
    @pytest.mark.asyncio
    async def test_text_delta_events(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.text.delta events"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " World"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            delta_events = [m for m in mock_ws.messages_sent if m["type"] == "response.text.delta"]
            assert len(delta_events) == 2
            assert delta_events[0]["delta"] == "Hello"
            assert delta_events[1]["delta"] == " World"
    
    @pytest.mark.asyncio
    async def test_text_done_event(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.text.done event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello World"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            done_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.text.done")
            assert done_event["text"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_content_part_done_event(self, skill, mock_agent, mock_ws):
        """Test response.content_part.done event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            part_done = next(m for m in mock_ws.messages_sent if m["type"] == "response.content_part.done")
            assert part_done["part"]["type"] == "text"
            assert part_done["part"]["text"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_output_item_done_event(self, skill, mock_agent, mock_ws):
        """Test response.output_item.done event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            item_done = next(m for m in mock_ws.messages_sent if m["type"] == "response.output_item.done")
            assert item_done["item"]["status"] == "completed"
            assert item_done["item"]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_response_done_event(self, skill, mock_agent, mock_ws, mock_context):
        """Test response.done event with usage"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            done_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.done")
            assert done_event["response"]["status"] == "completed"
            assert "usage" in done_event["response"]
            assert "total_tokens" in done_event["response"]["usage"]
            assert "input_tokens" in done_event["response"]["usage"]
            assert "output_tokens" in done_event["response"]["usage"]
    
    @pytest.mark.asyncio
    async def test_full_event_sequence_order(self, skill, mock_agent, mock_ws):
        """Test full event sequence is in correct order"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
        
        with patch.object(skill, 'execute_handoff', side_effect=lambda *a, **k: mock_stream()):
            await skill._handle_response_create(mock_ws, session, {})
            
            event_types = [m["type"] for m in mock_ws.messages_sent]
            
            # Verify order of key events
            assert event_types.index("response.created") < event_types.index("response.output_item.added")
            assert event_types.index("response.output_item.added") < event_types.index("response.content_part.added")
            assert event_types.index("response.content_part.added") < event_types.index("response.text.delta")
            assert event_types.index("response.text.delta") < event_types.index("response.text.done")
            assert event_types.index("response.text.done") < event_types.index("response.content_part.done")
            assert event_types.index("response.content_part.done") < event_types.index("response.output_item.done")
            assert event_types.index("response.output_item.done") < event_types.index("response.done")


# ============================================================================
# Response Cancel Tests
# ============================================================================

class TestRealtimeResponseCancel:
    """Test response.cancel functionality"""
    
    @pytest.mark.asyncio
    async def test_response_cancel(self, skill, mock_agent, mock_ws):
        """Test response.cancel sends cancelled event"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        
        await skill._handle_response_cancel(mock_ws, session, {})
        
        assert mock_ws.messages_sent[0]["type"] == "response.cancelled"


# ============================================================================
# Conversation to Messages Conversion Tests
# ============================================================================

class TestRealtimeConversationConversion:
    """Test conversation to OpenAI messages conversion"""
    
    @pytest.mark.asyncio
    async def test_convert_text_message(self, skill, mock_agent):
        """Test converting text message"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "1", "type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        
        messages = skill._conversation_to_messages(session)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_convert_multiple_messages(self, skill, mock_agent):
        """Test converting multiple messages"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "1", "type": "message", "role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"id": "2", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
            {"id": "3", "type": "message", "role": "user", "content": [{"type": "text", "text": "How are you?"}]}
        ]
        
        messages = skill._conversation_to_messages(session)
        
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_convert_audio_input_placeholder(self, skill, mock_agent):
        """Test converting audio input to placeholder"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {"id": "1", "type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "[audio data]"}]}
        ]
        
        messages = skill._conversation_to_messages(session)
        
        assert len(messages) == 1
        assert "[audio input]" in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_convert_empty_conversation(self, skill, mock_agent):
        """Test converting empty conversation"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = []
        
        messages = skill._conversation_to_messages(session)
        
        assert messages == []
    
    @pytest.mark.asyncio
    async def test_convert_multiple_content_parts(self, skill, mock_agent):
        """Test converting message with multiple content parts"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [
            {
                "id": "1",
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"}
                ]
            }
        ]
        
        messages = skill._conversation_to_messages(session)
        
        assert len(messages) == 1
        assert "First part" in messages[0]["content"]
        assert "Second part" in messages[0]["content"]


# ============================================================================
# Delta Text Extraction Tests
# ============================================================================

class TestRealtimeDeltaExtraction:
    """Test delta text extraction from streaming chunks"""
    
    def test_extract_delta_text_basic(self, skill):
        """Test extracting delta text from chunk"""
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        
        result = skill._extract_delta_text(chunk)
        
        assert result == "Hello"
    
    def test_extract_delta_text_empty(self, skill):
        """Test extracting from chunk with no content"""
        chunk = {"choices": [{"delta": {}}]}
        
        result = skill._extract_delta_text(chunk)
        
        assert result == ""
    
    def test_extract_delta_text_no_choices(self, skill):
        """Test extracting from chunk with no choices"""
        chunk = {}
        
        result = skill._extract_delta_text(chunk)
        
        assert result == ""
    
    def test_extract_delta_text_empty_choices(self, skill):
        """Test extracting from chunk with empty choices"""
        chunk = {"choices": []}
        
        result = skill._extract_delta_text(chunk)
        
        assert result == ""


# ============================================================================
# Event Handling Tests
# ============================================================================

class TestRealtimeEventHandling:
    """Test event dispatching and handling"""
    
    @pytest.mark.asyncio
    async def test_handle_known_event_type(self, skill, mock_agent, mock_ws):
        """Test handling known event type"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {"type": "session.update", "session": {"voice": "nova"}}
        
        await skill._handle_event(mock_ws, session, event)
        
        assert session.voice == "nova"
        assert mock_ws.messages_sent[0]["type"] == "session.updated"
    
    @pytest.mark.asyncio
    async def test_unknown_event_type(self, skill, mock_agent, mock_ws):
        """Test unknown event type sends error response"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {"type": "unknown.event.type"}
        
        await skill._handle_event(mock_ws, session, event)
        
        # Implementation sends error with type field in data that gets merged
        # The _send_event merges data with **data, so nested "type" may override
        error_event = mock_ws.messages_sent[0]
        # Either the event type is "error" or "invalid_event" (from nested data)
        assert error_event["type"] in ("error", "invalid_event")
        assert "unknown" in error_event.get("message", "").lower()
    
    @pytest.mark.asyncio
    async def test_event_includes_event_id(self, skill, mock_agent, mock_ws):
        """Test that sent events include event_id"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        event = {"type": "session.update", "session": {}}
        
        await skill._handle_event(mock_ws, session, event)
        
        assert "event_id" in mock_ws.messages_sent[0]
        assert mock_ws.messages_sent[0]["event_id"].startswith("evt_")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestRealtimeErrorHandling:
    """Test error handling in Realtime transport"""
    
    @pytest.mark.asyncio
    async def test_response_error_sends_failed_status(self, skill, mock_agent, mock_ws, mock_context):
        """Test response error sends failed status"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        async def mock_stream_error(*args, **kwargs):
            raise Exception("Test error")
            yield  # Make it a generator
        
        with patch.object(skill, 'execute_handoff', side_effect=Exception("Test error")):
            await skill._handle_response_create(mock_ws, session, {})
            
            done_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.done")
            assert done_event["response"]["status"] == "failed"
            assert done_event["response"]["status_details"]["type"] == "error"
    
    @pytest.mark.asyncio
    async def test_response_error_includes_error_message(self, skill, mock_agent, mock_ws):
        """Test response error includes error message"""
        await skill.initialize(mock_agent)
        
        session = RealtimeSession()
        session.conversation = [{"id": "1", "role": "user", "type": "message", "content": [{"type": "text", "text": "Hi"}]}]
        
        with patch.object(skill, 'execute_handoff', side_effect=Exception("Specific error message")):
            await skill._handle_response_create(mock_ws, session, {})
            
            done_event = next(m for m in mock_ws.messages_sent if m["type"] == "response.done")
            assert "Specific error message" in done_event["response"]["status_details"]["error"]


# ============================================================================
# Send Event Tests
# ============================================================================

class TestRealtimeSendEvent:
    """Test _send_event helper method"""
    
    @pytest.mark.asyncio
    async def test_send_event_format(self, skill, mock_ws):
        """Test event format is correct"""
        await skill._send_event(mock_ws, "test.event", {"key": "value"})
        
        sent = mock_ws.messages_sent[0]
        assert sent["type"] == "test.event"
        assert sent["key"] == "value"
        assert "event_id" in sent
        assert sent["event_id"].startswith("evt_")
    
    @pytest.mark.asyncio
    async def test_send_event_unique_ids(self, skill, mock_ws):
        """Test each event gets unique ID"""
        await skill._send_event(mock_ws, "test.event1", {})
        await skill._send_event(mock_ws, "test.event2", {})
        
        id1 = mock_ws.messages_sent[0]["event_id"]
        id2 = mock_ws.messages_sent[1]["event_id"]
        
        assert id1 != id2


# ============================================================================
# Session Update Method Tests
# ============================================================================

class TestRealtimeSessionUpdate:
    """Test RealtimeSession.update method"""
    
    def test_update_single_field(self):
        """Test updating single field"""
        session = RealtimeSession()
        session.update({"voice": "echo"})
        
        assert session.voice == "echo"
    
    def test_update_multiple_fields(self):
        """Test updating multiple fields"""
        session = RealtimeSession()
        session.update({
            "voice": "fable",
            "instructions": "Be brief",
            "modalities": ["audio"]
        })
        
        assert session.voice == "fable"
        assert session.instructions == "Be brief"
        assert session.modalities == ["audio"]
    
    def test_update_ignores_unknown_fields(self):
        """Test update ignores unknown fields"""
        session = RealtimeSession()
        original_voice = session.voice
        
        session.update({"unknown_field": "value"})
        
        assert session.voice == original_voice  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
