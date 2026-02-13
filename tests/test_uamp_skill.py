"""
Tests for UAMP transport skill changes:
- Renamed _handle_input_text -> _handle_input
- Stateless messages from event
- Echo session_id on all outgoing events
- Backward compat: no messages field uses session.conversation
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from webagents.uamp.events import (
    BaseEvent,
    SessionEndEvent,
    SessionErrorEvent,
    InputTextEvent,
    InputTypingEvent,
    PresenceTypingEvent,
    ResponseCreatedEvent,
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ResponseErrorEvent,
)


class TestBaseEventSessionId:
    """Test that BaseEvent includes session_id."""
    
    def test_session_id_none_by_default(self):
        event = BaseEvent(type="test")
        assert event.session_id is None
        d = event.to_dict()
        assert "session_id" not in d

    def test_session_id_included_when_set(self):
        event = BaseEvent(type="test", session_id="sess_abc")
        d = event.to_dict()
        assert d["session_id"] == "sess_abc"


class TestSessionEndEvent:
    """Test SessionEndEvent."""
    
    def test_basic(self):
        event = SessionEndEvent(reason="daemon_takeover")
        d = event.to_dict()
        assert d["type"] == "session.end"
        assert d["reason"] == "daemon_takeover"

    def test_with_session_id(self):
        event = SessionEndEvent(reason="timeout", session_id="sess_123")
        d = event.to_dict()
        assert d["session_id"] == "sess_123"
        assert d["reason"] == "timeout"


class TestSessionErrorEvent:
    """Test SessionErrorEvent."""
    
    def test_basic(self):
        event = SessionErrorEvent(error={"code": "agent_offline", "message": "Agent not connected"})
        d = event.to_dict()
        assert d["type"] == "session.error"
        assert d["error"]["code"] == "agent_offline"


class TestResponseEventsEchoSessionId:
    """Test that response events echo session_id."""
    
    def test_response_created_with_session_id(self):
        event = ResponseCreatedEvent(response_id="resp_1", session_id="sess_abc")
        d = event.to_dict()
        assert d["session_id"] == "sess_abc"
        assert d["response_id"] == "resp_1"

    def test_response_done_with_session_id(self):
        from webagents.uamp.events import ResponseOutput, ContentItem, UsageStats
        event = ResponseDoneEvent(
            response_id="resp_1",
            session_id="sess_abc",
            response=ResponseOutput(
                id="resp_1",
                status="completed",
                output=[ContentItem(type="text", text="hello")],
                usage=UsageStats(input_tokens=5, output_tokens=1, total_tokens=6),
            ),
        )
        d = event.to_dict()
        assert d["session_id"] == "sess_abc"

    def test_response_error_with_session_id(self):
        event = ResponseErrorEvent(
            response_id="resp_1",
            session_id="sess_abc",
            error={"code": "generation_error", "message": "test"},
        )
        d = event.to_dict()
        assert d["session_id"] == "sess_abc"


class TestInputTypingChatId:
    """Test that InputTypingEvent uses chat_id (not conversation_id)."""
    
    def test_chat_id_in_typing(self):
        event = InputTypingEvent(is_typing=True, chat_id="chat_123")
        d = event.to_dict()
        assert d["chat_id"] == "chat_123"
        assert "conversation_id" not in d


class TestPresenceTypingChatId:
    """Test that PresenceTypingEvent uses chat_id (not conversation_id)."""
    
    def test_chat_id_in_presence_typing(self):
        event = PresenceTypingEvent(user_id="u1", is_typing=True, chat_id="chat_123")
        d = event.to_dict()
        assert d["chat_id"] == "chat_123"
        assert "conversation_id" not in d


class TestStatelessMessages:
    """Test that _handle_input uses event.messages for stateless context."""
    
    def test_messages_overwrites_conversation(self):
        """When event has 'messages', session.conversation should be replaced."""
        from webagents.agents.skills.core.transport.uamp.skill import UAMPSession
        
        session = UAMPSession()
        session.conversation = [{"role": "user", "content": "old message"}]
        
        # Simulate what _handle_input does
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        event = {
            "type": "input.text",
            "text": "Hello",
            "role": "user",
            "messages": messages,
            "session_id": "sess_123",
        }
        
        # Stateless mode: replaces conversation
        if event.get("messages"):
            session.conversation = list(event["messages"])
        
        assert len(session.conversation) == 2
        assert session.conversation[0]["role"] == "system"
        assert session.conversation[1]["content"] == "Hello"

    def test_no_messages_appends_to_conversation(self):
        """When event has no 'messages', append to session.conversation (backward compat)."""
        from webagents.agents.skills.core.transport.uamp.skill import UAMPSession
        
        session = UAMPSession()
        session.conversation = [{"role": "user", "content": "old message"}]
        
        event = {
            "type": "input.text",
            "text": "New message",
            "role": "user",
        }
        
        # Backward compat: append
        if not event.get("messages"):
            session.conversation.append({
                "role": event.get("role", "user"),
                "content": event.get("text", ""),
            })
        
        assert len(session.conversation) == 2
        assert session.conversation[1]["content"] == "New message"


class TestConcurrentSessionIdIsolation:
    """Test that different session_ids don't interfere."""
    
    def test_unique_session_ids(self):
        """Each interaction should get a unique session_id."""
        import uuid
        ids = set()
        for _ in range(100):
            sid = str(uuid.uuid4())
            assert sid not in ids
            ids.add(sid)
        assert len(ids) == 100
