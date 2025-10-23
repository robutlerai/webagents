"""
Server Models Tests - WebAgents V2.0

Test Pydantic models for request/response validation:
- ChatCompletionRequest validation
- OpenAI response models
- Error handling for invalid data

Run with: pytest tests/server/test_models.py -v  
"""

import pytest
from pydantic import ValidationError
from webagents.server.models import (
    ChatCompletionRequest, ChatMessage, OpenAIResponse, 
    AgentInfoResponse, ServerInfo, HealthResponse
)


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model validation"""
    
    def test_valid_basic_request(self):
        """Test basic valid request"""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        request = ChatCompletionRequest(**data)
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello"
        assert request.stream is False
    
    def test_valid_streaming_request(self):
        """Test valid streaming request"""
        data = {
            "messages": [{"role": "user", "content": "Stream this"}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 100
        }
        request = ChatCompletionRequest(**data)
        assert request.stream is True
        assert request.temperature == 0.7
        assert request.max_tokens == 100
    
    def test_request_with_tools(self):
        """Test request with external tools"""
        data = {
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info"
                    }
                }
            ]
        }
        request = ChatCompletionRequest(**data)
        assert len(request.tools) == 1
        assert request.tools[0]["type"] == "function"
    
    def test_empty_messages_allowed(self):
        """Test that empty messages list is actually allowed"""
        data = {"messages": []}
        # This should work - empty messages appear to be allowed
        request = ChatCompletionRequest(**data)
        assert len(request.messages) == 0
    
    def test_invalid_missing_messages(self):
        """Test that missing messages fails validation"""
        data = {"stream": True}
        with pytest.raises(ValidationError):
            ChatCompletionRequest(**data)


class TestChatMessage:
    """Test ChatMessage model validation"""
    
    def test_valid_user_message(self):
        """Test valid user message"""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_valid_assistant_message(self):
        """Test valid assistant message"""
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_message_with_tool_calls(self):
        """Test message with tool calls"""
        msg = ChatMessage(
            role="assistant",
            content="I'll check the weather",
            tool_calls=[{"id": "call_123", "type": "function"}]
        )
        assert len(msg.tool_calls) == 1


class TestResponseModels:
    """Test response model validation"""
    
    def test_agent_info_response(self):
        """Test AgentInfoResponse model"""
        data = {
            "name": "test-agent",
            "description": "Test agent",
            "capabilities": ["chat", "tools"],
            "skills": ["openai"],
            "tools": ["get_weather"],
            "model": "gpt-4o-mini"
        }
        response = AgentInfoResponse(**data)
        assert response.name == "test-agent"
        assert len(response.capabilities) == 2
    
    def test_server_info_response(self):
        """Test ServerInfo model"""
        data = {
            "agents": ["agent1", "agent2"],
            "endpoints": {
                "chat": "/{agent}/chat/completions",
                "health": "/health"
            }
        }
        response = ServerInfo(**data)
        assert len(response.agents) == 2
        assert "chat" in response.endpoints
    
    def test_health_response(self):
        """Test HealthResponse model"""
        from datetime import datetime
        data = {
            "status": "healthy",
            "timestamp": datetime.utcnow()
        }
        response = HealthResponse(**data)
        assert response.status == "healthy"
        assert response.version == "2.0.0" 