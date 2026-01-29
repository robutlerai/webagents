"""
Comprehensive tests for A2ATransportSkill

Tests Google Agent2Agent Protocol compatibility:
- Agent Card discovery
- Task lifecycle
- Message formats (TextPart, FilePart, DataPart)
- Artifacts
- Authentication
- SSE streaming
"""

import pytest
import json
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from webagents.agents.skills.core.transport.a2a.skill import (
    A2ATransportSkill, A2ATask, TaskState
)


# ============================================================================
# Mock Fixtures
# ============================================================================

class MockAgent:
    """Mock agent for testing"""
    def __init__(self):
        self.name = "test-agent"
        self.description = "A test agent"
        self.skills = {}
        self.api_key = None
        self.scopes = "all"
        self._registered_handoffs = []
        self.active_handoff = None
    
    def get_all_tools(self):
        return [
            {"function": {"name": "test_tool", "description": "A test tool"}}
        ]
    
    async def run_streaming(self, messages, **kwargs):
        yield {"choices": [{"delta": {"content": "Response"}}]}


class MockContext:
    """Mock context for testing"""
    def __init__(self, agent=None):
        self.agent = agent
        self.messages = []
        self.stream = True
        self.auth = None


@pytest.fixture
def skill():
    return A2ATransportSkill()


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_context(mock_agent):
    return MockContext(mock_agent)


# ============================================================================
# Agent Card Tests (/.well-known/agent.json)
# ============================================================================

class TestA2AAgentCard:
    """Test Agent Card discovery endpoint"""
    
    @pytest.mark.asyncio
    async def test_agent_card_basic_fields(self, skill, mock_agent, mock_context):
        """Test Agent Card contains required fields"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert card["name"] == "test-agent"
            assert card["description"] == "A test agent"
            assert "url" in card
            assert "version" in card
            assert "protocolVersion" in card
    
    @pytest.mark.asyncio
    async def test_agent_card_protocol_version(self, skill, mock_agent, mock_context):
        """Test Agent Card has correct protocol version"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert card["protocolVersion"] == "0.2.1"
    
    @pytest.mark.asyncio
    async def test_agent_card_capabilities(self, skill, mock_agent, mock_context):
        """Test Agent Card capabilities"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert card["capabilities"]["streaming"] is True
            assert card["capabilities"]["artifacts"] is True
            assert "stateTransitionHistory" in card["capabilities"]
    
    @pytest.mark.asyncio
    async def test_agent_card_provider(self, skill, mock_agent, mock_context):
        """Test Agent Card includes provider info"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert "provider" in card
            assert "organization" in card["provider"]
            assert "url" in card["provider"]
    
    @pytest.mark.asyncio
    async def test_agent_card_input_output_modes(self, skill, mock_agent, mock_context):
        """Test Agent Card lists input/output modes"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert "text" in card["defaultInputModes"]
            assert "file" in card["defaultInputModes"]
            assert "data" in card["defaultInputModes"]
            assert "text" in card["defaultOutputModes"]
    
    @pytest.mark.asyncio
    async def test_agent_card_skills_list(self, skill, mock_agent, mock_context):
        """Test Agent Card includes skills list"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert "skills" in card
            assert len(card["skills"]) > 0
            assert card["skills"][0]["name"] == "test_tool"


# ============================================================================
# Agent Card Authentication Tests
# ============================================================================

class TestA2AAgentCardAuthentication:
    """Test Agent Card authentication schemes"""
    
    @pytest.mark.asyncio
    async def test_auth_none_when_no_api_key(self, skill, mock_context):
        """Test 'none' auth when no API key configured"""
        agent = MockAgent()
        agent.api_key = None
        agent.scopes = "all"
        mock_context.agent = agent
        
        await skill.initialize(agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            assert "authentication" in card
            auth_types = [a["type"] for a in card["authentication"]]
            assert "none" in auth_types
    
    @pytest.mark.asyncio
    async def test_auth_bearer_when_api_key(self, skill, mock_context):
        """Test 'bearer' auth when API key configured"""
        agent = MockAgent()
        agent.api_key = "secret-key"
        mock_context.agent = agent
        
        await skill.initialize(agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            auth_types = [a["type"] for a in card["authentication"]]
            assert "bearer" in auth_types
    
    @pytest.mark.asyncio
    async def test_auth_oauth2_when_restricted_scopes(self, skill, mock_context):
        """Test 'oauth2' auth when scopes are restricted"""
        agent = MockAgent()
        agent.scopes = "owner"
        mock_context.agent = agent
        
        await skill.initialize(agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            auth_types = [a["type"] for a in card["authentication"]]
            assert "oauth2" in auth_types


# ============================================================================
# Task Lifecycle Tests
# ============================================================================

class TestA2ATaskLifecycle:
    """Test A2A task lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_create_task_returns_id(self, skill, mock_agent, mock_context):
        """Test task creation returns task ID"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "Response"}}]}
                mock_handoff.return_value = mock_stream()
                
                events = []
                async for event in skill.create_task(
                    message={"role": "user", "parts": [{"type": "text", "text": "Hello"}]}
                ):
                    events.append(event)
                
                # First event should be task.started
                started = json.loads(events[0].split("data: ")[1])
                assert "id" in started
    
    @pytest.mark.asyncio
    async def test_task_status_progression(self, skill, mock_agent, mock_context):
        """Test task status progresses: pending -> running -> completed"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "Done"}}]}
                mock_handoff.return_value = mock_stream()
                
                events = []
                async for event in skill.create_task(
                    message={"role": "user", "parts": [{"type": "text", "text": "Hello"}]}
                ):
                    events.append(event)
                
                # Parse events - SSE format is "event: type\ndata: {...}\n\n"
                event_types = []
                for e in events:
                    if e.startswith("event:"):
                        # Extract just the event type from the first line
                        first_line = e.split("\n")[0]
                        event_types.append(first_line.split("event: ")[1].strip())
                
                assert "task.started" in event_types
                assert "task.completed" in event_types
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, skill, mock_agent):
        """Test getting task status by ID"""
        await skill.initialize(mock_agent)
        
        # Create a task manually
        task = A2ATask()
        task.status = TaskState.COMPLETED
        task.result = {"content": "Test result"}
        skill._tasks[task.id] = task
        
        result = await skill.get_task(task.id)
        
        assert result["id"] == task.id
        assert result["status"] == "completed"
        assert result["result"]["content"] == "Test result"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, skill, mock_agent):
        """Test getting nonexistent task returns error"""
        await skill.initialize(mock_agent)
        
        result = await skill.get_task("nonexistent-id")
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_cancel_running_task(self, skill, mock_agent):
        """Test cancelling a running task"""
        await skill.initialize(mock_agent)
        
        task = A2ATask()
        task.status = TaskState.RUNNING
        skill._tasks[task.id] = task
        
        result = await skill.cancel_task(task.id)
        
        assert result["status"] == "cancelled"
        assert result["cancelled"] is True
    
    @pytest.mark.asyncio
    async def test_cancel_completed_task_unchanged(self, skill, mock_agent):
        """Test cancelling completed task doesn't change status"""
        await skill.initialize(mock_agent)
        
        task = A2ATask()
        task.status = TaskState.COMPLETED
        skill._tasks[task.id] = task
        
        result = await skill.cancel_task(task.id)
        
        assert result["status"] == "completed"
        assert result["cancelled"] is False


# ============================================================================
# Message Part Tests
# ============================================================================

class TestA2AMessageParts:
    """Test A2A message part handling"""
    
    @pytest.mark.asyncio
    async def test_text_part(self, skill, mock_agent):
        """Test TextPart conversion"""
        await skill.initialize(mock_agent)
        
        parts = [{"type": "text", "text": "Hello World"}]
        content = skill._parts_to_content(parts)
        
        # Simple text returns string
        assert content == "Hello World"
    
    @pytest.mark.asyncio
    async def test_multiple_text_parts(self, skill, mock_agent):
        """Test multiple TextParts are joined"""
        await skill.initialize(mock_agent)
        
        parts = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"}
        ]
        content = skill._parts_to_content(parts)
        
        assert "Line 1" in content
        assert "Line 2" in content
    
    @pytest.mark.asyncio
    async def test_file_part_image(self, skill, mock_agent):
        """Test FilePart with image converts to image_url"""
        await skill.initialize(mock_agent)
        
        parts = [
            {"type": "text", "text": "What is this?"},
            {
                "type": "file",
                "file": {
                    "name": "photo.jpg",
                    "mimeType": "image/jpeg",
                    "data": "base64encodeddata"
                }
            }
        ]
        content = skill._parts_to_content(parts)
        
        # Multimodal returns list
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in content[1]["image_url"]["url"]
    
    @pytest.mark.asyncio
    async def test_file_part_image_uri(self, skill, mock_agent):
        """Test FilePart with image URI"""
        await skill.initialize(mock_agent)
        
        parts = [{
            "type": "file",
            "file": {
                "name": "photo.jpg",
                "mimeType": "image/jpeg",
                "uri": "https://example.com/image.jpg"
            }
        }]
        content = skill._parts_to_content(parts)
        
        assert isinstance(content, list)
        assert content[0]["image_url"]["url"] == "https://example.com/image.jpg"
    
    @pytest.mark.asyncio
    async def test_file_part_non_image(self, skill, mock_agent):
        """Test FilePart with non-image file"""
        await skill.initialize(mock_agent)
        
        parts = [{
            "type": "file",
            "file": {
                "name": "document.pdf",
                "mimeType": "application/pdf"
            }
        }]
        content = skill._parts_to_content(parts)
        
        # Non-image files are converted to text description
        # Returns list format for multimodal compatibility
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            assert any("[File: document.pdf" in t for t in text_parts)
        else:
            assert "[File: document.pdf" in content
    
    @pytest.mark.asyncio
    async def test_data_part(self, skill, mock_agent):
        """Test DataPart with JSON data"""
        await skill.initialize(mock_agent)
        
        parts = [{
            "type": "data",
            "data": {"key": "value", "count": 42},
            "mimeType": "application/json"
        }]
        content = skill._parts_to_content(parts)
        
        assert "application/json" in content
        assert '"key"' in content or "'key'" in content


# ============================================================================
# A2A to OpenAI Conversion Tests
# ============================================================================

class TestA2AToOpenAIConversion:
    """Test A2A message format to OpenAI format conversion"""
    
    @pytest.mark.asyncio
    async def test_convert_user_message(self, skill, mock_agent):
        """Test converting user message"""
        await skill.initialize(mock_agent)
        
        message = {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello"}]
        }
        
        openai_msgs = skill._a2a_to_openai(message, None)
        
        assert len(openai_msgs) == 1
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_convert_agent_role_to_assistant(self, skill, mock_agent):
        """Test 'agent' role converts to 'assistant'"""
        await skill.initialize(mock_agent)
        
        message = {
            "role": "agent",
            "parts": [{"type": "text", "text": "Response"}]
        }
        
        openai_msgs = skill._a2a_to_openai(message, None)
        
        assert openai_msgs[0]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_convert_message_list(self, skill, mock_agent):
        """Test converting message list"""
        await skill.initialize(mock_agent)
        
        messages = [
            {"role": "user", "parts": [{"type": "text", "text": "Hi"}]},
            {"role": "agent", "parts": [{"type": "text", "text": "Hello!"}]},
            {"role": "user", "parts": [{"type": "text", "text": "How are you?"}]}
        ]
        
        openai_msgs = skill._a2a_to_openai(None, messages)
        
        assert len(openai_msgs) == 3
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[1]["role"] == "assistant"
        assert openai_msgs[2]["role"] == "user"


# ============================================================================
# OpenAI to A2A Conversion Tests
# ============================================================================

class TestOpenAIToA2AConversion:
    """Test OpenAI chunk to A2A message conversion"""
    
    @pytest.mark.asyncio
    async def test_convert_content_chunk(self, skill, mock_agent):
        """Test converting content chunk"""
        await skill.initialize(mock_agent)
        
        chunk = {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        
        a2a_msg = skill._openai_chunk_to_a2a(chunk)
        
        assert a2a_msg["role"] == "agent"
        assert a2a_msg["parts"][0]["type"] == "text"
        assert a2a_msg["parts"][0]["text"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_skip_empty_content(self, skill, mock_agent):
        """Test empty content chunks are skipped"""
        await skill.initialize(mock_agent)
        
        chunk = {
            "choices": [{"delta": {}}]
        }
        
        a2a_msg = skill._openai_chunk_to_a2a(chunk)
        
        assert a2a_msg is None
    
    @pytest.mark.asyncio
    async def test_skip_no_choices(self, skill, mock_agent):
        """Test chunks without choices are skipped"""
        await skill.initialize(mock_agent)
        
        chunk = {"choices": []}
        
        a2a_msg = skill._openai_chunk_to_a2a(chunk)
        
        assert a2a_msg is None


# ============================================================================
# Artifacts Tests
# ============================================================================

class TestA2AArtifacts:
    """Test A2A artifacts functionality"""
    
    @pytest.mark.asyncio
    async def test_get_artifacts_for_completed_task(self, skill, mock_agent):
        """Test getting artifacts for completed task"""
        await skill.initialize(mock_agent)
        
        task = A2ATask()
        task.status = TaskState.COMPLETED
        task.result = {"content": "Generated content here"}
        skill._tasks[task.id] = task
        
        result = await skill.get_task_artifacts(task.id)
        
        assert "artifacts" in result
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["data"] == "Generated content here"
        assert result["artifacts"][0]["type"] == "text"
    
    @pytest.mark.asyncio
    async def test_get_artifacts_nonexistent_task(self, skill, mock_agent):
        """Test getting artifacts for nonexistent task"""
        await skill.initialize(mock_agent)
        
        result = await skill.get_task_artifacts("nonexistent")
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_artifacts_empty_for_pending_task(self, skill, mock_agent):
        """Test artifacts empty for pending task"""
        await skill.initialize(mock_agent)
        
        task = A2ATask()
        task.status = TaskState.PENDING
        skill._tasks[task.id] = task
        
        result = await skill.get_task_artifacts(task.id)
        
        assert result["artifacts"] == []


# ============================================================================
# SSE Streaming Tests
# ============================================================================

class TestA2ASSEStreaming:
    """Test A2A SSE streaming format"""
    
    @pytest.mark.asyncio
    async def test_sse_event_format(self, skill, mock_agent):
        """Test SSE event format is correct"""
        await skill.initialize(mock_agent)
        
        event = skill._sse_event("task.started", {"id": "123"})
        
        assert event.startswith("event: task.started\n")
        assert "data: " in event
        assert event.endswith("\n\n")
    
    @pytest.mark.asyncio
    async def test_sse_stream_events(self, skill, mock_agent, mock_context):
        """Test SSE stream produces correct events"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff') as mock_handoff:
                async def mock_stream(*args, **kwargs):
                    yield {"choices": [{"delta": {"content": "Hi"}}]}
                mock_handoff.return_value = mock_stream()
                
                events = []
                async for event in skill.create_task(
                    message={"role": "user", "parts": [{"type": "text", "text": "Hello"}]}
                ):
                    events.append(event)
                
                # Should have started, message(s), completed events
                assert any("task.started" in e for e in events)
                assert any("task.completed" in e for e in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
