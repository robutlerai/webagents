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
from webagents.agents.skills.core.transport.a2a.uamp_adapter import A2AUAMPAdapter
from webagents.uamp import (
    InputTextEvent,
    InputImageEvent,
    ResponseDeltaEvent,
    ContentDelta,
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
    
    async def process_uamp(self, events, **kwargs):
        """Mock UAMP processing - yields UAMP server events"""
        from webagents.uamp import (
            ResponseCreatedEvent, ResponseDeltaEvent, ResponseDoneEvent,
            ContentDelta, ContentItem, ResponseOutput
        )
        response_id = "resp_test123"
        yield ResponseCreatedEvent(response_id=response_id)
        yield ResponseDeltaEvent(
            response_id=response_id,
            delta=ContentDelta(type="text", text="Done")
        )
        yield ResponseDoneEvent(
            response_id=response_id,
            response=ResponseOutput(
                id=response_id,
                status="completed",
                output=[ContentItem(type="text", text="Done")]
            )
        )


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
        """Test Agent Card lists input/output modes based on capabilities"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            # Text is always supported
            assert "text" in card["defaultInputModes"]
            assert "text" in card["defaultOutputModes"]
            # Input modes are dynamically determined by LLM capabilities
            assert isinstance(card["defaultInputModes"], list)
    
    @pytest.mark.asyncio
    async def test_agent_card_model_capabilities(self, skill, mock_agent, mock_context):
        """Test Agent Card includes model capabilities"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            card = await skill.agent_card()
            
            # Model capabilities should be present
            assert "modelCapabilities" in card
            assert "modalities" in card["modelCapabilities"]
            assert "text" in card["modelCapabilities"]["modalities"]
    
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
# Message Part Tests (via UAMP Adapter)
# ============================================================================

class TestA2AMessageParts:
    """Test A2A message part handling via UAMP adapter"""
    
    @pytest.fixture
    def adapter(self):
        return A2AUAMPAdapter()
    
    def test_text_part(self, adapter):
        """Test TextPart conversion to UAMP"""
        request = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello World"}]
            }
        }
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello World"
    
    def test_multiple_text_parts(self, adapter):
        """Test multiple TextParts are converted"""
        request = {
            "message": {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "Line 1"},
                    {"type": "text", "text": "Line 2"}
                ]
            }
        }
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 2
        assert text_events[0].text == "Line 1"
        assert text_events[1].text == "Line 2"
    
    def test_file_part_image(self, adapter):
        """Test FilePart with image converts to InputImageEvent"""
        request = {
            "message": {
                "role": "user",
                "parts": [
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
            }
        }
        events = adapter.to_uamp(request)
        
        image_events = [e for e in events if isinstance(e, InputImageEvent)]
        assert len(image_events) == 1
        assert "data:image/jpeg;base64,base64encodeddata" in image_events[0].image
    
    def test_file_part_image_uri(self, adapter):
        """Test FilePart with image URI"""
        request = {
            "message": {
                "role": "user",
                "parts": [{
                    "type": "file",
                    "file": {
                        "name": "photo.jpg",
                        "mimeType": "image/jpeg",
                        "uri": "https://example.com/image.jpg"
                    }
                }]
            }
        }
        events = adapter.to_uamp(request)
        
        image_events = [e for e in events if isinstance(e, InputImageEvent)]
        assert len(image_events) == 1
        assert image_events[0].image["url"] == "https://example.com/image.jpg"
    
    def test_file_part_non_image(self, adapter):
        """Test FilePart with non-image file creates InputFileEvent"""
        from webagents.uamp import InputFileEvent
        
        request = {
            "message": {
                "role": "user",
                "parts": [{
                    "type": "file",
                    "file": {
                        "name": "document.pdf",
                        "mimeType": "application/pdf",
                        "data": "pdfdata"
                    }
                }]
            }
        }
        events = adapter.to_uamp(request)
        
        file_events = [e for e in events if isinstance(e, InputFileEvent)]
        assert len(file_events) == 1
        assert file_events[0].filename == "document.pdf"
        assert file_events[0].mime_type == "application/pdf"
    
    def test_data_part(self, adapter):
        """Test DataPart with JSON data"""
        request = {
            "message": {
                "role": "user",
                "parts": [{
                    "type": "data",
                    "data": {"key": "value", "count": 42},
                    "mimeType": "application/json"
                }]
            }
        }
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert "application/json" in text_events[0].text
        assert '"key"' in text_events[0].text


# ============================================================================
# A2A to OpenAI Conversion Tests (via UAMP Adapter)
# ============================================================================

class TestA2AToOpenAIConversion:
    """Test A2A message format to UAMP conversion"""
    
    @pytest.fixture
    def adapter(self):
        return A2AUAMPAdapter()
    
    def test_convert_user_message(self, adapter):
        """Test converting user message to UAMP"""
        request = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello"}]
            }
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert text_events[0].role == "user"
        assert text_events[0].text == "Hello"
    
    def test_convert_agent_role_to_assistant(self, adapter):
        """Test 'agent' role converts to 'assistant' in UAMP"""
        request = {
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": "Response"}]
            }
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 1
        assert text_events[0].role == "assistant"
    
    def test_convert_message_list(self, adapter):
        """Test converting message list"""
        request = {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "Hi"}]},
                {"role": "agent", "parts": [{"type": "text", "text": "Hello!"}]},
                {"role": "user", "parts": [{"type": "text", "text": "How are you?"}]}
            ]
        }
        
        events = adapter.to_uamp(request)
        
        text_events = [e for e in events if isinstance(e, InputTextEvent)]
        assert len(text_events) == 3
        assert text_events[0].role == "user"
        assert text_events[1].role == "assistant"
        assert text_events[2].role == "user"


# ============================================================================
# OpenAI to A2A Conversion Tests (via UAMP)
# ============================================================================

class TestOpenAIToA2AConversion:
    """Test UAMP to A2A message conversion"""
    
    @pytest.fixture
    def adapter(self):
        return A2AUAMPAdapter()
    
    def test_convert_content_chunk(self, adapter):
        """Test converting UAMP ResponseDeltaEvent to A2A"""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="Hello")
        )
        
        a2a_event = adapter.from_uamp_streaming(event)
        
        assert a2a_event["event"] == "task.message"
        assert a2a_event["data"]["role"] == "agent"
        assert a2a_event["data"]["parts"][0]["type"] == "text"
        assert a2a_event["data"]["parts"][0]["text"] == "Hello"
    
    def test_skip_empty_content(self, adapter):
        """Test empty delta events return None for A2A"""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=ContentDelta(type="text", text="")
        )
        
        a2a_event = adapter.from_uamp_streaming(event)
        
        # Empty text should return None
        assert a2a_event is None
    
    def test_skip_none_delta(self, adapter):
        """Test events without delta are skipped"""
        event = ResponseDeltaEvent(
            response_id="resp_123",
            delta=None
        )
        
        a2a_event = adapter.from_uamp_streaming(event)
        
        assert a2a_event is None


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
