"""
Server Test Fixtures - WebAgents V2.0

Shared pytest fixtures for server testing including:
- Mock agents and skills
- Test clients and servers
- OpenAI compliance validators
- Request/response helpers
"""

import json
import pytest
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.server.core.app import WebAgentsServer
from webagents.server.models import ChatCompletionRequest, OpenAIResponse


class MockLLMSkill:
    """Mock LLM skill for testing without external APIs"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = config.get('model', 'test-gpt-4o-mini')
        self.agent = None
    
    async def initialize(self, agent: BaseAgent) -> None:
        """Initialize mock skill"""
        self.agent = agent
        return True
    
    async def chat_completion(self, messages: List[Dict], stream: bool = False, tools: Optional[List] = None) -> Dict[str, Any]:
        """Mock chat completion - returns dict format"""
        if stream:
            # This shouldn't be called for streaming - use chat_completion_stream instead
            raise ValueError("Use chat_completion_stream for streaming")
        
        return self._create_mock_response(messages)
    
    async def chat_completion_stream(self, messages: List[Dict], tools: Optional[List] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Mock streaming chat completion"""
        completion_id = f"chatcmpl-test-{hash(str(messages))}"
        created = 1699999999
        
        # First chunk with role
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        }
        
        # Content chunks
        content_parts = ["Test ", "streaming ", "response ", "chunk ", "by ", "chunk."]
        for i, part in enumerate(content_parts):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": part},
                    "finish_reason": None
                }]
            }
            # Small delay to simulate real streaming
            await asyncio.sleep(0.01)
        
        # Final chunk with finish_reason and usage
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }
    
    def _create_mock_response(self, messages: List[Dict]) -> Dict[str, Any]:
        """Create OpenAI-compatible response"""
        last_message = messages[-1] if messages else {"content": "empty"}
        content = last_message.get("content", "empty")
        
        return {
            "id": f"chatcmpl-test-{hash(str(messages))}",
            "object": "chat.completion",
            "created": 1699999999,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response to: {content}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }


@pytest.fixture
def mock_llm_skill():
    """Create a mock LLM skill"""
    return MockLLMSkill({"model": "test-gpt-4o-mini"})


@pytest.fixture 
def test_agent(mock_llm_skill):
    """Create a test agent with mock LLM skill"""
    agent = BaseAgent(
        name="test-agent",
        instructions="Test agent for server testing",
        scope="all"
    )
    
    # Set up mock skill
    agent.skills = {"primary_llm": mock_llm_skill}
    
    # Mock the agent's run methods to use our mock skill
    async def mock_run(messages, stream=False, tools=None):
        return await mock_llm_skill.chat_completion(messages, stream=stream, tools=tools)
    
    async def mock_run_streaming(messages, tools=None):
        async for chunk in mock_llm_skill.chat_completion_stream(messages, tools=tools):
            yield chunk
    
    agent.run = mock_run
    agent.run_streaming = mock_run_streaming
    
    return agent


@pytest.fixture
def multi_agent_setup(mock_llm_skill):
    """Create multiple test agents for multi-agent testing"""
    agents = []
    
    for i, (name, instructions) in enumerate([
        ("assistant", "General purpose AI assistant"),
        ("calculator", "Mathematical calculations and computations"),  
        ("weather", "Weather information and forecasts")
    ]):
        agent = BaseAgent(
            name=name,
            instructions=instructions,
            scope="all"
        )
        
        # Each agent gets its own mock skill instance
        skill = MockLLMSkill({"model": f"test-{name}-model"})
        agent.skills = {"primary_llm": skill}
        
        # Mock methods
        async def mock_run(messages, stream=False, tools=None, skill_ref=skill):
            return await skill_ref.chat_completion(messages, stream=stream, tools=tools)
        
        async def mock_run_streaming(messages, tools=None, skill_ref=skill):
            async for chunk in skill_ref.chat_completion_stream(messages, tools=tools):
                yield chunk
        
        agent.run = mock_run
        agent.run_streaming = mock_run_streaming
        agents.append(agent)
    
    return agents


@pytest.fixture
def test_server(test_agent):
    """Create a test server with single agent"""
    return WebAgentsServer(agents=[test_agent])


@pytest.fixture
def multi_agent_server(multi_agent_setup):
    """Create a test server with multiple agents"""
    return WebAgentsServer(agents=multi_agent_setup)


@pytest.fixture  
def test_client(test_server):
    """Create FastAPI test client"""
    return TestClient(test_server.app)


@pytest.fixture
def multi_client(multi_agent_server):
    """Create FastAPI test client with multiple agents"""
    return TestClient(multi_agent_server.app)


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with something?"}
    ]


@pytest.fixture
def sample_request_data():
    """Sample request data for testing"""
    return {
        "messages": [{"role": "user", "content": "Test message"}],
        "stream": False,
        "model": "test-agent"
    }


@pytest.fixture
def streaming_request_data():
    """Sample streaming request data"""
    return {
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 100
    }


# OpenAI Compliance Validation Helpers
OPENAI_COMPLIANCE_CONFIG = {
    'required_response_fields': [
        'id', 'object', 'created', 'model', 'choices', 'usage'
    ],
    'required_choice_fields': [
        'index', 'message', 'finish_reason'
    ],
    'required_message_fields': [
        'role', 'content'
    ],
    'required_usage_fields': [
        'prompt_tokens', 'completion_tokens', 'total_tokens'
    ],
    'required_streaming_fields': [
        'id', 'object', 'created', 'model', 'choices'
    ],
    'required_streaming_choice_fields': [
        'index', 'delta'
    ]
}


@pytest.fixture
def openai_validator():
    """OpenAI compliance validation helper"""
    
    def validate_response(response: Dict[str, Any], is_streaming: bool = False):
        """Validate OpenAI compliance for responses"""
        
        if is_streaming:
            required_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_fields']
            required_choice_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_choice_fields']
            expected_object = 'chat.completion.chunk'
        else:
            required_fields = OPENAI_COMPLIANCE_CONFIG['required_response_fields']
            required_choice_fields = OPENAI_COMPLIANCE_CONFIG['required_choice_fields']
            expected_object = 'chat.completion'
        
        # Check top-level fields
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # Check object type
        assert response['object'] == expected_object, f"Invalid object type: {response['object']}"
        
        # Check choices
        assert isinstance(response['choices'], list), "choices must be a list"
        assert len(response['choices']) > 0, "choices must not be empty"
        
        choice = response['choices'][0]
        for field in required_choice_fields:
            assert field in choice, f"Missing required choice field: {field}"
        
        # Additional validation for non-streaming
        if not is_streaming:
            message = choice['message']
            for field in OPENAI_COMPLIANCE_CONFIG['required_message_fields']:
                assert field in message, f"Missing required message field: {field}"
            
            usage = response['usage']
            for field in OPENAI_COMPLIANCE_CONFIG['required_usage_fields']:
                assert field in usage, f"Missing required usage field: {field}"
        
        return True
    
    return validate_response


@pytest.fixture  
def streaming_helper():
    """Helper for testing streaming responses"""
    
    def parse_sse_chunks(response_text: str) -> List[Dict[str, Any]]:
        """Parse SSE formatted streaming response"""
        chunks = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
        
        return chunks
    
    def collect_streaming_content(chunks: List[Dict[str, Any]]) -> str:
        """Collect content from streaming chunks"""
        content = ""
        for chunk in chunks:
            if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                content += chunk['choices'][0]['delta']['content']
        return content
    
    def validate_streaming_sequence(chunks: List[Dict[str, Any]]) -> bool:
        """Validate streaming chunk sequence is correct"""
        if not chunks:
            return False
        
        # First chunk should have role
        first_chunk = chunks[0]
        if first_chunk['choices'][0]['delta'].get('role') != 'assistant':
            return False
        
        # Last chunk should have finish_reason
        last_chunk = chunks[-1]
        if last_chunk['choices'][0].get('finish_reason') != 'stop':
            return False
        
        # All chunks should have same id and model
        first_id = first_chunk['id']
        first_model = first_chunk['model']
        
        for chunk in chunks:
            if chunk['id'] != first_id or chunk['model'] != first_model:
                return False
        
        return True
    
    return {
        'parse_sse_chunks': parse_sse_chunks,
        'collect_content': collect_streaming_content,
        'validate_sequence': validate_streaming_sequence
    }


@pytest.fixture
def request_helper():
    """Helper for making HTTP requests with proper headers"""
    
    def make_request_with_context(client: TestClient, method: str, url: str, **kwargs):
        """Make request with proper context headers"""
        headers = kwargs.get('headers', {})
        headers.update({
            'x-user-id': 'test-user',
            'x-payment-user-id': 'test-payment-user',
            'x-user-scope': 'all'
        })
        kwargs['headers'] = headers
        
        if method.upper() == 'GET':
            return client.get(url, **kwargs)
        elif method.upper() == 'POST':
            return client.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    return make_request_with_context 