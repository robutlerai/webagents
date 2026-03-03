"""
OpenAI Skill Integration Tests - WebAgents V2.0

Integration tests for OpenAI skill with real/mock API calls.
Tests OpenAI compliance for streaming and non-streaming responses.

Currently tests mock implementation - can be updated for real OpenAI API calls.

Run with: pytest tests/test_integration_openai.py -m integration -v
"""

import pytest
import asyncio
import os
import time
import json
from typing import Dict, Any, List, AsyncGenerator
from unittest.mock import patch

# Integration test configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'mock-api-key-for-testing')
USE_REAL_OPENAI = os.getenv('USE_REAL_OPENAI', 'false').lower() == 'true'
TEST_MODEL = 'gpt-4o-mini'

# OpenAI Compliance Configuration
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

def is_integration_tests_enabled() -> bool:
    """Check if integration tests should run"""
    return os.getenv('RUN_INTEGRATION_TESTS', 'true').lower() == 'true'

def get_test_message() -> Dict[str, Any]:
    """Get a standard test message for integration tests"""
    return {"role": "user", "content": "Hello! Please respond with 'OpenAI integration test successful' and nothing else."}

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.openai import OpenAISkill

if OpenAISkill is None:
    pytest.skip("OpenAISkill not available (openai SDK may be missing)", allow_module_level=True)

# Pytest markers for test organization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not USE_REAL_OPENAI,
        reason='Requires USE_REAL_OPENAI=true to run real OpenAI integration tests'
    ),
]


@pytest.fixture
def openai_skill():
    """Create OpenAISkill configured for integration testing"""
    api_key = os.environ.get('OPENAI_API_KEY', OPENAI_API_KEY)
    config = {
        'api_key': api_key,
        'model': TEST_MODEL
    }
    skill = OpenAISkill(config)
    return skill


def validate_openai_response(response: Dict[str, Any], is_streaming: bool = False) -> None:
    """Validate OpenAI compliance for API responses"""
    
    if is_streaming:
        required_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_fields']
        required_choice_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_choice_fields']
    else:
        required_fields = OPENAI_COMPLIANCE_CONFIG['required_response_fields']
        required_choice_fields = OPENAI_COMPLIANCE_CONFIG['required_choice_fields']
    
    # Check top-level response structure
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"
    
    # Check response object type
    if is_streaming:
        assert response['object'] == 'chat.completion.chunk', f"Invalid object type: {response['object']}"
    else:
        assert response['object'] == 'chat.completion', f"Invalid object type: {response['object']}"
    
    # Check choices structure
    assert isinstance(response['choices'], list), "choices must be a list"
    assert len(response['choices']) > 0, "choices must not be empty"
    
    choice = response['choices'][0]
    for field in required_choice_fields:
        assert field in choice, f"Missing required choice field: {field}"
    
    # Check message/delta structure for non-streaming
    if not is_streaming and choice.get('message') is not None:
        message = choice['message']
        for field in OPENAI_COMPLIANCE_CONFIG['required_message_fields']:
            assert field in message, f"Missing required message field: {field}"
        
        # Check usage fields for final response
        if 'usage' in response and response['usage'] is not None:
            usage = response['usage']
            for field in OPENAI_COMPLIANCE_CONFIG['required_usage_fields']:
                assert field in usage, f"Missing required usage field: {field}"
            
            assert isinstance(usage['prompt_tokens'], int), "prompt_tokens must be integer"
            assert isinstance(usage['completion_tokens'], int), "completion_tokens must be integer"  
            assert isinstance(usage['total_tokens'], int), "total_tokens must be integer"
            assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']


@pytest.mark.asyncio
async def test_openai_skill_initialization(openai_skill):
    """Test OpenAI skill initialization and configuration"""
    
    agent = BaseAgent(
        name="openai-init-test-agent",
        instructions="Test agent for OpenAI skill initialization.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Check skill initialization
    assert "primary_llm" in agent.skills
    skill = agent.skills["primary_llm"]
    
    assert skill.model == TEST_MODEL
    assert skill.api_key == OPENAI_API_KEY
    
    print(f"✅ OpenAI skill initialized with model: {skill.model}")


@pytest.mark.asyncio
async def test_nonstreaming_openai_compliance(openai_skill):
    """Test non-streaming response OpenAI compliance"""
    
    agent = BaseAgent(
        name="openai-compliance-test-agent",
        instructions="You are a helpful assistant for testing OpenAI compliance.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test with simple message
    messages = [get_test_message()]
    
    # Make non-streaming request
    start_time = time.time()
    response = await openai_skill.chat_completion(
        messages=messages,
        stream=False
    )
    duration = time.time() - start_time
    
    print(f"Non-streaming response time: {duration:.2f}s")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Validate OpenAI compliance
    validate_openai_response(response, is_streaming=False)
    
    # Additional checks
    assert 'content' in response['choices'][0]['message']
    content = response['choices'][0]['message']['content']
    assert len(content) > 0, "Response content should not be empty"
    
    # Check model name (API may return versioned name like gpt-4o-mini-2024-07-18)
    assert response['model'].startswith(TEST_MODEL), \
        f"Model '{response['model']}' should start with '{TEST_MODEL}'"
    
    print(f"✅ Non-streaming OpenAI compliance validated")


@pytest.mark.asyncio 
async def test_streaming_openai_compliance(openai_skill):
    """Test streaming response OpenAI compliance"""
    
    agent = BaseAgent(
        name="openai-streaming-test-agent", 
        instructions="You are a helpful assistant for testing streaming.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    messages = [get_test_message()]
    
    # Make streaming request
    start_time = time.time()
    chunks = []
    full_content = ""
    
    async for chunk in openai_skill.chat_completion_stream(
        messages=messages
    ):
        chunks.append(chunk)
        
        # Validate each chunk for OpenAI compliance
        validate_openai_response(chunk, is_streaming=True)
        
        # Extract content if available
        if (chunk.get('choices', []) and 
            chunk['choices'][0].get('delta', {}).get('content')):
            content = chunk['choices'][0]['delta']['content']
            full_content += content
    
    duration = time.time() - start_time
    
    print(f"Streaming response time: {duration:.2f}s")
    print(f"Total chunks: {len(chunks)}")
    print(f"Full content: {full_content}")
    
    assert len(chunks) > 0, "Should receive at least one chunk"
    assert len(full_content) > 0, "Should receive some content"
    
    # Check that final chunk has finish_reason
    if chunks:
        final_chunk = chunks[-1]
        if final_chunk.get('choices'):
            finish_reason = final_chunk['choices'][0].get('finish_reason')
            assert finish_reason is not None, "Final chunk should have finish_reason"
    
    # Check model consistency across chunks (API may return versioned name)
    for chunk in chunks:
        assert chunk['model'].startswith(TEST_MODEL), \
            f"Model '{chunk['model']}' should start with '{TEST_MODEL}'"
    
    print(f"✅ Streaming OpenAI compliance validated")


@pytest.mark.asyncio
async def test_openai_tools_functionality(openai_skill):
    """Test OpenAI skill registers correctly with BaseAgent"""
    
    agent = BaseAgent(
        name="openai-tools-test-agent",
        instructions="Test agent for OpenAI skill registration.",
        skills={"primary_llm": openai_skill}
    )
    
    # Ensure skills are initialized (registers handoff)
    await agent._ensure_skills_initialized()
    
    # OpenAI skill registers as a handoff (completion handler), not individual tools
    assert "primary_llm" in agent.skills
    skill = agent.skills["primary_llm"]
    assert skill.model == TEST_MODEL
    
    # Verify the skill has the required completion methods
    assert hasattr(skill, 'chat_completion')
    assert hasattr(skill, 'chat_completion_stream')
    assert callable(skill.chat_completion)
    assert callable(skill.chat_completion_stream)
    
    # Verify the handoff was registered
    assert len(agent._registered_handoffs) > 0, \
        "OpenAI skill should register as a handoff"
    
    print(f"✅ OpenAI skill registration validated")


@pytest.mark.asyncio
async def test_baseagent_integration_nonstreaming(openai_skill):
    """Test full BaseAgent integration with OpenAI skill (non-streaming)"""
    
    agent = BaseAgent(
        name="openai-agent-integration-test",
        instructions="You are a test assistant. Respond concisely and accurately.",
        skills={"primary_llm": openai_skill}
    )
    
    # Ensure skills are initialized (registers handoff, sets agent ref)
    await agent._ensure_skills_initialized()
    
    messages = [
        {"role": "user", "content": "What is 3+3? Answer with just the number."}
    ]
    
    # Test through BaseAgent.run()
    response = await agent.run(messages, stream=False)
    
    assert response is not None, "agent.run() should return a response dict"
    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    
    # Response should already be in correct format from BaseAgent
    print(f"Response type: {type(response)}")
    print(f"Response keys: {response.keys()}")
    
    # Validate OpenAI compliance at agent level
    validate_openai_response(response, is_streaming=False)
    
    # Check response content
    content = response['choices'][0]['message']['content']
    assert len(content) > 0
    
    print(f"Agent non-streaming response: {content}")
    print(f"✅ BaseAgent non-streaming integration successful")


@pytest.mark.asyncio
async def test_baseagent_integration_streaming(openai_skill):
    """Test full BaseAgent integration with OpenAI skill (streaming)"""
    
    agent = BaseAgent(
        name="openai-agent-streaming-test",
        instructions="You are a helpful assistant.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5, each number on a new line."}
    ]
    
    # Test through BaseAgent.run_streaming()
    chunks = []
    full_content = ""
    
    async for chunk in agent.run_streaming(messages):
        chunks.append(chunk)
        
        # Validate each chunk
        validate_openai_response(chunk, is_streaming=True)
        
        # Collect content
        if (chunk.get('choices', []) and
            chunk['choices'][0].get('delta', {}).get('content')):
            content = chunk['choices'][0]['delta']['content']
            full_content += content
    
    assert len(chunks) > 0
    assert len(full_content) > 0
    
    print(f"Agent streaming chunks: {len(chunks)}")
    print(f"Agent streaming content: {full_content}")
    print(f"✅ BaseAgent streaming integration successful")


@pytest.mark.asyncio
async def test_openai_error_handling(openai_skill):
    """Test OpenAI skill error handling and edge cases"""
    
    agent = BaseAgent(
        name="openai-error-test-agent",
        instructions="Test agent for error handling.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test with empty messages (should handle gracefully)
    try:
        response = await openai_skill.chat_completion(
            messages=[],
            stream=False  
        )
        
        # Should still return a valid response structure
        validate_openai_response(response, is_streaming=False)
        print("✅ Empty messages handled gracefully")
        
    except Exception as e:
        # Error handling should be graceful
        assert isinstance(e, Exception)
        print(f"✅ Empty messages error handled: {type(e).__name__}")
    
    # Test with very long message (should handle gracefully)
    long_message = {"role": "user", "content": "A" * 1000}
    response = await openai_skill.chat_completion(
        messages=[long_message],
        stream=False
    )
    
    validate_openai_response(response, is_streaming=False)
    print("✅ Long message handled gracefully")


@pytest.mark.asyncio
async def test_openai_performance_benchmarks(openai_skill):
    """Test OpenAI skill performance benchmarks"""
    
    agent = BaseAgent(
        name="openai-perf-test-agent",
        instructions="You are a helpful assistant. Respond concisely.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    test_message = {"role": "user", "content": "Hello"}
    
    # Test non-streaming performance
    start_time = time.time()
    response = await openai_skill.chat_completion(
        messages=[test_message],
        stream=False
    )
    nonstreaming_time = time.time() - start_time
    
    validate_openai_response(response, is_streaming=False)
    
    # Test streaming performance (first chunk)
    start_time = time.time()
    first_chunk_time = None
    total_chunks = 0
    
    async for chunk in openai_skill.chat_completion_stream(
        messages=[test_message]
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time
        total_chunks += 1
    
    total_streaming_time = time.time() - start_time
    
    # Performance assertions (reasonable thresholds for mock implementation)
    assert nonstreaming_time < 5, f"Non-streaming too slow: {nonstreaming_time:.2f}s"
    assert first_chunk_time < 2, f"First chunk too slow: {first_chunk_time:.2f}s"
    assert total_streaming_time < 10, f"Total streaming too slow: {total_streaming_time:.2f}s"
    assert total_chunks > 0, "Should receive at least one chunk"
    
    print(f"📊 Performance Results:")
    print(f"  Non-streaming: {nonstreaming_time:.2f}s")
    print(f"  First chunk: {first_chunk_time:.2f}s")  
    print(f"  Total streaming: {total_streaming_time:.2f}s")
    print(f"  Total chunks: {total_chunks}")
    print(f"✅ Performance benchmarks completed")


@pytest.mark.asyncio 
async def test_openai_concurrent_requests(openai_skill):
    """Test concurrent request handling with OpenAI skill"""
    
    agent = BaseAgent(
        name="openai-concurrent-test-agent",
        instructions="You are a helpful assistant.",
        skills={"primary_llm": openai_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Create multiple concurrent requests
    async def make_request(request_id: int):
        message = {"role": "user", "content": f"Request {request_id}: What is {request_id} * 2?"}
        response = await openai_skill.chat_completion(
            messages=[message],
            stream=False
        )
        validate_openai_response(response, is_streaming=False)
        return response
    
    # Run 3 concurrent requests
    start_time = time.time()
    tasks = [make_request(i) for i in range(1, 4)]
    responses = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    assert len(responses) == 3
    for response in responses:
        validate_openai_response(response, is_streaming=False)
    
    print(f"📊 Concurrent requests completed in {total_time:.2f}s")
    print(f"✅ Concurrent request handling successful")


@pytest.mark.asyncio
async def test_openai_hook_integration(openai_skill):
    """Test OpenAI skill handoff integration with BaseAgent"""
    
    agent = BaseAgent(
        name="openai-hook-test-agent",
        instructions="Test agent for handoff registration.",
        skills={"primary_llm": openai_skill}
    )
    
    # Ensure skills are initialized so handoff is registered
    await agent._ensure_skills_initialized()
    
    # OpenAI skill registers as a handoff (completion handler), not hooks
    assert len(agent._registered_handoffs) > 0, \
        "OpenAI skill should register a handoff"
    
    # Verify the handoff has a completion function
    handoff_entry = agent._registered_handoffs[0]
    handoff_config = handoff_entry.get('config', handoff_entry)
    assert handoff_config is not None, "Handoff should have a config"
    
    # Verify skill is properly linked to the agent
    skill = agent.skills["primary_llm"]
    assert skill.agent == agent, "Skill should be initialized with agent reference"
    
    print(f"Total handoffs: {len(agent._registered_handoffs)}")
    print(f"✅ OpenAI skill handoff integration validated")


if __name__ == "__main__":
    pytest.main([__file__, "-m", "integration", "-v"]) 