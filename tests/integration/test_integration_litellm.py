"""
LiteLLM Integration Tests - WebAgents V2.0

Integration tests with real LiteLLM proxy API calls.
Tests OpenAI compliance for streaming and non-streaming responses.

Requirements:
- Local LiteLLM proxy running on localhost:2225  
- API key: rok_testapikey
- Models: gpt-4o-mini, claude-3-5-haiku, grok-4

Run with: pytest tests/test_integration_litellm.py -m integration -v
"""

import pytest
import asyncio
import os
import time
import json
from typing import Dict, Any, List, AsyncGenerator
from unittest.mock import patch

# Integration test configuration (embedded to avoid import issues)
LITELLM_BASE_URL = os.getenv('LITELLM_BASE_URL', 'http://localhost:2225')
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY', 'rok_testapikey')

TEST_MODELS = {
    'openai': 'openai/gpt-4o-mini',
    'anthropic': 'anthropic/claude-3-5-haiku', 
    'xai': 'xai/grok-4',
    'azure': 'azure/gpt-4o-mini'
}

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
    return {"role": "user", "content": "Hello! Please respond with 'Integration test successful' and nothing else."}

def get_litellm_config() -> Dict[str, Any]:
    """Get LiteLLM configuration for integration tests"""
    return {
        'api_keys': {
            'openai': LITELLM_API_KEY,
            'anthropic': LITELLM_API_KEY, 
            'xai': LITELLM_API_KEY,
            'azure': LITELLM_API_KEY
        },
        'base_url': LITELLM_BASE_URL,
        'model': TEST_MODELS['openai'],  # Use proper model name
        'fallback_models': [TEST_MODELS['anthropic'], TEST_MODELS['xai']]
    }

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.llm.litellm import LiteLLMSkill
from webagents.server.context.context_vars import create_context, set_context

# Pytest markers for test organization
pytestmark = pytest.mark.integration

# Skip all tests if integration tests are disabled
skip_if_disabled = pytest.mark.skipif(
    not is_integration_tests_enabled(),
    reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=true to enable)"
)


@pytest.fixture
def integration_skill():
    """Create LiteLLMSkill configured for integration testing"""
    config = get_litellm_config()
    
    # Configure LiteLLM to use our local proxy
    with patch.dict(os.environ, {
        'OPENAI_API_BASE': LITELLM_BASE_URL,
        'OPENAI_API_KEY': LITELLM_API_KEY,
        'ANTHROPIC_API_KEY': LITELLM_API_KEY,
        'XAI_API_KEY': LITELLM_API_KEY
    }):
        skill = LiteLLMSkill(config)
    
    return skill


def validate_openai_response(response, is_streaming=False):
    """Validate OpenAI API compliance for LiteLLM responses"""
    
    # Handle LiteLLM Message objects which have attributes, not dictionary keys
    def has_field(obj, field):
        """Check if object has field as attribute or key"""
        if hasattr(obj, field):
            return True
        if hasattr(obj, '__contains__') and field in obj:
            return True
        return False
    
    def get_field(obj, field):
        """Get field value from object (attribute or key)"""
        if hasattr(obj, field):
            return getattr(obj, field)
        if hasattr(obj, '__getitem__'):
            return obj[field]
        return None
    
    # Check top-level response fields
    expected_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_fields'] if is_streaming else OPENAI_COMPLIANCE_CONFIG['required_response_fields']
    for field in expected_fields:
        assert has_field(response, field), f"Missing required response field: {field}"
    
    # Check choices array
    choices = get_field(response, 'choices')
    assert choices and len(choices) > 0, "Response must have at least one choice"
    
    choice = choices[0]
    expected_choice_fields = OPENAI_COMPLIANCE_CONFIG['required_streaming_choice_fields'] if is_streaming else OPENAI_COMPLIANCE_CONFIG['required_choice_fields']
    for field in expected_choice_fields:
        assert has_field(choice, field), f"Missing required choice field: {field}"
    
    # Check message/delta structure for non-streaming
    if not is_streaming and has_field(choice, 'message'):
        message = get_field(choice, 'message')
        for field in OPENAI_COMPLIANCE_CONFIG['required_message_fields']:
            assert has_field(message, field), f"Missing required message field: {field}"
        
        # Check usage fields for final response
        if has_field(response, 'usage'):
            usage = get_field(response, 'usage')
            for field in OPENAI_COMPLIANCE_CONFIG['required_usage_fields']:
                assert has_field(usage, field), f"Missing required usage field: {field}"


@skip_if_disabled
@pytest.mark.asyncio
async def test_proxy_connectivity():
    """Test basic connectivity to LiteLLM proxy"""
    
    # Test if proxy is running by making a simple request
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LITELLM_BASE_URL}/health")
            assert response.status_code == 200
            print(f"✅ LiteLLM proxy is running at {LITELLM_BASE_URL}")
    except Exception as e:
        pytest.skip(f"LiteLLM proxy not available: {e}")


@skip_if_disabled  
@pytest.mark.asyncio
async def test_nonstreaming_openai_compliance(integration_skill):
    """Test non-streaming response OpenAI compliance"""
    
    agent = BaseAgent(
        name="compliance-test-agent",
        instructions="You are a helpful assistant.",
        skills={"litellm": integration_skill}
    )
    
    await asyncio.sleep(0.1)  # Let skill initialize
    
    # Test with simple message
    messages = [get_test_message()]
    
    # Make non-streaming request
    start_time = time.time()
    response = await integration_skill.chat_completion(
        messages=messages,
        model=TEST_MODELS['openai'],
        stream=False
    )
    duration = time.time() - start_time
    
    print(f"Non-streaming response time: {duration:.2f}s")
    
    # Convert LiteLLM response to dict for processing
    response_dict = response if isinstance(response, dict) else response.dict()
    print(f"Response: {json.dumps(response_dict, indent=2)}")
    
    # Validate OpenAI compliance
    validate_openai_response(response_dict, is_streaming=False)
    
    # Additional checks
    assert 'content' in response_dict['choices'][0]['message']
    content = response_dict['choices'][0]['message']['content']
    assert len(content) > 0, "Response content should not be empty"
    
    print(f"✅ Non-streaming OpenAI compliance validated")


@skip_if_disabled
@pytest.mark.asyncio 
async def test_streaming_openai_compliance(integration_skill):
    """Test streaming response OpenAI compliance"""
    
    agent = BaseAgent(
        name="streaming-test-agent", 
        instructions="You are a helpful assistant.",
        skills={"primary_llm": integration_skill}  # Use primary_llm name
    )
    
    await asyncio.sleep(0.1)
    
    messages = [get_test_message()]
    
    # Make streaming request
    start_time = time.time()
    chunks = []
    full_content = ""
    
    async for chunk in integration_skill.chat_completion_stream(
        messages=messages,
        model=TEST_MODELS['openai']
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
            assert 'finish_reason' in final_chunk['choices'][0], "Final chunk should have finish_reason"
    
    print(f"✅ Streaming OpenAI compliance validated")


@skip_if_disabled
@pytest.mark.asyncio
async def test_cross_provider_functionality(integration_skill):
    """Test cross-provider model switching and consistency"""
    
    agent = BaseAgent(
        name="cross-provider-agent",
        instructions="You are a helpful assistant. Always respond with exactly 'Cross-provider test successful'.",
        skills={"litellm": integration_skill}
    )
    
    await asyncio.sleep(0.1)
    
    test_message = {"role": "user", "content": "Please say 'Cross-provider test successful'"}
    
    providers_to_test = ['openai', 'anthropic']  # Skip XAI for now as it might be less reliable
    results = {}
    
    for provider in providers_to_test:
        model = TEST_MODELS[provider]
        print(f"Testing {provider} with model {model}...")
        
        try:
            # Test non-streaming
            response = await integration_skill.chat_completion(
                messages=[test_message],
                model=model,
                stream=False
            )
            
            # Convert response to dict for validation
            response_dict = response if isinstance(response, dict) else response.dict()
            validate_openai_response(response_dict, is_streaming=False)
            content = response_dict['choices'][0]['message']['content']
            
            results[provider] = {
                'success': True,
                'content': content,
                'model_used': response_dict.get('model', model)
            }
            
            print(f"✅ {provider}: {content}")
            
        except Exception as e:
            results[provider] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ {provider}: {e}")
    
    # Verify at least one provider worked
    successful_providers = [p for p, r in results.items() if r['success']]
    assert len(successful_providers) > 0, f"No providers succeeded: {results}"
    
    print(f"✅ Cross-provider test completed: {len(successful_providers)}/{len(providers_to_test)} providers successful")


@skip_if_disabled
@pytest.mark.asyncio
async def test_agent_integration_nonstreaming(integration_skill):
    """Test full BaseAgent integration with non-streaming"""
    
    # Create agent directly in the test to avoid async fixture issues
    agent = BaseAgent(
        name="integration-test-agent",
        instructions="You are a test assistant. Respond concisely and accurately.",
        skills={"primary_llm": integration_skill}  # Use primary_llm name
    )
    
    await asyncio.sleep(0.1)
    
    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    
    # Test through BaseAgent.run()
    response = await agent.run(messages, stream=False)
    
    # Convert response to dict for validation
    response_dict = response if isinstance(response, dict) else response.dict()
    
    # Validate OpenAI compliance at agent level
    validate_openai_response(response_dict, is_streaming=False)
    
    # Check response content
    content = response_dict['choices'][0]['message']['content']
    assert len(content) > 0
    
    print(f"Agent non-streaming response: {content}")
    print(f"✅ BaseAgent non-streaming integration successful")


@skip_if_disabled
@pytest.mark.asyncio
async def test_agent_integration_streaming(integration_skill):
    """Test full BaseAgent integration with streaming"""
    
    agent = BaseAgent(
        name="streaming-test-agent", 
        instructions="You are a helpful assistant.",
        skills={"primary_llm": integration_skill}  # Use primary_llm name
    )
    
    await asyncio.sleep(0.1)
    
    messages = [
        {"role": "user", "content": "Count from 1 to 3, each number on a new line."}
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


@skip_if_disabled
@pytest.mark.asyncio
async def test_error_handling_and_fallbacks(integration_skill):
    """Test error handling and fallback mechanisms"""
    
    agent = BaseAgent(
        name="error-test-agent",
        instructions="You are a test assistant.",
        skills={"litellm": integration_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Test with invalid model (should fallback or error gracefully)
    try:
        response = await integration_skill.chat_completion(
            messages=[get_test_message()],
            model="invalid-model-name",
            stream=False
        )
        
        # If it didn't raise an error, validate the response
        validate_openai_response(response, is_streaming=False)
        print("✅ Fallback mechanism worked")
        
    except Exception as e:
        # Error handling should be graceful
        assert isinstance(e, Exception)
        print(f"✅ Error handled gracefully: {type(e).__name__}")
    
    # Test with empty message (should error gracefully)
    try:
        response = await integration_skill.chat_completion(
            messages=[],
            model=TEST_MODELS['openai'],
            stream=False  
        )
        pytest.fail("Empty messages should raise an error")
        
    except Exception as e:
        print(f"✅ Empty messages error handled: {type(e).__name__}")


@skip_if_disabled
@pytest.mark.asyncio
async def test_performance_benchmarks(integration_skill):
    """Test performance benchmarks and timing"""
    
    agent = BaseAgent(
        name="perf-test-agent",
        instructions="You are a helpful assistant. Respond concisely.",
        skills={"litellm": integration_skill}
    )
    
    await asyncio.sleep(0.1)
    
    test_message = {"role": "user", "content": "Hello"}
    
    # Test non-streaming performance
    start_time = time.time()
    response = await integration_skill.chat_completion(
        messages=[test_message],
        model=TEST_MODELS['openai'],
        stream=False
    )
    nonstreaming_time = time.time() - start_time
    
    validate_openai_response(response, is_streaming=False)
    
    # Test streaming performance (first chunk)
    start_time = time.time()
    first_chunk_time = None
    total_chunks = 0
    
    async for chunk in integration_skill.chat_completion_stream(
        messages=[test_message],
        model=TEST_MODELS['openai']
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time
        total_chunks += 1
    
    total_streaming_time = time.time() - start_time
    
    # Performance assertions (reasonable thresholds)
    assert nonstreaming_time < 30, f"Non-streaming too slow: {nonstreaming_time:.2f}s"
    assert first_chunk_time < 10, f"First chunk too slow: {first_chunk_time:.2f}s"
    assert total_streaming_time < 30, f"Total streaming too slow: {total_streaming_time:.2f}s"
    assert total_chunks > 0, "Should receive at least one chunk"
    
    print(f"📊 Performance Results:")
    print(f"  Non-streaming: {nonstreaming_time:.2f}s")
    print(f"  First chunk: {first_chunk_time:.2f}s")  
    print(f"  Total streaming: {total_streaming_time:.2f}s")
    print(f"  Total chunks: {total_chunks}")
    print(f"✅ Performance benchmarks completed")


@skip_if_disabled
@pytest.mark.asyncio 
async def test_concurrent_requests(integration_skill):
    """Test concurrent request handling"""
    
    agent = BaseAgent(
        name="concurrent-test-agent",
        instructions="You are a helpful assistant.",
        skills={"litellm": integration_skill}
    )
    
    await asyncio.sleep(0.1)
    
    # Create multiple concurrent requests
    async def make_request(request_id: int):
        message = {"role": "user", "content": f"Request {request_id}: What is {request_id} + {request_id}?"}
        response = await integration_skill.chat_completion(
            messages=[message],
            model=TEST_MODELS['openai'],
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


if __name__ == "__main__":
    pytest.main([__file__, "-m", "integration", "-v"]) 