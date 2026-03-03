"""
Integration Test Configuration - WebAgents V2.0

Configuration for integration tests with real LiteLLM proxy API calls.
"""

import os
from typing import Dict, Any

# LiteLLM Proxy Configuration
LITELLM_BASE_URL = os.getenv('LITELLM_BASE_URL', 'http://localhost:2225')
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY', 'rok_testapikey')

# Test Models (available on the local proxy)
TEST_MODELS = {
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-haiku', 
    'xai': 'grok-4',
    'azure': 'azure/gpt-4o-mini'
}

# OpenAI Compliance Test Configuration
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

# Integration Test Settings
INTEGRATION_TEST_CONFIG = {
    'timeout': int(os.getenv('INTEGRATION_TEST_TIMEOUT', '30')),
    'run_integration_tests': os.getenv('RUN_INTEGRATION_TESTS', 'true').lower() == 'true',
    'max_retries': 3,
    'retry_delay': 1.0
}

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
        'custom_models': {
            model_name: {
                'name': model_name,
                'provider': provider,
                'max_tokens': 4096,
                'supports_tools': True,
                'supports_streaming': True
            }
            for provider, model_name in TEST_MODELS.items()
        }
    }

def is_integration_tests_enabled() -> bool:
    """Check if integration tests should run"""
    return INTEGRATION_TEST_CONFIG['run_integration_tests']

def get_test_message() -> Dict[str, Any]:
    """Get a standard test message for integration tests"""
    return {"role": "user", "content": "Hello! Please respond with 'Integration test successful' and nothing else."} 