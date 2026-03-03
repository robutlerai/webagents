"""
Test suite for Replicate skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.replicate.skill import ReplicateSkill


@pytest.fixture
def replicate_skill():
    """Create ReplicateSkill instance for testing"""
    skill = ReplicateSkill()
    
    # Create proper mock agent
    class MockAgent:
        def __init__(self):
            self.name = 'test-agent'
            self.skills = {
                'kv': MockKVSkill(),
                'auth': MockAuthSkill()
            }
            self._replicate_credentials = {}
    
    skill.agent = MockAgent()
    skill.logger = MagicMock()
    return skill


class MockKVSkill:
    """Mock KV skill"""
    def __init__(self):
        self.storage = {}
    
    async def kv_set(self, key, value, namespace=None):
        full_key = f"{namespace}:{key}" if namespace else key
        self.storage[full_key] = value
        return "✅ Saved"
    
    async def kv_get(self, key, namespace=None):
        full_key = f"{namespace}:{key}" if namespace else key
        return self.storage.get(full_key, "")


class MockAuthSkill:
    """Mock auth skill"""
    pass


@pytest.fixture
def mock_context():
    """Mock context with user authentication"""
    context = MagicMock()
    context.auth = MagicMock()
    context.auth.user_id = 'test_user_123'
    context.auth.authenticated = True
    return context


@pytest.fixture
def mock_credentials():
    """Mock Replicate credentials"""
    return {
        'api_token': 'r8_test_replicate_token',
        'created_at': '2024-01-01T00:00:00'
    }


class TestReplicateSkill:
    """Test cases for Replicate skill"""

    def test_skill_initialization(self, replicate_skill):
        """Test skill initializes correctly"""
        assert replicate_skill.api_base == 'https://api.replicate.com/v1'
        assert 'test-agent' == replicate_skill.agent.name
        assert replicate_skill.get_dependencies() == ['auth', 'kv']

    def test_prompt_mentions_tools(self, replicate_skill):
        """Test prompt mentions all tools"""
        prompt = replicate_skill.replicate_prompt()
        assert 'replicate_setup(' in prompt
        assert 'replicate_list_models(' in prompt
        assert 'replicate_run_prediction(' in prompt
        assert 'replicate_get_prediction(' in prompt
        assert 'replicate_cancel_prediction(' in prompt
        assert 'replicate_get_model_info(' in prompt
        assert 'Replicate integration' in prompt

    def test_has_6_tools(self, replicate_skill):
        """Test skill has exactly 6 tools"""
        replicate_tools = [m for m in dir(replicate_skill) if m.startswith('replicate_') and not m.startswith('replicate_prompt')]
        assert len(replicate_tools) == 6
        assert 'replicate_setup' in replicate_tools
        assert 'replicate_list_models' in replicate_tools
        assert 'replicate_run_prediction' in replicate_tools
        assert 'replicate_get_prediction' in replicate_tools
        assert 'replicate_cancel_prediction' in replicate_tools
        assert 'replicate_get_model_info' in replicate_tools

    @pytest.mark.asyncio
    async def test_save_load_credentials_kv(self, replicate_skill, mock_context):
        """Test saving and loading credentials via KV skill"""
        user_id = 'test_user_123'
        api_token = 'r8_test_token'
        
        # Test save
        success = await replicate_skill._save_replicate_credentials(user_id, api_token)
        assert success == True
        
        # Test load
        credentials = await replicate_skill._load_replicate_credentials(user_id)
        assert credentials['api_token'] == api_token
        assert 'created_at' in credentials

    @pytest.mark.asyncio
    async def test_save_load_credentials_memory_fallback(self, replicate_skill):
        """Test memory fallback when KV skill unavailable"""
        # Remove KV skill to test fallback
        replicate_skill.agent.skills['kv'] = None
        
        user_id = 'test_user_123'
        api_token = 'r8_test_token'
        
        # Test save (should use memory fallback)
        success = await replicate_skill._save_replicate_credentials(user_id, api_token)
        assert success == True
        
        # Test load (should use memory fallback)
        credentials = await replicate_skill._load_replicate_credentials(user_id)
        assert credentials['api_token'] == api_token

    @pytest.mark.asyncio
    async def test_replicate_setup_success(self, replicate_skill, mock_context):
        """Test successful Replicate setup"""
        mock_models_response = {'results': [{'name': 'model1'}, {'name': 'model2'}]}
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_models_response):
                result = await replicate_skill.replicate_setup('r8_test_token')
                
                assert '✅ Replicate credentials saved successfully!' in result
                assert 'Connection to Replicate API verified' in result

    @pytest.mark.asyncio
    async def test_replicate_setup_invalid_token(self, replicate_skill, mock_context):
        """Test Replicate setup with invalid API token"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            # Mock HTTP 401 error
            mock_response = MagicMock()
            mock_response.status_code = 401
            
            with patch.object(replicate_skill, '_make_replicate_request', 
                            side_effect=httpx.HTTPStatusError("401 Unauthorized", 
                                                            request=MagicMock(), 
                                                            response=mock_response)):
                result = await replicate_skill.replicate_setup('invalid_token')
                assert '❌ Invalid API token' in result

    @pytest.mark.asyncio
    async def test_replicate_setup_empty_token(self, replicate_skill, mock_context):
        """Test Replicate setup with empty API token"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            result = await replicate_skill.replicate_setup('')
            assert '❌ API token is required' in result

    @pytest.mark.asyncio
    async def test_replicate_setup_no_auth(self, replicate_skill):
        """Test Replicate setup without authentication"""
        mock_context = MagicMock()
        mock_context.auth.authenticated = False
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=None):
            result = await replicate_skill.replicate_setup('r8_test_token')
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_replicate_list_models_success(self, replicate_skill, mock_context):
        """Test successful model listing"""
        mock_models_response = {
            'results': [
                {
                    'name': 'stable-diffusion',
                    'owner': 'stability-ai',
                    'description': 'Generate images from text',
                    'visibility': 'public'
                },
                {
                    'name': 'whisper',
                    'owner': 'openai',
                    'description': 'Speech to text',
                    'visibility': 'public'
                }
            ]
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_models_response):
                result = await replicate_skill.replicate_list_models()
                
                assert '🤖 Available Models:' in result
                assert 'stability-ai/stable-diffusion' in result
                assert 'openai/whisper' in result
                assert 'Generate images from text' in result

    @pytest.mark.asyncio
    async def test_replicate_list_models_with_owner(self, replicate_skill, mock_context):
        """Test model listing with specific owner"""
        mock_models_response = {
            'results': [
                {
                    'name': 'stable-diffusion',
                    'owner': 'stability-ai',
                    'description': 'Generate images from text',
                    'visibility': 'public'
                }
            ]
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_models_response) as mock_request:
                result = await replicate_skill.replicate_list_models(owner='stability-ai')
                
                # Check that the correct endpoint was called
                mock_request.assert_called_with('GET', '/models?owner=stability-ai', user_id='test_user_123')
                assert '🤖 Available Models from stability-ai:' in result

    @pytest.mark.asyncio
    async def test_replicate_list_models_empty(self, replicate_skill, mock_context):
        """Test model listing with no results"""
        mock_models_response = {'results': []}
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_models_response):
                result = await replicate_skill.replicate_list_models()
                assert '📭 No models found' in result

    @pytest.mark.asyncio
    async def test_replicate_get_model_info_success(self, replicate_skill, mock_context):
        """Test successful model info retrieval"""
        mock_model_response = {
            'name': 'stable-diffusion',
            'owner': 'stability-ai',
            'description': 'Generate images from text prompts',
            'visibility': 'public',
            'github_url': 'https://github.com/stability-ai/stable-diffusion',
            'latest_version': {
                'id': 'v1.2.3',
                'created_at': '2024-01-15T10:30:00Z',
                'openapi_schema': {
                    'components': {
                        'schemas': {
                            'Input': {
                                'properties': {
                                    'prompt': {
                                        'type': 'string',
                                        'description': 'Text prompt for image generation'
                                    },
                                    'width': {
                                        'type': 'integer',
                                        'description': 'Width of output image'
                                    }
                                },
                                'required': ['prompt']
                            }
                        }
                    }
                }
            }
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_model_response):
                result = await replicate_skill.replicate_get_model_info('stability-ai/stable-diffusion')
                
                assert '🤖 Model Information: stability-ai/stable-diffusion' in result
                assert '🌍 Visibility: public' in result
                assert 'Generate images from text prompts' in result
                assert '📋 Input Parameters:' in result
                assert 'prompt (string)' in result

    @pytest.mark.asyncio
    async def test_replicate_get_model_info_invalid_format(self, replicate_skill, mock_context):
        """Test model info with invalid model name format"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            result = await replicate_skill.replicate_get_model_info('invalid-model-name')
            assert '❌ Model name must be in format: owner/model-name' in result

    @pytest.mark.asyncio
    async def test_replicate_run_prediction_success(self, replicate_skill, mock_context):
        """Test successful prediction creation"""
        mock_prediction_response = {
            'id': 'pred_abc123def456',
            'status': 'starting',
            'model': 'stability-ai/stable-diffusion'
        }
        
        input_data = {
            'prompt': 'A beautiful sunset',
            'width': 512,
            'height': 512
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_prediction_response):
                result = await replicate_skill.replicate_run_prediction('stability-ai/stable-diffusion', input_data)
                
                assert '🚀 Prediction started successfully!' in result
                assert 'pred_abc123def456' in result
                assert 'Status: starting' in result

    @pytest.mark.asyncio
    async def test_replicate_run_prediction_completed_immediately(self, replicate_skill, mock_context):
        """Test prediction that completes immediately"""
        mock_prediction_response = {
            'id': 'pred_abc123def456',
            'status': 'succeeded',
            'output': 'https://replicate.delivery/output.png'
        }
        
        input_data = {'prompt': 'Test prompt'}
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_prediction_response):
                result = await replicate_skill.replicate_run_prediction('test/model', input_data)
                
                assert '✅ Output:' in result
                assert 'https://replicate.delivery/output.png' in result

    @pytest.mark.asyncio
    async def test_replicate_run_prediction_invalid_model(self, replicate_skill, mock_context):
        """Test prediction with invalid model name"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            result = await replicate_skill.replicate_run_prediction('invalid-model', {'prompt': 'test'})
            assert '❌ Model name must be in format: owner/model-name' in result

    @pytest.mark.asyncio
    async def test_replicate_run_prediction_no_input(self, replicate_skill, mock_context):
        """Test prediction with no input data"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            result = await replicate_skill.replicate_run_prediction('test/model', None)
            assert '❌ Input data is required' in result

    @pytest.mark.asyncio
    async def test_replicate_get_prediction_success(self, replicate_skill, mock_context):
        """Test successful prediction status retrieval"""
        mock_prediction_response = {
            'id': 'pred_abc123def456',
            'status': 'succeeded',
            'model': 'stability-ai/stable-diffusion',
            'created_at': '2024-01-15T10:30:00Z',
            'started_at': '2024-01-15T10:30:05Z',
            'completed_at': '2024-01-15T10:32:15Z',
            'output': 'https://replicate.delivery/output.png'
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_prediction_response):
                result = await replicate_skill.replicate_get_prediction('pred_abc123def456')
                
                assert '📊 Prediction Status Report' in result
                assert 'pred_abc123def456' in result
                assert '✅ Status: succeeded' in result
                assert 'https://replicate.delivery/output.png' in result

    @pytest.mark.asyncio
    async def test_replicate_get_prediction_failed(self, replicate_skill, mock_context):
        """Test prediction status for failed prediction"""
        mock_prediction_response = {
            'id': 'pred_abc123def456',
            'status': 'failed',
            'error': 'Invalid input parameters',
            'logs': 'Error: prompt is required\nValidation failed'
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_prediction_response):
                result = await replicate_skill.replicate_get_prediction('pred_abc123def456')
                
                assert '❌ Status: failed' in result
                assert 'Invalid input parameters' in result
                assert '📝 Logs:' in result

    @pytest.mark.asyncio
    async def test_replicate_get_prediction_not_found(self, replicate_skill, mock_context):
        """Test prediction status for non-existent prediction"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            mock_response = MagicMock()
            mock_response.status_code = 404
            
            with patch.object(replicate_skill, '_make_replicate_request',
                            side_effect=httpx.HTTPStatusError("404 Not Found",
                                                            request=MagicMock(),
                                                            response=mock_response)):
                result = await replicate_skill.replicate_get_prediction('invalid_pred_id')
                assert "❌ Prediction 'invalid_pred_id' not found" in result

    @pytest.mark.asyncio
    async def test_replicate_cancel_prediction_success(self, replicate_skill, mock_context):
        """Test successful prediction cancellation"""
        mock_cancel_response = {
            'id': 'pred_abc123def456',
            'status': 'canceled'
        }
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch.object(replicate_skill, '_make_replicate_request', return_value=mock_cancel_response):
                result = await replicate_skill.replicate_cancel_prediction('pred_abc123def456')
                
                assert '✅ Prediction pred_abc123def456 canceled successfully' in result

    @pytest.mark.asyncio
    async def test_replicate_cancel_prediction_cannot_cancel(self, replicate_skill, mock_context):
        """Test cancellation of prediction that cannot be canceled"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            mock_response = MagicMock()
            mock_response.status_code = 422
            
            with patch.object(replicate_skill, '_make_replicate_request',
                            side_effect=httpx.HTTPStatusError("422 Unprocessable Entity",
                                                            request=MagicMock(),
                                                            response=mock_response)):
                result = await replicate_skill.replicate_cancel_prediction('pred_abc123def456')
                assert '❌ Cannot cancel prediction' in result

    @pytest.mark.asyncio
    async def test_make_replicate_request_no_credentials(self, replicate_skill, mock_context):
        """Test API request without saved credentials"""
        # Don't save any credentials
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with pytest.raises(Exception, match="Replicate credentials not found"):
                await replicate_skill._make_replicate_request('GET', '/models', user_id='test_user_123')

    @pytest.mark.asyncio
    async def test_make_replicate_request_with_credentials(self, replicate_skill, mock_context, mock_credentials):
        """Test API request with saved credentials"""
        # Save credentials first
        await replicate_skill._save_replicate_credentials('test_user_123', 'r8_test_token')
        
        mock_response = MagicMock()
        mock_response.json.return_value = {'results': []}
        mock_response.content = b'{"results": []}'
        
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                result = await replicate_skill._make_replicate_request('GET', '/models', user_id='test_user_123')
                
                assert result == {'results': []}
                # Verify correct headers were used
                call_args = mock_client.return_value.__aenter__.return_value.get.call_args
                headers = call_args[1]['headers']
                assert headers['Authorization'] == 'Token r8_test_token'

    @pytest.mark.asyncio
    async def test_authentication_flow(self, replicate_skill):
        """Test authentication flow without proper context"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=None):
            user_id = await replicate_skill._get_authenticated_user_id()
            assert user_id is None

    @pytest.mark.asyncio
    async def test_all_tools_require_authentication(self, replicate_skill):
        """Test that all tools properly check for authentication"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=None):
            # Test each tool without authentication
            result1 = await replicate_skill.replicate_setup('token')
            assert '❌ Authentication required' in result1
            
            result2 = await replicate_skill.replicate_list_models()
            assert '❌ Authentication required' in result2
            
            result3 = await replicate_skill.replicate_run_prediction('test/model', {'prompt': 'test'})
            assert '❌ Authentication required' in result3
            
            result4 = await replicate_skill.replicate_get_prediction('pred_123')
            assert '❌ Authentication required' in result4
            
            result5 = await replicate_skill.replicate_cancel_prediction('pred_123')
            assert '❌ Authentication required' in result5
            
            result6 = await replicate_skill.replicate_get_model_info('test/model')
            assert '❌ Authentication required' in result6

    def test_dependencies_are_correct(self, replicate_skill):
        """Test that skill dependencies are correctly specified"""
        deps = replicate_skill.get_dependencies()
        assert 'auth' in deps
        assert 'kv' in deps
        assert len(deps) == 2

    @pytest.mark.asyncio
    async def test_credential_storage_error_handling(self, replicate_skill):
        """Test error handling in credential storage"""
        # Mock KV skill that raises an exception
        class FailingKVSkill:
            async def kv_set(self, key, value, namespace=None):
                raise Exception("Storage failed")
            
            async def kv_get(self, key, namespace=None):
                raise Exception("Retrieval failed")
        
        replicate_skill.agent.skills['kv'] = FailingKVSkill()
        
        # Should fall back to memory storage
        success = await replicate_skill._save_replicate_credentials('user_123', 'token')
        assert success == True
        
        # Should fall back to memory retrieval
        credentials = await replicate_skill._load_replicate_credentials('user_123')
        assert credentials['api_token'] == 'token'

    @pytest.mark.asyncio
    async def test_empty_prediction_id_handling(self, replicate_skill, mock_context):
        """Test handling of empty prediction IDs"""
        with patch('webagents.agents.skills.ecosystem.replicate.skill.get_context', return_value=mock_context):
            result1 = await replicate_skill.replicate_get_prediction('')
            assert '❌ Prediction ID is required' in result1
            
            result2 = await replicate_skill.replicate_cancel_prediction('   ')
            assert '❌ Prediction ID is required' in result2

