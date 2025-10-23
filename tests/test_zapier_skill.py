"""
Test suite for minimalistic Zapier skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.zapier.skill import ZapierSkill


@pytest.fixture
def zapier_skill():
    """Create ZapierSkill instance for testing"""
    skill = ZapierSkill()
    
    # Create proper mock agent
    class MockAgent:
        def __init__(self):
            self.name = 'test-agent'
            self.skills = {
                'kv': MockKVSkill(),
                'auth': MockAuthSkill()
            }
            self._zapier_credentials = {}
    
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
    """Mock Zapier credentials"""
    return {
        'api_key': 'zapier_test_api_key',
        'created_at': '2024-01-01T00:00:00'
    }


class TestZapierSkill:
    """Test cases for minimalistic Zapier skill"""

    def test_skill_initialization(self, zapier_skill):
        """Test skill initializes correctly"""
        assert zapier_skill.api_base == 'https://api.zapier.com/v1'
        assert 'test-agent' == zapier_skill.agent.name
        assert zapier_skill.get_dependencies() == ['auth', 'kv']

    def test_prompt_mentions_tools(self, zapier_skill):
        """Test prompt mentions all tools"""
        prompt = zapier_skill.zapier_prompt()
        assert 'zapier_setup(' in prompt
        assert 'zapier_trigger(' in prompt
        assert 'zapier_list_zaps(' in prompt
        assert 'zapier_status(' in prompt
        assert 'Minimalistic Zapier integration' in prompt

    def test_has_4_tools(self, zapier_skill):
        """Test skill has exactly 4 tools"""
        zapier_tools = [m for m in dir(zapier_skill) if m.startswith('zapier_') and not m.startswith('zapier_prompt')]
        assert len(zapier_tools) == 4
        assert 'zapier_setup' in zapier_tools
        assert 'zapier_trigger' in zapier_tools
        assert 'zapier_list_zaps' in zapier_tools
        assert 'zapier_status' in zapier_tools

    @pytest.mark.asyncio
    async def test_save_load_credentials_kv(self, zapier_skill, mock_context):
        """Test saving and loading credentials via KV skill"""
        user_id = 'test_user_123'
        api_key = 'test_api_key'
        
        # Test save
        success = await zapier_skill._save_zapier_credentials(user_id, api_key)
        assert success == True
        
        # Test load
        credentials = await zapier_skill._load_zapier_credentials(user_id)
        assert credentials['api_key'] == api_key
        assert 'created_at' in credentials

    @pytest.mark.asyncio
    async def test_save_load_credentials_memory_fallback(self, zapier_skill):
        """Test memory fallback when KV skill unavailable"""
        # Remove KV skill to test fallback
        zapier_skill.agent.skills['kv'] = None
        
        user_id = 'test_user_123'
        api_key = 'test_api_key'
        
        # Test save (should use memory fallback)
        success = await zapier_skill._save_zapier_credentials(user_id, api_key)
        assert success == True
        
        # Test load (should use memory fallback)
        credentials = await zapier_skill._load_zapier_credentials(user_id)
        assert credentials['api_key'] == api_key

    @pytest.mark.asyncio
    async def test_zapier_setup_success(self, zapier_skill, mock_context):
        """Test successful Zapier setup"""
        mock_zaps_response = {'data': [{'id': 'zap1'}, {'id': 'zap2'}]}
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_zaps_response):
                result = await zapier_skill.zapier_setup('test_api_key')
                
                assert '✅ Zapier credentials saved successfully!' in result
                assert 'Found 2 Zaps' in result

    @pytest.mark.asyncio
    async def test_zapier_setup_invalid_api_key(self, zapier_skill, mock_context):
        """Test Zapier setup with invalid API key"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            # Mock HTTP 401 error
            mock_response = MagicMock()
            mock_response.status_code = 401
            
            with patch.object(zapier_skill, '_make_zapier_request', side_effect=httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_response)):
                result = await zapier_skill.zapier_setup('invalid_key')
                
                assert '❌ Invalid API key' in result

    @pytest.mark.asyncio
    async def test_zapier_setup_no_auth(self, zapier_skill):
        """Test Zapier setup without authentication"""
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=None):
            result = await zapier_skill.zapier_setup('test_api_key')
            
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_zapier_setup_empty_api_key(self, zapier_skill, mock_context):
        """Test Zapier setup with empty API key"""
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            result = await zapier_skill.zapier_setup('')
            
            assert '❌ API key is required' in result

    @pytest.mark.asyncio
    async def test_zapier_trigger_success(self, zapier_skill, mock_context, mock_credentials):
        """Test successful Zap trigger"""
        mock_response = {
            'id': 'task_123',
            'status': 'triggered'
        }
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_response):
                    result = await zapier_skill.zapier_trigger('zap_123', {'input': 'test'})
                    
                    assert '✅ Zap triggered successfully!' in result
                    assert 'task_123' in result
                    assert 'triggered' in result

    @pytest.mark.asyncio
    async def test_zapier_trigger_no_credentials(self, zapier_skill, mock_context):
        """Test Zap trigger without credentials"""
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=None):
                result = await zapier_skill.zapier_trigger('zap_123')
                
                assert '❌' in result
                assert 'zapier_setup()' in result

    @pytest.mark.asyncio
    async def test_zapier_trigger_zap_not_found(self, zapier_skill, mock_context, mock_credentials):
        """Test Zap trigger with non-existent Zap"""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)):
                    result = await zapier_skill.zapier_trigger('nonexistent')
                    
                    assert '❌ Zap \'nonexistent\' not found' in result

    @pytest.mark.asyncio
    async def test_zapier_list_zaps_success(self, zapier_skill, mock_context, mock_credentials):
        """Test successful Zap listing"""
        mock_response = {
            'data': [
                {
                    'id': 'zap_1',
                    'title': 'Email to Slack',
                    'status': 'on',
                    'steps': [
                        {'app': {'title': 'Gmail'}},
                        {'app': {'title': 'Slack'}}
                    ]
                },
                {
                    'id': 'zap_2', 
                    'title': 'Form to Spreadsheet',
                    'status': 'off',
                    'steps': [
                        {'app': {'title': 'Typeform'}},
                        {'app': {'title': 'Google Sheets'}}
                    ]
                }
            ]
        }
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_response):
                    result = await zapier_skill.zapier_list_zaps()
                    
                    assert '📋 Available Zapier Zaps:' in result
                    assert 'Email to Slack' in result
                    assert 'Form to Spreadsheet' in result
                    assert '🟢' in result  # Active Zap
                    assert '🔴' in result  # Inactive Zap
                    assert 'Gmail' in result  # Trigger app

    @pytest.mark.asyncio
    async def test_zapier_list_zaps_empty(self, zapier_skill, mock_context, mock_credentials):
        """Test Zap listing with no Zaps"""
        mock_response = {'data': []}
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_response):
                    result = await zapier_skill.zapier_list_zaps()
                    
                    assert '📭 No Zaps found' in result

    @pytest.mark.asyncio
    async def test_zapier_status_success(self, zapier_skill, mock_context, mock_credentials):
        """Test successful task status check"""
        mock_response = {
            'status': 'success',
            'zap_id': 'zap_123',
            'created_at': '2024-01-01T10:00:00Z',
            'updated_at': '2024-01-01T10:05:00Z'
        }
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_response):
                    result = await zapier_skill.zapier_status('task_123')
                    
                    assert '📊 Zap Execution Status Report' in result
                    assert '✅ Status: success' in result
                    assert 'zap_123' in result
                    assert 'task_123' in result

    @pytest.mark.asyncio
    async def test_zapier_status_error(self, zapier_skill, mock_context, mock_credentials):
        """Test task status check for failed execution"""
        mock_response = {
            'status': 'error',
            'zap_id': 'zap_123',
            'created_at': '2024-01-01T10:00:00Z',
            'updated_at': '2024-01-01T10:01:00Z',
            'error': 'Connection timeout'
        }
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', return_value=mock_response):
                    result = await zapier_skill.zapier_status('task_123')
                    
                    assert '❌ Status: error' in result
                    assert 'Connection timeout' in result

    @pytest.mark.asyncio
    async def test_zapier_status_task_not_found(self, zapier_skill, mock_context, mock_credentials):
        """Test status check for non-existent task"""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)):
                    result = await zapier_skill.zapier_status('nonexistent')
                    
                    assert '❌ Task \'nonexistent\' not found' in result

    @pytest.mark.asyncio
    async def test_make_zapier_request_methods(self, zapier_skill, mock_credentials):
        """Test different HTTP methods in _make_zapier_request"""
        user_id = 'test_user'
        
        with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {'success': True}
                mock_response.content = b'{"success": true}'
                
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.put.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.delete.return_value = mock_response
                
                # Test GET
                result = await zapier_skill._make_zapier_request('GET', '/test', user_id=user_id)
                assert result == {'success': True}
                
                # Test POST
                result = await zapier_skill._make_zapier_request('POST', '/test', {'data': 'test'}, user_id)
                assert result == {'success': True}
                
                # Test PUT
                result = await zapier_skill._make_zapier_request('PUT', '/test', {'data': 'test'}, user_id)
                assert result == {'success': True}
                
                # Test DELETE
                result = await zapier_skill._make_zapier_request('DELETE', '/test', user_id=user_id)
                assert result == {'success': True}

    @pytest.mark.asyncio
    async def test_make_zapier_request_unsupported_method(self, zapier_skill, mock_credentials):
        """Test unsupported HTTP method"""
        user_id = 'test_user'
        
        with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
            with pytest.raises(Exception, match="Unsupported HTTP method"):
                await zapier_skill._make_zapier_request('PATCH', '/test', user_id=user_id)

    @pytest.mark.asyncio
    async def test_tools_require_authentication(self, zapier_skill):
        """Test that all tools require authentication"""
        unauthenticated_context = MagicMock()
        unauthenticated_context.auth = None
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=unauthenticated_context):
            # Test all tools return authentication required
            result = await zapier_skill.zapier_setup('test_key')
            assert '❌ Authentication required' in result
            
            result = await zapier_skill.zapier_trigger('zap_id')
            assert '❌ Authentication required' in result
            
            result = await zapier_skill.zapier_list_zaps()
            assert '❌ Authentication required' in result
            
            result = await zapier_skill.zapier_status('task_id')
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_zapier_setup_permission_denied(self, zapier_skill, mock_context):
        """Test Zapier setup with permission denied"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            # Mock HTTP 403 error
            mock_response = MagicMock()
            mock_response.status_code = 403
            
            with patch.object(zapier_skill, '_make_zapier_request', side_effect=httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)):
                result = await zapier_skill.zapier_setup('limited_key')
                
                assert '❌ API key doesn\'t have required permissions' in result

    @pytest.mark.asyncio
    async def test_zapier_trigger_permission_denied(self, zapier_skill, mock_context, mock_credentials):
        """Test Zap trigger with permission denied"""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 403
        
        with patch('webagents.agents.skills.ecosystem.zapier.skill.get_context', return_value=mock_context):
            with patch.object(zapier_skill, '_load_zapier_credentials', return_value=mock_credentials):
                with patch.object(zapier_skill, '_make_zapier_request', side_effect=httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)):
                    result = await zapier_skill.zapier_trigger('restricted_zap')
                    
                    assert '❌ Permission denied' in result
                    assert 'Check if Zap is enabled' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
