"""
Test suite for minimalistic n8n skill
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.n8n.skill import N8nSkill


@pytest.fixture
def n8n_skill():
    """Create N8nSkill instance for testing"""
    with patch.dict('os.environ', {
        'N8N_BASE_URL': 'http://localhost:5678'
    }):
        skill = N8nSkill()
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {
                    'kv': MockKVSkill(),
                    'auth': MockAuthSkill()
                }
                self._n8n_credentials = {}
        
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
    """Mock n8n credentials"""
    return {
        'api_key': 'n8n_test_api_key',
        'base_url': 'http://localhost:5678',
        'created_at': '2024-01-01T00:00:00'
    }


class TestN8nSkill:
    """Test cases for minimalistic n8n skill"""

    def test_skill_initialization(self, n8n_skill):
        """Test skill initializes correctly"""
        assert n8n_skill.default_n8n_url == 'http://localhost:5678'
        assert 'test-agent' == n8n_skill.agent.name
        assert n8n_skill.get_dependencies() == ['auth', 'kv']

    def test_prompt_mentions_tools(self, n8n_skill):
        """Test prompt mentions all tools"""
        prompt = n8n_skill.n8n_prompt()
        assert 'n8n_setup(' in prompt
        assert 'n8n_execute(' in prompt
        assert 'n8n_list_workflows(' in prompt
        assert 'n8n_status(' in prompt
        assert 'Minimalistic n8n integration' in prompt

    def test_has_4_tools(self, n8n_skill):
        """Test skill has exactly 4 tools"""
        n8n_tools = [m for m in dir(n8n_skill) if m.startswith('n8n_') and not m.startswith('n8n_prompt')]
        assert len(n8n_tools) == 4
        assert 'n8n_setup' in n8n_tools
        assert 'n8n_execute' in n8n_tools
        assert 'n8n_list_workflows' in n8n_tools
        assert 'n8n_status' in n8n_tools

    @pytest.mark.asyncio
    async def test_save_load_credentials_kv(self, n8n_skill, mock_context):
        """Test saving and loading credentials via KV skill"""
        user_id = 'test_user_123'
        api_key = 'test_api_key'
        base_url = 'http://localhost:5678'
        
        # Test save
        success = await n8n_skill._save_n8n_credentials(user_id, api_key, base_url)
        assert success == True
        
        # Test load
        credentials = await n8n_skill._load_n8n_credentials(user_id)
        assert credentials['api_key'] == api_key
        assert credentials['base_url'] == base_url
        assert 'created_at' in credentials

    @pytest.mark.asyncio
    async def test_save_load_credentials_memory_fallback(self, n8n_skill):
        """Test memory fallback when KV skill unavailable"""
        # Remove KV skill to test fallback
        n8n_skill.agent.skills['kv'] = None
        
        user_id = 'test_user_123'
        api_key = 'test_api_key'
        base_url = 'http://localhost:5678'
        
        # Test save (should use memory fallback)
        success = await n8n_skill._save_n8n_credentials(user_id, api_key, base_url)
        assert success == True
        
        # Test load (should use memory fallback)
        credentials = await n8n_skill._load_n8n_credentials(user_id)
        assert credentials['api_key'] == api_key
        assert credentials['base_url'] == base_url

    @pytest.mark.asyncio
    async def test_n8n_setup_success(self, n8n_skill, mock_context):
        """Test successful n8n setup"""
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_make_n8n_request', return_value={'data': []}):
                result = await n8n_skill.n8n_setup('test_api_key', 'http://localhost:5678')
                
                assert '✅ n8n credentials saved successfully!' in result
                assert 'http://localhost:5678' in result

    @pytest.mark.asyncio
    async def test_n8n_setup_invalid_api_key(self, n8n_skill, mock_context):
        """Test n8n setup with invalid API key"""
        import httpx
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            # Mock HTTP 401 error
            mock_response = MagicMock()
            mock_response.status_code = 401
            
            with patch.object(n8n_skill, '_make_n8n_request', side_effect=httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_response)):
                result = await n8n_skill.n8n_setup('invalid_key')
                
                assert '❌ Invalid API key' in result

    @pytest.mark.asyncio
    async def test_n8n_setup_no_auth(self, n8n_skill):
        """Test n8n setup without authentication"""
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=None):
            result = await n8n_skill.n8n_setup('test_api_key')
            
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_n8n_setup_empty_api_key(self, n8n_skill, mock_context):
        """Test n8n setup with empty API key"""
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            result = await n8n_skill.n8n_setup('')
            
            assert '❌ API key is required' in result

    @pytest.mark.asyncio
    async def test_n8n_execute_success(self, n8n_skill, mock_context, mock_credentials):
        """Test successful workflow execution"""
        mock_response = {
            'id': 'exec_123',
            'status': 'running'
        }
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', return_value=mock_response):
                    result = await n8n_skill.n8n_execute('workflow_123', {'input': 'test'})
                    
                    assert '✅ Workflow executed successfully!' in result
                    assert 'exec_123' in result
                    assert 'running' in result

    @pytest.mark.asyncio
    async def test_n8n_execute_no_credentials(self, n8n_skill, mock_context):
        """Test workflow execution without credentials"""
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=None):
                result = await n8n_skill.n8n_execute('workflow_123')
                
                assert '❌' in result
                assert 'n8n_setup()' in result

    @pytest.mark.asyncio
    async def test_n8n_execute_workflow_not_found(self, n8n_skill, mock_context, mock_credentials):
        """Test workflow execution with non-existent workflow"""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)):
                    result = await n8n_skill.n8n_execute('nonexistent')
                    
                    assert '❌ Workflow \'nonexistent\' not found' in result

    @pytest.mark.asyncio
    async def test_n8n_list_workflows_success(self, n8n_skill, mock_context, mock_credentials):
        """Test successful workflow listing"""
        mock_response = {
            'data': [
                {
                    'id': 'workflow_1',
                    'name': 'Test Workflow 1',
                    'active': True,
                    'tags': [{'name': 'automation'}]
                },
                {
                    'id': 'workflow_2', 
                    'name': 'Test Workflow 2',
                    'active': False,
                    'tags': []
                }
            ]
        }
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', return_value=mock_response):
                    result = await n8n_skill.n8n_list_workflows()
                    
                    assert '📋 Available n8n Workflows:' in result
                    assert 'Test Workflow 1' in result
                    assert 'Test Workflow 2' in result
                    assert '🟢' in result  # Active workflow
                    assert '🔴' in result  # Inactive workflow
                    assert 'automation' in result  # Tag

    @pytest.mark.asyncio
    async def test_n8n_list_workflows_empty(self, n8n_skill, mock_context, mock_credentials):
        """Test workflow listing with no workflows"""
        mock_response = {'data': []}
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', return_value=mock_response):
                    result = await n8n_skill.n8n_list_workflows()
                    
                    assert '📭 No workflows found' in result

    @pytest.mark.asyncio
    async def test_n8n_status_success(self, n8n_skill, mock_context, mock_credentials):
        """Test successful execution status check"""
        mock_response = {
            'status': 'success',
            'workflowId': 'workflow_123',
            'startedAt': '2024-01-01T10:00:00Z',
            'stoppedAt': '2024-01-01T10:05:00Z'
        }
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', return_value=mock_response):
                    result = await n8n_skill.n8n_status('exec_123')
                    
                    assert '📊 Execution Status Report' in result
                    assert '✅ Status: success' in result
                    assert 'workflow_123' in result
                    assert 'exec_123' in result

    @pytest.mark.asyncio
    async def test_n8n_status_error(self, n8n_skill, mock_context, mock_credentials):
        """Test execution status check for failed execution"""
        mock_response = {
            'status': 'error',
            'workflowId': 'workflow_123',
            'startedAt': '2024-01-01T10:00:00Z',
            'stoppedAt': '2024-01-01T10:01:00Z',
            'data': {'resultData': {'error': 'Something went wrong'}}
        }
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', return_value=mock_response):
                    result = await n8n_skill.n8n_status('exec_123')
                    
                    assert '❌ Status: error' in result
                    assert 'Error details available' in result

    @pytest.mark.asyncio
    async def test_n8n_status_execution_not_found(self, n8n_skill, mock_context, mock_credentials):
        """Test status check for non-existent execution"""
        import httpx
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
                with patch.object(n8n_skill, '_make_n8n_request', side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)):
                    result = await n8n_skill.n8n_status('nonexistent')
                    
                    assert '❌ Execution \'nonexistent\' not found' in result

    @pytest.mark.asyncio
    async def test_make_n8n_request_methods(self, n8n_skill, mock_credentials):
        """Test different HTTP methods in _make_n8n_request"""
        user_id = 'test_user'
        
        with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = {'success': True}
                mock_response.content = b'{"success": true}'
                
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.put.return_value = mock_response
                mock_client.return_value.__aenter__.return_value.delete.return_value = mock_response
                
                # Test GET
                result = await n8n_skill._make_n8n_request('GET', '/test', user_id=user_id)
                assert result == {'success': True}
                
                # Test POST
                result = await n8n_skill._make_n8n_request('POST', '/test', {'data': 'test'}, user_id)
                assert result == {'success': True}
                
                # Test PUT
                result = await n8n_skill._make_n8n_request('PUT', '/test', {'data': 'test'}, user_id)
                assert result == {'success': True}
                
                # Test DELETE
                result = await n8n_skill._make_n8n_request('DELETE', '/test', user_id=user_id)
                assert result == {'success': True}

    @pytest.mark.asyncio
    async def test_make_n8n_request_unsupported_method(self, n8n_skill, mock_credentials):
        """Test unsupported HTTP method"""
        user_id = 'test_user'
        
        with patch.object(n8n_skill, '_load_n8n_credentials', return_value=mock_credentials):
            with pytest.raises(Exception, match="Unsupported HTTP method"):
                await n8n_skill._make_n8n_request('PATCH', '/test', user_id=user_id)

    @pytest.mark.asyncio
    async def test_tools_require_authentication(self, n8n_skill):
        """Test that all tools require authentication"""
        unauthenticated_context = MagicMock()
        unauthenticated_context.auth = None
        
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=unauthenticated_context):
            # Test all tools return authentication required
            result = await n8n_skill.n8n_setup('test_key')
            assert '❌ Authentication required' in result
            
            result = await n8n_skill.n8n_execute('workflow_id')
            assert '❌ Authentication required' in result
            
            result = await n8n_skill.n8n_list_workflows()
            assert '❌ Authentication required' in result
            
            result = await n8n_skill.n8n_status('exec_id')
            assert '❌ Authentication required' in result

    @pytest.mark.asyncio
    async def test_url_normalization(self, n8n_skill, mock_context):
        """Test URL normalization in setup"""
        with patch('webagents.agents.skills.ecosystem.n8n.skill.get_context', return_value=mock_context):
            with patch.object(n8n_skill, '_make_n8n_request', return_value={'data': []}):
                # Test URL without protocol gets https added
                result = await n8n_skill.n8n_setup('test_key', 'example.com')
                assert 'https://example.com' in result
                
                # Test URL with http protocol is preserved
                result = await n8n_skill.n8n_setup('test_key', 'http://localhost:5678')
                assert 'http://localhost:5678' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
