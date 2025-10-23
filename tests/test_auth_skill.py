"""
AuthSkill Unit Tests - WebAgents V2.0

Comprehensive tests for WebAgents Platform integration and authentication functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.auth import AuthSkill, AuthScope, AuthContext, AuthenticationError, AuthorizationError
from robutler.api.types import User, UserRole, AuthResponse, ApiResponse


class MockContext:
    """Mock context for testing"""
    def __init__(self):
        self._data = {}
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def set(self, key, value):
        self._data[key] = value


@pytest.fixture
def mock_webagents_client():
    """Create mock WebAgents client for testing"""
    client = Mock()
    
    # Mock successful health check
    client.health_check = AsyncMock(return_value=ApiResponse(
        success=True,
        data={'status': 'healthy', 'service': 'webagents-platform'}
    ))
    
    # Mock successful user validation
    test_user = User(
        id='test-user-123',
        email='test@webagents.ai',
        name='Test User',
        role=UserRole.USER,
        total_credits=1000,
        used_credits=100
    )
    
    client.validate_api_key = AsyncMock(return_value=AuthResponse(
        success=True,
        user=test_user
    ))
    
    client.get_user = AsyncMock(return_value=AuthResponse(
        success=True,
        user=test_user
    ))
    
    client.get_user_credits = AsyncMock(return_value=ApiResponse(
        success=True,
        data={
            'total_credits': '1000',
            'used_credits': '100',
            'available_credits': '900'
        }
    ))
    
    client.track_usage = AsyncMock(return_value=ApiResponse(
        success=True,
        data={'transaction_id': 'tx_123', 'amount': '10', 'new_balance': '890'}
    ))
    
    client.get_user_transactions = AsyncMock(return_value=ApiResponse(
        success=True,
        data={
            'transactions': [
                {'id': 'tx_123', 'amount': '10', 'type': 'usage', 'description': 'API usage'}
            ]
        }
    ))
    
    client.list_api_keys = AsyncMock(return_value=ApiResponse(
        success=True,
        data={
            'api_keys': [
                {'id': 'key_123', 'name': 'Test Key', 'is_active': True}
            ]
        }
    ))
    
    client.close = AsyncMock()
    
    return client


@pytest.fixture
def auth_skill(mock_webagents_client):
    """Create AuthSkill with mocked client for testing"""
    config = {
        'require_auth': True,
        'platform_api_url': 'http://localhost:3000',
        'platform_api_key': 'rok_testapikey',
        'cache_ttl': 300
    }
    
    skill = AuthSkill(config)
    return skill


@pytest.mark.asyncio
async def test_auth_skill_initialization(auth_skill, mock_webagents_client):
    """Test AuthSkill initialization with WebAgents Platform client"""
    
    # Mock the RobutlerClient creation
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        agent = BaseAgent(
            name="auth-test-agent",
            instructions="Test agent for auth functionality.",
            skills={"auth": auth_skill}
        )
        await asyncio.sleep(0.1)  # Allow initialization
        
        # Check skill configuration
        assert auth_skill.require_auth == True
        assert auth_skill.platform_api_url == 'http://localhost:3000'
        assert auth_skill.platform_api_key == 'rok_testapikey'
        assert auth_skill.client is not None
        
        # Check client health check was called
        mock_webagents_client.health_check.assert_called_once()
        
        print("✅ AuthSkill initialization test passed")


@pytest.mark.asyncio
async def test_auth_skill_with_agent(auth_skill, mock_webagents_client):
    """Test AuthSkill integration with BaseAgent"""
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        agent = BaseAgent(
            name="auth-integration-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await asyncio.sleep(0.1)
        
        # Check skill is registered
        assert "auth" in agent.skills
        auth_skill_instance = agent.skills["auth"]
        assert isinstance(auth_skill_instance, AuthSkill)
        
        # Check tools are registered
        all_tools = agent.get_all_tools()
        tool_names = [tool['name'] for tool in all_tools]
        
        expected_tools = [
            'get_current_user', 'validate_api_key', 'get_user_credits',
            'track_usage', 'get_user_transactions', 'list_api_keys'
        ]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Tool {expected_tool} should be registered"
        
        # Check hooks are registered  
        request_hooks = agent._registered_hooks.get('on_request', [])
        auth_hooks = [hook for hook in request_hooks if hook.get('source') == 'auth']
        assert len(auth_hooks) > 0, "Auth request hooks should be registered"
        
        print("✅ AuthSkill-BaseAgent integration test passed")


@pytest.mark.asyncio
async def test_api_key_authentication(auth_skill, mock_webagents_client):
    """Test API key authentication with WebAgents Platform"""
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        agent = BaseAgent(
            name="auth-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await asyncio.sleep(0.1)
        
        # Test API key validation tool
        validate_tool = None
        for tool in agent.get_all_tools():
            if tool['name'] == 'validate_api_key':
                validate_tool = tool['function']
                break
        
        assert validate_tool is not None, "validate_api_key tool should be available"
        
        # Test successful validation
        result = await validate_tool('rok_testapikey')
        
        assert result['success'] == True
        assert 'user' in result
        assert result['user']['email'] == 'test@webagents.ai'
        assert result['message'] == 'API key is valid'
        
        # Verify platform client was called
        mock_webagents_client.validate_api_key.assert_called_with('rok_testapikey')
        
        print("✅ API key authentication test passed")


@pytest.mark.asyncio
async def test_request_authentication_hook(auth_skill, mock_webagents_client):
    """Test request authentication hook"""
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        auth_skill.client = mock_webagents_client
        
        # Test request with API key
        context = MockContext()
        context.set('headers', {'Authorization': 'Bearer rok_testapikey'})
        context.set('path', '/api/test')
        
        # Mock the authentication cache
        test_user = User(id='test-user-123', email='test@webagents.ai', role=UserRole.USER)
        auth_context = AuthContext(
            user=test_user,
            authenticated=True,
            scope=AuthScope.USER,
            permissions=['read', 'write']
        )
        auth_skill._auth_cache['rok_testapikey'] = auth_context
        
        # Test authentication
        result_context = await auth_skill.validate_request_auth(context)
        
        assert result_context.get('authenticated') == True
        assert result_context.get('user_id') == 'test-user-123'
        assert result_context.get('auth_scope') == 'user'
        
        print("✅ Request authentication hook test passed")


@pytest.mark.asyncio
async def test_user_tools(auth_skill, mock_webagents_client):
    """Test user-related tools"""
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        agent = BaseAgent(
            name="user-tools-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await asyncio.sleep(0.1)
        
        # Create authenticated context
        test_user = User(id='test-user-123', email='test@webagents.ai', role=UserRole.USER)
        auth_context = AuthContext(
            user=test_user,
            authenticated=True,
            scope=AuthScope.USER,
            permissions=['read', 'write']
        )
        context = MockContext()
        context.set('auth_context', auth_context)
        
        # Test get_current_user tool
        get_user_tool = None
        for tool in agent.get_all_tools():
            if tool['name'] == 'get_current_user':
                get_user_tool = tool['function']
                break
        
        result = await get_user_tool(context=context)
        assert result['success'] == True
        assert result['user']['id'] == 'test-user-123'
        
        # Test get_user_credits tool
        credits_tool = None
        for tool in agent.get_all_tools():
            if tool['name'] == 'get_user_credits':
                credits_tool = tool['function']
                break
        
        result = await credits_tool(context=context)
        assert result['success'] == True
        assert 'credits' in result
        
        # Test track_usage tool
        usage_tool = None
        for tool in agent.get_all_tools():
            if tool['name'] == 'track_usage':
                usage_tool = tool['function']
                break
        
        result = await usage_tool(10.0, "Test usage", context=context)
        assert result['success'] == True
        assert 'transaction' in result
        
        print("✅ User tools test passed")


@pytest.mark.asyncio
async def test_error_handling(auth_skill, mock_webagents_client):
    """Test error handling scenarios"""
    
    # Test with failed client initialization
    mock_failed_client = Mock()
    mock_failed_client.health_check = AsyncMock(side_effect=Exception("Connection failed"))
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_failed_client):
        agent = BaseAgent(
            name="error-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await asyncio.sleep(0.1)
        
        # Should continue without crashing
        assert agent.skills["auth"] is not None
        
    # Test API key validation failure
    mock_webagents_client.validate_api_key.return_value = AuthResponse(
        success=False,
        error='Invalid API key'
    )
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        auth_skill.client = mock_webagents_client
        
        result = await auth_skill.validate_api_key('invalid_key')
        assert result['success'] == False
        assert 'error' in result
    
    print("✅ Error handling test passed")


@pytest.mark.asyncio
async def test_authorization_scopes(auth_skill, mock_webagents_client):
    """Test authorization scope validation"""
    
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_webagents_client):
        auth_skill.client = mock_webagents_client
        
        # Test admin user authorization
        admin_user = User(id='admin-123', email='admin@webagents.ai', role=UserRole.ADMIN)
        admin_auth = AuthContext(
            user=admin_user,
            authenticated=True,
            scope=AuthScope.ADMIN,
            permissions=['*']
        )
        
        # Admin should be authorized for any tool
        assert auth_skill._is_tool_authorized('any_tool', admin_auth) == True
        
        # Test regular user authorization
        user_auth = AuthContext(
            user=User(id='user-123', email='user@webagents.ai', role=UserRole.USER),
            authenticated=True,
            scope=AuthScope.USER,
            permissions=['read', 'write']
        )
        
        # User should be authorized for basic tools
        assert auth_skill._is_tool_authorized('get_current_user', user_auth) == True
        assert auth_skill._is_tool_authorized('get_user_credits', user_auth) == True
        
        # Test unauthenticated access
        unauth_context = AuthContext(authenticated=False)
        assert auth_skill._is_tool_authorized('get_current_user', unauth_context) == False
        
        print("✅ Authorization scopes test passed")


@pytest.mark.asyncio
async def test_public_endpoints(auth_skill):
    """Test public endpoint detection"""
    
    # Test public endpoints
    context = MockContext()
    context.set('path', '/health')
    assert auth_skill._is_public_endpoint(context) == True
    
    context.set('path', '/docs')
    assert auth_skill._is_public_endpoint(context) == True
    
    context.set('path', '/openapi.json')
    assert auth_skill._is_public_endpoint(context) == True
    
    # Test protected endpoint
    context.set('path', '/api/private')
    assert auth_skill._is_public_endpoint(context) == False
    
    print("✅ Public endpoints test passed")


@pytest.mark.asyncio
async def test_api_key_extraction(auth_skill):
    """Test API key extraction from different sources"""
    
    # Test Bearer token extraction
    context = MockContext()
    context.set('headers', {'Authorization': 'Bearer test_api_key'})
    
    api_key = auth_skill._extract_api_key_from_context(context)
    assert api_key == 'test_api_key'
    
    # Test X-API-Key header extraction
    context = MockContext()
    context.set('headers', {'X-API-Key': 'test_api_key_2'})
    
    api_key = auth_skill._extract_api_key_from_context(context)
    assert api_key == 'test_api_key_2'
    
    # Test query parameter extraction
    context = MockContext()
    context.set('query_params', {'api_key': 'test_api_key_3'})
    
    api_key = auth_skill._extract_api_key_from_context(context)
    assert api_key == 'test_api_key_3'
    
    # Test no API key
    context = MockContext()
    api_key = auth_skill._extract_api_key_from_context(context)
    assert api_key is None
    
    print("✅ API key extraction test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 