"""
AuthSkill Unit Tests - WebAgents V2.0

Tests for authentication and authorization functionality.
AuthSkill now provides:
- on_connection hook for request authentication
- API key extraction from context.request.headers
- API key validation via RobutlerClient
- Owner assertion JWT verification
- Service token authentication
"""

import pytest
try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch, AsyncMock

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.robutler.auth import AuthSkill, AuthScope, AuthContext, AuthenticationError, AuthorizationError
from webagents.server.context.context_vars import Context, create_context
from robutler.api.types import User, UserRole, AuthResponse, ApiResponse


def _make_request(headers=None, query_params=None):
    """Create a mock request object with headers and query_params."""
    return SimpleNamespace(
        headers=headers or {},
        query_params=query_params or {},
    )


@pytest.fixture
def mock_robutler_client():
    """Create mock RobutlerClient for testing"""
    client = Mock()

    client.health_check = AsyncMock(return_value=ApiResponse(
        success=True,
        data={'status': 'healthy', 'service': 'webagents-platform'}
    ))

    test_user = User(
        id='test-user-123',
        email='test@webagents.ai',
        name='Test User',
        role=UserRole.USER,
    )

    client.validate_api_key = AsyncMock(return_value=AuthResponse(
        success=True,
        user=test_user
    ))

    client.close = AsyncMock()

    return client


@pytest.fixture
def auth_skill():
    """Create AuthSkill with test config"""
    config = {
        'require_auth': True,
        'platform_api_url': 'http://localhost:3000',
        'api_key': 'rok_testapikey',
        'cache_ttl': 300,
    }
    return AuthSkill(config)


@pytest.mark.asyncio
async def test_auth_skill_initialization(auth_skill, mock_robutler_client):
    """Test AuthSkill initialization with RobutlerClient"""
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="auth-test-agent",
            instructions="Test agent for auth functionality.",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()

        assert auth_skill.require_auth is True
        assert auth_skill.platform_api_url == 'http://localhost:3000'
        assert auth_skill.api_key == 'rok_testapikey'
        assert auth_skill.client is not None

        mock_robutler_client.health_check.assert_called_once()


@pytest.mark.asyncio
async def test_auth_skill_with_agent(auth_skill, mock_robutler_client):
    """Test AuthSkill integration with BaseAgent"""
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="auth-integration-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )

        assert "auth" in agent.skills
        assert isinstance(agent.skills["auth"], AuthSkill)

        connection_hooks = agent.get_all_hooks("on_connection")
        auth_hooks = [h for h in connection_hooks if h.get('source') == 'auth']
        assert len(auth_hooks) > 0, "Auth on_connection hook should be registered"


@pytest.mark.asyncio
async def test_api_key_authentication(auth_skill, mock_robutler_client):
    """Test API key authentication via _authenticate_api_key"""
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="auth-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()

        auth_context = await auth_skill._authenticate_api_key('rok_testapikey')

        assert auth_context is not None
        assert auth_context.authenticated is True
        assert auth_context.user_id == 'test-user-123'
        mock_robutler_client.validate_api_key.assert_called_with('rok_testapikey')


@pytest.mark.asyncio
async def test_request_authentication_hook(auth_skill, mock_robutler_client):
    """Test on_connection hook sets context.auth on success"""
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="auth-hook-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()

        context = create_context(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            request=_make_request(headers={'Authorization': 'Bearer rok_testapikey'}),
        )

        result_context = await auth_skill.validate_request_auth(context)

        assert result_context.auth is not None
        assert result_context.auth.authenticated is True
        assert result_context.auth.user_id == 'test-user-123'


@pytest.mark.asyncio
async def test_error_handling(auth_skill, mock_robutler_client):
    """Test error handling scenarios"""
    mock_failed_client = Mock()
    mock_failed_client.health_check = AsyncMock(side_effect=Exception("Connection failed"))

    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_failed_client):
        agent = BaseAgent(
            name="error-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()
        assert agent.skills["auth"] is not None

    # Test API key validation failure
    mock_robutler_client.validate_api_key.return_value = AuthResponse(
        success=False,
        error='Invalid API key'
    )

    auth_skill.client = mock_robutler_client

    result = await auth_skill._authenticate_api_key('invalid_key')
    assert result is None


@pytest.mark.asyncio
async def test_authentication_raises_on_missing_key(auth_skill, mock_robutler_client):
    """Test that validate_request_auth raises AuthenticationError when no key is provided"""
    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="no-key-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()

        context = create_context(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            request=_make_request(),
        )

        with pytest.raises(AuthenticationError):
            await auth_skill.validate_request_auth(context)


@pytest.mark.asyncio
async def test_auth_scope_from_admin_user(auth_skill, mock_robutler_client):
    """Test that admin users get ADMIN scope"""
    admin_user = User(id='admin-123', email='admin@webagents.ai', role=UserRole.ADMIN)
    mock_robutler_client.validate_api_key.return_value = AuthResponse(
        success=True,
        user=admin_user
    )

    with patch('webagents.agents.skills.robutler.auth.skill.RobutlerClient', return_value=mock_robutler_client):
        agent = BaseAgent(
            name="admin-scope-test",
            instructions="Test agent",
            skills={"auth": auth_skill}
        )
        await agent._ensure_skills_initialized()

        auth_context = await auth_skill._authenticate_api_key('rok_adminkey')

        assert auth_context is not None
        assert auth_context.scope == AuthScope.ADMIN


@pytest.mark.asyncio
async def test_auth_context_dataclass():
    """Test AuthContext defaults and field access"""
    ctx = AuthContext()
    assert ctx.authenticated is False
    assert ctx.scope == AuthScope.USER
    assert ctx.user_id is None
    assert ctx.agent_id is None
    assert ctx.assertion is None

    ctx = AuthContext(user_id='u1', authenticated=True, scope=AuthScope.OWNER)
    assert ctx.user_id == 'u1'
    assert ctx.scope == AuthScope.OWNER


@pytest.mark.asyncio
async def test_api_key_extraction(auth_skill):
    """Test API key extraction from different sources"""
    # Bearer token
    context = create_context(
        messages=[], stream=False,
        request=_make_request(headers={'Authorization': 'Bearer test_api_key'}),
    )
    assert auth_skill._extract_api_key_from_context(context) == 'test_api_key'

    # X-API-Key header
    context = create_context(
        messages=[], stream=False,
        request=_make_request(headers={'X-API-Key': 'test_api_key_2'}),
    )
    assert auth_skill._extract_api_key_from_context(context) == 'test_api_key_2'

    # Query parameter
    context = create_context(
        messages=[], stream=False,
        request=_make_request(query_params={'api_key': 'test_api_key_3'}),
    )
    assert auth_skill._extract_api_key_from_context(context) == 'test_api_key_3'

    # No key
    context = create_context(
        messages=[], stream=False,
        request=_make_request(),
    )
    assert auth_skill._extract_api_key_from_context(context) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
