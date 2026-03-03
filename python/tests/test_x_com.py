"""
Test suite for ultra-minimal X.com skill (3 tools only)
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from webagents.agents.skills.ecosystem.x_com.skill import XComSkill


@pytest.fixture
def minimal_x_com_skill():
    """Create minimal XComSkill instance for testing"""
    with patch.dict('os.environ', {
        'X_API_KEY': 'test_key',
        'X_API_SECRET': 'test_secret',
        'AGENTS_BASE_URL': 'http://localhost:2224'
    }):
        skill = XComSkill()
        
        # Create proper mock agent
        class MockAgent:
            def __init__(self):
                self.name = 'test-agent'
                self.skills = {
                    'notifications': MockNotificationSkill(),
                    'kv': MockKVSkill(),
                    'auth': MockAuthSkill()
                }
                self._x_tokens = {}
                self._temp_x_tokens = {}
        
        skill.agent = MockAgent()
        skill.logger = MagicMock()
        return skill


class MockNotificationSkill:
    """Mock notification skill"""
    async def send_notification(self, title, body, tag=None, type=None, priority=None):
        return "✅ Notification queued"


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


class TestMinimalXComSkill:
    """Test cases for ultra-minimal X.com skill"""

    def test_skill_initialization(self, minimal_x_com_skill):
        """Test skill initializes correctly"""
        assert minimal_x_com_skill.api_key == 'test_key'
        assert minimal_x_com_skill.api_secret == 'test_secret'
        assert 'test-agent' == minimal_x_com_skill.agent.name

    def test_prompt_mentions_3_tools(self, minimal_x_com_skill):
        """Test prompt mentions all 3 minimal tools"""
        prompt = minimal_x_com_skill.x_com_prompt()
        assert 'x_subscribe()' in prompt
        assert 'x_post()' in prompt
        assert 'x_manage()' in prompt
        assert '3 tools' in prompt
        assert 'Ultra-minimal' in prompt

    def test_has_exactly_3_tools(self, minimal_x_com_skill):
        """Test skill has exactly 3 x_ prefixed tools"""
        x_tools = [m for m in dir(minimal_x_com_skill) if m.startswith('x_') and not m.startswith('x_com')]
        assert len(x_tools) == 3
        assert 'x_subscribe' in x_tools
        assert 'x_post' in x_tools
        assert 'x_manage' in x_tools

    @pytest.mark.asyncio
    async def test_x_subscribe_no_auth_provides_url(self, minimal_x_com_skill, mock_context):
        """Test x_subscribe provides auth URL when not authenticated"""
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            # Mock no existing tokens
            with patch.object(minimal_x_com_skill, '_load_user_tokens', return_value=None):
                # Mock request token generation
                mock_tokens = {
                    'oauth_token': 'request_token',
                    'oauth_token_secret': 'request_secret',
                    'oauth_callback_confirmed': 'true'
                }
                with patch.object(minimal_x_com_skill, '_get_request_token', return_value=mock_tokens):
                    result = await minimal_x_com_skill.x_subscribe('elonmusk', 'SpaceX launches')
                    
                    assert '🔗 First, authorize X.com access:' in result
                    assert 'api.x.com/oauth/authorize' in result
                    assert 'elonmusk' in result

    @pytest.mark.asyncio
    async def test_x_subscribe_with_auth_success(self, minimal_x_com_skill, mock_context):
        """Test successful subscription when authenticated"""
        user_tokens = {
            'oauth_token': 'access_token',
            'oauth_token_secret': 'access_secret'
        }
        
        mock_user_response = {
            'data': {
                'id': '12345',
                'name': 'Elon Musk',
                'username': 'elonmusk'
            }
        }
        
        mock_webhook_response = {
            'webhook_id': 'webhook123'
        }
        
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_user_tokens', return_value=user_tokens):
                with patch.object(minimal_x_com_skill, '_make_authenticated_request', return_value=mock_user_response):
                    with patch.object(minimal_x_com_skill, '_register_webhook_with_x', return_value=mock_webhook_response):
                        result = await minimal_x_com_skill.x_subscribe('elonmusk', 'SpaceX launches only')
                        
                        assert '✅ Subscribed to @elonmusk (Elon Musk)!' in result
                        assert 'SpaceX launches only' in result

    @pytest.mark.asyncio
    async def test_x_subscribe_user_not_found(self, minimal_x_com_skill, mock_context):
        """Test subscription fails when user not found"""
        user_tokens = {
            'oauth_token': 'access_token',
            'oauth_token_secret': 'access_secret'
        }
        
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_user_tokens', return_value=user_tokens):
                with patch.object(minimal_x_com_skill, '_make_authenticated_request', return_value={'data': None}):
                    result = await minimal_x_com_skill.x_subscribe('nonexistentuser')
                    
                    assert '❌ User @nonexistentuser not found on X.com' in result

    @pytest.mark.asyncio
    async def test_x_post_no_auth_provides_url(self, minimal_x_com_skill, mock_context):
        """Test x_post provides auth URL when not authenticated"""
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_user_tokens', return_value=None):
                mock_tokens = {
                    'oauth_token': 'request_token',
                    'oauth_token_secret': 'request_secret',
                    'oauth_callback_confirmed': 'true'
                }
                with patch.object(minimal_x_com_skill, '_get_request_token', return_value=mock_tokens):
                    result = await minimal_x_com_skill.x_post('Hello world!')
                    
                    assert '🔗 First, authorize X.com access:' in result
                    assert 'post your tweet' in result

    @pytest.mark.asyncio
    async def test_x_post_success(self, minimal_x_com_skill, mock_context):
        """Test successful tweet posting"""
        user_tokens = {
            'oauth_token': 'access_token',
            'oauth_token_secret': 'access_secret'
        }
        
        mock_response = {
            'data': {
                'id': 'tweet123'
            }
        }
        
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_user_tokens', return_value=user_tokens):
                with patch.object(minimal_x_com_skill, '_make_authenticated_request', return_value=mock_response):
                    result = await minimal_x_com_skill.x_post('Hello world!')
                    
                    assert '✅ Tweet posted! ID: tweet123' in result

    @pytest.mark.asyncio
    async def test_x_manage_list_empty(self, minimal_x_com_skill, mock_context):
        """Test x_manage list with no subscriptions"""
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_subscriptions', return_value={'users': {}}):
                result = await minimal_x_com_skill.x_manage('list')
                
                assert '📭 No subscriptions yet' in result
                assert 'x_subscribe(username, instructions)' in result

    @pytest.mark.asyncio
    async def test_x_manage_list_with_subscriptions(self, minimal_x_com_skill, mock_context):
        """Test x_manage list with existing subscriptions"""
        subscriptions = {
            'users': {
                'elonmusk': {
                    'display_name': 'Elon Musk',
                    'instructions': 'SpaceX launches only'
                },
                'openai': {
                    'display_name': 'OpenAI',
                    'instructions': 'AI model updates'
                }
            }
        }
        
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_subscriptions', return_value=subscriptions):
                result = await minimal_x_com_skill.x_manage('list')
                
                assert '📋 Your X.com Subscriptions:' in result
                assert '@elonmusk (Elon Musk)' in result
                assert 'SpaceX launches only' in result
                assert '@openai (OpenAI)' in result
                assert 'AI model updates' in result

    @pytest.mark.asyncio
    async def test_x_manage_unsubscribe_success(self, minimal_x_com_skill, mock_context):
        """Test successful unsubscribe"""
        subscriptions = {
            'users': {
                'elonmusk': {
                    'display_name': 'Elon Musk',
                    'instructions': 'SpaceX launches only'
                }
            }
        }
        
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_subscriptions', return_value=subscriptions):
                with patch.object(minimal_x_com_skill, '_save_subscriptions') as mock_save:
                    result = await minimal_x_com_skill.x_manage('unsubscribe', 'elonmusk')
                    
                    assert '✅ Unsubscribed from @elonmusk (Elon Musk)' in result
                    mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_x_manage_unsubscribe_not_found(self, minimal_x_com_skill, mock_context):
        """Test unsubscribe from non-existent user"""
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            with patch.object(minimal_x_com_skill, '_load_subscriptions', return_value={'users': {}}):
                result = await minimal_x_com_skill.x_manage('unsubscribe', 'nonexistent')
                
                assert '❌ Not subscribed to @nonexistent' in result

    @pytest.mark.asyncio
    async def test_x_manage_invalid_action(self, minimal_x_com_skill, mock_context):
        """Test x_manage with invalid action"""
        with patch('webagents.agents.skills.ecosystem.x_com.skill.get_context', return_value=mock_context):
            result = await minimal_x_com_skill.x_manage('invalid_action')
            
            assert '❌ Invalid action. Use \'list\' or \'unsubscribe\'' in result

    @pytest.mark.asyncio
    async def test_notification_integration(self, minimal_x_com_skill):
        """Test notification skill integration"""
        await minimal_x_com_skill._send_post_notification(
            'Test tweet content',
            'testuser',
            'tweet123',
            'owner123'
        )
        
        # Should not raise exception - notification skill is mocked

    @pytest.mark.asyncio
    async def test_subscription_storage(self, minimal_x_com_skill):
        """Test subscription storage via KV skill"""
        subscriptions = {
            'users': {'testuser': {'active': True}},
            'webhook_active': True
        }
        
        # Test save
        await minimal_x_com_skill._save_subscriptions('owner123', subscriptions)
        
        # Test load
        loaded = await minimal_x_com_skill._load_subscriptions('owner123')
        assert loaded['users']['testuser']['active'] == True
        assert loaded['webhook_active'] == True

    @pytest.mark.asyncio
    async def test_webhook_signature_verification(self, minimal_x_com_skill):
        """Test webhook signature verification method exists"""
        # Just test the method exists and doesn't crash
        result = await minimal_x_com_skill._verify_webhook_signature('sig', '123', 'body')
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_relevance_checking_with_instructions(self, minimal_x_com_skill):
        """Test post relevance checking with instructions"""
        # Test with instructions
        is_relevant = await minimal_x_com_skill._check_post_relevance(
            'SpaceX launched Starship today!',
            'SpaceX launches and space news',
            'spacex',
            'owner123'
        )
        # Should return boolean
        assert isinstance(is_relevant, bool)
        
        # Test without instructions (should default to relevant)
        is_relevant_no_instructions = await minimal_x_com_skill._check_post_relevance(
            'Random tweet',
            '',
            'user',
            'owner123'
        )
        assert is_relevant_no_instructions == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
