"""
Unit Tests for DiscoverySkill

Tests agent discovery, intent search, and capability filtering.
Intent publishing tests are postponed until server implementation 
(requires agent-to-portal handshake).
"""

import pytest
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Import DiscoverySkill and related classes
from webagents.agents.skills.robutler.discovery import (
    DiscoverySkill,
    AgentSearchResult,
    IntentRegistration,
    SearchMode
)
from webagents.agents.core.base_agent import BaseAgent
from robutler.api import RobutlerClient
from robutler.api.types import ApiResponse


class MockAgentContext:
    """Mock agent context for testing DiscoverySkill"""
    def __init__(self, agent=None):
        self.agent = agent or Mock()
        self.agent.name = getattr(self.agent, 'name', 'test-agent')
        self.agent.api_key = getattr(self.agent, 'api_key', None)


class MockResponse:
    """Mock response for API calls"""
    def __init__(self, success=True, data=None, message="OK"):
        self.success = success
        self.data = data or {}
        self.message = message


@pytest.fixture
def mock_webagents_client():
    """Mock RobutlerClient for testing"""
    client = Mock(spec=RobutlerClient)
    client._make_request = AsyncMock()
    client.health_check = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def discovery_config():
    """DiscoverySkill configuration for testing"""
    return {
        'enable_discovery': True,
        'search_mode': 'semantic',
        'max_results': 5,
        'webagents_api_url': 'http://test.localhost',
        'robutler_api_key': 'test_discovery_key',
        'cache_ttl': 300
    }


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = Mock()
    agent.name = 'test-discovery-agent'
    agent.api_key = None  # Will test API key resolution
    return agent


@pytest.fixture
async def discovery_skill_with_client(discovery_config, mock_webagents_client, mock_agent):
    """DiscoverySkill instance with mocked client for testing"""
    with patch('webagents.agents.skills.robutler.discovery.skill.RobutlerClient') as mock_client_class:
        mock_client_class.return_value = mock_webagents_client
        
        # Mock successful health check
        mock_webagents_client.health_check.return_value = MockResponse(success=True)
        
        skill = DiscoverySkill(discovery_config)
        
        # Mock initialize without full agent context setup
        mock_context = MockAgentContext(mock_agent)
        with patch('webagents.utils.logging.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            skill.logger = Mock()  # Set logger directly
            await skill.initialize(mock_context)
        
        yield skill
        
        # Cleanup
        if hasattr(skill, 'cleanup'):
            await skill.cleanup()


@pytest.fixture
async def discovery_skill_no_client(discovery_config, mock_agent):
    """DiscoverySkill instance without client (simulating platform unavailable)"""
    with patch('webagents.agents.skills.robutler.discovery.skill.RobutlerClient') as mock_client_class:
        # Simulate client creation failure
        mock_client_class.side_effect = Exception("Platform unavailable")
        
        skill = DiscoverySkill(discovery_config)
        
        # Mock initialize 
        mock_context = MockAgentContext(mock_agent)
        with patch('webagents.utils.logging.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            skill.logger = Mock()
            await skill.initialize(mock_context)
        
        yield skill


@pytest.fixture
def sample_platform_agents():
    """Sample agent data from platform API"""
    return [
        {
            'agent_id': 'coding-assistant',
            'name': 'Coding Assistant',
            'description': 'Expert programming assistant',
            'intents': ['help with programming', 'debug code'],
            'url': 'http://localhost:8001/coding-assistant',
            'similarity_score': 0.95,
            'capabilities': ['python', 'javascript'],
            'min_balance': 0.0
        },
        {
            'agent_id': 'data-analyst', 
            'name': 'Data Analyst',
            'description': 'Advanced data analysis expert',
            'intents': ['analyze data', 'create charts'],
            'url': 'http://localhost:8002/data-analyst',
            'similarity_score': 0.85,
            'capabilities': ['pandas', 'visualization'],
            'min_balance': 5.0
        }
    ]


# ===== UNIT TESTS =====

class TestDiscoverySkillInitialization:
    """Test DiscoverySkill initialization and configuration"""
    
    def test_discovery_skill_creation(self, discovery_config):
        """Test DiscoverySkill creation with configuration"""
        skill = DiscoverySkill(discovery_config)
        
        assert skill.enable_discovery == True
        assert skill.default_search_mode == SearchMode.SEMANTIC
        assert skill.max_results == 5
        assert skill.webagents_api_url == 'http://test.localhost'
        assert skill.robutler_api_key == 'test_discovery_key'
    
    def test_discovery_skill_default_config(self):
        """Test DiscoverySkill creation with default configuration"""
        skill = DiscoverySkill()
        
        assert skill.enable_discovery == True
        assert skill.default_search_mode == SearchMode.SEMANTIC
        assert skill.max_results == 10
        assert skill.robutler_api_key is None  # No default in config, resolved in initialize
    
    def test_api_key_resolution_hierarchy(self):
        """Test API key resolution hierarchy: config > agent > env > default"""
        
        # Test 1: Config has priority
        skill_with_config = DiscoverySkill({'robutler_api_key': 'config_key'})
        assert skill_with_config.robutler_api_key == 'config_key'
        
        # Test 2: No config key, will be resolved in initialize
        skill_no_config = DiscoverySkill({})
        assert skill_no_config.robutler_api_key is None
    
    def test_base_url_resolution(self):
        """Test base URL resolution: env > config > default"""
        
        # Test with environment variable
        with patch.dict(os.environ, {'ROBUTLER_API_URL': 'http://env.localhost'}):
            skill = DiscoverySkill({'webagents_api_url': 'http://config.localhost'})
            assert skill.webagents_api_url == 'http://env.localhost'  # Environment wins
        
        # Test with config only
        skill = DiscoverySkill({'webagents_api_url': 'http://config.localhost'})
        assert skill.webagents_api_url == 'http://config.localhost'
        
        # Test with defaults
        skill = DiscoverySkill({})
        assert skill.webagents_api_url == 'http://localhost:3000'  # Default
    
    @pytest.mark.asyncio
    async def test_api_key_resolution_in_initialize(self, mock_webagents_client):
        """Test API key resolution during initialize"""
        
        with patch('webagents.agents.skills.robutler.discovery.skill.RobutlerClient') as mock_client_class:
            mock_client_class.return_value = mock_webagents_client
            mock_webagents_client.health_check.return_value = MockResponse(success=True)
            
            # Test agent.api_key resolution
            agent_with_key = Mock()
            agent_with_key.name = 'test-agent'
            agent_with_key.api_key = 'agent_api_key'
            
            skill = DiscoverySkill({})  # No config key
            mock_context = MockAgentContext(agent_with_key)
            
            with patch('webagents.utils.logging.get_logger') as mock_logger:
                mock_logger.return_value = Mock()
                await skill.initialize(mock_context)
            
            assert skill.robutler_api_key == 'agent_api_key'
            
            # Test environment variable resolution
            with patch.dict(os.environ, {'ROBUTLER_API_KEY': 'env_api_key'}):
                agent_no_key = Mock()
                agent_no_key.name = 'test-agent'
                agent_no_key.api_key = None
                
                skill2 = DiscoverySkill({})
                mock_context2 = MockAgentContext(agent_no_key)
                
                with patch('webagents.utils.logging.get_logger') as mock_logger:
                    mock_logger.return_value = Mock()
                    await skill2.initialize(mock_context2)
                
                assert skill2.robutler_api_key == 'env_api_key'


class TestAgentSearchResults:
    """Test AgentSearchResult data structure"""
    
    def test_agent_search_result_creation(self):
        """Test creating AgentSearchResult"""
        result = AgentSearchResult(
            agent_id="test-agent",
            name="Test Agent",
            description="Test description",
            intents=["test intent"],
            url="http://test.com"
        )
        
        assert result.agent_id == "test-agent"
        assert result.name == "Test Agent"
        assert result.similarity_score == 0.0  # Default
        assert result.capabilities == []  # Default empty list


class TestAgentSearch:
    """Test agent search functionality with platform integration"""
    
    @pytest.mark.asyncio
    async def test_search_agents_with_platform_success(self, discovery_skill_with_client, sample_platform_agents):
        """Test agent search with successful platform response"""
        # Mock platform search response
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True,
            data={'agents': sample_platform_agents}
        )
        
        result = await discovery_skill_with_client.search_agents(
            query="help with programming",
            max_results=3,
            context=None
        )
        
        assert result['success'] == True
        assert result['query'] == "help with programming"
        assert result['search_mode'] == 'semantic'
        assert result['total_results'] == 2
        assert len(result['agents']) == 2
        
        # Verify API call was made with correct parameters
        discovery_skill_with_client.client._make_request.assert_called_once_with(
            'GET', '/agents/search',
            params={
                'query': 'help with programming',
                'limit': 3,
                'mode': 'semantic',
                'min_similarity': 0.5
            }
        )
        
        # Check agent data
        agent = result['agents'][0]
        assert agent['agent_id'] == 'coding-assistant'
        assert agent['name'] == 'Coding Assistant'
        assert agent['similarity_score'] == 0.95
    
    @pytest.mark.asyncio
    async def test_search_agents_with_different_modes(self, discovery_skill_with_client):
        """Test search with different search modes"""
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True, data={'agents': []}
        )
        
        # Test semantic search
        await discovery_skill_with_client.search_agents(
            query="programming help",
            search_mode="semantic",
            context=None
        )
        
        call_args = discovery_skill_with_client.client._make_request.call_args
        assert call_args[1]['params']['mode'] == 'semantic'
        
        # Test exact search
        await discovery_skill_with_client.search_agents(
            query="programming help", 
            search_mode="exact",
            context=None
        )
        
        call_args = discovery_skill_with_client.client._make_request.call_args
        assert call_args[1]['params']['mode'] == 'exact'
    
    @pytest.mark.asyncio
    async def test_search_agents_platform_failure(self, discovery_skill_with_client):
        """Test search when platform returns failure"""
        # Mock platform failure
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=False, message="Platform error"
        )
        
        result = await discovery_skill_with_client.search_agents(
            query="programming", 
            context=None
        )
        
        assert result['success'] == False
        assert "Platform search failed" in result['error']
    
    @pytest.mark.asyncio
    async def test_search_agents_no_client(self, discovery_skill_no_client):
        """Test search when client is not available"""
        result = await discovery_skill_no_client.search_agents("test query", context=None)
        
        assert result['success'] == False
        assert 'client not available' in result['error']
    
    @pytest.mark.asyncio
    async def test_search_agents_disabled(self):
        """Test search when discovery is disabled"""
        skill = DiscoverySkill({'enable_discovery': False})
        
        result = await skill.search_agents("test query", context=None)
        
        assert result['success'] == False
        assert 'disabled' in result['error'].lower()


class TestAgentDiscovery:
    """Test agent discovery by capabilities"""
    
    @pytest.mark.asyncio
    async def test_discover_agents_by_capabilities(self, discovery_skill_with_client):
        """Test discovering agents by capabilities"""
        # Mock platform discovery response
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True,
            data={
                'agents': [{
                    'agent_id': 'python-expert',
                    'name': 'Python Expert',
                    'description': 'Python programming specialist',
                    'intents': ['python help'],
                    'url': 'http://localhost:8001',
                    'capabilities': ['python', 'django', 'flask'],
                    'min_balance': 0.0
                }]
            }
        )
        
        result = await discovery_skill_with_client.discover_agents(
            capabilities=["python", "web"],
            max_results=5,
            context=None
        )
        
        assert result['success'] == True
        assert result['capabilities'] == ["python", "web"]
        assert result['total_results'] == 1
        
        # Verify API call was made correctly
        discovery_skill_with_client.client._make_request.assert_called_once_with(
            'GET', '/agents/discover',
            params={
                'capabilities': ["python", "web"],
                'limit': 5
            }
        )
        
        agent = result['agents'][0]
        assert agent['agent_id'] == 'python-expert'
        assert 'python' in agent['capabilities']


class TestSimilarAgents:
    """Test finding similar agents"""
    
    @pytest.mark.asyncio
    async def test_find_similar_agents(self, discovery_skill_with_client):
        """Test finding similar agents to a reference agent"""
        # Mock platform similar agents response
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True,
            data={
                'agents': [{
                    'agent_id': 'similar-agent-1',
                    'name': 'Similar Agent 1',
                    'description': 'Similar functionality',
                    'intents': ['similar intent'],
                    'url': 'http://localhost:8002',
                    'similarity_score': 0.85
                }]
            }
        )
        
        result = await discovery_skill_with_client.find_similar_agents(
            agent_id="reference-agent",
            max_results=3,
            context=None
        )
        
        assert result['success'] == True
        assert result['reference_agent'] == "reference-agent"
        assert result['total_results'] == 1
        
        # Verify API call
        discovery_skill_with_client.client._make_request.assert_called_once_with(
            'GET', '/agents/reference-agent/similar',
            params={'limit': 3}
        )
        
        agent = result['agents'][0]
        assert agent['agent_id'] == 'similar-agent-1'
        assert agent['similarity_score'] == 0.85


class TestIntentPublishing:
    """
    Test intent publishing functionality
    
    Note: Full integration tests postponed until server implementation
    (requires agent-to-portal handshake for authentication)
    """
    
    @pytest.mark.asyncio
    async def test_publish_intents_no_client(self, discovery_skill_no_client):
        """Test intent publishing when client is not available"""
        result = await discovery_skill_no_client.publish_intents(
            intents=["test intent"],
            description="Test description",
            context=None
        )
        
        assert result['success'] == False
        assert 'client not available' in result['error']
    
    @pytest.mark.asyncio
    async def test_publish_intents_with_platform_failure(self, discovery_skill_with_client):
        """Test intent publishing when platform fails (simulating missing handshake)"""
        # Mock platform failure (likely due to missing handshake)
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=False,
            message="Authentication required - agent handshake needed"
        )
        
        result = await discovery_skill_with_client.publish_intents(
            intents=["test intent"],
            description="Test description",
            context=None
        )
        
        assert result['success'] == False
        assert 'authentication' in result['error'].lower() or 'handshake' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_publish_intents_success(self, discovery_skill_with_client):
        """Test successful intent publishing"""
        # Mock successful platform response
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True,
            data={
                'results': [{
                    'intent': 'test intent',
                    'status': 'published',
                    'message': 'Successfully published'
                }]
            }
        )
        
        result = await discovery_skill_with_client.publish_intents(
            intents=["test intent"],
            description="Test description",
            capabilities=["testing"],
            context=None
        )
        
        assert result['success'] == True
        assert result['agent_id'] == "test-discovery-agent"
        assert result['published_intents'] == ["test intent"]
        
        # Verify API call structure
        call_args = discovery_skill_with_client.client._make_request.call_args
        assert call_args[0] == ('POST', '/intents/publish')
        assert 'intents' in call_args[1]['data']
        assert call_args[1]['data']['intents'][0]['intent'] == 'test intent'
    
    @pytest.mark.asyncio
    async def test_get_published_intents(self, discovery_skill_with_client):
        """Test getting published intents for current agent"""
        # Mock platform response
        discovery_skill_with_client.client._make_request.return_value = MockResponse(
            success=True,
            data={
                'intents': [
                    {
                        'intent': 'test intent',
                        'description': 'Test description',
                        'status': 'published'
                    }
                ]
            }
        )
        
        result = await discovery_skill_with_client.get_published_intents(context=None)
        
        assert result['success'] == True
        assert result['agent_id'] == "test-discovery-agent"
        assert len(result['intents']) == 1
        assert result['intents'][0]['intent'] == 'test intent'


class TestDataParsing:
    """Test data parsing and conversion methods"""
    
    def test_parse_agent_result(self, discovery_skill_with_client):
        """Test parsing agent data from platform response"""
        agent_data = {
            'agent_id': 'test-agent',
            'name': 'Test Agent',
            'description': 'Test description',
            'intents': ['test intent'],
            'url': 'http://test.com',
            'similarity_score': 0.95,
            'capabilities': ['python'],
            'min_balance': 5.0
        }
        
        result = discovery_skill_with_client._parse_agent_result(agent_data)
        
        assert isinstance(result, AgentSearchResult)
        assert result.agent_id == 'test-agent'
        assert result.similarity_score == 0.95
        assert result.capabilities == ['python']
        assert result.min_balance == 5.0
    
    def test_agent_result_to_dict(self, discovery_skill_with_client):
        """Test converting AgentSearchResult to dictionary"""
        result = AgentSearchResult(
            agent_id="test-agent",
            name="Test Agent",
            description="Test description",
            intents=["test intent"],
            url="http://test.com",
            similarity_score=0.95,
            capabilities=["python"],
            min_balance=5.0
        )
        
        result_dict = discovery_skill_with_client._agent_result_to_dict(result)
        
        assert result_dict['agent_id'] == 'test-agent'
        assert result_dict['similarity_score'] == 0.95
        assert result_dict['capabilities'] == ['python']
        assert result_dict['min_balance'] == 5.0


class TestErrorHandling:
    """Test error handling in DiscoverySkill"""
    
    @pytest.mark.asyncio
    async def test_search_with_invalid_search_mode(self, discovery_skill_with_client):
        """Test search with invalid search mode"""
        result = await discovery_skill_with_client.search_agents(
            query="test",
            search_mode="invalid_mode",
            context=None
        )
        
        # Should return error response due to invalid search mode
        assert result['success'] == False
        assert 'invalid_mode' in result['error'] or 'not a valid SearchMode' in result['error']
    
    @pytest.mark.asyncio
    async def test_search_with_platform_exception(self, discovery_skill_with_client):
        """Test search when platform raises exception"""
        # Mock platform exception
        discovery_skill_with_client.client._make_request.side_effect = Exception("Connection error")
        
        result = await discovery_skill_with_client.search_agents(
            query="test query",
            context=None
        )
        
        # Should handle exception and return error
        assert result['success'] == False
        assert 'connection error' in result['error'].lower()
    
    @pytest.mark.asyncio 
    async def test_discovery_disabled_error_handling(self):
        """Test error handling when discovery is disabled"""
        skill = DiscoverySkill({'enable_discovery': False})
        
        # All tools should return disabled error
        search_result = await skill.search_agents("test", context=None)
        discover_result = await skill.discover_agents(["test"], context=None)
        similar_result = await skill.find_similar_agents("test", context=None)
        publish_result = await skill.publish_intents(["test"], "test", context=None)
        intents_result = await skill.get_published_intents(context=None)
        
        for result in [search_result, discover_result, similar_result, publish_result, intents_result]:
            assert result['success'] == False
            assert 'disabled' in result['error'].lower()


class TestDiscoverySkillIntegration:
    """Integration tests for DiscoverySkill with BaseAgent"""
    
    @pytest.mark.asyncio
    async def test_discovery_tools_available(self, discovery_skill_with_client):
        """Test that discovery tools are properly decorated and available"""
        
        # Check that tools are decorated correctly
        assert hasattr(discovery_skill_with_client.search_agents, '_webagents_is_tool')
        assert hasattr(discovery_skill_with_client.discover_agents, '_webagents_is_tool')
        assert hasattr(discovery_skill_with_client.find_similar_agents, '_webagents_is_tool')
        assert hasattr(discovery_skill_with_client.publish_intents, '_webagents_is_tool')
        assert hasattr(discovery_skill_with_client.get_published_intents, '_webagents_is_tool')
        
        # Check tool scopes
        assert discovery_skill_with_client.search_agents._tool_scope == 'all'
        assert discovery_skill_with_client.discover_agents._tool_scope == 'all'
        assert discovery_skill_with_client.find_similar_agents._tool_scope == 'all'
        assert discovery_skill_with_client.publish_intents._tool_scope == 'owner'
        assert discovery_skill_with_client.get_published_intents._tool_scope == 'all'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 