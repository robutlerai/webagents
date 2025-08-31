"""
Test suite for Dynamic Agent System - WebAgents V2.0

Tests the DynamicAgentFactory and server integration.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from webagents.agents.core.dynamic_factory import DynamicAgentFactory, get_dynamic_agent_factory
from webagents.agents.core.base_agent import BaseAgent
from webagents.server.core.app import WebAgentsServer, create_server


class TestDynamicAgentFactory:
    """Test suite for DynamicAgentFactory"""
    
    @pytest.fixture
    def factory(self):
        """Create a factory instance for testing"""
        return DynamicAgentFactory(
            portal_base_url="https://test-portal.com",
            service_token="test-token-123",
            enable_caching=True,
            cache_ttl=300
        )
    
    @pytest.fixture
    def sample_agent_data(self):
        """Sample agent data from portal API"""
        return {
            "id": "agent-123",
            "name": "test-agent",
            "instructions": "You are a test agent",
            "model": "gpt-4o-mini",
            "api_key": "sk-test123",
            "intents": ["testing", "demo"],
            "canTalkToOtherAgents": True,
            "minimumBalance": "0.01",
            "creditsPerToken": "0.000001"
        }
    
    def test_factory_initialization(self):
        """Test factory initialization with different configurations"""
        factory = DynamicAgentFactory(
            portal_base_url="https://custom-portal.com",
            service_token="custom-token",
            enable_caching=False,
            cache_ttl=600
        )
        assert factory.portal_base_url == "https://custom-portal.com"
        assert factory.service_token == "custom-token"
        assert factory.enable_caching is False
        assert factory.cache_ttl == 600
    
    def test_cache_validation(self, factory):
        """Test cache validation logic"""
        current_time = time.time()
        assert factory._is_cache_valid(current_time - 100) is True  # Valid
        assert factory._is_cache_valid(current_time - 400) is False  # Expired
        
        # Test disabled cache
        factory.enable_caching = False
        assert factory._is_cache_valid(current_time) is False
    
    def test_cache_cleanup(self, factory):
        """Test cache cleanup functionality"""
        current_time = time.time()
        
        # Add cache entries
        factory.agent_data_cache = {
            "agent1": {"data": {"name": "agent1"}, "cached_at": current_time - 100},  # Valid
            "agent2": {"data": {"name": "agent2"}, "cached_at": current_time - 400},  # Expired
        }
        
        factory.agent_cache = {
            "agent1": {"agent": Mock(), "created_at": current_time - 100, "data": {}},  # Valid
            "agent3": {"agent": Mock(), "created_at": current_time - 500, "data": {}},  # Expired
        }
        
        # Run cleanup
        factory._cleanup_expired_cache()
        
        # Check results
        assert "agent1" in factory.agent_data_cache
        assert "agent2" not in factory.agent_data_cache
        assert "agent1" in factory.agent_cache
        assert "agent3" not in factory.agent_cache
    
    @pytest.mark.asyncio
    async def test_get_agent_data_success(self, factory, sample_agent_data):
        """Test successful agent data fetching from portal"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful API responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"agents": [sample_agent_data]}
            
            mock_api_key_response = Mock()
            mock_api_key_response.status_code = 200
            mock_api_key_response.json.return_value = {"apiKey": "sk-test123"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [mock_response, mock_api_key_response]
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Test fetching agent data
            result = await factory.get_agent_data("test-agent")
            
            # Verify result
            assert result is not None
            assert result["name"] == "test-agent"
            assert result["api_key"] == "sk-test123"
            
            # Verify API calls
            assert mock_client_instance.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_agent_data_not_found(self, factory):
        """Test agent data fetching when agent not found"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"agents": []}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            result = await factory.get_agent_data("nonexistent-agent")
            assert result is None
    
    def test_create_agent_skills(self, factory, sample_agent_data):
        """Test agent skills creation"""
        skills = factory._create_agent_skills(sample_agent_data)
        
        # Should have primary LLM skill
        assert "primary_llm" in skills
        
        # Should have platform skills if API key available
        assert "payment" in skills
        assert "discovery" in skills  # canTalkToOtherAgents is True
        assert "nli" in skills
    
    def test_create_agent_from_data(self, factory, sample_agent_data):
        """Test BaseAgent creation from portal data"""
        agent = factory.create_agent_from_data(sample_agent_data)
        
        # Verify agent properties
        assert isinstance(agent, BaseAgent)
        assert agent.name == "test-agent"
        assert agent.api_key == "sk-test123"
        assert agent.intents == ["testing", "demo"]
        assert agent.can_use_other_agents is True
        assert agent._portal_data == sample_agent_data
        assert agent._dynamic_agent is True
        
        # Verify instructions contain system prompt
        assert "test-agent" in agent.instructions
        assert "webagents_discovery" in agent.instructions  # Has discovery tool
        
        # Verify skills were created
        assert len(agent.skills) > 0
    
    @pytest.mark.asyncio
    async def test_get_or_create_agent_success(self, factory, sample_agent_data):
        """Test successful agent creation through full pipeline"""
        with patch.object(factory, 'get_agent_data', return_value=sample_agent_data):
            agent = await factory.get_or_create_agent("test-agent")
            
            assert agent is not None
            assert isinstance(agent, BaseAgent)
            assert agent.name == "test-agent"
            
            # Verify agent was cached
            assert "test-agent" in factory.agent_cache
    
    def test_get_cache_stats(self, factory):
        """Test cache statistics reporting"""
        # Add some cache entries
        factory.agent_data_cache["test1"] = {"data": {}, "cached_at": time.time()}
        factory.agent_cache["test2"] = {"agent": Mock(), "created_at": time.time(), "data": {}}
        
        stats = factory.get_cache_stats()
        
        assert stats["caching_enabled"] is True
        assert stats["cache_ttl"] == 300
        assert stats["agent_data_cache_size"] == 1
        assert stats["agent_instance_cache_size"] == 1
        assert stats["service_token_configured"] is True


class TestServerDynamicAgentIntegration:
    """Test suite for server integration with dynamic agents"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock BaseAgent for testing"""
        agent = Mock(spec=BaseAgent)
        agent.name = "test-agent"
        agent.api_key = "sk-test123"
        agent._dynamic_agent = True
        return agent
    
    @pytest.fixture
    def custom_resolver(self, mock_agent):
        """Custom dynamic agent resolver for testing"""
        def resolver(agent_name: str):
            if agent_name == "test-agent":
                return mock_agent
            return False
        return resolver
    
    def test_server_creation_with_custom_resolver(self, custom_resolver):
        """Test server creation with custom dynamic agent resolver"""
        server = WebAgentsServer(dynamic_agents=custom_resolver)
        
        assert server.dynamic_agents == custom_resolver
        assert not hasattr(server, 'dynamic_agent_factory')  # Should not create factory
    
    def test_server_creation_with_portal_resolver(self):
        """Test server creation with default portal-based resolver"""
        with patch('webagents.server.core.app.get_dynamic_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_get_factory.return_value = mock_factory
            
            server = WebAgentsServer()
            
            assert server.dynamic_agents is not None
            assert hasattr(server, 'dynamic_agent_factory')
            assert server.dynamic_agent_factory == mock_factory
    
    @pytest.mark.asyncio
    async def test_resolve_agent_static_first(self):
        """Test that static agents are resolved first"""
        static_agent = Mock(spec=BaseAgent)
        static_agent.name = "static-agent"
        
        server = WebAgentsServer(agents=[static_agent])
        
        # Should resolve static agent
        result = await server._resolve_agent("static-agent", is_dynamic=True)
        assert result == static_agent
    
    @pytest.mark.asyncio
    async def test_resolve_agent_dynamic_sync(self, custom_resolver, mock_agent):
        """Test dynamic agent resolution with sync resolver"""
        server = WebAgentsServer(dynamic_agents=custom_resolver)
        
        result = await server._resolve_agent("test-agent", is_dynamic=True)
        assert result == mock_agent
    
    @pytest.mark.asyncio
    async def test_resolve_agent_dynamic_async(self, mock_agent):
        """Test dynamic agent resolution with async resolver"""
        async def async_resolver(agent_name: str):
            if agent_name == "test-agent":
                return mock_agent
            return False
        
        server = WebAgentsServer(dynamic_agents=async_resolver)
        
        result = await server._resolve_agent("test-agent", is_dynamic=True)
        assert result == mock_agent
    
    def test_create_server_factory_function(self):
        """Test create_server factory function with dynamic agents"""
        # Test with custom resolver
        def custom_resolver(name):
            return Mock(spec=BaseAgent) if name == "test" else False
        
        server = create_server(dynamic_agents=custom_resolver)
        assert server.dynamic_agents == custom_resolver
        
        # Test with default (portal-based) resolver
        with patch('webagents.server.core.app.get_dynamic_agent_factory'):
            server = create_server()
            assert server.dynamic_agents is not None


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_factory_without_service_token(self):
        """Test factory behavior without service token"""
        factory = DynamicAgentFactory(service_token=None)
        assert factory.service_token is None
    
    def test_reserved_agent_names(self):
        """Test that reserved names are properly excluded"""
        factory = DynamicAgentFactory()
        
        # Test reserved names
        reserved_names = ["health", "agents", "ready", "docs", "openapi.json", "favicon.ico", "metrics", "status"]
        
        for name in reserved_names:
            assert name.lower() in factory.reserved_names
    
    @pytest.mark.asyncio
    async def test_get_agent_data_reserved_name(self):
        """Test that reserved names return None without API calls"""
        factory = DynamicAgentFactory(service_token="test")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            result = await factory.get_agent_data("health")
            
            # Should return None without making API calls
            assert result is None
            assert mock_client_instance.get.call_count == 0
    
    def test_agent_creation_with_minimal_data(self):
        """Test agent creation with minimal portal data"""
        factory = DynamicAgentFactory()
        minimal_data = {
            "name": "minimal-agent",
            "instructions": "Basic agent"
        }
        
        agent = factory.create_agent_from_data(minimal_data)
        
        assert agent is not None
        assert agent.name == "minimal-agent"
        assert agent.api_key is None
        assert agent.intents == []
        assert agent.can_use_other_agents is False 