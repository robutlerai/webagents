"""
Tests for NLISkill - Natural Language Interface for Agent-to-Agent Communication

Tests NLI functionality including HTTP communication, error handling, and statistics tracking.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from webagents.agents.skills.robutler.nli import NLISkill, NLICommunication, AgentEndpoint


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, name="test-agent"):
        self.name = name
        self.api_key = "test_api_key"


class MockResponse:
    """Mock HTTP response"""
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
    
    def json(self):
        return self._json_data


@pytest.fixture
def nli_skill():
    """NLI skill with default configuration"""
    config = {
        'timeout': 10.0,
        'max_retries': 1,
        'default_authorization': 0.05,
        'max_authorization': 1.0,
        'known_agents': [
            {
                'url': 'http://test-agent:8001',
                'name': 'Test Agent',
                'description': 'Test agent for unit tests',
                'capabilities': ['testing', 'mocking']
            }
        ]
    }
    return NLISkill(config)


@pytest.fixture
async def initialized_nli_skill(nli_skill):
    """Initialized NLI skill with mock agent"""
    mock_agent = MockAgent()
    await nli_skill.initialize(mock_agent)
    yield nli_skill
    await nli_skill.cleanup()


class TestNLISkillInitialization:
    """Test NLI skill initialization and configuration"""
    
    def test_nli_skill_creation(self):
        """Test NLI skill creation with various configurations"""
        # Default configuration
        skill = NLISkill()
        assert skill.default_timeout == 30.0
        assert skill.max_retries == 2
        assert skill.default_authorization == 0.10
        assert skill.max_authorization == 5.00
        
        # Custom configuration
        config = {
            'timeout': 15.0,
            'max_retries': 3,
            'default_authorization': 0.20,
            'max_authorization': 2.50
        }
        skill_custom = NLISkill(config)
        assert skill_custom.default_timeout == 15.0
        assert skill_custom.max_retries == 3
        assert skill_custom.default_authorization == 0.20
        assert skill_custom.max_authorization == 2.50
    
    @pytest.mark.asyncio
    async def test_nli_skill_initialization(self, nli_skill):
        """Test NLI skill initialization with agent"""
        mock_agent = MockAgent()
        
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', True):
            with patch('webagents.agents.skills.robutler.nli.skill.httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                
                await nli_skill.initialize(mock_agent)
                
                assert nli_skill.agent == mock_agent
                assert nli_skill.http_client == mock_client
                assert len(nli_skill.known_agents) == 1
                
                # Check known agent was registered
                agent_key = "test-agent:8001"  # No trailing slash
                assert agent_key in nli_skill.known_agents
                known_agent = nli_skill.known_agents[agent_key]
                assert known_agent.name == "Test Agent"
                assert known_agent.capabilities == ['testing', 'mocking']
                
                await nli_skill.cleanup()
    
    @pytest.mark.asyncio 
    async def test_nli_skill_initialization_no_httpx(self, nli_skill):
        """Test NLI skill initialization when httpx not available"""
        mock_agent = MockAgent()
        
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', False):
            await nli_skill.initialize(mock_agent)
            
            assert nli_skill.agent == mock_agent
            assert nli_skill.http_client is None


class TestAgentEndpointManagement:
    """Test agent endpoint registration and management"""
    
    def test_register_agent_endpoint(self, nli_skill):
        """Test agent endpoint registration"""
        endpoint_key = nli_skill._register_agent_endpoint(
            url="http://localhost:8002/coding-assistant",
            name="Coding Assistant",
            description="AI coding assistant",
            capabilities=['python', 'debugging', 'code-review']
        )
        
        assert endpoint_key == "localhost:8002/coding-assistant"
        assert endpoint_key in nli_skill.known_agents
        
        agent = nli_skill.known_agents[endpoint_key]
        assert agent.url == "http://localhost:8002/coding-assistant"
        assert agent.name == "Coding Assistant"
        assert agent.description == "AI coding assistant"
        assert agent.capabilities == ['python', 'debugging', 'code-review']
        assert agent.success_rate == 1.0
        assert agent.last_contact is None
    
    def test_update_agent_stats(self, nli_skill):
        """Test agent statistics updating"""
        agent_url = "http://localhost:8001/test-agent"
        
        # Register agent first
        nli_skill._register_agent_endpoint(agent_url, name="Test Agent")
        
        # Update with success
        nli_skill._update_agent_stats(agent_url, success=True, duration_ms=150.0)
        
        endpoint_key = "localhost:8001/test-agent"
        agent = nli_skill.known_agents[endpoint_key]
        assert agent.last_contact is not None
        assert agent.success_rate == 1.0  # Still 1.0 since it started at 1.0
        
        # Update with failure
        nli_skill._update_agent_stats(agent_url, success=False, duration_ms=250.0)
        
        # Success rate should decrease slightly (exponential moving average)
        assert agent.success_rate < 1.0


class TestNLICommunication:
    """Test NLI communication functionality"""
    
    @pytest.mark.asyncio
    async def test_successful_nli_communication(self, initialized_nli_skill):
        """Test successful NLI communication"""
        skill = initialized_nli_skill
        
        # Mock successful HTTP response
        mock_response = MockResponse(
            status_code=200,
            json_data={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I received your message and I'm ready to help."
                    }
                }]
            }
        )
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/test-agent",
                message="Hello, can you help me with a task?",
                authorized_amount=0.10
            )
            
            assert result == "Hello! I received your message and I'm ready to help."
            
            # Verify HTTP request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:8001/test-agent/chat/completions"
            
            # Check request payload
            payload = call_args[1]['json']
            assert payload['messages'][0]['content'] == "Hello, can you help me with a task?"
            assert payload['model'] == skill.agent.name
            assert payload['stream'] is False
            
            # Check headers
            headers = call_args[1]['headers']
            assert headers['X-Authorization-Amount'] == '0.1'
            assert headers['X-Origin-Agent'] == skill.agent.name
            
            # Check communication was recorded
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is True
            assert comm.message == "Hello, can you help me with a task?"
            assert comm.response == "Hello! I received your message and I'm ready to help."
    
    @pytest.mark.asyncio
    async def test_nli_communication_url_formatting(self, initialized_nli_skill):
        """Test NLI communication URL formatting"""
        skill = initialized_nli_skill
        
        mock_response = MockResponse(
            status_code=200,
            json_data={"choices": [{"message": {"content": "OK"}}]}
        )
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            # Test URL without trailing slash
            await skill.nli_tool("http://localhost:8001/agent", "test")
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:8001/agent/chat/completions"
            
            mock_post.reset_mock()
            
            # Test URL with trailing slash
            await skill.nli_tool("http://localhost:8001/agent/", "test")
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:8001/agent/chat/completions"
            
            mock_post.reset_mock()
            
            # Test URL already with chat/completions
            await skill.nli_tool("http://localhost:8001/agent/chat/completions", "test")
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:8001/agent/chat/completions"
    
    @pytest.mark.asyncio
    async def test_nli_communication_authorization_limits(self, initialized_nli_skill):
        """Test NLI communication authorization limits"""
        skill = initialized_nli_skill
        
        # Test exceeding maximum authorization
        result = await skill.nli_tool(
            agent_url="http://localhost:8001/test-agent",
            message="Test message",
            authorized_amount=10.0  # Exceeds max_authorization of 1.0
        )
        
        assert "exceeds maximum allowed" in result
        assert len(skill.communication_history) == 0  # No communication should be recorded
    
    @pytest.mark.asyncio
    async def test_nli_communication_http_error(self, initialized_nli_skill):
        """Test NLI communication with HTTP errors"""
        skill = initialized_nli_skill
        
        # Mock HTTP error response
        mock_response = MockResponse(status_code=500, text="Internal Server Error")
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/error-agent",
                message="This should fail"
            )
            
            assert "Failed to communicate" in result
            assert "HTTP 500" in result
            
            # Check failed communication was recorded
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is False
            assert "HTTP 500" in comm.error
    
    @pytest.mark.asyncio
    async def test_nli_communication_timeout(self, initialized_nli_skill):
        """Test NLI communication timeout handling"""
        skill = initialized_nli_skill
        
        from webagents.agents.skills.robutler.nli.skill import httpx
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/slow-agent",
                message="This will timeout",
                timeout=1.0
            )
            
            assert "Failed to communicate" in result
            assert "Request timeout" in result
            
            # Check failed communication was recorded
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is False
            assert "Request timeout" in comm.error
    
    @pytest.mark.asyncio
    async def test_nli_communication_retry_logic(self, initialized_nli_skill):
        """Test NLI communication retry logic"""
        skill = initialized_nli_skill
        skill.max_retries = 2  # Set specific retry count
        
        # Mock first call fails, second succeeds
        mock_responses = [
            MockResponse(status_code=503, text="Service Unavailable"),  # First attempt fails
            MockResponse(  # Second attempt succeeds
                status_code=200,
                json_data={"choices": [{"message": {"content": "Success on retry"}}]}
            )
        ]
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = mock_responses
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/retry-agent",
                message="Test retry logic"
            )
            
            assert result == "Success on retry"
            assert mock_post.call_count == 2  # Should have retried once
            
            # Check successful communication was recorded
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is True
    
    @pytest.mark.asyncio
    async def test_nli_communication_no_httpx(self):
        """Test NLI communication when httpx not available"""
        skill = NLISkill()
        
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', False):
            await skill.initialize(MockAgent())
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/test",
                message="This should fail"
            )
            
            assert "HTTP client not available" in result


class TestNLITools:
    """Test NLI skill tools"""
    
    @pytest.mark.asyncio
    async def test_list_known_agents_tool(self, initialized_nli_skill):
        """Test list_known_agents tool"""
        skill = initialized_nli_skill
        
        # Should have one agent from initialization
        result = await skill.list_known_agents()
        
        assert "Known Agent Endpoints" in result
        assert "Test Agent" in result
        assert "http://test-agent:8001" in result
        assert "testing, mocking" in result
        assert "Never" in result  # Last contact should be "Never"
    
    @pytest.mark.asyncio
    async def test_list_known_agents_empty(self):
        """Test list_known_agents tool with no agents"""
        skill = NLISkill()
        await skill.initialize(MockAgent())
        
        result = await skill.list_known_agents()
        assert "No known agent endpoints" in result
        
        await skill.cleanup()
    
    @pytest.mark.asyncio
    async def test_show_communication_history_tool(self, initialized_nli_skill):
        """Test show_communication_history tool"""
        skill = initialized_nli_skill
        
        # Add some mock communications
        skill.communication_history = [
            NLICommunication(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                target_agent_url="http://localhost:8001/agent1",
                message="Hello agent 1",
                response="Hello back!",
                cost_usd=0.05,
                duration_ms=150.0,
                success=True
            ),
            NLICommunication(
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                target_agent_url="http://localhost:8002/agent2",
                message="Help me with a task",
                response="",
                cost_usd=0.0,
                duration_ms=5000.0,
                success=False,
                error="Timeout"
            )
        ]
        
        result = await skill.show_communication_history(limit=5)
        
        assert "Recent NLI Communications" in result
        assert "agent1" in result
        assert "agent2" in result
        assert "Hello agent 1" in result
        assert "Hello back!" in result
        assert "Timeout" in result
        assert "Summary Statistics" in result
        assert "Total Communications: 2" in result
        assert "Success Rate: 50.0%" in result
    
    @pytest.mark.asyncio
    async def test_show_communication_history_empty(self, initialized_nli_skill):
        """Test show_communication_history tool with no history"""
        skill = initialized_nli_skill
        
        result = await skill.show_communication_history()
        assert "No NLI communications recorded" in result
    
    @pytest.mark.asyncio
    async def test_register_agent_tool(self, initialized_nli_skill):
        """Test register_agent tool"""
        skill = initialized_nli_skill
        
        result = await skill.register_agent(
            agent_url="http://localhost:8003/new-agent",
            name="New Agent",
            description="A newly registered agent",
            capabilities="python,data-analysis,visualization"
        )
        
        assert "Registered agent: New Agent" in result
        assert "http://localhost:8003/new-agent" in result
        assert "python, data-analysis, visualization" in result
        
        # Check agent was actually registered
        endpoint_key = "localhost:8003/new-agent"
        assert endpoint_key in skill.known_agents
        
        agent = skill.known_agents[endpoint_key]
        assert agent.name == "New Agent"
        assert agent.capabilities == ["python", "data-analysis", "visualization"]


class TestNLIStatistics:
    """Test NLI statistics and tracking"""
    
    def test_get_statistics(self, nli_skill):
        """Test get_statistics method"""
        # Add mock communications
        nli_skill.communication_history = [
            NLICommunication(
                timestamp=datetime.now(),
                target_agent_url="http://test1",
                message="msg1",
                response="resp1",
                cost_usd=0.10,
                duration_ms=100.0,
                success=True
            ),
            NLICommunication(
                timestamp=datetime.now(),
                target_agent_url="http://test2",
                message="msg2",
                response="",
                cost_usd=0.0,
                duration_ms=200.0,
                success=False
            )
        ]
        
        nli_skill.known_agents["test1"] = AgentEndpoint(url="http://test1")
        nli_skill.known_agents["test2"] = AgentEndpoint(url="http://test2")
        
        stats = nli_skill.get_statistics()
        
        assert stats['total_communications'] == 2
        assert stats['successful_communications'] == 1
        assert stats['success_rate'] == 0.5
        assert stats['total_cost_usd'] == 0.10
        assert stats['average_duration_ms'] == 150.0
        assert stats['known_agents'] == 2
        assert 'httpx_available' in stats
    
    def test_get_statistics_empty(self, nli_skill):
        """Test get_statistics with no data"""
        stats = nli_skill.get_statistics()
        
        assert stats['total_communications'] == 0
        assert stats['successful_communications'] == 0
        assert stats['success_rate'] == 0
        assert stats['total_cost_usd'] == 0
        assert stats['average_duration_ms'] == 0
        assert stats['known_agents'] == 0


class TestNLIEdgeCases:
    """Test edge cases and error scenarios"""
    
    @pytest.mark.asyncio
    async def test_cleanup_with_no_client(self):
        """Test cleanup when no HTTP client exists"""
        skill = NLISkill()
        # Should not raise any exception
        await skill.cleanup()
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, initialized_nli_skill):
        """Test handling of malformed API responses"""
        skill = initialized_nli_skill
        
        # Mock response with missing expected fields
        mock_response = MockResponse(
            status_code=200,
            json_data={"unexpected": "format"}
        )
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/malformed",
                message="Test malformed response"
            )
            
            # Should still succeed but return the raw response
            assert "unexpected" in result or "format" in result
            
            # Communication should still be recorded as successful
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is True
    
    @pytest.mark.asyncio
    async def test_json_decode_error(self, initialized_nli_skill):
        """Test handling of JSON decode errors"""
        skill = initialized_nli_skill
        
        # Mock response that will raise JSON decode error
        mock_response = MockResponse(status_code=200, text="Invalid JSON")
        mock_response.json = Mock(side_effect=ValueError("Invalid JSON"))
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(
                agent_url="http://localhost:8001/invalid-json",
                message="Test invalid JSON"
            )
            
            assert "Failed to communicate" in result 