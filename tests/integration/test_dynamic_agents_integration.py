"""
Integration Tests for Dynamic Agent System - WebAgents V2.0

Tests the dynamic agent system integration with the FastAPI server.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from webagents.server.core.app import create_server, WebAgentsServer
from webagents.agents.core.dynamic_factory import DynamicAgentFactory


class TestDynamicAgentsServerIntegration:
    """Integration tests for dynamic agents with FastAPI server"""
    
    @pytest.fixture
    def portal_agent_data(self):
        """Sample portal agent configuration"""
        return {
            "id": "test-agent-id-123",
            "name": "integration-test-agent",
            "instructions": "You are a helpful integration test agent.",
            "model": "gpt-4o-mini",
            "api_key": "sk-test-integration-key",
            "intents": ["testing", "integration"],
            "canTalkToOtherAgents": True,
            "minimumBalance": "0.01",
            "creditsPerToken": "0.000001",
            "userId": "test-user-123"
        }
    
    @pytest.fixture
    def test_server(self):
        """Create a test server with dynamic agents"""
        server = create_server(
            agents=None,  # No static agents
            dynamic_agents=None,  # Use portal-based resolver
            title="Dynamic Agents Integration Test Server"
        )
        return server
    
    @pytest.fixture
    def test_client(self, test_server):
        """Create a test client"""
        return TestClient(test_server.fastapi_app)
    
    def test_server_stats_with_dynamic_agents(self, test_client):
        """Test server stats endpoint shows dynamic agent information"""
        response = test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify server info
        assert "server_info" in data
        server_info = data["server_info"]
        
        assert "static_agents" in server_info
        assert server_info["static_agents"] == 0  # No static agents
        
        assert "dynamic_agents_enabled" in server_info
        assert server_info["dynamic_agents_enabled"] is True
        
        # Verify dynamic agent factory stats
        assert "dynamic_agent_factory" in data
        factory_stats = data["dynamic_agent_factory"]
        
        assert "caching_enabled" in factory_stats
        assert "cache_ttl" in factory_stats
    
    def test_dynamic_agent_not_found(self, test_client):
        """Test dynamic agent resolution for non-existent agent"""
        # Mock empty agents response
        with patch('httpx.AsyncClient') as mock_client:
            empty_response = Mock()
            empty_response.status_code = 200
            empty_response.json.return_value = {"agents": []}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = empty_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            response = test_client.get("/nonexistent-agent")
            
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert "not found" in data["detail"].lower()
    
    def test_dynamic_agent_info_endpoint(self, test_client, portal_agent_data):
        """Test dynamic agent info endpoint"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful API responses
            agents_response = Mock()
            agents_response.status_code = 200
            agents_response.json.return_value = {"agents": [portal_agent_data]}
            
            api_key_response = Mock()
            api_key_response.status_code = 200
            api_key_response.json.return_value = {"apiKey": "sk-test-integration-key"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [agents_response, api_key_response]
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Request agent info
            response = test_client.get("/integration-test-agent")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "agent" in data
            assert data["agent"] == "integration-test-agent"
            assert "description" in data
            assert "agent_data" in data
            assert "endpoints" in data
            
            # Verify endpoints
            endpoints = data["endpoints"]
            assert "control" in endpoints
            assert "info" in endpoints
            assert "chat" in endpoints
            assert endpoints["chat"] == "/integration-test-agent/chat/completions"
    
    def test_dynamic_agent_chat_completion_non_streaming(self, test_client, portal_agent_data):
        """Test non-streaming chat completion with dynamic agent"""
        mock_response = "Hello! I'm the integration test agent."
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock portal API responses
            agents_response = Mock()
            agents_response.status_code = 200
            agents_response.json.return_value = {"agents": [portal_agent_data]}
            
            api_key_response = Mock()
            api_key_response.status_code = 200
            api_key_response.json.return_value = {"apiKey": "sk-test-integration-key"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [agents_response, api_key_response]
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Mock BaseAgent response
            with patch('webagents.agents.core.base_agent.BaseAgent.run') as mock_run:
                mock_run.return_value = mock_response
                
                # Send chat completion request
                request_data = {
                    "model": "integration-test-agent",
                    "messages": [
                        {"role": "user", "content": "Hello, can you help me?"}
                    ],
                    "stream": False
                }
                
                response = test_client.post(
                    "/integration-test-agent/chat/completions",
                    json=request_data
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify OpenAI-compatible response structure
                assert "id" in data
                assert "object" in data
                assert "created" in data
                assert "model" in data
                assert "choices" in data
                
                # Verify response content
                choices = data["choices"]
                assert len(choices) == 1
                assert choices[0]["index"] == 0
                assert choices[0]["message"]["role"] == "assistant"
                assert choices[0]["message"]["content"] == mock_response
                assert choices[0]["finish_reason"] == "stop"
    
    def test_dynamic_agent_with_external_tools(self, test_client, portal_agent_data):
        """Test dynamic agent handling external tools in request"""
        # Mock BaseAgent to return tool calls
        mock_tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}'
                }
            }
        ]
        
        mock_response = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": mock_tool_calls
                },
                "finish_reason": "tool_calls"
            }],
            "_external_tools_only": True
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock portal API responses
            agents_response = Mock()
            agents_response.status_code = 200
            agents_response.json.return_value = {"agents": [portal_agent_data]}
            
            api_key_response = Mock()
            api_key_response.status_code = 200
            api_key_response.json.return_value = {"apiKey": "sk-test-integration-key"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [agents_response, api_key_response]
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Mock BaseAgent response
            with patch('webagents.agents.core.base_agent.BaseAgent.run') as mock_run:
                mock_run.return_value = mock_response
                
                # Request with external tools
                request_data = {
                    "model": "integration-test-agent",
                    "messages": [
                        {"role": "user", "content": "What's the weather in San Francisco?"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather for a location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"}
                                    }
                                }
                            }
                        }
                    ],
                    "stream": False
                }
                
                response = test_client.post(
                    "/integration-test-agent/chat/completions",
                    json=request_data
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify tool calls in response
                assert "choices" in data
                choice = data["choices"][0]
                assert "message" in choice
                message = choice["message"]
                assert "tool_calls" in message
                assert len(message["tool_calls"]) == 1
                
                tool_call = message["tool_calls"][0]
                assert tool_call["function"]["name"] == "get_weather"
                assert "San Francisco" in tool_call["function"]["arguments"]


class TestCustomDynamicResolver:
    """Test server with custom dynamic resolver"""
    
    def test_server_with_custom_resolver(self):
        """Test server creation with custom dynamic agent resolver"""
        def custom_resolver(agent_name: str):
            if agent_name == "custom-agent":
                # Return a mock agent
                from unittest.mock import Mock
                from webagents.agents.core.base_agent import BaseAgent
                
                agent = Mock(spec=BaseAgent)
                agent.name = "custom-agent"
                agent.run.return_value = "Custom response"
                return agent
            return False
        
        # Create server with custom resolver
        server = create_server(dynamic_agents=custom_resolver)
        client = TestClient(server.fastapi_app)
        
        # Test custom agent info
        response = client.get("/custom-agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent"] == "custom-agent"
    
    def test_server_resolver_precedence(self):
        """Test that custom resolver takes precedence over portal resolver"""
        def custom_resolver(agent_name: str):
            if agent_name == "test-agent":
                from unittest.mock import Mock
                from webagents.agents.core.base_agent import BaseAgent
                
                agent = Mock(spec=BaseAgent)
                agent.name = "test-agent"
                return agent
            return False
        
        # Create server with custom resolver (should not create portal factory)
        server = WebAgentsServer(dynamic_agents=custom_resolver)
        
        # Verify custom resolver is used
        assert server.dynamic_agents == custom_resolver
        
        # Verify portal factory is not created
        assert not hasattr(server, 'dynamic_agent_factory') 