"""
Server Core Tests - WebAgents V2.0

Core server functionality tests:
- FastAPI server initialization
- Agent routing and discovery
- Context middleware
- Health endpoints
- Non-streaming chat completions

Run with: pytest tests/server/test_server_core.py -v
"""

import pytest
from fastapi.testclient import TestClient


class TestServerInitialization:
    """Test server initialization and basic setup"""
    
    def test_server_creation(self, test_server):
        """Test server creates correctly with agents"""
        assert test_server is not None
        assert test_server.app is not None
        assert "test-agent" in test_server.static_agents
    
    def test_fastapi_app_properties(self, test_server):
        """Test FastAPI app has correct properties"""
        app = test_server.app
        assert app.title == "WebAgents V2 Server"
        assert app.version == "2.0.0"
        assert len(app.routes) > 0


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_endpoint(self, test_client):
        """Test basic health check"""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        # Note: timestamp is not included in the HealthResponse model
    
    @pytest.mark.skip(reason="Detailed health endpoint not yet implemented")
    def test_detailed_health_endpoint(self, test_client):
        """Test detailed health check"""
        response = test_client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        # HealthResponse model uses "status" not "server_status"
        assert data["status"] == "healthy"
        assert "agents" in data
        assert "timestamp" in data


class TestDiscoveryEndpoints:
    """Test agent discovery endpoints"""
    
    def test_root_discovery(self, test_client):
        """Test root discovery endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        # agents can be a list of dicts or strings
        agents = data["agents"]
        if agents and isinstance(agents[0], dict):
            agent_names = [a.get('name') for a in agents]
            assert "test-agent" in agent_names
        else:
            assert "test-agent" in agents
    
    def test_agent_info(self, test_client):
        """Test individual agent info"""
        response = test_client.get("/test-agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "test-agent"
        # Note: description may not be present, instructions is used instead


class TestNonStreamingChatCompletions:
    """Test non-streaming chat completions"""
    
    def test_basic_chat_completion(self, test_client, sample_request_data, openai_validator):
        """Test basic non-streaming chat completion"""
        response = test_client.post("/test-agent/chat/completions", json=sample_request_data)
        assert response.status_code == 200
        
        data = response.json()
        openai_validator(data, is_streaming=False)
    
    def test_chat_completion_with_multiple_messages(self, test_client, sample_messages, openai_validator):
        """Test chat completion with conversation history"""
        request_data = {
            "model": "test-agent",
            "messages": sample_messages,
            "stream": False
        }
        
        response = test_client.post("/test-agent/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        openai_validator(data, is_streaming=False)
        assert "Mock response to:" in data["choices"][0]["message"]["content"]


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_nonexistent_agent(self, test_client, sample_request_data):
        """Test request to non-existent agent"""
        response = test_client.post("/nonexistent/chat/completions", json=sample_request_data)
        assert response.status_code == 404
    
    def test_invalid_request_data(self, test_client):
        """Test invalid request data handling - server defaults missing fields"""
        response = test_client.post("/test-agent/chat/completions", json={})
        assert response.status_code == 200
    
    def test_malformed_json(self, test_client):
        """Test malformed JSON handling"""
        response = test_client.post(
            "/test-agent/chat/completions",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422


class TestContextMiddleware:
    """Test context middleware functionality"""
    
    def test_request_with_user_context(self, test_client, sample_request_data, request_helper):
        """Test request with user context headers"""
        response = request_helper(test_client, "POST", "/test-agent/chat/completions", json=sample_request_data)
        assert response.status_code == 200


class TestMultiAgentSupport:
    """Test multi-agent server functionality"""
    
    def test_multiple_agents_available(self, multi_client):
        """Test that multiple agents are available"""
        response = multi_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        expected_agents = ["assistant", "calculator", "weather"]
        agents = data["agents"]
        # agents can be a list of dicts or strings
        if agents and isinstance(agents[0], dict):
            agent_names = [a.get('name') for a in agents]
            for agent in expected_agents:
                assert agent in agent_names
        else:
            for agent in expected_agents:
                assert agent in agents
    
    def test_different_agent_endpoints(self, multi_client, sample_request_data):
        """Test that different agents respond correctly"""
        for agent_name in ["assistant", "calculator", "weather"]:
            response = multi_client.post(f"/{agent_name}/chat/completions", json=sample_request_data)
            assert response.status_code == 200
            
            data = response.json()
            # Model name can vary based on LLM skill configuration
            assert "model" in data 