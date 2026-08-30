"""
Integration Tests for Dynamic Agent System - WebAgents V2.0

Tests the dynamic agent system integration with the FastAPI server.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.server.core.app import create_server, WebAgentsServer


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
            "creditsPerToken": {"inputPer1k": "0.001", "outputPer1k": "0.003"},
            "userId": "test-user-123"
        }
    
    @pytest.fixture
    def mock_resolver(self, portal_agent_data):
        """Create a mock async dynamic agent resolver"""
        from webagents.agents.skills.core.transport import CompletionsTransportSkill
        
        async def resolver(agent_name: str):
            if agent_name == portal_agent_data["name"]:
                agent = BaseAgent(
                    name=portal_agent_data["name"],
                    instructions=portal_agent_data["instructions"],
                    scopes=["all"]
                )
                
                mock_response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1699999999,
                    "model": portal_agent_data["name"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello! I'm the integration test agent."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
                
                async def mock_run(messages, stream=False, tools=None):
                    return mock_response
                
                async def mock_run_streaming(messages, tools=None, **kwargs):
                    yield {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1699999999, "model": portal_agent_data["name"], "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
                    yield {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1699999999, "model": portal_agent_data["name"], "choices": [{"index": 0, "delta": {"content": "Hello!"}, "finish_reason": None}]}
                    yield {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1699999999, "model": portal_agent_data["name"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
                
                agent.run = mock_run
                agent.run_streaming = mock_run_streaming
                
                transport_skill = CompletionsTransportSkill()
                agent.skills["completions_transport"] = transport_skill
                await transport_skill.initialize(agent)
                
                if not hasattr(agent, '_registered_http_handlers'):
                    agent._registered_http_handlers = []
                for attr_name in dir(transport_skill):
                    attr = getattr(transport_skill, attr_name, None)
                    if callable(attr) and hasattr(attr, '_http_subpath'):
                        agent._registered_http_handlers.append({
                            'subpath': attr._http_subpath,
                            'method': getattr(attr, '_http_method', 'get'),
                            'function': attr,
                            'scope': getattr(attr, '_http_scope', 'all'),
                            'description': getattr(attr, '_http_description', ''),
                            'source': 'completions_transport'
                        })
                
                return agent
            return None
        return resolver
    
    @pytest.fixture
    def test_server(self, mock_resolver):
        """Create a test server with dynamic agents"""
        server = create_server(
            agents=None,
            dynamic_agents=mock_resolver,
            title="Dynamic Agents Integration Test Server"
        )
        return server
    
    @pytest.fixture
    def test_client(self, test_server):
        """Create a test client.

        `POST /{agent}/chat/completions` runs the model on the OWNER's credit
        and refuses a request that carries no credential at all — for a DYNAMIC
        agent that endpoint is served by the catch-all HTTP dispatch, which
        applies the same floor (`webagents.server.core.app.COMPLETIONS_PATHS`).
        Every real caller presents a credential, so this client does too;
        `test_dynamic_agent_completions_refuses_an_anonymous_caller` below
        asserts the floor itself.
        """
        return TestClient(
            test_server.fastapi_app,
            headers={"Authorization": "Bearer test-service-token"},
        )

    def test_dynamic_agent_completions_refuses_an_anonymous_caller(self, test_server):
        """The dynamic door to the billable endpoint has the same floor as the
        statically registered route. It did not: the static route was guarded
        and this path was not, so every dynamic agent was still an open,
        billable model endpoint for anyone who could reach the port."""
        anonymous = TestClient(test_server.fastapi_app)
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response = anonymous.post(
            "/integration-test-agent/chat/completions", json=request_data
        )
        assert response.status_code == 401, response.text

        # A bare `Bearer` is not a credential either.
        response = anonymous.post(
            "/integration-test-agent/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer"},
        )
        assert response.status_code == 401, response.text

        # Any of the three credential headers clears it.
        for header in ("Authorization", "X-Api-Key", "X-Owner-Assertion"):
            response = anonymous.post(
                "/integration-test-agent/chat/completions",
                json=request_data,
                headers={header: "Bearer test-service-token"},
            )
            assert response.status_code == 200, (header, response.text)

    def test_dynamic_agent_custom_http_handlers_are_not_gated(self, test_server):
        """The floor is scoped to the billable endpoint on purpose: an agent's
        own `@http` handlers may be public by design, and blanket-gating the
        catch-all would break them.

        The probe MUST be a path that actually reaches
        `dynamic_agent_http_dispatch`. `/{agent}/health` does not: it is served
        by the dedicated route registered before the catch-all, and `"health"`
        is in the dispatch's own `reserved_suffixes` where it would 404 — so a
        `/health` probe stays green even if someone widens the floor to gate
        every path in the dispatch, which is the exact regression this test is
        named to prevent. `/capabilities` and `/models` are
        `CompletionsTransportSkill` `@http` handlers with no dedicated route,
        so they are only reachable THROUGH the dispatch.
        """
        anonymous = TestClient(test_server.fastapi_app)
        for subpath in ("capabilities", "models"):
            response = anonymous.get(f"/integration-test-agent/{subpath}")
            assert response.status_code == 200, (subpath, response.text)

        # Guard the guard: these are exactly the paths the dispatch would
        # short-circuit before ever looking for a handler.
        from webagents.server.core.app import COMPLETIONS_PATHS

        assert "capabilities" not in COMPLETIONS_PATHS
        assert "models" not in COMPLETIONS_PATHS
    
    def test_server_stats_with_dynamic_agents(self, test_client):
        """Test server stats endpoint shows dynamic agent information"""
        response = test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "server" in data
        assert "agents" in data
        assert data["agents"]["static_count"] == 0
        
        assert "dynamic_agents" in data
        assert data["dynamic_agents"]["enabled"] is True
    
    def test_dynamic_agent_not_found(self, test_client):
        """Test dynamic agent resolution for non-existent agent"""
        response = test_client.get("/nonexistent-agent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_dynamic_agent_info_endpoint(self, test_client, portal_agent_data):
        """Test dynamic agent info endpoint"""
        response = test_client.get("/integration-test-agent")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert data["name"] == "integration-test-agent"
        assert "instructions" in data
        assert "endpoints" in data
        
        endpoints = data["endpoints"]
        assert "chat_completions" in endpoints
        assert endpoints["chat_completions"] == "/integration-test-agent/chat/completions"
    
    def test_dynamic_agent_chat_completion_non_streaming(self, test_client, portal_agent_data):
        """Test non-streaming chat completion with dynamic agent"""
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
    
    def test_dynamic_agent_with_external_tools(self, test_client, portal_agent_data):
        """Test dynamic agent handling external tools in request"""
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


class TestCustomDynamicResolver:
    """Test server with custom dynamic resolver"""
    
    def _make_mock_agent(self, name: str):
        """Create a properly configured mock agent"""
        agent = Mock(spec=BaseAgent)
        agent.name = name
        agent.instructions = f"I am {name}"
        agent.scopes = ["all"]
        agent.skills = {}
        agent.get_tools_for_scope = Mock(return_value=[])
        agent.get_all_http_handlers = Mock(return_value=[])
        agent.get_all_websocket_handlers = Mock(return_value=[])
        agent.run = AsyncMock(return_value="Custom response")
        agent.list_commands = Mock(return_value=[])
        return agent
    
    def test_server_with_custom_resolver(self):
        """Test server creation with custom dynamic agent resolver"""
        def custom_resolver(agent_name: str):
            if agent_name == "custom-agent":
                agent = Mock(spec=BaseAgent)
                agent.name = "custom-agent"
                agent.instructions = "I am a custom agent"
                agent.scopes = ["all"]
                agent.skills = {}
                agent.get_tools_for_scope = Mock(return_value=[])
                agent.get_all_http_handlers = Mock(return_value=[])
                agent.get_all_websocket_handlers = Mock(return_value=[])
                agent.list_commands = Mock(return_value=[])
                return agent
            return False
        
        server = create_server(dynamic_agents=custom_resolver)
        client = TestClient(server.fastapi_app)
        
        response = client.get("/custom-agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "custom-agent"
    
    def test_server_resolver_precedence(self):
        """Test that custom resolver takes precedence over portal resolver"""
        def custom_resolver(agent_name: str):
            if agent_name == "test-agent":
                agent = Mock(spec=BaseAgent)
                agent.name = "test-agent"
                agent.instructions = "Test"
                return agent
            return False
        
        server = WebAgentsServer(dynamic_agents=custom_resolver)
        
        assert server.dynamic_agents == custom_resolver 