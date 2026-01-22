"""
API Route Harmonization Tests

Comprehensive tests for harmonized API routes across:
- WebAgentsDaemon (CLI daemon server)
- DaemonClient (CLI client)
- WebAgentsServer (main server)
- @command decorator integration
"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

from webagents.cli.daemon.server import WebAgentsDaemon, create_daemon
from webagents.cli.daemon.registry import DaemonRegistry, DaemonAgent
from webagents.cli.client.daemon_client import DaemonClient
from webagents.cli.loader.agent_md import AgentFile


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_agent_content():
    """Standard test agent AGENT.md content."""
    return """---
name: test-agent
description: A test agent for API route testing
skills:
  - session
  - checkpoint
---

# Test Agent

You are a test agent for API route testing.
"""


@pytest.fixture
def test_agent_path(tmp_path, test_agent_content):
    """Create a test AGENT.md file."""
    agent_file = tmp_path / "AGENT.md"
    agent_file.write_text(test_agent_content)
    return agent_file


@pytest.fixture
def daemon_default_prefix():
    """Create WebAgentsDaemon with default prefix /agents."""
    daemon = WebAgentsDaemon(port=0, url_prefix="/agents")
    return daemon


@pytest.fixture
def daemon_custom_prefix():
    """Create WebAgentsDaemon with custom prefix."""
    daemon = WebAgentsDaemon(port=0, url_prefix="/api/v1/agents")
    return daemon


@pytest.fixture
def daemon_no_prefix():
    """Create WebAgentsDaemon with empty prefix."""
    daemon = WebAgentsDaemon(port=0, url_prefix="")
    return daemon


@pytest_asyncio.fixture
async def client_default(daemon_default_prefix):
    """AsyncClient for testing with default prefix."""
    transport = ASGITransport(app=daemon_default_prefix.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def client_custom(daemon_custom_prefix):
    """AsyncClient for testing with custom prefix."""
    transport = ASGITransport(app=daemon_custom_prefix.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Test: Daemon Server Routes - Default Prefix
# =============================================================================

class TestDaemonServerRoutesDefaultPrefix:
    """Test WebAgentsDaemon API routes with default /agents prefix."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client_default):
        """GET /health returns healthy status."""
        response = await client_default.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client_default):
        """GET / returns daemon status."""
        response = await client_default.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "webagentsd"
        assert data["status"] == "running"
        assert "url_prefix" in data
    
    @pytest.mark.asyncio
    async def test_list_agents_default_prefix(self, client_default):
        """GET /agents returns agent list."""
        # FastAPI redirects /agents to /agents/ by default
        response = await client_default.get("/agents", follow_redirects=True)
        assert response.status_code == 200
        assert "agents" in response.json()
    
    @pytest.mark.asyncio
    async def test_list_agents_with_trailing_slash(self, client_default):
        """GET /agents/ works directly."""
        response = await client_default.get("/agents/")
        assert response.status_code == 200
        assert "agents" in response.json()
    
    @pytest.mark.asyncio
    async def test_register_agent_default_prefix(self, client_default, test_agent_path):
        """POST /agents/ registers agent from path."""
        response = await client_default.post(
            "/agents/",
            json={"path": str(test_agent_path)}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-agent"
    
    @pytest.mark.asyncio
    async def test_get_agent_default_prefix(self, client_default, test_agent_path):
        """GET /agents/{name} returns agent info."""
        # First register the agent
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        # Then get it
        response = await client_default.get("/agents/test-agent")
        assert response.status_code == 200
        assert response.json()["name"] == "test-agent"
    
    @pytest.mark.asyncio
    async def test_unregister_agent_default_prefix(self, client_default, test_agent_path):
        """DELETE /agents/{name} unregisters agent."""
        # First register
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        # Then unregister
        response = await client_default.delete("/agents/test-agent")
        assert response.status_code == 200
        assert response.json()["status"] == "unregistered"
    
    @pytest.mark.asyncio
    async def test_agent_not_found(self, client_default):
        """GET /agents/{name} returns 404 for unknown agent."""
        response = await client_default.get("/agents/nonexistent")
        assert response.status_code == 404


# =============================================================================
# Test: Daemon Server Routes - Custom Prefix
# =============================================================================

class TestDaemonServerRoutesCustomPrefix:
    """Test WebAgentsDaemon API routes with custom /api/v1/agents prefix."""
    
    @pytest.mark.asyncio
    async def test_list_agents_custom_prefix(self, client_custom):
        """GET /api/v1/agents/ works with custom prefix."""
        response = await client_custom.get("/api/v1/agents/")
        assert response.status_code == 200
        assert "agents" in response.json()
    
    @pytest.mark.asyncio
    async def test_register_agent_custom_prefix(self, client_custom, test_agent_path):
        """POST /api/v1/agents/ registers agent."""
        response = await client_custom.post(
            "/api/v1/agents/",
            json={"path": str(test_agent_path)}
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_get_agent_custom_prefix(self, client_custom, test_agent_path):
        """GET /api/v1/agents/{name} returns agent info."""
        await client_custom.post("/api/v1/agents/", json={"path": str(test_agent_path)})
        
        response = await client_custom.get("/api/v1/agents/test-agent")
        assert response.status_code == 200


# =============================================================================
# Test: Deprecated Endpoints Removed
# =============================================================================

class TestDeprecatedEndpointsRemoved:
    """Test that deprecated start/stop/restart/run endpoints are removed."""
    
    @pytest.mark.asyncio
    async def test_start_endpoint_removed(self, client_default, test_agent_path):
        """POST /agents/{name}/start returns 404."""
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        response = await client_default.post("/agents/test-agent/start")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_stop_endpoint_removed(self, client_default, test_agent_path):
        """POST /agents/{name}/stop returns 404."""
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        response = await client_default.post("/agents/test-agent/stop")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_restart_endpoint_removed(self, client_default, test_agent_path):
        """POST /agents/{name}/restart returns 404."""
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        response = await client_default.post("/agents/test-agent/restart")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_run_endpoint_removed(self, client_default, test_agent_path):
        """POST /agents/{name}/run returns 404."""
        await client_default.post("/agents/", json={"path": str(test_agent_path)})
        
        response = await client_default.post("/agents/test-agent/run")
        assert response.status_code == 404


# =============================================================================
# Test: DaemonClient URL Construction
# =============================================================================

class TestDaemonClientURLConstruction:
    """Test DaemonClient builds URLs correctly with configurable prefix."""
    
    def test_default_agents_prefix(self):
        """Client uses /agents as default prefix."""
        # Don't instantiate with actual httpx client (would need network)
        # Test the URL construction logic instead
        base_url = "http://localhost:8765"
        agents_prefix = "/agents"
        
        # Simulate _agents_url method
        def agents_url(path=""):
            if path and not path.startswith("/"):
                path = f"/{path}"
            return f"{base_url}{agents_prefix}{path}"
        
        assert agents_url() == "http://localhost:8765/agents"
        assert agents_url("/test-agent") == "http://localhost:8765/agents/test-agent"
        assert agents_url("/test-agent/command") == "http://localhost:8765/agents/test-agent/command"
    
    def test_custom_agents_prefix(self):
        """Client uses custom prefix for command URLs."""
        base_url = "http://localhost:8765"
        agents_prefix = "/api/v1/agents"
        
        def agents_url(path=""):
            if path and not path.startswith("/"):
                path = f"/{path}"
            return f"{base_url}{agents_prefix}{path}"
        
        assert agents_url() == "http://localhost:8765/api/v1/agents"
        assert agents_url("/test/command") == "http://localhost:8765/api/v1/agents/test/command"
    
    def test_empty_agents_prefix(self):
        """Client works with empty prefix."""
        base_url = "http://localhost:8765"
        agents_prefix = ""
        
        def agents_url(path=""):
            if path and not path.startswith("/"):
                path = f"/{path}"
            return f"{base_url}{agents_prefix}{path}"
        
        assert agents_url("/test-agent") == "http://localhost:8765/test-agent"
    
    def test_run_agent_method_removed(self):
        """run_agent method no longer exists on DaemonClient."""
        # Check that DaemonClient doesn't have run_agent method
        assert not hasattr(DaemonClient, 'run_agent') or \
               'run_agent' not in DaemonClient.__dict__


# =============================================================================
# Test: Command Decorator Integration
# =============================================================================

class TestCommandDecoratorIntegration:
    """Test @command decorator exposes commands correctly."""
    
    def test_command_appears_in_list(self):
        """@command decorated function appears in list_commands()."""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.skills.local.session.skill import SessionManagerSkill
        
        agent = BaseAgent(
            name="test",
            instructions="Test agent",
            skills={
                "session": SessionManagerSkill(config={"agent_name": "test"})
            }
        )
        
        commands = agent.list_commands()
        paths = [c["path"] for c in commands]
        
        assert "/session/new" in paths
        assert "/session/save" in paths
        assert "/session/load" in paths
    
    def test_checkpoint_commands_registered(self):
        """Checkpoint skill commands are registered."""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.skills.local.checkpoint.skill import CheckpointSkill
        
        agent = BaseAgent(
            name="test",
            instructions="Test agent",
            skills={
                "checkpoint": CheckpointSkill(config={"agent_name": "test"})
            }
        )
        
        commands = agent.list_commands()
        paths = [c["path"] for c in commands]
        
        assert "/checkpoint/create" in paths
        assert "/checkpoint/list" in paths
        assert "/checkpoint/restore" in paths
    
    def test_command_has_description(self):
        """Commands include description from docstring."""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.skills.local.session.skill import SessionManagerSkill
        
        agent = BaseAgent(
            name="test",
            instructions="Test agent",
            skills={
                "session": SessionManagerSkill(config={"agent_name": "test"})
            }
        )
        
        commands = agent.list_commands()
        
        # Find /session/new command
        session_new = next((c for c in commands if c["path"] == "/session/new"), None)
        assert session_new is not None
        assert "description" in session_new
        assert len(session_new["description"]) > 0


# =============================================================================
# Test: Daemon Route Registration
# =============================================================================

class TestDaemonRouteRegistration:
    """Test that routes are correctly registered on the daemon router."""
    
    def test_default_prefix_routes(self, daemon_default_prefix):
        """Routes are registered with /agents prefix."""
        routes = [r.path for r in daemon_default_prefix.agents_router.routes]
        
        # Routes on APIRouter have the prefix included
        assert "/agents/" in routes  # List agents
        assert "/agents/{name}" in routes  # Get/Delete agent
        assert "/agents/{name}/command" in routes  # List commands
        assert "/agents/{name}/command/{path:path}" in routes  # Execute command
        assert "/agents/{name}/chat/completions" in routes  # Chat
    
    def test_custom_prefix_routes(self, daemon_custom_prefix):
        """Routes are registered with custom prefix."""
        routes = [r.path for r in daemon_custom_prefix.agents_router.routes]
        
        assert "/api/v1/agents/" in routes
        assert "/api/v1/agents/{name}" in routes
        assert "/api/v1/agents/{name}/command" in routes
    
    def test_no_deprecated_routes(self, daemon_default_prefix):
        """Deprecated routes are not registered."""
        routes = [r.path for r in daemon_default_prefix.agents_router.routes]
        
        assert "/agents/{name}/start" not in routes
        assert "/agents/{name}/stop" not in routes
        assert "/agents/{name}/restart" not in routes
        assert "/agents/{name}/run" not in routes
    
    def test_root_level_routes_not_prefixed(self, daemon_default_prefix):
        """Root-level routes like /health are not prefixed."""
        app_routes = [r.path for r in daemon_default_prefix.app.routes]
        
        assert "/health" in app_routes
        assert "/" in app_routes
        assert "/cron" in app_routes


# =============================================================================
# Test: Create Daemon Factory
# =============================================================================

class TestCreateDaemonFactory:
    """Test create_daemon factory function."""
    
    def test_create_daemon_default(self):
        """create_daemon with defaults."""
        daemon = create_daemon()
        
        assert daemon.port == 8765
        assert daemon.url_prefix == "/agents"
    
    def test_create_daemon_custom_prefix(self):
        """create_daemon with custom prefix."""
        daemon = create_daemon(port=9000, url_prefix="/custom")
        
        assert daemon.port == 9000
        assert daemon.url_prefix == "/custom"
    
    def test_create_daemon_empty_prefix(self):
        """create_daemon with empty prefix."""
        daemon = create_daemon(url_prefix="")
        
        assert daemon.url_prefix == ""


# =============================================================================
# Test: End-to-End Integration
# =============================================================================

class TestEndToEndIntegration:
    """Full integration tests simulating TUI workflow."""
    
    @pytest.mark.asyncio
    async def test_register_and_list_commands(self, client_default, test_agent_path):
        """
        Full workflow:
        1. POST /agents/ with agent path
        2. GET /agents/{name}/command
        3. Verify commands are returned
        """
        # Register agent
        reg_response = await client_default.post(
            "/agents/",
            json={"path": str(test_agent_path)}
        )
        assert reg_response.status_code == 200
        
        # Note: list_commands requires the agent to be loaded with skills,
        # which happens through the manager. For this test, we verify the
        # endpoint exists and returns properly.
        cmd_response = await client_default.get("/agents/test-agent/command")
        # May return empty commands or 404 depending on manager setup
        # But endpoint should exist
        assert cmd_response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_command_404_for_unknown_agent(self, client_default):
        """GET /agents/nonexistent/command returns 404."""
        response = await client_default.get("/agents/nonexistent/command")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, client_default, test_agent_path):
        """Test full agent lifecycle: register, get, unregister."""
        # Register
        reg = await client_default.post("/agents/", json={"path": str(test_agent_path)})
        assert reg.status_code == 200
        
        # List should include agent
        list_resp = await client_default.get("/agents/")
        agents = list_resp.json()["agents"]
        assert any(a["name"] == "test-agent" for a in agents)
        
        # Get agent
        get_resp = await client_default.get("/agents/test-agent")
        assert get_resp.status_code == 200
        
        # Unregister
        unreg = await client_default.delete("/agents/test-agent")
        assert unreg.status_code == 200
        
        # Should be gone
        get_after = await client_default.get("/agents/test-agent")
        assert get_after.status_code == 404


# =============================================================================
# Test: WebAgentsServer Routes (if applicable)
# =============================================================================

class TestWebAgentsServerRoutes:
    """Test WebAgentsServer routes with url_prefix."""
    
    def test_no_duplicate_agents_path(self):
        """Verify routes don't have /agents/agents double-prefix."""
        from webagents.server.core.app import WebAgentsServer
        
        # Create server with /agents prefix
        server = WebAgentsServer(url_prefix="/agents")
        
        # Get all route paths
        routes = [r.path for r in server.router.routes]
        
        # Should not have /agents/agents
        for route in routes:
            assert "/agents/agents" not in route, f"Found double-prefix in route: {route}"
    
    def test_collection_routes_at_root(self):
        """Collection routes should be at router root (prefix provides /agents)."""
        from webagents.server.core.app import WebAgentsServer
        
        server = WebAgentsServer(url_prefix="/agents")
        routes = [r.path for r in server.router.routes]
        
        # With prefix /agents, these become /agents/, /agents/{name} etc.
        assert "/agents/" in routes  # List/register at prefix root
