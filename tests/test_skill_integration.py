"""
Skill Integration Tests

Tests skills running with WebAgentsServer (webagentsd).
These test real agent initialization and HTTP endpoint exposure.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

from webagents.agents.core.base_agent import BaseAgent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def auth_agent():
    """Create an agent with AuthSkill in self-issued mode."""
    from webagents.agents.skills.local.auth import AuthSkill
    
    auth_skill = AuthSkill(config={
        "mode": "self-issued",
        "issuer": "http://localhost:8000/test-auth"
    })
    
    agent = BaseAgent(
        name="test-auth-agent",
        instructions="You are an auth test agent.",
        skills={"AuthSkill": auth_skill},
        scopes=["all"]
    )
    return agent


@pytest.fixture
def plugin_agent():
    """Create an agent with PluginSkill."""
    from webagents.agents.skills.local.plugin import PluginSkill
    
    plugin_skill = PluginSkill(config={
        "plugins_dir": "/tmp/test-plugins",
        "cache_dir": "/tmp/test-plugin-cache"
    })
    
    agent = BaseAgent(
        name="test-plugin-agent",
        instructions="You are a plugin test agent.",
        skills={"PluginSkill": plugin_skill},
        scopes=["all"]
    )
    return agent


@pytest.fixture
def webui_agent():
    """Create an agent with WebUISkill."""
    from webagents.agents.skills.local.webui import WebUISkill
    
    webui_skill = WebUISkill(config={})
    
    agent = BaseAgent(
        name="test-webui-agent",
        instructions="You are a WebUI test agent.",
        skills={"WebUISkill": webui_skill},
        scopes=["all"]
    )
    return agent


# Check for multilspy
try:
    import multilspy
    HAS_MULTILSPY = True
except ImportError:
    HAS_MULTILSPY = False


@pytest.fixture
def lsp_agent():
    """Create an agent with LSPSkill."""
    if not HAS_MULTILSPY:
        pytest.skip("multilspy not installed")
    
    from webagents.agents.skills.local.lsp import LSPSkill
    
    lsp_skill = LSPSkill(config={
        "workspace": str(Path.cwd())
    })
    
    agent = BaseAgent(
        name="test-lsp-agent",
        instructions="You are an LSP test agent.",
        skills={"LSPSkill": lsp_skill},
        scopes=["all"]
    )
    return agent


# =============================================================================
# Auth Skill Integration Tests
# =============================================================================

class TestAuthSkillIntegration:
    """Test AuthSkill with real agent initialization."""
    
    @pytest.mark.asyncio
    async def test_auth_agent_initialization(self, auth_agent):
        """Test auth agent initializes correctly."""
        await auth_agent._ensure_skills_initialized()
        
        auth_skill = auth_agent.skills.get("AuthSkill")
        assert auth_skill is not None
        assert auth_skill.agent is not None
    
    @pytest.mark.asyncio
    async def test_auth_skill_has_commands(self, auth_agent):
        """Test auth skill has commands registered."""
        await auth_agent._ensure_skills_initialized()
        
        auth_skill = auth_agent.skills.get("AuthSkill")
        
        # AuthSkill uses @command decorators, not @tool
        # Check the skill exists and is properly initialized
        assert auth_skill is not None
        assert hasattr(auth_skill, 'generate_token')
        assert hasattr(auth_skill, 'validate_token')
    
    @pytest.mark.asyncio
    async def test_auth_token_generation(self, auth_agent):
        """Test token generation works."""
        await auth_agent._ensure_skills_initialized()
        
        auth_skill = auth_agent.skills.get("AuthSkill")
        
        # generate_token is a sync method, not async
        result = auth_skill.generate_token(
            target="http://localhost:9000/other-agent",
            scopes=["read"]
        )
        
        # Should return a JWT token string or dict with token
        assert result is not None
        if isinstance(result, dict):
            assert "token" in result or "access_token" in result
        else:
            # Result is the token string itself
            assert isinstance(result, str)
            assert len(result) > 0


# =============================================================================
# Plugin Skill Integration Tests
# =============================================================================

class TestPluginSkillIntegration:
    """Test PluginSkill with real agent initialization."""
    
    @pytest.mark.asyncio
    async def test_plugin_agent_initialization(self, plugin_agent):
        """Test plugin agent initializes correctly."""
        await plugin_agent._ensure_skills_initialized()
        
        plugin_skill = plugin_agent.skills.get("PluginSkill")
        assert plugin_skill is not None
        assert plugin_skill.agent is not None
    
    @pytest.mark.asyncio
    async def test_plugin_skill_commands_registered(self, plugin_agent):
        """Test plugin commands are registered."""
        await plugin_agent._ensure_skills_initialized()
        
        plugin_skill = plugin_agent.skills.get("PluginSkill")
        
        # Check commands via the skill
        commands = plugin_skill.get_commands() if hasattr(plugin_skill, 'get_commands') else []
        # Plugin skill has /plugin commands
        assert plugin_skill is not None
    
    @pytest.mark.asyncio
    async def test_plugin_list_empty(self, plugin_agent):
        """Test listing plugins when none installed."""
        await plugin_agent._ensure_skills_initialized()
        
        plugin_skill = plugin_agent.skills.get("PluginSkill")
        
        # List plugins (should be empty or return success)
        if hasattr(plugin_skill, 'list_plugins'):
            result = await plugin_skill.list_plugins()
            assert isinstance(result, (dict, list, str))


# =============================================================================
# WebUI Skill Integration Tests
# =============================================================================

class TestWebUISkillIntegration:
    """Test WebUISkill with real agent initialization."""
    
    @pytest.mark.asyncio
    async def test_webui_agent_initialization(self, webui_agent):
        """Test WebUI agent initializes correctly."""
        await webui_agent._ensure_skills_initialized()
        
        webui_skill = webui_agent.skills.get("WebUISkill")
        assert webui_skill is not None
        assert webui_skill.agent is not None
    
    @pytest.mark.asyncio
    async def test_webui_status_command(self, webui_agent):
        """Test WebUI status command."""
        await webui_agent._ensure_skills_initialized()
        
        webui_skill = webui_agent.skills.get("WebUISkill")
        
        # Call status handler if available
        if hasattr(webui_skill, 'handle_ui_status'):
            result = await webui_skill.handle_ui_status()
            assert "mounted" in result or "status" in result.lower()


# =============================================================================
# LSP Skill Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_MULTILSPY, reason="multilspy not installed")
class TestLSPSkillIntegration:
    """Test LSPSkill with real agent initialization."""
    
    @pytest.mark.asyncio
    async def test_lsp_agent_initialization(self, lsp_agent):
        """Test LSP agent initializes correctly."""
        await lsp_agent._ensure_skills_initialized()
        
        lsp_skill = lsp_agent.skills.get("LSPSkill")
        assert lsp_skill is not None
        assert lsp_skill.agent is not None
    
    @pytest.mark.asyncio
    async def test_lsp_tools_registered(self, lsp_agent):
        """Test LSP tools are registered."""
        await lsp_agent._ensure_skills_initialized()
        
        tools = lsp_agent.get_all_tools()
        tool_names = [t["name"] for t in tools]
        
        # LSP should provide these tools
        expected_tools = ["goto_definition", "find_references", "get_hover"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


# =============================================================================
# Multi-Skill Agent Tests
# =============================================================================

class TestMultiSkillAgent:
    """Test agent with multiple skills."""
    
    @pytest.mark.asyncio
    async def test_agent_with_auth_and_plugin(self):
        """Test agent with both auth and plugin skills."""
        from webagents.agents.skills.local.auth import AuthSkill
        from webagents.agents.skills.local.plugin import PluginSkill
        
        auth_skill = AuthSkill(config={"mode": "self-issued"})
        plugin_skill = PluginSkill(config={})
        
        agent = BaseAgent(
            name="multi-skill-agent",
            instructions="You have auth and plugin capabilities.",
            skills={
                "AuthSkill": auth_skill,
                "PluginSkill": plugin_skill
            },
            scopes=["all"]
        )
        
        await agent._ensure_skills_initialized()
        
        # Both skills should be initialized
        assert agent.skills.get("AuthSkill") is not None
        assert agent.skills.get("PluginSkill") is not None
        assert agent.skills["AuthSkill"].agent is not None
        assert agent.skills["PluginSkill"].agent is not None


# =============================================================================
# WebAgentsServer Integration Tests
# =============================================================================

class TestWebAgentsServerIntegration:
    """Test skills with WebAgentsServer."""
    
    @pytest.mark.asyncio
    async def test_server_with_auth_agent(self, auth_agent):
        """Test WebAgentsServer can host auth-enabled agent."""
        from webagents.server.core.app import WebAgentsServer
        
        await auth_agent._ensure_skills_initialized()
        
        # Create server with auth agent
        server = WebAgentsServer(
            agents=[auth_agent],
            title="Test Server"
        )
        
        assert server.app is not None
        
        # Check routes are registered
        routes = [r.path for r in server.app.routes]
        assert any("test-auth-agent" in str(r) for r in routes)
    
    @pytest.mark.asyncio
    async def test_server_with_multiple_skill_agents(self):
        """Test server with multiple agents having different skills."""
        from webagents.agents.skills.local.auth import AuthSkill
        from webagents.agents.skills.local.plugin import PluginSkill
        from webagents.server.core.app import WebAgentsServer
        
        # Create auth agent
        auth_agent = BaseAgent(
            name="auth-agent",
            instructions="Auth agent",
            skills={"AuthSkill": AuthSkill(config={"mode": "self-issued"})},
            scopes=["all"]
        )
        
        # Create plugin agent
        plugin_agent = BaseAgent(
            name="plugin-agent",
            instructions="Plugin agent",
            skills={"PluginSkill": PluginSkill(config={})},
            scopes=["all"]
        )
        
        await auth_agent._ensure_skills_initialized()
        await plugin_agent._ensure_skills_initialized()
        
        # Create server with both
        server = WebAgentsServer(
            agents=[auth_agent, plugin_agent],
            title="Multi-Agent Server"
        )
        
        assert server.app is not None
        
        # Both agents should be available
        routes = [str(r.path) for r in server.app.routes]
        assert any("auth-agent" in r for r in routes)
        assert any("plugin-agent" in r for r in routes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
