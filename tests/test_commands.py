"""
Tests for the hierarchical command system.
"""

import pytest
from typing import Dict, Any


class TestCommandDecorator:
    """Tests for the @command decorator"""
    
    def test_command_decorator_basic(self):
        """Test basic command decoration"""
        from webagents.agents.tools.decorators import command
        
        @command("/test/action", description="Test action")
        async def test_action(param: str = "") -> Dict[str, Any]:
            return {"param": param}
        
        assert hasattr(test_action, '_webagents_is_command')
        assert test_action._webagents_is_command is True
        assert test_action._command_path == "/test/action"
        assert test_action._command_description == "Test action"
        assert test_action._command_scope == "all"
    
    def test_command_decorator_with_alias(self):
        """Test command with alias"""
        from webagents.agents.tools.decorators import command
        
        @command("/checkpoint/create", alias="/checkpoint", description="Create checkpoint")
        async def create_checkpoint() -> Dict[str, Any]:
            return {"status": "created"}
        
        assert create_checkpoint._command_path == "/checkpoint/create"
        assert create_checkpoint._command_alias == "/checkpoint"
    
    def test_command_decorator_with_scope(self):
        """Test command with restricted scope"""
        from webagents.agents.tools.decorators import command
        
        @command("/admin/reset", scope="owner")
        async def reset() -> Dict[str, Any]:
            return {"status": "reset"}
        
        assert reset._command_scope == "owner"
    
    def test_command_decorator_default_path(self):
        """Test command with default path from function name"""
        from webagents.agents.tools.decorators import command
        
        @command
        async def my_command() -> Dict[str, Any]:
            return {}
        
        assert my_command._command_path == "/my_command"


class TestBaseAgentCommands:
    """Tests for BaseAgent command registration and execution"""
    
    def test_register_command(self):
        """Test command registration"""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import command
        
        agent = BaseAgent(name="test-agent")
        
        @command("/test/hello", description="Say hello")
        async def hello(name: str = "World") -> Dict[str, Any]:
            return {"message": f"Hello, {name}!"}
        
        agent.register_command(hello, source="test")
        
        commands = agent.list_commands()
        assert len(commands) == 1
        assert commands[0]["path"] == "/test/hello"
        assert commands[0]["description"] == "Say hello"
    
    def test_list_commands_with_scope_filter(self):
        """Test listing commands with scope filter"""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import command
        
        agent = BaseAgent(name="test-agent")
        
        @command("/public/action", scope="all")
        async def public_action() -> Dict[str, Any]:
            return {}
        
        @command("/owner/action", scope="owner")
        async def owner_action() -> Dict[str, Any]:
            return {}
        
        agent.register_command(public_action, source="test")
        agent.register_command(owner_action, source="test")
        
        # All commands
        all_commands = agent.list_commands()
        assert len(all_commands) == 2
        
        # Filter by scope
        owner_commands = agent.list_commands(scope="owner")
        assert len(owner_commands) == 2  # Both included (all + owner)
    
    def test_get_command(self):
        """Test getting command by path"""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import command
        
        agent = BaseAgent(name="test-agent")
        
        @command("/test/action", description="Test action")
        async def test_action() -> Dict[str, Any]:
            return {}
        
        agent.register_command(test_action, source="test")
        
        cmd = agent.get_command("/test/action")
        assert cmd is not None
        assert cmd["path"] == "/test/action"
        
        # Non-existent command
        assert agent.get_command("/nonexistent") is None
    
    def test_get_command_by_alias(self):
        """Test getting command by alias"""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import command
        
        agent = BaseAgent(name="test-agent")
        
        @command("/checkpoint/create", alias="/checkpoint")
        async def create_checkpoint() -> Dict[str, Any]:
            return {}
        
        agent.register_command(create_checkpoint, source="test")
        
        # Get by alias
        cmd = agent.get_command("/checkpoint")
        assert cmd is not None
        assert cmd["path"] == "/checkpoint/create"
    
    @pytest.mark.asyncio
    async def test_execute_command(self):
        """Test command execution"""
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.tools.decorators import command
        
        agent = BaseAgent(name="test-agent")
        
        @command("/test/greet")
        async def greet(name: str = "World") -> Dict[str, Any]:
            return {"greeting": f"Hello, {name}!"}
        
        agent.register_command(greet, source="test")
        
        result = await agent.execute_command("/test/greet", {"name": "Test"})
        assert result["greeting"] == "Hello, Test!"
    
    @pytest.mark.asyncio
    async def test_execute_command_not_found(self):
        """Test executing non-existent command"""
        from webagents.agents.core.base_agent import BaseAgent
        
        agent = BaseAgent(name="test-agent")
        
        with pytest.raises(ValueError, match="Command not found"):
            await agent.execute_command("/nonexistent", {})


class TestSlashCommandRegistry:
    """Tests for SlashCommandRegistry"""
    
    def test_register_agent_commands(self):
        """Test registering commands from agent"""
        from webagents.cli.repl.slash_commands import SlashCommandRegistry
        
        registry = SlashCommandRegistry()
        
        commands = [
            {
                "path": "/checkpoint/create",
                "alias": "/checkpoint",
                "description": "Create checkpoint",
                "scope": "owner",
                "parameters": {},
                "required": [],
            },
            {
                "path": "/session/save",
                "alias": None,
                "description": "Save session",
                "scope": "all",
                "parameters": {},
                "required": [],
            },
        ]
        
        registry.register_agent_commands(commands)
        
        # Check registration
        assert "checkpoint/create" in registry.agent_commands
        assert "session/save" in registry.agent_commands
    
    def test_get_subcommands(self):
        """Test getting subcommands for a prefix"""
        from webagents.cli.repl.slash_commands import SlashCommandRegistry
        
        registry = SlashCommandRegistry()
        
        commands = [
            {"path": "/checkpoint/create", "description": "Create"},
            {"path": "/checkpoint/restore", "description": "Restore"},
            {"path": "/checkpoint/list", "description": "List"},
            {"path": "/session/save", "description": "Save"},
        ]
        
        registry.register_agent_commands(commands)
        
        subcommands = registry.get_subcommands("checkpoint")
        assert len(subcommands) == 3
        
        paths = [sc["path"] for sc in subcommands]
        assert "checkpoint/create" in paths
        assert "checkpoint/restore" in paths
        assert "checkpoint/list" in paths
