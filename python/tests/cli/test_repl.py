"""
REPL Tests

Test interactive REPL functionality.
"""

import pytest
from pathlib import Path
import tempfile

from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.repl.slash_commands import SlashCommandRegistry, handle_slash_command


class TestSlashCommands:
    """Test slash command functionality."""
    
    def test_registry_has_builtins(self):
        """Test registry has built-in commands."""
        registry = SlashCommandRegistry()
        
        # Core commands that should always exist
        assert registry.get("help") is not None
        assert registry.get("exit") is not None
        assert registry.get("clear") is not None
        # Note: save/load are now agent-provided commands via SessionSkill, not builtins
    
    def test_list_commands(self):
        """Test listing all commands."""
        registry = SlashCommandRegistry()
        commands = registry.list_commands()
        
        assert len(commands) > 5
        assert "help" in commands
        assert "exit" in commands
    
    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Test handling unknown command."""
        # handle_slash_command is now async
        session = WebAgentsSession()
        result = await handle_slash_command("/unknown_command", session)
        # Should return None for unknown
        assert result is None


class TestSession:
    """Test REPL session."""
    
    def test_session_init(self):
        """Test session initialization."""
        session = WebAgentsSession()
        
        assert session.running == True
        assert session.slash_commands is not None
    
    def test_session_with_agent(self):
        """Test session with agent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_path = Path(tmpdir) / "AGENT.md"
            agent_path.write_text("---\nname: test\n---\nTest agent")
            
            session = WebAgentsSession(agent_path=agent_path)
            
            assert session.agent_path == agent_path
            assert session.agent_name == "AGENT"
    
    def test_session_token_stats(self):
        """Test token statistics."""
        session = WebAgentsSession()
        
        assert session.input_tokens == 0
        assert session.output_tokens == 0
