"""
CLI Command Tests

Test CLI command functionality.
"""

import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import os

from webagents.cli.main import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def preserve_cwd():
    """Ensure working directory is restored after each test."""
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


class TestBasicCommands:
    """Test basic CLI commands."""
    
    def test_help(self):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "WebAgents" in result.output or "webagents" in result.output.lower()
    
    def test_version(self):
        """Test version display."""
        # Version should be accessible
        from webagents import __version__
        assert __version__
    
    def test_list_no_agents(self):
        """Test list command with no agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["list"])
            # Should not error, just show empty or message
            assert result.exit_code == 0


class TestInitCommand:
    """Test agent initialization."""
    
    def test_init_default(self):
        """Test creating default AGENT.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["init"])
            
            assert result.exit_code == 0
            assert (Path(tmpdir) / "AGENT.md").exists()
    
    def test_init_named(self):
        """Test creating named agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["init", "planner"])
            
            assert result.exit_code == 0
            assert (Path(tmpdir) / "AGENT-planner.md").exists()
    
    def test_init_context(self):
        """Test creating AGENTS.md context file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["init", "--context"])
            
            assert result.exit_code == 0
            assert (Path(tmpdir) / "AGENTS.md").exists()
    
    def test_init_already_exists(self):
        """Test init when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # Create first
            runner.invoke(app, ["init"])
            # Try again
            result = runner.invoke(app, ["init"])
            
            assert result.exit_code == 1


@pytest.mark.skip(reason="scan command removed - use 'list' instead")
class TestScanCommand:
    """Test agent scanning."""
    
    def test_scan_empty(self):
        """Test scan with no agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["scan"])
            
            assert result.exit_code == 0
    
    def test_scan_with_agents(self):
        """Test scan finds agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # Create some agents
            (Path(tmpdir) / "AGENT.md").write_text("---\nname: test\n---\nTest")
            (Path(tmpdir) / "AGENT-other.md").write_text("---\nname: other\n---\nOther")
            
            result = runner.invoke(app, ["scan"])
            
            assert result.exit_code == 0
            assert "AGENT.md" in result.output or "agent" in result.output.lower()


@pytest.mark.skip(reason="register command removed - agents auto-register via daemon")
class TestRegisterCommand:
    """Test agent registration."""
    
    def test_register_current_dir(self):
        """Test registering agents in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            (Path(tmpdir) / "AGENT.md").write_text("---\nname: test\n---\nTest")
            
            result = runner.invoke(app, ["register"])
            
            assert result.exit_code == 0


class TestConfigCommand:
    """Test configuration commands."""
    
    def test_config_help(self):
        """Test config help shows subcommands."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "get" in result.output or "set" in result.output
    
    def test_sandbox_help(self):
        """Test sandbox subcommand help."""
        result = runner.invoke(app, ["config", "sandbox", "--help"])
        assert result.exit_code == 0


@pytest.mark.skip(reason="template command removed - use 'init' with options instead")
class TestTemplateCommand:
    """Test template commands."""
    
    def test_template_list(self):
        """Test listing templates."""
        result = runner.invoke(app, ["template", "list"])
        assert result.exit_code == 0


class TestSkillCommand:
    """Test skill commands."""
    
    def test_skill_list(self):
        """Test listing skills."""
        result = runner.invoke(app, ["skill", "list"])
        assert result.exit_code == 0


class TestDaemonCommand:
    """Test daemon commands."""
    
    def test_daemon_status(self):
        """Test daemon status."""
        result = runner.invoke(app, ["daemon", "status"])
        assert result.exit_code == 0


@pytest.mark.skip(reason="discover command removed - discovery is handled by daemon")
class TestDiscoverCommand:
    """Test discovery commands."""
    
    def test_discover_no_intent(self):
        """Test discover without intent."""
        result = runner.invoke(app, ["discover"])
        # Should prompt for intent
        assert result.exit_code == 1 or "intent" in result.output.lower()


class TestAuthCommand:
    """Test auth commands."""
    
    def test_whoami_not_logged_in(self):
        """Test whoami when not logged in."""
        result = runner.invoke(app, ["auth", "whoami"])
        assert result.exit_code == 0
        assert "not logged in" in result.output.lower() or "login" in result.output.lower()
