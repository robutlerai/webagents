import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from webagents.cli.daemon.server import WebAgentsDaemon
from webagents.cli.daemon.registry import DaemonRegistry

@pytest.fixture
def daemon_setup(tmp_path):
    # Setup directories
    watch_dir = tmp_path / "agents"
    watch_dir.mkdir()
    
    daemon = WebAgentsDaemon(port=0, watch_dirs=[watch_dir])
    
    # Mock manager methods to avoid actual execution
    daemon.manager.start = AsyncMock(return_value=True)
    daemon.manager.restart = AsyncMock(return_value=True)
    daemon.manager.get_running_agents = MagicMock(return_value=[])
    
    return daemon, watch_dir

@pytest.mark.asyncio
async def test_file_change_integration(daemon_setup):
    daemon, watch_dir = daemon_setup
    
    # Create an agent file
    agent_file = watch_dir / "AGENT-test.md"
    content = """---
name: test-agent
cron: "0 * * * *"
---
# Test Agent
"""
    agent_file.write_text(content.strip())
    
    # Simulate file creation event
    # Update registry first (simulating FileWatcher behavior)
    daemon.registry.update_from_file(agent_file)
    # Direct call to handler since we're not running the full watchdog loop in unit test
    await daemon._handle_file_change("created", agent_file)
    
    # Verify registry update
    agent = daemon.registry.get("test-agent")
    assert agent is not None
    assert agent.cron == "0 * * * *"
    
    # Verify cron update
    jobs = daemon.cron.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].agent_name == "test-agent"
    assert jobs[0].schedule == "0 * * * *"

@pytest.mark.asyncio
async def test_hot_reload(daemon_setup):
    daemon, watch_dir = daemon_setup
    
    # Mock agent running
    daemon.manager.get_running_agents.return_value = ["test-agent"]
    
    # Register initial agent
    agent_file = watch_dir / "AGENT-test.md"
    content = "---\nname: test-agent\n---"
    agent_file.write_text(content.strip())
    daemon.registry.update_from_file(agent_file)
    
    # Simulate file modification
    daemon.registry.update_from_file(agent_file) # Ensure registry has latest file info
    await daemon._handle_file_change("modified", agent_file)
    
    # Verify restart called
    daemon.manager.restart.assert_called_once_with("test-agent")

@pytest.mark.asyncio
async def test_cron_sync(daemon_setup):
    daemon, watch_dir = daemon_setup
    
    # Create agent with cron
    agent_file = watch_dir / "AGENT-cron.md"
    content = """---
name: cron-agent
cron: "@daily"
---"""
    agent_file.write_text(content.strip())
    
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("created", agent_file)
    
    assert len(daemon.cron.list_jobs()) == 1
    job = daemon.cron.list_jobs()[0]
    assert job.schedule == "0 0 * * *"  # @daily resolves to this
    
    # Update cron
    content = """---
name: cron-agent
cron: "@hourly"
---"""
    agent_file.write_text(content.strip())
    
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("modified", agent_file)
    
    assert len(daemon.cron.list_jobs()) == 1
    job = daemon.cron.list_jobs()[0]
    assert job.schedule == "0 * * * *"  # @hourly resolves to this
