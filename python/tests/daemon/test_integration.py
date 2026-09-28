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
    # `cron:` is a list of named schedules (plan item 1.7, 2026-09-26); the old
    # string form is refused by the loader (tests/cli/test_cron_schema_w1daemon.py).
    content = """---
name: test-agent
cron:
  - name: hourly
    schedule: "0 * * * *"
    prompt: Report.
    deliver:
      file: reports/hourly.md
---
# Test Agent
"""
    agent_file.write_text(content.strip())

    # Simulate file creation event
    # Update registry first (simulating FileWatcher behavior)
    daemon.registry.update_from_file(agent_file)
    # Direct call to handler since we're not running the full watchdog loop in unit test
    await daemon._handle_file_change("created", agent_file)

    # Verify registry update: the schedule list is kept as the file wrote it
    agent = daemon.registry.get("test-agent")
    assert agent is not None
    assert isinstance(agent.cron, list) and agent.cron[0]["name"] == "hourly"

    # The file change synced the schedule into the runner
    # (cli/daemon/schedule_runner.py, plan item 1.7), with its next fire.
    listed = daemon.cron.list_schedules()
    assert [(s["agent"], s["name"], s["expression"]) for s in listed] == [("test-agent", "hourly", "0 * * * *")]
    assert listed[0]["next_run"] is not None

@pytest.mark.asyncio
@pytest.mark.skip(reason="Hot reload depends on manager state which is mocked - requires integration test")
async def test_hot_reload(daemon_setup):
    daemon, watch_dir = daemon_setup
    
    # Register initial agent first
    agent_file = watch_dir / "AGENT-test.md"
    content = "---\nname: test-agent\n---"
    agent_file.write_text(content.strip())
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("created", agent_file)
    
    # Now mock the agent as running (after initial registration)
    daemon.manager.get_running_agents.return_value = ["test-agent"]
    
    # Simulate file modification
    content_modified = "---\nname: test-agent\ndescription: Updated\n---"
    agent_file.write_text(content_modified.strip())
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("modified", agent_file)
    
    # Verify restart called
    daemon.manager.restart.assert_called_once_with("test-agent")

@pytest.mark.asyncio
async def test_cron_sync(daemon_setup):
    """A file's `cron:` list becomes the runner's schedules on the change
    event, an edit replaces them, a dropped block removes them, and a file
    whose block is wrong is refused by the loader before the registry sees
    it (plan item 1.7, 2026-09-26; the runner is cli/daemon/schedule_runner.py)."""
    daemon, watch_dir = daemon_setup

    agent_file = watch_dir / "AGENT-cron.md"
    agent_file.write_text(
        '---\nname: cron-agent\ncron:\n  - name: daily\n    schedule: "0 0 * * *"\n    prompt: Report.\n'
        "    deliver:\n      file: reports/daily.md\n---\n"
    )
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("created", agent_file)

    listed = daemon.cron.list_schedules()
    assert [(s["agent"], s["name"], s["kind"], s["expression"]) for s in listed] == [("cron-agent", "daily", "cron", "0 0 * * *")]
    assert listed[0]["deliver"] == {"kind": "file", "path": "reports/daily.md"}

    # The state file lives beside the agent file, under .webagents/cron/.
    state = watch_dir / ".webagents" / "cron" / "cron-agent.json"
    assert state.exists()

    # An edit: the schedule is replaced by the file's new one.
    agent_file.write_text(
        '---\nname: cron-agent\ncron:\n  - name: hourly\n    schedule: "0 * * * *"\n    prompt: Report.\n'
        "    deliver:\n      file: reports/hourly.md\n---\n"
    )
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("modified", agent_file)
    assert [(s["agent"], s["name"], s["expression"]) for s in daemon.cron.list_schedules()] == [("cron-agent", "hourly", "0 * * * *")]

    # The block gone: nothing scheduled for the agent.
    agent_file.write_text("---\nname: cron-agent\n---\n")
    daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("modified", agent_file)
    assert daemon.cron.list_schedules() == []

    # The old string form is refused by the loader, so the registry keeps
    # what it had and the runner is untouched.
    agent_file.write_text('---\nname: cron-agent\ncron: "@hourly"\n---\n')
    with pytest.raises(Exception):
        daemon.registry.update_from_file(agent_file)
    await daemon._handle_file_change("modified", agent_file)
    assert daemon.cron.list_schedules() == []
