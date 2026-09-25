"""
Defects found by the 2026-09-23 end-to-end test campaign, pinned.

Each of these passed every unit test in the suite and was only visible by
running the real CLI against a real daemon in a real project directory.
"""

import asyncio
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()


class TestDiscoveryStaysOutOfToolDirectories:
    """`checkpoint create` copies the project into `.webagents/history/`,
    including `AGENT-demo.md`. Discovery descended into it, the snapshot
    declared the same `name:`, and the registry (keyed by name) replaced the
    real agent with the snapshot. From then on the daemon served the agent
    FROM THE SNAPSHOT: edits to the real file stopped taking effect, and
    `checkpoint list` read the snapshot's empty directory and reported none."""

    @pytest.mark.parametrize(
        "relative",
        [
            ".webagents/history/AGENT-demo.md",
            ".git/AGENT.md",
            "node_modules/pkg/AGENT.md",
            ".venv/lib/AGENT-x.md",
            "a/b/__pycache__/AGENT.md",
        ],
    )
    def test_tool_directories_are_not_discoverable(self, relative):
        from webagents.cli.daemon.registry import is_discoverable

        assert is_discoverable(Path(relative)) is False

    @pytest.mark.parametrize("relative", ["AGENT.md", "AGENT-demo.md", "team/AGENT-x.md"])
    def test_real_agent_files_are(self, relative):
        from webagents.cli.daemon.registry import is_discoverable

        assert is_discoverable(Path(relative)) is True

    def test_a_project_under_a_build_directory_is_still_found(self):
        # The watcher and restore paths hand over ABSOLUTE paths, and
        # `~/build/myproj/` is a real layout, so `build`/`dist` are not ignored.
        from webagents.cli.daemon.registry import is_discoverable

        assert is_discoverable(Path("/Users/someone/build/myproj/AGENT.md")) is True

    def test_the_context_file_is_not_an_agent(self):
        # It is watched so an edit reloads the agents beneath it. Registering
        # it made a phantom agent named `assistant` (the schema default) for
        # every context file on disk.
        from webagents.cli.daemon.registry import is_discoverable

        assert is_discoverable(Path("WEBAGENTS.md")) is False

    def test_scan_skips_the_snapshot_and_keeps_the_real_agent(self, tmp_path):
        from webagents.cli.daemon.registry import DaemonRegistry

        (tmp_path / "AGENT-demo.md").write_text("---\nname: demo\n---\n\nReal.\n")
        snapshot = tmp_path / ".webagents" / "history"
        snapshot.mkdir(parents=True)
        (snapshot / "AGENT-demo.md").write_text("---\nname: demo\n---\n\nStale.\n")
        (tmp_path / "WEBAGENTS.md").write_text("---\nnamespace: x\n---\n\nctx\n")

        registry = DaemonRegistry()
        asyncio.run(registry.scan_directory(tmp_path))

        assert set(registry.agents) == {"demo"}
        assert registry.agents["demo"].source_path == str(tmp_path / "AGENT-demo.md")

    def test_the_watcher_path_refuses_a_snapshot(self, tmp_path):
        # `update_from_file` is where file events AND restore-on-start arrive,
        # so a registration persisted by an earlier daemon is refused too.
        from webagents.cli.daemon.registry import DaemonRegistry

        snapshot = tmp_path / ".webagents" / "history" / "AGENT-demo.md"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("---\nname: demo\n---\n\nStale.\n")

        registry = DaemonRegistry()
        assert registry.update_from_file(snapshot) is None
        assert registry.agents == {}

    def test_the_watcher_path_refuses_the_context_file(self, tmp_path):
        from webagents.cli.daemon.registry import DaemonRegistry

        context = tmp_path / "WEBAGENTS.md"
        context.write_text("---\nnamespace: x\n---\n\nctx\n")

        registry = DaemonRegistry()
        assert registry.update_from_file(context) is None
        assert "assistant" not in registry.agents

    def test_a_name_collision_is_logged_not_silent(self, tmp_path):
        import logging

        from webagents.cli.daemon.registry import DaemonRegistry
        from webagents.cli.loader import AgentFile

        a = tmp_path / "one" / "AGENT.md"
        b = tmp_path / "two" / "AGENT.md"
        for path in (a, b):
            path.parent.mkdir()
            path.write_text("---\nname: same\n---\n\nBody.\n")

        # A handler on the logger itself: webagents' logging does not
        # propagate to the root logger, so `caplog` never sees it.
        seen = []
        handler = logging.Handler()
        handler.emit = lambda record: seen.append(record.getMessage())
        logger = logging.getLogger("webagents.cli.daemon.registry")
        logger.addHandler(handler)
        try:
            registry = DaemonRegistry()
            registry.register(AgentFile(a))
            registry.register(AgentFile(b))
        finally:
            logger.removeHandler(handler)

        assert any("declared by two files" in message for message in seen)


class TestWhoamiExitsNonZeroWhenLoggedOut:
    def test_human_and_json_modes_agree(self, tmp_path, monkeypatch):
        # The `--json` path already exited 1; the human path exited 0, so
        # `webagents whoami && webagents deploy` went ahead while logged out.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("WEBAGENTS_PROFILE", "pytest-campaign-whoami")
        monkeypatch.delenv("WEBAGENTS_TOKEN", raising=False)

        human = runner.invoke(app, ["whoami"])
        machine = runner.invoke(app, ["--json", "whoami"])

        assert human.exit_code == 1
        assert machine.exit_code == 1
