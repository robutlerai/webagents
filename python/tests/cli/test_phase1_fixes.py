"""
Regression tests for the CLI defects fixed on 2026-09-23.

Each test pins one thing that was shipping broken. They are grouped here rather
than scattered into the existing files because they share a motivation: every
one of these was invisible to the suite at the time, and several were invisible
to the user too, which is why they lasted.
"""

import os

import pytest
from typer.testing import CliRunner


runner = CliRunner()


@pytest.fixture(autouse=True)
def preserve_cwd():
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


class TestAnAgentIsItsDeclaredName:
    """An agent is known by its frontmatter name, not its filename (`agent_files.py`).

    `webagents list` used `af.stem`, so `AGENT-foo.md` listed as `AGENT-foo`.
    `list` is gone; `-a` and `/agent` use the same rule."""

    def test_the_declared_name_is_the_name(self, tmp_path):
        from webagents.cli.agent_files import folder_agents

        (tmp_path / "AGENT-foo.md").write_text("---\nname: bar\n---\nBody\n")
        assert [a.name for a in folder_agents(tmp_path)] == ["bar"]

    def test_either_name_finds_it(self, tmp_path):
        from webagents.cli.agent_files import agent_file_for

        (tmp_path / "AGENT-foo.md").write_text("---\nname: bar\n---\nBody\n")
        assert agent_file_for(tmp_path, "bar") == tmp_path / "AGENT-foo.md"
        assert agent_file_for(tmp_path, "foo") == tmp_path / "AGENT-foo.md"

    def test_a_broken_file_is_not_offered(self, tmp_path):
        from webagents.cli.agent_files import folder_agents

        (tmp_path / "AGENT-good.md").write_text("---\nname: good\n---\nBody\n")
        (tmp_path / "AGENT-bad.md").write_text("---\nname: [unclosed\n---\n")
        assert "good" in [a.name for a in folder_agents(tmp_path)]

    def test_the_built_in_agent_by_name(self, tmp_path):
        from webagents.cli.agent_files import agent_file_for

        assert agent_file_for(tmp_path, "robutler") is None


class TestDaemonClientModel:
    """The CLI does not override the model the agent declares."""

    def test_model_is_omitted_when_not_explicitly_set(self):
        # Both chat paths hardcoded `"model": "gpt-4o-mini"`, and the daemon
        # path is the only live one for the REPL and the TUI, so every
        # interactive session silently ignored the agent's own `model:`.
        from webagents.cli.client.daemon_client import DaemonClient

        payload = DaemonClient._chat_payload(
            [{"role": "user", "content": "hi"}], stream=False, model=None
        )
        assert "model" not in payload
        assert payload["stream"] is False

    def test_an_explicit_model_is_still_sent(self):
        from webagents.cli.client.daemon_client import DaemonClient

        payload = DaemonClient._chat_payload(
            [{"role": "user", "content": "hi"}], stream=True, model="openai/x"
        )
        assert payload["model"] == "openai/x"
        assert payload["stream"] is True


class TestNoDeadCommandModules:
    """Every module in `cli/commands/` is reachable."""

    def test_command_package_exports_only_reachable_modules(self):
        # Six modules sat in `cli/commands/` that `main.py` never registered and
        # nothing imported, so every command they defined was unreachable. This
        # pins the cleanup: a module added here must be wired up or justified.
        import webagents.cli.commands as commands

        assert set(commands.__all__) == {"config", "daemon", "secrets"}

    def test_deleted_modules_are_gone(self):
        import importlib

        # And, on 2026-09-24, the groups the TypeScript CLI never had.
        for name in ["cron", "discover", "index", "intent", "namespace", "sync",
                     "agent", "auth", "checkpoint", "deploy", "doctor", "register",
                     "session", "skill", "template", "ui"]:
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(f"webagents.cli.commands.{name}")
