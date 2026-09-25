"""
Phase 5a: the frontmatter schema stops accepting things it will not act on.

Both models used to set `extra = "allow"`. A typo parsed, validated, was stored
on the model and was never read, so `skils: [mcp]` produced an agent with no
skills and no message. These tests pin the rejection, the suggestion, and the
three things the rejection must NOT break:

  * a directory scan, which must survive one unreadable file
  * every agent file the repo and `webagents init` have ever produced
  * the seven declared-but-inert keys, which stay legal on purpose
"""

from pathlib import Path

import pytest

from webagents.cli.loader import (
    INERT_FIELDS,
    AgentFile,
    AgentFormatError,
    AgentMetadata,
    ContextFile,
    ContextMetadata,
    scan_agent_files,
)
from webagents.cli.loader.context import CONTEXT_FILENAME


class TestUnknownKeysAreRejected:
    def test_a_typo_is_refused_rather_than_kept(self):
        with pytest.raises(AgentFormatError) as e:
            AgentMetadata(name="x", skils=["mcp"])
        assert "skils" in str(e.value)

    def test_the_message_names_the_nearest_known_key(self):
        # "unknown key" alone sends the reader to the documentation for
        # something already on their screen.
        with pytest.raises(AgentFormatError) as e:
            AgentMetadata(name="x", modle="openai/gpt-4o")
        assert "did you mean 'model'" in str(e.value)

    def test_a_key_with_no_near_match_still_lists_what_is_known(self):
        with pytest.raises(AgentFormatError) as e:
            AgentMetadata(name="x", qqqqq=1)
        message = str(e.value)
        assert "did you mean" not in message
        assert "skills" in message and "namespace" in message

    def test_the_context_schema_is_strict_too(self):
        with pytest.raises(AgentFormatError) as e:
            ContextMetadata(namespce="x")
        assert "did you mean 'namespace'" in str(e.value)

    def test_it_is_not_a_pydantic_validation_error(self):
        # Pydantic re-wraps ValueError raised inside a validator, which buries
        # the message and makes the class uncatchable. This one must come out
        # intact.
        from pydantic import ValidationError

        with pytest.raises(AgentFormatError) as e:
            AgentMetadata(name="x", skils=[])
        assert not isinstance(e.value, ValidationError)
        assert str(e.value).startswith("AGENT.md frontmatter:")


class TestTheErrorSaysWhichFile:
    def test_an_agent_file_error_carries_its_path(self, tmp_path):
        bad = tmp_path / "AGENT-bad.md"
        bad.write_text("---\nname: b\nskils: [mcp]\n---\nBody\n")

        with pytest.raises(AgentFormatError) as e:
            AgentFile(bad)
        assert str(bad) in str(e.value)

    def test_a_context_file_error_carries_its_path(self, tmp_path):
        bad = tmp_path / CONTEXT_FILENAME
        bad.write_text("---\nnamespce: x\n---\nBody\n")

        with pytest.raises(AgentFormatError) as e:
            ContextFile(bad)
        assert str(bad) in str(e.value)


class TestOneBadFileDoesNotHideTheRest:
    def test_scan_returns_the_good_files_and_reports_the_bad(self, tmp_path):
        (tmp_path / "AGENT-ok.md").write_text("---\nname: ok\n---\nBody\n")
        (tmp_path / "AGENT-bad.md").write_text("---\nname: bad\nskils: []\n---\nBody\n")

        files, errors = scan_agent_files(tmp_path)

        assert [f.name for f in files] == ["ok"]
        assert len(errors) == 1
        assert errors[0][0].name == "AGENT-bad.md"
        assert "skils" in errors[0][1]

    def test_find_agent_files_still_returns_a_plain_list(self, tmp_path):
        # The old signature, still used by `AgentLoader.load_all`.
        (tmp_path / "AGENT-ok.md").write_text("---\nname: ok\n---\nBody\n")
        (tmp_path / "AGENT-bad.md").write_text("---\nname: bad\nskils: []\n---\nBody\n")

        from webagents.cli.loader import find_agent_files

        assert [f.name for f in find_agent_files(tmp_path)] == ["ok"]


class TestTheInertKeysStayLegal:
    def test_every_inert_key_is_still_accepted(self):
        # Dropping them would reject every agent file `webagents init` has ever
        # written, and every example in this repo. Measured 2026-09-23: all
        # seven repo files set at least one.
        meta = AgentMetadata(
            name="x",
            tools=["a"],
            visibility="public",
            version="2.0.0",
            author="someone",
            tags=["t"],
            mcp_servers=["s"],
        )
        assert meta.visibility == "public"

    def test_the_inert_set_matches_the_schema(self):
        # A key removed from the model but left in INERT_FIELDS would make
        # `doctor` report something that can no longer be written.
        assert INERT_FIELDS <= set(AgentMetadata.model_fields)

    def test_what_init_writes_parses_cleanly_and_is_not_inert(self, tmp_path, monkeypatch):
        import os

        from typer.testing import CliRunner

        from webagents.cli.main import app

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert CliRunner().invoke(app, ["init", "demo"]).exit_code == 0
            written = AgentFile(tmp_path / "demo" / "AGENT.md")
        finally:
            os.chdir(cwd)

        # `init` used to write `visibility: local`, teaching a dead field on
        # the very first file a user sees.
        assert set(written._raw_yaml) & INERT_FIELDS == set()
        assert written.name == "demo"


class TestWatchIsReportedAsInert:
    def test_watch_is_listed(self):
        # It IS read, into `DaemonAgent.watch_patterns`, and then nothing
        # consumes it. From outside, "stored in a struct" and "ignored" are
        # the same thing.
        assert "watch" in INERT_FIELDS

    def test_the_trigger_refuses_instead_of_returning_silently(self):
        import asyncio

        from webagents.cli.daemon.watcher import FileWatchTrigger

        trigger = FileWatchTrigger(agent_manager=None, patterns=["*.csv"])
        with pytest.raises(NotImplementedError):
            asyncio.run(trigger.start(Path("."), "some-agent"))


class TestTheDeadSandboxModuleIsNotWired:
    def test_nothing_outside_it_imports_the_enforcer(self):
        # S-217. Two classes share the name `SandboxConfig` and nothing else;
        # a caller added here would AttributeError on the schema's version.
        import subprocess

        root = Path(__file__).resolve().parents[2] / "webagents"
        hits = subprocess.run(
            ["grep", "-rn", "SandboxEnforcer", str(root), "--include=*.py"],
            capture_output=True,
            text=True,
        ).stdout.splitlines()

        outside = [h for h in hits if "cli/daemon/sandbox.py" not in h]
        assert outside == [], f"daemon/sandbox.py gained a caller: {outside}"
