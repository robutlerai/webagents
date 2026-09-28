"""
The chat's typed-line history on disk (owner decision D1, the S-291 fix,
2026-09-26): one file per profile, owner-only, in prompt_toolkit's
`FileHistory` format, which the TypeScript chat reads and writes too
(`cli/chat-history.ts`). The shape is pinned by
`tests/fixtures/cli/chat_commands.json` (`history`), which
`tests/unit/cli/chat-history-interactive2.test.ts` runs as well.
"""

import json
import os
import stat
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.history import FileHistory
from rich.console import Console

from webagents.cli.repl.commands import CHAT_HISTORY
from webagents.cli.repl.session import WebAgentsSession, chat_history_file, secure_chat_history

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())["history"]
POSIX = os.name == "posix"


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return tmp_path / "home"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_fixture_shape_is_the_one_both_chats_keep():
    assert FIXTURE["file"] == CHAT_HISTORY["file"]
    assert int(FIXTURE["dir_mode"], 8) == CHAT_HISTORY["dir_mode"]
    assert int(FIXTURE["file_mode"], 8) == CHAT_HISTORY["file_mode"]
    assert FIXTURE["keep"] == CHAT_HISTORY["keep"]


def test_prompt_toolkit_reads_the_sample_the_typescript_chat_writes(newcomer):
    file = chat_history_file()
    assert secure_chat_history(file)
    file.write_text(FIXTURE["encoded"])
    # `load_history_strings` yields newest first.
    assert list(FileHistory(str(file)).load_history_strings()) == list(reversed(FIXTURE["entries"]))


def test_the_file_lives_in_the_profile_folder_and_is_owner_only(newcomer, monkeypatch):
    monkeypatch.setenv("WEBAGENTS_PROFILE", "work")
    file = chat_history_file()
    assert file == newcomer / ".webagents-work" / "history"
    assert secure_chat_history(file)
    history = FileHistory(str(file))
    history.store_string("hello")
    history.store_string("two\nlines")
    text = file.read_text()
    assert "\n+hello\n" in text and "\n+two\n+lines\n" in text
    if POSIX:
        assert _mode(file.parent) == 0o700
        assert _mode(file) == 0o600


def test_a_file_an_older_version_left_readable_is_closed(newcomer):
    file = chat_history_file()
    file.parent.mkdir(parents=True, mode=0o755)
    file.write_text(FIXTURE["encoded"])
    os.chmod(file, 0o644)
    assert secure_chat_history(file)
    if POSIX:
        assert _mode(file.parent) == 0o700
        assert _mode(file) == 0o600
    assert list(FileHistory(str(file)).load_history_strings()) == list(reversed(FIXTURE["entries"]))


def test_the_chat_keeps_its_box_history_in_that_file(newcomer, monkeypatch):
    monkeypatch.setenv("WEBAGENTS_PROFILE", "work")
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    history = session.prompt_box.history
    assert isinstance(history, FileHistory)
    assert Path(history.filename) == newcomer / ".webagents-work" / "history"
    if POSIX:
        assert _mode(Path(history.filename)) == 0o600
        assert _mode(Path(history.filename).parent) == 0o700
