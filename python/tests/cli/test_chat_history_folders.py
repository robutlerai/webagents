"""
Each folder its own history lines (2026-09-29), against the shared fixture
`tests/fixtures/cli/chat_commands.json` (`history.folders`), which the
TypeScript suite reads too (`tests/unit/cli/chat-history-folders.test.ts`).

↑ and the suggestions offered every line typed under the profile, in any
folder and by anything that ran the chat as its owner: an e2e test agent that
drove the chat under the owner's profile left its prompts in the owner's ↑
(the owner: "random stuff shows up in history.. prob leakage from other
sessions?"). Pinned here: an entry names its folder on a comment line
prompt_toolkit skips, a chat offers only its own folder's lines, an older entry
with no folder is offered nowhere, and a reader that ignores folders still reads
every entry.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.history import FileHistory
from rich.console import Console

from webagents.cli.repl.session import (
    FOLDER_LINE,
    FolderHistory,
    WebAgentsSession,
    chat_history_file,
    chat_history_folder,
    parse_history_entries,
    secure_chat_history,
)

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())["history"]
FOLDERS = FIXTURE["folders"]


@pytest.fixture
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "home"


def test_the_folder_line_is_the_fixtures():
    assert FOLDER_LINE == FOLDERS["line"]


def test_entries_name_their_folder():
    parsed = parse_history_entries(FOLDERS["encoded"])
    assert [text for text, _ in parsed] == FOLDERS["all"]
    assert [folder for _, folder in parsed] == ["/work/a", "/work/b", None]


@pytest.mark.parametrize("folder", sorted(FOLDERS["for_folder"]))
def test_a_chat_offers_its_own_folders_lines_only(tmp_path, folder):
    file = tmp_path / "history"
    file.write_text(FOLDERS["encoded"])
    # `load_history_strings` yields newest first.
    assert list(FolderHistory(str(file), folder).load_history_strings()) == list(reversed(FOLDERS["for_folder"][folder]))


def test_a_reader_that_ignores_folders_still_reads_every_entry(tmp_path):
    file = tmp_path / "history"
    file.write_text(FOLDERS["encoded"])
    assert list(FileHistory(str(file)).load_history_strings()) == list(reversed(FOLDERS["all"]))


def test_a_stored_line_names_its_folder_and_comes_back_there_only(tmp_path):
    file = tmp_path / "history"
    FolderHistory(str(file), "/work/a").store_string("two\nlines")
    text = file.read_text()
    assert f"\n{FOLDER_LINE}/work/a\n+two\n+lines\n" in text
    assert list(FolderHistory(str(file), "/work/a").load_history_strings()) == ["two\nlines"]
    assert list(FolderHistory(str(file), "/work/b").load_history_strings()) == []


def test_the_chat_keeps_its_folders_lines(newcomer, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    history = session.prompt_box.history
    assert isinstance(history, FolderHistory)
    assert history.folder == chat_history_folder() == str(work.resolve())
    assert Path(history.filename) == chat_history_file()
    assert secure_chat_history(chat_history_file())
