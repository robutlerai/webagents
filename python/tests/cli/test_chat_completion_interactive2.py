"""
Argument completion in the chat's box (interactive-mode spec 3.8, 2026-09-26):
after `/<command> ` the menu offers the next word's values, enter and tab
insert one and never run the command, and the menu closes when nothing more is
offered. The commands that complete, and the first values each offers, are
pinned by `tests/fixtures/cli/chat_commands.json` (`completion`), which
`tests/unit/cli/chat-completion-interactive2.test.ts` runs too.
"""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer
from rich.console import Console

from webagents.cli.repl.commands import COMPLETED_COMMANDS
from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.ui.prompt_box import PromptBox, Slot, argument_words
from webagents.cli.ui.theme import theme_for

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())["completion"]


def _agent_completer(args: str):
    before, partial = argument_words(args)
    if before == ["edit"]:
        return Slot([("helper", "A helper")], partial)
    if before:
        return None
    return Slot([("helper", "A helper"), ("robutler", "The general assistant"), ("new", "make one here"), ("edit", "open its file")], partial)


def _skills_completer(args: str):
    before, partial = argument_words(args)
    if before[:1] == ["add"]:
        return Slot([(name, "") for name in ("todo", "shell", "memory") if name not in before[1:]], partial)
    if before:
        return None
    return Slot([("add", ""), ("remove", "")], partial)


def _box() -> PromptBox:
    return PromptBox(
        theme_for(Console(file=StringIO(), width=100, force_terminal=False, color_system=None)),
        commands=[("/help", "Show the commands and keys"), ("/agent", "List this folder"), ("/skills", "Skills"), ("/exit", "Leave")],
        footer=lambda: [],
        completers={"agent": _agent_completer, "skills": _skills_completer},
    )


def test_the_menu_stays_open_after_a_command_with_the_values_it_offers():
    box = _box()
    assert [c[0] for c in box.menu_items("/agent ")] == ["helper", "robutler", "new", "edit"]
    # By prefix first, then by substring, in the order offered.
    assert [c[0] for c in box.menu_items("/agent e")] == ["edit", "helper", "robutler", "new"]
    assert [c[0] for c in box.menu_items("/agent ed")] == ["edit"]
    assert [c[0] for c in box.menu_items("/agent edit ")] == ["helper"]
    assert box.menu_items("/agent new ") == []


def test_an_inserted_argument_replaces_the_word_being_typed_and_never_runs():
    box = _box()
    buffer = Buffer()
    buffer.text = "/agent he"
    box._insert_argument(buffer, "helper")
    assert buffer.text == "/agent helper " and buffer.cursor_position == len(buffer.text)
    # Nothing more is offered after a name: the menu is closed.
    assert box.menu_items(buffer.text) == []
    buffer.text = "/skills "
    box._insert_argument(buffer, "add")
    assert buffer.text == "/skills add "
    assert [c[0] for c in box.menu_items(buffer.text)] == ["todo", "shell", "memory"]
    buffer.text = "/skills add me"
    box._insert_argument(buffer, "memory")
    assert buffer.text == "/skills add memory "
    # A list command keeps offering the names not yet typed; esc closes its menu, then enter sends.
    assert [c[0] for c in box.menu_items(buffer.text)] == ["todo", "shell"]


def test_the_menu_closes_after_a_command_that_completes_nothing():
    box = _box()
    assert box.menu_items("/help ") == []
    # The command menu itself still narrows on the name, with the slash.
    assert [c[0] for c in box.menu_items("/ex")] == ["/exit"]


def test_the_command_menu_finds_a_command_by_prefix_then_by_substring_as_the_typescript_box_does():
    # 2026-09-28: the query kept its `/`, so "/mo" found /model but never /memory,
    # and "/e" found /exit alone where the TypeScript box offers three.
    box = _box()
    assert [c[0] for c in box.menu_items("/e")] == ["/exit", "/help", "/agent"]
    assert [c[0] for c in box.menu_items("/")] == ["/help", "/agent", "/skills", "/exit"]
    assert [c[0] for c in box.menu_items("/ills")] == ["/skills"]


@pytest.fixture
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
    return project


def test_the_commands_that_complete_are_the_fixture_list():
    assert list(COMPLETED_COMMANDS) == FIXTURE["commands"]


def test_the_chats_completers_offer_the_first_values_the_fixture_pins(newcomer):
    (newcomer / "AGENT.md").write_text("---\nname: helper\ndescription: A helper\nskills:\n  - todo\n---\nHelp.\n")
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    asyncio.run(session._refresh_completion_data())
    completers = session._completers()
    assert sorted(completers) == sorted(FIXTURE["commands"])
    assert sorted(session.prompt_box.completers) == sorted(FIXTURE["commands"])
    def values(command: str, args: str):
        slot = completers[command](args)
        return None if slot is None else [row[0] for row in slot.rows]

    for command, first in FIXTURE["first_values"].items():
        offered = values(command, " ")
        for value in first:
            assert value in offered, command
    agents = values("agent", " ")
    assert "helper" in agents and "robutler" in agents
    # A completer is told everything typed after the command: a finished word ends in a space.
    assert values("agent", " edit ") == ["helper"]
    assert values("agent", " helper ") is None
    assert "todo" in values("skills", " add ")
    assert "todo" not in values("skills", " add todo ")
    assert values("skills", " remove ") == ["todo"]
    assert values("keys", " set OPENAI_API_KEY ") is None
    assert "status" in values("help", " ")
    assert "OPENAI_API_KEY" in values("keys", " set ")
    assert values("cron", " run ") == []
    assert values("memory", " forget ") == []
