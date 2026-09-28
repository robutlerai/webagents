"""
The chat's command table, help and refusals, against the shared fixture
`tests/fixtures/cli/chat_commands.json`, which the TypeScript suite reads too
(`chat-commands-fixture-interactive.test.ts`). This replaces the old
source-parsing parity test's role: the fixture is the reference both chats are
held to (2026-09-26, interactive-mode spec 3.8, 3.9).

Everything runs under a throwaway HOME, the FILE secrets backend, no key and
no sign-in, at width 100.
"""

import asyncio
import json
import re
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.commands import CHAT_COMMANDS, CHAT_KEYS, MOVED_COMMANDS, OUTSIDE_THE_CHAT
from webagents.cli.repl.session import WebAgentsSession

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY")


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in KEY_VARS + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    yield project


def _chat(agent_text=None):
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=240, force_terminal=False, color_system=None, record=True)
    if agent_text:
        Path("AGENT.md").write_text(agent_text)
    asyncio.run(session.initialize())
    return session


def _say(session, line):
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.handle_input(line))
    text = session.console.export_text(clear=False)[before:]
    return [l.rstrip() for l in text.split("\n")]


def _collapsed(lines):
    return re.sub(r"\s+", " ", " ".join(lines))


def test_the_command_table_matches_the_fixture():
    table = [
        {"name": c.name, "usage": c.usage, "description": c.description, "group": c.group, "details": list(c.details)}
        for c in CHAT_COMMANDS
    ]
    assert table == FIXTURE["commands"]
    assert [list(k) for k in CHAT_KEYS] == FIXTURE["keys"]
    assert MOVED_COMMANDS == FIXTURE["moved"]
    assert OUTSIDE_THE_CHAT == FIXTURE["outside"]


def test_help_prints_every_group_the_keys_and_the_outside_line(newcomer):
    # rich wraps the long "outside" line at width 100, so match each fixture
    # line, collapsed, in order (the TypeScript console does not wrap).
    hay = _collapsed(_say(_chat("---\nname: helper\n---\nHelp.\n"), "/help"))
    at = 0
    for line in FIXTURE["help"]:
        wanted = re.sub(r"\s+", " ", line).strip()
        if not wanted:
            continue
        found = hay.find(wanted, at)
        assert found >= 0, f"{wanted!r} not found in order in: {hay}"
        at = found + len(wanted)


@pytest.mark.parametrize("command,lines", list(FIXTURE["help_command"].items()))
def test_help_for_a_command_shows_its_forms(newcomer, command, lines):
    hay = _collapsed(_say(_chat("---\nname: helper\n---\nHelp.\n"), f"/help {command}"))
    for line in lines:
        wanted = re.sub(r"\s+", " ", re.sub(r"^[✦✗▲✓] ", "", line)).strip()
        assert wanted in hay


@pytest.mark.parametrize("case", FIXTURE["refusals"], ids=[c["typed"] for c in FIXTURE["refusals"]])
def test_refusals_and_did_you_mean(newcomer, case):
    hay = _collapsed(_say(_chat("---\nname: helper\n---\nHelp.\n"), case["typed"]))
    for line in case["lines"]:
        assert re.sub(r"\s+", " ", line).strip() in hay


def test_the_tip_shows_on_the_built_in_agent_in_an_empty_folder(newcomer):
    session = _chat()  # built-in
    before = len(session.console.export_text(clear=False))
    session._say_new_agent_tip()
    assert FIXTURE["tip"] in session.console.export_text(clear=False)[before:]


def test_the_tip_does_not_show_when_the_folder_has_an_agent_file(newcomer):
    session = _chat("---\nname: helper\n---\nHelp.\n")
    before = len(session.console.export_text(clear=False))
    session._say_new_agent_tip()
    assert FIXTURE["tip"] not in session.console.export_text(clear=False)[before:]
