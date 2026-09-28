"""
The opt-out is said in one wording, as a notice in the chat, and once in
`doctor` (the ptypass-fixes lane, 2026-09-27, brief item 12; fixture
`tests/fixtures/sandbox/srt.json` `unrestricted`, `status.warnings`). The
TypeScript twin is `typescript/tests/unit/cli/ptypass-fixes-opt-out.test.ts`.

The real-terminal PTY pass saw the `sandbox: off` load line printed raw above
the welcome card, wrapped mid-word ("Use `deve" / "lopment`"), saying "Use
development or strict" while `/sandbox` said "Remove `sandbox: off`", and
`doctor` printing it above its own sandbox line.
"""

import asyncio
import re
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.agents.skills.local.shell.skill import NO_SANDBOX_WARNING, UNRESTRICTED_WARNING
from webagents.cli.repl.chat_words import CHAT_WORDS, fill
from webagents.cli.repl.session import WebAgentsSession

OFF = "---\nname: helper\nskills:\n  - openai\n  - shell\nsandbox: off\n---\nHelp.\n"


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for var in ("ANTHROPIC_API_KEY", "WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL", "WEBAGENTS_NO_SANDBOX"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def test_one_wording_the_load_line_is_sandboxs_headline_and_fix():
    assert UNRESTRICTED_WARNING == f"Sandbox: {fill('sandboxOff', state='off (agent file)')} {CHAT_WORDS['sandboxOffFix']}"
    assert NO_SANDBOX_WARNING == f"Sandbox: {fill('sandboxOff', state='off (--no-sandbox)')} {CHAT_WORDS['sandboxFlagFix']}"
    assert "Use `development` or `strict`" not in UNRESTRICTED_WARNING


def test_the_chat_says_it_as_its_own_notice_wrapped_at_words(capsys):
    Path("AGENT.md").write_text(OFF)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=60, force_terminal=False, color_system=None, record=True)
    session._chatting = True
    asyncio.run(session.initialize())
    printed = session.console.export_text(clear=False)
    assert "▲ Sandbox: off (agent file): commands are not confined" in re.sub(r"\s+", " ", printed)
    assert CHAT_WORDS["sandboxOffFix"] in re.sub(r"\s+", " ", printed)
    # Not also the raw stderr line.
    assert UNRESTRICTED_WARNING not in capsys.readouterr().err
    # Wrapped at words: every line ends and starts on a whole word.
    words = set(re.sub(r"[^\w`:.()-]+", " ", UNRESTRICTED_WARNING).split())
    for line in printed.splitlines():
        for token in re.sub(r"[^\w`:.()-]+", " ", line).split():
            assert token in words or not token.isalpha() or token in ("✓", "▲"), (token, line)


def test_outside_the_chat_it_goes_to_stderr(capsys):
    Path("AGENT.md").write_text(OFF)
    session = WebAgentsSession(agent_path=None, interactive=False)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    assert UNRESTRICTED_WARNING in capsys.readouterr().err
    assert "Sandbox: off" not in session.console.export_text(clear=False)


def test_doctor_says_it_once_in_its_sandbox_line(capsys, newcomer):
    from webagents.cli.doctor import run_checks

    Path("AGENT.md").write_text(OFF)
    checks = run_checks(folder=newcomer)
    captured = capsys.readouterr()
    assert UNRESTRICTED_WARNING not in captured.err + captured.out
    sandbox = next(c for c in checks if c.name == "sandbox")
    assert sandbox.status == "warn" and sandbox.detail.startswith("off (agent file)")
