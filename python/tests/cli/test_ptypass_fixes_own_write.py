"""
The chat's own `always` write to the agent file is not "a change during the
reply", and the sandbox state follows it at once (the ptypass-fixes lane,
2026-09-27, brief item 11). The TypeScript twin is
`typescript/tests/unit/cli/ptypass-fixes-own-write.test.ts`.

The real-terminal PTY pass answered `always` to a refused host: the chat
wrote the host into AGENT.md, then its next prompt said "AGENT.md changed
during the last reply", and `/sandbox` kept saying "(default)" until
`/reload`, though the session already ran what the file said.
"""

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.session import WebAgentsSession, _HostAsker

AGENT = "---\nname: helper\nskills:\n  - openai\n  - shell\n---\nHelp.\n"


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


def _chat() -> WebAgentsSession:
    Path("AGENT.md").write_text(AGENT)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


def _shell(session: WebAgentsSession):
    return session.built.agent.skills["shell"]


def test_always_during_a_reply_is_not_a_change_and_sandbox_says_agent_file(monkeypatch):
    session = _chat()
    shell = _shell(session)
    assert shell.sandbox_state_line() == "development (default)"
    asker = _HostAsker(session, shell)

    async def turn(_message, snapshot=False):
        # What `always` does in the middle of a reply.
        await asker.allow_host_always("example.com")

    monkeypatch.setattr(session, "_turn", turn)
    asyncio.run(session.handle_input("fetch it"))
    assert "example.com" in Path("AGENT.md").read_text()
    at = len(session.console.export_text(clear=False))
    session._say_file_changed()
    said = session.console.export_text(clear=False)[at:]
    assert "changed" not in said, said
    assert shell.sandbox_state_line() == "development (agent file)"
    assert "example.com" in shell.policy.network_domains
    session.cmd_sandbox()
    assert "development (agent file)" in session.console.export_text(clear=False)[at:]


def test_a_change_of_the_persons_own_is_still_said(monkeypatch):
    session = _chat()
    shell = _shell(session)
    asker = _HostAsker(session, shell)

    async def turn(_message, snapshot=False):
        # Someone else edits the file during the reply, then `always` writes too.
        Path("AGENT.md").write_text(AGENT.replace("Help.", "Help more."))
        await asker.allow_host_always("example.com")

    monkeypatch.setattr(session, "_turn", turn)
    asyncio.run(session.handle_input("fetch it"))
    at = len(session.console.export_text(clear=False))
    session._say_file_changed()
    assert "AGENT.md changed during the last reply" in session.console.export_text(clear=False)[at:]
