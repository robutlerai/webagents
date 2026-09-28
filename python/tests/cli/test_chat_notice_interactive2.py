"""
The chat bugs the 2026-09-26 e2e run found, pinned (interactive-mode part 2):

  * the change notice says WHEN the agent file changed: an edit made while the
    chat sat idle is "changed since the chat loaded it", only one made between
    a message and its reply is "changed during the last reply";
  * the goodbye line counts this chat's replies and tokens, not a resumed
    conversation's;
  * /sandbox says "Invalid" for a declaration that did not resolve (D6);
  * /tools leaves the turn-scoped content tools out.

The TypeScript twin is `tests/unit/cli/chat-notice-interactive2.test.ts`.
"""

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.sessions import save_session, sessions_dir

AGENT = "---\nname: helper\nskills:\n  - openai\n---\nHelp.\n"


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def _chat(text: str = AGENT) -> WebAgentsSession:
    Path("AGENT.md").write_text(text)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


def _printed(session: WebAgentsSession, since: int = 0) -> str:
    return session.console.export_text(clear=False)[since:]


def _mark(session: WebAgentsSession) -> int:
    return len(session.console.export_text(clear=False))


def test_an_edit_made_while_idle_is_changed_since_the_chat_loaded_it(newcomer, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    session = _chat()

    async def turn(_message):
        return None

    monkeypatch.setattr(session, "_turn", turn)
    # The person edits the file at the prompt, then sends a message.
    Path("AGENT.md").write_text(AGENT.replace("Help.", "Help more."))
    asyncio.run(session.handle_input("hi"))
    at = _mark(session)
    session._say_file_changed()
    out = _printed(session, at)
    assert "✦ AGENT.md changed since the chat loaded it. /reload uses the new version." in out
    assert "during the last reply" not in out


def test_a_change_between_the_message_and_the_reply_is_during_the_last_reply(newcomer, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    session = _chat()

    async def turn(_message):
        Path("AGENT.md").write_text(AGENT.replace("Help.", "Rewritten by the agent."))

    monkeypatch.setattr(session, "_turn", turn)
    asyncio.run(session.handle_input("hi"))
    at = _mark(session)
    session._say_file_changed()
    assert "▲ AGENT.md changed during the last reply. /reload shows what changed." in _printed(session, at)


def test_the_goodbye_line_counts_this_chats_replies_and_tokens_only(newcomer):
    session = _chat()
    directory = sessions_dir(newcomer, "helper")
    directory.mkdir(parents=True)
    save_session(
        directory,
        {
            "session_id": "0b7f8e2a-3333-4c2d-9a3e-000000000003",
            "agent_name": "helper",
            "created_at": "2026-09-24T10:00:00.000Z",
            "updated_at": "2026-09-24T10:05:00.123Z",
            "messages": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}],
            "metadata": {"sdk": "typescript"},
            "input_tokens": 30,
            "output_tokens": 6,
        },
    )
    asyncio.run(session.handle_input("/resume 1"))
    assert session.input_tokens + session.output_tokens == 36
    # Nothing said in THIS chat yet: no goodbye line at all.
    at = _mark(session)
    session._goodbye()
    assert _printed(session, at).strip() == ""
    # One reply of 18 tokens here.
    session.turns = 1
    session.session_tokens = 18
    at = _mark(session)
    session._goodbye()
    out = _printed(session, at)
    assert "✦ 1 reply · 18 tokens ·" in out and "54 tokens" not in out
    # /status still counts the conversation's tokens, resumed ones included.
    at = _mark(session)
    asyncio.run(session.handle_input("/status"))
    assert "2 messages, 36 tokens" in _printed(session, at)


def test_sandbox_says_invalid_for_a_declaration_that_did_not_resolve(newcomer):
    session = _chat("---\nname: boxed\nskills:\n  - shell\n---\nHelp.\n")
    shell = session.built.agent.skills["shell"]
    shell._sandbox_error = "network must be a list of domains"
    assert shell.sandbox_error == "network must be a list of domains"
    at = _mark(session)
    asyncio.run(session.handle_input("/sandbox"))
    out = _printed(session, at)
    assert "▲ Sandbox: Invalid: network must be a list of domains" in out
    assert "Every command is refused until the declaration is fixed." in out


def test_tools_leaves_the_turn_scoped_content_tools_out(newcomer):
    session = _chat("---\nname: helper\nskills:\n  - todo\n---\nHelp.\n")
    # As the TypeScript agent registers them for a turn (`present`, `read_content`).
    session.built.agent._registered_tools.append({"name": "present", "description": "Display a piece of content", "scope": "all"})
    session.built.agent._registered_tools.append({"name": "read_content", "description": "Load media", "scope": "all"})
    at = _mark(session)
    asyncio.run(session.handle_input("/tools"))
    out = _printed(session, at)
    assert "todo_add" in out and "present" not in out and "read_content" not in out
