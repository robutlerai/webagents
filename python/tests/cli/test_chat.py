"""
The chat's commands, driven the way a person types them (2026-09-24).

The chat runs its agent in-process and takes the same commands, in the same
words, as the TypeScript chat (`repl/commands.py`; the wording is pinned by
`test_chat_command_parity.py`). These tests drive `handle_input` with what a
person types and read what the chat printed.

Every test runs under a throwaway HOME with the FILE secrets backend, no
provider key and no sign-in, so nothing here touches the machine's keychain or
its real login.
"""

import asyncio
import json
import os
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.commands import CHAT_COMMANDS
from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.sessions import sessions_dir, slug_for

KEY_VARS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "XAI_API_KEY", "FIREWORKS_API_KEY",
)


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


def _agent(folder: Path, text: str, name: str = "AGENT.md") -> Path:
    path = folder / name
    path.write_text(text)
    return path


def _chat(agent_path=None) -> WebAgentsSession:
    session = WebAgentsSession(agent_path=agent_path)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


def _say(session: WebAgentsSession, line: str) -> str:
    """Type `line`, return what the chat printed for it."""
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.handle_input(line))
    return session.console.export_text(clear=False)[before:]


def test_help_lists_every_command_with_its_usage_and_the_keys(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    out = _say(chat, "/help")
    for command in CHAT_COMMANDS:
        assert command.usage in out and command.description in out
    assert "Keys" in out and "ctrl+c" in out


def test_help_for_one_command_shows_how_to_type_it(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    out = _say(chat, "/help keys")
    assert "/keys [set|unset NAME]" in out
    assert "Unknown command /nope." in _say(chat, "/help nope")


def test_an_unknown_command_says_how_to_find_the_right_one(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    out = _say(chat, "/nope")
    assert "Unknown command /nope." in out and "/help" in out


def test_a_message_is_not_sent_when_there_is_no_model_and_the_way_out_is_named(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    out = _say(chat, "hello")
    assert "webagents login" in out and "/login" in out
    assert chat.messages == []


def test_new_starts_a_fresh_conversation(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    chat.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    before = chat.session_id
    out = _say(chat, "/new")
    assert "Started a new conversation." in out
    assert chat.messages == [] and chat.session_id != before


def test_conversations_are_kept_under_the_profile_not_in_the_project(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    chat.messages = [{"role": "user", "content": "remember this"}]
    chat.save_conversation()
    expected = Path(os.environ["HOME"]) / ".webagents" / "sessions" / slug_for(str(newcomer.resolve())) / "helper"
    assert (expected / f"{chat.session_id}.json").exists()
    assert sessions_dir(newcomer, "helper") == expected
    assert not (newcomer / ".webagents" / "sessions").exists()
    assert oct((expected / f"{chat.session_id}.json").stat().st_mode & 0o777) == "0o600"


def test_resume_lists_earlier_conversations_and_continues_one(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    assert "No earlier conversations with helper in this folder." in _say(chat, "/resume")

    chat.messages = [{"role": "user", "content": "plan the launch"}, {"role": "assistant", "content": "Step one."}]
    chat.save_conversation()
    _say(chat, "/new")
    listing = _say(chat, "/resume")
    assert "Earlier conversations" in listing and "plan the launch" in listing and "/resume <number>" in listing

    assert "There is no conversation 9." in _say(chat, "/resume 9")

    out = _say(chat, "/resume 1")
    assert "Continuing the conversation from" in out and "(2 messages)" in out
    assert "── Earlier in this conversation ──" in out and "Step one." in out
    assert chat.messages[0]["content"] == "plan the launch"


def test_resume_reads_a_conversation_the_typescript_chat_wrote(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    directory = sessions_dir(newcomer, "helper")
    directory.mkdir(parents=True)
    written_by_ts = {
        "session_id": "0b7f8e2a-1111-4c2d-9a3e-000000000001",
        "agent_name": "helper",
        "created_at": "2026-09-24T10:00:00.000Z",
        "updated_at": "2026-09-24T10:05:00.123Z",
        "messages": [{"role": "user", "content": "from typescript"}, {"role": "assistant", "content": "ok"}],
        "metadata": {"sdk": "typescript"},
        "input_tokens": 12,
        "output_tokens": 3,
    }
    (directory / f"{written_by_ts['session_id']}.json").write_text(json.dumps(written_by_ts))
    out = _say(chat, "/resume 1")
    assert "Continuing the conversation" in out
    assert chat.messages[0]["content"] == "from typescript" and chat.input_tokens == 12


def test_agent_lists_this_folders_agents_and_the_built_in_one_and_switches(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\ndescription: Helps.\n---\nHelp.\n"))
    _agent(newcomer, "---\nname: writer\ndescription: Writes.\n---\nWrite.\n", "AGENT-writer.md")
    listing = _say(chat, "/agent")
    assert "helper" in listing and "writer" in listing and "robutler" in listing and "built in" in listing

    out = _say(chat, "/agent writer")
    assert "Now talking to writer." in out and chat.agent_name == "writer"
    assert "Already talking to writer." in _say(chat, "/agent writer")
    assert "There is no agent called ghost in this folder." in _say(chat, "/agent ghost")


def test_keys_lists_where_each_comes_from_and_sets_and_removes_a_stored_one(newcomer, monkeypatch):
    chat = _chat(_agent(newcomer, "---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n"))
    listing = _say(chat, "/keys")
    assert "Model provider keys" in listing and "OPENAI_API_KEY" in listing and "not set" in listing

    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-entered")
    out = _say(chat, "/keys set OPENAI_API_KEY")
    assert "Stored OPENAI_API_KEY (an owner-only file)." in out
    assert "stored in an owner-only file" in _say(chat, "/keys")
    assert chat.model_problem is None  # the agent now runs on the key

    out = _say(chat, "/keys unset OPENAI_API_KEY")
    assert "Removed OPENAI_API_KEY." in out
    assert "is not a model provider key" in _say(chat, "/keys set NOT_A_KEY")


def test_status_says_who_what_and_where(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    out = _say(chat, "/status")
    assert "Not signed in. /login signs in." in out
    assert "helper (AGENT.md)" in out
    assert "none (no provider key is set). /login, or /keys set <NAME>." in out
    assert "0 messages" in out


def test_sandbox_says_what_the_agents_commands_may_do(newcomer):
    no_shell = _chat(_agent(newcomer, "---\nname: helper\nskills:\n  - filesystem\n---\nHelp.\n"))
    assert "Sandbox: Not needed: this agent cannot run commands." in _say(no_shell, "/sandbox")


def test_an_at_path_includes_a_file_and_leaves_anything_else_as_typed(newcomer):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    (newcomer / "notes.md").write_text("the notes")
    out = chat._expand_file_references("read @notes.md and ask @someone")
    assert '<file path="notes.md">\nthe notes\n</file>' in out and "ask @someone" in out
    assert chat._expand_file_references("mail me@example.com") == "mail me@example.com"


def test_publish_needs_an_agent_file(newcomer):
    chat = _chat(None)  # the built-in agent
    assert chat.agent_name == "robutler"
    out = _say(chat, "/publish")
    assert "Publishing needs an AGENT.md in this folder." in out and "webagents init" in out


def test_the_offer_signs_in_and_the_agent_then_runs_on_robutler(newcomer, monkeypatch):
    chat = _chat(_agent(newcomer, "---\nname: helper\n---\nHelp.\n"))
    assert chat.model_problem

    async def fake_login(api_key=None, say=print):
        say("Opening your browser to confirm the sign-in...")
        monkeypatch.setattr("webagents.cli.model_access.is_signed_in", lambda: True)
        monkeypatch.setattr("webagents.cli.credentials.get_token", lambda *a, **k: "jwt.login")
        return {"username": "dev"}

    monkeypatch.setattr("webagents.cli.platform.auth.login", fake_login)
    answers = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    before = len(chat.console.export_text(clear=False))
    asyncio.run(chat.offer_model_access())
    out = chat.console.export_text(clear=False)[before:]
    assert "Sign in to Robutler and use its models, paid from your credits" in out
    assert "Signed in as @dev on 127.0.0.1:9." in out
    assert chat.model_problem is None and chat.model_label() == "auto/balanced via Robutler"


def test_the_offer_takes_a_key_with_echo_off_and_keeps_it(newcomer, monkeypatch):
    chat = _chat(_agent(newcomer, "---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n"))
    answers = iter(["2"])  # "Enter OPENAI_API_KEY, kept for next time"
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-entered")
    asyncio.run(chat.offer_model_access())
    assert chat.model_problem is None
    from webagents.cli.commands.secrets import _store

    assert _store(quiet=True).get("OPENAI_API_KEY") == "sk-entered"


def test_no_streaming_shows_the_reply_whole(newcomer, monkeypatch):
    """`--no-streaming`: no live region redrawn as the reply arrives; the
    reply is printed once, when it is done, and kept in the conversation."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat = _chat(_agent(newcomer, "---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n"))
    chat.streaming = False
    assert chat.model_problem is None

    from webagents.agents.core.base_agent import BaseAgent

    async def canned(self, messages, **kwargs):
        yield {"choices": [{"delta": {"content": "Hello "}}]}
        yield {"choices": [{"delta": {"content": "world."}}]}

    def no_live(*args, **kwargs):
        raise AssertionError("a live region was drawn with streaming off")

    monkeypatch.setattr(BaseAgent, "run_streaming", canned)
    monkeypatch.setattr("rich.live.Live", no_live)
    out = _say(chat, "hi")
    assert "Hello world." in out
    assert chat.messages[-1] == {"role": "assistant", "content": "Hello world."}


# -- conversations on Robutler (`session: {backend: robutler}`, 2026-09-25) ------------------------

ROBUTLER_AGENT = "---\nname: helper\nskills:\n  - session: {backend: robutler}\n---\nHelp.\n"
AGENT_ID = "9b1f2c3d-4e5f-4a6b-8c7d-0e1f2a3b4c5d"


class FakePlatform:
    """The portal's `/api/agents/{id}/conversations` routes, recording what was asked."""

    def __init__(self):
        self.requests = []
        self.recorded = []
        self.listing = []
        self.chats = {}

    def handler(self, request):
        import httpx

        self.requests.append(request)
        if request.headers.get("authorization") != "Bearer person-token":
            return httpx.Response(401, json={"error": "Unauthorized"})
        path = request.url.path
        base = f"/api/agents/{AGENT_ID}/conversations"
        if request.method == "POST" and path == base:
            body = json.loads(request.content)
            self.recorded.append(body)
            return httpx.Response(201, json={"chatId": body.get("chatId") or "c-terminal", "created": True})
        if request.method == "GET" and path == base:
            return httpx.Response(200, json={"conversations": self.listing})
        if request.method == "GET" and path.startswith(base + "/"):
            chat = path[len(base) + 1:]
            return httpx.Response(200, json={"chatId": chat, "messages": self.chats.get(chat, [])})
        return httpx.Response(404, json={"error": "Agent not found"})


@pytest.fixture
def platform(monkeypatch, newcomer):
    import httpx

    fake = FakePlatform()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(fake.handler), **kw))
    monkeypatch.setenv("ROBUTLER_API_URL", "https://robutler.example")
    return fake


def _signed_in_and_published(folder: Path, monkeypatch) -> None:
    monkeypatch.setenv("WEBAGENTS_TOKEN", "person-token")
    (folder / ".webagents").mkdir(exist_ok=True)
    (folder / ".webagents" / "config.json").write_text(
        json.dumps({"link.agentId": AGENT_ID, "link.agentName": "ada.helper"})
    )


def _record(chat: WebAgentsSession) -> None:
    async def go():
        chat.record_on_robutler()
        await chat.settle_recording()

    asyncio.run(go())


def test_signed_out_resume_says_the_conversations_stay_here(newcomer, platform):
    chat = _chat(_agent(newcomer, ROBUTLER_AGENT))
    out = _say(chat, "/resume")
    assert "Conversations stay on this machine: sign in with `webagents login` to keep them on Robutler too." in out
    assert platform.requests == []


def test_a_turn_is_recorded_into_the_chat_with_the_agent_and_remembered(newcomer, platform, monkeypatch):
    _signed_in_and_published(newcomer, monkeypatch)
    chat = _chat(_agent(newcomer, ROBUTLER_AGENT))
    chat.messages = [{"role": "user", "content": "plan the launch"}, {"role": "assistant", "content": "Step one."}]
    chat.save_conversation()
    _record(chat)

    assert platform.recorded == [
        {
            "sessionId": chat.session_id,
            "messages": [{"role": "user", "content": "plan the launch"}, {"role": "assistant", "content": "Step one."}],
        }
    ]
    assert chat.platform_chat_id == "c-terminal" and chat.recorded_count == 2
    saved = json.loads((sessions_dir(newcomer, "helper") / f"{chat.session_id}.json").read_text())
    assert saved["metadata"]["robutler_chat_id"] == "c-terminal" and saved["metadata"]["robutler_recorded"] == 2

    # The next turn sends only what it added, into the same chat.
    chat.messages += [{"role": "user", "content": "and then?"}, {"role": "assistant", "content": "Step two."}]
    _record(chat)
    assert platform.recorded[-1] == {
        "sessionId": chat.session_id,
        "chatId": "c-terminal",
        "messages": [{"role": "user", "content": "and then?"}, {"role": "assistant", "content": "Step two."}],
    }


def test_resume_lists_robutler_and_continues_a_chat_started_on_the_web(newcomer, platform, monkeypatch):
    _signed_in_and_published(newcomer, monkeypatch)
    chat = _chat(_agent(newcomer, ROBUTLER_AGENT))
    chat.messages = [{"role": "user", "content": "plan the launch"}, {"role": "assistant", "content": "Step one."}]
    chat.save_conversation()
    here = chat.session_id
    _say(chat, "/new")
    platform.listing = [
        {"chatId": "c-web", "sessionId": None, "updatedAt": "2099-01-01T00:00:00.000Z", "messageCount": 2, "preview": "from the web"},
    ]
    platform.chats["c-web"] = [{"role": "user", "content": "from the web"}, {"role": "assistant", "content": "Hello."}]

    listing = _say(chat, "/resume")
    assert "Robutler: from the web" in listing and "plan the launch" in listing

    out = _say(chat, "/resume 1")
    assert "Continuing the conversation" in out and "(2 messages)" in out
    assert chat.messages[0]["content"] == "from the web"
    assert chat.platform_chat_id == "c-web" and chat.recorded_count == 2 and chat.session_id != here
    # Kept on this machine as well from now on.
    assert (sessions_dir(newcomer, "helper") / f"{chat.session_id}.json").exists()


def test_a_failing_platform_is_said_once_and_the_conversation_goes_on(newcomer, platform, monkeypatch):
    _signed_in_and_published(newcomer, monkeypatch)
    monkeypatch.setenv("WEBAGENTS_TOKEN", "expired")
    chat = _chat(_agent(newcomer, ROBUTLER_AGENT))
    chat.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    _record(chat)
    before = len(chat.console.export_text(clear=False))
    chat.say_recording_problem()
    chat.say_recording_problem()
    said = chat.console.export_text(clear=False)[before:]
    assert said.count("This conversation is not being kept on Robutler: your sign-in has expired: run `webagents login`.") == 1
    assert chat.platform_chat_id is None


# -- /undo and /rewind (`checkpoints.py`, 2026-09-25) -------------------------------------------------

FILE_AGENT = "---\nname: helper\nskills: [filesystem]\n---\nHelp.\n"


@pytest.fixture
def answers(monkeypatch):
    """What the person types at a confirmation, at a terminal."""
    typed = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    async def ask(self, question):
        return typed.pop(0) if typed else None

    monkeypatch.setattr(WebAgentsSession, "_ask", ask)
    return typed


def test_undo_puts_back_what_the_last_message_changed(newcomer, answers):
    chat = _chat(_agent(newcomer, FILE_AGENT))
    (newcomer / "plan.md").write_text("first\n")
    chat.snapshot_before_turn("rewrite the plan")
    (newcomer / "plan.md").write_text("rewritten by the agent\n")
    (newcomer / "new.md").write_text("made by the agent\n")

    answers.append("y")
    out = _say(chat, "/undo")

    assert "Undo your last message's changes to this folder:" in out
    assert "restore  plan.md" in out and "remove   new.md" in out
    assert "Put back 1 file and removed 1 file made since." in out
    assert (newcomer / "plan.md").read_text() == "first\n" and not (newcomer / "new.md").exists()
    assert "Nothing to undo in this conversation." in _say(chat, "/undo")


def test_undo_asks_first_and_a_no_leaves_the_folder_alone(newcomer, answers):
    chat = _chat(_agent(newcomer, FILE_AGENT))
    (newcomer / "plan.md").write_text("first\n")
    chat.snapshot_before_turn("rewrite the plan")
    (newcomer / "plan.md").write_text("rewritten\n")

    answers.append("n")
    assert "Left as it is." in _say(chat, "/undo")
    assert (newcomer / "plan.md").read_text() == "rewritten\n"


def test_undo_when_the_last_message_changed_nothing(newcomer, answers):
    chat = _chat(_agent(newcomer, FILE_AGENT))
    chat.snapshot_before_turn("just asking")
    assert "Nothing to undo: the folder is as it was before your last message." in _say(chat, "/undo")


def test_undo_is_off_in_the_home_folder(newcomer, monkeypatch):
    home = Path(os.environ["HOME"])
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(home)
    _agent(home, FILE_AGENT)
    chat = _chat(home / "AGENT.md")
    chat.snapshot_before_turn("anything")
    assert chat.turn_snapshots == []
    assert "/undo is off in your home folder and above: start the chat in a project folder to use it." in _say(chat, "/undo")


def test_rewind_lists_the_snapshots_and_puts_one_back(newcomer, answers):
    chat = _chat(_agent(newcomer, FILE_AGENT))
    assert "No snapshots of this folder yet." in _say(chat, "/rewind")
    (newcomer / "plan.md").write_text("first\n")
    chat.snapshot_before_turn("plan the launch")
    (newcomer / "plan.md").write_text("second\n")

    listing = _say(chat, "/rewind")
    assert "Snapshots of this folder" in listing and 'before "plan the launch"' in listing
    assert "Put the folder back with /rewind <number>." in listing
    assert "There is no snapshot 9." in _say(chat, "/rewind 9")

    answers.append("yes")
    out = _say(chat, "/rewind 1")
    assert 'Put the folder back as it was just now (before "plan the launch"):' in out
    assert "Put back 1 file." in out and (newcomer / "plan.md").read_text() == "first\n"
    # The restore took a snapshot of how things were first, so /rewind can go back.
    assert "before /rewind" in _say(chat, "/rewind")
