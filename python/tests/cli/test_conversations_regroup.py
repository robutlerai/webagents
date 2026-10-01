"""
The chat's commands regrouped, and the conversation's life cycle (2026-09-29),
the same in the TypeScript chat (`typescript/tests/unit/cli/conversations-regroup.test.ts`).

The owner asked for a better grouping of the commands, for one level of
subcommands rather than `/agent model`, and how conversations are started,
continued and deleted. Pinned here: `/resume delete <number>` removes one
earlier conversation after asking (never the current one, never a copy on
Robutler); the chat starts a new conversation and says, in one faint line,
when the last one here is less than a day old; `-c` and `-r [number]` open the
chat on an earlier one; `/keys remove` is the verb (`unset` still taken); and
`new` and `edit`, the words `/agent` keeps for itself, are refused as names.

Every test runs under a throwaway HOME with the file secrets backend, no
provider key and no sign-in.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from webagents.cli.init_templates import RESERVED_AGENT_NAMES, RESERVED_NAME, reserved_name
from webagents.cli.main import app
from webagents.cli.repl.chat_words import CHAT_WORDS, fill
from webagents.cli.repl.session import WebAgentsSession
from webagents.cli.sessions import delete_session, save_session, sessions_dir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cli"
GRAMMAR = json.loads((FIXTURES / "init_templates.json").read_text())["name_grammar"]
JSON_ERRORS = json.loads((FIXTURES / "json_errors.json").read_text())
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY")
AGENT = "---\nname: helper\n---\nHelp.\n"
runner = CliRunner()


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in KEY_VARS + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    (project / "AGENT.md").write_text(AGENT)
    yield project


def _chat(resume=None, answer=True) -> WebAgentsSession:
    session = WebAgentsSession(agent_path=Path("AGENT.md"), interactive=True, resume=resume)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())

    async def confirm(question: str) -> bool:
        session.asked = question  # type: ignore[attr-defined]
        return answer

    session._confirm = confirm  # type: ignore[method-assign]
    return session


def _say(session: WebAgentsSession, line: str) -> str:
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.handle_input(line))
    return session.console.export_text(clear=False)[before:]


def _printed_by(session: WebAgentsSession, call) -> str:
    before = len(session.console.export_text(clear=False))
    result = call()
    if asyncio.iscoroutine(result):
        asyncio.run(result)
    return session.console.export_text(clear=False)[before:]


def _saved(folder: Path, session_id: str, text: str, when: datetime, chat_id=None) -> None:
    """A conversation as the chat saves it, last used at `when`."""
    directory = sessions_dir(folder, "helper")
    save_session(directory, {
        "session_id": session_id,
        "agent_name": "helper",
        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}],
        "metadata": {"robutler_chat_id": chat_id} if chat_id else {},
        "input_tokens": 0,
        "output_tokens": 0,
    })
    file = directory / f"{session_id}.json"
    data = json.loads(file.read_text())
    data["updated_at"] = when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    file.write_text(json.dumps(data))


# ---------------------------------------------------------------- deleting


def test_resume_delete_removes_an_earlier_conversation_after_asking(newcomer):
    now = datetime.now(timezone.utc)
    _saved(newcomer, "11111111-1111-4111-8111-111111111111", "the older one", now - timedelta(hours=5))
    _saved(newcomer, "22222222-2222-4222-8222-222222222222", "the newer one", now - timedelta(hours=1))
    directory = sessions_dir(newcomer, "helper")

    chat = _chat(answer=False)
    out = _say(chat, "/resume delete 2")
    assert chat.asked == fill("resumeDeleteAsk", when="5 h ago", count=2)
    assert CHAT_WORDS["resumeNotDeleted"] in out
    assert (directory / "11111111-1111-4111-8111-111111111111.json").exists()

    chat = _chat(answer=True)
    out = _say(chat, "/resume delete 2")
    assert fill("resumeDeleted", when="5 h ago") in out
    assert CHAT_WORDS["resumeDeletedRemote"] not in out
    assert not (directory / "11111111-1111-4111-8111-111111111111.json").exists()
    listing = _say(chat, "/resume")
    assert "the newer one" in listing and "the older one" not in listing
    assert "/resume delete <number> deletes one" in listing


def test_resume_delete_says_what_it_cannot_do(newcomer):
    now = datetime.now(timezone.utc)
    _saved(newcomer, "33333333-3333-4333-8333-333333333333", "on robutler too", now - timedelta(hours=2), chat_id="chat-1")
    chat = _chat()
    assert fill("usage", usage="/resume delete <number>") in _say(chat, "/resume delete")
    out = _say(chat, "/resume delete 7")
    assert fill("resumeNoNumber", pick="7") in out and CHAT_WORDS["resumeNoNumberHint"] in out
    out = _say(chat, "/resume delete 1")
    assert fill("resumeDeleted", when="2 h ago") in out
    assert CHAT_WORDS["resumeDeletedRemote"] in out


def test_the_current_conversation_is_not_in_the_list_so_it_cannot_be_deleted(newcomer):
    chat = _chat()
    chat.messages = [{"role": "user", "content": "the current one"}, {"role": "assistant", "content": "ok"}]
    chat.save_conversation()
    out = _say(chat, "/resume delete 1")
    assert "No earlier conversations with helper in this folder." in out
    assert (sessions_dir(newcomer, "helper") / f"{chat.session_id}.json").exists()


def test_delete_session_takes_the_latest_pointer_with_it(tmp_path):
    save_session(tmp_path, {"session_id": "a", "agent_name": "x", "messages": [], "metadata": {}})
    save_session(tmp_path, {"session_id": "b", "agent_name": "x", "messages": [], "metadata": {}})
    assert (tmp_path / ".latest").read_text() == "b"
    assert delete_session(tmp_path, "a") and (tmp_path / ".latest").read_text() == "b"
    assert delete_session(tmp_path, "b") and not (tmp_path / ".latest").exists()
    assert delete_session(tmp_path, "b") is False


# ---------------------------------------------------------------- starting


def test_the_start_says_when_the_last_conversation_here_is_recent(newcomer):
    now = datetime.now(timezone.utc)
    _saved(newcomer, "44444444-4444-4444-8444-444444444444", "yesterday's plan", now - timedelta(hours=3))
    chat = _chat()
    out = _printed_by(chat, chat._start_where_asked)
    assert fill("lastConversation", when="3 h ago", count=2) in out
    assert chat.messages == [], "the chat itself starts a new conversation"


def test_an_old_conversation_is_not_mentioned(newcomer):
    _saved(newcomer, "55555555-5555-4555-8555-555555555555", "last week", datetime.now(timezone.utc) - timedelta(days=3))
    chat = _chat()
    assert _printed_by(chat, chat._start_where_asked).strip() == ""


def test_dash_c_continues_the_last_one_and_dash_r_lists_them(newcomer):
    now = datetime.now(timezone.utc)
    _saved(newcomer, "66666666-6666-4666-8666-666666666666", "the older one", now - timedelta(hours=5))
    _saved(newcomer, "77777777-7777-4777-8777-777777777777", "the newer one", now - timedelta(hours=1))
    chat = _chat(resume="1")
    out = _printed_by(chat, chat._start_where_asked)
    assert "Continuing the conversation" in out and chat.messages[0]["content"] == "the newer one"
    chat = _chat(resume="")
    out = _printed_by(chat, chat._start_where_asked)
    assert "Earlier conversations" in out and "the older one" in out and chat.messages == []
    chat = _chat(resume="2")
    _printed_by(chat, chat._start_where_asked)
    assert chat.messages[0]["content"] == "the older one"


def test_the_flags_reach_the_chat_and_refuse_what_does_not_go_together():
    seen = []
    with mock.patch("webagents.cli.repl.session.start_repl", side_effect=lambda **kw: seen.append(kw.get("resume"))):
        for argv, resume in ((["-c"], "1"), (["-r"], ""), (["-r", "2"], "2"), (["chat", "-r"], ""), (["chat", "--resume", "3"], "3"), ([], None)):
            result = runner.invoke(app, argv)
            assert result.exit_code == 0, (argv, result.output)
            assert seen[-1] == resume, argv
    both = runner.invoke(app, ["-c", "-r"])
    assert both.exit_code == 2 and "Use --continue or --resume, not both." in both.output
    with_prompt = runner.invoke(app, ["-c", "-p", "hi"])
    assert with_prompt.exit_code == 2 and "--continue and --resume open the chat; they do not go with -p." in with_prompt.output


# ---------------------------------------------------------------- the subcommand rule


def test_keys_remove_is_the_verb_and_unset_is_still_taken(newcomer):
    chat = _chat()
    out = _say(chat, "/keys")
    assert "/keys remove <NAME> removes a stored one" in out
    assert "Usage: /keys [set|remove NAME]" in _say(chat, "/keys drop OPENAI_API_KEY")
    assert "OPENAI_API_KEY was not stored." in _say(chat, "/keys remove OPENAI_API_KEY")
    assert "OPENAI_API_KEY was not stored." in _say(chat, "/keys unset OPENAI_API_KEY")


def test_the_words_agent_keeps_for_itself_are_refused_as_names(newcomer):
    assert list(RESERVED_AGENT_NAMES) == GRAMMAR["reserved"]
    assert RESERVED_NAME == GRAMMAR["reserved_sentence"] == JSON_ERRORS["reserved_name"]["message"]
    assert reserved_name("edit") and reserved_name("some/where/New") and not reserved_name("editor")
    chat = _chat()
    out = _say(chat, "/agent new edit")
    assert RESERVED_NAME.replace("{name}", "edit") in out
    assert not Path("AGENT-edit.md").exists()
    result = runner.invoke(app, ["init", "new"])
    assert result.exit_code == 1 and RESERVED_NAME.replace("{name}", "new") in result.output
    assert not Path("new").exists()
    as_json = runner.invoke(app, ["--json", "init", "edit"])
    assert as_json.exit_code == JSON_ERRORS["reserved_name"]["exit"]
    assert json.loads(as_json.stdout)["error"]["code"] == "reserved_name"
