"""
Search and pickers in the chat's menu (2026-09-30, the owner: "/resume and
other commands/subcommands should have search/filter on typing and up/down
arrow selection for the filtered options"), the same in the TypeScript box
(`typescript/tests/unit/cli/chat-menu-search.test.ts`).

Pinned here:
  * how rows are found (`rank_rows`), and how a completer splits what is typed
    (`argument_words`, `rest_after`), by the fixture's `menu_search`;
  * in the box: a query of several words; enter on a row that completes the
    command sends the line with the text that row inserts, enter on any other
    row inserts it, tab always inserts; enter on `/resume` opens its list;
  * in the chat: `/resume` lists this folder's conversations numbered as the
    command numbers them, each inserting the start of its id, `delete` last
    and `/resume delete` choosing among the same; an id that is all digits
    still finds its conversation; `/rewind` lists the snapshots; `/model` the
    models the agent may switch to; `/help` finds a command by its
    description; a key to set runs, a key to remove is inserted.
"""

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer
from rich.console import Console

from webagents.cli import checkpoints
from webagents.cli.repl.commands import COMPLETED_COMMANDS, MODEL_TIERS, PICKER_COMMANDS
from webagents.cli.repl.session import WebAgentsSession, _pick_conversation
from webagents.cli.sessions import new_session_id, save_session
from webagents.cli.ui.prompt_box import COMMAND_SEARCH_MIN, MenuRow, PromptBox, Slot, argument_words, rank_rows, rest_after
from webagents.cli.ui.theme import theme_for

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_commands.json").read_text())
SEARCH = FIXTURE["menu_search"]


@pytest.mark.parametrize("case", SEARCH["cases"], ids=[c["about"] for c in SEARCH["cases"]])
def test_rows_are_found_as_the_fixture_says(case):
    rows = [MenuRow(*row) for row in case["rows"]]
    assert [r.value for r in rank_rows(rows, case["query"], commands=case["commands"])] == case["expected"]


def test_a_completer_splits_what_is_typed_as_the_fixture_says():
    assert COMMAND_SEARCH_MIN == SEARCH["command_search_min"]
    for case in SEARCH["argument_words"]:
        assert argument_words(case["args"]) == (case["before"], case["partial"]), case
    for case in SEARCH["rest_after"]:
        assert rest_after(case["args"], case["words"]) == case["rest"], case


def test_the_constants_are_the_fixtures():
    assert list(COMPLETED_COMMANDS) == FIXTURE["completion"]["commands"]
    assert list(PICKER_COMMANDS) == FIXTURE["completion"]["pickers"]
    assert list(MODEL_TIERS) == FIXTURE["completion"]["model_tiers"]


# -- the box ------------------------------------------------------------------

CONVERSATIONS = [
    MenuRow("1", "2h ago · 14 messages · plan the launch", "5b1f2c9a-0000", runs=True, insert="5b1f2c9a"),
    MenuRow("2", "yesterday · 3 messages · fix the budget sheet", "0c3d4e5f-0000", runs=True, insert="0c3d4e5f"),
]


def _resume(args: str):
    words = args.split()
    if words[:1] == ["delete"] and (len(words) > 1 or args[-1:].isspace()):
        return Slot(CONVERSATIONS, rest_after(args, 1))
    return Slot(CONVERSATIONS + [MenuRow("delete", "delete an earlier conversation")], rest_after(args, 0))


def _mcp(args: str):
    before, partial = argument_words(args)
    if before == ["remove"]:
        return Slot([MenuRow("sqlite", "this agent's server")], partial)
    return None if before else Slot([MenuRow("remove", "take one out")], partial)


def _box(pickers=lambda name: name == "resume") -> PromptBox:
    return PromptBox(
        theme_for(Console(file=StringIO(), width=100, force_terminal=False, color_system=None)),
        commands=[("/resume", "Continue an earlier conversation, or delete one"), ("/mcp", "The MCP servers"), ("/login", "Sign in to Robutler")],
        footer=lambda: [],
        completers={"resume": _resume, "mcp": _mcp},
        pickers=pickers,
    )


def test_a_query_of_several_words_narrows_the_list():
    box = _box()
    assert [r.value for r in box.menu_items("/resume ")] == ["1", "2", "delete"]
    assert [r.value for r in box.menu_items("/resume launch")] == ["1"]
    assert [r.value for r in box.menu_items("/resume the bud")] == ["2"]
    assert box.menu_items("/resume nothing like it") == []
    # The command list finds a command by a word of its description.
    assert [r.value for r in box.menu_items("/sign")] == ["/login"]


def test_enter_on_a_row_that_completes_the_command_sends_the_line_with_what_it_inserts():
    box = _box()
    assert box.enter_choice("/resume launch pl") == ("send", "/resume 5b1f2c9a")
    box.menu_index = 1
    assert box.enter_choice("/resume ") == ("send", "/resume 0c3d4e5f")
    box.menu_index = 0
    assert box.enter_choice("/resume delete bud") == ("send", "/resume delete 0c3d4e5f")
    # The number itself, typed out, is sent as typed.
    assert box.enter_choice("/resume 2") == ("send", "/resume 2")


def test_enter_on_any_other_row_inserts_it_and_tab_always_inserts():
    box = _box()
    assert box.enter_choice("/resume del") == ("insert", "delete")
    assert box.enter_choice("/mcp remove sq") == ("insert", "sqlite")
    buffer = Buffer()
    buffer.text = "/resume launch pl"
    box._insert_argument(buffer, "5b1f2c9a")
    assert buffer.text == "/resume 5b1f2c9a "


def test_enter_on_a_picker_opens_its_list_when_there_is_one_to_choose():
    assert _box().enter_choice("/res") == ("open", "/resume ")
    assert _box(pickers=lambda _name: False).enter_choice("/res") == ("send", "/resume")
    # No conversation to choose: the command runs, and says so.
    box = _box()
    box.completers["resume"] = lambda args: Slot([], rest_after(args, 0))
    assert box.enter_choice("/res") == ("send", "/resume")


# -- the chat -----------------------------------------------------------------


@pytest.fixture
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def _chat(agent_file: str) -> WebAgentsSession:
    (Path.cwd() / "AGENT.md").write_text(agent_file)
    session = WebAgentsSession(agent_path=Path("AGENT.md"), interactive=True)
    session.console = Console(file=StringIO(), width=100, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    asyncio.run(session._refresh_completion_data())
    return session


def _said_by(session: WebAgentsSession, line: str) -> str:
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.handle_input(line))
    return session.console.export_text(clear=False)[before:]


def _kept(session: WebAgentsSession, first_message: str, session_id: str = "") -> str:
    sid = session_id or new_session_id()
    save_session(
        session.session_dir(),
        {"session_id": sid, "agent_name": "helper", "messages": [{"role": "user", "content": first_message}, {"role": "assistant", "content": "ok"}], "metadata": {}},
    )
    return sid


HELPER = "---\nname: helper\nskills:\n  - openai\nmodel: openai/gpt-4o-mini\n---\nHelp.\n"


def test_resume_lists_the_conversations_as_the_command_numbers_them(newcomer):
    chat = _chat(HELPER)
    older = _kept(chat, "fix the budget sheet")
    newer = _kept(chat, "plan the launch")
    asyncio.run(chat._refresh_completion_data())
    slot = chat._completers()["resume"](" ")
    assert [r.value for r in slot.rows] == ["1", "2", "delete"]
    first, second = slot.rows[0], slot.rows[1]
    assert first.description.endswith("· 2 messages · plan the launch") and first.runs and first.insert == newer[:8]
    assert second.insert == older[:8]
    assert not slot.rows[2].runs
    # `/resume delete ` chooses among the same conversations.
    assert [r.value for r in chat._completers()["resume"](" delete ").rows] == ["1", "2"]
    # What a row inserts finds that conversation when the command runs.
    entries = chat._conversation_entries([])
    assert _pick_conversation(entries, first.insert).id == newer
    assert _pick_conversation(entries, "2").id == older


def test_an_id_that_is_all_digits_still_finds_its_conversation(newcomer):
    chat = _chat(HELPER)
    digits = "12345678-0000-4000-8000-000000000000"
    _kept(chat, "plan the launch", digits)
    entries = chat._conversation_entries([])
    assert _pick_conversation(entries, "12345678").id == digits
    assert _pick_conversation(entries, "1").id == digits


def test_a_short_number_is_a_position_never_the_start_of_an_id(newcomer):
    # The gate run of 2026-09-30 caught this: `/resume 9` with one conversation
    # continued it when its random id happened to start with 9.
    chat = _chat(HELPER)
    nine = "9aaaaaaa-0000-4000-8000-000000000000"
    _kept(chat, "plan the launch", nine)
    entries = chat._conversation_entries([])
    assert _pick_conversation(entries, "9") is None
    assert "There is no conversation 9." in _said_by(chat, "/resume 9")
    assert _pick_conversation(entries, "9aaa").id == nine


def test_a_new_folder_offers_no_conversation_and_no_delete(newcomer):
    chat = _chat(HELPER)
    assert chat._completers()["resume"](" ").rows == []
    assert chat._completers()["resume"](" delete ") is None


def test_rewind_lists_the_snapshots(newcomer):
    chat = _chat(HELPER)
    (newcomer / "notes.txt").write_text("one")
    checkpoints.take_snapshot(newcomer, "before: plan the launch")
    asyncio.run(chat._refresh_completion_data())
    rows = chat._completers()["rewind"](" ").rows
    assert [r.value for r in rows] == ["1"] and rows[0].runs
    assert rows[0].description.endswith("· before: plan the launch")


def test_model_offers_the_declared_providers_models_with_the_one_in_use_first(newcomer, monkeypatch):
    # "in use" is the model the agent runs on: with a key for it.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    chat = _chat(HELPER)
    rows = chat._completers()["model"](" ").rows
    assert rows[0] == MenuRow("openai/gpt-4o-mini", "in use", runs=True)
    assert all(r.value.startswith("openai/") for r in rows)
    assert [r.value for r in rank_rows(*chat._completers()["model"](" 4.1"))] == ["openai/gpt-4.1"]


def test_model_offers_the_tiers_and_every_known_model_without_a_provider_skill(newcomer):
    chat = _chat("---\nname: helper\nskills:\n  - todo\n---\nHelp.\n")
    values = [r.value for r in chat._completers()["model"](" ").rows]
    assert values[values.index("auto/fastest"):][:3] == list(MODEL_TIERS)
    assert "anthropic/claude-sonnet-4-6" in values and "openai/gpt-4o-mini" in values


def test_help_finds_a_command_by_its_description_and_keys_run_only_to_set(newcomer):
    chat = _chat(HELPER)
    completers = chat._completers()
    assert [r.value for r in rank_rows(*completers["help"](" sign in"))] == ["login"]
    assert all(r.runs for r in completers["keys"](" set ").rows)
    assert not any(r.runs for r in completers["keys"](" remove ").rows)


def test_enter_on_resume_opens_the_list_unless_the_conversations_are_on_robutler_too(newcomer, monkeypatch):
    chat = _chat(HELPER)
    _kept(chat, "plan the launch")
    asyncio.run(chat._refresh_completion_data())
    assert chat.prompt_box.pickers("resume") and chat.prompt_box.pickers("rewind")
    assert chat.prompt_box.enter_choice("/resume") == ("open", "/resume ")
    monkeypatch.setattr(WebAgentsSession, "session_backend", property(lambda _self: "robutler"))
    assert not chat.prompt_box.pickers("resume")
