"""
`webagents mcp list` / `mcp add` and the chat's `/mcp list` / `/mcp add`
(`webagents/cli/mcp_import.py`, 2026-09-29), against the shared fixture
`tests/fixtures/mcp_tool/import.json`, which the TypeScript suite reads too
(`typescript/tests/unit/cli/mcp-import.test.ts`).

The owner asked whether the MCP servers already set up in other apps could be
listed and used. What is pinned: the settings files read and how each app's
entry is read; that a listing never prints a key (a key-looking value, the
value after a flag named like a key, a key in an address's query, variable and
header values); that `add` moves keys into this profile's secret store and the
entry reads `${secret:NAME}`, refuses a key on the command line, turns VS
Code's inputs into secrets to set, and writes the entry where the agent reads
its servers without touching any other byte of the agent file. Then the real
command and the chat command, in a temporary home with the file secret store,
so nothing reaches the owner's own settings or keystore.
"""

from __future__ import annotations

import asyncio
import json
import os
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pytest
from rich.console import Console
from typer.testing import CliRunner

from webagents.cli import mcp_import as m
from webagents.cli.main import app

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "mcp_tool" / "import.json").read_text(encoding="utf-8"))
runner = CliRunner()


def _found(raw: dict) -> m.FoundServer:
    return m.FoundServer(raw["app_id"], raw["app"], raw["file"], raw["name"], raw["entry"])


class FakeStore:
    """The store `convert_entry` writes keys to, in memory."""

    def __init__(self, held: Optional[Dict[str, str]] = None):
        self.held = dict(held or {})
        self.writes = 0

    def get(self, name: str) -> Optional[str]:
        return self.held.get(name)

    def set(self, name: str, value: str) -> None:
        self.held[name] = value
        self.writes += 1


# ---------------------------------------------------------------- the words


def test_the_words_and_apps_are_the_fixtures():
    assert m.WORDS == FIXTURE["words"]
    assert [list(a) for a in m.APPS] == FIXTURE["apps"]


@pytest.mark.parametrize("system", ["darwin", "linux"])
def test_the_settings_files_read(system):
    case = FIXTURE["settings_files"][system]
    got = m.settings_files(Path(case["home"]), Path(FIXTURE["settings_files"]["folder"]), system)
    assert [[a, b, str(f), s] for a, b, f, s in got] == case["files"]


def test_windows_reads_appdata():
    files = m.settings_files(Path("/h"), Path("/w"), "win32", appdata="/appdata")
    assert files[0][2] == Path("/appdata") / "Claude" / "claude_desktop_config.json"
    assert files[5][2] == Path("/appdata") / "Code" / "User" / "mcp.json"


# ---------------------------------------------------------------- reading


@pytest.mark.parametrize("case", FIXTURE["jsonc"], ids=lambda c: c["about"])
def test_jsonc_comments_and_trailing_commas_go_strings_stay(case):
    assert json.loads(m.strip_jsonc(case["text"])) == case["data"]


@pytest.mark.parametrize("case", FIXTURE["normalize"], ids=lambda c: c["about"])
def test_another_apps_entry_in_this_sdks_shape(case):
    assert m.normalize_entry(case["raw"]) == case["entry"]


def test_a_command_that_is_a_file_with_spaces_is_not_split(tmp_path):
    server = tmp_path / "My Server" / "run server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\n")
    assert m.normalize_entry({"command": str(server)}) == {"command": str(server), "args": []}


def test_discover_reads_every_app_and_says_which_file_it_could_not(tmp_path):
    case = FIXTURE["discover"]
    home, folder = tmp_path / "home", tmp_path / "folder"
    home.mkdir()
    folder.mkdir()
    real = folder.resolve()
    for relative, text in case["files"].items():
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text.replace("{folder}", str(real)), encoding="utf-8")

    def shown(file: str) -> str:
        return file.replace(str(real), "folder").replace(str(home), "home")

    found = m.discover(home, folder, case["system"], appdata="")
    assert [[f.app_id, f.app, shown(f.file), f.name, f.entry] for f in found.found] == case["found"]
    assert [[a, shown(f), r] for a, f, r in found.unreadable] == case["unreadable"]


def test_a_settings_file_is_read_again_when_it_changes(tmp_path):
    """Parsed files are kept between reads (the chat reads them before every
    prompt), and read again once their size or time stamp changes. A
    byte-order mark is dropped, and bytes that are not UTF-8 are replaced, as
    Node does."""
    home, folder = tmp_path / "home", tmp_path / "folder"
    (home / ".cursor").mkdir(parents=True)
    folder.mkdir()
    file = home / ".cursor" / "mcp.json"
    file.write_text("\ufeff" + json.dumps({"mcpServers": {"a": {"command": "x"}}}), encoding="utf-8")
    assert [f.name for f in m.discover(home, folder, "linux").found] == ["a"]
    file.write_text(json.dumps({"mcpServers": {"a": {"command": "x"}, "bb": {"command": "y"}}}), encoding="utf-8")
    assert [f.name for f in m.discover(home, folder, "linux").found] == ["a", "bb"]
    file.write_bytes(b'{"mcpServers": {"c": {"command": "caf\xe9"}}}')
    assert [f.entry for f in m.discover(home, folder, "linux").found] == [{"command": "caf\ufffd", "args": []}]


# ---------------------------------------------------------------- showing


@pytest.mark.parametrize("case", FIXTURE["masked_args"], ids=lambda c: " ".join(c["args"]))
def test_keys_on_a_command_line_are_masked(case):
    masked, carried = m.masked_args(case["args"])
    assert masked == case["masked"]
    assert carried is case["carried"]


@pytest.mark.parametrize("case", FIXTURE["describe"], ids=lambda c: c["says"])
def test_one_line_per_entry_names_never_values(case):
    assert m.describe(case["entry"]) == case["says"]


@pytest.mark.parametrize("case", FIXTURE["list"]["cases"], ids=lambda c: c["about"])
def test_the_listing(case):
    discovery = m.Discovery([_found(f) for f in case["found"]], [tuple(u) for u in case["unreadable"]])
    assert m.list_lines(discovery, Path(FIXTURE["list"]["home"])) == case["lines"]


# ---------------------------------------------------------------- converting


@pytest.mark.parametrize("case", FIXTURE["convert"]["cases"], ids=lambda c: c["about"])
def test_keys_move_to_the_secret_store(case):
    store = FakeStore(case["store"])
    found = _found(case["found"])
    if "refused" in case:
        with pytest.raises(m.AddRefused) as refused:
            m.convert_entry(found, Path(FIXTURE["convert"]["folder"]), store)
        assert str(refused.value) == case["refused"]
        assert store.held == case["store"]
        return
    converted = m.convert_entry(found, Path(FIXTURE["convert"]["folder"]), store)
    assert converted.entry == case["entry"]
    assert converted.stored == case["stored"]
    assert converted.to_set == case["to_set"]
    assert store.held == case["store_after"]
    if "writes" in case:
        assert store.writes == case["writes"]
    assert found.entry == case["found"]["entry"], "the discovered entry is not changed"


# ---------------------------------------------------------------- writing


@pytest.mark.parametrize("case", FIXTURE["insert"], ids=lambda c: c["about"])
def test_the_entry_goes_into_the_agent_files_own_mcp_block(case):
    if case.get("cannot_edit"):
        with pytest.raises(m.CannotEdit):
            m.insert_into_agent_file(case["text"], case["name"], case["entry"], "AGENT.md")
        return
    assert m.insert_into_agent_file(case["text"], case["name"], case["entry"], "AGENT.md") == case["after"]


@pytest.mark.parametrize("case", FIXTURE["choose"]["cases"], ids=lambda c: f"{c['name']} from {c['from']}")
def test_choosing_the_server(case):
    choose = FIXTURE["choose"]
    discovery = m.Discovery([_found(f) for f in choose["found"]], [])
    if "refused" in case:
        with pytest.raises(m.AddRefused) as refused:
            m.choose(discovery, case["name"], case["from"], choose["list_command"])
        assert str(refused.value) == case["refused"]
        return
    assert discovery.found.index(m.choose(discovery, case["name"], case["from"], choose["list_command"])) == case["picks"]


@pytest.mark.parametrize("case", FIXTURE["plan"], ids=lambda c: c["about"])
def test_where_the_entry_is_written(case, tmp_path):
    agent_file = tmp_path / case["agent_file"]
    agent_file.write_text(case["agent"], encoding="utf-8")
    if case["mcp_json"] is not None:
        (tmp_path / "mcp.json").write_text(case["mcp_json"], encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    store = FakeStore(case["store"])
    name = m.agent_name_of(agent_file, case["agent"])
    if "refused" in case:
        with pytest.raises(m.AddRefused) as refused:
            m.plan_add(_found(case["found"]), agent_file, store, name)
        assert str(refused.value) == case["refused"]
    else:
        plan = m.plan_add(_found(case["found"]), agent_file, store, name)
        assert plan.where == case["where"]
        assert {file.name: text for file, text in plan.writes} == case["writes"]
        assert plan.converted.stored == case["stored"]
        m.write_plan(plan)
        for file, text in case["writes"].items():
            assert (tmp_path / file).read_text(encoding="utf-8") == text
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    unchanged = {k: v for k, v in before.items() if k not in case.get("writes", {})}
    assert {k: after[k] for k in unchanged} == unchanged


@pytest.mark.parametrize("case", FIXTURE["agent_name"], ids=lambda c: c["file"])
def test_the_agents_name(case):
    assert m.agent_name_of(Path(case["file"]), case["text"]) == case["name"]


def test_the_agent_file_a_path_names(tmp_path, monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "AGENT.md").write_text("x")
    assert m.agent_file_for(tmp_path / "one") == tmp_path / "one" / "AGENT.md"
    assert m.agent_file_for(tmp_path / "one" / "AGENT.md") == tmp_path / "one" / "AGENT.md"
    (tmp_path / "named").mkdir()
    (tmp_path / "named" / "AGENT-bob.md").write_text("x")
    assert m.agent_file_for(tmp_path / "named") == tmp_path / "named" / "AGENT-bob.md"
    (tmp_path / "named" / "AGENT-amy.md").write_text("x")
    with pytest.raises(m.AddRefused) as many:
        m.agent_file_for(tmp_path / "named")
    assert str(many.value) == f"More than one agent file in {tmp_path / 'named'}; name one: AGENT-amy.md, AGENT-bob.md."
    (tmp_path / "empty").mkdir()
    with pytest.raises(m.AddRefused) as none:
        m.agent_file_for(tmp_path / "empty")
    assert str(none.value) == f"No agent file in {tmp_path / 'empty'}; webagents init makes one."


def test_what_add_says(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    plan = m.AddPlan([], "mcp.json", m.Converted({}, stored=["REMOTE_TOKEN"], to_set=["API_KEY", "GITHUB_TOKEN"]))
    assert m.result_lines(plan, "remote", "helper", in_chat=True) == [
        "Added remote to helper (mcp.json).",
        "Stored REMOTE_TOKEN in this profile's secrets; the entry reads them as ${secret:NAME}.",
        "Set API_KEY, GITHUB_TOKEN before it connects: `webagents secrets set API_KEY`, `webagents secrets set GITHUB_TOKEN`.",
        "/reload connects it.",
    ]
    bare = m.AddPlan([], "AGENT.md", m.Converted({}))
    assert m.result_lines(bare, "x", "yoyo", in_chat=False) == ["Added x to yoyo (AGENT.md).", "The agent connects it the next time it starts."]


# ---------------------------------------------------------------- the commands


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temporary home with the file secret store: nothing reaches the owner's settings or keystore."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "APPDATA"):
        monkeypatch.delenv(var, raising=False)
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "chrome-devtools": {"command": "npx -y chrome-devtools-mcp@latest"},
                    "remote": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer abc123-not-a-key"}},
                }
            }
        )
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text("---\nname: helper\nskills:\n  - memory\n---\nBe brief.\n")
    monkeypatch.chdir(project)
    return home


def _secret(name: str) -> Optional[str]:
    return m.secret_store().get(name)


def test_the_cli_lists_and_adds(home):
    listed = runner.invoke(app, ["mcp", "list"])
    assert listed.exit_code == 0, listed.output
    assert listed.output.splitlines() == [
        "MCP servers other apps on this machine use",
        "Cursor  ~/.cursor/mcp.json",
        "  chrome-devtools  npx -y chrome-devtools-mcp@latest",
        "  remote  https://mcp.example.com/mcp  headers Authorization",
        "Add one to an agent: webagents mcp add <name>, or /mcp add <name> in the chat.",
    ]
    assert "abc123" not in listed.output

    added = runner.invoke(app, ["mcp", "add", "remote"])
    assert added.exit_code == 0, added.output
    assert added.output.splitlines() == [
        "Added remote to helper (mcp.json).",
        "Stored REMOTE_TOKEN in this profile's secrets; the entry reads them as ${secret:NAME}.",
        "The agent connects it the next time it starts.",
    ]
    assert _secret("REMOTE_TOKEN") == "abc123-not-a-key"
    written = json.loads(Path("mcp.json").read_text())
    assert written == {"mcpServers": {"remote": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ${secret:REMOTE_TOKEN}"}}}}
    assert "abc123" not in Path("mcp.json").read_text() + Path("AGENT.md").read_text()
    assert Path("AGENT.md").read_text() == "---\nname: helper\nskills:\n  - memory\n  - mcp\n---\nBe brief.\n"

    again = runner.invoke(app, ["mcp", "add", "remote"])
    assert again.exit_code == 1
    assert "helper already has an MCP server named remote." in again.output

    missing = runner.invoke(app, ["mcp", "add", "nope"])
    assert missing.exit_code == 1
    assert "No MCP server named nope in other apps' settings; webagents mcp list shows them." in missing.output


def test_the_chat_lists_and_adds(home, monkeypatch):
    """In a loaded chat: the add's own edit of the agent file (the `- mcp` it
    adds) is not announced as a change, since its own last line says /reload
    connects it; an edit from outside after it still is."""
    from webagents.cli import credentials
    from webagents.cli.repl.session import WebAgentsSession

    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for var in ("ANTHROPIC_API_KEY", "WEBAGENTS_TOKEN", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    credentials.set_flag_token(None)
    Path("AGENT.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBe brief.\n")
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=200, force_terminal=False, color_system=None, record=True)

    async def main() -> str:
        await session.initialize()
        await session.handle_input("/mcp list")
        await session.handle_input("/mcp add chrome-devtools")
        session._say_file_changed()
        added = session.console.export_text(clear=False)
        assert "changed since the chat loaded it" not in added
        await session.handle_input("/mcp add chrome-devtools")
        await session.handle_input("/mcp add")
        Path("AGENT.md").write_text(Path("AGENT.md").read_text() + "Be kind.\n")
        session._say_file_changed()
        await session._cleanup_agent(session.built)
        return session.console.export_text()

    out = asyncio.run(main())
    assert "  chrome-devtools  npx -y chrome-devtools-mcp@latest" in out
    assert "Added chrome-devtools to helper (mcp.json)." in out
    assert "/reload connects it." in out
    assert "helper already has an MCP server named chrome-devtools." in out
    assert "/mcp [list | add <name> [--from <app>] | remove <name>]" in out
    assert "AGENT.md changed since the chat loaded it. /reload uses the new version." in out
    assert json.loads(Path("mcp.json").read_text())["mcpServers"]["chrome-devtools"] == {"command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"]}
    assert Path("AGENT.md").read_text().startswith("---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - mcp\n---\n")
    assert os.environ["HOME"] == str(home)


# ---------------------------------------------------------------- removing


@pytest.mark.parametrize("case", FIXTURE["remove"]["cases"], ids=lambda c: c["about"])
def test_the_entry_comes_out_of_the_agent_files_own_mcp_block(case):
    if case.get("cannot_edit"):
        with pytest.raises(m.CannotEdit):
            m.remove_from_agent_file(case["text"], case["name"], "AGENT.md")
        return
    assert m.remove_from_agent_file(case["text"], case["name"], "AGENT.md") == case["after"]


def test_remove_says_where_and_which_secrets_stay(tmp_path, monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    (tmp_path / "AGENT.md").write_text("---\nname: helper\nskills:\n  - mcp\n---\nBe brief.\n")
    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": {"remote": {"url": "https://x.example/mcp", "headers": {"Authorization": "Bearer ${secret:REMOTE_TOKEN}"}}, "time": {"command": "uvx"}}})
    )
    assert m.remove_from_agent("remote", tmp_path, in_chat=False) == [
        "Removed remote from helper (mcp.json).",
        "It read REMOTE_TOKEN from this profile's secrets; they stay stored (`webagents secrets remove <NAME>` removes one).",
        "The agent stops using it the next time it starts.",
    ]
    assert json.loads((tmp_path / "mcp.json").read_text()) == {"mcpServers": {"time": {"command": "uvx"}}}
    with pytest.raises(m.RemoveRefused) as refused:
        m.remove_from_agent("remote", tmp_path, in_chat=True)
    assert str(refused.value) == "helper has no MCP server named remote."
    assert m.remove_from_agent("time", tmp_path, in_chat=True)[-1] == "/reload stops it in this chat."


def test_the_cli_and_the_chat_remove(home):
    runner.invoke(app, ["mcp", "add", "chrome-devtools"])
    removed = runner.invoke(app, ["mcp", "remove", "chrome-devtools"])
    assert removed.exit_code == 0, removed.output
    assert removed.output.splitlines() == ["Removed chrome-devtools from helper (mcp.json).", "The agent stops using it the next time it starts."]
    again = runner.invoke(app, ["mcp", "remove", "chrome-devtools"])
    assert again.exit_code == 1 and "helper has no MCP server named chrome-devtools." in again.output

    from webagents.cli.repl.session import WebAgentsSession

    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=200, force_terminal=False, color_system=None, record=True)

    async def main() -> str:
        await session.handle_input("/mcp add chrome-devtools")
        await session.handle_input("/mcp remove chrome-devtools")
        await session.handle_input("/mcp remove chrome-devtools")
        return session.console.export_text()

    out = asyncio.run(main())
    assert "Removed chrome-devtools from helper (mcp.json)." in out and "/reload stops it in this chat." in out
    assert "helper has no MCP server named chrome-devtools." in out
