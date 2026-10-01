"""
`webagents conversations list | delete | prune` (`webagents/cli/conversations_command.py`,
2026-09-29), against the shared fixture `tests/fixtures/cli/conversations.json`,
which the TypeScript suite reads too (`typescript/tests/unit/cli/conversations-command.test.ts`).

Until this, a kept conversation could only be removed by hand. Pinned here:
the words; the `--older-than` grammar; what `list` prints, this folder's or
every folder's; which conversation an id prefix picks; which are older than an
age. Then the real commands in a throwaway HOME: `delete` asks, and without a
terminal it needs `--yes`; `prune --dry-run` removes nothing; `--json` answers
one document; a copy on Robutler is said to stay.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli import conversations_command as c
from webagents.cli.main import app
from webagents.cli.sessions import save_session, sessions_dir

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "conversations.json").read_text(encoding="utf-8"))
NOW = datetime.fromisoformat(FIXTURE["now"].replace("Z", "+00:00"))
runner = CliRunner()


def _kept(raw: dict) -> c.Kept:
    return c.Kept(Path("/nowhere"), raw["folder"], raw["agent"], raw["id"], raw["updated_at"], raw["messages"], raw["preview"], raw["on_robutler"])


def test_the_words_are_the_fixtures():
    assert c.WORDS == FIXTURE["words"]


@pytest.mark.parametrize("text,seconds", FIXTURE["ages"])
def test_the_age_grammar(text, seconds):
    assert c.parse_age(text) == seconds


@pytest.mark.parametrize("case", FIXTURE["list"]["cases"], ids=lambda case: case["about"])
def test_what_list_prints(case):
    kept = [_kept(k) for k in case["kept"]]
    lines = c.list_lines(kept, case["folder"], case["everywhere"], FIXTURE["list"]["width"], FIXTURE["list"]["delete_command"], NOW)
    assert lines == case["lines"]


@pytest.mark.parametrize("case", FIXTURE["pick"]["cases"], ids=lambda case: f"{case['prefix']!r} everywhere={case['everywhere']}")
def test_which_conversation_an_id_picks(case):
    p = FIXTURE["pick"]
    kept = [_kept(k) for k in p["kept"]]
    if "refused" in case:
        with pytest.raises(c.NotOne) as refused:
            c.pick(kept, case["prefix"], p["folder"], case["everywhere"], p["list_command"])
        assert str(refused.value) == case["refused"]
        return
    assert kept.index(c.pick(kept, case["prefix"], p["folder"], case["everywhere"], p["list_command"])) == case["picks"]


def test_which_are_older_than_an_age():
    o = FIXTURE["older_than"]
    assert [k.id for k in c.older_than([_kept(k) for k in o["kept"]], o["seconds"], NOW)] == o["ids"]


# ---------------------------------------------------------------- the commands


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    here = tmp_path / "here"
    elsewhere = tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(here)
    for folder, agent, session_id, text, chat_id in (
        (here, "helper", "aaaa1111-0000-4000-8000-000000000001", "plan the launch", None),
        (here, "writer", "bbbb2222-0000-4000-8000-000000000002", "draft the post", "chat-9"),
        (elsewhere, "helper", "cccc3333-0000-4000-8000-000000000003", "somewhere else", None),
    ):
        save_session(sessions_dir(folder, agent), {
            "session_id": session_id,
            "agent_name": agent,
            "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}],
            "metadata": {"folder": str(folder.resolve()), **({"robutler_chat_id": chat_id} if chat_id else {})},
        })
    # A conversation never started (no message from the person) is not one.
    save_session(sessions_dir(here, "helper"), {"session_id": "dddd4444-0000-4000-8000-000000000004", "agent_name": "helper", "messages": [], "metadata": {}})
    return here, elsewhere


def test_list_shows_this_folders_and_all_shows_every_folders(home):
    here, elsewhere = home
    out = runner.invoke(app, ["conversations", "list"]).output
    assert str(here.resolve()) in out and "plan the launch" in out and "draft the post" in out
    assert "somewhere else" not in out and "dddd4444" not in out
    everywhere = runner.invoke(app, ["conversations", "list", "--all"]).output
    assert "somewhere else" in everywhere and str(elsewhere.resolve()) in everywhere
    document = json.loads(runner.invoke(app, ["--json", "conversations", "list"]).stdout)
    assert document["ok"] is True
    assert sorted(d["id"][:4] for d in document["data"]["conversations"]) == ["aaaa", "bbbb"]


def test_delete_needs_a_yes_where_nothing_can_answer_and_says_the_robutler_copy_stays(home):
    here, _ = home
    refused = runner.invoke(app, ["conversations", "delete", "bbbb"])
    assert refused.exit_code == 1
    assert c.fill("needsYes", command="webagents conversations delete") in refused.output
    assert (sessions_dir(here, "writer") / "bbbb2222-0000-4000-8000-000000000002.json").exists()
    done = runner.invoke(app, ["conversations", "delete", "bbbb", "--yes"])
    assert done.exit_code == 0 and "Deleted the conversation last used just now." in done.output
    assert c.WORDS["remoteStays"] in done.output
    assert not (sessions_dir(here, "writer") / "bbbb2222-0000-4000-8000-000000000002.json").exists()
    missing = runner.invoke(app, ["conversations", "delete", "cccc", "--yes"])
    assert missing.exit_code == 1 and "No conversation cccc in" in missing.output
    anywhere = runner.invoke(app, ["conversations", "delete", "cccc", "--all", "--yes"])
    assert anywhere.exit_code == 0


def test_delete_asks_at_a_terminal(home):
    here, _ = home
    assert c.delete_command(here, "aaaa", False, False, False, ask=lambda q: "n") == 0
    assert (sessions_dir(here, "helper") / "aaaa1111-0000-4000-8000-000000000001.json").exists()
    questions = []
    assert c.delete_command(here, "aaaa", False, False, False, ask=lambda q: questions.append(q) or "y") == 0
    assert questions == [c.fill("askDelete", when="just now", count=2)]
    assert not (sessions_dir(here, "helper") / "aaaa1111-0000-4000-8000-000000000001.json").exists()


def test_prune_by_age(home):
    here, elsewhere = home
    bad = runner.invoke(app, ["conversations", "prune", "--older-than", "soon"])
    assert bad.exit_code == 2 and c.WORDS["badAge"] in bad.output
    assert runner.invoke(app, ["conversations", "prune"]).exit_code == 2
    none = runner.invoke(app, ["conversations", "prune", "--older-than", "1d", "--yes"])
    assert none.exit_code == 0 and "No conversations last used more than 1d ago." in none.output
    later = datetime.fromisoformat("2099-01-01T00:00:00+00:00")
    assert c.prune_command(here, "1d", True, False, True, False, now=later) == 0
    assert (sessions_dir(elsewhere, "helper") / "cccc3333-0000-4000-8000-000000000003.json").exists()
    assert c.prune_command(here, "1d", False, True, False, False, now=later) == 0
    assert not (sessions_dir(here, "helper") / "aaaa1111-0000-4000-8000-000000000001.json").exists()
    assert (sessions_dir(elsewhere, "helper") / "cccc3333-0000-4000-8000-000000000003.json").exists()
