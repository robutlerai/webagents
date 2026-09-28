"""
The chat's commands that read or change the agent file (2026-09-26,
interactive-mode spec section 3), against the shared fixture
`tests/fixtures/cli/chat_edits.json`, which the TypeScript suite reads too
(`chat-edits-interactive.test.ts`): the words both chats say, the loader's
refusals, and each command driven through `handle_input` with the prompts
mocked, as `test_chat.py` does.
"""

import asyncio
import json
import os
import re
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.loader import AgentFormatError
from webagents.cli.loader.agent_md import AgentFile
from webagents.cli.repl.chat_words import CHAT_WORDS
from webagents.cli.repl.session import WebAgentsSession

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "chat_edits.json").read_text())
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY")


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in KEY_VARS + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL", "VISUAL", "EDITOR", "WEBAGENTS_SRT_CLI", "WEBAGENTS_SRT_NODE"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return {"home": tmp_path / "home", "project": project, "monkeypatch": monkeypatch}


def _fill(text, subs):
    for key, value in subs.items():
        text = text.replace("{" + key + "}", value)
    return text


def _expect_in_order(lines, expected, subs):
    hay = re.sub(r"\s+", " ", " ".join(lines))
    at = 0
    for raw in expected:
        wanted = re.sub(r"\s+", " ", _fill(raw, subs)).strip()
        found = hay.find(wanted, at)
        assert found >= 0, f'"{wanted}" not found in order in:\n' + "\n".join(lines)
        at = found + len(wanted)


def _write_files(folder, files):
    for name, text in (files or {}).items():
        full = folder / name
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text)


def _editor_script(home, kind):
    path = home / f"editor-{kind}.sh"
    body = "#!/bin/sh\nprintf 'More.\\n' >> \"$1\"\n" if kind == "append" else "#!/bin/sh\nexit 3\n"
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def test_the_words_match_the_shared_fixture():
    assert CHAT_WORDS == FIXTURE["words"]


@pytest.mark.parametrize("case", FIXTURE["loader"]["cases"], ids=[c["case"] for c in FIXTURE["loader"]["cases"]])
def test_the_loader(newcomer, case):
    project = newcomer["project"]
    file = project / case["file"]
    if case.get("link"):
        (project / case["link"]).write_text(case["text"])
        os.symlink(case["link"], str(file))
    else:
        file.write_text(case["text"])
    loader = FIXTURE["loader"]
    if case.get("error"):
        subs = {"path": str(file), "key": case.get("key", ""), "near": case.get("near", ""), "known": loader["known_keys"]}
        template = {
            "linked": loader["linked"],
            "not_yaml": loader["not_yaml"],
            "not_yaml_at": loader["not_yaml_at"],
            "cron_string": loader["cron_string"],
            "unknown_key": loader["unknown_key"],
            "unknown_key_no_match": loader.get("unknown_key_no_match", ""),
        }[case["error"]]
        with pytest.raises(AgentFormatError) as raised:
            AgentFile(file)
        if "{line}" in template:
            # The position is each SDK's parser's (`not_yaml_at_about`): the shape is pinned, the numbers are not.
            import re

            expected = re.escape(_fill(template, subs)).replace(re.escape("{line}"), r"\d+").replace(re.escape("{column}"), r"\d+")
            assert re.fullmatch(expected, str(raised.value)), str(raised.value)
            return
        assert str(raised.value) == _fill(template, subs)
        return
    agent = AgentFile(file)
    if case.get("name"):
        assert agent.metadata.name == case["name"]
    if case.get("skills"):
        assert [s if isinstance(s, str) else next(iter(s)) for s in agent.metadata.skills] == case["skills"]
    if case.get("instructions"):
        assert agent.instructions == case["instructions"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["case"] for c in FIXTURE["cases"]])
def test_the_chat_cases(newcomer, monkeypatch, case):
    home, project = newcomer["home"], newcomer["project"]
    folder = home if case.get("cwd") == "home" else project
    if case.get("cwd") == "home":
        monkeypatch.chdir(home)
    for key, value in (case.get("env") or {}).items():
        monkeypatch.setenv(key, value)
    _write_files(folder, case.get("files"))
    if case.get("editor") in ("append", "fail"):
        monkeypatch.setenv("EDITOR", _editor_script(home, case["editor"]))

    session = WebAgentsSession(
        agent_path=None,
        model=case.get("model"),
        chosen=(case.get("agent") == "robutler"),
        interactive=bool(case.get("interactive")),
    )
    session.console = Console(file=StringIO(), width=240, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())

    _write_files(folder, case.get("then_write"))
    for name, target in (case.get("then_link") or {}).items():
        full = folder / name
        if full.exists() or full.is_symlink():
            full.unlink()
        os.symlink(target, str(full))

    answers = list(case.get("answers") or [])

    async def ask(_self, _question):
        return answers.pop(0) if answers else None

    monkeypatch.setattr(WebAgentsSession, "_ask", ask)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    before = len(session.console.export_text(clear=False))
    for line in case.get("typed") or []:
        asyncio.run(session.handle_input(line))
    printed = [l.rstrip() for l in session.console.export_text(clear=False)[before:].split("\n")]

    subs = {"folder": str(Path(os.getcwd())), "known": FIXTURE["loader"]["known_keys"]}
    _expect_in_order(printed, case.get("printed") or [], subs)

    for name, expected in (case.get("files_after") or {}).items():
        full = folder / name
        if expected is None:
            assert not full.exists(), f"{name} should not exist"
        else:
            assert full.read_text() == expected, name
    if case.get("lock_after") is not None:
        lock = json.loads((folder / ".webagents" / "skills.lock").read_text())
        assert sorted((lock.get("skills") or {}).keys()) == case["lock_after"]


@pytest.mark.parametrize("case", FIXTURE["offers"], ids=[c["case"] for c in FIXTURE["offers"]])
def test_the_first_run_offer(newcomer, monkeypatch, case):
    """A None answer is Ctrl+C or Ctrl+D at "Choose 1-3": the offer is cancelled
    and said so, and the chat goes on (2026-09-26, the e2e run, where this chat
    echoed ^C and hung until enter)."""
    project = newcomer["project"]
    _write_files(project, case.get("files"))
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=240, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    assert session.model_problem

    answers = list(case.get("answers") or [])
    asked: list = []

    async def ask(_self, question):
        asked.append(question)
        return answers.pop(0) if answers else None

    monkeypatch.setattr(WebAgentsSession, "_ask", ask)
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.offer_model_access())
    printed = asked + [l.rstrip() for l in session.console.export_text(clear=False)[before:].split("\n")]
    _expect_in_order(printed, case.get("printed") or [], {})
