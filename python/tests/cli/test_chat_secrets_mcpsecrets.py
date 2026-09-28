"""
The chat's `/secrets` (S-292, 2026-09-26), pinned by `chat` in
`tests/fixtures/cli/secrets.json` (the TypeScript suite runs the same,
`chat-secrets-mcpsecrets.test.ts`): it lists the names in the store an MCP
server's `${secret:NAME}` reads and how to add one; `/secrets set NAME` takes
the value at a hidden local prompt, like `/keys set`, and the value goes to
the store and nowhere else: not to the model, not to the environment, not to
the screen. Scratch HOME, file backend, never the keychain.
"""

from __future__ import annotations

import asyncio
import json
import os
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.session import WebAgentsSession

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "secrets.json").read_text())["chat"]
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY")
VALUE = "dummy-github-token-not-a-real-credential"


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in KEY_VARS + ("WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    (project / "AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
    yield project


def _chat() -> WebAgentsSession:
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=200, force_terminal=False, color_system=None, record=True)
    asyncio.run(session.initialize())
    return session


def _say(session: WebAgentsSession, line: str) -> str:
    before = len(session.console.export_text(clear=False))
    asyncio.run(session.handle_input(line))
    return session.console.export_text(clear=False)[before:]


def test_the_listing_says_none_and_how_to_add_one():
    out = _say(_chat(), "/secrets")
    assert FIXTURE["heading"] in out
    assert FIXTURE["none"] in out
    assert FIXTURE["hint"] in out


def test_set_takes_the_value_at_a_hidden_prompt_and_keeps_it_out_of_everything_else(monkeypatch):
    chat = _chat()
    prompts = []

    def hidden(prompt=""):
        prompts.append(prompt)
        return VALUE

    monkeypatch.setattr("getpass.getpass", hidden)
    out = _say(chat, "/secrets set GITHUB_TOKEN")
    assert prompts == ["  " + FIXTURE["prompt"].replace("{name}", "GITHUB_TOKEN")]
    assert FIXTURE["stored"].replace("{name}", "GITHUB_TOKEN").replace("{where}", FIXTURE["stored_where"]["file"]) in out
    assert FIXTURE["stored_detail"].replace("{name}", "GITHUB_TOKEN") in out
    assert VALUE not in out
    # Not in the conversation, and not in the environment the shell tool inherits.
    assert chat.messages == []
    assert "GITHUB_TOKEN" not in os.environ
    from webagents.cli.commands.secrets import _store

    assert _store(quiet=True).get("GITHUB_TOKEN") == VALUE

    listing = _say(chat, "/secrets")
    assert "GITHUB_TOKEN" in listing and FIXTURE["where"]["file"] in listing and VALUE not in listing

    assert FIXTURE["removed"].replace("{name}", "GITHUB_TOKEN") in _say(chat, "/secrets remove GITHUB_TOKEN")
    assert FIXTURE["not_stored"].replace("{name}", "GITHUB_TOKEN") in _say(chat, "/secrets remove GITHUB_TOKEN")
    assert _store(quiet=True).get("GITHUB_TOKEN") is None


def test_nothing_entered_stores_nothing(monkeypatch):
    chat = _chat()
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "   ")
    assert FIXTURE["nothing_entered"] in _say(chat, "/secrets set GITHUB_TOKEN")
    from webagents.cli.commands.secrets import _store

    assert _store(quiet=True).get("GITHUB_TOKEN") is None


def test_a_bad_name_and_a_bad_verb_are_refused():
    chat = _chat()
    assert FIXTURE["bad_name"].replace("{name}", "bad-name") in _say(chat, "/secrets set bad-name")
    assert FIXTURE["usage_error"] in _say(chat, "/secrets bogus")
    assert FIXTURE["usage_error"] in _say(chat, "/secrets set")


def test_a_provider_key_stored_this_way_reaches_the_model_as_keys_set_would(monkeypatch):
    (Path.cwd() / "AGENT.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n")
    chat = _chat()
    assert chat.model_problem
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-dummy-provider-key")
    _say(chat, "/secrets set OPENAI_API_KEY")
    assert chat.model_problem is None
