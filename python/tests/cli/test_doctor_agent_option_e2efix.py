"""
`webagents doctor -a <name>` checks that agent, as the chat's `-a` picks it
(2026-09-26, the new-developer e2e run: it was "unknown option '-a'"). An
unknown name is the chat's refusal (`agent_files.py`), raised before anything
is built. The TypeScript doctor is pinned the same way
(`tests/unit/cli/doctor-agent-option-e2efix.test.ts`).
"""

from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from webagents.cli.agent_files import AgentNotFound
from webagents.cli.doctor import run_checks
from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) or var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_DEBUG"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "AGENT.md").write_text("---\nname: bot\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n")
    (folder / "AGENT-helper.md").write_text("---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nHelps\n")
    monkeypatch.chdir(folder)
    return folder


def test_it_checks_the_named_agent_and_the_default_file_without_it(project):
    named = {c.name: c for c in run_checks(project, agent="helper")}["agent"]
    assert named.detail == "helper (AGENT-helper.md)"
    bare = {c.name: c for c in run_checks(project)}["agent"]
    assert bare.detail == "bot (AGENT.md)"


def test_an_unknown_name_is_the_chats_refusal(project):
    with pytest.raises(AgentNotFound, match="There is no agent called nosuch in this folder."):
        run_checks(project, agent="nosuch")


def test_the_command_takes_dash_a_and_answers_the_envelope_under_json(project):
    result = runner.invoke(app, ["doctor", "-a", "helper"])
    assert "helper (AGENT-helper.md)" in result.output
    refused = runner.invoke(app, ["doctor", "-a", "nosuch"])
    assert refused.exit_code == 1
    assert "There is no agent called nosuch in this folder." in refused.output
    as_json = runner.invoke(app, ["--json", "doctor", "-a", "nosuch"])
    assert as_json.exit_code == 1
    document = json.loads(as_json.output)
    assert document["ok"] is False and document["error"]["code"] == "agent_not_found"
