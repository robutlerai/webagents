"""
`--json` is honoured by `secrets list` and by the `-a <unknown agent>` refusal
(2026-09-26, the new-developer e2e run: both ignored it). Pinned by
`list_json` in `tests/fixtures/cli/secrets.json` and by
`tests/fixtures/cli/json_errors.json`, which the TypeScript suite runs too
(`tests/unit/cli/json-secrets-agent-e2efix.test.ts`). A scratch HOME with the
file backend; the value is piped in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cli"
LIST = json.loads((FIXTURES / "secrets.json").read_text())["list_json"]
ERRORS = json.loads((FIXTURES / "json_errors.json").read_text())
runner = CliRunner()


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) or var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_DEBUG"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_json_secrets_list_answers_one_document_with_the_fixtures_rows(monkeypatch):
    stored = runner.invoke(app, ["secrets", "set", LIST["case"]["stored"]], input="dummy-value-not-a-credential\n")
    assert stored.exit_code == 0, stored.output
    monkeypatch.setenv(LIST["case"]["in_shell"], "sk-from-the-shell")
    listed = runner.invoke(app, ["--json", "secrets", "list"])
    assert listed.exit_code == 0, listed.output
    document = json.loads(listed.output)
    assert document["ok"] is True
    assert document["data"] == LIST["case"]["data"]
    for key in document["data"]["keys"]:
        assert list(key) == LIST["fields"]
    assert "dummy-value" not in listed.output


def test_json_unknown_agent_answers_the_error_envelope(tmp_path, monkeypatch):
    (tmp_path / "AGENT.md").write_text("---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    expected = ERRORS["agent_not_found"]
    for argv in (["--json", "-a", "nosuch", "-p", "hi"], ["--json", "chat", "-a", "nosuch", "-p", "hi"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == expected["exit"], result.output
        document = json.loads(result.output)
        assert document["ok"] is False
        assert document["error"]["code"] == expected["code"]
        assert document["error"]["message"] == expected["message"].format(name="nosuch", agents="my-agent")
