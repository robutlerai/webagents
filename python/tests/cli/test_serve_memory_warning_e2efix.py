"""
`webagents serve` says at startup when the agent has `memory` but nothing
verifies a caller (2026-09-26, the new-developer e2e run): every served caller
is a caller nothing verified, who reads shared notes and writes nothing, and
the tester found that out one refused tool call at a time. The sentence is
`memory_without_auth` in `tests/fixtures/cli/serve_startup.json`, which the
TypeScript `serve()` prints too (`tests/unit/server/serve-memory-warning-e2efix.test.ts`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app
from webagents.cli.startup_lines import MEMORY_WITHOUT_AUTH, memory_without_auth_line

STARTUP = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "serve_startup.json").read_text())
runner = CliRunner()


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) or var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_PUBLIC_URL", "ROBUTLER_API_URL", "ROBUTLER_INTERNAL_API_URL"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.chdir(tmp_path)
    import uvicorn

    from webagents.cli import listen

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(listen, "port_is_free", lambda host, port: True)
    return tmp_path


def test_the_sentence_is_the_fixtures():
    assert MEMORY_WITHOUT_AUTH == STARTUP["memory_without_auth"]
    assert memory_without_auth_line("rememberer") == STARTUP["memory_without_auth"].format(name="rememberer")


def test_it_is_said_after_the_authskill_line_for_an_agent_with_memory(tmp_path):
    (tmp_path / "AGENT.md").write_text(f"---\nname: rememberer\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - {STARTUP['memory_skill']}\n---\nBody\n")
    result = runner.invoke(app, ["serve", "--port", "3999"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    auth = next(i for i, line in enumerate(lines) if "rememberer has no AuthSkill" in line)
    assert lines[auth + 1] == memory_without_auth_line("rememberer")


def test_it_is_not_said_for_an_agent_without_memory(tmp_path):
    (tmp_path / "AGENT.md").write_text("---\nname: plain\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n")
    result = runner.invoke(app, ["serve", "--port", "3999"])
    assert result.exit_code == 0, result.output
    assert "has memory but no AuthSkill" not in result.output
