"""
`webagents doctor`'s `mcp` check (S-292, 2026-09-26): the check computed from
the MCP skill's report, pinned by `doctor` in `tests/fixtures/cli/secrets.json`
(the TypeScript suite runs the same cases, `doctor-mcp-mcpsecrets.test.ts`),
and the check as doctor reaches it through a real agent file: a server whose
`${secret:NAME}` is not stored fails with the `webagents secrets set` command,
and a stored one connects. Scratch HOME, file backend, never the keychain.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from webagents.cli.doctor import MCP_CHECK_WORDS, mcp_check, run_checks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SECRETS = json.loads((FIXTURES / "cli" / "secrets.json").read_text())
PROBE = json.loads((FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.json").read_text())
PROBE_SERVER = FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.py"


def test_the_words_match_the_fixture():
    assert MCP_CHECK_WORDS == SECRETS["doctor"]["words"]


@pytest.mark.parametrize("case", SECRETS["doctor"]["cases"], ids=[c["name"] for c in SECRETS["doctor"]["cases"]])
def test_the_check_from_a_report(monkeypatch, case):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    assert mcp_check(case["report"]).as_dict() == case["check"]


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def _agent_with_probe(project: Path, env: dict) -> None:
    entry = {"command": sys.executable, "args": [str(PROBE_SERVER)], "env": env}
    (project / "AGENT.md").write_text(
        "---\nname: a\nskills:\n  - mcp:\n      " + PROBE["server_name"] + ": " + json.dumps(entry) + "\n---\nBody\n"
    )


def _mcp(folder: Path):
    return {c.name: c for c in run_checks(folder)}["mcp"]


def test_an_unstored_secret_fails_the_check_with_the_command_that_stores_it(scratch):
    _agent_with_probe(scratch, {"X": "${secret:NOT_STORED}"})
    check = _mcp(scratch)
    assert check.status == "fail"
    assert check.detail == f'{PROBE["server_name"]}: env X of MCP server "{PROBE["server_name"]}": ${{secret:NOT_STORED}} is not set: store it with `webagents secrets set NOT_STORED`'
    assert check.fix == "`webagents secrets set NOT_STORED`"


def test_a_stored_secret_connects_and_the_check_names_the_server(scratch):
    from webagents.cli.commands.secrets import _store

    _store(quiet=True).set("STORED_FOR_DOCTOR", "dummy-not-a-real-credential")
    _agent_with_probe(scratch, {"X": "${secret:STORED_FOR_DOCTOR}"})
    check = _mcp(scratch)
    assert check.status == "ok"
    assert check.detail == f"1 server connected: {PROBE['server_name']}"


def test_a_literal_that_looks_like_a_key_is_a_warning(scratch):
    _agent_with_probe(scratch, {"GH": "ghp_dummy0123456789abcdefghijklmnop"})
    check = _mcp(scratch)
    assert check.status == "warn"
    assert "ghp_dummy" not in check.detail and "ghp_dummy" not in (check.fix or "")
    suggested = f"{PROBE['server_name'].upper()}_GH"
    assert check.fix == f"${{secret:{suggested}}} in the agent file, then `webagents secrets set {suggested}`"


def test_no_mcp_entry_is_not_used(scratch):
    (scratch / "AGENT.md").write_text("---\nname: a\n---\nBody\n")
    check = _mcp(scratch)
    assert check.status == "ok" and check.detail == MCP_CHECK_WORDS["notUsed"]
    assert os.environ.get("OPENAI_API_KEY") is None
