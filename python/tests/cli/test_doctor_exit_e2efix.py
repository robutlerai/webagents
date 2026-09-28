"""
`webagents doctor` prints its report and nothing above it (2026-09-26, the
new-developer e2e run): building the agent for the checks printed the skills'
log lines (an MCP server that did not connect) above "Checks", and a server
that had connected was never closed, so asyncio printed "an error occurred
during closing of asynchronous generator" when the loop ended. The report's
`mcp` check carries the same finding; the log goes to the profile's file as
`-p` sends it, and the servers are closed in the loop that opened them. The
TypeScript doctor is pinned the same way (`tests/unit/cli/doctor-exit-e2efix.test.ts`).

The CLI is spawned as a person would run it, against the probe MCP server the
secrets tests use, with a scratch HOME and the file backend, never the keychain.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROBE = json.loads((FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.json").read_text())
PROBE_SERVER = FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.py"


def _clean_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not (k.endswith("_API_KEY") or k.endswith("_TOKEN") or k in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_DEBUG"))}
    env.update({
        "HOME": str(home),
        "WEBAGENTS_SECRETS_BACKEND": "file",
        "ROBUTLER_API_URL": "http://127.0.0.1:9",
        "OPENAI_API_KEY": "sk-dummy-not-a-real-key",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _project(tmp_path: Path, entry: dict) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text(
        "---\nname: a\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - mcp:\n      " + PROBE["server_name"] + ": " + json.dumps(entry) + "\n---\nBody\n"
    )
    return project


def _doctor(project: Path, home: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "webagents", "doctor"], cwd=project, env=_clean_env(home), capture_output=True, text=True, timeout=120)


def test_a_connected_server_is_closed_and_nothing_sits_above_the_report(tmp_path):
    project = _project(tmp_path, {"command": sys.executable, "args": [str(PROBE_SERVER)], "env": {"PROBE_REGION": "eu-west-9"}})
    result = _doctor(project, tmp_path / "home")
    assert result.stdout.splitlines()[0] == "Checks", result.stdout
    assert f"1 server connected: {PROBE['server_name']}" in result.stdout
    assert "asynchronous generator" not in result.stderr
    assert "[MCP]" not in result.stderr and "[MCPSkill]" not in result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_server_that_cannot_start_is_in_the_mcp_check_not_a_raw_line_above_the_report(tmp_path):
    project = _project(tmp_path, {"command": "/nonexistent/mcp-server"})
    result = _doctor(project, tmp_path / "home")
    assert result.returncode == 1
    assert result.stdout.splitlines()[0] == "Checks", result.stdout
    assert any(line.lstrip().startswith("✗ mcp") and PROBE["server_name"] + ":" in line for line in result.stdout.splitlines())
    assert "[MCP]" not in result.stdout and "[MCP]" not in result.stderr
    assert "ERROR" not in result.stderr
