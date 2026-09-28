"""
`webagents -p` says MCP problems on stderr, as the TypeScript `-p` does
(2026-09-26, the new-developer e2e run: it was silent, the agent's log going
to a file). The three sentences are `problem_lines` in
`tests/fixtures/mcp_tool/config_shapes.json`, which the TypeScript suite runs
too (`tests/unit/skills/mcp-problem-lines-e2efix.test.ts`): here the
templates, the lines a report yields, and the CLI itself, spawned in a scratch
HOME against a server that cannot start, printing the `failed` line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from webagents.agents.skills.local.mcp.config import MCP_PROBLEM_LINES, problem_lines

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LINES = json.loads((FIXTURES / "mcp_tool" / "config_shapes.json").read_text())["secret_refs"]["problem_lines"]


def test_the_templates_are_the_fixtures():
    assert MCP_PROBLEM_LINES == {"failed": LINES["failed"], "rejected": LINES["rejected"], "warning": LINES["warning"]}


@pytest.mark.parametrize("case", LINES["cases"], ids=[c["name"] for c in LINES["cases"]])
def test_the_lines_a_report_yields(case):
    assert problem_lines(case["report"]) == case["lines"]


def _clean_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not (k.endswith("_API_KEY") or k.endswith("_TOKEN") or k in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "WEBAGENTS_DEBUG"))}
    env.update({
        "HOME": str(home),
        "WEBAGENTS_SECRETS_BACKEND": "file",
        "ROBUTLER_API_URL": "http://127.0.0.1:9",
        # A key so the run starts, and a model endpoint that refuses, so it ends: the MCP lines come first.
        "OPENAI_API_KEY": "sk-dummy-not-a-real-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def test_the_cli_says_a_server_that_cannot_start_on_stderr(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text(
        "---\nname: mcpfail\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - mcp:\n      ghost:\n        command: /nonexistent/mcp-server\n---\nBody\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "webagents", "-p", "hello"],
        cwd=project,
        env=_clean_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    failed = [line for line in result.stderr.splitlines() if line.startswith('[MCPSkill] Server "ghost" failed to connect: ')]
    assert failed, result.stderr
    # The sentence only, masked and on stderr; stdout carries no MCP line.
    assert "[MCPSkill]" not in result.stdout
