"""
A stdio MCP server that stops before it answers (2026-09-29, the owner's yoyo
agent), against the shared fixture `tests/fixtures/mcp_tool/connect_errors.json`
(`server_stopped`), which the TypeScript suite reads too
(`tests/unit/skills/mcp-server-stopped.test.ts`).

`uvx mcp-server-sqlite` fetched the server's last release with `mcp` 2.2.0; the
server died at start (`@server.list_resources()` is gone from `mcp` 2), and the
chat's `/mcp` said `not connected: Connection closed`. The model's
`list_mcp_servers` said "No MCP servers connected.", so the model could not say
why either. The reason sat in the server's stderr log, which nothing named.
Pinned here:
  * the words and the error-line rule are the fixture's;
  * a real server that writes a traceback and exits gets a report row saying
    its own error line and where its output is, the log holding the whole
    traceback;
  * `list_mcp_servers` names the server that did not connect, with that reason.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp.connect_errors import (
    LINE_MAX,
    SERVER_STOPPED,
    SERVER_STOPPED_SILENT,
    SERVER_STOPPED_UNLOGGED,
    connection_closed,
    display_path,
    last_error_line,
    server_stopped_sentence,
)
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, owner_reference_sources

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
STOPPED = json.loads((FIXTURES / "mcp_tool" / "connect_errors.json").read_text())["server_stopped"]


def test_the_words_are_the_fixtures():
    assert SERVER_STOPPED == STOPPED["sentence"]
    assert SERVER_STOPPED_SILENT == STOPPED["silent"]
    assert SERVER_STOPPED_UNLOGGED == STOPPED["unlogged"]
    assert LINE_MAX == STOPPED["line_max"]


@pytest.mark.parametrize("case", STOPPED["lines"], ids=lambda case: case["about"])
def test_the_error_line_is_the_one_that_says_what_went_wrong(case):
    assert last_error_line(case["stderr"]) == case["line"]


def test_a_long_error_line_is_cut():
    line = last_error_line("Error: " + "x" * 400)
    assert line is not None and len(line) == LINE_MAX and line.endswith("…")


@pytest.mark.parametrize("case", STOPPED["paths"], ids=lambda case: case["log"])
def test_the_log_path_writes_home_as_a_tilde(case, monkeypatch):
    monkeypatch.setenv("HOME", case["home"])
    assert display_path(case["log"]) == case["shown"]


def test_the_sentence(monkeypatch):
    says = STOPPED["says"]
    monkeypatch.setenv("HOME", says["home"])
    assert server_stopped_sentence(says["stderr"], says["log"]) == says["sentence"]
    assert server_stopped_sentence("", says["log"]) == STOPPED["silent"]
    assert server_stopped_sentence(None, None) == STOPPED["unlogged"]


def test_what_counts_as_the_connection_closing():
    class ErrorData:
        code = -32000
        message = "Connection closed"

    class McpError(Exception):
        error = ErrorData()

    class EndOfStream(Exception):
        pass

    assert connection_closed(McpError("Connection closed"))
    assert connection_closed(EndOfStream())
    assert connection_closed(RuntimeError("Connection closed"))
    assert not connection_closed(RuntimeError("Request timed out"))
    assert not connection_closed(FileNotFoundError("[Errno 2] No such file or directory: 'uvx'"))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "home"


def test_a_server_that_crashes_at_start_is_said_with_its_own_error(tmp_path, home):
    script = tmp_path / "crashing_server.py"
    script.write_text(
        "import sys\n"
        f"sys.stderr.write({STOPPED['says']['stderr']!r})\n"
        "sys.exit(1)\n"
    )
    skill = LocalMcpSkill({
        "mcp": {"sqlite": {"command": sys.executable, "args": [str(script)]}},
        "references": owner_reference_sources(),
    })
    agent = BaseAgent(name="client", instructions="x", skills={"mcp": skill})

    async def after_initialize():
        await agent._ensure_skills_initialized()
        try:
            return skill.server_report(), await skill.list_mcp_servers()
        finally:
            await skill.cleanup()

    report, listed = asyncio.run(after_initialize())
    from webagents.cli.config_store import global_dir

    log = global_dir() / "logs" / "mcp-sqlite.log"
    expected = SERVER_STOPPED.replace("{line}", STOPPED["lines"][0]["line"]).replace("{log}", display_path(str(log)))
    row = report[0]
    assert row["connected"] is False
    assert row["error"] == expected
    assert "Connection closed" not in json.dumps(report)
    # The whole traceback is in the log the sentence names.
    assert STOPPED["says"]["stderr"] in log.read_text()
    # The model's tool names the server and why it is not there.
    assert listed.splitlines()[0] == "No MCP servers connected."
    assert "Not connected:" in listed
    assert f"  sqlite: {expected}" in listed.splitlines()
