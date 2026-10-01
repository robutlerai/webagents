"""
A stdio MCP server whose command is not there (2026-09-29), against the shared
fixture `tests/fixtures/mcp_tool/connect_errors.json` (`command_missing`), which
the TypeScript suite reads too (`tests/unit/skills/mcp-command-missing.test.ts`).

The row said `[Errno 2] No such file or directory: 'uvx'`, the operating
system's words, with nothing about what to install. Pinned here: the sentence
and the hints are the fixture's; a real skill pointed at a command that is not
there reports that sentence with `needs_command`, and `doctor`'s fix line says
what to install; a missing `uvx` is first looked for beside this Python and in
the `uv` package (`pip install 'webagents[uv]'`).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp import skill as mcp_skill
from webagents.agents.skills.local.mcp.connect_errors import COMMAND_HINTS, COMMAND_MISSING, command_hint, command_missing_sentence
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, owner_reference_sources
from webagents.cli.doctor import MCP_CHECK_WORDS, mcp_check

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
MISSING = json.loads((FIXTURES / "mcp_tool" / "connect_errors.json").read_text())["command_missing"]
DOCTOR_WORDS = json.loads((FIXTURES / "cli" / "secrets.json").read_text())["doctor"]["words"]


def test_the_words_are_the_fixtures():
    assert COMMAND_MISSING == MISSING["sentence"]
    assert COMMAND_HINTS == MISSING["hints"]
    assert MCP_CHECK_WORDS["fixCommand"] == DOCTOR_WORDS["fixCommand"]


@pytest.mark.parametrize("case", MISSING["cases"], ids=lambda case: case["command"])
def test_the_sentence_says_what_to_install(case):
    assert command_missing_sentence(case["command"]) == case["says"]


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_a_command_is_looked_for_on_path_and_as_a_file(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "my-server")
    assert mcp_skill._command_found("my-server", str(bin_dir), None)
    assert not mcp_skill._command_found("not-there", str(bin_dir), None)
    assert mcp_skill._command_found("bin/my-server", None, str(tmp_path))
    assert not mcp_skill._command_found("bin/not-there", None, str(tmp_path))


def test_a_missing_uvx_is_found_beside_this_python(tmp_path, monkeypatch):
    here = tmp_path / "venv" / "bin"
    here.mkdir(parents=True)
    uvx = _executable(here / "uvx")
    monkeypatch.setattr(mcp_skill.sys, "executable", str(here / "python"))
    assert mcp_skill._uv_fallback("uvx", ["mcp-server-time"]) == (str(uvx), ["mcp-server-time"])


def test_else_in_the_uv_package_run_as_uv_tool_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_skill.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setitem(sys.modules, "uv", types.SimpleNamespace(find_uv_bin=lambda: "/opt/uv/bin/uv"))
    assert mcp_skill._uv_fallback("uvx", ["mcp-server-time"]) == ("/opt/uv/bin/uv", ["tool", "run", "mcp-server-time"])
    assert mcp_skill._uv_fallback("uv", ["run", "x"]) == ("/opt/uv/bin/uv", ["run", "x"])
    assert mcp_skill._uv_fallback("npx", ["-y", "x"]) is None


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "home"


def test_a_server_whose_command_is_not_there_says_so_and_doctor_says_what_to_install(home):
    command = "webagents-test-no-such-command"
    skill = LocalMcpSkill({"mcp": {"tools": {"command": command, "args": []}}, "references": owner_reference_sources()})
    agent = BaseAgent(name="client", instructions="x", skills={"mcp": skill})

    async def report_after_initialize():
        await agent._ensure_skills_initialized()
        try:
            return skill.server_report()
        finally:
            await skill.cleanup()

    row = asyncio.run(report_after_initialize())[0]
    assert row["connected"] is False
    assert row["error"] == command_missing_sentence(command)
    assert row["needs_command"] == command
    assert "Errno" not in row["error"]
    check = mcp_check([row])
    assert check.status == "fail"
    assert check.fix == MCP_CHECK_WORDS["fixCommand"].replace("{command}", command).replace("{hint}", command_hint(command))
