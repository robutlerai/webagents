"""
The `mcp` entry of an agent file (plan item 0.3, 2026-09-26): the shapes it
accepts and the servers each resolves to, the one name a server's tool gets,
and the error when the SDK cannot load, all pinned by
`tests/fixtures/mcp_tool/config_shapes.json`, which the TypeScript suite runs
too (`tests/unit/skills/mcp-agent-file.test.ts`). Then the client itself,
against the fixture echo server (`tests/fixtures/mcp_tool/echo_server.py`,
`echo_server.json`) over stdio and over Streamable HTTP, and the access block
reaching the tools it registers on start.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp import skill as skill_module
from webagents.agents.skills.local.mcp.config import TOOL_SEPARATOR, qualified_tool_name, sdk_missing, servers_from_config
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill
from webagents.cli.agent_builder import load_skills

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_tool"
SHAPES = json.loads((FIXTURES / "config_shapes.json").read_text())
ECHO = json.loads((FIXTURES / "echo_server.json").read_text())
ECHO_SERVER = FIXTURES / "echo_server.py"
STDIO_SERVER = {"command": sys.executable, "args": [str(ECHO_SERVER)]}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_port(port: int, process: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"the server exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"nothing listened on {port} within {timeout}s")


# -- the shapes -----------------------------------------------------------------------------


@pytest.mark.parametrize("case", SHAPES["shapes"], ids=[c["name"] for c in SHAPES["shapes"]])
def test_the_shapes_an_mcp_entry_accepts(case):
    resolution = servers_from_config(case["config"])
    assert resolution.servers == case["servers"]
    assert resolution.rejected == case["rejected"]


def test_the_file_s_extra_keys_stay_on_the_server_config():
    resolution = servers_from_config({"s": {"command": "x", "pricing": {"creditsPerCall": 1}}})
    assert resolution.configs["s"]["pricing"] == {"creditsPerCall": 1}
    assert resolution.configs["s"]["transport"] == "stdio"


def test_one_name_per_tool():
    assert TOOL_SEPARATOR == SHAPES["naming"]["separator"]
    for case in SHAPES["naming"]["cases"]:
        assert qualified_tool_name(case["server"], case["tool"]) == case["name"]


def test_the_sdk_error_says_what_failed(monkeypatch):
    assert sdk_missing("<reason>") == SHAPES["missing_sdk"]["python"]
    monkeypatch.setattr(skill_module, "MCP_AVAILABLE", False)
    monkeypatch.setattr(skill_module, "MCP_IMPORT_ERROR", "No module named 'mcp'")
    with pytest.raises(RuntimeError) as raised:
        LocalMcpSkill({"mcp": {"demo": STDIO_SERVER}})
    assert str(raised.value).startswith(SHAPES["missing_sdk"]["starts_with"])
    assert "No module named 'mcp'" in str(raised.value)


def test_streamable_http_is_a_transport_here():
    # Under either of mcp's names (`streamable_http_client` from 1.24, the
    # deprecated `streamablehttp_client` before it; 2026-09-29).
    assert skill_module.streamable_http_available()


# -- the agent-file loader ------------------------------------------------------------------


@pytest.mark.parametrize("shape", [{"demo": {"command": "x"}}, {"mcpServers": {"demo": {"command": "x"}}}], ids=["top level", "mcpServers"])
def test_load_skills_builds_the_mcp_skill_from_either_shape(shape, tmp_path):
    report: dict = {}
    skills = load_skills([{"mcp": shape}], agent_name="a", agent_path=tmp_path / "AGENT.md", report=report)
    assert report == {}
    assert isinstance(skills["mcp"], LocalMcpSkill)
    # The `mcpServers` shape used to resolve to no server at all (2026-09-26).
    assert [s["name"] for s in skills["mcp"]._load_mcp_config().servers] == ["demo"]


def test_a_bare_mcp_reads_mcp_json_next_to_the_agent_file(tmp_path):
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"demo": {"command": "x"}}}))
    skills = load_skills(["mcp"], agent_name="a", agent_path=tmp_path / "AGENT.md")
    assert [s["name"] for s in skills["mcp"]._load_mcp_config().servers] == ["demo"]
    skills = load_skills(["mcp"], agent_name="a", agent_path=tmp_path / "other" / "AGENT.md")
    assert skills["mcp"]._load_mcp_config().servers == []


# -- the client, against the fixture echo server ------------------------------------------------


async def _connected(servers: dict, extra_skills: dict | None = None) -> BaseAgent:
    skills = {"mcp": LocalMcpSkill({"mcp": servers}), **(extra_skills or {})}
    agent = BaseAgent(name="client", instructions="x", skills=skills)
    await agent._ensure_skills_initialized()
    return agent


def _echo_tools(agent: BaseAgent) -> list:
    prefix = ECHO["server_name"] + TOOL_SEPARATOR
    return sorted((t for t in agent.get_all_tools() if t["name"].startswith(prefix)), key=lambda t: t["name"])


async def test_lists_and_calls_the_tools_by_qualified_name_over_stdio():
    agent = await _connected({ECHO["server_name"]: STDIO_SERVER})
    try:
        tools = _echo_tools(agent)
        assert [t["name"] for t in tools] == ECHO["qualified"]
        for tool in ECHO["tools"]:
            listed = next(t for t in tools if t["name"] == qualified_tool_name(ECHO["server_name"], tool["name"]))
            assert listed["definition"]["function"]["description"] == tool["description"]
            assert listed["definition"]["function"]["parameters"] == tool["inputSchema"]
        for call in ECHO["calls"]:
            assert await agent.execute_tool(call["tool"], call["arguments"]) == call["text"]
    finally:
        await agent.skills["mcp"].cleanup()


async def test_over_streamable_http():
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(ECHO_SERVER), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        wait_for_port(port, process)
        url = f"http://127.0.0.1:{port}{ECHO['http_path']}"
        agent = await _connected({ECHO["server_name"]: {"url": url, "transport": "http"}})
        try:
            assert [t["name"] for t in _echo_tools(agent)] == ECHO["qualified"]
            call = ECHO["calls"][0]
            assert await agent.execute_tool(call["tool"], call["arguments"]) == call["text"]
        finally:
            await agent.skills["mcp"].cleanup()
    finally:
        process.terminate()
        process.wait(timeout=10)


async def test_a_server_that_fails_to_connect_is_reported_and_the_others_still_load():
    agent = await _connected({ECHO["server_name"]: STDIO_SERVER, "broken": {"url": "http://127.0.0.1:9/mcp", "transport": "http"}})
    try:
        assert [t["name"] for t in _echo_tools(agent)] == ECHO["qualified"]
    finally:
        await agent.skills["mcp"].cleanup()


async def test_access_tools_naming_the_skill_restricts_the_tools_it_registers_on_start():
    from webagents.access.install import add_access, finish_access

    skills = {"mcp": LocalMcpSkill({"mcp": {ECHO["server_name"]: STDIO_SERVER}})}
    policy = add_access(skills, {"groups": {"friends": ["user:@alice"]}, "tools": {"friends": ["mcp"]}}, None)
    agent = BaseAgent(name="client", instructions="x", skills=skills)
    finish_access(agent, policy, skills)
    await agent._ensure_skills_initialized()
    try:
        names = lambda scopes: [t["name"] for t in agent.get_tools_for_scopes(scopes)]  # noqa: E731
        assert set(ECHO["qualified"]) <= set(names(["owner"]))
        assert set(ECHO["qualified"]) <= set(names(["group:friends"]))
        assert not set(ECHO["qualified"]) & set(names(["group:everyone"]))
        assert not set(ECHO["qualified"]) & set(names([]))
    finally:
        await agent.skills["mcp"].cleanup()
