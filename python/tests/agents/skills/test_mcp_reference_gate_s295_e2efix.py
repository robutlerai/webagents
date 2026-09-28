"""
S-295 (CRITICAL, 2026-09-26): `${env:NAME}` and `${secret:NAME}` in an MCP
server's url, headers and env resolve ONLY when the skill's builder hands in
the sources (config key `references`). The skill resolved them for every
skill against `os.environ` and the CLI keystore, so a host that built one from
data its users saved could have its own environment expanded into a URL and
sent to that server.

Pinned by `secret_refs.resolution_gate` in
`tests/fixtures/mcp_tool/config_shapes.json`, which the TypeScript suite runs
too (`tests/unit/skills/mcp-reference-gate-s295-e2efix.test.ts`): a host-built
skill keeps every field literal (proved on the wire against the probe server:
the header arrives as the bytes written), while the agent-file loaders' skill
resolves as before.
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
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, owner_reference_sources

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
GATE = json.loads((FIXTURES / "mcp_tool" / "config_shapes.json").read_text())["secret_refs"]["resolution_gate"]
PROBE = json.loads((FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.json").read_text())
PROBE_SERVER = FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.py"


def _server_from(cases) -> dict:
    config: dict = {"command": "srv"}
    for case in cases:
        if case.get("key"):
            config.setdefault(case["field"], {})[case["key"]] = case["value"]
        else:
            config[case["field"]] = case["value"]
    return config


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def host_environment(tmp_path, monkeypatch):
    """A scratch HOME with the file backend, and the host's own variable set:
    what a saved URL must never reach."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(GATE["variable"], GATE["variable_value"])


@pytest.fixture
def probe_http():
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(PROBE_SERVER), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"the probe exited with {process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("the probe never listened")
        yield f"http://127.0.0.1:{port}{PROBE['http_path']}"
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_the_fixture_names_the_config_key_and_the_two_sources():
    assert GATE["config_key"] == "references"
    assert GATE["sources"] == ["env", "secret"]


def test_a_host_built_skill_keeps_every_field_literal():
    config = _server_from(GATE["literal_cases"])
    skill = LocalMcpSkill({"mcp": {"srv": config}})
    live, values = skill._resolve_references("srv", config)
    assert live == config
    assert values == []
    assert GATE["variable_value"] not in json.dumps(live)


def test_a_host_built_skill_given_a_url_with_a_reference_keeps_the_literal_url():
    config = {"url": f"https://mcp.example.test/mcp?k=${{env:{GATE['variable']}}}", "transport": "http"}
    skill = LocalMcpSkill({"mcp": {"srv": config}})
    live, _values = skill._resolve_references("srv", config)
    assert live["url"] == config["url"]


async def test_on_the_wire_a_host_built_skill_sends_the_header_as_the_literal_bytes_written(probe_http):
    header = GATE["resolved_header"]
    skill = LocalMcpSkill({"mcp": {PROBE["server_name"]: {"url": probe_http, "transport": "http", "headers": {header["key"]: header["value"]}}}})
    agent = BaseAgent(name="host", instructions="x", skills={"mcp": skill})
    await agent._ensure_skills_initialized()
    try:
        assert await agent.execute_tool(f"{PROBE['server_name']}__authorization", {}) == header["value"]
    finally:
        await skill.cleanup()


async def test_with_the_sources_handed_in_the_same_header_resolves(probe_http):
    header = GATE["resolved_header"]
    skill = LocalMcpSkill({
        "mcp": {PROBE["server_name"]: {"url": probe_http, "transport": "http", "headers": {header["key"]: header["value"]}}},
        "references": owner_reference_sources(),
    })
    agent = BaseAgent(name="owner", instructions="x", skills={"mcp": skill})
    await agent._ensure_skills_initialized()
    try:
        assert await agent.execute_tool(f"{PROBE['server_name']}__authorization", {}) == header["expected"]
    finally:
        await skill.cleanup()


def test_the_sources_are_injected_not_the_process_environment():
    config = {"url": "https://mcp.example.test/mcp", "transport": "http", "headers": {"X": f"${{env:{GATE['variable']}}}"}}
    skill = LocalMcpSkill({
        "mcp": {"srv": config},
        "references": {"env": {GATE["variable"]: "from-the-injected-mapping"}, "secret": lambda name: None},
    })
    live, values = skill._resolve_references("srv", config)
    assert live["headers"]["X"] == "from-the-injected-mapping"
    assert values == ["from-the-injected-mapping"]


async def test_the_agent_file_loader_builds_the_skill_with_the_owners_sources(tmp_path):
    from webagents.cli.agent_builder import build_agent

    (tmp_path / "AGENT.md").write_text(
        "---\nname: a\nskills:\n  - mcp:\n      srv:\n        url: https://mcp.example.test/mcp\n"
        f"        headers:\n          X: '${{env:{GATE['variable']}}}'\n---\nBody\n"
    )
    built = await build_agent(tmp_path / "AGENT.md", working_dir=tmp_path, initialize=False)
    skill = built.agent.skills["mcp"]
    assert isinstance(skill._references, dict) and set(skill._references) == {"env", "secret"}
    config = {"url": "https://mcp.example.test/mcp", "transport": "http", "headers": {"X": f"${{env:{GATE['variable']}}}"}}
    live, _values = skill._resolve_references("srv", config)
    assert live["headers"]["X"] == GATE["variable_value"]
