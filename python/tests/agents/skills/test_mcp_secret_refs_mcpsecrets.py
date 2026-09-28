"""
`${secret:NAME}` and `${env:NAME}` in an MCP server's env, headers and url,
and the environment a stdio server gets (S-292, 2026-09-26), pinned by
`secret_refs` in `tests/fixtures/mcp_tool/config_shapes.json`, which the
TypeScript suite runs too (`tests/unit/skills/mcp-secret-refs-mcpsecrets.test.ts`).

Then the client itself, against the probe server
(`tests/fixtures/mcp_tool/probe_server_mcpsecrets.py`): a stdio server sees
only the MCP SDK's default variables plus its own resolved `env`, never the
agent process's `OPENAI_API_KEY`; a remote server's Authorization header built
from a `${secret:...}` reaches it; and the value shows up in no log line, no
report and no error along the way. The secret store is the CLI's file
fallback under a throwaway HOME, never the keychain.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp.config import servers_from_config
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, owner_reference_sources
from webagents.agents.skills.local.secrets.references import (
    REFERENCE_NAME,
    REFERENCE_SENTENCES,
    SECRET_LOOKING_PATTERNS,
    SECRET_MASK,
    SecretReferenceError,
    at_connect_sentence,
    expand_references,
    literal_warning,
    looks_like_secret,
    mask_text,
    mask_url,
    mask_value,
    secrets_set_hint,
    suggested_secret_name,
)
from webagents.cli.config_store import cli_command

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REFS = json.loads((FIXTURES / "mcp_tool" / "config_shapes.json").read_text())["secret_refs"]
PROBE = json.loads((FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.json").read_text())
CLI_COMMAND = json.loads((FIXTURES / "cli" / "cli_command.json").read_text())
PROBE_SERVER = FIXTURES / "mcp_tool" / "probe_server_mcpsecrets.py"
STDIO_PROBE = {"command": sys.executable, "args": [str(PROBE_SERVER)]}

# Obvious dummies. Nothing here is or resembles a real credential.
TOKEN = "dummy-probe-token-not-a-real-credential-0123456789"
LEAK = "sk-dummy-agent-process-key-that-must-not-leak"


# -- the grammar --------------------------------------------------------------------------------


def test_sentences_mask_name_pattern_and_patterns_match_the_fixture():
    assert REFERENCE_SENTENCES == REFS["sentences"]
    assert SECRET_MASK == REFS["mask"]
    assert REFERENCE_NAME.pattern == REFS["name_pattern"]
    assert list(SECRET_LOOKING_PATTERNS) == REFS["literal_warnings"]["patterns"]


@pytest.mark.parametrize("case", REFS["hints"], ids=[str(c["profile"]) for c in REFS["hints"]])
def test_the_hint_names_the_profile(monkeypatch, case):
    if case["profile"]:
        monkeypatch.setenv("WEBAGENTS_PROFILE", case["profile"])
    else:
        monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    assert secrets_set_hint(case["name"]) == case["hint"]


def test_the_hint_is_the_cli_s_own_command_under_every_profile(monkeypatch):
    for case in CLI_COMMAND["cases"]:
        if case["profile"]:
            monkeypatch.setenv("WEBAGENTS_PROFILE", case["profile"])
        else:
            monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
        assert secrets_set_hint("X") == cli_command("secrets set X")


@pytest.mark.parametrize("case", REFS["expansion"], ids=[c["name"] for c in REFS["expansion"]])
def test_expansion(monkeypatch, case):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    secrets = case.get("secrets", {})
    env = case.get("env", {})
    if "error" in case:
        with pytest.raises(SecretReferenceError) as raised:
            expand_references(case["value"], secrets.get, env)
        assert str(raised.value) == case["error"]
        if case.get("missing_secret"):
            assert raised.value.missing_secret == case["missing_secret"]
        return
    assert expand_references(case["value"], secrets.get, env) == (case["expected"], case["values"])


@pytest.mark.parametrize("case", REFS["command_line"], ids=[c["name"] for c in REFS["command_line"]])
def test_a_reference_on_the_command_line_is_refused(case):
    resolution = servers_from_config(case["config"])
    assert resolution.servers == case["servers"]
    assert resolution.rejected == case["rejected"]


@pytest.mark.parametrize("case", REFS["literal_warnings"]["suggested_names"], ids=[c["suggested"] for c in REFS["literal_warnings"]["suggested_names"]])
def test_the_suggested_name(case):
    assert suggested_secret_name(case["server"], case["key"]) == case["suggested"]


@pytest.mark.parametrize("case", REFS["literal_warnings"]["cases"], ids=[c["name"] for c in REFS["literal_warnings"]["cases"]])
def test_literal_warnings(monkeypatch, case):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    assert servers_from_config(case["config"]).warnings == case["warnings"]


def test_a_reference_is_never_a_literal(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    assert looks_like_secret("Bearer ${secret:A_TOKEN_NAME_LONG_ENOUGH}") is False
    assert "${secret:GH_AUTHORIZATION}" in literal_warning("gh", "headers", "Authorization")


@pytest.mark.parametrize("case", REFS["masking"]["values"], ids=[c["value"] for c in REFS["masking"]["values"]])
def test_masked_values(case):
    assert mask_value(case["value"]) == case["masked"]


@pytest.mark.parametrize("case", REFS["masking"]["urls"], ids=[c["url"] for c in REFS["masking"]["urls"]])
def test_masked_urls(case):
    assert mask_url(case["url"]) == case["masked"]


def test_masked_text():
    case = REFS["masking"]["text"]
    assert mask_text(case["text"], case["values"]) == case["masked"]


@pytest.mark.parametrize("case", REFS["at_connect"], ids=[c["composed"] for c in REFS["at_connect"]])
def test_the_connect_time_sentence(case):
    assert at_connect_sentence(case["server"], case["field"], case["key"], case["sentence"]) == case["composed"]


# -- the client, against the probe server --------------------------------------------------------


@pytest.fixture
def scratch_store(tmp_path, monkeypatch):
    """The CLI's own store, file backend, under a throwaway HOME: what
    `webagents secrets set PROBE_TOKEN` would write. Never the keychain."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR", "OPENAI_API_KEY", "PROBE_FROM_ENV"):
        monkeypatch.delenv(var, raising=False)
    from webagents.cli.commands.secrets import _store

    assert _store(quiet=True).set("PROBE_TOKEN", TOKEN) == "file"
    return _store(quiet=True)


@pytest.fixture
def logged(caplog):
    # The package's loggers do not propagate to the root, so the capture
    # handler is attached to the skill's own logger, at DEBUG: the debug line
    # that describes a server is where a value would leak first.
    logger = logging.getLogger("webagents.skills.mcp")
    caplog.handler.setLevel(logging.DEBUG)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


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


@pytest.fixture
def probe_http():
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(PROBE_SERVER), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        wait_for_port(port, process)
        yield f"http://127.0.0.1:{port}{PROBE['http_path']}"
    finally:
        process.terminate()
        process.wait(timeout=10)


async def _connected(servers: dict, **config) -> BaseAgent:
    # As the agent-file loader builds it (S-295): with the owner's sources.
    # A skill built without them resolves nothing (test_mcp_reference_gate_s295_e2efix.py).
    config.setdefault("references", owner_reference_sources())
    skill = LocalMcpSkill({"mcp": servers, **config})
    agent = BaseAgent(name="client", instructions="x", skills={"mcp": skill})
    await agent._ensure_skills_initialized()
    return agent


def _nothing_logged_holds(caplog, value: str) -> None:
    for record in caplog.records:
        assert value not in record.getMessage(), record.getMessage()


async def test_a_stdio_server_gets_the_default_environment_plus_only_its_own_resolved_env(scratch_store, logged, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", LEAK)
    monkeypatch.setenv("PROBE_FROM_ENV", "from-the-environment")
    agent = await _connected({
        PROBE["server_name"]: {
            **STDIO_PROBE,
            "env": {"PROBE_DECLARED": "${secret:PROBE_TOKEN}", "PROBE_COPIED": "${env:PROBE_FROM_ENV}", "PROBE_PLAIN": "plain"},
        }
    })
    try:
        env = lambda name: agent.execute_tool(f"{PROBE['server_name']}__env", {"name": name})  # noqa: E731
        assert await env("OPENAI_API_KEY") == PROBE["unset"]
        assert await env("PROBE_DECLARED") == TOKEN
        assert await env("PROBE_COPIED") == "from-the-environment"
        assert await env("PROBE_PLAIN") == "plain"
        assert await env("PATH") != PROBE["unset"]
        report = agent.skills["mcp"].server_report()
        assert len(report) == 1
        row = report[0]
        assert row["name"] == PROBE["server_name"] and row["transport"] == "stdio" and row["connected"] is True
        assert row["tools"] == PROBE["qualified"]
        assert row["env"] == {"PROBE_DECLARED": "${secret:PROBE_TOKEN}", "PROBE_COPIED": "${env:PROBE_FROM_ENV}", "PROBE_PLAIN": SECRET_MASK}
        assert row["missing_secrets"] == [] and row["warnings"] == []
        assert TOKEN not in json.dumps(report)
        _nothing_logged_holds(logged, TOKEN)
        _nothing_logged_holds(logged, LEAK)
    finally:
        await agent.skills["mcp"].cleanup()


async def test_a_remote_server_s_authorization_header_built_from_a_secret_reaches_it(scratch_store, logged, probe_http):
    agent = await _connected({
        PROBE["server_name"]: {"url": probe_http, "transport": "http", "headers": {"Authorization": "Bearer ${secret:PROBE_TOKEN}"}}
    })
    try:
        assert await agent.execute_tool(f"{PROBE['server_name']}__authorization", {}) == f"Bearer {TOKEN}"
        row = agent.skills["mcp"].server_report()[0]
        assert row["connected"] is True
        assert row["headers"] == {"Authorization": "Bearer ${secret:PROBE_TOKEN}"}
        assert row["url"] == probe_http
        assert TOKEN not in json.dumps(agent.skills["mcp"].server_report())
        _nothing_logged_holds(logged, TOKEN)
    finally:
        await agent.skills["mcp"].cleanup()


async def test_a_reference_in_the_url_is_resolved_and_a_literal_query_is_masked(scratch_store, logged, probe_http, monkeypatch):
    monkeypatch.setenv("PROBE_FROM_ENV", probe_http)
    agent = await _connected({
        PROBE["server_name"]: {"url": "${env:PROBE_FROM_ENV}?token=${secret:PROBE_TOKEN}", "transport": "http"},
        "literal": {"url": f"{probe_http}?token=literal-not-a-secret", "transport": "http"},
    })
    try:
        assert [(r["name"], r["connected"], r["url"]) for r in agent.skills["mcp"].server_report()] == [
            (PROBE["server_name"], True, "${env:PROBE_FROM_ENV}?token=${secret:PROBE_TOKEN}"),
            ("literal", True, f"{probe_http}?token={SECRET_MASK}"),
        ]
        _nothing_logged_holds(logged, TOKEN)
    finally:
        await agent.skills["mcp"].cleanup()


async def test_an_unresolvable_reference_fails_that_server_by_name_and_the_others_still_load(scratch_store, logged):
    agent = await _connected({
        "broken": {**STDIO_PROBE, "env": {"X": "${secret:NOT_STORED}"}},
        PROBE["server_name"]: STDIO_PROBE,
    })
    try:
        rows = {r["name"]: r for r in agent.skills["mcp"].server_report()}
        assert rows[PROBE["server_name"]]["connected"] is True
        assert rows["broken"]["connected"] is False and rows["broken"]["tools"] == []
        assert rows["broken"]["error"] == 'env X of MCP server "broken": ${secret:NOT_STORED} is not set: store it with `webagents secrets set NOT_STORED`'
        assert rows["broken"]["missing_secrets"] == ["NOT_STORED"]
        assert any(
            "Server 'broken' failed to connect: env X of MCP server \"broken\": ${secret:NOT_STORED} is not set" in r.getMessage()
            for r in logged.records
        )
    finally:
        await agent.skills["mcp"].cleanup()


async def test_an_unset_variable_and_a_reference_in_args_are_each_said_by_name(scratch_store, logged):
    agent = await _connected({
        "novar": {"url": "http://127.0.0.1:9/mcp", "transport": "http", "headers": {"X-Key": "${env:PROBE_NOT_EXPORTED}"}},
        "onargs": {"command": sys.executable, "args": [str(PROBE_SERVER), "${secret:PROBE_TOKEN}"]},
    })
    try:
        rows = {r["name"]: r for r in agent.skills["mcp"].server_report()}
        assert rows["novar"]["error"] == 'headers X-Key of MCP server "novar": ${env:PROBE_NOT_EXPORTED} is not set in the environment'
        assert rows["onargs"]["rejected"] == REFS["sentences"]["commandLine"].replace("{field}", "args")
        assert rows["onargs"]["transport"] == "unknown"
        _nothing_logged_holds(logged, TOKEN)
    finally:
        await agent.skills["mcp"].cleanup()


async def test_a_literal_that_looks_like_a_key_is_warned_about_once_and_the_server_still_connects(scratch_store, logged):
    agent = await _connected({PROBE["server_name"]: {**STDIO_PROBE, "env": {"GH": "ghp_dummy0123456789abcdefghijklmnop"}}})
    try:
        expected = literal_warning(PROBE["server_name"], "env", "GH")
        assert sum(1 for r in logged.records if expected in r.getMessage()) == 1
        row = agent.skills["mcp"].server_report()[0]
        assert row["connected"] is True and row["warnings"] == [expected] and row["env"] == {"GH": SECRET_MASK}
        _nothing_logged_holds(logged, "ghp_dummy0123456789abcdefghijklmnop")
    finally:
        await agent.skills["mcp"].cleanup()


async def test_a_builder_may_hand_in_its_own_sources(scratch_store):
    agent = await _connected(
        {PROBE["server_name"]: {**STDIO_PROBE, "env": {"X": "${secret:FROM_HOST}"}}},
        references={"env": {}, "secret": lambda name: "host-supplied-dummy" if name == "FROM_HOST" else None},
    )
    try:
        assert await agent.execute_tool(f"{PROBE['server_name']}__env", {"name": "X"}) == "host-supplied-dummy"
    finally:
        await agent.skills["mcp"].cleanup()
