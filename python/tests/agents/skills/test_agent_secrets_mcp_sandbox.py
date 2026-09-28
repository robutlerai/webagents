"""
An MCP server entry that asks for a sandbox the agent cannot provide does not
start (S-313, 2026-09-27, the agent-secrets lane): refused by name when the
skill initializes, with the shared fixture's sentence
(`tests/fixtures/mcp_tool/config_shapes.json`, `sandbox_key`), listed as
`rejected` by `server_report()`, and honoured only when the Docker `sandbox`
skill is loaded. The TypeScript suite runs its side in
`tests/unit/skills/agent-secrets-mcp-sandbox.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.skills.local.mcp.config import SANDBOX_UNAVAILABLE, asks_for_sandbox, problem_lines, servers_from_config
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, McpConnectError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_tool"
CASE = json.loads((FIXTURES / "config_shapes.json").read_text())["sandbox_key"]


class SandboxSkill:  # noqa: D101 - the Docker skill, by the name the MCP skill looks for
    pass


def skill_with(agent_skills: dict) -> LocalMcpSkill:
    skill = LocalMcpSkill({"mcp": CASE["config"], "agent_name": "t"})
    skill.agent = SimpleNamespace(skills=agent_skills, name="t")
    return skill


def test_the_sentence_is_the_fixture_s():
    assert SANDBOX_UNAVAILABLE == CASE["refusal"]["python"]


def test_asks_when_the_key_holds_anything_but_false_and_none():
    assert asks_for_sandbox({"sandbox": True}) is True
    assert asks_for_sandbox({"sandbox": "docker"}) is True
    assert asks_for_sandbox({"sandbox": 1}) is True
    assert asks_for_sandbox({"sandbox": False}) is False
    assert asks_for_sandbox({"sandbox": None}) is False
    assert asks_for_sandbox({}) is False
    assert asks_for_sandbox(None) is False


def test_the_normalizer_alone_keeps_the_key_for_the_skill_to_decide():
    # The normalizer never sees the agent's skills, so every entry loads there
    # with its `sandbox` kept on the config.
    resolution = servers_from_config(CASE["config"])
    assert [s["name"] for s in resolution.servers] == list(CASE["config"])
    assert resolution.configs["boxed"]["sandbox"] is True


def test_without_the_sandbox_skill_the_entry_is_refused_by_name_and_the_others_load():
    expected = CASE["python"]["without_sandbox_skill"]
    skill = skill_with({})
    resolution = skill._load_mcp_config()
    skill._refuse_unprovidable_sandbox(resolution)
    assert [s["name"] for s in resolution.servers] == expected["loads"]
    assert resolution.rejected == [{"name": name, "reason": CASE["refusal"]["python"]} for name in expected["rejected"]]
    for name in expected["rejected"]:
        assert name not in resolution.configs
    # `server_report()` and the CLI's problem lines say it like any refused entry.
    skill._resolution = resolution
    report = skill.server_report()
    assert [row["name"] for row in report if row.get("rejected")] == expected["rejected"]
    assert problem_lines(report)[: len(expected["rejected"])] == [
        f'[MCPSkill] Server "{name}" {CASE["refusal"]["python"]}; skipping it.' for name in expected["rejected"]
    ]


def test_with_the_sandbox_skill_every_entry_loads():
    expected = CASE["python"]["with_sandbox_skill"]
    skill = skill_with({"sandbox": SandboxSkill()})
    resolution = skill._load_mcp_config()
    skill._refuse_unprovidable_sandbox(resolution)
    assert [s["name"] for s in resolution.servers] == expected["loads"]
    assert resolution.rejected == expected["rejected"]


def test_a_direct_connect_of_a_boxed_entry_without_the_skill_raises_rather_than_running_locally():
    pytest.importorskip("mcp")
    skill = skill_with({})
    server = {"name": "boxed", "transport": "stdio", "command": "srv", "args": []}
    with pytest.raises(McpConnectError) as caught:
        asyncio.run(skill._open_server("boxed", server, dict(CASE["config"]["boxed"])))
    assert str(caught.value) == CASE["refusal"]["python"]
