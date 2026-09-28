"""
S-286 addendum (2026-09-26): `enabledTools` and `toolPolicies: block` are
applied at discovery, as TypeScript applies them. This SDK registered every
tool a server listed, so a tool the owner blocked stayed callable. Pinned by
the shared fixture `tool_policies.filtering` in
`tests/fixtures/mcp_tool/config_shapes.json`, which the TypeScript suite runs
too (`tests/unit/skills/mcp-tool-filtering-s286-e2efix.test.ts`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp.config import filter_discovered_tools
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill

FILTERING = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "mcp_tool" / "config_shapes.json").read_text())["tool_policies"]["filtering"]


def _listed(names):
    return [SimpleNamespace(name=name, description=f"tool {name}", inputSchema={"type": "object", "properties": {}}) for name in names]


class _Session:
    def __init__(self, names):
        self.names = names

    async def list_tools(self):
        return SimpleNamespace(tools=_listed(self.names))


@pytest.mark.parametrize("case", FILTERING["cases"], ids=[c["name"] for c in FILTERING["cases"]])
def test_the_filter_keeps_what_the_fixture_says(case):
    kept = filter_discovered_tools(_listed(case["listed"]), case["entry"])
    assert [f"srv__{t.name}" for t in kept] == case["registered"]


@pytest.mark.parametrize("case", FILTERING["cases"], ids=[c["name"] for c in FILTERING["cases"]])
def test_discovery_registers_only_the_filtered_tools(case):
    skill = LocalMcpSkill({"mcp": {"srv": case["entry"]}})
    skill.agent = BaseAgent(name="t", instructions="x", skills={})
    skill._server_configs["srv"] = case["entry"]
    asyncio.run(skill._discover_capabilities("srv", _Session(case["listed"])))
    assert sorted(skill.tools_registry) == sorted(case["registered"])


def test_an_entry_the_file_wrote_reaches_discovery_through_connect(monkeypatch):
    # `_connect_server` records the entry before opening the server, so what
    # discovery filters on is the entry as written, not the resolved copy.
    skill = LocalMcpSkill({"mcp": {"srv": {"command": "srv", "enabledTools": ["a"]}}})
    recorded = {}

    async def fake_open(name, live, config):
        recorded["config"] = config

    monkeypatch.setattr(skill, "_open_server", fake_open)
    asyncio.run(skill._connect_server("srv", {"name": "srv", "transport": "stdio", "command": "srv", "args": []}, {"command": "srv", "enabledTools": ["a"]}))
    assert skill._server_configs["srv"] == {"command": "srv", "enabledTools": ["a"]}
    assert recorded["config"] == {"command": "srv", "enabledTools": ["a"]}
