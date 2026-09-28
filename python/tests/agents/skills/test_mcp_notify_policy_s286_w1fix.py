"""
S-286 (2026-09-26): a Python agent's MCP `notify` tool policy had no effect,
so an owner who set it expecting to approve each MCP call got no prompt and the
tool ran. Python has no approval channel (no `policyHook`, no notification
skill), so it now REFUSES a server whose `toolPolicies` names `notify`, by
name, at load, rather than running the tool unprompted. `block` and `allow`
are kept as before.

Pinned against the shared fixture `tests/fixtures/mcp_tool/config_shapes.json`
(`tool_policies`), which the TypeScript suite reads too
(`tests/unit/skills/mcp-notify-policy-s286-w1fix.test.ts`); the fixture records
that TypeScript honours `notify` through a host `policyHook` while Python
refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path

from webagents.agents.skills.local.mcp.config import NOTIFY_POLICY_UNSUPPORTED, servers_from_config

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mcp_tool"
SHAPES = json.loads((FIXTURES / "config_shapes.json").read_text())
POLICIES = SHAPES["tool_policies"]


def test_the_reason_matches_the_fixture():
    assert NOTIFY_POLICY_UNSUPPORTED == POLICIES["python"]["rejected"][0]["reason"]


def test_a_server_asking_for_notify_is_refused_by_name_and_the_others_load():
    resolution = servers_from_config(POLICIES["config"])
    assert [s["name"] for s in resolution.servers] == POLICIES["python"]["loads"]
    assert resolution.rejected == POLICIES["python"]["rejected"]


def test_block_and_allow_policies_are_kept_not_refused():
    # A tool the owner blocks or allows is not a refusal: only notify is, and
    # the config is kept untouched for the skill to read.
    config = {"srv": {"command": "npx", "args": ["s"], "toolPolicies": {"a": "block", "b": "allow"}}}
    resolution = servers_from_config(config)
    assert [s["name"] for s in resolution.servers] == ["srv"]
    assert resolution.rejected == []
    assert resolution.configs["srv"]["toolPolicies"] == {"a": "block", "b": "allow"}


def test_a_notify_among_other_policies_still_refuses_the_server():
    config = {"srv": {"url": "https://mcp.example/", "toolPolicies": {"a": "allow", "danger": "notify"}}}
    resolution = servers_from_config(config)
    assert resolution.servers == []
    assert resolution.rejected == [{"name": "srv", "reason": NOTIFY_POLICY_UNSUPPORTED}]
