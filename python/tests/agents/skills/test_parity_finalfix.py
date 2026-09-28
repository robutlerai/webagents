"""
Four parity gaps between the SDKs closed on the Python side or pinned here
(2026-09-27, the final e2e re-run), each against the shared fixture the
TypeScript suite reads too (`tests/unit/skills/parity-finalfix.test.ts`):

1. a bare `- mcp` entry, which reads the folder's mcp.json, resolves the
   owner's `${env:NAME}` and `${secret:NAME}` references, as a `- mcp: {...}`
   entry does (`mcp_tool/config_shapes.json`, `folder_mcp_json`);
2. the `- shell: {allowed_commands: [...]}` block widens the allow-list in
   both SDKs (`sandbox/srt.json`, `shell_block`);
3. `list_mcp_servers` and the shell tool carry the same model-facing
   description in both SDKs (`config_shapes.json` `tools`, `srt.json`
   `shell_tool`).

The daemon's `/cron` alias is `tests/daemon/test_cron_routes_parity_finalfix.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from webagents.agents.skills.local.mcp.skill import LocalMcpSkill
from webagents.agents.skills.local.shell.skill import ShellSkill
from webagents.cli.agent_builder import load_skills

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
MCP = json.loads((FIXTURES / "mcp_tool" / "config_shapes.json").read_text())
SRT = json.loads((FIXTURES / "sandbox" / "srt.json").read_text())


def test_a_bare_mcp_entry_resolves_the_owners_references_from_the_folders_mcp_json(tmp_path, monkeypatch):
    case = MCP["folder_mcp_json"]
    (tmp_path / "mcp.json").write_text(json.dumps(case["mcp_json"]))
    for name, value in case["environment"].items():
        monkeypatch.setenv(name, value)
    skills = load_skills(["mcp"], agent_name="a", agent_path=tmp_path / "AGENT.md")
    skill = skills["mcp"]
    assert isinstance(skill, LocalMcpSkill)
    # The sources are the owner's (the same the `- mcp: {...}` shape gets).
    assert isinstance(skill._references, dict) and "env" in skill._references and "secret" in skill._references
    servers = {s["name"]: s for s in skill._load_mcp_config().servers}
    assert skill.config_source == "mcp.json" and case["server"] in servers
    live, values = skill._resolve_references(case["server"], servers[case["server"]])
    assert live["env"] == case["resolved_env"]
    assert values == list(case["resolved_env"].values())
    # The configuration as written keeps the reference, never the value.
    assert servers[case["server"]]["env"] == case["mcp_json"]["mcpServers"][case["server"]]["env"]


def test_the_shell_block_spellings_widen_the_lists():
    case = SRT["shell_block"]["case"]
    skill = ShellSkill(dict(case["declared"]))
    for name in case["allowed_includes"]:
        assert name in skill.allowed_commands
    for name in case["blocked_includes"]:
        assert name in skill.blocked_commands


def test_the_tool_descriptions_are_the_fixtures():
    assert LocalMcpSkill.list_mcp_servers._tool_description == MCP["tools"]["list_mcp_servers"]["description"]
    assert LocalMcpSkill.list_mcp_servers._webagents_tool_definition["function"]["description"] == MCP["tools"]["list_mcp_servers"]["description"]
    tool = SRT["shell_tool"]
    assert ShellSkill.run_command._tool_name == tool["name"]["python"]
    assert ShellSkill.run_command._tool_description == tool["description"]
    assert ShellSkill.run_command._webagents_tool_definition["function"]["description"] == tool["description"]
    assert ShellSkill.run_command._tool_scope == "owner"
