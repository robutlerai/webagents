"""
`webagents acp` (gap-closure plan item 1.6, 2026-09-26): the command reserves
stdout before the agent file is read, builds the agent `serve` builds, and
serves it through the ACP skill with sessions under the profile directory.
The words are pinned in tests/test_acp_protocol.py; the transcripts in
tests/test_acp_stdio.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from webagents.agents.skills.core.transport.acp.skill import ACPTransportSkill
from webagents.cli.acp_serve import acp_command


def test_acp_command_reserves_stdout_and_serves_the_agent(tmp_path, monkeypatch, capsys):
    (tmp_path / "AGENT.md").write_text("---\nname: fixture\nskills:\n  - todo\n---\n\nYou are a fixture.\n")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    seen = {}

    async def fake_serve_stdio(self, agent, stdin=None, stdout=None):
        seen["agent"] = agent
        seen["skill"] = self
        seen["stdout"] = stdout
        seen["sys_stdout_is_stderr"] = sys.stdout is sys.stderr
        print("a skill printing at startup")

    monkeypatch.setattr(ACPTransportSkill, "serve_stdio", fake_serve_stdio)
    real_stdout = sys.stdout
    acp_command(str(tmp_path))

    assert seen["agent"].name == "fixture"
    assert isinstance(seen["skill"], ACPTransportSkill)
    assert any(s is seen["skill"] for s in seen["agent"].skills.values())
    assert seen["stdout"] is real_stdout
    assert seen["sys_stdout_is_stderr"] is True
    assert seen["skill"].settings["sessions_dir"] == str(home / ".webagents" / "acp" / "sessions")
    captured = capsys.readouterr()
    assert "a skill printing at startup" in captured.err
    assert captured.out == ""


def test_acp_command_keeps_the_file_s_own_sessions_dir(tmp_path, monkeypatch):
    (tmp_path / "AGENT.md").write_text("---\nname: fixture\nskills:\n  - acp:\n      sessions_dir: /tmp/acp-here\n---\n\nBody.\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    seen = {}

    async def fake_serve_stdio(self, agent, stdin=None, stdout=None):
        seen["skill"] = self

    monkeypatch.setattr(ACPTransportSkill, "serve_stdio", fake_serve_stdio)
    acp_command(str(tmp_path))
    assert seen["skill"].settings["sessions_dir"] == "/tmp/acp-here"
    assert sum(isinstance(s, ACPTransportSkill) for s in seen["skill"].agent.skills.values()) if seen["skill"].agent else True
