"""
SKILL.md scripts are confined commands too (the sandbox-default lane,
2026-09-27): an agent that runs them is not one that "cannot run commands",
so `doctor` and `/sandbox` say the state its scripts run under
(`SkillMdSkill.script_state`, `cli/doctor.py` `_sandbox_check`). The
TypeScript twin is `typescript/tests/unit/sandbox/sandbox-default-scripts.test.ts`;
the words come from `tests/fixtures/sandbox/srt.json` (`status`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from webagents.agents.skills.local.skillmd import SkillMdSkill, discover_skills

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SKILLMD = json.loads((FIXTURES / "skillmd" / "skillmd.json").read_text())
SRT = json.loads((FIXTURES / "sandbox" / "srt.json").read_text())
REPO = FIXTURES / "skillmd" / SKILLMD["sample"]["repo"]


def _skill(sandbox=None) -> SkillMdSkill:
    found = discover_skills(str(REPO), SKILLMD["sample"]["explicit"])
    return SkillMdSkill({"skills": found.skills, "skipped": found.skipped, "warnings": found.warnings, "agent_dir": str(REPO), "sandbox": sandbox})


def test_the_state_scripts_run_under():
    assert _skill().script_state() == "strict (default)"
    assert _skill({"preset": "development", "network": ["example.com"]}).script_state() == SRT["status"]["agent_file"].replace("{preset}", "development")
    assert _skill(False).script_state() == SRT["status"]["off_agent_file"]
    assert _skill("off").script_state() == SRT["status"]["off_agent_file"]
    assert _skill({"preset": "stirct"}).script_state() == "invalid (agent file)"
    assert SkillMdSkill({"skills": [], "agent_dir": str(REPO)}).script_state() == ""


def test_doctor_names_the_scripts_state_for_an_agent_with_no_shell(monkeypatch):
    import webagents.sandbox as sandbox
    from webagents.cli.doctor import _sandbox_check

    built = SimpleNamespace(agent=SimpleNamespace(skills={"agent_skills": _skill()}), sandbox=None)
    monkeypatch.setattr(sandbox, "backend_status", lambda: {"available": True, "backend": "srt", "version": "0.0.77", "reason": "", "found": ""})
    check = _sandbox_check(built)
    assert check.status == "ok" and check.detail == "strict (default) for SKILL.md scripts, enforced by srt 0.0.77"
    monkeypatch.setattr(sandbox, "backend_status", lambda: {"available": False, "backend": None, "version": None, "reason": "srt is not installed", "found": ""})
    check = _sandbox_check(built)
    assert check.status == "fail" and check.detail == "strict (default) for SKILL.md scripts, but srt is not installed: scripts are refused"
    assert "--no-sandbox" in check.fix
    # No skills at all is still "cannot run commands".
    none = SimpleNamespace(agent=SimpleNamespace(skills={}), sandbox=None)
    assert _sandbox_check(none).detail == "not needed: the agent cannot run commands"
