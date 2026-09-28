"""
The command list gates unconfined commands only (the ptypass-fixes lane,
2026-09-27; fixture `tests/fixtures/sandbox/srt.json` `shell_allowlist`).

The real-terminal PTY pass found `sleep` and `mkdir` refused in both SDKs,
`python3` in TypeScript only and `rg` in Python only, before the sandbox was
ever reached, while `docs/cli/sandbox.md` says the kernel is the boundary.
Confined, a command is now refused only for a name the agent file blocks;
unconfined, one list and one wording hold in both SDKs. The TypeScript twin
is `typescript/tests/unit/skills/ptypass-fixes-shell-allowlist.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from webagents.agents.skills.local.shell import skill as shell
from webagents.agents.skills.local.shell.skill import ShellSkill
from webagents.sandbox import backend_status, sandbox_available

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "sandbox" / "srt.json").read_text())["shell_allowlist"]
requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")


def test_one_list_and_one_wording():
    assert list(shell.DEFAULT_ALLOWED) == FIXTURE["allowed"]
    assert list(shell.DEFAULT_BLOCKED) == FIXTURE["blocked"]
    assert shell.NOT_ALLOWED == FIXTURE["not_allowed"]
    assert shell.IS_BLOCKED == FIXTURE["is_blocked"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["command"] for c in FIXTURE["cases"]])
def test_the_gate_case_by_case(case):
    config = {}
    if case.get("declared_blocked"):
        config["blocked_commands"] = case["declared_blocked"]
    if case.get("declared_allowed"):
        config["allowed_commands"] = case["declared_allowed"]
    skill = ShellSkill(config)
    for mode in ("confined", "unconfined"):
        allowed, reason = skill._check_command(case["command"], confined=(mode == "confined"))
        expected = case[mode]
        assert (allowed, reason) == ((True, "") if expected is None else (False, expected)), mode


def test_the_tool_says_it_with_the_prefix_when_unconfined(tmp_path):
    skill = ShellSkill({"base_dir": str(tmp_path), "sandbox": "off"})
    answer = asyncio.run(skill.run_command("sleep 0"))
    assert answer == FIXTURE["prefix"] + FIXTURE["not_allowed"].format(name="sleep")


@requires_backend
def test_confined_sleep_and_mkdir_run_and_the_kernel_still_holds(tmp_path):
    skill = ShellSkill({"base_dir": str(tmp_path)})
    answer = asyncio.run(skill.run_command("sleep 0.1 && mkdir made && echo MADE"))
    assert "MADE" in answer and (tmp_path / "made").is_dir()
    # Not refused for its name, and still confined: no write outside the folder.
    outside = tmp_path.parent / f"ptypass-fixes-outside-{tmp_path.name}"
    answer = asyncio.run(skill.run_command(f"mkdir {outside} && echo MADE"))
    assert "MADE" not in answer and not outside.exists()
