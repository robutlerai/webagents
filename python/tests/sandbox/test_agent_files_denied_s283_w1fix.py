"""
S-283 (2026-09-26): a confined command could rewrite the agent's own
definition and context files (`AGENT.md`, `AGENT-*.md`, `WEBAGENTS.md`,
`mcp.json`) and other skills' `SKILL.md`, and the daemon reloaded them. The
agent's folder is a write root under `development` and under the default
`strict`, and only `.webagents/*` was denied inside it; a SKILL.md script
protected only its own folder.

Pinned here, against the shared fixture's `agent_file_deny` section
(`tests/fixtures/sandbox/srt.json`; the TypeScript twin is
`tests/unit/sandbox/agent-files-denied-s283-w1fix.test.ts`): the escalation
set carries the four literal names and the `AGENT-*.md` pattern, resolved
per engine (enumerated by name on both, the glob itself on macOS); under
real srt, both presets refuse every write, creation and rename of those
files while a plain file in the agent folder still writes; and a SKILL.md
script cannot write a sibling skill's `SKILL.md` or `AGENT.md`, because
every skill folder is read-only for it now.

Enforcement tests run srt for real (`conftest.py` says where it comes from)
and skip, with the reason, where it cannot run. The one expectation that
differs by engine, a matching file that does not exist when the command
starts, is asserted on macOS and recorded as the Linux residual the fixture
states.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd import SkillMdSkill, discover_skills
from webagents.cli.loader.schema import SandboxConfig
from webagents.sandbox import (
    AGENT_FILE_PATTERNS,
    ESCALATION_DENY,
    agent_file_denies,
    backend_status,
    matches_agent_file_pattern,
    policy_from_metadata,
    run_sandboxed,
    sandbox_available,
)
from webagents.sandbox import srt as engine

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = json.loads((FIXTURES / "sandbox" / "srt.json").read_text())
SKILLMD = json.loads((FIXTURES / "skillmd" / "skillmd.json").read_text())
REPO = FIXTURES / "skillmd" / SKILLMD["sample"]["repo"]
PROBE = SKILLMD["scripts"]["probe"]

requires_backend = pytest.mark.skipif(not sandbox_available(), reason=f"no srt here: {backend_status()['reason']}")

#: The four literal names S-283 added to the escalation set.
LITERAL_AGENT_FILES = ("AGENT.md", "WEBAGENTS.md", "mcp.json", ".agents/skills")
DARWIN = platform.system() == "Darwin"

#: The declarations as an agent file makes them, through the loader's schema,
#: so `strict` carries its default `allowed_folders: ["."]` (the agent's own
#: folder) exactly as `sandbox:\n  preset: strict` does; the TypeScript
#: parser fills the same default.
PRESETS = [
    pytest.param(SandboxConfig(preset="development"), id="development"),
    pytest.param(SandboxConfig(preset="strict"), id="strict"),
]


@pytest.fixture
def agent_dir(tmp_path):
    """An agent folder as the daemon serves one: its definition, a named
    sibling agent file, the inherited context, an MCP config, and the sample
    repository's two skills installed under `.agents/skills`."""
    folder = tmp_path / "agent"
    skills = folder / ".agents" / "skills"
    skills.mkdir(parents=True)
    for name in ("pdf", "xlsx"):
        shutil.copytree(REPO / "skills" / name, skills / name)
    for name in ("AGENT.md", "AGENT-helper.md", "WEBAGENTS.md", "mcp.json"):
        (folder / name).write_text("original\n")
    return Path(os.path.realpath(folder))


class TestTheFixtureIsTheContract:
    def test_the_agent_files_are_in_the_escalation_set_and_the_patterns_beside_it(self):
        for name in LITERAL_AGENT_FILES:
            assert name in ESCALATION_DENY
        assert list(ESCALATION_DENY) == FIXTURE["escalation_deny"]
        assert list(AGENT_FILE_PATTERNS) == FIXTURE["agent_file_deny"]["patterns"]
        assert FIXTURE["agent_file_deny"]["glob_on"] == ["darwin"]

    def test_the_pattern_matches_the_daemons_agent_file_names_and_nothing_else(self):
        for name in ("AGENT-helper.md", "AGENT-evil.md", "AGENT-.md"):
            assert matches_agent_file_pattern(name), name
        for name in ("AGENT.md", "AGENTS.md", "agent-x.md", "AGENT-x.md.bak", "AGENT_x.md", "notes/AGENT-x.md"):
            assert not matches_agent_file_pattern(name), name

    def test_existing_matches_are_enumerated_on_both_engines_and_the_glob_only_on_macos(self, agent_dir):
        root = str(agent_dir)
        assert agent_file_denies(root, system="Linux") == [os.path.join(root, "AGENT-helper.md")]
        assert agent_file_denies(root, system="Darwin") == [os.path.join(root, "AGENT-helper.md"), os.path.join(root, "AGENT-*.md")]
        missing = os.path.join(root, "nowhere")
        assert agent_file_denies(missing, system="Linux") == []
        assert agent_file_denies(missing, system="Darwin") == [os.path.join(missing, "AGENT-*.md")]

    @pytest.mark.parametrize("declared", PRESETS)
    def test_they_are_denied_in_every_write_root(self, agent_dir, declared):
        root = str(agent_dir)
        policy = policy_from_metadata(declared, cwd=root)
        assert root in policy.write_roots
        denied = policy.deny_writes
        for name in (*LITERAL_AGENT_FILES, "AGENT-helper.md"):
            assert os.path.join(root, name) in denied, (declared, name)
        assert (os.path.join(root, "AGENT-*.md") in denied) is DARWIN

    def test_they_reach_the_settings_srt_reads(self, agent_dir):
        root = str(agent_dir)
        policy = policy_from_metadata({"preset": "development"}, cwd=root)
        settings = engine.build_settings(policy, deps={})
        for name in (*LITERAL_AGENT_FILES, "AGENT-helper.md"):
            assert os.path.join(root, name) in settings["filesystem"]["denyWrite"], name
        if DARWIN:
            assert os.path.join(root, "AGENT-*.md") in settings["filesystem"]["denyWrite"]


def _probe_command(agent_dir: Path) -> str:
    """One shell line per attempt, each reporting `WROTE <label>` or
    `denied <label>`, so a single confined command probes every file."""

    def write(relative: str) -> str:
        target = agent_dir / relative
        return f'if (echo pwned > "{target}") 2>/dev/null; then echo "WROTE {relative}"; else echo "denied {relative}"; fi'

    return "\n".join(
        [
            write("AGENT.md"),
            write("AGENT-helper.md"),
            write("AGENT-evil.md"),
            write("WEBAGENTS.md"),
            write("mcp.json"),
            write(".agents/skills/xlsx/SKILL.md"),
            f'if mkdir -p "{agent_dir / ".agents/skills/new"}" 2>/dev/null; then echo "WROTE .agents/skills/new"; else echo "denied .agents/skills/new"; fi',
            f'if mv "{agent_dir / "AGENT-helper.md"}" "{agent_dir / "AGENT-moved.md"}" 2>/dev/null; then echo "WROTE rename"; else echo "denied rename"; fi',
            f'if rm "{agent_dir / "AGENT.md"}" 2>/dev/null; then echo "WROTE unlink"; else echo "denied unlink"; fi',
            write("work.txt"),
        ]
    )


def _verdicts(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[0] in ("WROTE", "denied"):
            out[parts[1]] = parts[0]
    return out


@requires_backend
class TestAConfinedCommandCannotWriteTheAgentFiles:
    @pytest.mark.parametrize("declared", PRESETS)
    def test_under_the_preset(self, agent_dir, declared):
        policy = policy_from_metadata(declared, cwd=str(agent_dir))
        result = run_sandboxed(_probe_command(agent_dir), policy, timeout=30)
        seen = _verdicts(result.stdout)
        denied = [
            "AGENT.md",
            "AGENT-helper.md",
            "WEBAGENTS.md",
            "mcp.json",
            ".agents/skills/xlsx/SKILL.md",
            ".agents/skills/new",
            "rename",
            "unlink",
        ]
        # A matching file that does not exist yet: the glob covers it on
        # macOS; on Linux srt takes no write glob (fixture `agent_file_deny.linux`).
        if DARWIN:
            denied.append("AGENT-evil.md")
        for label in denied:
            assert seen.get(label) == "denied", (label, result.stdout, result.stderr)
        assert seen.get("work.txt") == "WROTE", result.stdout
        # The host agrees: nothing of the agent's changed, and the plain file landed.
        for name in ("AGENT.md", "AGENT-helper.md", "WEBAGENTS.md", "mcp.json"):
            assert (agent_dir / name).read_text() == "original\n", name
        assert (agent_dir / ".agents/skills/xlsx/SKILL.md").read_text() == (REPO / "skills" / "xlsx" / "SKILL.md").read_text()
        assert not (agent_dir / ".agents/skills/new").exists()
        assert not (agent_dir / "AGENT-moved.md").exists()
        if DARWIN:
            assert not (agent_dir / "AGENT-evil.md").exists()
        assert (agent_dir / "work.txt").read_text() == "pwned\n"


def _skill(agent_dir: Path, sandbox=None) -> SkillMdSkill:
    found = discover_skills(str(agent_dir))
    assert found.names() == ["pdf", "xlsx"]
    return SkillMdSkill({"skills": found.skills, "agent_dir": str(agent_dir), "sandbox": sandbox})


class TestASkillScriptCannotWriteASiblingSkillOrTheAgentFile:
    def test_every_skill_folder_is_read_only_for_a_script_not_only_its_own(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        found = discover_skills(str(agent_dir))
        pdf = next(s for s in found.skills if s.name == "pdf")
        policy = skill.script_policy(pdf)
        assert not isinstance(policy, str)
        for each in found.skills:
            assert each.directory in policy.read_only, each.name
        # And the agent's own files are in the denies the script's settings get.
        denied = policy.deny_writes
        for name in (*LITERAL_AGENT_FILES, "AGENT-helper.md"):
            assert os.path.join(str(agent_dir), name) in denied, name

    @requires_backend
    def test_the_sibling_skill_md_and_agent_md_are_refused_under_development(self, agent_dir):
        skill = _skill(agent_dir, sandbox={"preset": "development"})
        sibling = agent_dir / ".agents" / "skills" / "xlsx" / "SKILL.md"
        own = agent_dir / ".agents" / "skills" / "pdf" / "planted.md"
        definition = agent_dir / "AGENT.md"
        plain = agent_dir / "work.txt"
        out = asyncio.run(
            skill.run_skill_script(
                skill="pdf",
                script="scripts/probe.py",
                args=["--write", str(sibling), "--write", str(own), "--write", str(definition), "--write", str(plain)],
            )
        )
        lines = out.strip().split("\n")
        assert lines[0] == PROBE["ok"]
        assert lines[1].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[2].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[3].startswith(PROBE["write_refused_starts_with"]), out
        assert lines[4].startswith(PROBE["write_ok_starts_with"]), out
        assert sibling.read_text() == (REPO / "skills" / "xlsx" / "SKILL.md").read_text()
        assert not own.exists()
        assert definition.read_text() == "original\n"
        assert plain.exists()
