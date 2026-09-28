"""
Loading SKILL.md skills (gap-closure plan item 1.4, 2026-09-26), against the
shared fixture `tests/fixtures/skillmd/skillmd.json`, which the TypeScript
suite reads too (`tests/unit/skills/skillmd-loader.test.ts`): the parser's
cases (CRLF, a closing fence with no newline, a BOM, an unquoted colon,
metadata values as strings, the three spellings of allowed-tools, other
clients' keys kept, the two reasons a skill is skipped), discovery under
`.agents/skills` and through `agent_skills:`, and the words the model sees.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd.skillmd_loader import (
    AUTO_SKILLS_DIR,
    CATALOG_PREAMBLE,
    EXPLICIT_KEY,
    KNOWN_KEYS,
    SKILL_FILE_NAMES,
    SKIPPED_DIRS,
    SkillMd,
    SkippedSkill,
    activation_text,
    bundled_files,
    catalog_text,
    discover_skills,
    load_skill_dir,
    parse_skill_md,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "skillmd"
FIXTURE = json.loads((FIXTURES / "skillmd.json").read_text())
REPO = FIXTURES / FIXTURE["sample"]["repo"]


class TestTheFixtureIsTheContract:
    def test_the_layout(self):
        layout = FIXTURE["layout"]
        assert AUTO_SKILLS_DIR == layout["auto_dir"].replace("/", os.sep)
        assert EXPLICIT_KEY == layout["explicit_key"]
        assert list(SKILL_FILE_NAMES) == layout["skill_file_names"]
        assert sorted(SKIPPED_DIRS) == sorted(layout["skipped_dirs"])
        assert list(KNOWN_KEYS) == FIXTURE["frontmatter"]["known_keys"]
        assert CATALOG_PREAMBLE == FIXTURE["catalog"]["preamble"]


@pytest.mark.parametrize("case", FIXTURE["parse_cases"], ids=[c["case"] for c in FIXTURE["parse_cases"]])
def test_the_parser(case):
    parsed = parse_skill_md(case["text"], case["dir"])
    if "skipped" in case:
        assert isinstance(parsed, tuple), parsed
        problem, reason = parsed
        assert problem == case["skipped"]
        if "reason_starts_with" in case:
            assert reason.startswith(case["reason_starts_with"]), reason
        else:
            assert reason == case["reason"]
        return
    assert isinstance(parsed, SkillMd), parsed
    assert parsed.name == case["dir"]
    assert parsed.description == case["description"]
    assert parsed.allowed_tools == case["allowed_tools"]
    assert parsed.metadata == case["metadata"]
    assert parsed.extra == case["extra"]
    assert parsed.warnings == case["warnings"]
    assert parsed.body == case["body"]


def test_allowed_tools_is_a_hint_and_never_a_grant():
    # Nothing in the loader turns the hint into a scope; the skill's tools
    # stay owner-only whatever the file says (test_skillmd_skill.py).
    parsed = parse_skill_md("---\ndescription: d\nallowed-tools: Bash(*) Read Write\n---\n", "x")
    assert isinstance(parsed, SkillMd)
    assert parsed.allowed_tools == ["Bash(*)", "Read", "Write"]
    assert not hasattr(parsed, "scope")


class TestTheSampleRepository:
    """The fixture repository read as `agent_skills: [skills]`."""

    def test_the_skills_and_the_skipped_one(self):
        sample = FIXTURE["sample"]
        found = discover_skills(str(REPO), sample["explicit"])
        assert found.names() == sample["skills"]
        assert [(s.name, s.problem, s.reason) for s in found.skipped] == [
            (s["name"], s["problem"], s["reason"]) for s in sample["skipped"]
        ]
        assert found.warnings == []
        pdf = found.skills[0]
        assert pdf.source == "explicit"
        assert pdf.allowed_tools == sample["pdf"]["allowed_tools"]
        assert pdf.metadata == sample["pdf"]["metadata"]
        assert pdf.license == sample["pdf"]["license"]
        assert pdf.warnings == sample["pdf"]["warnings"]
        assert pdf.body == sample["pdf"]["body"]
        assert pdf.location == os.path.realpath(REPO / "skills" / "pdf" / "SKILL.md")
        assert pdf.directory == os.path.realpath(REPO / "skills" / "pdf")

    def test_the_catalog(self):
        found = discover_skills(str(REPO), FIXTURE["sample"]["explicit"])
        by_name = {s.name: s for s in found.skills}
        expected = FIXTURE["sample"]["catalog"].replace("{pdf_location}", by_name["pdf"].location).replace("{xlsx_location}", by_name["xlsx"].location)
        assert catalog_text(found.skills) == expected

    def test_the_activation_lists_files_without_reading_them(self):
        found = discover_skills(str(REPO), FIXTURE["sample"]["explicit"])
        by_name = {s.name: s for s in found.skills}
        assert bundled_files(by_name["pdf"].directory) == FIXTURE["sample"]["pdf"]["files"]
        expected = FIXTURE["sample"]["activation"].replace("{pdf_dir}", by_name["pdf"].directory)
        text = activation_text(by_name["pdf"])
        assert text == expected
        # The substitution stays text: nothing ran `date`.
        assert "!`date`" in text
        assert FIXTURE["sample"]["activation_xlsx_files"] in activation_text(by_name["xlsx"])

    def test_the_catalog_escapes_xml(self):
        escaped = FIXTURE["catalog"]["escaped"]
        skill = SkillMd(name=escaped["name"], description=escaped["description"], location=escaped["location"], directory="/x/a-b", body="")
        assert catalog_text([skill]) == escaped["text"]

    def test_the_catalog_is_empty_without_skills(self):
        assert catalog_text([]) == FIXTURE["catalog"]["empty"]


class TestDiscovery:
    def _skill(self, folder: Path, name: str, description: str = "Does things.") -> Path:
        directory = folder / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody of {name}.\n")
        return directory

    def test_agents_skills_is_found_on_its_own(self, tmp_path):
        self._skill(tmp_path / ".agents" / "skills", "alpha")
        self._skill(tmp_path / ".agents" / "skills", "beta")
        found = discover_skills(str(tmp_path))
        assert found.names() == ["alpha", "beta"]
        assert all(s.source == "auto" for s in found.skills)

    def test_lower_case_skill_md_counts(self, tmp_path):
        directory = tmp_path / ".agents" / "skills" / "low"
        directory.mkdir(parents=True)
        (directory / "skill.md").write_text("---\ndescription: d\n---\n")
        assert discover_skills(str(tmp_path)).names() == ["low"]

    def test_an_explicit_folder_is_one_skill_or_a_folder_of_skills(self, tmp_path):
        self._skill(tmp_path / "one", "one")
        shared = tmp_path / "shared"
        self._skill(shared, "s1")
        self._skill(shared, "s2")
        found = discover_skills(str(tmp_path), ["one/one", "shared", str(shared / "s1")])
        # `one` once, `s1` once (the second mention is shadowed), `s2` once.
        assert found.names() == ["one", "s1", "s2"]
        assert all(s.source == "explicit" for s in found.skills)
        assert len(found.warnings) == 1 and 's1' in found.warnings[0] and "shadowed" in found.warnings[0]

    def test_an_explicit_entry_wins_over_a_discovered_one(self, tmp_path):
        self._skill(tmp_path / ".agents" / "skills", "dup", "auto one")
        self._skill(tmp_path / "elsewhere", "dup", "explicit one")
        found = discover_skills(str(tmp_path), ["elsewhere/dup"])
        assert found.names() == ["dup"]
        assert found.skills[0].description == "explicit one"
        assert len(found.warnings) == 1 and "shadowed" in found.warnings[0]

    def test_a_missing_or_empty_explicit_entry_is_reported_not_raised(self, tmp_path):
        (tmp_path / "empty").mkdir()
        found = discover_skills(str(tmp_path), ["nowhere", "empty"])
        assert found.names() == []
        assert [s.problem for s in found.skipped] == ["not_found", "not_found"]
        assert found.skipped[0].reason == f"{EXPLICIT_KEY}: nowhere is not a folder"
        assert found.skipped[1].reason == f"{EXPLICIT_KEY}: empty holds no SKILL.md and no folder with one"

    def test_a_malformed_skill_is_reported_and_the_rest_load(self, tmp_path):
        self._skill(tmp_path / ".agents" / "skills", "good")
        broken = tmp_path / ".agents" / "skills" / "broken"
        broken.mkdir()
        (broken / "SKILL.md").write_text("---\nname: broken\n---\nNo description.\n")
        found = discover_skills(str(tmp_path))
        assert found.names() == ["good"]
        assert [(s.name, s.problem) for s in found.skipped] == [("broken", "no_description")]

    def test_git_and_node_modules_are_never_skills_or_bundled_files(self, tmp_path):
        directory = self._skill(tmp_path / ".agents" / "skills", "tidy")
        (directory / ".git").mkdir()
        (directory / ".git" / "config").write_text("x")
        (directory / "node_modules" / "m").mkdir(parents=True)
        (directory / "node_modules" / "m" / "index.js").write_text("x")
        (directory / "scripts").mkdir()
        (directory / "scripts" / "run.sh").write_text("echo hi\n")
        os.symlink("/etc/hosts", directory / "escape")
        for name in (".git", "node_modules"):
            (tmp_path / ".agents" / "skills" / name).mkdir(exist_ok=True)
            (tmp_path / ".agents" / "skills" / name / "SKILL.md").write_text("---\ndescription: d\n---\n")
        found = discover_skills(str(tmp_path))
        assert found.names() == ["tidy"]
        assert bundled_files(str(directory)) == ["scripts/run.sh"]

    def test_load_skill_dir_names_the_folder(self, tmp_path):
        directory = self._skill(tmp_path, "named")
        loaded = load_skill_dir(str(directory))
        assert isinstance(loaded, SkillMd)
        assert loaded.name == "named"
        assert loaded.location == os.path.realpath(directory / "SKILL.md")
        assert isinstance(load_skill_dir(str(tmp_path)), SkippedSkill)
