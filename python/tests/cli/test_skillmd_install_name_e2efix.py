"""
The installed SKILL.md folder is named after the skill's own validated
`name:` (2026-09-26, the new-developer e2e run): `skills add` named it after
the repository, so a skill declared `greeter` in a repository `greeter-skill`
landed as `.agents/skills/greeter-skill` and doctor warned about its own name
on every load. And doctor's fix line names the folders (it stopped at
"named"). Both are pinned by `install_name` and `doctor` in
`tests/fixtures/skillmd/skillmd.json`, which the TypeScript suite reads too
(`tests/unit/cli/skillmd-install-name-e2efix.test.ts`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webagents.agents.skills.local.skillmd.skillmd_install import locate_skills, parse_source, repo_name
from webagents.agents.skills.local.skillmd.skillmd_loader import discover_skills, doctor_report
from webagents.cli.skills_edit import skills_command

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "skillmd" / "skillmd.json").read_text())
CASES = FIXTURE["install_name"]["cases"]
DOCTOR = FIXTURE["doctor"]


@pytest.fixture(autouse=True)
def _no_profile(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)


def _source_folder(tmp_path: Path, repo: str, relative: str, declared) -> Path:
    """A local source folder named `repo` holding one SKILL.md at `relative`, declaring `declared` (or nothing)."""
    root = tmp_path / "src" / repo
    file = root / relative
    file.parent.mkdir(parents=True)
    front = "" if declared is None else f"name: {declared}\n"
    file.write_text(f"---\n{front}description: Greets the person by name.\n---\n\n# Greeter\n\nGreet them.\n")
    return root


def _agent_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "agent"
    folder.mkdir()
    (folder / "AGENT.md").write_text("---\nname: bot\nskills:\n  - openai\n---\n\nBody.\n")
    return folder


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_located_name_is_the_validated_declared_name(case, tmp_path):
    root = _source_folder(tmp_path, case["repo"], case["path"], case["declared"])
    source = parse_source(str(root))
    located, skipped = locate_skills(str(root), source.subpath, repo_name(source))
    assert skipped == []
    assert [name for name, _directory in located] == [case["installed"]]


def test_it_installs_under_that_name_and_doctor_then_has_nothing_to_warn_about(tmp_path):
    root = _source_folder(tmp_path, "greeter-skill", "SKILL.md", "greeter")
    folder = _agent_folder(tmp_path)
    out, err = [], []
    code = skills_command("add", [str(root)], folder=folder, out=out.append, err=err.append, tty=False, yes=True)
    assert code == 0, err
    assert (folder / ".agents" / "skills" / "greeter" / "SKILL.md").exists()
    assert not (folder / ".agents" / "skills" / "greeter-skill").exists()
    assert doctor_report(discover_skills(str(folder))) == {"status": "ok", "detail": DOCTOR["some"].format(count=1, s="", names="greeter"), "fix": None}


def test_doctor_names_the_folders_to_fix(tmp_path):
    folder = _agent_folder(tmp_path)
    skills = folder / ".agents" / "skills"
    (skills / "broken").mkdir(parents=True)
    (skills / "broken" / "SKILL.md").write_text("---\nname: broken\n---\n")
    (skills / "greeter-skill").mkdir()
    (skills / "greeter-skill" / "SKILL.md").write_text("---\nname: greeter\ndescription: Greets.\n---\nGreet.\n")
    report = doctor_report(discover_skills(str(folder)))
    assert report["status"] == "warn"
    assert report["fix"] == DOCTOR["fix"].format(names="broken, greeter-skill")
    assert not report["fix"].endswith("named")
