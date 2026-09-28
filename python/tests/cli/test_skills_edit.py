"""
`webagents skills add` and `skills remove` (2026-09-25), the same in both CLIs.

The cases are `tests/fixtures/cli/skills_edit.json`, which the TypeScript suite
runs too (`typescript/tests/unit/cli/skills-edit.test.ts`): the editor, byte for
byte, and the command's words, exit codes and choice of file. What is pinned
beside them is this CLI's wiring: the Typer commands reach the editor and exit
with its code.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.skills_edit import SkillListError, SkillsFacts, edit_skill_list, skills_command

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "skills_edit.json").read_text())


@pytest.fixture(autouse=True)
def _no_profile(monkeypatch):
    # The hints name commands as they must be typed, with `--profile` under one.
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    # The shared cases are about keys and sign-in, where both SDKs agree; a
    # provider library missing from this Python is its own refusal (B3,
    # `test_final_sdk_skills_add_library.py`), which the TypeScript CLI,
    # whose providers need no library, never has.
    monkeypatch.setattr("webagents.cli.model_access._client_installed", lambda provider: True)


@pytest.mark.parametrize("case", FIXTURE["edits"], ids=[c["case"] for c in FIXTURE["edits"]])
def test_the_editor(case):
    if "error" in case:
        with pytest.raises(SkillListError) as caught:
            edit_skill_list(case["text"], case["action"], case["names"], case["file"])
        assert caught.value.problem == case["error"]
        return
    edit = edit_skill_list(case["text"], case["action"], case["names"], case["file"])
    assert edit.text == case["result"]
    assert edit.changed == case["changed"]
    assert (edit.added, edit.already, edit.removed, edit.absent, edit.skills) == (
        case["added"], case["already"], case["removed"], case["absent"], case["skills"],
    )


def _layout(folder, files):
    """Write a case's files, making a symbolic link for a `{link}` value (S-290)."""
    import os

    for name, spec in files.items():
        full = folder / name
        full.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(spec, dict):
            os.symlink(spec["link"], str(full))
        else:
            full.write_bytes(spec.encode())


@pytest.mark.parametrize("case", FIXTURE["commands"], ids=[c["case"] for c in FIXTURE["commands"]])
def test_the_command(case, tmp_path):
    _layout(tmp_path, case["files"])
    keys = set(case["facts"]["keys"])
    facts = SkillsFacts(has_key=lambda variable: variable in keys, signed_in=case["facts"]["signed_in"])
    out, err = [], []

    code = skills_command(
        case["args"][0], case["args"][1:], agent=case.get("agent"), folder=tmp_path,
        facts=lambda: facts, out=out.append, err=err.append,
    )

    assert code == case["exit"]
    assert ("\n".join(out).split("\n") if out else []) == case["out"]
    assert ("\n".join(err).split("\n") if err else []) == case["err"]
    for name, text in case["files"].items():
        # A link's own bytes are its target's; the target is checked as its own entry.
        if isinstance(text, dict):
            continue
        assert (tmp_path / name).read_bytes().decode() == case.get("after", {}).get(name, text), name


@pytest.mark.parametrize("case", FIXTURE["plans"], ids=[c["case"] for c in FIXTURE["plans"]])
def test_the_plan(case, tmp_path):
    from webagents.cli.skills_edit import plan_skills

    _layout(tmp_path, case["files"])
    plan = plan_skills(case["args"][0], case["args"][1:], folder=tmp_path)
    expected = case["plan"]
    assert plan.errors == expected["errors"]
    assert plan.add == expected["add"]
    assert plan.remove == expected["remove"]
    assert plan.skills_after == expected["skills_after"]
    assert plan.installed_removals == expected["installed_removals"]
    assert plan.sources == expected["sources"]
    assert plan.changed == expected["changed"]
    assert (plan.file.name if plan.file is not None else None) == expected["file"]


class TestTheCommands:
    """`webagents skills add|remove` run the editor in the working folder."""

    def test_add_and_remove(self, tmp_path, monkeypatch):
        from webagents.cli import skills_edit
        from webagents.cli.main import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(skills_edit, "machine_facts", lambda: SkillsFacts(has_key=lambda v: True, signed_in=True))
        (tmp_path / "AGENT.md").write_text("---\nname: bot\nskills:\n  - openai\n---\n")

        added = CliRunner().invoke(app, ["skills", "add", "shell", "todo"])
        assert added.exit_code == 0, added.output
        assert "Added shell and todo to AGENT.md." in added.output
        removed = CliRunner().invoke(app, ["skills", "remove", "shell", "-a", "bot"])
        assert removed.exit_code == 0, removed.output
        assert (tmp_path / "AGENT.md").read_text() == "---\nname: bot\nskills:\n  - openai\n  - todo\n---\n"

    def test_a_refusal_exits_1(self, tmp_path, monkeypatch):
        from webagents.cli.main import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["skills", "add", "shell"])
        assert result.exit_code == 1
        assert "No agent file in this folder." in result.output

    def test_names_are_required(self, tmp_path, monkeypatch):
        from webagents.cli.main import app

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["skills", "add"])
        assert result.exit_code != 0
