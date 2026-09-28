"""
`edit_front_matter_scalar` (the chat's `/model --save`, interactive-mode spec
3.5, 2026-09-26), against the shared fixture `tests/fixtures/cli/skills_edit.json`
(`scalar_edits`), which the TypeScript suite runs too
(`tests/unit/cli/skills-edit-scalar-interactive2.test.ts`): the exact bytes
after each edit, and the sentence each refusal carries.
"""

import json
from pathlib import Path

import pytest

from webagents.cli.skills_edit import SkillListError, edit_front_matter_scalar

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "skills_edit.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["scalar_edits"], ids=[c["case"] for c in FIXTURE["scalar_edits"]])
def test_edit_front_matter_scalar(case):
    file = str(Path("/some/folder") / case["file"])
    if case.get("error"):
        with pytest.raises(SkillListError) as raised:
            edit_front_matter_scalar(case["before"], case["key"], case["value"], file)
        assert str(raised.value) == case["error"].replace("{file}", case["file"])
        return
    edit = edit_front_matter_scalar(case["before"], case["key"], case["value"], file)
    assert edit.previous == case["previous"]
    if case.get("unchanged"):
        assert edit.changed is False and edit.text == case["before"]
        return
    assert edit.changed is True
    assert edit.text == case["after"]
