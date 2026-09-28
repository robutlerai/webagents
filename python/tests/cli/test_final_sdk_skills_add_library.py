"""
`/skills add google` (and `webagents skills add google`) on a Python without
the Google SDK (B3, 2026-09-28).

It wrote `google` into the agent file and every start then said "Skill google
failed to load", while the file kept naming it and `/model` kept refusing
Google's models. The add is now refused before the file changes, with the
install command for the Python that runs webagents. Python only: the
TypeScript providers call their APIs over HTTP and need no library.
"""

import sys
from pathlib import Path

import pytest

from webagents.cli.skills_edit import SkillsFacts, missing_client_line, plan_skills, skills_command

AGENT = "---\nname: bot\nskills:\n  - filesystem\n---\nHi.\n"
LINE = (
    f"google needs the google-genai library, which this Python does not have. "
    f"Install it with `{sys.executable} -m pip install 'webagents[llm]'`, then add google again."
)


@pytest.fixture(autouse=True)
def _no_google_library(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    monkeypatch.setattr(
        "webagents.cli.model_access._client_installed", lambda provider: provider.id != "google"
    )


def test_the_line_names_the_install_command():
    assert missing_client_line("google") == LINE
    assert missing_client_line("gemini") == LINE.replace("google needs", "gemini needs").replace("add google again", "add gemini again")
    assert missing_client_line("openai") is None
    assert missing_client_line("filesystem") is None


def test_the_command_refuses_before_the_file_changes(tmp_path):
    (tmp_path / "AGENT.md").write_text(AGENT)
    out, err = [], []
    code = skills_command(
        "add", ["google"], agent=None, folder=tmp_path,
        facts=lambda: SkillsFacts(has_key=lambda variable: False, signed_in=True), out=out.append, err=err.append,
    )
    assert code == 1 and err == [LINE] and out == []
    assert (tmp_path / "AGENT.md").read_text() == AGENT


def test_the_chats_plan_refuses_it_too(tmp_path):
    (tmp_path / "AGENT.md").write_text(AGENT)
    plan = plan_skills("add", ["google"], folder=tmp_path, file=Path(tmp_path / "AGENT.md"))
    assert plan.errors == [LINE] and not plan.changed
