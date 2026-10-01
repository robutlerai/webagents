"""
The root help's sections (2026-09-29), against the shared fixture
`tests/fixtures/cli/help_groups.json`, which the TypeScript suite reads too
(`typescript/tests/unit/cli/help-groups.test.ts`): the headings and the
commands under each, in order, every listed command a real one, every visible
command listed once, and the hidden ones still working.
"""

import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from webagents.cli.main import HELP_GROUPS, app

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "help_groups.json").read_text())
runner = CliRunner()


def test_the_sections_are_the_fixtures():
    assert [[heading, list(names)] for heading, names in HELP_GROUPS] == FIXTURE["groups"]


def test_every_visible_command_is_listed_once_and_the_hidden_ones_are_not():
    root = typer.main.get_command(app)
    visible = sorted(name for name, command in root.commands.items() if not command.hidden) + ["help"]
    listed = [name for _heading, names in FIXTURE["groups"] for name in names]
    assert sorted(listed) == sorted(visible)
    assert sorted(name for name, command in root.commands.items() if command.hidden) == sorted(FIXTURE["hidden"])


def test_the_help_draws_the_sections_in_order():
    out = runner.invoke(app, ["--help"]).output
    at = [out.index(f"\n{heading}\n") for heading, _names in FIXTURE["groups"]]
    assert at == sorted(at)
    assert "\nCommands:\n" not in out
    for heading, names in FIXTURE["groups"]:
        section = out[out.index(f"\n{heading}\n"):].split("\n\n")[0]
        assert [line.split()[0] for line in section.splitlines()[1:] if line.startswith("  ") and not line.startswith("   ")] == names
    for hidden in FIXTURE["hidden"]:
        assert f"\n  {hidden} " not in out


def test_the_hidden_commands_still_work():
    assert runner.invoke(app, ["templates", "list"]).exit_code == 0
    assert "Available Templates" in runner.invoke(app, ["init", "--list"]).output
    assert runner.invoke(app, ["connect", "--help"]).exit_code == 0
