"""
Global options after the subcommand (the sandbox-default lane, 2026-09-27):
`webagents login --profile local` used to answer "No such option". The
hoisting is pinned against the shared fixture
`tests/fixtures/cli/sandbox_default_global_options.json` that
`typescript/tests/unit/cli/sandbox-default-global-options.test.ts` reads
too, and proved on the real CLI through `runner.invoke(app, ...)`, which
goes through the root group's `main` where the hoisting lives: `config path
--profile <name>` prints the profile's own config path, and `doctor
--no-sandbox` reports the sandbox off for this run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app
from webagents.cli.sandbox_default_argv import HOISTED_OPTIONS, hoist_global_options

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "sandbox_default_global_options.json").read_text())
runner = CliRunner()


def test_the_hoisted_options_are_the_fixtures():
    assert list(HOISTED_OPTIONS) == FIXTURE["hoisted"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[" ".join(c["argv"]) or "(none)" for c in FIXTURE["cases"]])
def test_each_case(case):
    assert hoist_global_options(case["argv"]) == case["hoisted"]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    # The root callback puts the flag and the profile into the process
    # environment; both are SET here (to values that mean "none") so the
    # teardown restores what was there, rather than leaving `sd-probe` behind
    # for the next test (which it did once).
    monkeypatch.setenv("WEBAGENTS_NO_SANDBOX", "0")
    monkeypatch.setenv("WEBAGENTS_PROFILE", "")
    monkeypatch.delenv("WEBAGENTS_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_profile_after_the_subcommand_reaches_the_root(isolated):
    profile = "sd-probe"
    result = runner.invoke(app, ["config", "path", "--profile", profile])
    assert result.exit_code == 0, result.output
    assert FIXTURE["real_cli"]["config_path_contains"].replace("{profile}", profile) in result.output


def test_no_sandbox_after_the_subcommand_turns_the_sandbox_off_for_the_run(isolated):
    (isolated / "AGENT.md").write_text("---\nname: a\nskills:\n  - shell\n---\nBody\n")
    # `--json` stays a root option typed before the command; only `--profile` and `--no-sandbox` are lifted.
    result = runner.invoke(app, ["--json", "doctor", "--no-sandbox"])
    document = json.loads(result.stdout)

    def checks_in(node):
        """Every check object in the document, wherever the envelope keeps them."""
        if isinstance(node, dict):
            if node.get("name") == "sandbox" and "detail" in node:
                yield node
            for value in node.values():
                yield from checks_in(value)
        elif isinstance(node, list):
            for item in node:
                yield from checks_in(item)

    sandbox = next(checks_in(document))
    assert sandbox["detail"] == FIXTURE["real_cli"]["doctor_sandbox_detail"]
    assert os.environ.get("WEBAGENTS_NO_SANDBOX") == "1"
