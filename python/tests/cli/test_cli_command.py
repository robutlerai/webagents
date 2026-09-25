"""A hint names the command as the user must type it here (2026-09-25).

Under `webagents --profile local`, "Run `webagents login`" signed the DEFAULT
profile in and left the local one signed out. Every hint now goes through
`cli_command`, which adds `--profile` while one is active; the cases are
shared with the TypeScript CLI (`tests/fixtures/cli/cli_command.json`).
"""

import json
from pathlib import Path

import pytest

from webagents.cli.config_store import cli_command
from webagents.cli.repl.failures import present_failure

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "cli_command.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: f"{c['profile']}:{c['rest']}")
def test_names_the_command_with_the_active_profile(case, monkeypatch):
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    if case["profile"] is not None:
        monkeypatch.setenv("WEBAGENTS_PROFILE", case["profile"])
    assert cli_command(case["rest"]) == case["expected"]


def test_the_failure_hints_name_the_profile(monkeypatch):
    hints = FIXTURE["hints"]
    monkeypatch.setenv("WEBAGENTS_PROFILE", hints["profile"])
    proxy = "wss://robutler.example/llm"
    refused = present_failure("does not run models for a CLI sign-in", proxy_url=proxy)
    assert refused.hint == hints["sign_in"]
    broke = present_failure("insufficient credits", proxy_url=proxy)
    assert broke.hint == hints["credits"]


def test_whoami_names_the_profile_in_its_fix(monkeypatch):
    from webagents.cli import account
    import webagents.cli.credentials as credentials

    monkeypatch.setenv("WEBAGENTS_PROFILE", "local")
    monkeypatch.setattr(credentials, "get_token", lambda: None)
    result = account.who_am_i()
    assert result.fix == "Run `webagents --profile local login`."
