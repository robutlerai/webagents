"""
`webagents secrets set|list|remove|get` (S-292, 2026-09-26), pinned by
`tests/fixtures/cli/secrets.json`, which the TypeScript suite runs too
(`tests/unit/cli/secrets-cli-mcpsecrets.test.ts`): the help words, the name
grammar (the `${secret:NAME}` grammar), the sentences, and a round trip
through the keystore's FILE fallback in a scratch HOME, with the value piped
in, never given as an argument and never printed back except by `get --show`.
Also: the stored provider keys are the only names that reach `os.environ`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

from webagents.cli.main import app

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "secrets.json").read_text())
runner = CliRunner()


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    # Every key and token another test may have left in os.environ (the chat's
    # `/keys set` and `load_into_environment` write there) goes first: `secrets
    # list` reports keys "set in this shell", and a leftover
    # GOOGLE_GEMINI_API_KEY broke the empty-list step in the full suite
    # (2026-09-26). A fixed list of names missed it.
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) or var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _words():
    from webagents.cli.help_format import _description

    root = typer.main.get_command(app)
    ctx = click.Context(root, info_name="webagents")
    group = root.get_command(ctx, "secrets")
    group_ctx = click.Context(group, parent=ctx, info_name="secrets")
    words = {"group": _description(group)}
    for name in ("list", "set", "remove", "unset", "get"):
        command = group.get_command(group_ctx, name)
        words[name] = _description(command)
        if name == "get":
            show = next(p for p in command.params if isinstance(p, click.Option) and "--show" in p.opts)
            words["get_show"] = show.help
    words["unset_hidden"] = group.get_command(group_ctx, "unset").hidden
    return words


def test_the_help_words_match_the_fixture():
    words = _words()
    assert words.pop("unset_hidden") is True
    assert words == FIXTURE["help"]


def test_unset_is_hidden_from_the_help_and_remove_is_shown():
    result = runner.invoke(app, ["secrets", "-h"])
    assert result.exit_code == 0
    assert "remove" in result.output and "unset" not in result.output


def test_the_name_grammar_is_the_reference_grammar():
    from webagents.agents.skills.local.secrets.references import REFERENCE_NAME

    assert REFERENCE_NAME.pattern == FIXTURE["name_pattern"]
    for name in FIXTURE["names"]["accepted"]:
        assert REFERENCE_NAME.match(name), name
    for name in FIXTURE["names"]["refused"]:
        assert not REFERENCE_NAME.match(name), name


def test_set_never_takes_the_value_as_an_argument():
    import inspect

    from webagents.cli.commands.secrets import set_secret

    assert "value" not in inspect.signature(set_secret).parameters
    result = runner.invoke(app, ["secrets", "set", "GITHUB_TOKEN", "the-value"])
    assert result.exit_code != 0


def test_the_round_trip_through_the_file_fallback(scratch):
    value = FIXTURE["round_trip"]["value"]
    for step in FIXTURE["round_trip"]["steps"]:
        result = runner.invoke(app, step["argv"], input=step.get("stdin"))
        assert result.exit_code == step["exit"], f"{step['argv']}: {result.output}"
        if "stdout" in step:
            assert result.stdout.rstrip("\n").split("\n") == step["stdout"], step["argv"]
        if "stderr" in step:
            for line in step["stderr"]:
                assert line in result.output, step["argv"]
        # Only `get --show` ever prints the value.
        if step["argv"][-1] != "--show":
            assert value not in result.output, step["argv"]
    stored_file = scratch / "home" / FIXTURE["round_trip"]["file"]
    assert stored_file.exists()
    assert oct(stored_file.stat().st_mode & 0o777) == "0o600"
    assert json.loads(stored_file.read_text()) == {}


def test_set_writes_the_owner_only_file_and_list_names_it_without_the_value(scratch):
    value = FIXTURE["round_trip"]["value"]
    assert runner.invoke(app, ["secrets", "set", "GITHUB_TOKEN"], input=f"{value}\n").exit_code == 0
    stored_file = scratch / "home" / FIXTURE["round_trip"]["file"]
    assert json.loads(stored_file.read_text()) == {"GITHUB_TOKEN": value}
    listing = runner.invoke(app, ["secrets", "list"])
    assert "GITHUB_TOKEN" in listing.output and value not in listing.output


def test_a_refused_name_stores_nothing(scratch):
    for name in FIXTURE["names"]["refused"]:
        if not name:
            continue
        result = runner.invoke(app, ["secrets", "set", name], input="x\n")
        assert result.exit_code == 1
        assert FIXTURE["lines"]["bad_name"].replace("{name}", name) in result.output
    assert not (scratch / "home" / FIXTURE["round_trip"]["file"]).exists()


def test_the_environment_wins_over_a_stored_value_and_says_so(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from-the-shell")
    result = runner.invoke(app, ["secrets", "set", "GITHUB_TOKEN"], input="stored-dummy\n")
    assert result.exit_code == 0
    assert FIXTURE["lines"]["environment_wins"].replace("{name}", "GITHUB_TOKEN") in result.output
    listing = runner.invoke(app, ["secrets", "list"]).output
    assert FIXTURE["lines"]["list_where"]["shell_over_stored"] in listing


def test_only_provider_keys_reach_the_environment_at_start(monkeypatch):
    """A secret stored for an MCP server is read by the skill at connect time
    and must not become a variable the whole agent process can read."""
    from webagents.cli.commands.secrets import _store, load_into_environment

    store = _store(quiet=True)
    store.set("GITHUB_TOKEN", "mcp-secret-dummy")
    store.set("OPENAI_API_KEY", "provider-key-dummy")
    assert load_into_environment() == 1
    assert os.environ.get("OPENAI_API_KEY") == "provider-key-dummy"
    assert "GITHUB_TOKEN" not in os.environ
