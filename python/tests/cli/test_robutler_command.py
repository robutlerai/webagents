"""The `robutler` command, the same in both CLIs (2026-09-25).

Its help, and the `webagents` command line its options become: the cases are
`tests/fixtures/cli/robutler.json`, which the TypeScript command runs too
(`typescript/tests/unit/cli/robutler-args.test.ts`). It used to start the chat
whatever it was given, `--help` included.
"""

import json
import sys
from pathlib import Path

import pytest

from webagents import robutler_entry
from webagents.robutler_entry import USAGE, robutler_command

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "robutler.json").read_text())


def test_prints_the_shared_help():
    assert USAGE == FIXTURE["usage"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: " ".join(c["argv"]) or "(none)")
def test_the_options_mean_the_same(case):
    command = robutler_command(case["argv"])
    if case.get("help"):
        assert command == ("help",)
    elif case.get("error"):
        assert command == ("error", case["error"])
    else:
        assert command == ("run", case["run"])


def test_help_prints_and_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["robutler", "--help"])
    with pytest.raises(SystemExit) as done:
        robutler_entry.main()
    assert done.value.code == 0
    assert capsys.readouterr().out == FIXTURE["usage"] + "\n"


def test_an_unknown_option_is_refused_with_the_help_and_exit_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["robutler", "--nope"])
    with pytest.raises(SystemExit) as done:
        robutler_entry.main()
    assert done.value.code == 2
    out = capsys.readouterr()
    assert out.err == "Unknown option: --nope\n\n"
    assert out.out == FIXTURE["usage"] + "\n"


def test_a_run_goes_through_the_main_cli(monkeypatch):
    seen = {}

    def fake_cli():
        seen["argv"] = list(sys.argv)

    import webagents.cli.main as main_module

    monkeypatch.setattr(main_module, "cli", fake_cli)
    monkeypatch.setattr(sys, "argv", ["robutler", "-p", "hello", "--json"])
    robutler_entry.main()
    assert seen["argv"] == ["webagents", "-a", "robutler", "-p", "hello", "--output-format", "json"]
