"""
Help and usage errors in commander's layout and words (2026-09-24,
`webagents/cli/help_format.py`). The suggestion cases are the TypeScript
ones (`typescript/tests/unit/cli/suggest.test.ts`), so both CLIs suggest alike.
"""

from typer.testing import CliRunner

from webagents.cli.help_format import box_wrap, format_item, suggest_similar
from webagents.cli.main import app

COMMANDS = ["chat", "connect", "serve", "daemon", "login", "logout", "whoami", "link", "unlink", "publish", "help"]


def test_suggest_similar_as_commander_does():
    assert suggest_similar("publsh", COMMANDS) == "\n(Did you mean publish?)"
    assert suggest_similar("logn", COMMANDS) == "\n(Did you mean login?)"
    assert suggest_similar("ab", ["abc", "abd"]) == "\n(Did you mean one of abc, abd?)"
    assert suggest_similar("xyzzy", COMMANDS) == ""
    assert suggest_similar("--modle", ["--model", "--agent", "--help"]) == "\n(Did you mean --model?)"


def test_items_pad_and_wrap_as_commander_does():
    assert format_item("-h, --help", 10, "display help for command", 80) == "  -h, --help  display help for command"
    wrapped = format_item("--token <token>", 24, "Use this platform token for this run, instead of the stored sign-in", 80)
    assert wrapped.split("\n") == [
        "  --token <token>           Use this platform token for this run, instead of the",
        "                            stored sign-in",
    ]
    assert box_wrap("short", 10) == "short"


def test_a_mistyped_command_is_named_with_a_suggestion():
    result = CliRunner().invoke(app, ["publsh"])
    assert result.exit_code == 1
    assert result.stderr == "error: unknown command 'publsh'\n(Did you mean publish?)\n"


def test_a_missing_value_names_the_option_as_declared():
    result = CliRunner().invoke(app, ["login", "--url"])
    assert result.exit_code == 1
    assert result.stderr == "error: option '-u, --url <url>' argument missing\n"


def test_a_group_with_no_command_shows_its_help_and_fails():
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 1
    assert result.stderr.startswith("Usage: webagents config [options] [command]\n")


def test_help_names_a_command():
    result = CliRunner().invoke(app, ["help", "serve"])
    assert result.exit_code == 0
    assert result.stdout.startswith("Usage: webagents serve [options] [path]\n\nServe an agent on HTTP\n")
