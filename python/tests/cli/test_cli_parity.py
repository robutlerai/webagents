"""
The two CLIs take the same commands, arguments and options (2026-09-24).

The TypeScript CLI (`typescript/src/cli/index.ts`) is the reference. This
reads its commander declarations and compares them with this CLI's Typer app,
so a command or flag added, renamed or dropped in one CLI only fails here,
rather than being found by a script that worked against the other one.

Help (`-h/--help`) is left out of the comparison: commander adds it to every
command and Typer to every command it builds, and `test_help_is_dash_h` below
checks the short form separately.
"""

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import click
import typer
from typer.testing import CliRunner

from webagents.cli.main import app

TS_FILE = Path(__file__).resolve().parents[3] / "typescript" / "src" / "cli" / "index.ts"

Spec = Dict[str, Tuple[Set[FrozenSet[str]], List[Tuple[str, bool]]]]


def _flags(declaration: str) -> FrozenSet[str]:
    """`'-m, --model <model>'` -> {'-m', '--model'}."""
    return frozenset(part.strip().split(" ")[0] for part in declaration.split(",") if part.strip())


def _arguments(text: str) -> List[Tuple[str, bool]]:
    """`'<key> <value>'` or `'[key]'` -> [(name, required)]."""
    return [(m.group(2).replace("-", "_"), m.group(1) == "<") for m in re.finditer(r"([<\[])([\w.-]+)[>\]]", text)]


def _ts_spec() -> Tuple[Spec, str]:
    """Every command path with its options and arguments, and the default command."""
    source = TS_FILE.read_text()
    groups: Dict[str, str] = {}
    for m in re.finditer(r"const (\w+) = program\.command\('([\w-]+)'\)", source):
        groups[m.group(1)] = m.group(2)

    spec: Spec = {"": (set(), [])}
    default = ""
    current = ""
    # One pass over the calls, in order: `.command(...)` opens a command, and
    # every `.option`/`.argument` after it belongs to it until the next one.
    call = re.compile(
        r"(?P<owner>\b\w+)?\s*\.command\('(?P<command>[^']+)'(?P<options>[^)]*)\)"
        r"|\.option\('(?P<option>[^']+)'"
        r"|\.argument\('(?P<argument>[^']+)'"
        r"|\bprogram\s*\.name\("
    )
    for m in call.finditer(source):
        if m.group("command"):
            name, _, rest = m.group("command").partition(" ")
            owner = m.group("owner")
            parent = groups.get(owner, "") if owner and owner != "program" else ""
            current = f"{parent} {name}".strip()
            spec.setdefault(current, (set(), []))
            spec[current][1].extend(_arguments(rest))
            if "isDefault: true" in (m.group("options") or ""):
                default = current
        elif m.group("option"):
            spec[current][0].add(_flags(m.group("option")))
        elif m.group("argument"):
            spec[current][1].extend(_arguments(m.group("argument")))
        else:
            current = ""
    # `.version(version)` gives commander's `-V, --version`.
    spec[""][0].add(frozenset({"-V", "--version"}))
    # A group is only its subcommands in both CLIs.
    return spec, default


def _py_spec() -> Spec:
    spec: Spec = {}

    def walk(command: click.Command, path: str) -> None:
        options: Set[FrozenSet[str]] = set()
        arguments: List[Tuple[str, bool]] = []
        for param in command.params:
            if isinstance(param, click.Option):
                flags = frozenset(param.opts) | frozenset(param.secondary_opts)
                if flags & {"-h", "--help"}:
                    continue
                options.add(flags)
            elif isinstance(param, click.Argument):
                arguments.append((param.name, param.required))
        spec[path] = (options, arguments)
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                walk(sub, f"{path} {name}".strip())

    walk(typer.main.get_command(app), "")
    return spec


def test_the_same_commands():
    ts, _ = _ts_spec()
    assert sorted(_py_spec()) == sorted(ts)


def test_the_same_options_and_arguments():
    ts, default = _ts_spec()
    py = _py_spec()
    for path, (options, arguments) in ts.items():
        if path == "":
            # The default command's options are the root's in Typer: `webagents -p ...`.
            expected = options | ts[default][0]
            assert py[""][0] == expected, "the root options"
            continue
        assert py[path][0] == options, f"`webagents {path}` options"
        assert py[path][1] == arguments, f"`webagents {path}` arguments"


def test_the_reference_is_read():
    # A parser that found nothing would make both tests above pass vacuously.
    ts, default = _ts_spec()
    assert default == "chat"
    assert {"chat", "serve", "login", "publish", "config set", "secrets get"} <= set(ts)
    assert frozenset({"-p", "--prompt"}) in ts["chat"][0]
    assert ts["config set"][1] == [("key", True), ("value", True)]


def test_help_is_dash_h():
    for args in (["-h"], ["doctor", "-h"], ["config", "set", "-h"]):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, args
        assert "Usage" in result.output


# -- the words: every help page reads the same --------------------------------------------------

_STR = r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\""""


def _js(single, double):
    # `re.findall` gives '' for the quote style that did not match.
    return re.sub(r"\\(.)", r"\1", single or double or "")


def _ts_words():
    """For each command path: its description, and (term, description, default)
    for its options and arguments, as declared in index.ts."""
    source = TS_FILE.read_text()
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"(?m)^\s*//.*$", "", source)
    groups = {m.group(1): m.group(2) for m in re.finditer(r"const (\w+) = program\.command\('([\w-]+)'\)", source)}
    words = {}
    current = None
    call = re.compile(
        r"(?P<owner>\b\w+)?\s*\.command\('(?P<command>[^']+)'[^)]*\)"
        r"|\.(?P<kind>description|option|argument)\((?P<args>(?:" + _STR + r")(?:\s*,\s*(?:" + _STR + r"))*)\s*,?\s*\)"
    )
    for m in call.finditer(source):
        if m.group("command"):
            name = m.group("command").split(" ")[0]
            owner = m.group("owner")
            parent = groups.get(owner, "") if owner and owner != "program" else ""
            current = f"{parent} {name}".strip()
            words.setdefault(current, {"description": "", "options": [], "arguments": []})
            continue
        if current is None:
            continue
        values = [_js(a, b) for a, b in re.findall(_STR, m.group("args"))]
        entry = words[current]
        if m.group("kind") == "description":
            entry["description"] = values[0]
        elif m.group("kind") == "option":
            entry["options"].append((values[0], values[1] if len(values) > 1 else "", values[2] if len(values) > 2 else None))
        else:
            entry["arguments"].append((values[0], values[1] if len(values) > 1 else "", values[2] if len(values) > 2 else None))
    return words


def _py_words():
    from webagents.cli.help_format import _description, _is_help, option_term

    words = {}

    def walk(command, ctx, path):
        options = []
        for param in command.get_params(ctx):
            if isinstance(param, click.Option) and not param.hidden and not _is_help(param):
                default = None if param.is_flag or param.default is None else str(param.default)
                options.append((option_term(param), param.help or "", default))
        arguments = []
        for param in command.params:
            if isinstance(param, click.Argument):
                name = f"<{param.name}>" if param.required else f"[{param.name}]"
                default = None if param.default is None else str(param.default)
                arguments.append((name, getattr(param, "help", None) or "", default))
        words[path] = {"description": _description(command), "options": options, "arguments": arguments}
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                walk(sub, click.Context(sub, parent=ctx, info_name=name), f"{path} {name}".strip())

    root = typer.main.get_command(app)
    walk(root, click.Context(root, info_name="webagents"), "")
    return words


def test_the_same_words():
    ts, py = _ts_words(), _py_words()
    for path, entry in ts.items():
        if path == "":
            continue  # the root's description names the SDK
        assert py[path]["description"] == entry["description"], f"`webagents {path}` description"
        assert py[path]["options"] == entry["options"], f"`webagents {path}` options"
        # Commander declares inline arguments (`get [key]`) with no description; compare the declared ones.
        declared = [a for a in entry["arguments"]]
        if declared:
            assert py[path]["arguments"] == declared, f"`webagents {path}` arguments"
        else:
            assert all(not a[1] for a in py[path]["arguments"]), f"`webagents {path}`: an argument described only in Python"


def test_the_words_are_read():
    ts = _ts_words()
    assert ts["serve"]["arguments"] == [("[path]", "Path to agent config file", ".")]
    assert ("--output-format <format>", "With -p: text, json, stream-json", "text") in ts["chat"]["options"]
    assert ts["link"]["arguments"][0][1] == "The agent's name; defaults to the name in this folder's agent file"


def test_commands_are_listed_in_the_same_order():
    from webagents.cli.main import COMMAND_ORDER

    ts, _ = _ts_spec()
    top = [path for path in ts if path and " " not in path]
    assert list(COMMAND_ORDER) == top
