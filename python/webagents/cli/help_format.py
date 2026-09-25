"""
Help and usage errors in the TypeScript CLI's words and layout (2026-09-24).

The TypeScript CLI is built on commander 14, and its `-h` pages and usage
errors are what people see first. Typer drew boxed panels and click wrote
"Error: No such command", so the same mistake read differently in each CLI.
This is commander's formatter and messages, ported line for line
(`commander/lib/help.js` `formatHelp`, `formatItem`, `boxWrap`;
`lib/command.js` `unknownCommand`, `unknownOption`, `missingArgument`,
`optionMissingArgument`, `_excessArguments`; `lib/suggestSimilar.js`), over the
click command tree:

  * `Usage:`, the description, then `Arguments:` (only when one is described),
    `Options:` and `Commands:`, every term padded to one column and every
    description wrapped at the terminal's width (80 when it is not a terminal);
  * `-h, --help` on every command and `help [command]` in every group;
  * errors on stderr, exit 1: `error: unknown command 'x'`,
    `error: unknown option '--x'`, with `(Did you mean ...?)` as commander
    suggests it;
  * a group run without a command prints its help to stderr and exits 1.

The command surface itself is held to the TypeScript one by
`tests/cli/test_cli_parity.py`; this makes the two print it the same way.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from typing import Any, List, Optional, Sequence

import click
from typer.core import TyperCommand, TyperGroup

MIN_WIDTH_TO_WRAP = 40
HELP_DESCRIPTION = "display help for command"


# -- commander's suggestSimilar -------------------------------------------------------------

MAX_DISTANCE = 3


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (commander's `editDistance`)."""
    if abs(len(a) - len(b)) > MAX_DISTANCE:
        return max(len(a), len(b))
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def suggest_similar(word: str, candidates: Sequence[str]) -> str:
    """`\\n(Did you mean x?)`, or `''`: commander's `suggestSimilar`."""
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return ""
    searching_options = word.startswith("--")
    if searching_options:
        word = word[2:]
        candidates = [c[2:] for c in candidates]
    similar: List[str] = []
    best = MAX_DISTANCE
    for candidate in candidates:
        if len(candidate) <= 1:
            continue
        distance = _edit_distance(word, candidate)
        length = max(len(word), len(candidate))
        if (length - distance) / length > 0.4:
            if distance < best:
                best = distance
                similar = [candidate]
            elif distance == best:
                similar.append(candidate)
    similar.sort()
    if searching_options:
        similar = [f"--{c}" for c in similar]
    if len(similar) > 1:
        return f"\n(Did you mean one of {', '.join(similar)}?)"
    if len(similar) == 1:
        return f"\n(Did you mean {similar[0]}?)"
    return ""


# -- commander's formatHelp ----------------------------------------------------------------


def help_width(stream: Any = None) -> int:
    stream = stream or sys.stdout
    try:
        if stream.isatty():
            return shutil.get_terminal_size((80, 24)).columns
    except (AttributeError, ValueError):
        pass
    return 80


def box_wrap(text: str, width: int) -> str:
    if width < MIN_WIDTH_TO_WRAP:
        return text
    lines: List[str] = []
    for raw in re.split(r"\r\n|\n", text):
        chunks = re.findall(r"\s*\S+", raw)
        if not chunks:
            lines.append("")
            continue
        current = [chunks[0]]
        size = len(chunks[0])
        for chunk in chunks[1:]:
            if size + len(chunk) <= width:
                current.append(chunk)
                size += len(chunk)
                continue
            lines.append("".join(current))
            chunk = chunk.lstrip()
            current, size = [chunk], len(chunk)
        lines.append("".join(current))
    return "\n".join(lines)


def format_item(term: str, term_width: int, description: str, width: int) -> str:
    if not description:
        return "  " + term
    remaining = width - term_width - 2 - 2
    if remaining < MIN_WIDTH_TO_WRAP or re.search(r"\n[^\S\r\n]", description):
        formatted = description
    else:
        formatted = box_wrap(description, remaining).replace("\n", "\n" + " " * (term_width + 2))
    return "  " + term.ljust(term_width) + "  " + formatted.replace("\n", "\n  ")


def _description(command: click.Command) -> str:
    """The first paragraph of the command's help, on one line."""
    text = (command.help or "").strip()
    return " ".join(text.split("\n\n", 1)[0].split())


def _visible_options(command: click.Command, ctx: click.Context) -> List[click.Option]:
    return [p for p in command.get_params(ctx) if isinstance(p, click.Option) and not p.hidden]


def _is_help(option: click.Option) -> bool:
    return "--help" in option.opts


def option_term(option: click.Option) -> str:
    flags = sorted(option.opts + option.secondary_opts, key=lambda f: (f.startswith("--"), f))
    term = ", ".join(flags)
    if not option.is_flag and not option.count:
        placeholder = option.metavar or f"<{option.name}>"
        term += f" {placeholder}"
    return term


def _option_description(option: click.Option) -> str:
    if _is_help(option):
        return HELP_DESCRIPTION
    text = option.help or ""
    default = option.default
    if not option.is_flag and default is not None and not callable(default):
        text = f"{text} (default: {json.dumps(str(default))})".strip()
    return text


def _arguments(command: click.Command) -> List[click.Argument]:
    return [p for p in command.params if isinstance(p, click.Argument)]


def _argument_name(argument: click.Argument) -> str:
    name = (argument.name or "") + ("..." if argument.nargs == -1 else "")
    return f"<{name}>" if argument.required else f"[{name}]"


def _argument_description(argument: click.Argument) -> str:
    text = getattr(argument, "help", None) or ""
    if argument.default is not None and not callable(argument.default):
        text = f"{text} (default: {json.dumps(str(argument.default))})".strip()
    return text


def subcommand_term(command: click.Command, ctx: click.Context) -> str:
    options = [o for o in _visible_options(command, ctx) if not _is_help(o)]
    args = " ".join(_argument_name(a) for a in _arguments(command))
    return command.name + (" [options]" if options else "") + (f" {args}" if args else "")


def command_usage(command: click.Command, ctx: click.Context) -> str:
    names = []
    walk: Optional[click.Context] = ctx
    while walk is not None:
        names.insert(0, walk.info_name or walk.command.name or "")
        walk = walk.parent
    usage = ["[options]"]
    if isinstance(command, click.Group):
        usage.append("[command]")
    usage += [_argument_name(a) for a in _arguments(command)]
    return " ".join(names + usage)


def format_help(command: click.Command, ctx: click.Context, width: Optional[int] = None) -> str:
    width = width or help_width()
    options = _visible_options(command, ctx)
    arguments = _arguments(command)
    shown_arguments = arguments if any(getattr(a, "help", None) for a in arguments) else []
    commands: List[click.Command] = []
    if isinstance(command, click.Group):
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                commands.append(sub)

    terms = [option_term(o) for o in options]
    terms += [a.name or "" for a in shown_arguments]
    terms += [subcommand_term(c, click.Context(c, parent=ctx, info_name=c.name)) for c in commands]
    term_width = max((len(t) for t in terms), default=0)

    out = [f"Usage: {command_usage(command, ctx)}", ""]
    description = _description(command)
    if description:
        out += [box_wrap(description, width), ""]
    if shown_arguments:
        out.append("Arguments:")
        out += [format_item(a.name or "", term_width, _argument_description(a), width) for a in shown_arguments]
        out.append("")
    if options:
        out.append("Options:")
        out += [format_item(option_term(o), term_width, _option_description(o), width) for o in options]
        out.append("")
    if commands:
        out.append("Commands:")
        for sub in commands:
            sub_ctx = click.Context(sub, parent=ctx, info_name=sub.name)
            out.append(format_item(subcommand_term(sub, sub_ctx), term_width, _description(sub), width))
        out.append("")
    return "\n".join(out)


# -- the classes Typer builds with ------------------------------------------------------------


class CommanderCommand(TyperCommand):
    def get_help(self, ctx: click.Context) -> str:
        return format_help(self, ctx).rstrip("\n")

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(format_help(self, ctx))


def _help_command(group: click.Group) -> click.Command:
    """`help [command]`, which commander gives every command that has subcommands."""

    @click.pass_context
    def show(ctx: click.Context, command: Optional[str]) -> None:
        parent = ctx.parent or ctx
        if not command:
            click.echo(group.get_help(parent))
            return
        sub = group.get_command(parent, command)
        if sub is None:
            raise click.UsageError(f"No such command '{command}'.", ctx=parent)
        click.echo(sub.get_help(click.Context(sub, parent=parent, info_name=command)))

    cmd = CommanderCommand(
        "help",
        callback=show,
        params=[click.Argument(["command"], required=False)],
        help=HELP_DESCRIPTION,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    return cmd


class CommanderGroup(TyperGroup):
    #: Command names in the TypeScript CLI's order; anything else follows.
    order: Sequence[str] = ()
    #: The command the top level's unknown options belong to, if any.
    default_command: Optional[str] = None

    def list_commands(self, ctx: click.Context) -> List[str]:
        position = {name: i for i, name in enumerate(self.commands)}
        rank = {name: i for i, name in enumerate(self.order)}
        names = sorted(self.commands, key=lambda n: (rank.get(n, len(rank)), position[n]))
        return names + ["help"]

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        if cmd_name == "help" and "help" not in self.commands:
            return _help_command(self)
        return super().get_command(ctx, cmd_name)

    def get_help(self, ctx: click.Context) -> str:
        return format_help(self, ctx).rstrip("\n")

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(format_help(self, ctx))

    def main(self, args: Optional[Sequence[str]] = None, prog_name: Optional[str] = None, complete_var: Optional[str] = None, standalone_mode: bool = True, **extra: Any) -> Any:
        """Usage errors in commander's words, on stderr, exit 1."""
        try:
            result = super().main(args=args, prog_name=prog_name, complete_var=complete_var, standalone_mode=False, **extra)
        except click.exceptions.NoArgsIsHelpError as error:
            sys.stderr.write(error.ctx.get_help() + "\n")
            return self._finish(1, standalone_mode)
        except click.exceptions.Exit as done:
            return self._finish(done.exit_code, standalone_mode)
        except click.ClickException as error:
            argv = list(args) if args is not None else sys.argv[1:]
            sys.stderr.write(commander_message(error, self, argv) + "\n")
            return self._finish(1, standalone_mode)
        except click.exceptions.Abort:
            sys.stderr.write("\n")
            return self._finish(130, standalone_mode)
        return self._finish(result if isinstance(result, int) else 0, standalone_mode)

    @staticmethod
    def _finish(code: int, standalone_mode: bool) -> int:
        if standalone_mode:
            sys.exit(code)
        return code


def _long_flags(command: click.Command, ctx: click.Context) -> List[str]:
    flags: List[str] = []
    for option in _visible_options(command, ctx):
        flags += [f for f in option.opts + option.secondary_opts if f.startswith("--")]
    return flags


def _command_for(root: click.Group, argv: Sequence[str]) -> click.Context:
    """The context of the command `argv` names: click's parser raises some
    errors (an option's missing value) before any context is attached."""
    ctx = click.Context(root, info_name="webagents")
    command: click.Command = root
    for token in argv:
        if token.startswith("-"):
            continue
        if not isinstance(command, click.Group):
            break
        sub = command.get_command(ctx, token)
        if sub is None:
            break
        command = sub
        ctx = click.Context(sub, parent=ctx, info_name=token)
    return ctx


def commander_message(error: click.ClickException, root: Optional[click.Group] = None, argv: Sequence[str] = ()) -> str:
    """A click usage error as commander words it."""
    ctx: Optional[click.Context] = getattr(error, "ctx", None)
    if ctx is None and root is not None:
        ctx = _command_for(root, argv)
    message = error.format_message()
    if isinstance(error, click.NoSuchOption):
        flag = error.option_name
        suggestion = ""
        if flag.startswith("--") and ctx is not None:
            candidates = _long_flags(ctx.command, ctx)
            default = getattr(ctx.command, "default_command", None)
            if ctx.parent is None and default:
                # At the top, commander hands an option it does not know to the default command.
                chat = ctx.command.get_command(ctx, default)
                if chat is not None:
                    candidates = _long_flags(chat, click.Context(chat, parent=ctx, info_name=default))
            suggestion = suggest_similar(flag, candidates)
        return f"error: unknown option '{flag}'{suggestion}"
    if isinstance(error, click.MissingParameter) and isinstance(error.param, click.Argument):
        return f"error: missing required argument '{error.param.name}'"
    if isinstance(error, click.BadOptionUsage) and "requires an argument" in message and ctx is not None:
        option = next((p for p in ctx.command.params if isinstance(p, click.Option) and error.option_name in p.opts + p.secondary_opts), None)
        if option is not None:
            return f"error: option '{option_term(option)}' argument missing"
    if isinstance(error, click.BadParameter) and isinstance(error.param, click.Option):
        value = re.search(r"'([^']*)' is not", message)
        shown = value.group(1) if value else ""
        return f"error: option '{option_term(error.param)}' argument '{shown}' is invalid."
    missing = re.match(r"No such command '(.+)'\.", message)
    if missing and ctx is not None:
        name = missing.group(1)
        names = [n for n in ctx.command.list_commands(ctx)] if isinstance(ctx.command, click.Group) else []
        return f"error: unknown command '{name}'{suggest_similar(name, names)}"
    extra = re.match(r"Got unexpected extra arguments? \((.*)\)", message)
    if extra and ctx is not None:
        try:
            got_extra = len(shlex.split(extra.group(1)))
        except ValueError:
            got_extra = len(extra.group(1).split())
        expected = len(_arguments(ctx.command))
        s = "" if expected == 1 else "s"
        where = f" for '{ctx.info_name}'" if ctx.parent is not None else ""
        return f"error: too many arguments{where}. Expected {expected} argument{s} but got {expected + got_extra}."
    return f"error: {message[0].lower() + message[1:] if message else message}"


def commander_group(order: Sequence[str] = (), default: Optional[str] = None) -> type:
    """A `CommanderGroup` class listing its commands in `order`; `default` is the
    command that takes the top level's options (commander's `isDefault`)."""
    return type("CommanderGroup", (CommanderGroup,), {"order": tuple(order), "default_command": default})
