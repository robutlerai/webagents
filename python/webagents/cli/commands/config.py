"""
`webagents config`: get, set, unset, validate, path (2026-09-24).

The TypeScript CLI's five commands, with its words and exit codes
(`typescript/src/cli/index.ts`). Both CLIs read and write the same files
through their `ConfigStore`, so they must also share its rules: an unknown key
is refused rather than stored and then ignored, and a value is typed by the
default its key already has, so `config set daemon.port 8821` stores a number.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Optional

import typer

from ..help_format import CommanderCommand, commander_group

app = typer.Typer(help="Manage configuration", no_args_is_help=True, cls=commander_group())


def _store() -> Any:
    """The profile's store: `--profile` is in the environment by the time a command runs."""
    from ..config_store import ConfigStore

    return ConfigStore()


def _refuse_unknown_key(key: str) -> None:
    from ..config_store import DEFAULTS

    if key not in DEFAULTS:
        print(f"Unknown config key: {key}", file=sys.stderr)
        print(f"Known keys: {', '.join(sorted(DEFAULTS))}", file=sys.stderr)
        raise typer.Exit(1)


def _as_text(value: Any) -> str:
    """A value as the TypeScript CLI prints it (`String(value)`)."""
    if value is None:
        return "(not set)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _coerce(key: str, raw: str) -> Any:
    """Typed by the default the key already has; `null`-defaulted keys stay text."""
    from ..config_store import DEFAULTS

    current = DEFAULTS[key]
    if isinstance(current, bool):
        if raw.lower() in ("true", "1", "yes", "on"):
            return True
        if raw.lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f'{key} expects true or false, got "{raw}"')
    if isinstance(current, (int, float)):
        try:
            number = float(raw)
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            raise ValueError(f'{key} expects a number, got "{raw}"')
        return int(number) if number.is_integer() else number
    return raw


@app.command("get", cls=CommanderCommand)
def get(key: Optional[str] = typer.Argument(None)) -> None:
    """Get a configuration value (the effective one, after every layer)"""
    from ..config_store import DEFAULTS

    store = _store()
    if key:
        _refuse_unknown_key(key)
        print(_as_text(store.get(key)))
        return
    print(json.dumps({k: store.get(k) for k in sorted(DEFAULTS)}, indent=2))


@app.command("set", cls=CommanderCommand)
def set_config(
    key: str = typer.Argument(...),
    value: str = typer.Argument(...),
    project: bool = typer.Option(False, "--project", help="Write to ./.webagents/config.json instead of the global file"),
) -> None:
    """Set a configuration value"""
    _refuse_unknown_key(key)
    try:
        typed = _coerce(key, value)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise typer.Exit(1)
    written = _store().set(key, typed, scope="project" if project else "global")
    print(f"{key} = {json.dumps(typed)} in {written}")


@app.command("unset", cls=CommanderCommand)
def unset(
    key: str = typer.Argument(...),
    project: bool = typer.Option(False, "--project", help="Remove from ./.webagents/config.json instead of the global file"),
) -> None:
    """Remove a configuration value"""
    _refuse_unknown_key(key)
    removed = _store().unset(key, scope="project" if project else "global")
    print(f"Removed {key}" if removed else f"{key} was not set there")


@app.command("validate", cls=CommanderCommand)
def validate() -> None:
    """Check the config files for unknown keys and bad values"""
    problems = _store().validate()
    if not problems:
        print("Config is valid.")
        return
    for problem in problems:
        print(problem, file=sys.stderr)
    raise typer.Exit(1)


@app.command("path", cls=CommanderCommand)
def path_cmd() -> None:
    """Show config file paths"""
    store = _store()
    print(f"global:  {store.global_path}")
    print(f"project: {store.project_path}")
