"""
The machine-readable output contract.

BORROWED FROM OpenClaw, which gets this right and which webagents did not have
at all. The plan originally proposed `--output-format` on a single command;
that is not enough. For a CLI whose users are themselves building agents, the
machine-readable contract IS the product, and it has to be the same shape on
every command or a script cannot rely on it.

THE CONTRACT, when `--json` is passed:

  * stdout carries EXACTLY ONE JSON document and nothing else. No banners, no
    progress, no tables, no colour.
  * every diagnostic goes to stderr, where a pipeline can ignore or capture it
    separately.
  * a failure still prints one JSON document, with `ok: false` and an `error`
    object, and exits non-zero. A script must never have to parse a traceback
    to find out what happened.

The envelope:

    {"ok": true,  "data": {...}}
    {"ok": false, "error": {"code": "...", "message": "...", "fix": "..."}}

`fix` carries the same next-step text a human would have been shown. An agent
reading this output is exactly the audience that benefits from being told what
to do rather than only what broke.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

import typer
from rich.console import Console

#: Diagnostics go here whenever `--json` is active, so stdout stays clean.
err_console = Console(stderr=True)


def json_enabled(ctx: Optional[typer.Context]) -> bool:
    """Whether this invocation asked for machine-readable output."""
    return bool(ctx and ctx.obj and ctx.obj.get("json"))


def emit(data: Dict[str, Any]) -> None:
    """Print the success envelope. The ONLY thing written to stdout."""
    # `print`, not `console.print`: rich would wrap, style and truncate, any of
    # which corrupts a document meant for a parser.
    print(json.dumps({"ok": True, "data": data}, indent=2, default=str))


def fail(
    code: str,
    message: str,
    fix: str = "",
    exit_code: int = 1,
) -> None:
    """Print the error envelope and exit non-zero.

    Always ONE document, even when the failure is a crash: a script that pipes
    stdout into a parser must not receive a half-written table followed by a
    traceback.
    """
    payload: Dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if fix:
        payload["error"]["fix"] = fix
    print(json.dumps(payload, indent=2))
    raise typer.Exit(exit_code)


def note(message: str) -> None:
    """A human-facing aside that must never pollute stdout."""
    err_console.print(f"[dim]{message}[/dim]")


class JsonOutput:
    """Context manager turning an unexpected exception into one error document.

    Without this, a command that raises halfway through `--json` leaves a
    partial document on stdout and a traceback on stderr, which is the worst of
    both: unparseable and unreadable.
    """

    def __init__(self, enabled: bool, code: str = "internal_error") -> None:
        self.enabled = enabled
        self.code = code

    def __enter__(self) -> "JsonOutput":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled or exc is None or isinstance(exc, (typer.Exit, typer.Abort)):
            return False
        print(json.dumps({"ok": False, "error": {"code": self.code, "message": str(exc)}}, indent=2))
        # Suppress the traceback: it went to stdout's sibling already, and the
        # document above is the contract.
        err_console.print(f"[red]{exc_type.__name__}: {exc}[/red]")
        sys.exit(1)
