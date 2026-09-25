"""
`webagents secrets` - the provider keys on THIS machine.

WHAT THIS IS NOT, and why. Every comparable CLI's `secrets` command pushes
values to the platform (`wrangler secret put`, `fly secrets set`,
`vercel env add`). That is not buildable here today, and the reason is worth
writing down rather than discovering twice: the portal's only agent-secret
surface is per FUNCTION
(`POST /api/agents/{id}/functions/{functionName}/secret`), and it authenticates
with `getSession()` alone. It takes a browser session cookie; it has no bearer
path, so the token this CLI holds cannot use it. Building `secrets set` against
it would ship a command that answers 401 forever. Making it work is a portal
change: that route needs to accept a scoped bearer.

What IS real is the local half, and it is the half people actually fumble.
Provider keys live in shell profiles and `.env` files, where they are
world-readable and easy to leak into a shell history or a screen share. The
CLI already carries a keystore-backed `SecretStore` (macOS Keychain, Linux
Secret Service, Windows Credential Manager, with a 0600 file fallback), and it
already holds the platform token there. This puts provider keys in the same
place and teaches the process to read them.

PRECEDENCE, so this cannot silently shadow something: an existing environment
variable always wins. `secrets` fills in what the environment does not already
set, which means adding a key here can never change the behaviour of a shell
that was already exporting one.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List

import typer

from ..help_format import CommanderCommand, commander_group
from ..config_store import cli_command

app = typer.Typer(help="Keys this CLI stored on this machine", no_args_is_help=True, cls=commander_group())

#: The keystore namespace. `cli` already holds the platform token; provider
#: keys are a different kind of thing with a different lifetime, so they get
#: their own service key rather than sharing one bag.
NAMESPACE = "providers"


def _store(quiet: bool = False):
    """The profile's provider-key store. `quiet` for the chat, which says where
    keys are kept in its own words (`/keys`) rather than as a log warning."""
    from webagents.agents.skills.local.secrets.store import open_secret_store

    from ..config_store import global_dir, profile_name, scoped_namespace

    # Profile-scoped in BOTH places. The keychain is keyed by namespace alone,
    # so scoping only the directory would share one entry between profiles
    # wherever a keystore exists (S-219).
    profile = profile_name()
    return open_secret_store(
        namespace=scoped_namespace(NAMESPACE, profile),
        secrets_dir=str(global_dir(profile) / "secrets"),
        quiet=quiet,
    )


def _known_env_vars() -> Dict[str, str]:
    """The env var each provider is read from, from the one registry.

    Not a second hardcoded list: `skills/core/llm/providers.py` is what
    `models` and `doctor` already use, so a provider added there shows up here
    without anyone remembering to.
    """
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    # `env_vars` is a tuple in the order the skill checks them, and a provider
    # with no credential (a local one) has none.
    mapping: Dict[str, str] = {}
    for provider in LLM_PROVIDERS:
        # Keys only: the Robutler socket's URL is not a secret to list.
        if provider.credential != "api_key":
            continue
        for env_var in provider.env_vars:
            mapping.setdefault(env_var, provider.id)
    return mapping


def _stored_where(backend: str) -> str:
    return "your keychain" if backend == "keystore" else "an owner-only file"


def listing_lines() -> List[str]:
    """`secrets list`, in the TypeScript CLI's words: each stored key, and each
    provider key this shell sets, with where it comes from. A keychain cannot
    be listed, so the names are those this CLI recorded, and it says so."""
    store = _store(quiet=True)
    names, complete = store.list()
    stored = set(names)
    shown = sorted(stored | {v for v in _known_env_vars() if os.environ.get(v)})
    caveat = "An OS keychain cannot be listed, so keys other tools wrote there do not appear."
    if not shown:
        out = ["No keys stored, and none set in this shell.", f"Add one with `{cli_command('secrets set OPENAI_API_KEY')}`."]
        return out + ([caveat] if not complete else [])
    width = max(len(n) for n in shown) + 3

    def where(name: str) -> str:
        if os.environ.get(name):
            return "set in this shell, which wins over the stored one" if name in stored else "set in this shell"
        return "stored in your keychain" if store.keystore else "stored in an owner-only file"

    out = [f"  {name.ljust(width)}{where(name)}" for name in shown]
    return out + (["", caveat] if not complete else [])


@app.command("list", cls=CommanderCommand)
def list_secrets() -> None:
    """Keys stored on this machine, and which this shell sets"""
    for line in listing_lines():
        print(line)


@app.command("set", cls=CommanderCommand)
def set_secret(name: str = typer.Argument(...)) -> None:
    """Store a key, for example OPENAI_API_KEY (asked for with echo off)"""
    # Never from an argument: that lands in shell history and the process list.
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        print(f"{name} does not look like an environment variable name.", file=sys.stderr)
        raise typer.Exit(1)
    import getpass

    try:
        value = getpass.getpass(f"Value for {name}: ").strip()
    except (EOFError, KeyboardInterrupt):
        value = ""
    if not value:
        print("Nothing entered; nothing stored.", file=sys.stderr)
        raise typer.Exit(1)
    backend = _store(quiet=True).set(name, value)
    print(f"Stored {name} ({_stored_where(backend)}).")
    if os.environ.get(name):
        print(f"{name} is also set in this environment, and the environment wins.")


@app.command("unset", cls=CommanderCommand)
def unset_secret(name: str = typer.Argument(...)) -> None:
    """Remove a stored key"""
    if _store(quiet=True).delete(name):
        print(f"Removed {name}.")
    else:
        print(f"{name} was not stored.", file=sys.stderr)
        raise typer.Exit(1)


@app.command("get", cls=CommanderCommand)
def get_secret(
    name: str = typer.Argument(...),
    show: bool = typer.Option(False, "--show", help="Print the value itself, for a script"),
) -> None:
    """Say whether a key is stored, or print it with --show

    It exists because `publish` stores the agent's API key, which the platform
    returns exactly once. Redacted by default, and the bare value on stdout
    under `--show`, for `$(...)` in a script.
    """
    try:
        value = _store(quiet=True).get(name)
    except Exception as error:  # noqa: BLE001 - one line, exit 1
        print(str(error), file=sys.stderr)
        raise typer.Exit(1)
    if value is None:
        print(f"{name} is not stored.", file=sys.stderr)
        if os.environ.get(name):
            print("It IS set in this environment, which takes precedence.", file=sys.stderr)
        raise typer.Exit(1)
    if show:
        sys.stdout.write(f"{value}\n")
        return
    print(f"{name} is stored. Use --show to print it.")


def load_into_environment() -> int:
    """Put stored keys into `os.environ`, WITHOUT overriding what is there.

    Called by the commands that start an agent. Returns how many were added.
    The non-overriding rule is what makes this safe to call unconditionally:
    a shell that exports a key keeps winning, so adding a key here can never
    change the behaviour of a session that already had one.
    """
    try:
        # Quiet: this runs before every chat and run, where a log warning about
        # the file backend is noise; `secrets set` and `secrets backend` say it.
        store = _store(quiet=True)
        names, _ = store.list()
    except Exception:
        return 0

    # The provider variables are tried by NAME as well as through the index:
    # in keystore mode the index is only what this CLI recorded, and a key
    # stored before the index was maintained (or by another tool) would
    # otherwise never load. A `get` for an absent name is cheap and quiet.
    names = sorted(set(names) | set(_known_env_vars()))

    added = 0
    for name in names:
        if os.environ.get(name):
            continue
        value = store.get(name)
        if value:
            os.environ[name] = value
            added += 1
    return added
