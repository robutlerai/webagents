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

SINCE S-292 (2026-09-26) THE SAME STORE HOLDS MCP SECRETS: an agent file's MCP
server names one as `${secret:NAME}` in its `env`, `headers` or `url`, and the
MCP skill reads it here at connect time. Two consequences. `set` accepts the
`${secret:NAME}` grammar (`references.REFERENCE_NAME`), so every reference a
file can write is one this command can store, and takes the value with echo off
at a terminal or from a pipe in a script, never from an argument. And
`load_into_environment` puts ONLY the provider variables into `os.environ`,
never every stored name: a secret meant for one MCP server must not become a
variable the whole agent process, its shell tool included, can read. `remove`
was `unset`; the old name still works, hidden. The words are pinned by
`tests/fixtures/cli/secrets.json`.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import typer

from ..help_format import CommanderCommand, commander_group
from ..config_store import cli_command

app = typer.Typer(help="Keys and secrets this CLI keeps on this machine", no_args_is_help=True, cls=commander_group())

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


def listing_rows() -> Dict[str, object]:
    """What `secrets list` knows, as data (the `--json` document, pinned by
    `tests/fixtures/cli/secrets.json` `list_json`, 2026-09-26): each stored
    key and each provider key this shell sets, sorted by name, whether it is
    stored and where, whether the shell sets it (the shell wins), and whether
    the listing is complete (a keychain cannot be listed)."""
    store = _store(quiet=True)
    names, complete = store.list()
    stored = set(names)
    where = "keychain" if store.keystore else "file"
    shown = sorted(stored | {v for v in _known_env_vars() if os.environ.get(v)})
    keys = [
        {"name": name, "stored": name in stored, "where": where if name in stored else None, "set_in_shell": bool(os.environ.get(name))}
        for name in shown
    ]
    return {"keys": keys, "complete": bool(complete)}


def listing_lines() -> List[str]:
    """`secrets list`, in the TypeScript CLI's words: each stored key, and each
    provider key this shell sets, with where it comes from. A keychain cannot
    be listed, so the names are those this CLI recorded, and it says so."""
    listing = listing_rows()
    keys: List[Dict[str, object]] = listing["keys"]  # type: ignore[assignment]
    complete = bool(listing["complete"])
    caveat = "An OS keychain cannot be listed, so keys other tools wrote there do not appear."
    if not keys:
        out = ["No keys stored, and none set in this shell.", f"Add one with `{cli_command('secrets set OPENAI_API_KEY')}`."]
        other = _other_runtime_keys_line()
        return out + ([other] if other else []) + ([caveat] if not complete else [])
    width = max(len(str(k["name"])) for k in keys) + 3

    def where(key: Dict[str, object]) -> str:
        if key["set_in_shell"]:
            return "set in this shell, which wins over the stored one" if key["stored"] else "set in this shell"
        return "stored in your keychain" if key["where"] == "keychain" else "stored in an owner-only file"

    out = [f"  {str(key['name']).ljust(width)}{where(key)}" for key in keys]
    return out + (["", caveat] if not complete else [])


@app.command("list", cls=CommanderCommand)
def list_secrets(ctx: typer.Context) -> None:
    """Keys stored on this machine, and which this shell sets"""
    from ..output import emit, json_enabled

    # `--json` was ignored here (2026-09-26, the e2e run): one document now.
    if json_enabled(ctx):
        emit(listing_rows())
        return
    for line in listing_lines():
        print(line)


def _valid_name(name: str) -> bool:
    """The `${secret:NAME}` grammar (`references.REFERENCE_NAME`), imported
    late so the CLI's start does not load the skills package."""
    from webagents.agents.skills.local.secrets.references import REFERENCE_NAME

    return bool(REFERENCE_NAME.match(name))


def _read_secret(prompt: str) -> str:
    """The value: with echo off at a terminal, else the first line of what
    is piped in, with no prompt printed, so a script's stdout holds nothing
    but the command's answer (the TypeScript CLI's `promptSecretOrPipe`)."""
    if sys.stdin.isatty():
        import getpass

        try:
            return getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
    return sys.stdin.readline().rstrip("\r\n")


#: What `secrets set NAME VALUE` answers (2026-09-28, the e2e pass): it said
#: "too many arguments for 'set'", which did not say where the value goes.
#: The TypeScript CLI says the same (fixture `cli/final_sdk_low_items.json`).
VALUE_AS_ARGUMENT = (
    "`secrets set` takes the name alone: it asks for the value with echo off, or reads it from a pipe, "
    "so the value never lands in your shell history. Nothing was stored."
)


@app.command("set", cls=CommanderCommand, context_settings={"allow_extra_args": True})
def set_secret(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
    """Store a key, for example OPENAI_API_KEY (asked for with echo off, or read from a pipe)"""
    # Never from an argument: that lands in shell history and the process list.
    if ctx.args:
        print(VALUE_AS_ARGUMENT, file=sys.stderr)
        raise typer.Exit(1)
    if not _valid_name(name):
        print(f"{name} does not look like an environment variable name.", file=sys.stderr)
        raise typer.Exit(1)
    value = _read_secret(f"Value for {name}: ").strip()
    if not value:
        print("Nothing entered; nothing stored.", file=sys.stderr)
        raise typer.Exit(1)
    from webagents.agents.skills.local.secrets.keychain_ux import KeychainDialogBlocked

    try:
        backend = _store(quiet=True).set(name, value)
    except KeychainDialogBlocked as blocked:
        # Replacing it needs macOS to ask, and nobody can answer here.
        print(str(blocked), file=sys.stderr)
        raise typer.Exit(1)
    print(f"Stored {name} ({_stored_where(backend)}).")
    if os.environ.get(name):
        print(f"{name} is also set in this environment, and the environment wins.")


def _other_runtime_keys_line() -> str:
    """Where this CLI finds no keys and the TypeScript CLI has some in the
    keychain, say that each CLI keeps its own (keychain-ux, 2026-09-27). The
    file fallback is shared, so this is said for the keychain only."""
    try:
        from webagents.agents.skills.local.secrets.keychain_ux import KeychainRecord, other_keys_sentence, other_runtime_items, record_path

        from ..config_store import global_dir, profile_name, scoped_namespace

        if not _store(quiet=True).keystore:
            return ""
        profile = profile_name()
        record = KeychainRecord(record_path(global_dir(profile) / "secrets"))
        return other_keys_sentence() if other_runtime_items(record, scoped_namespace(NAMESPACE, profile)) else ""
    except Exception:  # noqa: BLE001 - a note, never a failure
        return ""


def _remove(name: str) -> None:
    if not _valid_name(name):
        print(f"{name} does not look like an environment variable name.", file=sys.stderr)
        raise typer.Exit(1)
    from webagents.agents.skills.local.secrets.keychain_ux import KeychainDialogBlocked

    from ..account import left_behind_lines

    store = _store(quiet=True)
    try:
        removed = store.delete(name)
    except KeychainDialogBlocked as blocked:
        print(str(blocked), file=sys.stderr)
        raise typer.Exit(1)
    if removed:
        print(f"Removed {name}.")
        # An old `webagents:` item only a macOS dialog could remove is named.
        for line in left_behind_lines(store.left_behind):
            print(line)
    else:
        print(f"{name} was not stored.", file=sys.stderr)
        raise typer.Exit(1)


@app.command("remove", cls=CommanderCommand)
def remove_secret(name: str = typer.Argument(...)) -> None:
    """Remove a stored key"""
    _remove(name)


@app.command("unset", cls=CommanderCommand, hidden=True)
def unset_secret(name: str = typer.Argument(...)) -> None:
    """Remove a stored key (the old name of remove)"""
    _remove(name)


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
    """Put the stored PROVIDER keys into `os.environ`, WITHOUT overriding what
    is there.

    Called by the commands that start an agent. Returns how many were added.
    The non-overriding rule is what makes this safe to call unconditionally:
    a shell that exports a key keeps winning, so adding a key here can never
    change the behaviour of a session that already had one.

    Only the provider variables, by NAME (S-292, 2026-09-26). This used to
    load every name the store's index held as well, which was harmless while
    the store held provider keys and agent API keys, and is not once it holds
    the secrets an MCP server names as `${secret:NAME}`: those are read by
    the MCP skill at connect time and must not become variables the whole
    agent process, its shell tool included, can read. Going by name also
    covers a key stored before the index was maintained, or by another tool;
    a `get` for an absent name is cheap and quiet.
    """
    try:
        # Quiet: this runs before every chat and run, where a log warning about
        # the file backend is noise; `secrets set` and `secrets backend` say it.
        store = _store(quiet=True)
    except Exception:
        return 0

    names = sorted(_known_env_vars())

    added = 0
    for name in names:
        if os.environ.get(name):
            continue
        value = store.get(name)
        if value:
            os.environ[name] = value
            added += 1
    return added
