"""
The `webagents` command (2026-09-24).

THE TYPESCRIPT CLI'S COMMANDS, ARGUMENTS, OPTIONS AND WORDS
(`typescript/src/cli/index.ts`), which is the reference; `tests/cli/test_cli_parity.py`
reads that file and fails when the two drift. A script, a doc page or a person
moving between the SDKs meets one CLI:

    webagents [-m model] [-a agent] [-p prompt] [--output-format f] [--no-streaming]
    webagents chat | connect        the same, by name
    webagents serve [path]          one agent over HTTP (-p/--port, --host)
    webagents daemon                every agent in a folder (-p, --host, -w, --no-cron)
    webagents login | logout | whoami
    webagents link [name] | unlink
    webagents publish [path]        (-y, --dry-run)
    webagents init [name]           (-t chatbot|tool-agent)
    webagents doctor | models
    webagents skills list | templates list
    webagents config get|set|unset|validate|path
    webagents secrets list|set|unset|get
    global: --json, --profile, --token, -V/--version, -h/--help

It had grown its own surface: `run`, `dev`, `list`, `reset`, `completion`,
`version`, `deploy` and the `agent`, `auth`, `checkpoint`, `session`, `skill`,
`ui` and daemon `start|stop|restart|status|logs|endpoints` groups, none of
which the TypeScript CLI has. `webagents -p` replaced `run`, `publish`
replaced `deploy`, and `daemon` runs in the foreground as it does there.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from .commands import config, secrets
from .config_store import cli_command
from .help_format import CommanderCommand, commander_group

#: The TypeScript CLI's command order (`program.command(...)` in index.ts), which
#: `webagents -h` lists them in; `test_cli_parity.py` holds the two together.
COMMAND_ORDER = (
    "chat", "connect", "serve", "daemon", "login", "logout", "whoami", "link", "unlink",
    "doctor", "models", "skills", "templates", "config", "init", "publish", "secrets",
)

app = typer.Typer(
    name="webagents",
    help="Build and run AI agents",
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    cls=commander_group(COMMAND_ORDER, default="chat"),
)
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(secrets.app, name="secrets", help="Keys this CLI stored on this machine")

skills_app = typer.Typer(help="Skills an agent file can name", no_args_is_help=True, cls=commander_group())
templates_app = typer.Typer(help="Agent templates", no_args_is_help=True, cls=commander_group())
app.add_typer(skills_app, name="skills")
app.add_typer(templates_app, name="templates")

OUTPUT_FORMATS = ("text", "json", "stream-json")


def _json_enabled(ctx: typer.Context) -> bool:
    root = ctx.find_root()
    return bool((root.obj or {}).get("json"))


# ============================================================================
# chat (the default), connect
# ============================================================================


def _chat(model: Optional[str], agent: Optional[str], prompt: Optional[str], output_format: str, no_streaming: bool) -> None:
    """The chat, or one answer with `-p`: the TypeScript `chatAction`, flag for flag."""
    if output_format not in OUTPUT_FORMATS:
        print(f"Unknown --output-format '{output_format}'. Expected one of: {', '.join(OUTPUT_FORMATS)}.", file=sys.stderr)
        raise typer.Exit(2)

    from .agent_files import AgentNotFound, agent_file_for, default_agent_file

    folder = Path.cwd()
    if agent:
        # By name, as `/agent` does, or refused.
        try:
            agent_file = agent_file_for(folder, agent)
        except AgentNotFound as error:
            print(str(error), file=sys.stderr)
            raise typer.Exit(1)
    else:
        agent_file = default_agent_file(folder)

    from .loader import AgentFormatError

    if prompt:
        from .one_shot import run_prompt

        try:
            code = run_prompt(agent_file, prompt, model, output_format)
        except AgentFormatError as error:
            # The file's own sentence and exit 1, as the TypeScript CLI answers.
            print(str(error), file=sys.stderr)
            raise typer.Exit(1)
        raise typer.Exit(code)

    if output_format != "text":
        # Interactive output is a terminal transcript; there is no sane JSON of it.
        print(f"--output-format {output_format} applies to -p/--prompt only.", file=sys.stderr)
        raise typer.Exit(2)

    from .repl.session import start_repl

    try:
        start_repl(agent_path=agent_file, model=model, streaming=not no_streaming, chosen=True)
    except AgentFormatError as error:
        print(str(error), file=sys.stderr)
        raise typer.Exit(1)


def _chat_options(hidden: bool) -> Dict[str, Any]:
    """The chat's flags: on `chat` and `connect`, and (out of the help) on the
    root, where `webagents -p ...` reaches the chat as commander's default command."""
    return {
        "model": typer.Option(None, "-m", "--model", metavar="<model>", help="Model to use, as provider/model", hidden=hidden),
        "agent": typer.Option(None, "-a", "--agent", metavar="<agent>", help="Agent name", hidden=hidden),
        "prompt": typer.Option(None, "-p", "--prompt", metavar="<prompt>", help="Non-interactive prompt, then exit", hidden=hidden),
        "output_format": typer.Option("text", "--output-format", metavar="<format>", help="With -p: text, json, stream-json", hidden=hidden),
        "no_streaming": typer.Option(False, "--no-streaming", help="Disable streaming", hidden=hidden),
    }


_SHOWN = _chat_options(hidden=False)
_ROOT = _chat_options(hidden=True)
_MODEL, _AGENT, _PROMPT, _FORMAT, _NO_STREAMING = (_SHOWN[k] for k in ("model", "agent", "prompt", "output_format", "no_streaming"))


@app.command("chat", cls=CommanderCommand)
def chat(
    model: Optional[str] = _MODEL,
    agent: Optional[str] = _AGENT,
    prompt: Optional[str] = _PROMPT,
    output_format: str = _FORMAT,
    no_streaming: bool = _NO_STREAMING,
) -> None:
    """Start interactive chat session"""
    _chat(model, agent, prompt, output_format, no_streaming)


@app.command("connect", cls=CommanderCommand)
def connect(
    model: Optional[str] = _MODEL,
    agent: Optional[str] = _AGENT,
    prompt: Optional[str] = _PROMPT,
    output_format: str = _FORMAT,
    no_streaming: bool = _NO_STREAMING,
) -> None:
    """Start interactive session (alias for chat)"""
    _chat(model, agent, prompt, output_format, no_streaming)


# ============================================================================
# serve, daemon
# ============================================================================


@app.command("serve", cls=CommanderCommand)
def serve(
    path: str = typer.Argument(".", help="Path to agent config file"),
    port: int = typer.Option(3000, "-p", "--port", metavar="<port>", help="Port"),
    host: Optional[str] = typer.Option(None, "--host", metavar="<host>", help="Interface to listen on (0.0.0.0 for every interface)"),
) -> None:
    """Serve an agent on HTTP"""
    from .serve import serve_command

    serve_command(path, port, host)


@app.command("daemon", cls=CommanderCommand)
def daemon(
    port: Optional[int] = typer.Option(None, "-p", "--port", metavar="<port>", help="Port (default: daemon.port, 8765)"),
    host: Optional[str] = typer.Option(None, "--host", metavar="<host>", help="Interface to listen on (default: daemon.host, 127.0.0.1)"),
    watch: Optional[str] = typer.Option(None, "-w", "--watch", metavar="<dir>", help="Watch directory"),
    no_cron: bool = typer.Option(False, "--no-cron", help="Disable cron"),
) -> None:
    """Start the WebAgents daemon"""
    from .commands.daemon import run_daemon

    run_daemon(port=port, host=host, watch=watch, cron=not no_cron)


# ============================================================================
# login, logout, whoami, link, unlink, doctor
# ============================================================================


@app.command("login", cls=CommanderCommand)
def login(
    url: Optional[str] = typer.Option(None, "-u", "--url", metavar="<url>", help="Portal URL (default: platform.url, or ROBUTLER_API_URL)"),
    token: Optional[str] = typer.Option(None, "-t", "--token", metavar="<token>", help="API key to sign in with, instead of the browser"),
) -> None:
    """Authenticate with the portal"""
    from .account import login_command

    raise typer.Exit(login_command(url, token))


@app.command("logout", cls=CommanderCommand)
def logout() -> None:
    """Sign out of Robutler"""
    from .account import logout_command

    logout_command()


@app.command("whoami", cls=CommanderCommand)
def whoami(ctx: typer.Context) -> None:
    """Show who you are signed in as"""
    from .account import who_am_i
    from .output import emit, fail

    result = who_am_i()
    if _json_enabled(ctx):
        if not result.ok:
            fail(result.code, result.message, result.fix)
        emit({"username": result.username, "platform": result.platform})
        return
    if not result.ok:
        print(result.message, file=sys.stderr)
        if result.fix:
            print(result.fix, file=sys.stderr)
        raise typer.Exit(1)
    print(result.message)


@app.command("link", cls=CommanderCommand)
def link(
    name: Optional[str] = typer.Argument(None, help="The agent's name; defaults to the name in this folder's agent file"),
    show: bool = typer.Option(False, "--show", help="Say what this folder is linked to"),
) -> None:
    """Link this folder to one of your agents on Robutler"""
    from .account import link_folder, show_link

    def error(line: str) -> None:
        print(line, file=sys.stderr)

    ok = show_link(Path.cwd(), print) if show else link_folder(Path.cwd(), name, print, error)
    if not ok:
        raise typer.Exit(1)


@app.command("unlink", cls=CommanderCommand)
def unlink() -> None:
    """Forget the agent this folder is linked to"""
    from .account import unlink_folder

    unlink_folder(Path.cwd(), print)


@app.command("doctor", cls=CommanderCommand)
def doctor(ctx: typer.Context) -> None:
    """Check this setup and say what to fix"""
    from .doctor import run_checks, report_lines
    from .output import emit

    checks = run_checks()
    if _json_enabled(ctx):
        emit({"checks": [c.as_dict() for c in checks]})
    else:
        for line in report_lines(checks):
            print(line)
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(1)


# ============================================================================
# models, skills, templates
# ============================================================================


@app.command("models", cls=CommanderCommand)
def models() -> None:
    """List LLM providers and which are configured here"""
    from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

    from .commands.secrets import _store

    try:
        store = _store(quiet=True)
    except Exception:  # noqa: BLE001 - the shell alone still answers
        store = None

    def has_key(env_var: str) -> bool:
        if os.environ.get(env_var):
            return True
        try:
            return bool(store is not None and store.get(env_var))
        except Exception:  # noqa: BLE001
            return False

    print("\nLLM providers:\n")
    for p in LLM_PROVIDERS:
        ready = p.credential == "local" or any(has_key(v) for v in p.env_vars)
        # The first variable the skill reads, as the TypeScript CLI prints it.
        first = p.env_vars[0] if p.env_vars else ""
        if p.credential == "local":
            needs = "no credential needed"
        elif p.credential == "platform":
            needs = f"{first} (or {cli_command('login')})"
        else:
            needs = first
        print(f"  {('ready' if ready else '-').ljust(6)} {p.id.ljust(14)} {p.model_format.ljust(24)} {needs}")
    print(f'\n  "ready" means this machine has its key: set in this shell, or stored with `{cli_command("secrets set")}`.')
    print("  Pass --model as provider/model, using an id from the provider.\n")


@skills_app.command("list", cls=CommanderCommand)
def skills_list() -> None:
    """Skills an agent file can name"""
    from .agent_builder import SKILL_CLASSES

    print("\nSkills an agent file can name:\n")
    for name in sorted(SKILL_CLASSES):
        print(f"  {name}")
    print()


#: What `init` can make: the TypeScript CLI's table, name for name.
INIT_TEMPLATES = {
    "chatbot": ("A chat agent: one model, no tools", ["openai"]),
    "tool-agent": ("Can read and write files and run shell commands", ["openai", "filesystem", "shell"]),
}


@templates_app.command("list", cls=CommanderCommand)
def templates_list() -> None:
    """List available templates"""
    print("\nAvailable Templates:\n")
    for name, (description, _skills) in INIT_TEMPLATES.items():
        print(f"  {name.ljust(20)} {description}")
    print("\nUse: webagents init <name> --template <template>\n")


# ============================================================================
# init, publish
# ============================================================================


@app.command("init", cls=CommanderCommand)
def init(
    name: str = typer.Argument("my-agent", help="Project name"),
    template: str = typer.Option("chatbot", "-t", "--template", metavar="<template>", help="Template to use"),
) -> None:
    """Initialize a new agent project"""
    from webagents.agents.skills.core.llm.providers import find_provider

    # Checked before anything is created, so a typo leaves no directory behind.
    if template not in INIT_TEMPLATES:
        print(f"Unknown template '{template}'. Available: {', '.join(INIT_TEMPLATES)}.", file=sys.stderr)
        raise typer.Exit(1)
    folder = Path(name).resolve()
    if folder.exists():
        print(f"Directory {name} already exists.", file=sys.stderr)
        raise typer.Exit(1)
    folder.mkdir(parents=True)

    openai = find_provider("openai")
    model = f"openai/{(openai.default_model if openai else None) or 'gpt-4o-mini'}"
    _description, skills = INIT_TEMPLATES[template]
    # The file both SDKs read, byte for byte what the TypeScript `init` writes.
    lines = ["---", f"name: {name}", f"description: A {template} agent", f"model: {model}", "skills:"]
    lines += [f"  - {skill}" for skill in skills]
    lines += ["---", "", f"# {name}", "", "You are a helpful assistant.", ""]
    (folder / "AGENT.md").write_text("\n".join(lines))

    print(f"\nCreated agent project: {name}/")
    print("  AGENT.md       the agent: its model, skills and instructions")
    print("\nNext steps:")
    print(f"  cd {name}")
    # The commands as they must be typed here: with `--profile` under one.
    chat, serve_cmd = cli_command(), cli_command("serve")
    width = max(21, len(serve_cmd) + 2)
    print(f"  {chat.ljust(width)}chat with it")
    print(f"  {serve_cmd.ljust(width)}serve it over HTTP")
    print(
        f"\nIt runs on {model}: add your key with `{cli_command('secrets set OPENAI_API_KEY')}`, "
        f"or sign in with `{cli_command('login')}` to run it through Robutler.\n"
    )


@app.command("publish", cls=CommanderCommand)
def publish(
    path: str = typer.Argument(".", help="Path to agent config"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Do not ask before creating a new agent"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be sent, and send nothing"),
) -> None:
    """Publish the agent to Robutler, or update the one this folder is linked to"""
    from .publish import PublishIO, publish_agent

    async def confirm(question: str) -> bool:
        if not sys.stdin.isatty():
            print("Pass --yes to create it without a prompt.", file=sys.stderr)
            return False
        answer = await asyncio.to_thread(input, f"{question} [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    def error(line: str) -> None:
        print(line, file=sys.stderr)

    result = asyncio.run(
        publish_agent(Path(path), PublishIO(ok=print, print=print, error=error, confirm=confirm), yes=yes, dry_run=dry_run)
    )
    if not result.ok:
        raise typer.Exit(1)


# ============================================================================
# The root: global flags, and the chat when no command is named
# ============================================================================


def _print_version(value: bool) -> None:
    """`--version`, bare on stdout, so `$(webagents --version)` is just the number."""
    if value:
        from webagents import __version__

        print(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(False, "-V", "--version", help="output the version number", is_eager=True, callback=_print_version),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output: one JSON document on stdout, diagnostics on stderr"),
    profile: Optional[str] = typer.Option(None, "--profile", metavar="<name>", help="Use a separate set of settings, keys and sign-in", envvar="WEBAGENTS_PROFILE", show_envvar=False),
    token: Optional[str] = typer.Option(None, "--token", metavar="<token>", help="Use this platform token for this run, instead of the stored sign-in", envvar="WEBAGENTS_TOKEN", show_envvar=False),
    model: Optional[str] = _ROOT["model"],
    agent: Optional[str] = _ROOT["agent"],
    prompt: Optional[str] = _ROOT["prompt"],
    output_format: str = _ROOT["output_format"],
    no_streaming: bool = _ROOT["no_streaming"],
) -> None:
    """Build and run AI agents"""
    ctx.obj = {"profile": profile, "token": token, "json": json_out}

    # The profile goes into the environment, where every later lookup reads it
    # (and a child process inherits it). The token does NOT: a bearer handed
    # to every child is a wider blast radius than the flag asked for, so it
    # stays in this process (S-222).
    if profile:
        os.environ["WEBAGENTS_PROFILE"] = profile
    from .credentials import set_flag_token

    set_flag_token(token)

    # QUIET BY DEFAULT, AND ON STDERR: stdout is a command's answer, so `-p`
    # and `--json` output stays clean. The daemon keeps INFO, because that
    # output is its log; WEBAGENTS_LOG_LEVEL overrides.
    if ctx.invoked_subcommand != "daemon":
        from ..utils.logging import setup_logging

        setup_logging(level="WARNING", stream=sys.stderr, tracebacks=bool(os.environ.get("WEBAGENTS_DEBUG")))

    # Stored provider keys, for every command that runs an agent. Never over
    # what the shell exports (`load_into_environment`).
    if ctx.invoked_subcommand in (None, "chat", "connect", "serve", "daemon", "doctor"):
        try:
            from .commands.secrets import load_into_environment

            load_into_environment()
        except Exception:  # noqa: BLE001 - the command's own check names a missing key
            pass

    if ctx.invoked_subcommand is None:
        _chat(model, agent, prompt, output_format, no_streaming)


def cli() -> None:
    """Entry point for the CLI."""
    # The name the help and errors use, however it was started (`python -m webagents` too).
    app(prog_name="webagents")


if __name__ == "__main__":
    cli()
