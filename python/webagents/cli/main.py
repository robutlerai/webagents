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
    webagents mcp serve [path]      the agent's tools to an MCP client (--http <port>, --host)
    webagents acp [path]            the agent to a code editor over ACP (stdio)
    webagents login | logout | whoami
    webagents link [name] | unlink
    webagents publish [path]        (-y, --dry-run)
    webagents init [name]           (-t chatbot|tool-agent)
    webagents doctor | models
    webagents skills list | add|remove <names...> (-a; add: --skill, -y)
    webagents templates list
    webagents config get|set|unset|validate|path
    webagents secrets list|set|remove|get
    webagents sandbox setup         whether the sandbox engine runs on this machine
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
from typing import Any, Dict, List, Optional

import typer

from .commands import config, secrets
from .config_store import cli_command
from .help_format import CommanderCommand, commander_group

#: The TypeScript CLI's command order (`program.command(...)` in index.ts), which
#: `webagents -h` lists them in; `test_cli_parity.py` holds the two together.
COMMAND_ORDER = (
    "chat", "connect", "serve", "daemon", "mcp", "cron", "acp", "login", "logout", "whoami", "budget", "link", "unlink",
    "doctor", "models", "skills", "templates", "config", "init", "publish", "secrets", "sandbox",
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
app.add_typer(secrets.app, name="secrets", help="Keys and secrets this CLI keeps on this machine")

skills_app = typer.Typer(help="Skills an agent file can name", no_args_is_help=True, cls=commander_group())
templates_app = typer.Typer(help="Agent templates", no_args_is_help=True, cls=commander_group())
app.add_typer(skills_app, name="skills")
app.add_typer(templates_app, name="templates")

sandbox_app = typer.Typer(help="The sandbox that confines shell commands", no_args_is_help=True, cls=commander_group())
app.add_typer(sandbox_app, name="sandbox")

OUTPUT_FORMATS = ("text", "json", "stream-json")


def _json_enabled(ctx: typer.Context) -> bool:
    root = ctx.find_root()
    return bool((root.obj or {}).get("json"))


# ============================================================================
# chat (the default), connect
# ============================================================================


def _chat(model: Optional[str], agent: Optional[str], prompt: Optional[str], output_format: str, no_streaming: bool, json_out: bool = False) -> None:
    """The chat, or one answer with `-p`: the TypeScript `chatAction`, flag for flag."""
    if output_format not in OUTPUT_FORMATS:
        print(f"Unknown --output-format '{output_format}'. Expected one of: {', '.join(OUTPUT_FORMATS)}.", file=sys.stderr)
        raise typer.Exit(2)
    # `--json` with `-p` is `--output-format json` (2026-09-27, the final e2e
    # re-run: the global flag was ignored here). An explicit format wins, and
    # the flag alone still opens the chat.
    if prompt and json_out and output_format == "text":
        output_format = "json"

    from .agent_files import AgentNotFound, agent_file_for, default_agent_file

    folder = Path.cwd()
    if agent:
        # By name, as `/agent` does, or refused. Under `--json` the refusal
        # is the error envelope (`output.fail`, code `agent_not_found`), which
        # was ignored here (2026-09-26, the e2e run).
        try:
            agent_file = agent_file_for(folder, agent)
        except AgentNotFound as error:
            if json_out:
                from .output import fail

                fail("agent_not_found", str(error))
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
    ctx: typer.Context,
    model: Optional[str] = _MODEL,
    agent: Optional[str] = _AGENT,
    prompt: Optional[str] = _PROMPT,
    output_format: str = _FORMAT,
    no_streaming: bool = _NO_STREAMING,
) -> None:
    """Start interactive chat session"""
    _chat(model, agent, prompt, output_format, no_streaming, _json_enabled(ctx))


@app.command("connect", cls=CommanderCommand)
def connect(
    ctx: typer.Context,
    model: Optional[str] = _MODEL,
    agent: Optional[str] = _AGENT,
    prompt: Optional[str] = _PROMPT,
    output_format: str = _FORMAT,
    no_streaming: bool = _NO_STREAMING,
) -> None:
    """Start interactive session (alias for chat)"""
    _chat(model, agent, prompt, output_format, no_streaming, _json_enabled(ctx))


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
# mcp serve
# ============================================================================

# `webagents mcp serve` (plan item 1.8, 2026-09-26): the agent's tools to an
# MCP client, over stdio by default (how Claude Code, Codex and OpenCode start
# a local server) or over Streamable HTTP with `--http`. The TypeScript group
# and command, word for word (`test_cli_parity.py`); the words are also held
# by `tests/fixtures/mcp_tool/serve.json`.
mcp_app = typer.Typer(help="Serve an agent over the Model Context Protocol", no_args_is_help=True, cls=commander_group())
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("serve", cls=CommanderCommand)
def mcp_serve(
    path: str = typer.Argument(".", help="Path to agent config file"),
    http: Optional[int] = typer.Option(None, "--http", metavar="<port>", help="Serve over Streamable HTTP on this port instead of stdio"),
    host: Optional[str] = typer.Option(None, "--host", metavar="<host>", help="Interface to listen on with --http (0.0.0.0 for every interface)"),
) -> None:
    """Serve an agent's tools to an MCP client: over stdio, or over Streamable HTTP with --http"""
    from .mcp_serve import mcp_serve_command

    mcp_serve_command(path, http, host)


# ============================================================================
# cron list, cron run
# ============================================================================

# `webagents cron` (plan item 1.7, 2026-09-26): the `cron:` schedules the agent
# files in a folder declare, as the daemon runs them. `list` reads the files
# and the runner's state and writes nothing; `run` builds the one agent and
# runs the schedule now, delivering as configured. The TypeScript group and
# commands, word for word (`test_cli_parity.py`); the lines are held by
# `tests/fixtures/cli/cron.json`.
cron_app = typer.Typer(help="Schedules the agents in a folder declare", no_args_is_help=True, cls=commander_group())
app.add_typer(cron_app, name="cron")

_WATCH_HELP = "Folder whose agents to read (default: this folder)"


@cron_app.command("list", cls=CommanderCommand)
def cron_list(
    ctx: typer.Context,
    watch: Optional[str] = typer.Option(None, "-w", "--watch", metavar="<dir>", help=_WATCH_HELP),
) -> None:
    """List the schedules of the agents in a folder"""
    from .cron_command import list_schedules_command

    # Non-zero when a file was refused (2026-09-26): the listing says so and exits 1.
    code = list_schedules_command(Path(watch) if watch else Path.cwd(), _json_enabled(ctx))
    if code:
        raise typer.Exit(code)


@cron_app.command("run", cls=CommanderCommand)
def cron_run(
    ctx: typer.Context,
    agent: str = typer.Argument(..., help="The agent, by name"),
    name: str = typer.Argument(..., help="The schedule, by name"),
    watch: Optional[str] = typer.Option(None, "-w", "--watch", metavar="<dir>", help=_WATCH_HELP),
) -> None:
    """Run a schedule now and deliver as configured"""
    from .cron_command import run_schedule_command

    code = run_schedule_command(agent, name, Path(watch) if watch else Path.cwd(), _json_enabled(ctx))
    if code:
        raise typer.Exit(code)


# ============================================================================
# acp
# ============================================================================

# `webagents acp` (plan item 1.6, 2026-09-26): the agent to a code editor over
# the Agent Client Protocol, on stdin and stdout (how Zed and the JetBrains
# IDEs start an agent). The TypeScript command, word for word
# (`test_cli_parity.py`); the words are also held by `tests/fixtures/acp/acp_protocol.json`.


@app.command("acp", cls=CommanderCommand)
def acp(
    path: str = typer.Argument(".", help="Path to agent config file"),
) -> None:
    """Serve an agent to a code editor over the Agent Client Protocol (stdio)"""
    from .acp_serve import acp_command

    acp_command(path)


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

    code = logout_command()
    if code:
        raise typer.Exit(code)


@app.command("whoami", cls=CommanderCommand)
def whoami(ctx: typer.Context) -> None:
    """Show who you are signed in as"""
    from .account import settle_keychain, who_am_i
    from .output import emit, fail

    # In a terminal, what a run with nobody to answer macOS could not read is
    # read now, so macOS asks once, here (keychain-ux, 2026-09-27): this is
    # the command that run's one sentence names.
    settle_keychain()
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


@app.command("budget", cls=CommanderCommand)
def budget(ctx: typer.Context, token_id: str = typer.Argument(..., help="A payment token id, from your token list on the platform")) -> None:
    """Show the budget tree of a run: a payment token and every child a hop derived from it"""
    from .budget_tree import budget_tree
    from .output import emit, fail

    result = budget_tree(token_id)
    if _json_enabled(ctx):
        if not result.ok:
            fail(result.code, result.message, result.fix)
        emit({"tree": result.tree})
        return
    if not result.ok:
        print(result.message, file=sys.stderr)
        if result.fix:
            print(result.fix, file=sys.stderr)
        raise typer.Exit(1)
    for line in result.lines:
        print(line)
    print(result.totals)


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
def doctor(
    ctx: typer.Context,
    # `-a`, as the chat takes it (2026-09-26): `doctor -a helper` was "unknown option".
    agent: Optional[str] = typer.Option(None, "-a", "--agent", metavar="<agent>", help="Agent name"),
) -> None:
    """Check this setup and say what to fix"""
    from .agent_files import AgentNotFound
    from .doctor import run_checks, report_lines
    from .output import emit, fail

    if not os.environ.get("WEBAGENTS_DEBUG"):
        # THE AGENT'S LOG GOES TO A FILE, as `-p` sends it (2026-09-26, the
        # e2e run): building the agent for the checks printed the skills'
        # log lines (an MCP server that did not connect) ABOVE the report,
        # which carries the same finding in its `mcp` check. WEBAGENTS_DEBUG
        # keeps the log on stderr. The TypeScript doctor holds its console
        # the same way.
        from ..utils.logging import setup_logging

        # The profile's own folder, as the chat's `repl_log_path` (the
        # ptypass-fixes lane, 2026-09-27): this went to `~/.webagents/logs`
        # whatever `--profile` said.
        from .config_store import global_dir

        log_dir = global_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(level="INFO", log_file=str(log_dir / "repl.log"), console_output=False)

    try:
        checks = run_checks(agent=agent)
    except AgentNotFound as error:
        # An agent the folder does not have: the chat's sentence, or the error envelope.
        if _json_enabled(ctx):
            fail("agent_not_found", str(error))
        print(str(error), file=sys.stderr)
        raise typer.Exit(1)
    if _json_enabled(ctx):
        emit({"checks": [c.as_dict() for c in checks]})
    else:
        for line in report_lines(checks):
            print(line)
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(1)


@sandbox_app.command("setup", cls=CommanderCommand)
def sandbox_setup(ctx: typer.Context) -> None:
    """Check that the sandbox engine runs on this machine"""
    # A CHECK, NOT AN INSTALLER (the sandbox-engine lane, 2026-09-27). The
    # engine ships with this package and node comes from PATH or the
    # nodejs-wheel-binaries dependency, so what can still be missing is the
    # machine's: the Linux programs (named with this distribution's install
    # line), what a container must allow, WSL 2 on Windows. It installs and
    # downloads nothing, runs a real confined `true`, and exits 1 when shell
    # commands would be refused. The words are the TypeScript CLI's
    # (`srt.ts` `setupChecks`), pinned by tests/fixtures/sandbox/sandbox_engine.json.
    from ..sandbox.srt import setup_checks
    from .doctor import Check, report_lines
    from .output import emit

    checks = [Check(c["name"], c["status"], c["detail"], c.get("fix")) for c in setup_checks()]
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
def models(ctx: typer.Context) -> None:
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

    from webagents.agents.skills.core.llm.ollama.probe import probe_ollama
    from webagents.agents.skills.core.llm.providers import MODELS_READY_FOOTNOTE, provider_base_url, provider_needs

    rows = []
    for p in LLM_PROVIDERS:
        if p.credential == "none":
            # A local server (Ollama, plan item 2.8) is ready when it answers.
            base = provider_base_url(p)
            ready = bool(base) and probe_ollama(base).ok
        elif p.credential == "platform":
            # Robutler's models are ready when this profile is signed in
            # (2026-09-27): the row said `-` for a signed-in person, since
            # only the proxy URL variable was looked at.
            from .model_access import is_signed_in

            ready = is_signed_in()
        else:
            ready = p.credential == "local" or any(has_key(v) for v in p.env_vars)
        # The same column words as the TypeScript CLI (`provider_needs`).
        rows.append({"id": p.id, "ready": bool(ready), "model_format": p.model_format, "needs": provider_needs(p, cli_command("login"))})
    # `--json` (2026-09-27): the table's rows as one document (fixture `cli/json_documents.json`, `models`).
    if _json_enabled(ctx):
        from .output import emit

        emit({"providers": rows})
        return
    print("\nLLM providers:\n")
    for row in rows:
        print(f"  {('ready' if row['ready'] else '-').ljust(6)} {row['id'].ljust(14)} {row['model_format'].ljust(24)} {row['needs']}")
    print("\n" + MODELS_READY_FOOTNOTE.replace("{command}", cli_command("secrets set")))
    print("  Pass --model as provider/model, using an id from the provider.\n")


@skills_app.command("list", cls=CommanderCommand)
def skills_list() -> None:
    """Skills an agent file can name"""
    from .agent_builder import SKILL_CLASSES
    from .skills_edit import skillmd_list_lines

    print("\nSkills an agent file can name:\n")
    for name in sorted(SKILL_CLASSES):
        print(f"  {name}")
    # The folder's SKILL.md skills, apart from the coded names (plan item 1.4).
    print()
    for line in skillmd_list_lines(Path.cwd()):
        print(line)
    print()


# `skills add` and `skills remove` change the `skills:` list of this folder's
# agent file and nothing else (`skills_edit.py`, the TypeScript `skills-edit.ts`),
# except for a SOURCE (`owner/repo`, a git URL, a folder), which installs a
# SKILL.md skill into `.agents/skills` instead (plan item 1.4).
@skills_app.command("add", cls=CommanderCommand)
def skills_add(
    ctx: typer.Context,
    names: List[str] = typer.Argument(..., help="Skills to add, by the names `skills list` shows, or a SKILL.md source: owner/repo, a git URL or a folder"),
    agent: Optional[str] = typer.Option(None, "-a", "--agent", metavar="<agent>", help="Agent name"),
    skill: Optional[str] = typer.Option(None, "--skill", metavar="<name>", help="Install only this skill from the source"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Install without asking"),
) -> None:
    """Add skills to an agent file"""
    from .skills_edit import skills_command

    if not _json_enabled(ctx):
        code = skills_command("add", names, agent=agent, skill=skill, yes=yes)
        if code:
            raise typer.Exit(code)
        return
    # `--json` (2026-09-27): one document, the editor's facts and the lines
    # it would have printed (fixture `cli/json_documents.json`, `skills_add`);
    # a refusal is the error envelope with the lines it would have printed.
    from .output import emit, fail

    messages: List[str] = []
    errors: List[str] = []
    edited: Dict[str, Any] = {}
    code = skills_command("add", names, agent=agent, skill=skill, yes=yes, out=messages.append, err=errors.append, edited=edited.update)
    if code:
        fail("skills_add_failed", "\n".join([*errors, *messages]), exit_code=code)
    emit({"file": edited.get("file"), "added": edited.get("added", []), "already": edited.get("already", []), "messages": messages})


@skills_app.command("remove", cls=CommanderCommand)
def skills_remove(
    names: List[str] = typer.Argument(..., help="Skills to remove"),
    agent: Optional[str] = typer.Option(None, "-a", "--agent", metavar="<agent>", help="Agent name"),
) -> None:
    """Remove skills from an agent file"""
    from .skills_edit import skills_command

    code = skills_command("remove", names, agent=agent)
    if code:
        raise typer.Exit(code)


# The templates `init` and the chat's `/agent new` write live in
# `cli/init_templates.py` (2026-09-26, interactive-mode spec 3.3), so the two
# commands produce identical bytes. Pinned by `tests/fixtures/cli/init_templates.json`.
from .init_templates import INIT_TEMPLATES, TOOL_AGENT_ACCESS, agent_markdown  # noqa: F401,E402


@templates_app.command("list", cls=CommanderCommand)
def templates_list(ctx: typer.Context) -> None:
    """List available templates"""
    # `--json` (2026-09-27): the same table as one document (fixture `cli/json_documents.json`, `templates_list`).
    if _json_enabled(ctx):
        from .output import emit

        emit({"templates": [{"name": name, "description": description} for name, (description, _skills, _sandbox, _access) in INIT_TEMPLATES.items()]})
        return
    print("\nAvailable Templates:\n")
    for name, (description, _skills, _sandbox, _access) in INIT_TEMPLATES.items():
        print(f"  {name.ljust(20)} {description}")
    print("\nUse: webagents init <name> --template <template>\n")


# ============================================================================
# init, publish
# ============================================================================


@app.command("init", cls=CommanderCommand)
def init(
    ctx: typer.Context,
    name: str = typer.Argument("my-agent", help="Project name"),
    template: str = typer.Option("chatbot", "-t", "--template", metavar="<template>", help="Template to use"),
) -> None:
    """Initialize a new agent project"""
    from .init_templates import ROBUTLER_CHOICE_MODEL, init_line, init_model

    # `--json` (2026-09-27): one document either way (fixture
    # `cli/json_documents.json`, `init`; the refusals in `cli/json_errors.json`).
    from .output import emit, fail

    json_out = _json_enabled(ctx)
    # Checked before anything is created, so a typo leaves no directory behind.
    if template not in INIT_TEMPLATES:
        message = f"Unknown template '{template}'. Available: {', '.join(INIT_TEMPLATES)}."
        if json_out:
            fail("unknown_template", message)
        print(message, file=sys.stderr)
        raise typer.Exit(1)
    folder = Path(name).resolve()
    if folder.exists():
        if json_out:
            fail("directory_exists", f"Directory {name} already exists.")
        print(f"Directory {name} already exists.", file=sys.stderr)
        raise typer.Exit(1)
    folder.mkdir(parents=True)

    # A provider's model when this machine holds its key, else Robutler's
    # choice (B3, 2026-09-28; `init_templates.init_model`).
    keyed = init_model()
    model = keyed or ROBUTLER_CHOICE_MODEL
    # The file both SDKs read, byte for byte what the chat's /agent new writes.
    (folder / "AGENT.md").write_text(agent_markdown(name, template, keyed))

    if json_out:
        # The model the file names; null when it names none (B3).
        emit({"name": name, "template": template, "path": str(folder), "files": ["AGENT.md"], "model": keyed})
        return
    print(f"\nCreated agent project: {name}/")
    print("  AGENT.md       the agent: its model, skills and instructions")
    print("\nNext steps:")
    print(f"  cd {name}")
    # The commands as they must be typed here: with `--profile` under one.
    chat, serve_cmd = cli_command(), cli_command("serve")
    width = max(21, len(serve_cmd) + 2)
    print(f"  {chat.ljust(width)}chat with it")
    print(f"  {serve_cmd.ljust(width)}serve it over HTTP")
    from .model_access import is_signed_in

    # What runs it, and the way in only when one is needed: no `login` hint
    # for a person already signed in (2026-09-28, fixture `init_line`).
    print(f"\n{init_line(model, keyed is not None, is_signed_in())}\n")


@app.command("publish", cls=CommanderCommand)
def publish(
    ctx: typer.Context,
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

    # `--json` (2026-09-27): the lines go to stderr and the outcome is one
    # document, for `--dry-run` the request that would have been sent
    # (fixture `cli/json_documents.json`, `publish_dry_run`).
    json_out = _json_enabled(ctx)
    errors: List[str] = []

    def say(line: str) -> None:
        print(line, file=sys.stderr if json_out else sys.stdout)

    def error(line: str) -> None:
        if json_out:
            errors.append(line)
        else:
            print(line, file=sys.stderr)

    result = asyncio.run(
        publish_agent(Path(path), PublishIO(ok=say, print=say, error=error, confirm=confirm), yes=yes, dry_run=dry_run)
    )
    if json_out:
        from .output import emit, fail

        if not result.ok:
            fail("publish_failed", "\n".join(errors))
        if result.request:
            emit({"method": result.request["method"], "url": result.request["url"], "body": result.request["body"], "created": result.created})
        else:
            emit({"username": result.username, "agent_id": result.agent_id, "created": result.created})
        return
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
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Run shell commands with your permissions for this run, outside the operating-system sandbox", envvar="WEBAGENTS_NO_SANDBOX", show_envvar=False),
    max_tool_rounds: Optional[str] = typer.Option(None, "--max-tool-rounds", metavar="<n>", help="Tool rounds one turn may run before its last answer (default 50)"),
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
    # `--max-tool-rounds` (2026-09-28, `agents/core/tool_budget.py`): checked
    # here, once, then read by every agent this run builds (the chat, `-p`,
    # `serve`, the daemon) from the environment, as `--profile` is.
    if max_tool_rounds is not None:
        from webagents.agents.core.tool_budget import MAX_TOOL_ROUNDS_ENV, parse_max_tool_rounds

        try:
            os.environ[MAX_TOOL_ROUNDS_ENV] = str(parse_max_tool_rounds(max_tool_rounds, "--max-tool-rounds"))
        except ValueError as e:
            sys.stderr.write(f"{e}\n")
            raise typer.Exit(1)
    # The sandbox is on by default (2026-09-27): `--no-sandbox` is the one-run
    # opt-out, for the owner's shell commands only. The shell reads the
    # environment (`webagents.sandbox.ENV_NO_SANDBOX`), so a child
    # `webagents` inherits the choice for the run.
    if no_sandbox:
        os.environ["WEBAGENTS_NO_SANDBOX"] = "1"
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
    if ctx.invoked_subcommand in (None, "chat", "connect", "serve", "daemon", "mcp", "acp", "cron", "doctor"):
        try:
            from .commands.secrets import load_into_environment

            load_into_environment()
        except Exception:  # noqa: BLE001 - the command's own check names a missing key
            pass

    if ctx.invoked_subcommand is None:
        _chat(model, agent, prompt, output_format, no_streaming, json_out)


def cli() -> None:
    """Entry point for the CLI."""
    # The name the help and errors use, however it was started (`python -m webagents` too).
    app(prog_name="webagents")


if __name__ == "__main__":
    cli()
