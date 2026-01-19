"""
WebAgents CLI Main Entry Point

Typer-based CLI with all command groups.
"""

import typer
from typing import Optional
from rich.console import Console

# Create main Typer app
app = typer.Typer(
    name="webagents",
    help="WebAgents - AI Agent Framework. Build, run, and discover agents.",
    rich_markup_mode="rich",
    no_args_is_help=False,
    add_completion=True,
)

console = Console()

# Import command groups
from .commands import (
    agent,
    checkpoint,
    daemon,
    auth,
    config,
    session,
    skill,
)

# ===== COMMAND GROUP REGISTRATION =====

# Core command groups (mirrored in REPL)
app.add_typer(session.app, name="session", help="Session management")
app.add_typer(checkpoint.app, name="checkpoint", help="Checkpoint management (file snapshots)")
app.add_typer(skill.app, name="skill", help="Manage agent skills")
app.add_typer(daemon.app, name="daemon", help="Manage webagentsd daemon")
app.add_typer(auth.app, name="auth", help="Authentication with robutler.ai")
app.add_typer(config.app, name="config", help="Configuration management")

# Agent subcommands are exposed at top level
app.add_typer(agent.app, name="agent", help="Agent lifecycle management")


# ===== DIRECT COMMANDS =====

# Command functions used by main.py
from .commands.agent import connect_command, list_command, init_command


@app.command("connect")
def connect(
    agent: Optional[str] = typer.Argument(
        None,
        help="Agent name or path to AGENT.md file"
    ),
):
    """Start interactive REPL session with an agent."""
    connect_command(agent)


@app.command("list")
def list_agents(
    local: bool = typer.Option(False, "--local", "-l", help="Local agents only"),
    remote: bool = typer.Option(False, "--remote", "-r", help="Remote agents only"),
    running: bool = typer.Option(False, "--running", help="Currently running agents"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
):
    """List registered agents."""
    list_command(local=local, remote=remote, running=running, namespace=namespace)


@app.command("init")
def init(
    name: Optional[str] = typer.Argument(None, help="Agent name (creates AGENT-<name>.md)"),
    template_name: Optional[str] = typer.Option(None, "--template", "-t", help="Use template"),
    context: bool = typer.Option(False, "--context", "-c", help="Create AGENTS.md context file"),
):
    """Initialize a new agent in the current directory."""
    init_command(name=name, template_name=template_name, context=context)


@app.command("run")
def run(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path"),
    prompt: Optional[str] = typer.Option(None, "-p", "--prompt", help="Single prompt, exit after"),
):
    """Run an agent (headless execution)."""
    from .commands.agent import run as agent_run
    agent_run(agent=agent, prompt=prompt, all_agents=False)


@app.command("login")
def login(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Use API key instead of OAuth"),
):
    """Login to robutler.ai platform."""
    from .commands.auth import login_command
    login_command(api_key=api_key)


@app.command("version")
def version():
    """Show version information."""
    try:
        from importlib.metadata import version as get_version
        ver = get_version("webagents")
    except:
        ver = "0.1.0"
    
    console.print(f"[cyan]webagents[/cyan] version {ver}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    WebAgents CLI - Build, run, and discover AI agents.
    
    Run without arguments to start an interactive REPL session.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand - launch interactive REPL
        from .repl.session import start_repl
        start_repl()
    else:
        # Configure logging for subcommands (non-interactive)
        pass


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
