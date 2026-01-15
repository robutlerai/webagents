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
    daemon,
    discover,
    auth,
    config,
    skill,
    template,
    intent,
    namespace,
    cron,
    index,
)

# Register command groups
app.add_typer(agent.app, name="run", help="Run agents (headless execution)")
app.add_typer(daemon.app, name="daemon", help="Manage webagentsd daemon")
app.add_typer(discover.app, name="discover", help="Discover agents by intent")
app.add_typer(auth.app, name="auth", help="Authentication with robutler.ai")
app.add_typer(config.app, name="config", help="Configuration management")
app.add_typer(skill.app, name="skill", help="Manage agent skills")
app.add_typer(template.app, name="template", help="Agent templates")
app.add_typer(intent.app, name="intent", help="Intent publishing and subscriptions")
app.add_typer(namespace.app, name="namespace", help="Namespace management")
app.add_typer(cron.app, name="cron", help="Scheduled agent execution")
app.add_typer(index.app, name="index", help="Vector index management")

# Direct commands (not in subgroups)
from .commands.agent import connect_command, list_command, init_command
from .commands.register import register_command, scan_command
from .commands.sync import sync_command, publish_command


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


@app.command("register")
def register(
    path: Optional[str] = typer.Argument(None, help="Path to agent file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes"),
):
    """Register agents with the local daemon."""
    register_command(path=path, recursive=recursive, watch=watch)


@app.command("scan")
def scan(
    path: Optional[str] = typer.Argument(None, help="Path to scan for agents"),
):
    """Scan and list discoverable AGENT*.md files."""
    scan_command(path=path)


@app.command("sync")
def sync(
    agent: Optional[str] = typer.Argument(None, help="Specific agent to sync"),
    auto: bool = typer.Option(False, "--auto", help="Enable auto-sync"),
):
    """Sync agents with remote registry (robutler.ai)."""
    sync_command(agent=agent, auto=auto)


@app.command("publish")
def publish(
    agent: Optional[str] = typer.Argument(None, help="Agent to publish"),
    internal: bool = typer.Option(False, "--internal", help="Internal namespace only"),
    public: bool = typer.Option(False, "--public", help="Public discovery"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n", help="Target namespace"),
):
    """Publish agent to namespaced registry."""
    publish_command(agent=agent, internal=internal, public=public, namespace=namespace)


@app.command("login")
def login(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Use API key instead of OAuth"),
):
    """Login to robutler.ai platform."""
    from .commands.auth import login_command
    login_command(api_key=api_key)


@app.command("whoami")
def whoami():
    """Show current authenticated user."""
    from .commands.auth import whoami_command
    whoami_command()


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
        # We can keep console logging or redirect based on command
        # For now, default logging setup in utils/logging.py is fine
        pass


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
