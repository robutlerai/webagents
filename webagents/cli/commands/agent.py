"""
Agent lifecycle commands.

webagents run, connect, list, stop, logs, debug, info
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Agent lifecycle management")
console = Console()


def _find_default_agent(path: Path) -> Optional[Path]:
    """Find default agent using resolution rules."""
    # 1. Check for AGENT.md
    agent_md = path / "AGENT.md"
    if agent_md.exists():
        return agent_md
    
    # 2. Check for single AGENT-*.md
    agent_files = list(path.glob("AGENT-*.md"))
    if len(agent_files) == 1:
        return agent_files[0]
    elif len(agent_files) > 1:
        return None  # Multiple agents, can't determine default
    
    return None


def _resolve_agent(agent_id: Optional[str]) -> Optional[Path]:
    """Resolve agent from name or path."""
    if agent_id is None:
        return _find_default_agent(Path.cwd())
    
    # Check if it's a path
    path = Path(agent_id)
    if path.exists():
        return path
    
    # Check if it's relative to current directory
    cwd_path = Path.cwd() / agent_id
    if cwd_path.exists():
        return cwd_path
    
    # Try as AGENT-<name>.md
    agent_file = Path.cwd() / f"AGENT-{agent_id}.md"
    if agent_file.exists():
        return agent_file
    
    # TODO: Look up in registry
    return None


@app.command("run")
def run(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path"),
    prompt: Optional[str] = typer.Option(None, "-p", "--prompt", help="Single prompt, exit after"),
    all_agents: bool = typer.Option(False, "--all", help="Run all agents in directory"),
):
    """Run an agent (headless execution)."""
    if all_agents:
        agent_files = list(Path.cwd().glob("AGENT*.md"))
        if not agent_files:
            console.print("[red]No AGENT*.md files found in current directory[/red]")
            raise typer.Exit(1)
        
        console.print(f"[cyan]Running {len(agent_files)} agent(s)...[/cyan]")
        for af in agent_files:
            console.print(f"  - {af.name}")
        # TODO: Actually run agents
        return
    
    agent_path = _resolve_agent(agent)
    if agent_path is None:
        if agent:
            console.print(f"[red]Agent not found: {agent}[/red]")
        else:
            console.print("[red]No AGENT.md or AGENT-*.md found in current directory[/red]")
            console.print("[dim]Use 'webagents init' to create one, or specify an agent path.[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Running agent: {agent_path.name}[/cyan]")
    
    if prompt:
        # Single prompt mode
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        # TODO: Load agent and execute single prompt
        console.print("[yellow]Single prompt execution not yet implemented[/yellow]")
    else:
        # Headless mode
        # TODO: Load agent and run in headless mode
        console.print("[yellow]Headless execution not yet implemented[/yellow]")


@app.command("stop")
def stop(
    agent: str = typer.Argument(..., help="Agent to stop"),
):
    """Stop a running agent."""
    console.print(f"[cyan]Stopping agent: {agent}[/cyan]")
    # TODO: Implement stop via daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("restart")
def restart(
    agent: str = typer.Argument(..., help="Agent to restart"),
):
    """Restart an agent."""
    console.print(f"[cyan]Restarting agent: {agent}[/cyan]")
    # TODO: Implement restart via daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("logs")
def logs(
    agent: str = typer.Argument(..., help="Agent name"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream logs"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines"),
):
    """View agent logs."""
    console.print(f"[cyan]Logs for agent: {agent}[/cyan]")
    if follow:
        console.print("[dim]Following logs... (Ctrl+C to exit)[/dim]")
    # TODO: Read logs from .webagents/logs/<agent>/
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("debug")
def debug(
    agent: str = typer.Argument(..., help="Agent to debug"),
):
    """Start interactive debug mode for an agent."""
    console.print(f"[cyan]Debug mode for agent: {agent}[/cyan]")
    # TODO: Launch debug REPL
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("info")
def info(
    agent: str = typer.Argument(..., help="Agent name or path"),
):
    """Show detailed agent information."""
    agent_path = _resolve_agent(agent)
    if agent_path is None:
        console.print(f"[red]Agent not found: {agent}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]Agent: {agent_path.stem}[/bold]\n"
        f"Path: {agent_path}\n"
        f"[dim]Use 'webagents connect {agent}' to start a session[/dim]",
        title="Agent Info",
        border_style="cyan"
    ))
    # TODO: Parse AGENT.md and show full info


# Command functions used by main.py
def connect_command(agent: Optional[str] = None):
    """Start interactive REPL session."""
    agent_path = _resolve_agent(agent)
    
    if agent_path is None and agent:
        console.print(f"[red]Agent not found: {agent}[/red]")
        raise typer.Exit(1)
    
    # Start REPL
    from ..repl.session import start_repl
    start_repl(agent_path=agent_path)


def list_command(
    local: bool = False,
    remote: bool = False,
    running: bool = False,
    namespace: Optional[str] = None
):
    """List registered agents."""
    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Namespace", style="dim")
    table.add_column("Status", style="green")
    table.add_column("Source", style="dim")
    
    # Scan current directory for agents
    cwd = Path.cwd()
    agent_files = list(cwd.glob("AGENT.md")) + list(cwd.glob("AGENT-*.md"))
    
    for af in agent_files:
        name = af.stem if af.stem != "AGENT" else "default"
        table.add_row(
            name,
            "local",
            "idle",
            str(af.relative_to(cwd))
        )
    
    if len(agent_files) == 0:
        console.print("[dim]No agents found in current directory.[/dim]")
        console.print("[dim]Use 'webagents init' to create one.[/dim]")
    else:
        console.print(table)


def init_command(
    name: Optional[str] = None,
    template_name: Optional[str] = None,
    context: bool = False
):
    """Initialize a new agent."""
    if context:
        # Create AGENTS.md context file
        agents_md = Path.cwd() / "AGENTS.md"
        if agents_md.exists():
            console.print("[yellow]AGENTS.md already exists[/yellow]")
            raise typer.Exit(1)
        
        content = """---
# Context inherited by all AGENT*.md files in this directory
namespace: local
---

# Project Context

This file provides context for all agents in this directory.

## Guidelines

- Be helpful and concise
- Follow best practices
- Ask for clarification when needed
"""
        agents_md.write_text(content)
        console.print(f"[green]Created {agents_md}[/green]")
        return
    
    # Create AGENT.md or AGENT-<name>.md
    if name:
        filename = f"AGENT-{name}.md"
    else:
        filename = "AGENT.md"
    
    agent_path = Path.cwd() / filename
    if agent_path.exists():
        console.print(f"[yellow]{filename} already exists[/yellow]")
        raise typer.Exit(1)
    
    # Use template if specified
    if template_name:
        console.print(f"[cyan]Using template: {template_name}[/cyan]")
        # TODO: Load and apply template
    
    content = f"""---
name: {name or 'assistant'}
description: A helpful AI assistant
namespace: local
model: openai/gpt-4o-mini
intents:
  - answer questions
  - help with tasks
skills: []
visibility: local
---

# {name.title() if name else 'Assistant'} Agent

You are a helpful AI assistant.

## Capabilities

- Answer questions
- Help with various tasks
- Provide information

## Guidelines

- Be helpful and concise
- Ask for clarification when needed
"""
    agent_path.write_text(content)
    console.print(f"[green]Created {agent_path}[/green]")
    console.print(f"[dim]Edit the file to customize your agent.[/dim]")
    console.print(f"[dim]Run 'webagents connect' to start a session.[/dim]")
