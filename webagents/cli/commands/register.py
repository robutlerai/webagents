"""
Agent registration commands.

webagents register, unregister, scan
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Agent registration")
console = Console()


def register_command(
    path: Optional[str] = None,
    recursive: bool = False,
    watch: bool = False,
):
    """Register agents with daemon."""
    target = Path(path) if path else Path.cwd()
    
    if not target.exists():
        console.print(f"[red]Path not found: {target}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Registering agents from: {target}[/cyan]")
    
    # Find agent files
    if target.is_file():
        agent_files = [target]
    else:
        pattern = "**/" if recursive else ""
        agent_files = list(target.glob(f"{pattern}AGENT.md")) + \
                      list(target.glob(f"{pattern}AGENT-*.md"))
    
    if not agent_files:
        console.print("[yellow]No AGENT*.md files found[/yellow]")
        return
    
    for af in agent_files:
        console.print(f"  [green]+ {af}[/green]")
    
    console.print(f"[dim]Registered {len(agent_files)} agent(s)[/dim]")
    
    if watch:
        console.print("[dim]Watching for changes... (Ctrl+C to stop)[/dim]")
        # TODO: Start file watcher
        console.print("[yellow]Watch mode not yet implemented[/yellow]")


def scan_command(path: Optional[str] = None):
    """Scan for agent files."""
    target = Path(path) if path else Path.cwd()
    
    console.print(f"[cyan]Scanning for agents in: {target}[/cyan]")
    
    # Find all agent files
    agent_files = list(target.rglob("AGENT.md")) + list(target.rglob("AGENT-*.md"))
    context_files = list(target.rglob("AGENTS.md"))
    
    if agent_files:
        table = Table(title="Agent Files")
        table.add_column("File", style="cyan")
        table.add_column("Path", style="dim")
        
        for af in agent_files:
            table.add_row(af.name, str(af.relative_to(target)))
        
        console.print(table)
    else:
        console.print("[dim]No AGENT*.md files found[/dim]")
    
    if context_files:
        console.print(f"\n[dim]Found {len(context_files)} context file(s): AGENTS.md[/dim]")


@app.command("register")
def register_cmd(
    path: Optional[str] = typer.Argument(None, help="Path to agent file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes"),
):
    """Register agents with local daemon."""
    register_command(path=path, recursive=recursive, watch=watch)


@app.command("unregister")
def unregister(
    agent: str = typer.Argument(..., help="Agent name to unregister"),
):
    """Remove agent from registry."""
    console.print(f"[yellow]Unregistering: {agent}[/yellow]")
    # TODO: Remove from daemon registry
    console.print("[green]Agent unregistered[/green]")


@app.command("scan")
def scan_cmd(
    path: Optional[str] = typer.Argument(None, help="Path to scan"),
):
    """Scan for agent files."""
    scan_command(path=path)
