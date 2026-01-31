"""
Checkpoint management commands.

webagents checkpoint create, restore, list, info, delete
"""

import typer
import asyncio
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

app = typer.Typer(help="Checkpoint management (file snapshots)")
console = Console()


def _get_daemon_client():
    """Get daemon client for command execution."""
    from ..client.daemon_client import DaemonClient
    return DaemonClient()


def _resolve_agent() -> Optional[str]:
    """Resolve current agent from environment or cwd."""
    from pathlib import Path
    
    # Check for AGENT.md in cwd
    cwd = Path.cwd()
    agent_md = cwd / "AGENT.md"
    if agent_md.exists():
        return "default"
    
    # Check for single AGENT-*.md
    agent_files = list(cwd.glob("AGENT-*.md"))
    if len(agent_files) == 1:
        return agent_files[0].stem.replace("AGENT-", "")
    
    return None


async def _execute_checkpoint_command(subcommand: str, **kwargs):
    """Execute a checkpoint command via daemon."""
    client = _get_daemon_client()
    
    try:
        if not await client.is_running():
            console.print("[yellow]webagentsd is not running. Starting...[/yellow]")
            from ..daemon.manager import start_daemon
            await start_daemon()
        
        agent_name = kwargs.pop("agent", None) or _resolve_agent()
        if not agent_name:
            console.print("[red]No agent specified. Use --agent or run from an agent directory.[/red]")
            return
        
        result = await client.execute_command(agent_name, f"/checkpoint/{subcommand}", kwargs)
        return result
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()


@app.command("create")
def create(
    description: Optional[str] = typer.Argument(None, help="Checkpoint description"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Create a new checkpoint (snapshot of files)."""
    kwargs = {"agent": agent}
    if description:
        kwargs["description"] = description
    
    result = asyncio.run(_execute_checkpoint_command("create", **kwargs))
    if result:
        console.print("[green]✓ Checkpoint created[/green]")
        if isinstance(result, dict):
            if "id" in result:
                console.print(f"[dim]ID: {result['id']}[/dim]")
            if "description" in result:
                console.print(f"[dim]Description: {result['description']}[/dim]")


@app.command("restore")
def restore(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to restore"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Restore files from a checkpoint."""
    result = asyncio.run(_execute_checkpoint_command("restore", agent=agent, checkpoint_id=checkpoint_id))
    if result:
        console.print(f"[green]✓ Restored checkpoint: {checkpoint_id}[/green]")
        if isinstance(result, dict) and "files_restored" in result:
            console.print(f"[dim]{result['files_restored']} files restored[/dim]")


@app.command("list")
def list_checkpoints(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of checkpoints to show"),
):
    """List available checkpoints."""
    result = asyncio.run(_execute_checkpoint_command("list", agent=agent, limit=limit))
    
    if result and isinstance(result, dict):
        checkpoints = result.get("checkpoints", [])
        if not checkpoints:
            console.print("[dim]No checkpoints found.[/dim]")
            console.print("[dim]Use 'webagents checkpoint create' to create one.[/dim]")
            return
        
        table = Table(title="Checkpoints")
        table.add_column("ID", style="cyan", max_width=12)
        table.add_column("Description")
        table.add_column("Created")
        table.add_column("Files")
        
        for cp in checkpoints:
            created = cp.get("created", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created)
                    created = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            table.add_row(
                cp.get("id", "?")[:12],
                cp.get("description", "-"),
                created,
                str(cp.get("file_count", 0)),
            )
        
        console.print(table)


@app.command("info")
def info(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Show detailed checkpoint information."""
    result = asyncio.run(_execute_checkpoint_command("info", agent=agent, checkpoint_id=checkpoint_id))
    
    if result and isinstance(result, dict):
        console.print(Panel(
            f"[bold]Checkpoint: {result.get('id', '?')}[/bold]\n\n"
            f"Description: {result.get('description', 'No description')}\n"
            f"Created: {result.get('created', '?')}\n"
            f"Files: {result.get('file_count', 0)}\n"
            f"Commit: {result.get('commit', '?')[:12] if result.get('commit') else '?'}",
            title="Checkpoint Info",
            border_style="cyan"
        ))
        
        # Show files if available
        files = result.get("files", [])
        if files:
            console.print("\n[bold]Files:[/bold]")
            for f in files[:20]:  # Limit display
                console.print(f"  {f}")
            if len(files) > 20:
                console.print(f"  [dim]... and {len(files) - 20} more[/dim]")


@app.command("delete")
def delete(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to delete"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a checkpoint."""
    if not force:
        confirm = typer.confirm(f"Delete checkpoint {checkpoint_id}?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return
    
    result = asyncio.run(_execute_checkpoint_command("delete", agent=agent, checkpoint_id=checkpoint_id))
    if result:
        console.print(f"[green]✓ Deleted checkpoint: {checkpoint_id}[/green]")
