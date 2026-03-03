"""
Session management commands.

webagents session new, history, save, load, clear
"""

import typer
import asyncio
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Session management")
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


async def _execute_session_command(subcommand: str, **kwargs):
    """Execute a session command via daemon."""
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
        
        result = await client.execute_command(agent_name, f"/session/{subcommand}", kwargs)
        return result
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()


@app.command("new")
def new(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Start a new session (clears history)."""
    result = asyncio.run(_execute_session_command("new", agent=agent))
    if result:
        console.print("[green]✓ Started new session[/green]")
        if isinstance(result, dict) and "session_id" in result:
            console.print(f"[dim]Session ID: {result['session_id']}[/dim]")


@app.command("history")
def history(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of messages to show"),
):
    """Show conversation history."""
    result = asyncio.run(_execute_session_command("history", agent=agent))
    
    if result and isinstance(result, dict):
        messages = result.get("messages", [])
        if not messages:
            console.print("[dim]No conversation history.[/dim]")
            return
        
        console.print(f"[bold]Session History ({len(messages)} messages)[/bold]\n")
        
        for msg in messages[-limit:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            
            if role == "user":
                console.print(f"[cyan]You:[/cyan] {content}")
            elif role == "assistant":
                console.print(f"[green]Assistant:[/green] {content}")
            else:
                console.print(f"[dim]{role}:[/dim] {content}")


@app.command("save")
def save(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (optional)"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Save current session."""
    kwargs = {"agent": agent}
    if session_id:
        kwargs["session_id"] = session_id
    
    result = asyncio.run(_execute_session_command("save", **kwargs))
    if result:
        console.print("[green]✓ Session saved[/green]")
        if isinstance(result, dict) and "session_id" in result:
            console.print(f"[dim]Session ID: {result['session_id']}[/dim]")


@app.command("load")
def load(
    session_id: str = typer.Argument(..., help="Session ID to load"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Load a previous session."""
    result = asyncio.run(_execute_session_command("load", agent=agent, session_id=session_id))
    if result:
        console.print(f"[green]✓ Loaded session: {session_id}[/green]")
        if isinstance(result, dict) and "messages" in result:
            console.print(f"[dim]{len(result['messages'])} messages restored[/dim]")


@app.command("list")
def list_sessions(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """List saved sessions."""
    result = asyncio.run(_execute_session_command("list", agent=agent))
    
    if result and isinstance(result, dict):
        sessions = result.get("sessions", [])
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
            return
        
        table = Table(title="Saved Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Created")
        table.add_column("Messages")
        table.add_column("Tokens")
        
        for session in sessions:
            table.add_row(
                session.get("id", "?"),
                session.get("created", "?"),
                str(session.get("message_count", 0)),
                str(session.get("total_tokens", 0)),
            )
        
        console.print(table)


@app.command("clear")
def clear(
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Clear current session history."""
    result = asyncio.run(_execute_session_command("clear", agent=agent))
    if result:
        console.print("[green]✓ Session cleared[/green]")
