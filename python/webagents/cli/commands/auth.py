"""
Authentication commands.

webagents login, logout, whoami, token
"""

import asyncio
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from ..platform.auth import (
    login as platform_login,
    logout as platform_logout,
    is_authenticated,
    get_current_user,
)

app = typer.Typer(help="Authentication with robutler.ai")
console = Console()


@app.command("login")
def login(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Use API key (rok_*) for headless/CI"),
):
    """Login to robutler.ai platform (browser or --api-key)."""
    login_command(api_key=api_key)


@app.command("logout")
def logout():
    """Clear credentials and logout."""
    console.print("[cyan]Logging out...[/cyan]")
    platform_logout()
    console.print("[green]Successfully logged out[/green]")


@app.command("whoami")
def whoami():
    """Show current authenticated user."""
    whoami_command()


@app.command("token")
def token(
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Refresh token"),
):
    """Display or refresh current token."""
    if refresh:
        console.print("[yellow]JWT tokens expire in 7 days; run 'webagents login' again to get a new one.[/yellow]")
    elif is_authenticated():
        creds = __import__("webagents.cli.state.local", fromlist=["get_state"]).get_state().get_credentials()
        tok = creds.get("access_token", "")
        if tok:
            console.print(f"[dim]{tok[:20]}...[/dim]")
        else:
            console.print("[dim]No token found. Run 'webagents login' first.[/dim]")
    else:
        console.print("[dim]No token found. Run 'webagents login' first.[/dim]")


def login_command(api_key: Optional[str] = None) -> None:
    """Login to robutler.ai (browser or API key)."""
    try:
        result = asyncio.run(platform_login(api_key=api_key))
        username = result.get("username", "user")
        console.print(f"[green]Logged in as @{username}[/green]")
        console.print("[dim]Token stored in ~/.webagents/credentials.json (expires in 7 days)[/dim]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise typer.Exit(1)


def whoami_command() -> None:
    """Show current user from stored token or API."""
    if not is_authenticated():
        console.print(Panel(
            "[dim]Not logged in[/dim]\n\nRun [bold]webagents login[/bold] to connect to robutler.ai",
            title="Authentication",
            border_style="yellow",
        ))
        return
    try:
        user = asyncio.run(get_current_user())
        if user:
            username = user.get("username") or user.get("displayName") or "user"
            display = user.get("displayName") or username
            console.print(Panel(
                f"[bold]@{username}[/bold]\n{display}",
                title="Logged in",
                border_style="green",
            ))
        else:
            creds = __import__("webagents.cli.state.local", fromlist=["get_state"]).get_state().get_credentials()
            username = creds.get("username", "user")
            console.print(Panel(f"[bold]@{username}[/bold]", title="Logged in", border_style="green"))
    except Exception:
        creds = __import__("webagents.cli.state.local", fromlist=["get_state"]).get_state().get_credentials()
        username = creds.get("username", "user")
        console.print(Panel(f"[bold]@{username}[/bold]", title="Logged in", border_style="green"))
