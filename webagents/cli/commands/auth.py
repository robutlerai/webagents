"""
Authentication commands.

webagents login, logout, whoami, token
"""

import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Authentication with robutler.ai")
console = Console()


@app.command("login")
def login(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Use API key instead of OAuth"),
):
    """Login to robutler.ai platform."""
    login_command(api_key=api_key)


@app.command("logout")
def logout():
    """Clear credentials and logout."""
    console.print("[cyan]Logging out...[/cyan]")
    # TODO: Clear stored credentials
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
        console.print("[cyan]Refreshing token...[/cyan]")
        # TODO: Refresh token
        console.print("[yellow]Not yet implemented[/yellow]")
    else:
        console.print("[dim]No token found. Run 'webagents login' first.[/dim]")


# Command functions used by main.py
def login_command(api_key: Optional[str] = None):
    """Login to robutler.ai."""
    if api_key:
        console.print("[cyan]Authenticating with API key...[/cyan]")
        # TODO: Validate and store API key
        console.print("[yellow]API key auth not yet implemented[/yellow]")
    else:
        console.print("[cyan]Opening browser for authentication...[/cyan]")
        console.print("[dim]Complete the login in your browser.[/dim]")
        # TODO: OAuth flow with robutler.ai
        console.print("[yellow]OAuth not yet implemented[/yellow]")


def whoami_command():
    """Show current user."""
    # TODO: Check if authenticated and show user info
    console.print(Panel(
        "[dim]Not logged in[/dim]\n\n"
        "Run 'webagents login' to connect to robutler.ai",
        title="Authentication",
        border_style="yellow"
    ))
