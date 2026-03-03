"""
Configuration commands.

webagents config get/set, sandbox
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Configuration management")
console = Console()


@app.callback(invoke_without_command=True)
def config(ctx: typer.Context):
    """Show current configuration."""
    if ctx.invoked_subcommand is not None:
        return
    
    console.print(Panel(
        "[bold]WebAgents Configuration[/bold]\n\n"
        "[dim]Global config: ~/.webagents/config.yml[/dim]\n"
        "[dim]Project config: .webagents/config.yml[/dim]",
        title="Config",
        border_style="cyan"
    ))
    # TODO: Show actual config


@app.command("get")
def get(
    key: str = typer.Argument(..., help="Config key to get"),
):
    """Get a config value."""
    console.print(f"[dim]Getting config: {key}[/dim]")
    # TODO: Get from config
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Config key"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Set a config value."""
    console.print(f"[cyan]Setting {key} = {value}[/cyan]")
    # TODO: Set in config
    console.print("[green]Config updated[/green]")


@app.command("edit")
def edit():
    """Open config in $EDITOR."""
    import os
    editor = os.environ.get("EDITOR", "vim")
    console.print(f"[cyan]Opening config in {editor}...[/cyan]")
    # TODO: Open config file in editor
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("reset")
def reset():
    """Reset configuration to defaults."""
    console.print("[yellow]Resetting config to defaults...[/yellow]")
    # TODO: Reset config
    console.print("[green]Config reset[/green]")


# Sandbox subcommands
sandbox_app = typer.Typer(help="Sandbox security configuration")
app.add_typer(sandbox_app, name="sandbox")


@sandbox_app.callback(invoke_without_command=True)
def sandbox(ctx: typer.Context):
    """Show sandbox status."""
    if ctx.invoked_subcommand is not None:
        return
    
    console.print(Panel(
        "[bold]Sandbox Configuration[/bold]\n\n"
        "Preset: [cyan]development[/cyan]\n"
        "Allowed folders: [dim].[/dim]\n"
        "Allowed commands: [dim]ls, cat, git status, ...[/dim]",
        title="Sandbox",
        border_style="cyan"
    ))


@sandbox_app.command("status")
def sandbox_status(
    agent: Optional[str] = typer.Argument(None, help="Agent name"),
):
    """Show sandbox status for agent."""
    if agent:
        console.print(f"[cyan]Sandbox status for: {agent}[/cyan]")
    else:
        console.print("[cyan]Global sandbox status[/cyan]")
    # TODO: Show actual sandbox config


@sandbox_app.command("allow-folder")
def allow_folder(
    path: str = typer.Argument(..., help="Folder path"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Per-agent setting"),
):
    """Add folder to allowlist."""
    console.print(f"[green]Allowing folder: {path}[/green]")
    if agent:
        console.print(f"[dim]For agent: {agent}[/dim]")
    # TODO: Update sandbox config


@sandbox_app.command("deny-folder")
def deny_folder(
    path: str = typer.Argument(..., help="Folder path"),
):
    """Remove folder from allowlist."""
    console.print(f"[yellow]Denying folder: {path}[/yellow]")
    # TODO: Update sandbox config


@sandbox_app.command("list-folders")
def list_folders():
    """List allowed folders."""
    console.print("[cyan]Allowed folders:[/cyan]")
    console.print("  .")
    # TODO: List actual allowed folders


@sandbox_app.command("allow-command")
def allow_command(
    cmd: str = typer.Argument(..., help="Command pattern"),
):
    """Allow shell command."""
    console.print(f"[green]Allowing command: {cmd}[/green]")
    # TODO: Update sandbox config


@sandbox_app.command("deny-command")
def deny_command(
    cmd: str = typer.Argument(..., help="Command pattern"),
):
    """Deny shell command."""
    console.print(f"[yellow]Denying command: {cmd}[/yellow]")
    # TODO: Update sandbox config


@sandbox_app.command("list-commands")
def list_commands():
    """List allowed commands."""
    console.print("[cyan]Allowed commands:[/cyan]")
    console.print("  ls, cat, head, tail, grep, git status, git log, git diff")
    # TODO: List actual allowed commands


@sandbox_app.command("allow-import")
def allow_import(
    module: str = typer.Argument(..., help="Python module"),
):
    """Allow Python import."""
    console.print(f"[green]Allowing import: {module}[/green]")
    # TODO: Update sandbox config


@sandbox_app.command("deny-import")
def deny_import(
    module: str = typer.Argument(..., help="Python module"),
):
    """Deny Python import."""
    console.print(f"[yellow]Denying import: {module}[/yellow]")
    # TODO: Update sandbox config


@sandbox_app.command("list-imports")
def list_imports():
    """List allowed imports."""
    console.print("[cyan]Allowed imports:[/cyan]")
    console.print("  json, re, datetime, pathlib, collections, typing, pydantic")
    # TODO: List actual allowed imports


@sandbox_app.command("preset")
def preset(
    name: str = typer.Argument(..., help="Preset name: strict, development, unrestricted"),
):
    """Apply sandbox preset."""
    if name not in ["strict", "development", "unrestricted"]:
        console.print(f"[red]Unknown preset: {name}[/red]")
        console.print("[dim]Available: strict, development, unrestricted[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Applying preset: {name}[/cyan]")
    # TODO: Apply preset
    console.print(f"[green]Preset '{name}' applied[/green]")
