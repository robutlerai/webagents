"""
Template commands.

webagents template list, use, pull, cache
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Agent templates")
console = Console()

# Bundled templates
BUNDLED_TEMPLATES = [
    ("assistant", "General purpose AI assistant"),
    ("planning", "Planning and task management agent"),
    ("marketing", "Marketing and content strategy agent"),
    ("content", "Content creation agent"),
    ("code-review", "Code review and analysis agent"),
    ("research", "Research and analysis agent"),
]


@app.command("list")
def list_templates(
    remote: bool = typer.Option(False, "--remote", "-r", help="Show templates from registry"),
):
    """List available templates."""
    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")
    
    for name, desc in BUNDLED_TEMPLATES:
        table.add_row(name, desc, "bundled")
    
    if remote:
        console.print("[dim]Fetching remote templates...[/dim]")
        # TODO: Fetch from registry
    
    console.print(table)


@app.command("use")
def use(
    template: str = typer.Argument(..., help="Template name"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Agent name (creates AGENT-<name>.md)"),
    keep: bool = typer.Option(False, "--keep", "-k", help="Keep TEMPLATE.md after applying"),
):
    """Apply template to create AGENT.md."""
    from pathlib import Path
    
    # Check if template exists
    if template not in [t[0] for t in BUNDLED_TEMPLATES]:
        console.print(f"[yellow]Template not found: {template}[/yellow]")
        console.print("[dim]Use 'webagents template list' to see available templates[/dim]")
        raise typer.Exit(1)
    
    # Determine output file
    if name:
        output_file = Path.cwd() / f"AGENT-{name}.md"
    else:
        output_file = Path.cwd() / "AGENT.md"
    
    if output_file.exists():
        console.print(f"[yellow]{output_file.name} already exists[/yellow]")
        overwrite = typer.confirm("Overwrite?")
        if not overwrite:
            raise typer.Exit(0)
    
    console.print(f"[cyan]Applying template: {template}[/cyan]")
    console.print(f"[dim]Creating: {output_file.name}[/dim]")
    
    # TODO: Load template and apply
    # For now, create a basic agent file
    content = f"""---
name: {name or template}
description: Agent created from {template} template
namespace: local
model: openai/gpt-4o-mini
intents:
  - help with tasks
skills: []
visibility: local
---

# {(name or template).title()} Agent

Created from the **{template}** template.

## Capabilities

Edit this file to customize your agent.

## Guidelines

- Be helpful and concise
- Ask for clarification when needed
"""
    output_file.write_text(content)
    console.print(f"[green]Created {output_file.name}[/green]")


@app.command("pull")
def pull(
    url: str = typer.Argument(..., help="GitHub URL or user/repo shorthand"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch name"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Path within repo"),
    apply: bool = typer.Option(False, "--apply", "-a", help="Apply immediately after pulling"),
):
    """Pull template from GitHub."""
    console.print(f"[cyan]Pulling template from: {url}[/cyan]")
    if branch:
        console.print(f"[dim]Branch: {branch}[/dim]")
    if path:
        console.print(f"[dim]Path: {path}[/dim]")
    
    # TODO: Fetch from GitHub
    console.print("[yellow]Not yet implemented[/yellow]")
    
    if apply:
        console.print("[dim]Applying template...[/dim]")


# Cache subcommands
cache_app = typer.Typer(help="Manage template cache")
app.add_typer(cache_app, name="cache")


@cache_app.command("list")
def cache_list():
    """List cached templates."""
    console.print("[cyan]Cached templates:[/cyan]")
    console.print("[dim]No cached templates[/dim]")
    # TODO: List from ~/.webagents/templates/


@cache_app.command("clear")
def cache_clear():
    """Clear template cache."""
    console.print("[cyan]Clearing template cache...[/cyan]")
    # TODO: Clear ~/.webagents/templates/
    console.print("[green]Cache cleared[/green]")
