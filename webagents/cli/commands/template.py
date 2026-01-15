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
    is_bundled = template in [t[0] for t in BUNDLED_TEMPLATES]
    
    # Check if cached
    from ..state.local import get_state
    cache_dir = get_state().get_templates_dir()
    cached_template_file = cache_dir / template / "TEMPLATE.md"
    is_cached = cached_template_file.exists()
    
    if not is_bundled and not is_cached:
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
    
    # Check if it's a cached template
    from ..state.local import get_state
    cache_dir = get_state().get_templates_dir()
    cached_template_file = cache_dir / template / "TEMPLATE.md"
    
    if cached_template_file.exists():
        content = cached_template_file.read_text()
        # TODO: Simple variable replacement if needed
        # For now just use the content directly but update name
        if name:
            import re
            content = re.sub(r'name: .*', f'name: {name}', content, count=1)
        
        output_file.write_text(content)
        console.print(f"[green]Created {output_file.name} from cached template[/green]")
        if not keep and cached_template_file.exists():
             # We don't delete cached templates on use
             pass
        return

    # Fallback to bundled (mocked for now)
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
    branch: Optional[str] = typer.Option("main", "--branch", "-b", help="Branch name"),
    path: Optional[str] = typer.Option("TEMPLATE.md", "--path", "-p", help="Path within repo"),
    apply: bool = typer.Option(False, "--apply", "-a", help="Apply immediately after pulling"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Agent name (if applying)"),
):
    """Pull template from GitHub."""
    from ..templates import github
    import asyncio
    
    console.print(f"[cyan]Pulling template from: {url}[/cyan]")
    
    # Run async pull
    try:
        template_file = asyncio.run(github.pull_template(
            url=url,
            branch=branch,
            path=path
        ))
    except Exception as e:
        console.print(f"[red]Error pulling template: {e}[/red]")
        raise typer.Exit(1)
    
    if not template_file:
        console.print("[red]Failed to pull template. Check URL and path.[/red]")
        raise typer.Exit(1)
        
    console.print(f"[green]Template saved to cache: {template_file.parent.name}[/green]")
    
    if apply:
        template_name = template_file.parent.name
        use(template=template_name, name=name, keep=False)


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
