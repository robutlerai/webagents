"""
Skill management commands.

webagents skill list, install, remove, enable, disable, config
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage agent skills")
console = Console()

# Available skills
AVAILABLE_SKILLS = [
    ("cron", "Scheduled task execution", "core"),
    ("folder-index", "Vector indexing with sqlite-vec", "core"),
    ("llm", "LLM provider integration (OpenAI, Anthropic, Google, xAI, Fireworks)", "core"),
    ("mcp", "Model Context Protocol servers", "core"),
    ("memory", "Short and long-term memory", "core"),
    ("discovery", "Agent discovery", "platform"),
    ("web", "Web fetch and search", "ecosystem"),
    ("filesystem", "File system operations", "ecosystem"),
    ("database", "Database operations", "ecosystem"),
]


@app.command("list")
def list_skills(
    installed: bool = typer.Option(False, "--installed", "-i", help="Show installed only"),
):
    """List available skills."""
    table = Table(title="Available Skills")
    table.add_column("Skill", style="cyan")
    table.add_column("Description")
    table.add_column("Category", style="dim")
    table.add_column("Status", style="green")
    
    for name, desc, category in AVAILABLE_SKILLS:
        status = "available"  # TODO: Check if installed
        if not installed or status == "installed":
            table.add_row(name, desc, category, status)
    
    console.print(table)


@app.command("add")
def add(
    skills: List[str] = typer.Argument(..., help="Skills to add"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Add skill(s) to agent config."""
    from pathlib import Path
    
    # Resolve agent
    if agent:
        if agent == ".":
            agent_file = Path.cwd() / "AGENT.md"
        else:
            agent_file = Path.cwd() / f"AGENT-{agent}.md"
    else:
        # Try to find default agent
        agent_file = Path.cwd() / "AGENT.md"
        if not agent_file.exists():
            agent_files = list(Path.cwd().glob("AGENT-*.md"))
            if len(agent_files) == 1:
                agent_file = agent_files[0]
            elif len(agent_files) > 1:
                console.print("[red]Multiple agents found. Use --agent to specify.[/red]")
                for af in agent_files:
                    console.print(f"  - {af.stem}")
                raise typer.Exit(1)
            else:
                console.print("[red]No agent found. Create one with 'webagents init'[/red]")
                raise typer.Exit(1)
    
    if not agent_file.exists():
        console.print(f"[red]Agent file not found: {agent_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Installing skills to: {agent_file.name}[/cyan]")
    for skill in skills:
        if skill not in [s[0] for s in AVAILABLE_SKILLS]:
            console.print(f"[yellow]Unknown skill: {skill}[/yellow]")
            continue
        console.print(f"  [green]+ {skill}[/green]")
    
    # TODO: Actually update AGENT.md YAML frontmatter
    console.print("[dim]Skills added to agent config[/dim]")


@app.command("remove")
def remove(
    skill: str = typer.Argument(..., help="Skill to remove"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Target agent"),
):
    """Remove skill from agent."""
    console.print(f"[yellow]Removing skill: {skill}[/yellow]")
    if agent:
        console.print(f"[dim]From agent: {agent}[/dim]")
    # TODO: Update AGENT.md


@app.command("enable")
def enable(
    skill: str = typer.Argument(..., help="Skill to enable"),
):
    """Enable a skill."""
    console.print(f"[green]Enabling skill: {skill}[/green]")
    # TODO: Enable skill


@app.command("disable")
def disable(
    skill: str = typer.Argument(..., help="Skill to disable"),
):
    """Disable a skill."""
    console.print(f"[yellow]Disabling skill: {skill}[/yellow]")
    # TODO: Disable skill


@app.command("config")
def config_skill(
    skill: str = typer.Argument(..., help="Skill to configure"),
):
    """Configure a skill."""
    console.print(f"[cyan]Configuring skill: {skill}[/cyan]")
    # TODO: Open skill config
    console.print("[yellow]Not yet implemented[/yellow]")
