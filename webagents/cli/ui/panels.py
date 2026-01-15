"""
UI Panels and Components

Reusable panel components for the CLI.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def agent_panel(name: str, description: str, namespace: str = "local",
                status: str = "idle", intents: list = None) -> Panel:
    """Create an agent info panel."""
    content = f"[bold]{name}[/bold]\n\n"
    content += f"{description}\n\n"
    content += f"Namespace: [cyan]{namespace}[/cyan]\n"
    content += f"Status: [{_status_color(status)}]{status}[/{_status_color(status)}]"
    
    if intents:
        content += "\n\nIntents:"
        for intent in intents[:5]:
            content += f"\n  • {intent}"
    
    return Panel(content, title="Agent", border_style="cyan")


def config_panel(config: dict) -> Panel:
    """Create a configuration panel."""
    lines = []
    for key, value in config.items():
        lines.append(f"{key}: [cyan]{value}[/cyan]")
    
    return Panel(
        "\n".join(lines),
        title="Configuration",
        border_style="dim"
    )


def sandbox_panel(preset: str, allowed_folders: list, allowed_commands: list) -> Panel:
    """Create a sandbox status panel."""
    content = f"[bold]Preset: {preset}[/bold]\n\n"
    
    content += "[dim]Allowed Folders:[/dim]\n"
    for folder in allowed_folders[:5]:
        content += f"  • {folder}\n"
    
    content += "\n[dim]Allowed Commands:[/dim]\n"
    for cmd in allowed_commands[:5]:
        content += f"  • {cmd}\n"
    
    return Panel(content, title="Sandbox", border_style="yellow")


def namespace_tree(namespaces: list) -> Tree:
    """Create a namespace tree view."""
    tree = Tree("[bold]Namespaces[/bold]")
    
    for ns in namespaces:
        ns_node = tree.add(f"[cyan]{ns['name']}[/cyan]")
        ns_node.add(f"Type: {ns.get('type', 'user')}")
        ns_node.add(f"Role: {ns.get('role', 'member')}")
        if 'agents' in ns:
            agents_node = ns_node.add(f"Agents ({len(ns['agents'])})")
            for agent in ns['agents'][:5]:
                agents_node.add(f"[dim]{agent}[/dim]")
    
    return tree


def intent_table(intents: list) -> Table:
    """Create an intents table."""
    table = Table(title="Published Intents")
    table.add_column("Intent", style="cyan")
    table.add_column("Agent")
    table.add_column("Visibility", style="dim")
    table.add_column("Namespace", style="dim")
    
    for intent in intents:
        table.add_row(
            intent.get("text", ""),
            intent.get("agent", ""),
            intent.get("visibility", "local"),
            intent.get("namespace", "local"),
        )
    
    return table


def subscription_table(subscriptions: list) -> Table:
    """Create a subscriptions table."""
    table = Table(title="Intent Subscriptions")
    table.add_column("ID", style="dim")
    table.add_column("Intent", style="cyan")
    table.add_column("Route To")
    table.add_column("Status", style="green")
    
    for sub in subscriptions:
        table.add_row(
            sub.get("id", ""),
            sub.get("intent", ""),
            sub.get("route_to", ""),
            sub.get("status", "active"),
        )
    
    return table


def cron_table(jobs: list) -> Table:
    """Create a cron jobs table."""
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="dim")
    table.add_column("Agent", style="cyan")
    table.add_column("Schedule")
    table.add_column("Next Run", style="dim")
    table.add_column("Status", style="green")
    
    for job in jobs:
        table.add_row(
            job.get("id", ""),
            job.get("agent", ""),
            job.get("schedule", ""),
            job.get("next_run", "-"),
            job.get("status", "active"),
        )
    
    return table


def _status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "idle": "dim",
        "running": "green",
        "error": "red",
        "paused": "yellow",
        "pending": "blue",
    }
    return colors.get(status, "dim")
