"""
Splash Screen

Beautiful ASCII art banner like gemini-cli.
"""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel


# Block-style logo (like gemini-cli)
WEBAGENTS_LOGO_BLOCK = """
██╗    ██╗███████╗██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗
██║    ██║██╔════╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝
██║ █╗ ██║█████╗  ██████╔╝███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗
██║███╗██║██╔══╝  ██╔══██╗██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
╚███╔███╔╝███████╗██████╔╝██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║
 ╚══╝╚══╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
"""

# Double-line style logo (alternative)
WEBAGENTS_LOGO_DOUBLE = """
╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗
║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗
╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝
"""

# Gradient colors (cyan -> magenta like gemini-cli)
GRADIENT_COLORS = [
    "cyan",
    "dodger_blue2", 
    "blue_violet",
    "medium_orchid",
    "magenta",
]


def print_splash(console: Console, style: str = "auto"):
    """Print colorful gradient splash screen.
    
    Args:
        console: Rich Console instance
        style: Logo style - "block", "double", or "auto"
    """
    
    # Auto-detect appropriate size
    if style == "auto":
        style = "double" if console.width < 80 else "block"
        
    logo = WEBAGENTS_LOGO_BLOCK if style == "block" else WEBAGENTS_LOGO_DOUBLE
    lines = logo.strip().split('\n')
    
    # Print each line with gradient color
    for i, line in enumerate(lines):
        color = GRADIENT_COLORS[i % len(GRADIENT_COLORS)]
        console.print(Text(line, style=f"bold {color}"))
    
    # Tips
    console.print("[dim]Tips for getting started:[/dim]")
    console.print("[dim]1. Ask questions, edit files, or run commands.[/dim]")
    console.print("[dim]2. Be specific for the best results.[/dim]")
    console.print("[dim]3. /help for more information.[/dim]")


def print_status_bar(console: Console, agent: str, sandbox: str, mode: str):
    """Print bottom status bar.
    
    Args:
        console: Rich Console instance
        agent: Current agent name
        sandbox: Sandbox status (enabled/disabled)
        mode: Current mode (auto/manual)
    """
    import os
    
    # Get current directory (shortened)
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    
    # Sandbox color
    sandbox_color = "green" if sandbox == "enabled" else "yellow"
    
    # Build status line
    status = (
        f"[dim]{cwd}[/dim]  "
        f"[cyan]{agent}[/cyan]  "
        f"[{sandbox_color}]sandbox {sandbox}[/{sandbox_color}]  "
        f"[dim]{mode}[/dim]"
    )
    
    console.print(status)


def print_agent_info(console: Console, agent_count: int, mcp_count: int):
    """Print agent/MCP info line.
    
    Args:
        console: Rich Console instance
        agent_count: Number of loaded agents
        mcp_count: Number of MCP servers
    """
    parts = []
    if agent_count > 0:
        parts.append(f"{agent_count} AGENT.md file{'s' if agent_count > 1 else ''}")
    if mcp_count > 0:
        parts.append(f"{mcp_count} MCP server{'s' if mcp_count > 1 else ''}")
    
    if parts:
        console.print(f"[dim]Using: {' | '.join(parts)}[/dim]")
    console.print()


def print_welcome_panel(console: Console, agent_name: str = None):
    """Print a welcome panel with agent info."""
    content = "[bold]Welcome to WebAgents[/bold]\n\n"
    
    if agent_name:
        content += f"Agent: [cyan]{agent_name}[/cyan]\n"
    else:
        content += "No agent loaded\n"
    
    content += "\n[dim]Type a message or use /help for commands[/dim]"
    
    console.print(Panel(
        content,
        border_style="cyan",
        padding=(1, 2),
    ))
