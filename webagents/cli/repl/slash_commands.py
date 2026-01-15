"""
Slash Command Registry

Handle /commands in the REPL.
"""

from typing import Callable, Dict, Any, Optional, TYPE_CHECKING
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

if TYPE_CHECKING:
    from .session import WebAgentsSession

console = Console()


class SlashCommandRegistry:
    """Registry for slash commands."""
    
    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in commands."""
        self.register("help", cmd_help, "Show available commands")
        self.register("exit", cmd_exit, "Exit the session")
        self.register("quit", cmd_exit, "Exit the session")
        self.register("clear", cmd_clear, "Clear screen and conversation")
        self.register("cls", cmd_screen, "Clear screen only")
        self.register("save", cmd_save, "Save session checkpoint")
        self.register("load", cmd_load, "Load session checkpoint")
        self.register("agent", cmd_agent, "Switch or show current agent")
        self.register("discover", cmd_discover, "Discover agents")
        self.register("mcp", cmd_mcp, "MCP server management")
        self.register("history", cmd_history, "Show conversation history")
        self.register("tokens", cmd_tokens, "Show token usage")
        self.register("config", cmd_config, "Show/edit configuration")
        self.register("model", cmd_model, "Show or change model")
    
    def register(self, name: str, handler: Callable, description: str = ""):
        """Register a slash command."""
        self.commands[name] = {
            "handler": handler,
            "description": description,
        }
    
    def get(self, name: str) -> Optional[Dict]:
        """Get command info."""
        return self.commands.get(name)
    
    def list_commands(self) -> Dict[str, Dict]:
        """List all commands."""
        return self.commands


def handle_slash_command(input_str: str, session: "WebAgentsSession") -> Optional[str]:
    """Handle a slash command."""
    parts = input_str[1:].split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    cmd_info = session.slash_commands.get(command)
    if cmd_info:
        return cmd_info["handler"](session, args)
    else:
        console.print(f"[yellow]Unknown command: /{command}[/yellow]")
        console.print("[dim]Type /help for available commands[/dim]")
        return None


# Built-in command handlers

def cmd_help(session: "WebAgentsSession", args: str) -> None:
    """Show help for commands."""
    table = Table(title="Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    for name, info in session.slash_commands.list_commands().items():
        table.add_row(f"/{name}", info["description"])
    
    console.print(table)


def cmd_exit(session: "WebAgentsSession", args: str) -> str:
    """Exit the session."""
    console.print("[dim]Goodbye![/dim]")
    return "exit"


def cmd_clear(session: "WebAgentsSession", args: str) -> None:
    """Clear screen and conversation."""
    console.clear()
    session.clear_history()
    from ..ui.splash import print_splash
    print_splash(console)
    console.print("[dim]Conversation cleared. Starting fresh.[/dim]\n")


def cmd_screen(session: "WebAgentsSession", args: str) -> None:
    """Clear the screen only."""
    console.clear()
    from ..ui.splash import print_splash
    print_splash(console)


def cmd_save(session: "WebAgentsSession", args: str) -> None:
    """Save session checkpoint."""
    checkpoint_name = args.strip() if args.strip() else "latest"
    console.print(f"[cyan]Saving checkpoint: {checkpoint_name}[/cyan]")
    # TODO: Actually save checkpoint
    console.print("[green]Checkpoint saved[/green]")


def cmd_load(session: "WebAgentsSession", args: str) -> None:
    """Load session checkpoint."""
    checkpoint_name = args.strip() if args.strip() else "latest"
    console.print(f"[cyan]Loading checkpoint: {checkpoint_name}[/cyan]")
    # TODO: Actually load checkpoint
    console.print("[yellow]No checkpoints found[/yellow]")


def cmd_agent(session: "WebAgentsSession", args: str) -> None:
    """Show or switch agent."""
    if args.strip():
        # Switch agent
        new_agent = args.strip()
        console.print(f"[cyan]Switching to agent: {new_agent}[/cyan]")
        # TODO: Actually switch agent
        console.print("[yellow]Agent switching not yet implemented[/yellow]")
    else:
        # Show current agent
        console.print(Panel(
            f"[bold]Current Agent: {session.agent_name}[/bold]\n\n"
            f"Path: {session.agent_path or 'default'}\n"
            f"[dim]Use /agent <name> to switch[/dim]",
            title="Agent",
            border_style="cyan"
        ))


def cmd_discover(session: "WebAgentsSession", args: str) -> None:
    """Discover agents."""
    if args.strip():
        intent = args.strip()
        console.print(f"[cyan]Searching for: {intent}[/cyan]")
        # TODO: Call discovery
        console.print("[dim]No agents found[/dim]")
    else:
        console.print("[dim]Usage: /discover <intent>[/dim]")
        console.print("[dim]Example: /discover summarize documents[/dim]")


def cmd_mcp(session: "WebAgentsSession", args: str) -> None:
    """MCP server management."""
    console.print(Panel(
        "[bold]MCP Servers[/bold]\n\n"
        "[dim]No MCP servers configured[/dim]\n\n"
        "Add servers in AGENT.md or config.",
        title="MCP",
        border_style="cyan"
    ))


def cmd_history(session: "WebAgentsSession", args: str) -> None:
    """Show conversation history."""
    if not session.messages:
        console.print("[dim]No conversation history yet.[/dim]")
        return
    
    console.print(f"[bold]Conversation History ({len(session.messages)} messages)[/bold]\n")
    for i, msg in enumerate(session.messages):
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


def cmd_tokens(session: "WebAgentsSession", args: str) -> None:
    """Show token usage."""
    console.print(Panel(
        f"[bold]Token Usage[/bold]\n\n"
        f"Input tokens:  {session.input_tokens:,}\n"
        f"Output tokens: {session.output_tokens:,}\n"
        f"Total:         {session.input_tokens + session.output_tokens:,}",
        title="Tokens",
        border_style="cyan"
    ))


def cmd_config(session: "WebAgentsSession", args: str) -> None:
    """Show configuration."""
    console.print(Panel(
        f"[bold]Configuration[/bold]\n\n"
        f"Model: {session.model}\n"
        f"Agent: {session.agent_name}\n"
        f"Messages: {len(session.messages)}",
        title="Config",
        border_style="cyan"
    ))


def cmd_model(session: "WebAgentsSession", args: str) -> None:
    """Show or change model."""
    if args.strip():
        new_model = args.strip()
        session.model = new_model
        session._agent_initialized = False  # Force agent reload
        session._agent = None
        console.print(f"[green]Model changed to: {new_model}[/green]")
        console.print("[dim]Note: Conversation history preserved.[/dim]")
    else:
        console.print(Panel(
            f"[bold]Current Model[/bold]\n\n"
            f"{session.model}\n\n"
            "[dim]Usage: /model <provider/model-name>[/dim]\n"
            "[dim]Examples:[/dim]\n"
            "[dim]  /model openai/gpt-4o[/dim]\n"
            "[dim]  /model anthropic/claude-3-5-sonnet-20241022[/dim]\n"
            "[dim]  /model google/gemini-2.5-flash[/dim]",
            title="Model",
            border_style="cyan"
        ))
