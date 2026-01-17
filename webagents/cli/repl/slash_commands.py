"""
Slash Command Registry

Handle /commands in the REPL. Supports hierarchical subcommands like /checkpoint restore.
"""

from typing import Callable, Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import inspect

if TYPE_CHECKING:
    from .session import WebAgentsSession

console = Console()


class SlashCommandRegistry:
    """Registry for slash commands with hierarchical subcommand support."""
    
    def __init__(self):
        self.commands: Dict[str, Dict] = {}
        self.agent_commands: Dict[str, Dict] = {}  # Commands from connected agent
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in commands."""
        self.register("help", cmd_help, "Show available commands")
        self.register("exit", cmd_exit, "Exit the session")
        self.register("quit", cmd_exit, "Exit the session")
        self.register("clear", cmd_clear, "Clear screen and conversation")
        self.register("cls", cmd_screen, "Clear screen only")
        self.register("new", cmd_new, "Start a new session")
        self.register("agent", cmd_agent, "Switch or show current agent")
        self.register("discover", cmd_discover, "Discover agents")
        self.register("mcp", cmd_mcp, "MCP server management")
        self.register("history", cmd_history, "Show conversation history")
        self.register("tokens", cmd_tokens, "Show token usage")
        self.register("config", cmd_config, "Show/edit configuration")
        self.register("model", cmd_model, "Show or change model")
        
        # Daemon management commands
        self.register("status", cmd_daemon_status, "Show daemon status")
        self.register("list", cmd_list_agents, "List registered agents")
        self.register("register", cmd_register_agent, "Register agent with daemon")
        self.register("run", cmd_run_agent, "Run a registered agent")
    
    def register(self, name: str, handler: Callable, description: str = "", scope: str = "all"):
        """Register a slash command.
        
        Args:
            name: Command name or path (e.g., "help" or "checkpoint/create")
            handler: Command handler function
            description: Command description
            scope: Access scope (all, owner, admin)
        """
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "scope": scope,
        }
    
    def register_agent_commands(self, commands: List[Dict[str, Any]]):
        """Register commands from connected agent.
        
        Args:
            commands: List of command info dicts from agent
        """
        self.agent_commands.clear()
        for cmd in commands:
            path = cmd.get("path", "").lstrip("/")
            self.agent_commands[path] = {
                "path": cmd.get("path"),
                "alias": cmd.get("alias"),
                "description": cmd.get("description", ""),
                "scope": cmd.get("scope", "all"),
                "parameters": cmd.get("parameters", {}),
                "required": cmd.get("required", []),
                "is_agent_command": True,
            }
    
    def get(self, name: str) -> Optional[Dict]:
        """Get command info by name or path."""
        # Check built-in commands first
        if name in self.commands:
            return self.commands[name]
        
        # Check agent commands (with or without leading /)
        clean_name = name.lstrip("/")
        if clean_name in self.agent_commands:
            return self.agent_commands[clean_name]
        
        return None
    
    def get_subcommands(self, prefix: str) -> List[Dict[str, Any]]:
        """Get subcommands for a command prefix.
        
        Args:
            prefix: Command prefix (e.g., "checkpoint")
            
        Returns:
            List of subcommand info dicts
        """
        subcommands = []
        prefix_with_slash = f"{prefix}/"
        
        for path, cmd in self.agent_commands.items():
            if path.startswith(prefix_with_slash) or path == prefix:
                subcommands.append({
                    "path": path,
                    "description": cmd.get("description", ""),
                })
        
        return subcommands
    
    def list_commands(self) -> Dict[str, Dict]:
        """List all commands (built-in and agent)."""
        all_commands = dict(self.commands)
        for path, cmd in self.agent_commands.items():
            all_commands[path] = cmd
        return all_commands


async def handle_slash_command(input_str: str, session: "WebAgentsSession") -> Optional[str]:
    """Handle a slash command with subcommand support.
    
    Supports:
    - /help - Built-in help
    - /checkpoint - Show subcommands (if registered)
    - /checkpoint create - Execute subcommand
    - /checkpoint restore <args> - Execute with arguments
    """
    # Parse command and arguments
    parts = input_str[1:].split()
    if not parts:
        console.print("[yellow]Empty command[/yellow]")
        return None
    
    command = parts[0].lower()
    rest = parts[1:] if len(parts) > 1 else []
    
    # Try exact match first (built-in commands)
    cmd_info = session.slash_commands.get(command)
    if cmd_info and not cmd_info.get("is_agent_command"):
        handler = cmd_info["handler"]
        args = " ".join(rest)
        if inspect.iscoroutinefunction(handler):
            return await handler(session, args)
        else:
            return handler(session, args)
    
    # Try hierarchical command path (e.g., "checkpoint/restore")
    if rest:
        # Try combining command with first argument as subcommand
        subcommand_path = f"{command}/{rest[0]}"
        cmd_info = session.slash_commands.get(subcommand_path)
        if cmd_info:
            # Execute agent command via daemon
            args = rest[1:] if len(rest) > 1 else []
            return await _execute_agent_command(session, subcommand_path, args, cmd_info)
    
    # Check if this is a command group (show subcommands)
    subcommands = session.slash_commands.get_subcommands(command)
    if subcommands:
        _show_subcommands(command, subcommands)
        return None
    
    # Try exact agent command match
    cmd_info = session.slash_commands.get(command)
    if cmd_info and cmd_info.get("is_agent_command"):
        return await _execute_agent_command(session, command, rest, cmd_info)
    
    console.print(f"[yellow]Unknown command: /{command}[/yellow]")
    console.print("[dim]Type /help for available commands[/dim]")
    return None


async def _execute_agent_command(session: "WebAgentsSession", path: str, args: List[str], cmd_info: Dict) -> Optional[str]:
    """Execute an agent command via the daemon."""
    await session._ensure_daemon()
    
    # Build args dict from positional arguments
    # For now, simple approach: pass as a single 'args' list
    # TODO: Parse according to command parameters
    data = {}
    if args:
        # If there are required params, try to match
        required = cmd_info.get("required", [])
        params = cmd_info.get("parameters", {})
        
        for i, arg in enumerate(args):
            if i < len(required):
                data[required[i]] = arg
            else:
                # Extra args as list
                data.setdefault("_extra", []).append(arg)
    
    try:
        # POST to /agents/{name}/command/{path}
        full_path = f"/{path}" if not path.startswith("/") else path
        result = await session.daemon_client.execute_command(session.agent_name, full_path, data)
        
        if isinstance(result, dict) and "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]✓[/green] {result}")
        
        return None
    except Exception as e:
        console.print(f"[red]Command error: {e}[/red]")
        return None


def _show_subcommands(prefix: str, subcommands: List[Dict[str, Any]]) -> None:
    """Show available subcommands for a command group."""
    table = Table(title=f"/{prefix} Subcommands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    for cmd in subcommands:
        path = cmd.get("path", "")
        desc = cmd.get("description", "")
        table.add_row(f"/{path}", desc)
    
    console.print(table)
    console.print(f"\n[dim]Usage: /{prefix} <subcommand> [args][/dim]")


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


def cmd_new(session: "WebAgentsSession", args: str) -> None:
    """Start a new session."""
    session.clear_history()
    session.session_id = None  # Will be regenerated
    console.print("[green]✓ Started new session[/green]")
    console.print("[dim]Previous conversation cleared.[/dim]")


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


# Daemon management command handlers (async)

async def cmd_daemon_status(session: "WebAgentsSession", args: str) -> None:
    """Show daemon status"""
    if not session.daemon_client or not await session.daemon_client.is_running():
        console.print("[yellow]webagentsd is not running[/yellow]")
        console.print("[dim]It will auto-start when needed[/dim]")
        return
    
    response = await session.daemon_client.client.get(f"{session.daemon_client.base_url}/")
    data = response.json()
    
    console.print(Panel(
        f"[bold]webagentsd Status[/bold]\n\n"
        f"Status: [green]running[/green]\n"
        f"Registered Agents: {data['agents']}\n"
        f"Cron Jobs: {data['cron_jobs']}\n"
        f"Active Triggers: {data.get('triggers', 0)}",
        title="Daemon",
        border_style="green"
    ))


async def cmd_list_agents(session: "WebAgentsSession", args: str) -> None:
    """List or search registered agents on daemon
    
    Usage:
      /list              - List all agents
      /list <pattern>    - Search agents by name pattern (supports wildcards)
    
    Examples:
      /list
      /list customer-*
      /list *-test
    """
    await session._ensure_daemon()
    
    # Parse search query
    query = args.strip() if args.strip() else None
    agents = await session.daemon_client.list_agents(query=query)
    
    # Build table
    title = f"Registered Agents (found {len(agents)})"
    if query:
        title = f"Search Results for '{query}' ({len(agents)} matches)"
    
    table = Table(title=title)
    table.add_column("Name", style="cyan")
    table.add_column("Source")
    table.add_column("Triggers")
    table.add_column("Description", max_width=50)
    
    for agent in agents:
        triggers = ", ".join(agent.get("triggers", ["api"]))
        description = agent.get("description", "")
        if len(description) > 50:
            description = description[:47] + "..."
        
        table.add_row(
            agent["name"],
            agent.get("source", "local"),
            triggers,
            description
        )
    
    if agents:
        console.print(table)
    else:
        if query:
            console.print(f"[yellow]No agents found matching '{query}'[/yellow]")
        else:
            console.print("[yellow]No agents registered[/yellow]")
            console.print("[dim]Use /register <path> to register an agent[/dim]")


async def cmd_register_agent(session: "WebAgentsSession", args: str) -> None:
    """Register agent with daemon"""
    path_str = args.strip() if args.strip() else "."
    
    await session._ensure_daemon()
    path = Path(path_str).resolve() # Resolve to absolute path
    
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return
    
    try:
        result = await session.daemon_client.register_agent(path)
        if result.get("type") == "directory":
            console.print(f"[green]Registered {result.get('count', 0)} agents from directory: {path}[/green]")
        else:
            console.print(f"[green]Registered agent: {result.get('name', 'unknown')}[/green]")
    except Exception as e:
        console.print(f"[red]Error registering agent: {e}[/red]")


async def cmd_unregister_agent(session: "WebAgentsSession", args: str) -> None:
    """Unregister agent from daemon"""
    if not args.strip():
        console.print("[yellow]Usage: /unregister <agent-name>[/yellow]")
        return
    
    await session._ensure_daemon()
    agent_name = args.strip()
    
    try:
        await session.daemon_client.unregister_agent(agent_name)
        console.print(f"[green]Unregistered agent: {agent_name}[/green]")
    except Exception as e:
        console.print(f"[red]Error unregistering agent: {e}[/red]")


async def cmd_run_agent(session: "WebAgentsSession", args: str) -> None:
    """Run a registered agent"""
    if not args.strip():
        console.print("[yellow]Usage: /run <agent-name>[/yellow]")
        return
    
    await session._ensure_daemon()
    agent_name = args.strip()
    
    result = await session.daemon_client.run_agent(agent_name, trigger="manual")
    console.print(f"[green]Running agent: {agent_name}[/green]")
    console.print(f"[dim]Trigger: manual[/dim]")
