"""
Slash Command Registry

Handle /commands in the REPL. Supports hierarchical subcommands like /checkpoint restore.

Built-in commands are hardcoded here. Agent-provided commands (via @command decorator)
are dynamically registered when connecting to an agent.
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
        """Register built-in CLI commands (hardcoded handlers)."""
        # System commands (/system/*)
        self.register("help", cmd_help, "Show available commands")
        self.register("exit", cmd_exit, "Exit the session")
        self.register("quit", cmd_exit, "Exit the session")
        
        # System group
        self.register("system/status", cmd_daemon_status, "Show daemon/system status")
        self.register("system/clear", cmd_clear, "Clear screen and conversation")
        self.register("system/cls", cmd_screen, "Clear screen only")
        
        # Shortcuts for common system commands
        self.register("clear", cmd_clear, "Clear screen and conversation")
        self.register("cls", cmd_screen, "Clear screen only")
        self.register("status", cmd_daemon_status, "Show daemon status")
        
        # Agent commands (/agent/*)
        self.register("agent/list", cmd_list_agents, "List registered agents")
        self.register("agent/connect", cmd_switch_agent, "Switch to another agent")
        self.register("agent/info", cmd_agent_info, "Show current agent config")
        
        # Shortcuts
        self.register("list", cmd_list_agents, "List registered agents")
        self.register("agent", cmd_agent_info, "Show or switch agent")
        
        # Skill commands (/skill/*)
        self.register("skill/list", cmd_skill_list, "List active skills")
        self.register("skill/add", cmd_skill_add, "Add a skill")
        self.register("skill/remove", cmd_skill_remove, "Remove a skill")
        
        # Shortcuts
        self.register("skill", cmd_skill_list, "List active skills")
        
        # Utility commands
        self.register("tokens", cmd_tokens, "Show token usage")
        self.register("history", cmd_history, "Show conversation history")
        
        # Session alias - maps to /session/new from SessionSkill
        self.register("new", cmd_new, "Start a new session")
        
        # Settings commands
        self.register("settings", cmd_settings, "Show display settings")
        self.register("settings/toolcalls", cmd_settings_toolcalls, "Toggle tool call details (Ctrl+T, T)")
        self.register("settings/thinking", cmd_settings_thinking, "Toggle expanded thinking (Ctrl+T, R)")
    
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
    
    def clear_agent_commands(self):
        """Clear all agent-provided commands.
        
        Called when switching agents to reset the command registry.
        """
        self.agent_commands.clear()
    
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
            
            # Register alias if present
            alias = cmd.get("alias")
            if alias:
                alias_path = alias.lstrip("/")
                self.agent_commands[alias_path] = self.agent_commands[path].copy()
    
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
        
        # Check built-in commands
        for path, cmd in self.commands.items():
            if path.startswith(prefix_with_slash):
                subcommands.append({
                    "path": path,
                    "description": cmd.get("description", ""),
                })
        
        # Check agent commands
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
    
    # Try hierarchical command path FIRST (e.g., "/settings thinking" -> "settings/thinking")
    if rest:
        # Try combining command with first argument as subcommand
        subcommand_path = f"{command}/{rest[0]}"
        cmd_info = session.slash_commands.get(subcommand_path)
        if cmd_info:
            if cmd_info.get("is_agent_command"):
                # Execute agent command via daemon
                args = rest[1:] if len(rest) > 1 else []
                return await _execute_agent_command(session, subcommand_path, args, cmd_info)
            else:
                # Execute built-in subcommand
                handler = cmd_info["handler"]
                args = " ".join(rest[1:]) if len(rest) > 1 else ""
                if inspect.iscoroutinefunction(handler):
                    return await handler(session, args)
                else:
                    return handler(session, args)
    
    # Try exact match for single commands (no subcommand)
    cmd_info = session.slash_commands.get(command)
    if cmd_info and not cmd_info.get("is_agent_command"):
        handler = cmd_info["handler"]
        args = " ".join(rest)
        if inspect.iscoroutinefunction(handler):
            return await handler(session, args)
        else:
            return handler(session, args)
    
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


# =============================================================================
# BUILT-IN COMMAND HANDLERS
# =============================================================================

def cmd_help(session: "WebAgentsSession", args: str) -> None:
    """Show help for commands."""
    from rich.table import Table
    
    # Group commands by category
    categories = {
        "System": [],
        "Agent": [],
        "Skills": [],
        "Session": [],
        "Checkpoint": [],
        "Settings": [],
        "Other": [],
    }
    
    for name, info in session.slash_commands.list_commands().items():
        desc = info.get("description", "")
        # Format as "/cmd subcmd" instead of "cmd/subcmd"
        display_name = "/" + name.replace("/", " ")
        entry = (display_name, desc)
        
        if name.startswith("system/") or name in ("exit", "quit", "clear", "cls", "status", "help"):
            categories["System"].append(entry)
        elif name.startswith("agent/") or name in ("list", "agent"):
            categories["Agent"].append(entry)
        elif name.startswith("skill/") or name == "skill":
            categories["Skills"].append(entry)
        elif name.startswith("session/") or name == "new":
            categories["Session"].append(entry)
        elif name.startswith("checkpoint/") or name == "checkpoint":
            categories["Checkpoint"].append(entry)
        elif name.startswith("settings/") or name == "settings":
            categories["Settings"].append(entry)
        elif name in ("tokens", "history"):
            # Core utility commands
            categories["System"].append(entry)
        else:
            categories["Other"].append(entry)
    
    console.print("\n[bold]Slash Commands[/bold]")
    console.print("[dim]Use /command subcommand or /command/subcommand format[/dim]\n")
    
    for category, cmds in categories.items():
        if cmds:
            console.print(f"[cyan bold]{category}:[/cyan bold]")
            for name, desc in sorted(set(cmds)):
                console.print(f"  {name:<28} [dim]{desc}[/dim]")
            console.print()
    
    # Keyboard shortcuts
    console.print("[cyan bold]Keyboard Shortcuts:[/cyan bold]")
    console.print("  Ctrl+T, T                   Toggle tool call details")
    console.print("  Ctrl+T, R                   Toggle expanded thinking")
    console.print("  Ctrl+T, D                   Toggle todo list")
    console.print("  Alt+Enter                   Insert newline (multiline input)")
    console.print("  Up/Down                     Navigate history")
    console.print()


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


# Agent commands
def cmd_agent_info(session: "WebAgentsSession", args: str) -> None:
    """Show or switch agent."""
    if args.strip():
        # Switch agent - delegate to cmd_switch_agent
        return cmd_switch_agent(session, args)
    
    # Show current agent
    console.print(Panel(
        f"[bold]Current Agent: {session.agent_name}[/bold]\n\n"
        f"Path: {session.agent_path or 'default'}\n"
        f"[dim]Use /agent connect <name> to switch[/dim]",
        title="Agent",
        border_style="cyan"
    ))


async def cmd_switch_agent(session: "WebAgentsSession", args: str) -> None:
    """Switch to another agent."""
    if not args.strip():
        console.print("[yellow]Usage: /agent connect <name>[/yellow]")
        return
    
    new_agent = args.strip()
    
    # Update agent name
    old_agent = session.agent_name
    session.agent_name = new_agent
    session.agent_path = None  # Clear path since we're switching by name
    
    # Clear conversation history for new agent
    session.messages = []
    session.input_tokens = 0
    session.output_tokens = 0
    
    # Refresh commands from new agent
    await session._fetch_agent_commands()
    
    console.print(f"[green]✓ Switched from {old_agent} to {new_agent}[/green]")
    console.print(f"[dim]Commands refreshed. Type /help to see available commands.[/dim]")


# Skill commands
async def cmd_skill_list(session: "WebAgentsSession", args: str) -> None:
    """List active skills."""
    from rich.table import Table
    
    skills = []
    
    # Try to get skills from daemon
    if session.daemon_client:
        try:
            agent_info = await session.daemon_client.get_agent(session.agent_name)
            if agent_info:
                # Try to get skills from metadata
                metadata = agent_info.get("metadata", {})
                skills = metadata.get("skills", [])
        except Exception:
            pass
    
    # If no skills from daemon, try local agent config
    if not skills and session.agent_path:
        try:
            from ..loader.hierarchy import load_agent
            merged = load_agent(session.agent_path)
            skills = merged.metadata.skills or []
        except Exception:
            pass
    
    # Default skills if none found
    if not skills:
        skills = ["filesystem", "shell", "web", "todo", "session", "checkpoint"]
    
    # Build table
    table = Table(title="Active Skills", border_style="cyan")
    table.add_column("Skill", style="cyan")
    table.add_column("Status", style="green")
    
    for skill in skills:
        if isinstance(skill, dict):
            name = list(skill.keys())[0]
            table.add_row(name, "✓ configured")
        else:
            table.add_row(skill, "✓ loaded")
    
    console.print(table)
    console.print(f"\n[dim]Agent: {session.agent_name}[/dim]")


def cmd_skill_add(session: "WebAgentsSession", args: str) -> None:
    """Add a skill to the agent."""
    if not args.strip():
        console.print("[yellow]Usage: /skill add <skill-name>[/yellow]")
        return
    
    skill_name = args.strip()
    console.print(f"[cyan]Adding skill: {skill_name}[/cyan]")
    # TODO: Actually add skill
    console.print("[yellow]Not yet implemented[/yellow]")


def cmd_skill_remove(session: "WebAgentsSession", args: str) -> None:
    """Remove a skill from the agent."""
    if not args.strip():
        console.print("[yellow]Usage: /skill remove <skill-name>[/yellow]")
        return
    
    skill_name = args.strip()
    console.print(f"[yellow]Removing skill: {skill_name}[/yellow]")
    # TODO: Actually remove skill
    console.print("[yellow]Not yet implemented[/yellow]")


def cmd_discover(session: "WebAgentsSession", args: str) -> None:
    """Discover agents by intent."""
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


# =============================================================================
# ASYNC DAEMON COMMAND HANDLERS
# =============================================================================

async def cmd_daemon_status(session: "WebAgentsSession", args: str) -> None:
    """Show daemon status."""
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
    """List or search registered agents on daemon.
    
    Usage:
      /list              - List all agents
      /list <pattern>    - Search agents by name pattern
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
    """Register agent with daemon."""
    path_str = args.strip() if args.strip() else "."
    
    await session._ensure_daemon()
    path = Path(path_str).resolve()  # Resolve to absolute path
    
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


async def cmd_run_agent(session: "WebAgentsSession", args: str) -> None:
    """Run a registered agent."""
    if not args.strip():
        console.print("[yellow]Usage: /run <agent-name>[/yellow]")
        return
    
    await session._ensure_daemon()
    agent_name = args.strip()
    
    result = await session.daemon_client.run_agent(agent_name, trigger="manual")
    console.print(f"[green]Running agent: {agent_name}[/green]")
    console.print(f"[dim]Trigger: manual[/dim]")


# =============================================================================
# SETTINGS COMMAND HANDLERS
# =============================================================================

def cmd_settings(session: "WebAgentsSession", args: str) -> None:
    """Show current display settings."""
    from rich.table import Table
    
    table = Table(title="Display Settings", border_style="dim")
    table.add_column("Setting", style="cyan")
    table.add_column("Status")
    table.add_column("Toggle", style="dim")
    
    # Tool call details
    tool_status = "[green]Expanded[/green]" if session.show_tool_details else "[yellow]Collapsed[/yellow]"
    table.add_row("Tool Calls", tool_status, "Ctrl+T T  or  /settings toolcalls")
    
    # Thinking blocks
    think_status = "[green]Expanded[/green]" if session.expand_thinking else "[yellow]Collapsed[/yellow]"
    table.add_row("Thinking Blocks", think_status, "Ctrl+T R  or  /settings thinking")
    
    # Todo list
    todo_status = "[green]Visible[/green]" if session.show_todos else "[yellow]Hidden[/yellow]"
    table.add_row("Todo List", todo_status, "Ctrl+T D")
    
    console.print(table)


def cmd_settings_toolcalls(session: "WebAgentsSession", args: str) -> None:
    """Toggle tool call details display."""
    session.show_tool_details = not session.show_tool_details
    
    if session.show_tool_details:
        console.print("[green]✓ Tool call details: Expanded[/green]")
        console.print("[dim]Shows: ⚙ read_query({\"query\": \"SELECT...\"})[/dim]")
        console.print("[dim]        └ [{\"COUNT(*)\": 13}][/dim]")
    else:
        console.print("[yellow]✓ Tool call details: Collapsed[/yellow]")
        console.print("[dim]Shows: ⚙ Ran db query...[/dim]")


def cmd_settings_thinking(session: "WebAgentsSession", args: str) -> None:
    """Toggle thinking block expansion."""
    session.expand_thinking = not session.expand_thinking
    
    if session.expand_thinking:
        console.print("[green]✓ Thinking blocks: Expanded[/green]")
        console.print("[dim]Shows full thinking content in a panel[/dim]")
    else:
        console.print("[yellow]✓ Thinking blocks: Collapsed[/yellow]")
        console.print("[dim]Shows: 💭 Thinking... (summary)[/dim]")
