"""
WebAgents TUI - Built with Textual for a polished terminal experience.
"""

import asyncio
import json
import re
import textwrap
import shutil
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical, Horizontal, Center
from textual.widgets import Static, Input, Footer, Header, Markdown, OptionList, LoadingIndicator, ProgressBar, TextArea
from textual.widgets.option_list import Option
from textual.suggester import Suggester
from textual.command import Hit, Hits, Provider, DiscoveryHit
from textual.screen import ModalScreen
from textual import work, on
from textual.worker import Worker
from textual.reactive import reactive

# Available Textual themes
AVAILABLE_THEMES = [
    "textual-dark",
    "textual-light",
    "nord",
    "gruvbox",
    "tokyo-night",
    "dracula",
    "monokai",
    "catppuccin-mocha",
    "solarized-light",
]


def get_term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to fit within width."""
    lines = []
    for line in text.split('\n'):
        if len(line) > width:
            wrapped = textwrap.wrap(line, width=width)
            lines.extend(wrapped if wrapped else [''])
        else:
            lines.append(line)
    return lines


def format_command(path: str) -> str:
    """Format command path as '/session new' instead of '/session/new'."""
    if path.startswith("/"):
        return "/" + path[1:].replace("/", " ")
    return "/" + path.replace("/", " ")


class CommandSuggester(Suggester):
    """Suggester for slash commands and history with ghost text.
    
    Shows dimmed completion text inline as you type:
    - `he` → shows `he[dim]lp[/dim]` (ghost text is completion portion)
    - `/se` → shows `/se[dim]ssion new[/dim]`
    
    UP/DOWN cycles through matching suggestions (ghost text only).
    RIGHT accepts the current suggestion.
    
    Features:
    - Slash command completion (built-in + dynamic from agent)
    - History auto-suggestion for non-slash input
    - Case-insensitive matching
    - Prefix-based cycling through matches with UP/DOWN
    """
    
    # Built-in commands (space-separated format)
    BUILTIN_COMMANDS = [
        "/help", "/clear", "/quit", "/exit",
        "/agent", "/agent list", "/agent switch", "/agent info",
        "/system", "/system status",
        "/settings", "/settings thinking", "/settings tools", "/settings theme",
    ]
    
    def __init__(self, history: List[str], agent_commands: Optional[Dict[str, Any]] = None):
        # use_cache=False ensures fresh suggestions, case_sensitive=False for matching
        super().__init__(use_cache=False, case_sensitive=False)
        self._history = history
        self._agent_commands = agent_commands or {}
        self._command_completions: Dict[str, Dict[str, List[str]]] = {}
        # For cycling through matches
        self._current_prefix = ""
        self._match_index = 0
        self._matches: List[str] = []
        # Current suggestion (updated by cycling)
        self._current_suggestion: Optional[str] = None
        # For history cycling on empty input
        self._history_cycle_index = -1
    
    def update_history(self, history: List[str]) -> None:
        """Update history reference."""
        self._history = history
    
    def update_agent_commands(self, commands: Dict[str, Any]) -> None:
        """Update the list of agent commands for suggestions."""
        self._agent_commands = commands
    
    def update_command_completions(self, completions: Dict[str, Dict[str, List[str]]]) -> None:
        """Update parameter completions for commands.
        
        Args:
            completions: Dict mapping command path to {param_name: [values]}
        """
        self._command_completions = completions
    
    def _get_all_commands(self) -> List[str]:
        """Get all command completions (built-in + agent) in space-separated format.
        
        Also includes parent commands when subcommands exist.
        e.g., if /session/new exists, also includes /session.
        """
        completions = list(self.BUILTIN_COMMANDS)
        parent_cmds = set()
        
        # Add agent commands in space-separated format only
        for path, info in self._agent_commands.items():
            # Convert path like "session/new" to "/session new"
            cmd = format_command(path)
            if cmd not in completions:
                completions.append(cmd)
            
            # Also add parent command if this is a subcommand
            # e.g., "session/new" -> add "/session"
            if "/" in path:
                parent = "/" + path.split("/")[0]
                parent_cmds.add(parent)
        
        # Add parent commands that aren't already in completions
        for parent in parent_cmds:
            if parent not in completions:
                completions.append(parent)
        
        return completions
    
    def get_matches_for_prefix(self, prefix: str) -> List[str]:
        """Get all matching commands/history for a prefix.
        
        Also includes parameter completions when a command is fully typed
        and has dynamic completions (e.g., session/checkpoint IDs).
        
        Returns empty list for empty prefix - suggestions only appear after typing.
        """
        if not prefix:
            return []
        
        prefix_lower = prefix.lower()
        matches = []
        seen = set()
        
        if prefix.startswith("/"):
            # Check if this is a complete command (with or without trailing space)
            # e.g., "/session load" or "/session load " should suggest session IDs
            parts = prefix[1:].strip().split()
            
            if parts:
                # Try parameter completions for the command
                param_matches = self._get_parameter_completions(parts, prefix)
                if param_matches:
                    return param_matches
            
            # Slash commands - collect all matches
            all_cmds = self._get_all_commands()
            for cmd in all_cmds:
                if cmd.lower().startswith(prefix_lower) and cmd.lower() != prefix_lower:
                    if cmd.lower() not in seen:
                        matches.append(cmd)
                        seen.add(cmd.lower())
            
            # Sort: shorter commands first, then alphabetically
            # e.g., /session before /session clear
            matches.sort(key=lambda x: (len(x), x.lower()))
        
        # Then history (for both slash and non-slash)
        for item in reversed(self._history):
            if item.lower().startswith(prefix_lower) and item != prefix:
                if item.lower() not in seen:
                    matches.append(item)
                    seen.add(item.lower())
        
        return matches
    
    def _get_parameter_completions(self, parts: List[str], prefix: str) -> List[str]:
        """Get completions for command parameters.
        
        Only returns parameter completions if the typed command is a COMPLETE 
        known command. For partial commands like "/session lo", this returns
        empty so normal command completion (to "/session load") can work.
        
        Args:
            parts: Command parts (e.g., ["session", "load"] or ["session"])
            prefix: Full prefix including leading /
            
        Returns:
            List of completion suggestions (full command + value)
        """
        matches = []
        prefix_base = prefix.rstrip()  # Base command without trailing space
        
        # Build command path from parts
        # /session load -> session/load
        # /session -> session
        if len(parts) >= 2:
            cmd_path = f"{parts[0]}/{parts[1]}"
        else:
            cmd_path = parts[0]
        
        # Check if this is a COMPLETE known command (not partial)
        # We need to verify the command exists in agent_commands
        is_complete_command = False
        if hasattr(self, '_agent_commands') and self._agent_commands:
            # Check if cmd_path is a registered command
            is_complete_command = cmd_path in self._agent_commands
        
        # Also check if it's in the space-separated format commands
        all_cmds = self._get_all_commands()
        space_cmd = "/" + " ".join(parts)
        for cmd in all_cmds:
            if cmd.lower() == space_cmd.lower():
                is_complete_command = True
                break
        
        # Only show parameter completions for complete commands
        if not is_complete_command:
            return []
        
        # Look up completions for this command
        if cmd_path in self._command_completions:
            completions_dict = self._command_completions[cmd_path]
            if completions_dict:
                for param_name, values in completions_dict.items():
                    for value in values[:10]:  # Limit to 10 suggestions per param
                        # Always include space between command and parameter value
                        suggestion = f"{prefix_base} {value}"
                        if suggestion not in matches:
                            matches.append(suggestion)
        
        return matches
    
    def cycle_suggestion(self, prefix: str, direction: int = 1) -> Optional[str]:
        """Cycle through suggestions for prefix. Returns full suggestion.
        
        Args:
            prefix: Current input text
            direction: 1=next (UP key), -1=prev (DOWN key)
            
        Returns:
            Full suggestion string (Textual will show difference as ghost text)
        """
        # If prefix changed, rebuild matches and reset
        if prefix != self._current_prefix:
            self._current_prefix = prefix
            self._matches = self.get_matches_for_prefix(prefix)
            # Start before first (UP) or after last (DOWN)
            self._match_index = -1 if direction > 0 else len(self._matches)
        
        if not self._matches:
            self._current_suggestion = None
            return None
        
        # Cycle through matches
        self._match_index += direction
        if self._match_index >= len(self._matches):
            self._match_index = 0
        elif self._match_index < 0:
            self._match_index = len(self._matches) - 1
        
        self._current_suggestion = self._matches[self._match_index]
        return self._current_suggestion
    
    def get_current_suggestion(self) -> Optional[str]:
        """Get the current suggestion without cycling."""
        return self._current_suggestion
    
    def reset_cycle(self) -> None:
        """Reset cycling state."""
        self._current_prefix = ""
        self._match_index = 0
        self._matches = []
        self._current_suggestion = None
        self._history_cycle_index = -1  # Reset history cycling too
    
    def cycle_history(self, direction: int = 1) -> Optional[str]:
        """Cycle through history items (for empty input UP/DOWN).
        
        Args:
            direction: 1 for previous (UP), -1 for next (DOWN)
            
        Returns:
            History item or None if at boundary
        """
        if not self._history:
            return None
        
        # Initialize on first call
        if self._history_cycle_index < 0:
            if direction > 0:  # UP - start from most recent
                self._history_cycle_index = len(self._history) - 1
            else:  # DOWN from start - nothing to show
                return None
        else:
            # Move in direction
            new_index = self._history_cycle_index - direction  # UP goes back, DOWN goes forward
            if 0 <= new_index < len(self._history):
                self._history_cycle_index = new_index
            elif new_index >= len(self._history):
                # Past end (DOWN from newest) - return None to clear
                self._history_cycle_index = -1
                return ""  # Return empty to clear input
            else:
                # Past beginning (UP from oldest) - stay at oldest
                return self._history[self._history_cycle_index]
        
        return self._history[self._history_cycle_index]
    
    async def get_suggestion(self, value: str) -> Optional[str]:
        """Return full suggestion string (Textual shows the difference as ghost text).
        
        If user has cycled to a specific suggestion, return that.
        Otherwise return first match.
        
        Returns None for empty input - suggestions only appear after typing.
        """
        # No suggestions for empty input
        if not value:
            return None
        
        # If we have a cycled suggestion for this exact prefix, use it
        if self._current_suggestion and value == self._current_prefix:
            return self._current_suggestion
        
        # Special case: whitespace used for refresh trick - preserve suggestion
        if value.isspace():
            return self._current_suggestion
        
        # Otherwise get first match
        matches = self.get_matches_for_prefix(value)
        if matches:
            # Don't override cycled suggestion if prefix matches
            if value != self._current_prefix:
                self._current_suggestion = matches[0]
            return self._current_suggestion or matches[0]
        
        # Clear suggestion for real input with no matches
        if not value.isspace():
            self._current_suggestion = None
        return None


class SplashScreen(ModalScreen):
    """Loading splash screen shown on startup."""
    
    CSS = """
    SplashScreen {
        align: center middle;
        background: $surface;
    }
    
    #splash-container {
        width: 60;
        height: auto;
        padding: 2 4;
        background: $surface;
        border: thick $primary;
    }
    
    #splash-logo {
        text-align: center;
        color: $primary;
        text-style: bold;
        padding-bottom: 1;
    }
    
    #splash-powered-by {
        text-align: center;
        color: $text-muted;
        padding-bottom: 1;
    }
    
    #splash-status {
        text-align: center;
        color: $text-muted;
        padding: 1 0;
    }
    
    ProgressBar {
        padding: 1 2;
    }
    """
    
    # ASCII art logos
    WEBAGENTS_LOGO = """╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗
║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗
╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝"""

    ROBUTLER_LOGO = """╦═╗╔═╗╔╗ ╦ ╦╔╦╗╦  ╔═╗╦═╗
╠╦╝║ ║╠╩╗║ ║ ║ ║  ║╣ ╠╦╝
╩╚═╚═╝╚═╝╚═╝ ╩ ╩═╝╚═╝╩╚═"""
    
    progress = reactive(0.0)
    status = reactive("Initializing…")
    
    def __init__(self, agent_name: str = "assistant", **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name
    
    def compose(self) -> ComposeResult:
        is_robutler = self.agent_name.lower() == "robutler"
        logo = self.ROBUTLER_LOGO if is_robutler else self.WEBAGENTS_LOGO
        
        with Vertical(id="splash-container"):
            yield Static(logo, id="splash-logo")
            if is_robutler:
                yield Static("powered by WebAgents", id="splash-powered-by")
            yield ProgressBar(total=100, show_eta=False, id="splash-progress")
            yield Static(self.status, id="splash-status")
    
    def watch_progress(self, value: float) -> None:
        try:
            self.query_one("#splash-progress", ProgressBar).update(progress=value)
        except:
            pass
    
    def watch_status(self, value: str) -> None:
        try:
            self.query_one("#splash-status", Static).update(value)
        except:
            pass


class ConnectionMenu(ModalScreen):
    """Modal menu for connection options."""
    
    CSS = """
    ConnectionMenu {
        align: left top;
        background: rgba(0, 0, 0, 0.5);  /* Dimmed background */
    }
    
    #connection-menu-container {
        width: 30;
        height: auto;
        offset: 1 1;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }
    
    OptionList {
        height: auto;
        background: transparent;
        border: none;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]
    
    def compose(self) -> ComposeResult:
        with Vertical(id="connection-menu-container"):
            yield OptionList(
                Option("Login (coming soon)", id="login", disabled=True),
                Option("Exit", id="exit"),
            )
    
    def on_mount(self) -> None:
        """Handle click outside to dismiss."""
        # Focus option list on mount
        self.query_one(OptionList).focus()

    def on_click(self, event) -> None:
        """Dismiss if clicking outside the menu container."""
        # Use region check (event.x/y are relative to the screen)
        if not self.query_one("#connection-menu-container").region.contains(event.x, event.y):
            self.dismiss()
            
    def action_dismiss(self) -> None:
        self.dismiss()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id == "exit":
            self.app.exit()
        self.dismiss()


class ConnectionStatus(Static):
    """Connection status indicator widget."""
    
    is_connected = reactive(False)
    can_focus = True
    
    def render(self) -> str:
        icon = "●" if self.is_connected else "○"
        color = "$warning" if self.is_connected else "$text-muted"
        return f"[{color}]{icon}[/]"
    
    def watch_is_connected(self, value: bool) -> None:
        self.refresh()
    
    async def on_click(self) -> None:
        """Show connection menu on click."""
        self.app.push_screen(ConnectionMenu())
    
    async def on_key(self, event) -> None:
        """Handle keyboard activation (Enter/Space)."""
        if event.key in ("enter", "space"):
            self.app.push_screen(ConnectionMenu())


class MenuButton(Static):
    """Menu button that opens the command palette."""
    
    can_focus = True
    
    def render(self) -> str:
        # Use simple triple horizontal lines (hamburger menu)
        return "[$text-muted]≡[/]"
    
    async def on_click(self) -> None:
        """Open command palette on click."""
        self.app.action_command_palette()
    
    async def on_key(self, event) -> None:
        """Handle keyboard activation (Enter/Space)."""
        if event.key in ("enter", "space"):
            self.app.action_command_palette()


class CustomHeader(Horizontal):
    """Custom header replacing the standard Textual header."""
    
    DEFAULT_CSS = """
    CustomHeader {
        height: 1;
        dock: top;
        background: $surface;
        color: $text-muted;
        width: 100%;
    }
    
    CustomHeader #header-title {
        width: 1fr;
        text-align: center;
        content-align: center middle;
        text-style: bold;
    }
    
    CustomHeader MenuButton {
        dock: right;
        width: 3;
        content-align: center middle;
    }
    
    CustomHeader MenuButton:hover {
        background: $boost;
    }
    
    CustomHeader MenuButton:focus {
        background: $boost;
    }
    
    CustomHeader ConnectionStatus {
        dock: left;
        width: 3;
        content-align: center middle;
    }
    
    CustomHeader ConnectionStatus:hover {
        background: $boost;
    }
    
    CustomHeader ConnectionStatus:focus {
        background: $boost;
    }
    """
    
    def __init__(self, agent_name: str = "assistant", **kwargs):
        super().__init__(**kwargs)
        self.agent_name = agent_name

    def compose(self) -> ComposeResult:
        yield ConnectionStatus(id="connection-status")
        title = "Robutler" if self.agent_name.lower() == "robutler" else "WebAgents"
        yield Static(title, id="header-title")
        yield MenuButton(id="menu-button")


class ThemePickerScreen(ModalScreen):
    """Modal screen for picking a theme."""
    
    CSS = """
    ThemePickerScreen {
        align: center middle;
    }
    
    #theme-picker {
        width: 40;
        height: auto;
        max-height: 20;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }
    
    #theme-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        padding-bottom: 1;
    }
    
    OptionList {
        height: auto;
        max-height: 15;
        background: $surface;
        border: none;
    }
    
    OptionList > .option-list--option-highlighted {
        background: $primary 30%;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]
    
    def compose(self) -> ComposeResult:
        with Vertical(id="theme-picker"):
            yield Static("🎨 Select Theme", id="theme-title")
            yield OptionList(*[Option(t, id=t) for t in AVAILABLE_THEMES])
    
    @on(OptionList.OptionSelected)
    def on_theme_selected(self, event: OptionList.OptionSelected) -> None:
        theme_name = str(event.option_id)
        try:
            self.app.theme = theme_name
            self.app.notify(f"Theme: {theme_name}")
        except Exception as e:
            self.app.notify(f"Theme error: {e}", severity="error")
        self.dismiss()
    
    def action_dismiss(self) -> None:
        self.dismiss()


class WebAgentsCommands(Provider):
    """Command palette provider for WebAgents.
    
    Includes built-in commands and dynamically discovered agent commands.
    """
    
    BUILTIN = [
        ("Clear Chat", "clear", "Clear all messages"),
        ("Toggle Thinking", "toggle_thinking", "Show/hide reasoning"),
        ("Toggle Tools", "toggle_tools", "Show/hide tool details"),
        ("🎨 Pick Theme", "pick_theme", "Change UI theme"),
        ("Help", "/help", "Show help"),
        ("Quit", "quit", "Exit"),
        # Agent management (as slash commands)
        ("Agent List", "/agent list", "List registered agents"),
        ("Agent Switch", "/agent switch", "Switch to another agent"),
        ("Agent Info", "/agent info", "Show current agent info"),
        # System
        ("System Status", "/system status", "Show daemon status"),
    ]
    
    def _get_all_commands(self):
        """Get all commands including agent commands."""
        commands = list(self.BUILTIN)
        
        # Add agent commands from app if available (dynamic only, no fallbacks)
        app = self.app
        
        if hasattr(app, "_agent_commands") and app._agent_commands:
            seen = set()
            for path, info in app._agent_commands.items():
                # Skip aliases to avoid duplicates  
                primary_path = info.get("path", "").lstrip("/")
                if primary_path in seen:
                    continue
                seen.add(primary_path)
                
                # Format nicely - convert session/save to "/session save"
                display_name = f"/{primary_path.replace('/', ' ')}"
                desc = info.get("description", "")
                # Use a unique action name that won't conflict
                commands.append((display_name, f"agent_cmd:{primary_path}", desc))
        
        return commands
    
    def _make_action(self, action: str):
        """Create a callable action for a command."""
        app = self.app
        
        if action.startswith("agent_cmd:"):
            # Agent command - execute via slash handler
            cmd_path = action.split(":", 1)[1]
            return lambda p=cmd_path: asyncio.create_task(app._handle_slash_command(f"/{p}"))
        elif action.startswith("/"):
            # Built-in slash command (like /agent list)
            cmd = action.replace(" ", "/").lstrip("/")
            return lambda c=cmd: asyncio.create_task(app._handle_slash_command(f"/{c}"))
        else:
            # Regular action
            return lambda a=action: getattr(app, f"action_{a}")()
    
    async def discover(self) -> Hits:
        """Show commands when palette first opens."""
        all_commands = self._get_all_commands()
        # Log command count for debugging
        app = self.app
        agent_cmd_count = len(app._agent_commands) if hasattr(app, "_agent_commands") else 0
        app.log.debug(f"Command palette: {len(all_commands)} total, {agent_cmd_count} from agent")
        
        for name, action, help_text in all_commands:
            yield DiscoveryHit(
                name,
                self._make_action(action),
                help=help_text,
            )
    
    async def search(self, query: str) -> Hits:
        """Search commands."""
        matcher = self.matcher(query)
        
        for name, action, help_text in self._get_all_commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(name),
                    self._make_action(action),
                    help=help_text,
                )


class UserMessage(Static):
    """Widget for displaying user messages with vertical bars."""
    
    def __init__(self, content: str):
        super().__init__()
        self.user_content = content
    
    def compose(self) -> ComposeResult:
        yield Static("┃", classes="accent-bar")
        yield Static(f"┃  {self.user_content}", classes="accent-bar")
        yield Static("┃", classes="accent-bar")


class ThinkingBlock(Static):
    """Widget for displaying thinking/reasoning blocks."""
    
    def __init__(self, content: str = "", subtitle: str = "", duration: float = 0, expanded: bool = False):
        super().__init__()
        self.thinking_content = content
        self.subtitle = subtitle
        self.duration = duration
        self.expanded = expanded
    
    def compose(self) -> ComposeResult:
        header = f"∵ {self.subtitle}" if self.subtitle else "∵ Thinking…"
        if self.duration > 0:
            header += f" · {self.duration:.1f}s"
        
        yield Static(header, classes="muted-text italic")
        
        if self.expanded and self.thinking_content:
            # Add newline before content and render with Markdown
            yield Static(" ")  # Empty line for spacing
            yield Markdown(self.thinking_content)


class ToolCallBlock(Static):
    """Widget for displaying tool call results."""
    
    def __init__(self, name: str, args: str = "", result: str = "", status: str = "success", expanded: bool = False):
        super().__init__()
        self.tool_name = name
        self.tool_args = args
        self.tool_result = result
        self.tool_status = status
        self.expanded = expanded
    
    def compose(self) -> ComposeResult:
        icon = "■" if self.tool_status == "success" else "◇" if self.tool_status == "running" else "✗"
        summary = self._get_summary()
        
        yield Static(f"{icon} {summary}", classes="muted-text")
        
        if self.expanded and self.tool_result:
            width = get_term_width() - 10
            result_lines = wrap_text(str(self.tool_result)[:500], width)[:10]
            for line in result_lines:
                yield Static(f"│  {line}", classes="muted-bar")
    
    def _get_summary(self) -> str:
        name = self.tool_name
        try:
            args_dict = json.loads(self.tool_args) if isinstance(self.tool_args, str) else self.tool_args
        except:
            args_dict = {}
        
        if name in ("read_file", "read"):
            path = args_dict.get("path", args_dict.get("file_path", "file"))
            return f"Read {Path(path).name if path else 'file'}"
        elif name in ("write_file", "write"):
            path = args_dict.get("file_path", args_dict.get("path", "file"))
            return f"Wrote {Path(path).name if path else 'file'}"
        elif name == "list_directory":
            return "Listed directory"
        elif name in ("shell", "bash", "run_command"):
            return "Ran command"
        elif name == "read_query":
            return "Ran db query"
        elif name == "list_tables":
            return "Listed db tables"
        else:
            return f"Ran {name}"


class AssistantMessage(Static):
    """Widget for displaying assistant responses with markdown."""
    
    def __init__(self, content: str):
        super().__init__()
        self.response_content = content
    
    def compose(self) -> ComposeResult:
        # Strip thinking tags and tool markers for display
        clean = re.sub(r'<think>.*?</think>', '', self.response_content, flags=re.DOTALL).strip()
        clean = re.sub(r'<tool_call id="[^"]*"/>', '', clean)
        if clean:
            yield Markdown(clean)


class StreamingIndicator(Static):
    """Widget showing streaming progress with tool calls and spinner."""
    
    spinner_frame = reactive(0)
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self):
        super().__init__()
        self._text = ""
        self._tools: Dict[str, Dict] = {}
        self._thinking = ""
        self._timer = None
    
    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._next_frame)
    
    def _next_frame(self) -> None:
        self.spinner_frame = (self.spinner_frame + 1) % len(self.SPINNER_FRAMES)
        self._update_display()
    
    def update_content(self, text: str) -> None:
        self._text = text
        self._update_display()
    
    def update_thinking(self, content: str) -> None:
        self._thinking = content
        self._update_display()
    
    def add_tool(self, tool_id: str, name: str, args: str) -> None:
        self._tools[tool_id] = {"name": name, "args": args, "status": "running"}
        self._update_display()
    
    def complete_tool(self, tool_id: str, status: str, result: str) -> None:
        if tool_id in self._tools:
            self._tools[tool_id]["status"] = status
            self._tools[tool_id]["result"] = result
            self._update_display()
    
    def _update_display(self) -> None:
        parts = []
        spinner = self.SPINNER_FRAMES[self.spinner_frame]
        
        # Show thinking subtitle - parse in real-time as it streams
        if self._thinking:
            # Look for the most recent complete or incomplete subtitle
            matches = re.findall(r'\*\*([^\*]+)\*\*', self._thinking)
            if matches:
                # Use the last complete subtitle
                subtitle = matches[-1].strip()
            else:
                # Try to catch an incomplete/streaming subtitle at the end
                open_match = re.search(r'\*\*([^\*]+)$', self._thinking)
                if open_match:
                    subtitle = open_match.group(1).strip()
                else:
                    subtitle = "Thinking…"
            
            # Truncate long subtitles
            if len(subtitle) > 60:
                subtitle = subtitle[:57] + "…"
                
            # Use spinner while thinking
            parts.append(f"[#666666 italic]{spinner} {subtitle}[/]")
        
        # Show active tools
        for tid, tool in self._tools.items():
            is_running = tool["status"] == "running"
            icon = spinner if is_running else "■"
            color = "#888888" if not is_running else "#aaaaaa"
            parts.append(f"[{color}]{icon} {tool['name']}[/]")
        
        # Show content preview
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', self._text, flags=re.DOTALL)
        clean = re.sub(r'<tool_call id="[^"]*"/>', '', clean).strip()
        if clean:
            preview = clean[-300:] if len(clean) > 300 else clean
            parts.append(preview)
        
        if not parts:
            parts.append(f"[#666666]{spinner}[/]")
        
        self.update("\n".join(parts))


from textual.message import Message

class ChatInput(Input):
    """Custom Input with ghost text suggestions and key handling.
    
    Features:
    - Ghost text autocomplete (dimmed suggestion inline via Suggester)
    - UP key: Navigate history backwards  
    - DOWN key: Navigate history forwards
    - RIGHT key: Accept current suggestion
    - Enter: Submit message
    """
    
    class Submitted(Message):
        """Posted when user submits the message."""
        def __init__(self, value: str):
            self.value = value
            super().__init__()
    
    class HistoryUp(Message):
        """Cycle to previous suggestion/history."""
        pass
    
    class HistoryDown(Message):
        """Cycle to next suggestion/history."""
        pass
    
    class AcceptSuggestion(Message):
        """Accept current suggestion (RIGHT key)."""
        pass

    def _on_key(self, event) -> None:
        """Handle key events for cycling and suggestion acceptance."""
        # UP key - cycle to previous suggestion
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.post_message(self.HistoryUp())
            return
        
        # DOWN key - cycle to next suggestion
        if event.key == "down":
            event.stop()
            event.prevent_default()
            self.post_message(self.HistoryDown())
            return
        
        # RIGHT key at end of input - accept suggestion
        if event.key == "right" and self.cursor_position == len(self.value):
            event.stop()
            event.prevent_default()
            self.post_message(self.AcceptSuggestion())
            return
        
        # Let parent handle everything else (including Enter for submit via action)
        super()._on_key(event)
    
    def action_submit(self) -> None:
        """Override submit to post our custom message."""
        if self.value.strip():
            self.post_message(self.Submitted(self.value))


class PromptInput(Static):
    """Input prompt area with ghost text suggestions and history.
    
    Features:
    - Ghost text autocomplete (dimmed suggestion inline)
    - UP/DOWN: Navigate command history
    - RIGHT: Accept current suggestion
    """
    
    is_busy = reactive(False)
    
    def __init__(self, agent_name: str = "assistant", agent_path: str = ""):
        super().__init__()
        self._agent_name = agent_name
        self._agent_path = agent_path
        self._history: List[str] = []
        self._history_index = -1
        self._suggester = CommandSuggester(self._history)
        self._load_history()
    
    def compose(self) -> ComposeResult:
        yield Static("┃", id="bar-top", classes="accent-bar")
        with Horizontal(id="input-row"):
            yield Static("┃", id="bar-left", classes="accent-bar")
            # Using Input with Suggester for ghost text autocomplete
            yield ChatInput(
                id="chat-input",
                placeholder=f"Message @{self._agent_name} or /command…",
                suggester=self._suggester,
            )
        yield Static("┃", id="bar-bottom", classes="accent-bar")
        
        # Agent info
        path_display = self._agent_path
        if len(path_display) > 45:
            path_display = "…" + path_display[-42:]
        yield Static(f"┃  @{self._agent_name}  {path_display}", id="agent-info", classes="agent-info-bar")
    
    def watch_is_busy(self, value: bool) -> None:
        try:
            input_widget = self.query_one("#chat-input", ChatInput)
            input_widget.disabled = value
        except:
            pass
            
    async def on_click(self) -> None:
        """Focus input on click."""
        try:
            self.query_one("#chat-input", ChatInput).focus()
        except:
            pass
    
    def update_agent(self, name: str, path: str = "") -> None:
        self._agent_name = name
        self._agent_path = path
        # Update agent info bar
        agent_info = self.query_one("#agent-info", Static)
        path_display = path if len(path) <= 45 else "…" + path[-42:]
        agent_info.update(f"┃  @{name}  {path_display}")
        # Update placeholder
        try:
            input_widget = self.query_one("#chat-input", ChatInput)
            input_widget.placeholder = f"Message @{name} or /command…"
        except:
            pass
    
    def update_agent_commands(self, commands: Dict[str, Any]) -> None:
        """Update the suggester with agent commands."""
        self._suggester.update_agent_commands(commands)
    
    def update_command_completions(self, completions: Dict[str, Dict[str, List[str]]]) -> None:
        """Update the suggester with command parameter completions."""
        self._suggester.update_command_completions(completions)
    
    def add_to_history(self, text: str) -> None:
        """Add text to history and update suggester."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
            self._save_history()
            # Update suggester with new history
            self._suggester.update_history(self._history)
        # Reset state
        self._history_index = len(self._history)
        self._suggester.reset_cycle()
    
    def cycle_suggestion_up(self, current_text: str) -> tuple:
        """Cycle suggestion UP (previous match).
        
        Returns: (should_replace_value, suggestion)
        - Returns (False, suggestion) for ghost text display
        """
        if not current_text:
            return (False, None)
        
        # Has text - cycle through suggestions (ghost text only)
        suggestion = self._suggester.cycle_suggestion(current_text, direction=1)
        return (False, suggestion)
    
    def cycle_suggestion_down(self, current_text: str) -> tuple:
        """Cycle suggestion DOWN (next match).
        
        Returns: (should_replace_value, suggestion)
        - Returns (False, suggestion) for ghost text display
        """
        if not current_text:
            return (False, None)
        
        # Has text - cycle through suggestions (ghost text only)
        suggestion = self._suggester.cycle_suggestion(current_text, direction=-1)
        return (False, suggestion)
    
    def get_current_suggestion(self) -> Optional[str]:
        """Get the current suggestion for accepting with RIGHT key."""
        return self._suggester.get_current_suggestion()
    
    def _get_history_file(self) -> Path:
        """Get path to history file."""
        history_dir = Path.home() / ".webagents"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir / "tui_history"
    
    def _load_history(self) -> None:
        """Load history from file."""
        try:
            history_file = self._get_history_file()
            if history_file.exists():
                with open(history_file, "r") as f:
                    self._history = [line.strip() for line in f.readlines() if line.strip()]
                # Keep only last 1000 items
                self._history = self._history[-1000:]
                self._history_index = len(self._history)
        except Exception:
            pass
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            history_file = self._get_history_file()
            with open(history_file, "w") as f:
                # Save last 1000 items
                for item in self._history[-1000:]:
                    f.write(item + "\n")
        except Exception:
            pass


class WebAgentsTUI(App):
    """The main WebAgents TUI application."""
    
    COMMANDS = {WebAgentsCommands}
    TITLE = "WebAgents"
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #chat-container {
        height: 1fr;
        padding: 1 0 1 1;
        scrollbar-gutter: stable;
    }
    
    ScrollableContainer {
        scrollbar-size: 1 1;
        scrollbar-color: $scrollbar;
        scrollbar-color-hover: $scrollbar-hover;
        scrollbar-color-active: $scrollbar-active;
    }
    
    #welcome {
        text-align: center;
        color: $text-muted;
        padding: 2 0;
    }
    
    /* Theme-responsive classes */
    .accent-bar {
        color: $primary;
    }
    
    .muted-text {
        color: $text-muted;
    }
    
    .muted-bar {
        color: $text-muted;
    }
    
    .italic {
        text-style: italic;
    }
    
    .agent-info-bar {
        color: $primary;
    }
    
    UserMessage {
        height: auto;
        margin: 0 0 1 0;
        padding: 0;
    }
    
    AssistantMessage {
        height: auto;
        margin: 0 0 1 0;
        padding-left: 3;
        color: $text;
    }
    
    ThinkingBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 0 3;
    }
    
    ThinkingBlock Markdown {
        margin: 0;
        padding: 0 0 0 2;
        color: $text-muted;
    }
    
    ToolCallBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 0 3;
    }
    
    StreamingIndicator {
        height: auto;
        margin: 0 0 1 0;
        padding-left: 3;
        color: $text-muted;
    }
    
    PromptInput {
        dock: bottom;
        height: auto;
        padding: 0 0 0 1;
        background: $background;
    }
    
    PromptInput #input-row {
        height: 1;
    }
    
    PromptInput #bar-left {
        width: 1;
    }
    
    PromptInput:focus-within {
        background: $boost;
    }

    PromptInput #chat-input {
        border: none;
        background: transparent;
        color: $text;
        padding: 0 0 0 1;
        height: 1;
        width: 1fr;
    }
    
    PromptInput #chat-input:focus {
        border: none;
    }
    
    /* Ghost text suggestion styling - dimmed */
    PromptInput #chat-input.-has-suggestion Input.-has-value {
        color: $text;
    }
    
    Footer {
        background: $surface;
        color: $text-muted;
    }
    
    Header {
        background: $surface;
        color: $text-muted;
        height: 1;
    }
    
    .help-text {
        padding-left: 3;
        margin-bottom: 1;
    }
    
    /* Command output - aligned with messages */
    .cmd-output {
        height: auto;
        margin: 0 0 1 0;
        padding-left: 3;
    }
    
    Markdown {
        margin: 0;
        padding: 0;
        color: $text;
    }
    """
    
    ENABLE_COMMAND_PALETTE = True
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "clear_prompt", "Clear", show=False),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_thinking", "Thinking", show=False),
        Binding("ctrl+d", "toggle_tools", "Tools", show=False),
        Binding("ctrl+shift+p", "pick_theme", "Theme", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("f1", "help", "Help"),
        Binding("ctrl+shift+up", "scroll_up", show=False),
        Binding("ctrl+shift+down", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
    ]
    
    def __init__(
        self,
        agent_name: str = "assistant",
        agent_path: Optional[Path] = None,
        daemon_client = None,
        use_daemon: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.agent_path = agent_path
        self.daemon_client = daemon_client
        self.use_daemon = use_daemon
        self.messages: List[Dict[str, str]] = []
        self.expand_thinking = False
        self.show_tool_details = False
        self._active_tools: Dict[str, Dict] = {}
        self._tool_index_to_id: Dict[int, str] = {}
        self._streaming_widget: Optional[StreamingIndicator] = None
        
        # Session management
        self.session_id: Optional[str] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._session_manager = None
        self._agent_commands: Dict[str, Dict[str, Any]] = {}  # Commands from agent skills
        self._command_completions: Dict[str, Dict[str, List[str]]] = {}  # Parameter completions
        self._init_session_manager()
    
    def _init_session_manager(self) -> None:
        """Initialize session manager for auto-save/resume."""
        if not self.agent_path:
            return
        try:
            from webagents.agents.skills.local.session.skill import SessionManager
            agent_dir = self.agent_path.parent if self.agent_path.is_file() else self.agent_path
            self._session_manager = SessionManager(agent_dir, self.agent_name)
        except Exception:
            self._session_manager = None
    
    async def _fetch_agent_commands(self) -> None:
        """Fetch and register commands from the connected agent.
        
        Commands are dynamically discovered from the agent's skills.
        Also pre-fetches completions for commands with parameters.
        """
        self._agent_commands.clear()
        self._command_completions.clear()
        
        if not self.daemon_client:
            self.log.warning("No daemon client, cannot fetch commands")
            return
        
        try:
            commands = await self.daemon_client.list_commands(self.agent_name)
            self.log.info(f"Fetched {len(commands)} commands from agent '{self.agent_name}'")
            
            for cmd in commands:
                path = cmd.get("path") or ""
                path = path.lstrip("/") if path else ""
                if path:
                    self._agent_commands[path] = cmd
                    self.log.debug(f"  Registered command: /{path}")
                    # Also register alias (handle None explicitly)
                    alias = cmd.get("alias") or ""
                    alias = alias.lstrip("/") if alias else ""
                    if alias and alias != path:
                        self._agent_commands[alias] = cmd
            
            # Fetch completions for commands with parameters
            await self._fetch_command_completions()
            
            # Update suggester with agent commands and completions
            try:
                prompt = self.query_one(PromptInput)
                prompt.update_agent_commands(self._agent_commands)
                prompt.update_command_completions(self._command_completions)
            except Exception as e:
                self.log.warning(f"Could not update suggester: {e}")
            
            # Log summary
            cmd_count = len([p for p in self._agent_commands if "/" in p or p in ["session", "checkpoint"]])
            self.log.info(f"Registered {len(self._agent_commands)} commands (with aliases)")
            
            # Show user notification with command count
            if commands:
                self.notify(f"Loaded {len(commands)} commands", timeout=2)
        except Exception as e:
            import traceback
            self.log.error(f"Failed to fetch agent commands: {e}\n{traceback.format_exc()}")
            self.notify(f"Failed to load commands: {e}", severity="warning")
    
    async def _fetch_command_completions(self) -> None:
        """Fetch completions for commands with parameters.
        
        Commands can provide dynamic completions (e.g., session/checkpoint IDs).
        Completions are fetched lazily when needed, not upfront.
        """
        # Don't fetch upfront - will be fetched on-demand when autocomplete is triggered
        pass
    
    async def _fetch_completions_for_command(self, cmd_path: str) -> Dict[str, List[str]]:
        """Fetch completions for a specific command on-demand.
        
        Called every time user presses UP/DOWN on a slash command.
        Always makes a GET request (no negative caching) to discover
        newly available completions.
        """
        if not self.daemon_client:
            return {}
        
        # Check cache first - only cache positive results (with completions)
        if cmd_path in self._command_completions and self._command_completions[cmd_path]:
            return self._command_completions[cmd_path]
        
        try:
            # GET /command/{path} returns docs + completions
            docs = await self.daemon_client.get_command_docs(self.agent_name, cmd_path)
            if docs and docs.get("completions"):
                self._command_completions[cmd_path] = docs["completions"]
                self.log.debug(f"  Completions for /{cmd_path}: {list(docs['completions'].keys())}")
                return docs["completions"]
        except Exception as e:
            # Don't cache failed lookups - try again next time
            self.log.debug(f"Could not fetch completions for /{cmd_path}: {e}")
        
        return {}
    
    async def _auto_resume_session(self) -> None:
        """Auto-resume the latest session if available."""
        if not self._session_manager:
            import uuid
            self.session_id = str(uuid.uuid4())
            return
        
        try:
            session = self._session_manager.load_latest(max_messages=100)
            if session and session.messages:
                # Restore messages
                self.messages = [
                    m.to_dict() if hasattr(m, 'to_dict') else m
                    for m in session.messages
                ]
                self.session_id = session.session_id
                self.input_tokens = session.input_tokens
                self.output_tokens = session.output_tokens
                
                # Load user messages into history for autocomplete
                self._load_session_messages_to_history()
                
                # Display restored session in chat
                await self._display_restored_session()
            else:
                import uuid
                self.session_id = str(uuid.uuid4())
        except Exception:
            import uuid
            self.session_id = str(uuid.uuid4())
    
    def _load_session_messages_to_history(self) -> None:
        """Load user messages from current session into autocomplete history.
        
        Only includes prompts from the current session, not global inter-session history.
        This keeps autocomplete suggestions relevant to the current conversation.
        """
        try:
            prompt = self.query_one(PromptInput)
            
            # Clear any existing history - only session messages should be in autocomplete
            prompt._history = []
            
            for msg in self.messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content:
                        # Add to history (avoiding duplicates)
                        if content and (not prompt._history or prompt._history[-1] != content):
                            prompt._history.append(content)
            
            # Update suggester with session-only history
            prompt._suggester.update_history(prompt._history)
            prompt._history_index = len(prompt._history)
        except Exception:
            pass
    
    async def _display_restored_session(self) -> None:
        """Display restored session messages in the chat container."""
        container = self.query_one("#chat-container", ScrollableContainer)
        
        # Build tool results map from 'tool' role messages (OpenAI format)
        tool_results = {}
        for msg in self.messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                tool_results[msg["tool_call_id"]] = msg.get("content", "")
        
        # Filter out 'tool' role messages for display count
        display_messages = [m for m in self.messages if m.get("role") != "tool"]
        
        # Show header with earlier messages count first
        if len(display_messages) > 6:
            await container.mount(Static(
                f"[dim]… {len(display_messages) - 6} earlier messages[/dim]",
                classes="muted-text"
            ))
        
        await container.mount(Static(
            f"[dim]── Restored ({len(display_messages)} messages) ──[/dim]",
            classes="muted-text"
        ))
        
        # Show last few messages with proper UI elements
        for msg in display_messages[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            
            if role == "user":
                display = content[:120] + "…" if len(content) > 120 else content
                await container.mount(UserMessage(display))
            elif role == "assistant":
                # Parse and render thinking blocks
                think_matches = list(re.finditer(r'<think>(.*?)</think>', content, re.DOTALL))
                for m in think_matches:
                    think_content = m.group(1).strip()
                    # Get last subtitle
                    subtitles = re.findall(r'\*\*([^\*]+)\*\*', think_content)
                    subtitle = subtitles[-1].strip() if subtitles else ""
                    await container.mount(ThinkingBlock(
                        content=think_content,
                        subtitle=subtitle,
                        duration=0,
                        expanded=self.expand_thinking
                    ))
                
                # Parse and render tool calls from metadata
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            tc_id = tc.get("id", "")
                            # Get result from tool_results map (OpenAI format) or embedded (legacy)
                            result = tool_results.get(tc_id, "") or tc.get("result", "")
                            await container.mount(ToolCallBlock(
                                name=func.get("name", "tool"),
                                args=func.get("arguments", ""),
                                result=result,
                                status="success",
                                expanded=self.show_tool_details
                            ))
                
                # Clean content (strip thinking tags)
                clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if clean:
                    # Truncate for preview
                    if len(clean) > 200:
                        clean = clean[:200] + "…"
                    await container.mount(AssistantMessage(clean))
        
        container.scroll_end(animate=False)
    
    def _save_session(self) -> None:
        """Save current session."""
        if not self._session_manager or not self.session_id:
            return
        
        try:
            from webagents.agents.skills.local.session.skill import Session, Message
            from datetime import datetime
            
            messages = []
            for m in self.messages:
                if isinstance(m, dict):
                    messages.append(Message(
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                        timestamp=m.get("timestamp"),
                        tool_calls=m.get("tool_calls"),
                        tool_call_id=m.get("tool_call_id"),
                    ))
                else:
                    messages.append(m)
            
            session = Session(
                session_id=self.session_id,
                agent_name=self.agent_name,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                messages=messages,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )
            self._session_manager.save(session)
        except Exception:
            pass  # Don't fail on save errors
    
    def compose(self) -> ComposeResult:
        yield CustomHeader(agent_name=self.agent_name)
        
        with ScrollableContainer(id="chat-container"):
            yield Static(self._get_welcome(), id="welcome")
        
        yield PromptInput(
            agent_name=self.agent_name,
            agent_path=str(self.agent_path) if self.agent_path else ""
        )
        yield Footer()
    
    def _get_welcome(self) -> str:
        if self.agent_name.lower() == "robutler":
            return """
[bold $primary]╦═╗╔═╗╔╗ ╦ ╦╔╦╗╦  ╔═╗╦═╗[/]
[bold $primary]╠╦╝║ ║╠╩╗║ ║ ║ ║  ║╣ ╠╦╝[/]
[bold $primary]╩╚═╚═╝╚═╝╚═╝ ╩ ╩═╝╚═╝╩╚═[/]

[$text-muted]powered by WebAgents[/]

[$text-muted]/help or Ctrl+P for commands[/]
"""
        return """
[bold $primary]╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗[/]
[bold $primary]║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗[/]
[bold $primary]╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝[/]

[$text-muted]/help or Ctrl+P for commands[/]
"""
    
    async def on_mount(self) -> None:
        self.query_one("#chat-input", ChatInput).focus()
        if self.use_daemon and self.daemon_client:
            # Show splash screen immediately and run connection in background
            splash = SplashScreen(agent_name=self.agent_name)
            self.push_screen(splash)
            self.run_worker(self._startup_sequence(splash))
    
    async def _startup_sequence(self, splash: SplashScreen) -> None:
        """Run startup connection sequence."""
        # Ensure splash is visible for a moment
        await asyncio.sleep(1.0)
        
        # Connect and register
        await self._ensure_daemon_and_agent(splash)
        
        # Auto-resume session
        splash.status = "Restoring session…"
        splash.progress = 95
        await self._auto_resume_session()
        
        # Ensure splash stays for completion
        await asyncio.sleep(0.5)
        self.pop_screen()
        self.query_one("#chat-input", ChatInput).focus()
    
    async def _ensure_daemon_and_agent(self, splash: Optional[SplashScreen] = None) -> None:
        def update_splash(progress: float, status: str):
            if splash:
                splash.progress = progress
                splash.status = status
        
        update_splash(10, "Checking daemon…")
        try:
            await self.daemon_client.health()
            update_splash(30, "Daemon ready")
            # Update connection status
            self.query_one("#connection-status", ConnectionStatus).is_connected = True
        except Exception:
            update_splash(20, "Starting daemon…")
            try:
                import subprocess
                import sys
                subprocess.Popen(
                    [sys.executable, "-m", "webagents.cli.daemon.server"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                for i in range(10):
                    await asyncio.sleep(0.5)
                    update_splash(20 + i * 3, f"Waiting for daemon… ({i+1}/10)")
                    try:
                        await self.daemon_client.health()
                        # Update connection status
                        self.query_one("#connection-status", ConnectionStatus).is_connected = True
                        break
                    except:
                        continue
            except Exception as e:
                self.notify(f"Daemon failed: {e}", severity="error")
                return
        
        update_splash(60, "Registering agent…")
        if self.agent_path:
            try:
                result = await self.daemon_client.register_agent(self.agent_path)
                update_splash(90, f"Loading @{result.get('name', 'agent')}…")
                if result.get("name"):
                    self.agent_name = result["name"]
                    prompt = self.query_one(PromptInput)
                    prompt.update_agent(
                        self.agent_name,
                        str(self.agent_path.parent) if self.agent_path else ""
                    )
                # Fetch agent commands
                update_splash(95, "Loading commands…")
                await self._fetch_agent_commands()
                
                update_splash(100, "Ready!")
                
                # Force connection status update
                try:
                    self.query_one("#connection-status", ConnectionStatus).is_connected = True
                except:
                    pass
                    
                await asyncio.sleep(0.3)  # Brief pause to show "Ready!"
            except Exception as e:
                self.notify(f"{e}", severity="warning")
    
    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        
        input_widget = self.query_one("#chat-input", ChatInput)
        input_widget.value = ""
        
        prompt = self.query_one(PromptInput)
        prompt.add_to_history(text)
        
        if text.startswith("/"):
            await self._handle_slash_command(text)
            return
        
        container = self.query_one("#chat-container", ScrollableContainer)
        await container.mount(UserMessage(text))
        container.scroll_end(animate=False)
        
        self.messages.append({"role": "user", "content": text})
        self._stream_response(text)
    
    async def _handle_slash_command(self, text: str) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        parts = text[1:].strip().split()
        cmd = parts[0].lower() if parts else ""
        sub = parts[1].lower() if len(parts) > 1 else ""
        args = " ".join(parts[2:]) if len(parts) > 2 else ""
        
        # Check if this is an agent command first
        # Try full path (cmd/sub), then just cmd
        cmd_path = f"{cmd}/{sub}" if sub else cmd
        
        # Find matching agent command
        agent_cmd = None
        agent_args = args
        if cmd_path in self._agent_commands:
            agent_cmd = cmd_path
        elif cmd in self._agent_commands:
            agent_cmd = cmd
            # sub becomes part of args for single-word commands
            if sub:
                agent_args = f"{sub} {args}".strip()
        
        if agent_cmd:
            await self._execute_agent_command(container, agent_cmd, agent_args)
            return
        
        if cmd in ("help", "h", "?"):
            # Build organized help text - most used commands first
            help_lines = []
            
            # Group all agent commands by prefix
            cmd_groups: Dict[str, List[tuple]] = {}
            seen_paths = set()
            
            for path, info in self._agent_commands.items():
                # Skip duplicates (aliases registered separately)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                
                if "/" in path:
                    prefix = path.split("/")[0].title()
                else:
                    prefix = "Other"
                
                if prefix not in cmd_groups:
                    cmd_groups[prefix] = []
                
                desc = info.get("description", "") if isinstance(info, dict) else ""
                cmd_display = format_command(path)
                cmd_groups[prefix].append((cmd_display, desc))
            
            def format_cmd_list(items: List[tuple], col_width: int = 24) -> List[str]:
                """Format command list with aligned descriptions."""
                lines = []
                for cmd, desc in sorted(items):
                    padding = " " * max(1, col_width - len(cmd))
                    lines.append(f"  [dim]{cmd}[/]{padding}{desc}")
                return lines
            
            # Built-in commands first
            help_lines.append("[bold]Commands:[/]")
            help_lines.extend(format_cmd_list([
                ("/clear", "Clear chat"),
                ("/quit", "Exit"),
            ]))
            
            # Agent management
            help_lines.append("\n[bold]Agents:[/]")
            help_lines.extend(format_cmd_list([
                ("/agent list", "List agents"),
                ("/agent switch <n>", "Switch agent"),
                ("/agent info", "Current agent"),
            ]))
            
            # Settings
            help_lines.append("\n[bold]Settings:[/]")
            help_lines.extend(format_cmd_list([
                ("/settings thinking", "Toggle thinking"),
                ("/settings tools", "Toggle tools"),
                ("/settings theme", "Pick theme"),
                ("/system status", "Daemon status"),
            ]))
            
            # Shortcuts (compact)
            help_lines.append("\n[bold]Keys:[/]")
            help_lines.append("  [dim]Ctrl+Q[/] quit  [dim]Ctrl+P[/] palette  [dim]↑↓[/] history  [dim]→[/] complete")
            
            # Agent commands at the end (dynamic)
            if "Session" in cmd_groups:
                help_lines.append("\n[bold]Session:[/]")
                help_lines.extend(format_cmd_list(cmd_groups["Session"]))
                del cmd_groups["Session"]
            
            if "Checkpoint" in cmd_groups:
                help_lines.append("\n[bold]Checkpoint:[/]")
                help_lines.extend(format_cmd_list(cmd_groups["Checkpoint"]))
                del cmd_groups["Checkpoint"]
            
            # Other dynamic command groups (skip "Other")
            for group in sorted(cmd_groups.keys()):
                if group != "Other":
                    help_lines.append(f"\n[bold]{group}:[/]")
                    help_lines.extend(format_cmd_list(cmd_groups[group]))
            
            await container.mount(Static("\n".join(help_lines), classes="help-text"))
        
        elif cmd in ("clear", "c"):
            for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock, StreamingIndicator"):
                w.remove()
            self.messages.clear()
        
        # System commands
        elif cmd == "system":
            if sub == "status":
                await self._show_system_status(container)
            else:
                await container.mount(Static("[dim]/system status[/dim]", classes="cmd-output"))
        
        # Agent commands
        elif cmd == "agent":
            if sub == "info":
                await self._show_agent_info(container)
            elif sub == "list":
                await self._list_agents(container)
            elif sub == "switch":
                await self._switch_agent(container, args)
            else:
                await container.mount(Static("[dim]/agent list | switch | info[/dim]", classes="cmd-output"))
        
        elif cmd == "settings":
            if sub == "thinking":
                self.expand_thinking = not self.expand_thinking
                self.notify(f"Thinking: {'on' if self.expand_thinking else 'off'}")
            elif sub == "tools":
                self.show_tool_details = not self.show_tool_details
                self.notify(f"Tools: {'on' if self.show_tool_details else 'off'}")
            elif sub == "theme":
                self.action_pick_theme()
            else:
                await container.mount(Static(f"""[dim]thinking: {'on' if self.expand_thinking else 'off'}
tools: {'on' if self.show_tool_details else 'off'}
theme: {self.theme}[/]""", classes="cmd-output"))

        elif cmd in ("quit", "q", "exit"):
            self.exit()
        
        else:
            # Check if this is a command prefix that has subcommands
            # e.g., /session should show /session save, /session load, etc.
            matching_paths = [
                path for path in self._agent_commands 
                if path.startswith(f"{cmd}/") or path == cmd
            ]
            
            if matching_paths:
                # Show available subcommands for this prefix
                lines = [f"[bold]/{cmd}:[/]"]
                for path in sorted(matching_paths):
                    info = self._agent_commands.get(path, {})
                    desc = info.get("description", "") if isinstance(info, dict) else ""
                    cmd_display = format_command(path)
                    padding = " " * max(1, 24 - len(cmd_display))
                    lines.append(f"  [dim]{cmd_display}[/]{padding}{desc}")
                await container.mount(Static("\n".join(lines), classes="cmd-output"))
            else:
                self.notify(f"Unknown: /{cmd}", severity="warning")
        
        container.scroll_end(animate=False)
    
    async def _execute_agent_command(self, container: ScrollableContainer, cmd_path: str, args: str) -> None:
        """Execute a command from the agent's skills via daemon."""
        if not self.daemon_client:
            await container.mount(Static("[dim]Not connected[/dim]", classes="cmd-output"))
            return
        
        try:
            # Get command info to check parameters
            cmd_info = self._agent_commands.get(cmd_path, {})
            params = cmd_info.get("parameters", {})
            required = cmd_info.get("required", [])
            
            # Parse args into a dict
            data = {}
            if args:
                # Try to parse as JSON first
                if args.startswith("{"):
                    data = json.loads(args)
                else:
                    # For simple args, try to map to first required parameter
                    if required:
                        data[required[0]] = args
                    else:
                        # Check if there's a single parameter
                        param_names = list(params.keys())
                        if len(param_names) == 1:
                            data[param_names[0]] = args
                        else:
                            # Default: pass as first meaningful param name
                            # IDs first (for restore/load), then descriptive names
                            for name in ["checkpoint_id", "session_id", "id", "name", "description", "arg"]:
                                if name in params:
                                    data[name] = args
                                    break
                            else:
                                data["arg"] = args
            
            result = await self.daemon_client.execute_command(
                self.agent_name, f"/{cmd_path}", data
            )
            
            # The daemon returns {"result": actual_result}
            actual_result = result.get("result", result) if isinstance(result, dict) else result
            
            # Display result - prefer 'display' field (markdown) if present
            if isinstance(actual_result, dict):
                if actual_result.get("display"):
                    # Use markdown display field - this is the preferred format
                    # Commands should return display in markdown for uniform rendering
                    await container.mount(Static(actual_result["display"], classes="cmd-output"))
                elif actual_result.get("error"):
                    await container.mount(Static(f"[red]Error: {actual_result['error']}[/red]", classes="cmd-output"))
                else:
                    # Fallback: format as JSON for commands that don't provide display
                    formatted = json.dumps(actual_result, indent=2)
                    await container.mount(Static(f"[dim]{formatted}[/dim]", classes="cmd-output"))
            else:
                await container.mount(Static(f"{actual_result}", classes="cmd-output"))
            
            container.scroll_end(animate=False)
            
            # Invalidate completions cache for this command (execution may alter state)
            # Also invalidate parent command group cache
            if cmd_path in self._command_completions:
                del self._command_completions[cmd_path]
            # Invalidate parent group (e.g., "session" when executing "session/load")
            if "/" in cmd_path:
                parent = cmd_path.rsplit("/", 1)[0]
                if parent in self._command_completions:
                    del self._command_completions[parent]
            
        except Exception as e:
            await container.mount(Static(f"[red]Error: {e}[/red]", classes="cmd-output"))
    
    async def _show_system_status(self, container: ScrollableContainer) -> None:
        """Show system status."""
        lines = []
        
        # Daemon status
        if self.daemon_client:
            try:
                health = await self.daemon_client.health()
                lines.append(f"[green]●[/] Daemon connected")
                if health.get("agents"):
                    lines.append(f"  Agents: {len(health['agents'])}")
            except:
                lines.append(f"[red]○[/] Daemon disconnected")
        else:
            lines.append(f"[dim]○[/] Daemon not initialized")
        
        # Session info
        lines.append(f"  Session: [dim]{self.session_id[:8] if self.session_id else 'none'}[/]")
        lines.append(f"  Messages: {len(self.messages)}")
        lines.append(f"  Tokens: {self.input_tokens}↓ {self.output_tokens}↑")
        
        await container.mount(Static("\n".join(lines), classes="cmd-output"))
    
    async def _show_agent_info(self, container: ScrollableContainer) -> None:
        """Show agent info."""
        lines = [f"[bold]@{self.agent_name}[/]"]
        if self.agent_path:
            lines.append(f"  [dim]{self.agent_path}[/]")
        
        if self.daemon_client and self.agent_name:
            try:
                info = await self.daemon_client.get_agent(self.agent_name)
                if info:
                    if info.get('model'):
                        lines.append(f"  Model: {info.get('model')}")
                    if info.get('skills'):
                        lines.append(f"  Skills: {', '.join(info['skills'])}")
            except:
                pass
        
        # Show available commands count
        if self._agent_commands:
            lines.append(f"  Commands: {len(self._agent_commands)}")
        
        await container.mount(Static("\n".join(lines), classes="cmd-output"))
    
    async def _list_agents(self, container: ScrollableContainer) -> None:
        """List registered agents."""
        if not self.daemon_client:
            await container.mount(Static("[dim]Not connected[/dim]", classes="cmd-output"))
            return
        
        try:
            agents = await self.daemon_client.list_agents()
            if not agents:
                await container.mount(Static("[dim]No agents[/dim]", classes="cmd-output"))
                return
            
            lines = []
            for agent in agents:
                name = agent.get("name", "?")
                is_current = "[green]●[/]" if name == self.agent_name else "[dim]○[/]"
                lines.append(f"{is_current} @{name}")
            
            await container.mount(Static("\n".join(lines), classes="cmd-output"))
        except Exception as e:
            await container.mount(Static(f"[red]Error: {e}[/red]", classes="cmd-output"))
    
    async def _switch_agent(self, container: ScrollableContainer, agent_name: str) -> None:
        """Switch to another agent."""
        if not agent_name:
            await container.mount(Static("[dim]/agent switch <name>[/dim]", classes="cmd-output"))
            return
        
        if not self.daemon_client:
            await container.mount(Static("[dim]Not connected[/dim]", classes="cmd-output"))
            return
        
        try:
            # Check if agent exists
            info = await self.daemon_client.get_agent(agent_name)
            if not info:
                await container.mount(Static(f"[red]Agent not found: {agent_name}[/red]", classes="cmd-output"))
                return
            
            # Switch
            self.agent_name = agent_name
            self.agent_path = Path(info.get("path")) if info.get("path") else None
            
            # Refresh agent commands
            await self._fetch_agent_commands()
            
            # Update UI
            prompt = self.query_one(PromptInput)
            prompt.update_agent(self.agent_name, str(self.agent_path) if self.agent_path else "")
            
            # Clear messages for new agent context
            self.messages.clear()
            self.session_id = None
            self._init_session_manager()
            
            await container.mount(Static(f"[green]✓[/] @{agent_name}", classes="cmd-output"))
            self.notify(f"@{agent_name}")
        except Exception as e:
            await container.mount(Static(f"[red]Error: {e}[/red]", classes="cmd-output"))
    
    @work(exclusive=True)
    async def _stream_response(self, text: str) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        prompt = self.query_one(PromptInput)
        
        # Show spinner
        prompt.is_busy = True
        
        self._streaming_widget = StreamingIndicator()
        await container.mount(self._streaming_widget)
        container.scroll_end(animate=False)
        
        response_text = ""
        self._active_tools = {}
        self._tool_index_to_id: Dict[int, str] = {}  # Map index to tool id for incremental updates
        thinking_content = ""
        thinking_start = None
        
        try:
            if self.use_daemon and self.daemon_client:
                import time
                async for chunk_str in self.daemon_client.chat_stream(
                    self.agent_name, text, self.messages[:-1]
                ):
                    if chunk_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(chunk_str)
                        
                        if chunk.get("object") == "metadata":
                            mtype = chunk.get("type")
                            payload = chunk.get("payload", {})
                            
                            if mtype == "tool_start":
                                self._active_tools[payload["id"]] = {
                                    "name": payload["name"],
                                    "args": payload["arguments"],
                                    "status": "running",
                                    "result": ""
                                }
                                self._streaming_widget.add_tool(
                                    payload["id"], payload["name"], payload["arguments"]
                                )
                            elif mtype == "tool_result":
                                # Find the tool call to update - handle duplicate IDs (call_0)
                                # by matching ID and finding first without a result
                                result_id = payload["id"]
                                target_key = None
                                for key, tool in self._active_tools.items():
                                    orig_id = tool.get("original_id", key)
                                    if orig_id == result_id and not tool.get("result"):
                                        target_key = key
                                        break
                                
                                if target_key:
                                    self._active_tools[target_key]["status"] = payload["status"]
                                    self._active_tools[target_key]["result"] = payload.get("result", "")
                                    self._streaming_widget.complete_tool(
                                        result_id, payload["status"], payload.get("result", "")
                                    )
                            elif mtype == "thought_start":
                                thinking_start = time.time()
                            continue
                        
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            
                            # Track tool calls from standard streaming
                            # Note: OpenAI streams tool calls incrementally:
                            # - First chunk has id, function.name, and partial arguments
                            # - Subsequent chunks only have index and function.arguments
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    tc_index = tc.get("index", 0)
                                    tc_id = tc.get("id")
                                    func = tc.get("function", {})
                                    
                                    if tc_id and func.get("name"):
                                        # Check if we already have a completed/different tool with same ID
                                        # (happens in agentic loops where Gemini reuses call_0)
                                        existing = self._active_tools.get(tc_id)
                                        if existing and existing.get("result") and existing.get("name") != func["name"]:
                                            # Different tool with same ID - use unique tracking key
                                            tracking_key = f"{tc_id}_{len(self._active_tools)}"
                                        else:
                                            tracking_key = tc_id
                                        
                                        # Register the tool call
                                        self._tool_index_to_id[tc_index] = tracking_key
                                        self._active_tools[tracking_key] = {
                                            "name": func.get("name", ""),
                                            "args": func.get("arguments", ""),
                                            "status": "running",
                                            "result": "",
                                            "original_id": tc_id  # Keep original for result matching
                                        }
                                        self._streaming_widget.add_tool(
                                            tc_id, func["name"], func.get("arguments", "")
                                        )
                                    elif tc_id and not func.get("name"):
                                        # Has ID but no name yet - register with ID as key
                                        self._tool_index_to_id[tc_index] = tc_id
                                        if tc_id not in self._active_tools:
                                            self._active_tools[tc_id] = {
                                                "name": "",
                                                "args": func.get("arguments", ""),
                                                "status": "running",
                                                "result": "",
                                                "original_id": tc_id
                                            }
                                    elif tc_index in self._tool_index_to_id:
                                        # Incremental update - append arguments
                                        existing_id = self._tool_index_to_id[tc_index]
                                        if existing_id in self._active_tools and func.get("arguments"):
                                            self._active_tools[existing_id]["args"] += func["arguments"]
                            
                            if content:
                                response_text += content
                                self._streaming_widget.update_content(response_text)
                                
                                # Extract thinking content for real-time display
                                think_match = re.search(r'<think>(.*?)(?:</think>|$)', response_text, re.DOTALL)
                                if think_match:
                                    thinking_content = think_match.group(1)
                                    self._streaming_widget.update_thinking(thinking_content)
                                
                                container.scroll_end(animate=False)
                    
                    except json.JSONDecodeError:
                        continue
            
            # Remove streaming widget
            if self._streaming_widget:
                self._streaming_widget.remove()
                self._streaming_widget = None
            
            # Add thinking blocks
            think_matches = list(re.finditer(r'<think>(.*?)</think>', response_text, re.DOTALL))
            for m in think_matches:
                content = m.group(1).strip()
                matches = re.findall(r'\*\*([^\*]+)\*\*', content)
                subtitle = matches[-1].strip() if matches else ""
                duration = time.time() - thinking_start if thinking_start else 0
                await container.mount(ThinkingBlock(
                    content=content,
                    subtitle=subtitle,
                    duration=duration,
                    expanded=self.expand_thinking
                ))
            
            # Add tool call blocks
            for tool_id, tool_data in self._active_tools.items():
                await container.mount(ToolCallBlock(
                    name=tool_data["name"],
                    args=tool_data["args"],
                    result=tool_data.get("result", ""),
                    status=tool_data["status"],
                    expanded=self.show_tool_details
                ))
            
            # Add assistant message (rendered as markdown)
            if response_text:
                await container.mount(AssistantMessage(response_text))
                # Save with tool calls for session restoration
                msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
                if self._active_tools:
                    msg["tool_calls"] = [
                        {
                            "id": tid,
                            "function": {
                                "name": td.get("name", ""),
                                "arguments": td.get("args", ""),
                            },
                            "result": td.get("result", ""),
                        }
                        for tid, td in self._active_tools.items()
                    ]
                self.messages.append(msg)
            
            container.scroll_end(animate=False)
        
        except Exception as e:
            if self._streaming_widget:
                self._streaming_widget.remove()
            await container.mount(Static(f"[red]Error: {e}[/]"))
            container.scroll_end(animate=False)
        
        finally:
            # Hide spinner and restore focus
            prompt.is_busy = False
            self.query_one("#chat-input", ChatInput).focus()
            
            # Auto-save session
            self._save_session()
    
    def action_quit(self) -> None:
        self.exit()
    
    def action_clear_prompt(self) -> None:
        """Clear the current prompt input."""
        try:
            input_widget = self.query_one("#chat-input", ChatInput)
            input_widget.value = ""
            input_widget.focus()
        except:
            pass
    
    def action_clear(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock, StreamingIndicator, Static"):
            if w.id != "welcome":
                w.remove()
        welcome = self.query_one("#welcome", Static)
        welcome.update(self._get_welcome())
    
    def action_new(self) -> None:
        """Start a new session."""
        import uuid
        self.messages.clear()
        self.session_id = str(uuid.uuid4())
        self.input_tokens = 0
        self.output_tokens = 0
        container = self.query_one("#chat-container", ScrollableContainer)
        for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock"):
            w.remove()
        self.notify("New session")
    
    def action_toggle_thinking(self) -> None:
        self.expand_thinking = not self.expand_thinking
        self.notify(f"Thinking: {'on' if self.expand_thinking else 'off'}")
    
    def action_toggle_tools(self) -> None:
        self.show_tool_details = not self.show_tool_details
        self.notify(f"Tools: {'on' if self.show_tool_details else 'off'}")
    
    def action_pick_theme(self) -> None:
        """Open theme picker dialog."""
        self.push_screen(ThemePickerScreen())
    
    def action_cancel(self) -> None:
        pass
    
    def action_help(self) -> None:
        asyncio.create_task(self._handle_slash_command("/help"))
    
    async def on_chat_input_history_up(self, event: ChatInput.HistoryUp) -> None:
        """Handle UP key - cycle through history or suggestions.
        
        Empty input: shows history item in placeholder (RIGHT to accept)
        Has text: cycles through suggestions (ghost text only)
        """
        prompt = self.query_one(PromptInput)
        input_widget = self.query_one("#chat-input", ChatInput)
        current = input_widget.value
        
        if not current:
            # Empty input - cycle history and show in placeholder
            suggestion = prompt._suggester.cycle_history(direction=1)
            if suggestion:
                input_widget.placeholder = suggestion
            return
        
        # Fetch completions for slash commands
        if current.startswith("/") and len(current) > 1:
            await self._ensure_command_completions(current)
        
        _, suggestion = prompt.cycle_suggestion_up(current)
        
        if suggestion is not None:
            # Has text - show as ghost text by forcing refresh
            saved_suggestion = prompt._suggester._current_suggestion
            saved_prefix = prompt._suggester._current_prefix
            saved_index = prompt._suggester._match_index
            saved_matches = prompt._suggester._matches
            
            input_widget.value = current + " "
            prompt._suggester._current_suggestion = saved_suggestion
            prompt._suggester._current_prefix = saved_prefix
            prompt._suggester._match_index = saved_index
            prompt._suggester._matches = saved_matches
            input_widget.value = current
            input_widget.cursor_position = len(current)
    
    async def on_chat_input_history_down(self, event: ChatInput.HistoryDown) -> None:
        """Handle DOWN key - cycle through history or suggestions.
        
        Empty input: shows history item in placeholder (RIGHT to accept)
        Has text: cycles through suggestions (ghost text only)
        """
        prompt = self.query_one(PromptInput)
        input_widget = self.query_one("#chat-input", ChatInput)
        current = input_widget.value
        
        if not current:
            # Empty input - cycle history and show in placeholder
            suggestion = prompt._suggester.cycle_history(direction=-1)
            if suggestion:
                input_widget.placeholder = suggestion
            elif suggestion == "":
                # Past newest - reset to default placeholder
                input_widget.placeholder = f"Message @{self.agent_name} or /command…"
                prompt._suggester._history_cycle_index = -1
            return
        
        # Fetch completions for slash commands
        if current.startswith("/") and len(current) > 1:
            await self._ensure_command_completions(current)
        
        _, suggestion = prompt.cycle_suggestion_down(current)
        
        if suggestion is not None:
            # Has text - show as ghost text by forcing refresh
            saved_suggestion = prompt._suggester._current_suggestion
            saved_prefix = prompt._suggester._current_prefix
            saved_index = prompt._suggester._match_index
            saved_matches = prompt._suggester._matches
            
            input_widget.value = current + " "
            prompt._suggester._current_suggestion = saved_suggestion
            prompt._suggester._current_prefix = saved_prefix
            prompt._suggester._match_index = saved_index
            prompt._suggester._matches = saved_matches
            input_widget.value = current
            input_widget.cursor_position = len(current)
    
    async def _ensure_command_completions(self, input_text: str) -> None:
        """Ensure completions are fetched for commands being typed.
        
        Called every time user presses UP/DOWN on a slash command.
        Tries the most specific command path first (e.g., session/load before session).
        """
        if not input_text.startswith("/"):
            return
        
        parts = input_text[1:].strip().split()
        if not parts:
            return
        
        # Build command path from parts (most specific)
        # /session load -> session/load
        # /session -> session
        cmd_path = "/".join(parts)
        
        # Only fetch if not already cached with positive results
        if cmd_path not in self._command_completions or not self._command_completions.get(cmd_path):
            completions = await self._fetch_completions_for_command(cmd_path)
            if completions:
                # Update suggester
                try:
                    prompt = self.query_one(PromptInput)
                    prompt.update_command_completions(self._command_completions)
                except Exception:
                    pass
    
    def on_chat_input_accept_suggestion(self, event: ChatInput.AcceptSuggestion) -> None:
        """Handle RIGHT key - accept current suggestion or placeholder."""
        prompt = self.query_one(PromptInput)
        input_widget = self.query_one("#chat-input", ChatInput)
        current = input_widget.value
        
        # If input is empty and placeholder shows history, accept the placeholder
        if not current and prompt._suggester._history_cycle_index >= 0:
            placeholder = input_widget.placeholder
            default_placeholder = f"Message @{self.agent_name} or /command…"
            if placeholder and placeholder != default_placeholder:
                input_widget.value = placeholder
                input_widget.cursor_position = len(placeholder)
                # Reset placeholder and history cycling
                input_widget.placeholder = default_placeholder
                prompt._suggester._history_cycle_index = -1
                return
        
        suggestion = prompt.get_current_suggestion()
        
        # If no current suggestion, try to get first match directly
        # This handles the case where get_suggestion hasn't been called yet
        if not suggestion:
            matches = prompt._suggester.get_matches_for_prefix(current)
            if matches:
                suggestion = matches[0]
        
        if suggestion:
            input_widget.value = suggestion
            input_widget.cursor_position = len(suggestion)
            
            # Don't fully reset - just update prefix to match new value
            # This allows immediate re-completion for the new prefix
            prompt._suggester._current_prefix = suggestion
            prompt._suggester._current_suggestion = None
            prompt._suggester._match_index = 0
            prompt._suggester._matches = []
            
            # Force Textual to refresh ghost text for the new value
            # by triggering value change
            input_widget.value = suggestion + " "
            prompt._suggester._current_prefix = suggestion
            input_widget.value = suggestion
            input_widget.cursor_position = len(suggestion)
    
    def action_scroll_up(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_up(animate=False)
    
    def action_scroll_down(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_down(animate=False)
    
    def action_page_up(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_page_up(animate=False)
    
    def action_page_down(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_page_down(animate=False)


async def run_tui(
    agent_name: str = "assistant",
    agent_path: Optional[Path] = None,
    daemon_client = None,
    use_daemon: bool = True
) -> None:
    """Run the WebAgents TUI."""
    app = WebAgentsTUI(
        agent_name=agent_name,
        agent_path=agent_path,
        daemon_client=daemon_client,
        use_daemon=use_daemon
    )
    await app.run_async()
