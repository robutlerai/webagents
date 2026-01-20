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
from textual.widgets import Static, Input, Footer, Header, Markdown, OptionList, LoadingIndicator, ProgressBar
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


class SlashCommandSuggester(Suggester):
    """Suggester for slash commands."""
    
    COMMANDS = [
        "/help", "/clear", "/new", "/quit", "/history", "/tokens",
        "/settings", "/settings thinking", "/settings tools", "/theme",
    ]
    
    async def get_suggestion(self, value: str) -> Optional[str]:
        if not value.startswith("/"):
            return None
        value_lower = value.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(value_lower) and cmd != value_lower:
                return cmd
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
    
    #splash-status {
        text-align: center;
        color: $text-muted;
        padding: 1 0;
    }
    
    ProgressBar {
        padding: 1 2;
    }
    """
    
    progress = reactive(0.0)
    status = reactive("Initializing…")
    
    def compose(self) -> ComposeResult:
        with Vertical(id="splash-container"):
            yield Static("""╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗
║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗
╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝""", id="splash-logo")
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


class HeaderClock(Static):
    """Simple clock widget."""
    
    def on_mount(self) -> None:
        self.set_interval(1, self.update_time)
        self.update_time()
        
    def update_time(self) -> None:
        self.update(datetime.datetime.now().strftime("%H:%M:%S"))


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
    
    CustomHeader HeaderClock {
        dock: right;
        padding: 0 1;
        content-align: center middle;
        width: auto;
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

    def compose(self) -> ComposeResult:
        yield ConnectionStatus(id="connection-status")
        yield Static("WebAgents", id="header-title")
        yield HeaderClock()


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
    """Command palette provider for WebAgents."""
    
    COMMANDS = [
        ("Clear Chat", "clear", "Clear all messages"),
        ("New Session", "new", "Start fresh session"),
        ("Toggle Thinking", "toggle_thinking", "Show/hide reasoning"),
        ("Toggle Tools", "toggle_tools", "Show/hide tool details"),
        ("🎨 Pick Theme", "pick_theme", "Change UI theme"),
        ("Help", "help", "Show help"),
        ("Quit", "quit", "Exit"),
    ]
    
    async def discover(self) -> Hits:
        """Show commands when palette first opens."""
        app = self.app
        for name, action, help_text in self.COMMANDS:
            yield DiscoveryHit(
                name,
                lambda a=action: getattr(app, f"action_{a}")(),
                help=help_text,
            )
    
    async def search(self, query: str) -> Hits:
        """Search commands."""
        app = self.app
        matcher = self.matcher(query)
        
        for name, action, help_text in self.COMMANDS:
            score = matcher.match(name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(name),
                    lambda a=action: getattr(app, f"action_{a}")(),
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


class PromptInput(Static):
    """Input prompt area with vertical bars."""
    
    is_busy = reactive(False)
    
    def __init__(self, agent_name: str = "assistant", agent_path: str = ""):
        super().__init__()
        self._agent_name = agent_name
        self._agent_path = agent_path
        self._history: List[str] = []
        self._history_index = -1
    
    def compose(self) -> ComposeResult:
        yield Static("┃", id="bar-top", classes="accent-bar")
        with Horizontal(id="input-row"):
            yield Static("┃", id="bar-left", classes="accent-bar")
            yield Input(
                placeholder="Type a message or /help…",
                id="chat-input",
                suggester=SlashCommandSuggester()
            )
        yield Static("┃", id="bar-bottom", classes="accent-bar")
        
        # Agent info
        path_display = self._agent_path
        if len(path_display) > 45:
            path_display = "…" + path_display[-42:]
        yield Static(f"┃  @{self._agent_name}  {path_display}", id="agent-info", classes="agent-info-bar")
    
    def watch_is_busy(self, value: bool) -> None:
        try:
            input_widget = self.query_one("#chat-input", Input)
            input_widget.disabled = value
        except:
            pass
    
    def update_agent(self, name: str, path: str = "") -> None:
        self._agent_name = name
        self._agent_path = path
        agent_info = self.query_one("#agent-info", Static)
        path_display = path if len(path) <= 45 else "…" + path[-42:]
        agent_info.update(f"┃  @{name}  {path_display}")
    
    def add_to_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = len(self._history)
    
    def get_prev_history(self) -> Optional[str]:
        if self._history and self._history_index > 0:
            self._history_index -= 1
            return self._history[self._history_index]
        return None
    
    def get_next_history(self) -> Optional[str]:
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            return self._history[self._history_index]
        elif self._history_index == len(self._history) - 1:
            self._history_index = len(self._history)
            return ""
        return None


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
        color: $text-muted;
        padding: 1;
    }
    
    Markdown {
        margin: 0;
        padding: 0;
        color: $text;
    }
    """
    
    ENABLE_COMMAND_PALETTE = True
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "toggle_thinking", "Thinking"),
        Binding("ctrl+d", "toggle_tools", "Tools"),
        Binding("ctrl+shift+p", "pick_theme", "Theme"),
        Binding("escape", "cancel", "Cancel"),
        Binding("f1", "help", "Help"),
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
        Binding("ctrl+up", "scroll_up", show=False),
        Binding("ctrl+down", "scroll_down", show=False),
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
        self._streaming_widget: Optional[StreamingIndicator] = None
    
    def compose(self) -> ComposeResult:
        yield CustomHeader()
        
        with ScrollableContainer(id="chat-container"):
            yield Static(self._get_welcome(), id="welcome")
        
        yield PromptInput(
            agent_name=self.agent_name,
            agent_path=str(self.agent_path) if self.agent_path else ""
        )
        yield Footer()
    
    def _get_welcome(self) -> str:
        return """
[bold $primary]╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗[/]
[bold $primary]║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗[/]
[bold $primary]╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝[/]

[$text-muted]Type a message, /help, or Ctrl+P for commands[/]
"""
    
    async def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        if self.use_daemon and self.daemon_client:
            # Show splash screen immediately and run connection in background
            splash = SplashScreen()
            self.push_screen(splash)
            self.run_worker(self._startup_sequence(splash))
    
    async def _startup_sequence(self, splash: SplashScreen) -> None:
        """Run startup connection sequence."""
        # Ensure splash is visible for a moment
        await asyncio.sleep(1.0)
        
        # Connect and register
        await self._ensure_daemon_and_agent(splash)
        
        # Ensure splash stays for completion
        await asyncio.sleep(0.5)
        self.pop_screen()
        self.query_one("#chat-input", Input).focus()
    
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
                update_splash(100, "Ready!")
                
                # Force connection status update
                try:
                    self.query_one("#connection-status", ConnectionStatus).is_connected = True
                except:
                    pass
                    
                await asyncio.sleep(0.3)  # Brief pause to show "Ready!"
            except Exception as e:
                self.notify(f"{e}", severity="warning")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        
        input_widget = self.query_one("#chat-input", Input)
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
        parts = text[1:].strip().lower().split()
        cmd = parts[0] if parts else ""
        args = " ".join(parts[1:]) if len(parts) > 1 else ""
        
        if cmd in ("help", "h", "?"):
            await container.mount(Static("""
[bold]Commands:[/]
  [$text-muted]/help[/]              Show this help
  [$text-muted]/clear[/]             Clear chat history
  [$text-muted]/new[/]               Start new session
  [$text-muted]/theme[/]             Pick UI theme
  [$text-muted]/quit[/]              Exit

[bold]Settings:[/]
  [$text-muted]/settings thinking[/]  Toggle thinking blocks
  [$text-muted]/settings tools[/]     Toggle tool details

[bold]Shortcuts:[/]
  [$primary]Ctrl+P[/]    Command palette
  [$primary]Ctrl+L[/]    Clear chat
  [$primary]Ctrl+T[/]    Toggle thinking
  [$primary]Ctrl+D[/]    Toggle tools
  [$primary]↑ / ↓[/]     Input history
  [$primary]Ctrl+↑/↓[/]  Scroll messages
  [$primary]PgUp/Dn[/]   Page scroll
""", classes="help-text"))
        
        elif cmd in ("clear", "c"):
            for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock, StreamingIndicator"):
                w.remove()
            self.messages.clear()
            self.notify("Cleared")
        
        elif cmd in ("new", "n"):
            self.messages.clear()
            for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock"):
                w.remove()
            self.notify("New session")
        
        elif cmd == "settings":
            if args == "thinking":
                self.expand_thinking = not self.expand_thinking
                self.notify(f"Thinking: {'on' if self.expand_thinking else 'off'}")
            elif args == "tools":
                self.show_tool_details = not self.show_tool_details
                self.notify(f"Tools: {'on' if self.show_tool_details else 'off'}")
            else:
                await container.mount(Static(f"""[#666666]Settings:
  thinking: {'on' if self.expand_thinking else 'off'}
  tools: {'on' if self.show_tool_details else 'off'}[/]"""))
        
        elif cmd == "theme":
            self.action_pick_theme()
        
        elif cmd in ("quit", "q", "exit"):
            self.exit()
        
        else:
            self.notify(f"Unknown: /{cmd}", severity="warning")
        
        container.scroll_end(animate=False)
    
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
                                if payload["id"] in self._active_tools:
                                    self._active_tools[payload["id"]]["status"] = payload["status"]
                                    self._active_tools[payload["id"]]["result"] = payload.get("result", "")
                                    self._streaming_widget.complete_tool(
                                        payload["id"], payload["status"], payload.get("result", "")
                                    )
                            elif mtype == "thought_start":
                                thinking_start = time.time()
                            continue
                        
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            
                            # Track tool calls from standard streaming
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    tc_id = tc.get("id")
                                    func = tc.get("function", {})
                                    if tc_id and func.get("name"):
                                        self._active_tools[tc_id] = {
                                            "name": func["name"],
                                            "args": func.get("arguments", ""),
                                            "status": "running",
                                            "result": ""
                                        }
                                        self._streaming_widget.add_tool(
                                            tc_id, func["name"], func.get("arguments", "")
                                        )
                            
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
                self.messages.append({"role": "assistant", "content": response_text})
            
            container.scroll_end(animate=False)
        
        except Exception as e:
            if self._streaming_widget:
                self._streaming_widget.remove()
            await container.mount(Static(f"[red]Error: {e}[/]"))
            container.scroll_end(animate=False)
        
        finally:
            # Hide spinner and restore focus
            prompt.is_busy = False
            self.query_one("#chat-input", Input).focus()
    
    def action_quit(self) -> None:
        self.exit()
    
    def action_clear(self) -> None:
        container = self.query_one("#chat-container", ScrollableContainer)
        for w in container.query("UserMessage, AssistantMessage, ThinkingBlock, ToolCallBlock, StreamingIndicator, Static"):
            if w.id != "welcome":
                w.remove()
        welcome = self.query_one("#welcome", Static)
        welcome.update(self._get_welcome())
    
    def action_new(self) -> None:
        self.messages.clear()
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
    
    def action_history_prev(self) -> None:
        prompt = self.query_one(PromptInput)
        prev = prompt.get_prev_history()
        if prev is not None:
            input_widget = self.query_one("#chat-input", Input)
            input_widget.value = prev
            input_widget.cursor_position = len(prev)
    
    def action_history_next(self) -> None:
        prompt = self.query_one(PromptInput)
        next_val = prompt.get_next_history()
        if next_val is not None:
            input_widget = self.query_one("#chat-input", Input)
            input_widget.value = next_val
            input_widget.cursor_position = len(next_val)
    
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
