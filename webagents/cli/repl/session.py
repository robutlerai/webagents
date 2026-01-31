"""
Interactive REPL Session

Prompt Toolkit + Rich for a premium terminal experience.
"""

import asyncio
import os
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion, WordCompleter, NestedCompleter
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from .slash_commands import SlashCommandRegistry, handle_slash_command
from ..ui.splash import print_splash, print_status_bar
from ..client.daemon_client import DaemonClient
from ..client.auto_start import ensure_daemon_running


# Custom prompt style - no backgrounds
PROMPT_STYLE = Style.from_dict({
    'prompt': 'bold',
    'prompt-bar': '#ff6b6b',  # Coral for user input bar only
    'prompt-agent': '#888888',
    'prompt-path': '#666666',
    'completion-menu.completion': '#aaaaaa',
    'completion-menu.completion.current': '#ffffff bold',
    'scrollbar.background': '',
    'scrollbar.button': '#666666',
    'bottom-toolbar': '',
    'bottom-toolbar.text': '#666666',
    'bottom-toolbar.right': '#888888',  
    'bottom-toolbar.status-on': '#888888',
    'bottom-toolbar.version': '#555555',
})

# Default system prompt for assistant mode
DEFAULT_INSTRUCTIONS = """You are a helpful AI assistant running in the WebAgents CLI.
Be concise and helpful. Use markdown for formatting. For newlines (e.g. poetry, lyrics, lists), use two spaces at the end of the line.
"""


from prompt_toolkit.completion import Completer, Completion

class SlashCommandCompleter(Completer):
    """Completer for slash commands with hierarchical support.
    
    Supports:
    - /h -> shows /help, /history
    - /agent -> shows /agent and /agent list, /agent info, /agent connect
    - /agent l -> shows /agent list
    - /agent list -> exact match
    """
    
    def __init__(self, commands):
        # commands is a list like ["/help", "/agent", "/agent/list", "/checkpoint/create", ...]
        self.commands = sorted(commands)
        # Build hierarchy: {"agent": ["list", "info", "connect"], ...}
        self.hierarchy = self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build command hierarchy from flat list."""
        hierarchy = {}
        for cmd in self.commands:
            parts = cmd.lstrip("/").split("/")
            if len(parts) == 1:
                # Top-level command
                if parts[0] not in hierarchy:
                    hierarchy[parts[0]] = []
            else:
                # Subcommand: /agent/list -> agent: [list]
                base = parts[0]
                sub = parts[1]
                if base not in hierarchy:
                    hierarchy[base] = []
                if sub not in hierarchy[base]:
                    hierarchy[base].append(sub)
        return hierarchy

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        if not text.startswith('/'):
            return
        
        # Handle trailing space case: "/agent " -> show subcommands
        if text.endswith(' '):
            # Parse base command (without trailing space)
            cmd_text = text.strip()[1:]  # Remove / and trailing space
            
            if cmd_text in self.hierarchy and self.hierarchy[cmd_text]:
                for sub in self.hierarchy[cmd_text]:
                    yield Completion(
                        sub,
                        start_position=0,  # Insert at cursor position
                        display=f"/{cmd_text} {sub}",
                        display_meta="subcommand"
                    )
            return
        
        # Remove leading /
        cmd_text = text[1:]
        
        # Check for space-separated composite commands: "/agent li"
        if ' ' in cmd_text:
            parts = cmd_text.split()
            base_cmd = parts[0]
            partial_sub = parts[1] if len(parts) > 1 else ""
            
            # Get subcommands for base command
            if base_cmd in self.hierarchy:
                for sub in self.hierarchy[base_cmd]:
                    if sub.startswith(partial_sub):
                        # Complete the subcommand part only
                        yield Completion(
                            sub,
                            start_position=-len(partial_sub),
                            display=f"/{base_cmd} {sub}",
                            display_meta="subcommand"
                        )
            return
        
        # Standard completion: /help, /agent, /checkpoint/create
        for cmd in self.commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta="command"
                )
        
        # Also suggest subcommands for partial matches like /agent -> /agent list
        if cmd_text in self.hierarchy and self.hierarchy[cmd_text]:
            for sub in self.hierarchy[cmd_text]:
                full_cmd = f"/{cmd_text} {sub}"
                yield Completion(
                    full_cmd,
                    start_position=-len(text),
                    display_meta="subcommand"
                )


class WebAgentsSession:
    """Interactive REPL session with an agent."""
    
    def __init__(self, agent_path: Optional[Path] = None):
        self.console = Console()
        
        # If no path provided, try to find default agent in current directory
        if agent_path is None:
            from ..loader.hierarchy import find_default_agent
            merged_default = find_default_agent(Path.cwd())
            if merged_default:
                agent_path = merged_default.path
                
        self.agent_path = agent_path
        self.agent_name = agent_path.stem if agent_path else "assistant"
        
        # Ensure history directory exists
        history_dir = Path.home() / ".webagents"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare completer for slash commands
        self.slash_commands = SlashCommandRegistry()
        # Build command list with both /agent/list and /agent list formats
        slash_cmds = []
        for cmd in self.slash_commands.list_commands().keys():
            slash_cmds.append(f"/{cmd}")
            # Also add space-separated format for hierarchical commands
            if "/" in cmd:
                slash_cmds.append(f"/{cmd.replace('/', ' ')}")
        self.completer = SlashCommandCompleter(slash_cmds)
        
        # Key bindings for multiline input
        kb = KeyBindings()

        @kb.add('enter')
        def _(event):
            """Submit on Enter."""
            event.current_buffer.validate_and_handle()

        @kb.add('escape', 'enter')
        def _(event):
            """Insert newline on Alt-Enter (Meta-Enter)."""
            event.current_buffer.insert_text('\n')
            
        # Display toggle shortcuts using Ctrl+T prefix (like tmux)
        # Ctrl+T then T = toggle toolcalls
        # Ctrl+T then R = toggle thinking  
        # Ctrl+T then D = toggle todos
        @kb.add('c-t', 't')
        def toggle_toolcalls(event):
            """Toggle tool call details (Ctrl+T, T)."""
            self.show_tool_details = not self.show_tool_details
            self._toggle_status = "Tool calls: " + ("expanded" if self.show_tool_details else "collapsed")
            event.app.invalidate()
        
        @kb.add('c-t', 'r')
        def toggle_thinking(event):
            """Toggle expanded thinking blocks (Ctrl+T, R)."""
            self.expand_thinking = not self.expand_thinking
            self._toggle_status = "Thinking: " + ("expanded" if self.expand_thinking else "collapsed")
            event.app.invalidate()
        
        @kb.add('c-t', 'd')
        def toggle_todos(event):
            """Toggle todo list visibility (Ctrl+T, D)."""
            self.show_todos = not self.show_todos
            self._toggle_status = "Todos: " + ("visible" if self.show_todos else "hidden")
            event.app.invalidate()
        
        # Also support Ctrl+T Ctrl+T for quick toggle of all
        @kb.add('c-t', 'c-t')
        def toggle_toolcalls_quick(event):
            """Toggle tool call details (Ctrl+T, Ctrl+T)."""
            self.show_tool_details = not self.show_tool_details
            self._toggle_status = "Tool calls: " + ("expanded" if self.show_tool_details else "collapsed")
            event.app.invalidate()
        
        self.session = PromptSession(
            history=FileHistory(str(history_dir / "history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            complete_while_typing=True,
            multiline=True,
            enable_history_search=True,
            key_bindings=kb,
            style=PROMPT_STYLE,
            bottom_toolbar=self._get_toolbar,
            reserve_space_for_menu=4,
        )
        
        self.running = True
        self.current_checkpoint = None
        
        # Token stats
        self.input_tokens = 0
        self.output_tokens = 0
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        
        # Session management
        self.session_id: Optional[str] = None
        self._session_manager = None
        self._auto_resume_enabled = True
        
        # Todo state
        self.todos: List[Dict[str, str]] = []
        self.show_todos = False
        
        # Display settings (toggleable via Ctrl+T shortcuts)
        self.show_tool_details = False  # Collapsed by default (Ctrl+T T to toggle)
        self.expand_thinking = False    # Collapsed by default (Ctrl+T R to toggle)
        self._toggle_status = ""        # Temporary status message for toggles
        
        # Agent instance (lazy loaded)
        self._agent = None
        self._agent_initialized = False
        
        # Model configuration - default to Google Gemini
        self.model = os.environ.get("WEBAGENTS_MODEL", "google/gemini-2.5-flash")
        # self.model = os.environ.get("WEBAGENTS_MODEL", "google/gemini-3-flash-preview")
        self.instructions = DEFAULT_INSTRUCTIONS
        
        # Daemon client (for future daemon mode)
        self.daemon_client: Optional[DaemonClient] = None
        self.use_daemon = True  # Enable daemon mode by default
        
        # Initialize session manager if agent path is available
        if self.agent_path:
            self._init_session_manager()
    
    def _init_session_manager(self):
        """Initialize session manager for auto-save/resume."""
        try:
            from webagents.agents.skills.local.session.skill import SessionManager
            agent_dir = self.agent_path.parent if self.agent_path else Path.cwd()
            self._session_manager = SessionManager(agent_dir, self.agent_name)
        except Exception:
            self._session_manager = None
    
    async def _auto_resume_session(self):
        """Auto-resume the latest session if available."""
        if not self._session_manager or not self._auto_resume_enabled:
            return
        
        try:
            session = self._session_manager.load_latest(max_messages=100)
            if session and session.messages:
                # Restore messages (convert Message objects to dicts)
                self.messages = [
                    m.to_dict() if hasattr(m, 'to_dict') else m 
                    for m in session.messages
                ]
                self.session_id = session.session_id
                self.input_tokens = session.input_tokens
                self.output_tokens = session.output_tokens
                
                # Show conversation history
                self._display_session_history()
            else:
                import uuid
                self.session_id = str(uuid.uuid4())
        except Exception as e:
            import uuid
            self.session_id = str(uuid.uuid4())
    
    def _display_session_history(self):
        """Display restored session history."""
        from rich.text import Text
        from rich.markdown import Markdown
        import re
        
        self.console.print()
        self.console.print(f"[dim]── Restored session ({len(self.messages)} messages) ──[/dim]")
        self.console.print()
        
        for msg in self.messages[-6:]:  # Show last 6 messages max
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                # User message with coral vertical bar (same as input prompt)
                display_content = content[:120] + "…" if len(content) > 120 else content
                self.console.print("[#ff6b6b]┃[/]")
                self.console.print(f"[#ff6b6b]┃[/]  {display_content}")
                self.console.print("[#ff6b6b]┃[/]")
                self.console.print()
            elif role == "assistant":
                # Assistant message (truncated preview)
                # Strip thinking tags for preview
                clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if len(clean) > 200:
                    clean = clean[:200] + "…"
                if clean:
                    self.console.print(f"    {clean}", style="dim")
                    self.console.print()
        
        if len(self.messages) > 6:
            self.console.print(f"[dim]  ... {len(self.messages) - 6} earlier messages[/dim]")
            self.console.print()
        
        self.console.print("[dim]── Continue conversation below ──[/dim]")
        self.console.print()
    
    def _save_session(self):
        """Save current session."""
        if not self._session_manager or not self.session_id:
            return
        
        try:
            from webagents.agents.skills.local.session.skill import Session, Message
            from datetime import datetime
            
            # Convert messages to Message objects
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
    
    async def _fetch_agent_commands(self):
        """Fetch and register commands from the connected agent.
        
        Commands are dynamically discovered from the agent's skills.
        When switching agents, this is called to refresh the available commands.
        """
        # Clear any previously registered agent commands
        self.slash_commands.clear_agent_commands()
        
        if not self.daemon_client:
            return
        
        try:
            commands = await self.daemon_client.list_commands(self.agent_name)
            if commands:
                self.slash_commands.register_agent_commands(commands)
        except Exception:
            pass  # Agent might not have commands or daemon might not be ready
    
    async def _ensure_agent(self):
        """Ensure agent is loaded and initialized."""
        if self._agent_initialized:
            return
            
        if self.use_daemon:
            await self._ensure_daemon()
            self._agent_initialized = True
            return
        
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.skills.local.filesystem.skill import FilesystemSkill
        from webagents.agents.skills.local.shell.skill import ShellSkill
        from webagents.agents.skills.local.cli.skill import CLISkill
        from webagents.agents.skills.local.web.skill import WebSkill
        from webagents.agents.skills.local.todo.skill import TodoSkill
        
        # Skills to include
        skills = {
            "files": FilesystemSkill(),
            "shell": ShellSkill(),
            "cli": CLISkill(session=self),
            "web": WebSkill(session=self),
            "todo": TodoSkill(session=self),
        }
        
        # Load from AGENT.md if provided
        if self.agent_path and self.agent_path.exists():
            try:
                from ..loader.hierarchy import load_agent
                merged = load_agent(self.agent_path)
                self.agent_name = merged.name
                
                # Prepend default instructions to agent instructions if provided
                if merged.instructions:
                    self.instructions = f"{DEFAULT_INSTRUCTIONS}\n\n## Agent Specific Instructions\n\n{merged.instructions}"
                else:
                    self.instructions = DEFAULT_INSTRUCTIONS
                
                # Only use model from metadata if explicitly set
                if merged.metadata.model:
                    self.model = merged.metadata.model
                
                # TODO: Add skills defined in AGENT.md metadata
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load agent: {e}[/yellow]")
        
        # Create the agent
        self._agent = BaseAgent(
            name=self.agent_name,
            instructions=self.instructions,
            model=self.model,
            skills=skills,
        )
        
        self._agent_initialized = True
    
    async def _ensure_daemon(self):
        """Ensure daemon is running and connected."""
        first_connect = self.daemon_client is None
        
        if not self.daemon_client:
            with self.console.status("[dim]Connecting to daemon...[/dim]", spinner="dots"):
                self.daemon_client = await ensure_daemon_running(
                    watch_dirs=[Path.cwd()]
                )
        
        # Always register agent and fetch commands (even on reconnect)
        if self.agent_path:
            try:
                # Register agent
                result = await self.daemon_client.register_agent(self.agent_path)
                
                # Update name from registration
                if result.get("name"):
                    self.agent_name = result["name"]
                
                # Fetch agent details to get instructions
                try:
                    agent_info = await self.daemon_client.get_agent(self.agent_name)
                    if agent_info.get("instructions"):
                        self.instructions = agent_info["instructions"]
                except Exception:
                    # Ignore if we can't get details (might be transient)
                    pass
                    
            except Exception as e:
                if first_connect:
                    self.console.print(f"[yellow]Warning: Failed to register agent with daemon: {e}[/yellow]")
        
        # Fetch and register agent commands (always refresh)
        await self._fetch_agent_commands()
        
        # Auto-resume latest session (only on first connect)
        if first_connect:
            await self._auto_resume_session()
    
    def _print_welcome(self):
        """Print welcome banner and tips."""
        self.console.print("")
        print_splash(self.console)
        self.console.print("")
        
        if self.agent_path:
            self.console.print(f"[dim]Agent: {self.agent_name} ({self.agent_path})[/dim]")
        else:
            self.console.print("[dim]No agent loaded. Using default assistant.[/dim]")
        
        self.console.print(f"[dim]Daemon mode: {'Enabled' if self.use_daemon else 'Disabled'}[/dim]")
        self.console.print("\n\n\n\n")
    
    def _print_response(self, response: str):
        """Print agent response with markdown rendering."""
        self.console.print(Markdown(response))
    
    def _print_tool_execution(self, tool_name: str, result: str):
        """Display tool execution in a panel."""
        self.console.print(Panel(
            result[:500] + ("…" if len(result) > 500 else ""),
            title=f"[cyan]{tool_name}[/cyan]",
            border_style="dim"
        ))
    
    def _get_toolbar(self):
        """Get bottom toolbar with agent info and footer."""
        
        # Get terminal width
        try:
            width = shutil.get_terminal_size().columns
        except:
            width = 80
        
        result = []
        
        # Show toggle status message (temporary)
        if self._toggle_status:
            result.append(('class:bottom-toolbar.text', f" {self._toggle_status}\n"))
            self._toggle_status = ""
        
        # Add full todo list if toggled
        if self.show_todos and self.todos:
            for t in self.todos:
                if not isinstance(t, dict): continue
                status = t.get('status', 'pending')
                desc = t.get('description', 'Task')
                icon = "✓" if status == 'completed' else "●" if status == 'in_progress' else "○" if status == 'pending' else "×"
                result.append(('class:bottom-toolbar.text', f"  {icon} {desc}\n"))
        
        # Get in-progress task
        in_progress_task = next((t['description'] for t in self.todos if t['status'] == 'in_progress'), None)
        if in_progress_task and not self.show_todos:
            result.append(('class:bottom-toolbar.text', f"  ● {in_progress_task}\n"))
        
        # Agent info line: ┃  @agent                     path
        agent_name = f"@{self.agent_name}"
        if self.agent_path:
            path_obj = self.agent_path.parent if self.agent_path.is_file() else self.agent_path
            path_str = str(path_obj)
            if len(path_str) > 40:
                path_str = "…" + path_str[-37:]
            padding = max(1, width - len(agent_name) - len(path_str) - 4)
            result.append(('class:prompt-bar', '┃  '))
            result.append(('class:prompt-agent', agent_name))
            result.append(('class:prompt-path', ' ' * padding + path_str))
        else:
            result.append(('class:prompt-bar', '┃  '))
            result.append(('class:prompt-agent', agent_name))
        result.append(('class:bottom-toolbar.text', '\n'))
        
        # Footer line: version | sandbox | help
        version_text = "WebAgents v0.1"
        sandbox_text = "sandbox: on"
        help_text = "'/' for commands"
        
        total_content = len(version_text) + len(sandbox_text) + len(help_text) + 2
        available_space = width - total_content
        left_pad = available_space // 3
        right_pad = available_space - left_pad - (available_space // 3)
        
        result.append(('class:bottom-toolbar.version', f" {version_text}"))
        result.append(('class:bottom-toolbar.text', " " * left_pad))
        result.append(('class:bottom-toolbar.status-on', sandbox_text))
        result.append(('class:bottom-toolbar.text', " " * right_pad))
        result.append(('class:bottom-toolbar.text', help_text))
        
        return result

    def _render_streaming_state(self, response_text: str, active_tools: Dict[str, Any], thinking_duration: float, thinking_start_time: Optional[float]):
        """Render current state with tools and thinking."""
        from rich.console import Group
        from rich.text import Text
        from rich.markdown import Markdown
        from rich.panel import Panel
        import time
        import re
        
        elements = []
        
        # Helper to render a thought block with vertical bar  
        def render_thought(content, is_open=False, duration=None):
            from rich.console import Group
            import textwrap
            import shutil
            
            try:
                term_width = shutil.get_terminal_size().columns
            except:
                term_width = 80
            wrap_width = max(40, term_width - 8)  # Leave room for "    │  "
            
            subtitle = ""
            clean_content = content.strip()
            
            if not clean_content and is_open:
                return Text("  ∵ Thinking…", style="#666666 italic")
            
            # Find ALL bold text and use the LAST one as subtitle
            matches = re.findall(r'\*\*([^\*]+)\*\*', clean_content)
            if matches:
                subtitle = matches[-1].strip()
            elif len(clean_content.split('\n')) > 0:
                first_line = clean_content.split('\n')[0].strip()
                if len(first_line) < 50:
                    subtitle = first_line
            
            header = f"  ∵ {subtitle}" if subtitle else "  ∵ Thinking…"
            if duration is not None:
                header += f" · {duration:.1f}s"
            
            # Collapsed mode (default) - just show header with dimmed style
            if not self.expand_thinking:
                return Text(header, style="#666666 italic")
            
            # Expanded mode - show header + content with dimmed vertical bar
            if clean_content:
                lines = []
                lines.append(Text(header, style="#666666 italic"))
                # Add content with dimmed vertical bar, wrap long lines
                for line in clean_content.split('\n'):
                    wrapped = textwrap.wrap(line, width=wrap_width) if line.strip() else ['']
                    for wrapped_line in wrapped:
                        lines.append(Text(f"    │  {wrapped_line}", style="#666666"))
                return Group(*lines)
            return Text(header, style="#666666 italic")

        # Helper to create smart summary for collapsed tool calls
        def get_tool_summary(name, args, result=None):
            """Generate a human-friendly summary for collapsed tool view."""
            try:
                import json
                args_dict = json.loads(args) if isinstance(args, str) else args
            except:
                args_dict = {}
            
            # If we have a result, try to include meaningful info
            result_str = str(result) if result else ""
            
            # Smart summaries based on tool name
            if name in ("read_query", "write_query"):
                # Show row count or first result if available
                if result_str:
                    lines = result_str.strip().split('\n')
                    if len(lines) > 1:
                        return f"Query returned {len(lines)} rows"
                    elif result_str[:80]:
                        return f"→ {result_str[:80]}…" if len(result_str) > 80 else f"→ {result_str}"
                return f"Ran db query…"
            elif name == "create_table":
                table = args_dict.get("table_name", "table")
                return f"Created table {table}"
            elif name == "list_tables":
                if result_str:
                    tables = [t.strip() for t in result_str.split('\n') if t.strip()]
                    if tables:
                        return f"Found {len(tables)} tables: {', '.join(tables[:3])}{'…' if len(tables) > 3 else ''}"
                return "Listed db tables"
            elif name == "describe_table":
                table = args_dict.get("table_name", "table")
                return f"Described {table}"
            elif name in ("read_file", "read"):
                path = args_dict.get("path", args_dict.get("file_path", "file"))
                fname = Path(path).name if path else 'file'
                if result_str:
                    lines = result_str.count('\n') + 1
                    return f"Read {fname} ({lines} lines)"
                return f"Read {fname}"
            elif name in ("write_file", "write"):
                path = args_dict.get("file_path", args_dict.get("path", "file"))
                return f"Wrote {Path(path).name if path else 'file'}"
            elif name in ("list_directory", "list_files"):
                path = args_dict.get("path", ".")
                if result_str:
                    items = [i.strip() for i in result_str.split('\n') if i.strip()]
                    if items:
                        preview = ', '.join(items[:4])
                        return f"Listed {len(items)} items: {preview}{'…' if len(items) > 4 else ''}"
                return f"Listed {path}"
            elif name == "shell" or name == "bash" or name == "run_command":
                cmd = args_dict.get("command", "")[:30]
                if result_str:
                    first_line = result_str.split('\n')[0][:60]
                    return f"→ {first_line}{'…' if len(result_str) > 60 else ''}"
                return f"Ran command"
            elif name == "glob":
                if result_str:
                    files = [f.strip() for f in result_str.split('\n') if f.strip()]
                    if files:
                        return f"Found {len(files)} files"
                return f"Searched files"
            elif name == "search_file_content":
                if result_str:
                    matches = result_str.count('\n') + 1
                    return f"Found {matches} matches"
                return f"Searched content"
            elif name == "replace":
                return "Replaced text"
            elif name == "web_fetch":
                return "Fetched webpage"
            elif name == "write_todos":
                return "Updated todos"
            elif name == "analyze_image":
                return "Analyzed image"
            else:
                return f"Ran {name}"

        # Helper to render a tool call
        def render_tool(tool_id, tool_data):
            import textwrap
            import shutil
            
            try:
                term_width = shutil.get_terminal_size().columns
            except:
                term_width = 80
            wrap_width = max(40, term_width - 8)
            
            name = tool_data["name"]
            args = tool_data["args"]
            status = tool_data["status"]
            result = tool_data.get("result", "")
            
            icon = "■"
            res = []
            summary = get_tool_summary(name, args, result)
            
            if self.show_tool_details and result:
                # Expanded mode with dimmed vertical bar
                header = Text.assemble(
                    ("  ", ""),
                    (f"{icon} ", "#666666"),
                    (f"{name}", "#777777"),
                    (f" · ", "#555555"),
                    (f"{args[:40]}{'…' if len(args) > 40 else ''}", "#555555") if args else ("", "")
                )
                res.append(header)
                
                # Add result with dimmed vertical bar, wrap long lines
                result_str = str(result)[:500]
                line_count = 0
                for line in result_str.split('\n')[:10]:
                    wrapped = textwrap.wrap(line, width=wrap_width) if line.strip() else ['']
                    for wrapped_line in wrapped:
                        res.append(Text(f"    │  {wrapped_line}", style="#666666"))
                        line_count += 1
                        if line_count >= 15:
                            break
                    if line_count >= 15:
                        break
                if len(result_str.split('\n')) > 10:
                    res.append(Text(f"    │  … ({len(result_str.split(chr(10)))} more lines)", style="#555555 italic"))
            elif result and len(str(result)) < 80:
                # Short result - show inline
                header = Text(f"  {icon} {summary}", style="#666666 italic")
                res.append(header)
                res.append(Text(f"    └ {str(result)[:80]}", style="#555555"))
            else:
                # Collapsed mode (default)
                line = Text(f"  {icon} {summary}", style="#666666 italic")
                res.append(line)
            
            return res

        # Parse thinking blocks first
        think_pattern = r'<think>(.*?)(?:</think>|$)'
        think_matches = list(re.finditer(think_pattern, response_text, re.DOTALL))
        
        # Render thinking blocks first
        for i, m in enumerate(think_matches):
            content = m.group(1).strip()
            is_closed = m.group(0).endswith('</think>')
            is_open = not is_closed
            
            duration = None
            if is_open and thinking_start_time:
                duration = time.time() - thinking_start_time
            elif is_closed and thinking_duration > 0:
                duration = thinking_duration
            
            elements.append(render_thought(content, is_open, duration))
            elements.append(Text(""))
        
        # Render completed tools (status=success) BEFORE content
        completed_tools = [(tid, td) for tid, td in active_tools.items() if td.get("status") == "success"]
        for tid, tool_data in completed_tools:
            tool_elements = render_tool(tid, tool_data)
            elements.extend(tool_elements)
            elements.append(Text(""))
        
        # Get clean content (strip thinking blocks and tool markers)
        clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.DOTALL)
        clean_text = re.sub(r'<tool_call id="[^"]*"/>', '', clean_text)
        clean_text = clean_text.strip()
        
        # Render main content
        if clean_text:
            elements.append(Markdown(clean_text))
            elements.append(Text(""))
        
        # Render running tools at the end (still in progress)
        running_tools = [(tid, td) for tid, td in active_tools.items() if td.get("status") == "running"]
        for tid, tool_data in running_tools:
            tool_elements = render_tool(tid, tool_data)
            elements.extend(tool_elements)
            elements.append(Text(""))
        
        # Fallback: If nothing but we are thinking
        if not elements and (thinking_start_time or thinking_duration > 0):
            d = thinking_duration if thinking_duration > 0 else (time.time() - thinking_start_time)
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = frames[int(time.time() * 10) % len(frames)]
            elements.append(Text(f"  {frame} Thinking… · {d:.1f}s", style="#666666"))
            
        if not elements:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = frames[int(time.time() * 10) % len(frames)]
            return Text(f"  {frame} …", style="#666666")
            
        return Group(*elements)

    def _summarize_tool_result(self, name: str, result: str) -> str:
        """Create a short summary of tool result."""
        if not result:
            return "(empty)"
        
        if name == "read_file":
            if result.startswith("Error") or "not found" in result or "Access denied" in result:
                return result[:300]
            lines = result.splitlines()
            return f"Read {len(lines)} lines"
        elif name == "list_files" or name == "list_directory":
            if "No files found" in result:
                return "No files found"
            lines = result.splitlines()
            if len(lines) > 20 and not result.startswith("Directory listing"):
                return f"Found {len(lines)} files"
            return result[:300] + ("…" if len(result) > 300 else "")
        elif name == "shell" or name == "bash" or name == "run_command":
            return result[:300] + ("…" if len(result) > 300 else "")
            
        return result[:300] + ("…" if len(result) > 300 else "")
    
    async def _handle_input(self, user_input: str):
        """Handle user input."""
        user_input = user_input.strip()
        
        if not user_input:
            return
        
        # Handle slash commands
        if user_input.startswith("/"):
            result = await handle_slash_command(user_input, self)
            if result == "exit":
                self.running = False
            return
        
        # Handle @ file references
        expanded_input = self._expand_file_references(user_input)
        
        # Ensure agent is ready
        await self._ensure_agent()
        
        # Add user message to history
        self.messages.append({"role": "user", "content": expanded_input})
        
        # Prepare messages with system prompt
        messages_to_send = [
            {"role": "system", "content": self.instructions},
            *self.messages
        ]
        
        # Stream response from agent
        try:
            response_text = ""
            active_tools = {}
            thinking_start_time = None
            thinking_duration = 0
            
            # Use Live display for streaming
            from rich.spinner import Spinner
            status_widget = Spinner("dots", text="Waiting for response...", style="dim")
            with Live(status_widget, console=self.console, refresh_per_second=10) as live:
                if self.use_daemon:
                    retries = 1
                    while retries >= 0:
                        try:
                            # History for daemon is already in self.messages (excluding system prompt which daemon adds)
                            async for chunk_str in self.daemon_client.chat_stream(self.agent_name, expanded_input, self.messages[:-1]):
                                if chunk_str == "[DONE]":
                                    break
                                try:
                                    import json
                                    import time
                                    chunk = json.loads(chunk_str)
                                    
                                    # Handle metadata chunks for tool calls and thinking
                                    if chunk.get("object") == "metadata":
                                        mtype = chunk.get("type")
                                        payload = chunk.get("payload", {})
                                        if mtype == "tool_start":
                                            active_tools[payload["id"]] = {
                                                "name": payload["name"],
                                                "args": payload["arguments"],
                                                "status": "running"
                                            }
                                            
                                            # Special handling for todo tool to update UI immediately
                                            if payload["name"] == "write_todos":
                                                try:
                                                    import json
                                                    args_val = payload["arguments"]
                                                    if isinstance(args_val, str):
                                                        args = json.loads(args_val)
                                                    else:
                                                        args = args_val
                                                        
                                                    if "todos" in args:
                                                        todos_val = args["todos"]
                                                        if isinstance(todos_val, str):
                                                            self.todos = json.loads(todos_val)
                                                        else:
                                                            self.todos = todos_val
                                                except:
                                                    pass
                                            
                                            # Insert marker to track tool call position in stream
                                            response_text += f"<tool_call id=\"{payload['id']}\"/>"
                                        elif mtype == "tool_result":
                                            tool = active_tools.get(payload["id"])
                                            if tool:
                                                tool["status"] = payload["status"]
                                                tool["result"] = payload.get("result", "")
                                                tool["summary"] = self._summarize_tool_result(payload["name"], payload["result"])
                                        elif mtype == "thought_start":
                                            thinking_start_time = time.time()
                                        elif mtype == "thought_end":
                                            if thinking_start_time:
                                                thinking_duration = time.time() - thinking_start_time
                                        
                                        # Update UI with metadata changes
                                        live.update(self._render_streaming_state(response_text, active_tools, thinking_duration, thinking_start_time))
                                        continue

                                    # Extract content from chunk
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        
                                        # Track tool calls from delta (standard OpenAI streaming format)
                                        tool_calls = delta.get("tool_calls")
                                        if tool_calls:
                                            for tc in tool_calls:
                                                tc_index = tc.get("index", 0)
                                                tc_id = tc.get("id")
                                                tc_func = tc.get("function", {})
                                                tc_name = tc_func.get("name")
                                                tc_args = tc_func.get("arguments", "")
                                                
                                                # Create or update tool entry
                                                tool_key = f"tool_{tc_index}"
                                                if tc_id:
                                                    tool_key = tc_id
                                                    
                                                if tool_key not in active_tools:
                                                    active_tools[tool_key] = {
                                                        "name": tc_name or "…",
                                                        "args": "",
                                                        "status": "running"
                                                    }
                                                
                                                # Update name if we got one
                                                if tc_name:
                                                    active_tools[tool_key]["name"] = tc_name
                                                    
                                                # Accumulate arguments
                                                if tc_args:
                                                    active_tools[tool_key]["args"] += tc_args
                                                    
                                                # Insert marker for tool position in stream
                                                if tc_id and f"<tool_call id=\"{tc_id}\"/>" not in response_text:
                                                    response_text += f"<tool_call id=\"{tc_id}\"/>"
                                        
                                        # Check for finish_reason to mark tool calls as complete
                                        finish_reason = choices[0].get("finish_reason")
                                        if finish_reason == "tool_calls":
                                            # Mark all active tools as in-progress (will be completed later)
                                            for tool_key in active_tools:
                                                if active_tools[tool_key]["status"] == "running":
                                                    active_tools[tool_key]["status"] = "success"
                                        
                                    if content:
                                        response_text += content
                                        
                                    # Update UI with intelligent thought parsing
                                    if content or tool_calls:
                                        live.update(self._render_streaming_state(response_text, active_tools, thinking_duration, thinking_start_time))
                                    
                                    # Track token usage
                                    usage = chunk.get("usage", {})
                                    if usage:
                                        self.input_tokens += usage.get("prompt_tokens", 0)
                                        self.output_tokens += usage.get("completion_tokens", 0)
                                except Exception as e:
                                    # Might be raw text or malformed JSON
                                    pass
                            
                            # Stream finished successfully
                            break
                            
                        except Exception as e:
                            # Handle 404 (Agent not found) - likely daemon restart
                            import httpx
                            is_404 = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404
                            
                            if is_404 and self.agent_path and retries > 0:
                                retries -= 1
                                live.update(Text("Daemon restarted. Re-registering agent...", style="yellow"))
                                try:
                                    await self.daemon_client.register_agent(self.agent_path)
                                    live.update(Text("Agent re-registered. Retrying...", style="green"))
                                    continue # Retry loop
                                except Exception as reg_err:
                                    live.update(Text(f"Error re-registering agent: {reg_err}", style="red"))
                                    break
                            else:
                                live.update(Text(f"Error communicating with daemon: {e}", style="red"))
                                break
                else:
                    async for chunk in self._agent.run_streaming(messages_to_send):
                        # Extract content from chunk
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                response_text += content
                                # Update live display with current response
                                live.update(Markdown(response_text))
                            
                            # Handle tool calls display
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    if tc.get("function", {}).get("name"):
                                        tool_name = tc["function"]["name"]
                                        live.update(Text(f"Using tool: {tool_name}...", style="dim cyan"))
                        
                        # Track token usage
                        usage = chunk.get("usage", {})
                        if usage:
                            self.input_tokens += usage.get("prompt_tokens", 0)
                            self.output_tokens += usage.get("completion_tokens", 0)
            
            # Add assistant response to history
            if response_text:
                # Strip tool markers before adding to history
                import re
                history_text = re.sub(r'<tool_call id=".*?"/>', '', response_text)
                self.messages.append({"role": "assistant", "content": history_text.strip()})
            
            self.console.print("")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
            # Remove the failed user message from history
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
        finally:
            # Auto-save session after each interaction
            self._save_session()
    
    def _expand_file_references(self, text: str) -> str:
        """Expand @path/to/file references to include file contents."""
        import re
        
        def replace_file_ref(match):
            file_path = match.group(1)
            path = Path(file_path).expanduser()
            
            # Try relative to cwd first
            if not path.is_absolute():
                path = Path.cwd() / path
            
            if path.exists() and path.is_file():
                try:
                    content = path.read_text()
                    return f"\n\n<file path=\"{file_path}\">\n{content}\n</file>\n\n"
                except Exception as e:
                    return f"@{file_path} (error reading: {e})"
            else:
                return match.group(0)  # Keep original if not found
        
        # Match @path patterns (not email-like patterns)
        pattern = r'@((?:[a-zA-Z0-9_\-./~]+)+(?:\.[a-zA-Z0-9]+)?)'
        return re.sub(pattern, replace_file_ref, text)
    
    async def run(self):
        """Run the interactive session."""
        # Ensure agent is loaded (for correct name in welcome and toolbar)
        await self._ensure_agent()
        
        self._print_welcome()
        
        while self.running:
            try:
                # Simple prompt with coral vertical bar:
                # ┃
                # ┃  [cursor]
                
                prompt_parts = [
                    ('class:prompt-bar', '┃\n'),
                    ('class:prompt-bar', '┃  '),
                ]
                
                user_input = await self.session.prompt_async(
                    prompt_parts,
                    rprompt=[],
                )
                
                # Print closing bar after input
                self.console.print("[#ff6b6b]┃[/]")
                self.console.print()
                
                await self._handle_input(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /exit or Ctrl+D to quit[/dim]")
                continue
            except EOFError:
                # Ctrl+D
                self.console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def run_sync(self):
        """Run session synchronously."""
        asyncio.run(self.run())
    
    def clear_history(self):
        """Clear conversation history and start fresh session."""
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0
        # Generate new session ID
        import uuid
        self.session_id = str(uuid.uuid4())

    async def confirm(self, message: str) -> bool:
        """Ask user for confirmation."""
        from prompt_toolkit.shortcuts import confirm
        return await asyncio.to_thread(confirm, f"{message} (y/n): ")


def start_repl(agent_path: Optional[Path] = None):
    """Start interactive REPL session."""
    # Configure logging to file and disable console output for REPL
    from webagents.utils.logging import setup_logging
    
    # Ensure log directory exists
    log_dir = Path.home() / ".webagents" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "repl.log"
    setup_logging(level="INFO", log_file=str(log_file), console_output=False)
    
    session = WebAgentsSession(agent_path=agent_path)
    session.run_sync()
