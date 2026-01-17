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


# Custom prompt style
PROMPT_STYLE = Style.from_dict({
    'prompt': 'cyan bold',
    'completion-menu.completion': 'bg:#008888 #ffffff',
    'completion-menu.completion.current': 'bg:#00aaaa #000000',
    'scrollbar.background': 'bg:#88aaaa',
    'scrollbar.button': 'bg:#222222',
    'bottom-toolbar': '', # Transparent/Default
    'bottom-toolbar.text': 'bg:#1a1a1a #aaaaaa',
    'bottom-toolbar.right': 'bg:#1a1a1a #eeeeee bold',
    'bottom-toolbar.status-on': 'bg:#1a1a1a bold',
    # 'bottom-toolbar.separator': 'bg:default #444444',
    'bottom-toolbar.pad': 'bg:#1a1a1a #aaaaaa',
})

# Default system prompt for assistant mode
DEFAULT_INSTRUCTIONS = """You are a helpful AI assistant running in the WebAgents CLI.
Be concise and helpful. Use markdown for formatting. For newlines (e.g. poetry, lyrics, lists), use two spaces at the end of the line.
"""


from prompt_toolkit.completion import Completer, Completion

class SlashCommandCompleter(Completer):
    """Completer for slash commands that only triggers when starting with /."""
    
    def __init__(self, commands):
        self.commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith('/'):
            return
            
        for cmd in self.commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))


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
        slash_cmds = [f"/{cmd}" for cmd in self.slash_commands.list_commands().keys()]
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
            
        @kb.add('c-t')
        def _(event):
            """Toggle todo list visibility."""
            self.show_todos = not self.show_todos
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
            reserve_space_for_menu=0,
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
                self.console.print(f"[dim]Restored session: {len(self.messages)} messages[/dim]")
            else:
                import uuid
                self.session_id = str(uuid.uuid4())
        except Exception as e:
            import uuid
            self.session_id = str(uuid.uuid4())
    
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
        """Fetch and register commands from the connected agent."""
        if not self.daemon_client:
            return
        
        try:
            commands = await self.daemon_client.list_commands(self.agent_name)
            self.slash_commands.register_agent_commands(commands)
        except Exception:
            pass  # Agent might not have commands
    
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
        if not self.daemon_client:
            with self.console.status("[dim]Connecting to daemon...[/dim]", spinner="dots"):
                self.daemon_client = await ensure_daemon_running(
                    watch_dirs=[Path.cwd()]
                )
                
                # Register current agent with daemon if using local file
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
                        self.console.print(f"[yellow]Warning: Failed to register agent with daemon: {e}[/yellow]")
                
                # Fetch and register agent commands
                await self._fetch_agent_commands()
                
                # Auto-resume latest session
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
            result[:500] + ("..." if len(result) > 500 else ""),
            title=f"[cyan]{tool_name}[/cyan]",
            border_style="dim"
        ))
    
    def _get_toolbar(self):
        """Get bottom toolbar content with centered sandbox status."""
        
        # Get in-progress task for display above toolbar
        in_progress_task = next((t['description'] for t in self.todos if t['status'] == 'in_progress'), None)
        
        # Show directory if possible
        if self.agent_path:
            path_obj = self.agent_path.parent if self.agent_path.is_file() else self.agent_path
            path = str(path_obj)
        else:
            path = str(Path.cwd())
            
        # Truncate path if too long
        if len(path) > 30:
            path = "..." + path[-27:]
        
        # Get terminal width
        try:
            width = shutil.get_terminal_size().columns
        except:
            width = 80
            
        # Components
        # Separator line above status bar (removed as per layout request)
        # separator = "─" * width + "\n"
        
        path_text = f" {path} "
        spacer = " "
        sandbox_text = " sandbox: on "
        agent_text = f" {self.agent_name} "
        
        # Calculate centering
        # Layout: [Path][Spacer][Padding1][Sandbox][Padding2][Agent]
        
        sandbox_len = len(sandbox_text)
        path_len = len(path_text)
        spacer_len = len(spacer)
        agent_len = len(agent_text)
        
        # Target start for sandbox (center of screen)
        target_sandbox_start = (width // 2) - (sandbox_len // 2)
        
        # Calculate padding1 (Left of sandbox)
        current_left_len = path_len + spacer_len
        padding1_len = max(0, target_sandbox_start - current_left_len)
        
        # Calculate padding2 (Right of sandbox)
        current_filled = current_left_len + padding1_len + sandbox_len + agent_len
        padding2_len = max(0, width - current_filled)
        
        padding1 = " " * padding1_len
        padding2 = " " * padding2_len
        
        result = []
        
        # Add full todo list if toggled
        if self.show_todos and self.todos:
            result.append(('class:bottom-toolbar.text', " Todo List (Ctrl+T to hide):\n"))
            for t in self.todos:
                if not isinstance(t, dict): continue
                status = t.get('status', 'pending')
                desc = t.get('description', 'Task')
                icon = "✅" if status == 'completed' else "🟡" if status == 'in_progress' else "⚪" if status == 'pending' else "🔴"
                style = "bold" if status == 'in_progress' else "dim" if status in ['completed', 'cancelled'] else ""
                result.append(('class:bottom-toolbar.text', f" {icon} {desc}\n"))
            result.append(('class:bottom-toolbar.text', "─" * width + "\n"))
        
        # Add current task if any
        if in_progress_task and not self.show_todos:
            result.append(('class:bottom-toolbar.status-on', f" 🟡 Working on: {in_progress_task} "))
            result.append(('class:bottom-toolbar.text', "\n"))

        result.extend([
            # ('class:bottom-toolbar.separator', separator),
            ('class:bottom-toolbar.pad', path_text),
            ('class:bottom-toolbar.pad', spacer),
            ('class:bottom-toolbar.pad', padding1),
            ('class:bottom-toolbar.status-on', sandbox_text),
            ('class:bottom-toolbar.pad', padding2),
            ('class:bottom-toolbar.right', agent_text),
        ])
        
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
        
        # Helper to render a thought block
        def render_thought(content, is_open=False, duration=None):
            subtitle = ""
            # Simple heuristic for subtitle: first line or bold text
            clean_content = content.strip()
            if not clean_content and is_open:
                return Text("Thinking...", style="dim italic")
                
            match = re.search(r'\*\*([^\*]+)\*\*', clean_content)
            if match:
                subtitle = match.group(1).strip()
            elif len(clean_content.split('\n')) > 0:
                 first_line = clean_content.split('\n')[0].strip()
                 if len(first_line) < 50:
                     subtitle = first_line
            
            header = f"Thinking... {subtitle}" if subtitle else "Thinking..."
            
            if duration is not None:
                header += f" ({duration:.1f}s)"
            
            return Text(header, style="dim italic")

        # Helper to render a tool call
        def render_tool(tool_id, tool_data):
            name = tool_data["name"]
            args = tool_data["args"]
            status = tool_data["status"]
            
            icon = "●"
            color = "green" if status == "success" else "red" if status == "error" else "yellow"
            if status == "running":
                icon = "⚙"
                color = "yellow"
                
            line = Text.assemble(
                (f"{icon} ", color),
                (f"{name}", "bold"),
                (f"({args})", "dim")
            )
            
            res = [line]
            if "summary" in tool_data:
                res.append(Text(f"  └ {tool_data['summary']}", style="dim"))
            
            return res

        # 1. Parse content into interleaved segments
        # Find all <think> blocks and tool markers <tool_call id="...">
        # Regex matches <think>...</think> OR <think>... OR <tool_call id="...">
        pattern = r'(<think>.*?(?:</think>|$))|(<tool_call id="(.*?)"/>)'
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        
        rendered_tool_ids = set()
        last_pos = 0
        
        for i, m in enumerate(matches):
            # Interleaved text BEFORE the thought/tool
            pre_text = response_text[last_pos:m.start()].strip()
            if pre_text:
                elements.append(Markdown(pre_text))
                elements.append(Text("")) # Spacer after text
            
            # Identify what matched
            think_match = m.group(1)
            tool_marker = m.group(2)
            tool_id_from_marker = m.group(3)
            
            if think_match:
                # The thought content
                content = think_match[7:] # strip <think>
                is_closed = content.endswith('</think>')
                if is_closed:
                    content = content[:-8]
                
                content = content.strip()
                is_open = not is_closed
                
                # Duration logic
                duration = None
                if is_open and thinking_start_time:
                    duration = time.time() - thinking_start_time
                elif is_closed and i == len(matches) - 1 and thinking_duration > 0:
                    duration = thinking_duration
                
                # Add thought block
                elements.append(render_thought(content, is_open, duration))
                elements.append(Text("")) # Spacer after thought
                
            elif tool_marker and tool_id_from_marker:
                if tool_id_from_marker in active_tools:
                    tool_elements = render_tool(tool_id_from_marker, active_tools[tool_id_from_marker])
                    elements.extend(tool_elements)
                    elements.append(Text("")) # Spacer after tool
                    rendered_tool_ids.add(tool_id_from_marker)
            
            last_pos = m.end()
            
        # 2. Text AFTER all matches
        post_text = response_text[last_pos:].strip()
        if post_text:
            elements.append(Markdown(post_text))
            
        # 3. Fallback: If absolutely nothing but we are active thinking
        if not elements and (thinking_start_time or thinking_duration > 0) and not active_tools:
             d = thinking_duration if thinking_duration > 0 else (time.time() - thinking_start_time)
             frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
             frame = frames[int(time.time() * 10) % len(frames)]
             elements.append(Text(f"{frame} Thinking... ({d:.1f}s)", style="dim cyan"))

        # 4. Tools that haven't been rendered yet (e.g. running or no marker yet)
        # Always render tools that weren't caught by markers at the bottom
        remaining_tools = [tid for tid in active_tools if tid not in rendered_tool_ids]
        
        if remaining_tools:
            # Ensure spacer before tools
            if elements and (not isinstance(elements[-1], Text) or elements[-1].plain != ""):
                elements.append(Text(""))
                
            for tid in remaining_tools:
                tool_elements = render_tool(tid, active_tools[tid])
                elements.extend(tool_elements)
                
            elements.append(Text("")) # Spacer after tools
            
        if not elements:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = frames[int(time.time() * 10) % len(frames)]
            return Text(f"{frame} ...", style="dim")
            
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
            return result[:300] + ("..." if len(result) > 300 else "")
        elif name == "shell" or name == "bash":
            return result[:300] + ("..." if len(result) > 300 else "")
            
        return result[:300] + ("..." if len(result) > 300 else "")
    
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
                                    if content:
                                        response_text += content
                                        
                                        # Update UI with intelligent thought parsing
                                        live.update(self._render_streaming_state(response_text, active_tools, thinking_duration, thinking_start_time))
                                        
                                        # Handle tool calls display
                                        tool_calls = delta.get("tool_calls")
                                        if tool_calls:
                                            for tc in tool_calls:
                                                if tc.get("function", {}).get("name"):
                                                    # Tool calls handled via metadata chunks
                                                    pass
                                    
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
                # Get user input
                user_input = await self.session.prompt_async(
                    [('class:prompt', '❯❯ ')],
                )
                
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
