"""
Interactive REPL Session

Prompt Toolkit + Rich for a premium terminal experience.
"""

import asyncio
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion, WordCompleter, NestedCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from .slash_commands import SlashCommandRegistry, handle_slash_command
from ..ui.splash import print_splash, print_status_bar


# Custom prompt style
PROMPT_STYLE = Style.from_dict({
    'prompt': 'cyan bold',
    'completion-menu.completion': 'bg:#008888 #ffffff',
    'completion-menu.completion.current': 'bg:#00aaaa #000000',
    'scrollbar.background': 'bg:#88aaaa',
    'scrollbar.button': 'bg:#222222',
})

# Default system prompt for assistant mode
DEFAULT_INSTRUCTIONS = """You are a helpful AI assistant running in the WebAgents CLI.
Be concise and helpful. Use markdown for formatting.
For poetry, ALWAYS use the 'two spaces at the end of line' markdown syntax to preserve line breaks between lines, and use double newlines (blank lines) between verses."""


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
        
        self.session = PromptSession(
            history=FileHistory(str(history_dir / "history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            complete_while_typing=True,
            multiline=False,  # Will enable with special key
            style=PROMPT_STYLE,
        )
        
        self.running = True
        self.current_checkpoint = None
        
        # Token stats
        self.input_tokens = 0
        self.output_tokens = 0
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        
        # Agent instance (lazy loaded)
        self._agent = None
        self._agent_initialized = False
        
        # Model configuration - default to Google Gemini
        self.model = os.environ.get("WEBAGENTS_MODEL", "google/gemini-3-flash-preview")
        self.instructions = DEFAULT_INSTRUCTIONS
    
    async def _ensure_agent(self):
        """Ensure agent is loaded and initialized."""
        if self._agent_initialized:
            return
        
        from webagents.agents.core.base_agent import BaseAgent
        from webagents.agents.skills.local.filesystem.skill import FilesystemSkill
        from webagents.agents.skills.local.shell.skill import ShellSkill
        from webagents.agents.skills.local.cli.skill import CLISkill
        
        # Skills to include
        skills = {
            "files": FilesystemSkill(),
            "shell": ShellSkill(),
            "cli": CLISkill(session=self),
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
    
    def _print_welcome(self):
        """Print welcome banner and tips."""
        print_splash(self.console)
        
        if self.agent_path:
            self.console.print(f"[dim]Agent: {self.agent_name} ({self.agent_path})[/dim]")
        else:
            self.console.print("[dim]No agent loaded. Using default assistant.[/dim]")
        
        self.console.print()
    
    def _print_response(self, response: str):
        """Print agent response with markdown rendering."""
        # Use a fresh line
        self.console.print()
        self.console.print(Markdown(response))
        self.console.print()
    
    def _print_tool_execution(self, tool_name: str, result: str):
        """Display tool execution in a panel."""
        self.console.print(Panel(
            result[:500] + ("..." if len(result) > 500 else ""),
            title=f"[cyan]{tool_name}[/cyan]",
            border_style="dim"
        ))
    
    def _print_footer(self):
        """Print status footer."""
        agent_info = self.agent_name
        sandbox = "enabled"
        mode = "auto"
        print_status_bar(self.console, agent_info, sandbox, mode)
    
    async def _handle_input(self, user_input: str):
        """Handle user input."""
        user_input = user_input.strip()
        
        if not user_input:
            return
        
        # Handle slash commands
        if user_input.startswith("/"):
            result = handle_slash_command(user_input, self)
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
        self.console.print()
        
        try:
            response_text = ""
            
            # Use Live display for streaming
            with Live(Text("Thinking...", style="dim"), console=self.console, refresh_per_second=10) as live:
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
                self.messages.append({"role": "assistant", "content": response_text})
            
            self.console.print()
            
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
            # Remove the failed user message from history
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
    
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
        self._print_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = await self.session.prompt_async(
                    [('class:prompt', '❯ ')],
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
        """Clear conversation history."""
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.console.print("[dim]Conversation cleared.[/dim]")


def start_repl(agent_path: Optional[Path] = None):
    """Start interactive REPL session."""
    session = WebAgentsSession(agent_path=agent_path)
    session.run_sync()
