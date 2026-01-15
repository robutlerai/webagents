"""
Interactive REPL Session

Prompt Toolkit + Rich for a premium terminal experience.
"""

import asyncio
from typing import Optional
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .slash_commands import SlashCommandRegistry, handle_slash_command
from ..ui.splash import print_splash, print_status_bar


# Custom prompt style
PROMPT_STYLE = Style.from_dict({
    'prompt': 'cyan bold',
})


class WebAgentsSession:
    """Interactive REPL session with an agent."""
    
    def __init__(self, agent_path: Optional[Path] = None):
        self.console = Console()
        self.agent_path = agent_path
        self.agent_name = agent_path.stem if agent_path else "assistant"
        
        # Ensure history directory exists
        history_dir = Path.home() / ".webagents"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = PromptSession(
            history=FileHistory(str(history_dir / "history")),
            auto_suggest=AutoSuggestFromHistory(),
            multiline=False,  # Will enable with special key
            style=PROMPT_STYLE,
        )
        
        self.slash_commands = SlashCommandRegistry()
        self.running = True
        self.current_checkpoint = None
        
        # Token stats
        self.input_tokens = 0
        self.output_tokens = 0
    
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
        # TODO: Parse @path/to/file and include file contents
        
        # Send to agent
        self.console.print()
        self.console.print("[dim]Thinking...[/dim]")
        
        # TODO: Actually call the agent
        # For now, show a placeholder response
        response = f"I received your message: \"{user_input}\"\n\n*Agent functionality is not yet implemented.*"
        self._print_response(response)
    
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


def start_repl(agent_path: Optional[Path] = None):
    """Start interactive REPL session."""
    session = WebAgentsSession(agent_path=agent_path)
    session.run_sync()
