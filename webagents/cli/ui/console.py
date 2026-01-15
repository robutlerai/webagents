"""
WebAgents Console

Rich console wrapper with streaming and tool display.
"""

from typing import AsyncIterator, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn


class WebAgentsConsole:
    """Enhanced console for WebAgents with streaming and tool display."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def print_agent_card(self, name: str, description: str, namespace: str, 
                          intents: list = None, status: str = "idle"):
        """Display agent info as a beautiful card."""
        content = f"[bold]{name}[/bold]\n\n"
        content += f"{description}\n\n"
        content += f"[dim]Namespace: {namespace}[/dim]\n"
        content += f"[dim]Status: {status}[/dim]"
        
        if intents:
            content += "\n\n[dim]Intents:[/dim]"
            for intent in intents[:5]:
                content += f"\n  - {intent}"
        
        self.console.print(Panel(
            content,
            title=f"[cyan]{name}[/cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))
    
    async def stream_response(self, chunks: AsyncIterator[str]):
        """Stream LLM response with live Markdown rendering."""
        buffer = ""
        
        with Live(Markdown(""), console=self.console, refresh_per_second=10) as live:
            async for chunk in chunks:
                buffer += chunk
                live.update(Markdown(buffer))
        
        return buffer
    
    def print_tool_execution(self, tool_name: str, args: dict = None, 
                              result: str = None, status: str = "success"):
        """Display tool execution with syntax highlighting."""
        # Determine border color
        border_color = "green" if status == "success" else "red" if status == "error" else "yellow"
        
        content = ""
        if args:
            content += f"[dim]Args: {args}[/dim]\n\n"
        
        if result:
            # Truncate long results
            if len(result) > 500:
                result = result[:500] + "\n...[truncated]"
            content += result
        
        self.console.print(Panel(
            content or "[dim]No output[/dim]",
            title=f"[cyan]{tool_name}[/cyan]",
            border_style=border_color,
        ))
    
    def print_code(self, code: str, language: str = "python", 
                   title: str = None, line_numbers: bool = True):
        """Display code with syntax highlighting."""
        syntax = Syntax(
            code, 
            language, 
            theme="monokai",
            line_numbers=line_numbers,
        )
        
        if title:
            self.console.print(Panel(syntax, title=title, border_style="dim"))
        else:
            self.console.print(syntax)
    
    def print_file_write(self, path: str, preview: str = None):
        """Display file write operation."""
        content = f"[green]Writing to {path}[/green]"
        if preview:
            # Show first few lines
            lines = preview.split('\n')[:5]
            content += "\n\n" + "\n".join(f"[dim]{i+1}|[/dim] {line}" for i, line in enumerate(lines))
            if len(preview.split('\n')) > 5:
                content += "\n[dim]...[/dim]"
        
        self.console.print(Panel(
            content,
            title="[cyan]WriteFile[/cyan]",
            border_style="green",
        ))
    
    def print_agents_table(self, agents: list):
        """Display agents in a table."""
        table = Table(title="Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Namespace", style="dim")
        table.add_column("Status", style="green")
        table.add_column("Intents")
        
        for agent in agents:
            intents_str = ", ".join(agent.get("intents", [])[:3])
            if len(agent.get("intents", [])) > 3:
                intents_str += ", ..."
            
            table.add_row(
                agent.get("name", "unknown"),
                agent.get("namespace", "local"),
                agent.get("status", "idle"),
                intents_str or "[dim]none[/dim]",
            )
        
        self.console.print(table)
    
    def print_discovery_results(self, results: list, intent: str):
        """Display discovery results."""
        if not results:
            self.console.print(f"[dim]No agents found for: {intent}[/dim]")
            return
        
        table = Table(title=f"Agents matching: {intent}")
        table.add_column("Agent", style="cyan")
        table.add_column("Namespace", style="dim")
        table.add_column("Score", style="green")
        table.add_column("Intents")
        
        for result in results:
            table.add_row(
                result.get("name"),
                result.get("namespace"),
                f"{result.get('score', 0):.2f}",
                ", ".join(result.get("matching_intents", [])),
            )
        
        self.console.print(table)
    
    def print_token_usage(self, input_tokens: int, output_tokens: int, 
                          cached_tokens: int = 0, cost: float = None):
        """Display token usage."""
        total = input_tokens + output_tokens
        
        content = f"[bold]Token Usage[/bold]\n\n"
        content += f"Input:  {input_tokens:,}\n"
        content += f"Output: {output_tokens:,}\n"
        if cached_tokens:
            content += f"Cached: {cached_tokens:,}\n"
        content += f"[bold]Total:  {total:,}[/bold]"
        
        if cost is not None:
            content += f"\n\n[dim]Estimated cost: ${cost:.4f}[/dim]"
        
        self.console.print(Panel(
            content,
            title="Tokens",
            border_style="cyan",
        ))
    
    def spinner(self, message: str = "Thinking..."):
        """Return a spinner context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            console=self.console,
            transient=True,
        )
    
    def error(self, message: str, details: str = None):
        """Display an error message."""
        content = f"[red]{message}[/red]"
        if details:
            content += f"\n\n[dim]{details}[/dim]"
        
        self.console.print(Panel(
            content,
            title="[red]Error[/red]",
            border_style="red",
        ))
    
    def success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]✓[/green] {message}")
    
    def warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {message}")
    
    def info(self, message: str):
        """Display an info message."""
        self.console.print(f"[cyan]ℹ[/cyan] {message}")
