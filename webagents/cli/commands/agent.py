"""
Agent lifecycle commands.

webagents run, connect, list, stop, logs, debug, info
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Agent lifecycle management")
console = Console()


def _find_default_agent(path: Path) -> tuple[Optional[Path], Path]:
    """Find default agent using resolution rules.
    
    Returns:
        Tuple of (agent_path, working_dir). The working_dir is always
        the original path (cwd), even when using embedded agents.
    """
    # 1. Check for AGENT.md
    agent_md = path / "AGENT.md"
    if agent_md.exists():
        return (agent_md, path)
    
    # 2. Check for single AGENT-*.md
    agent_files = list(path.glob("AGENT-*.md"))
    if len(agent_files) == 1:
        return (agent_files[0], path)
    elif len(agent_files) > 1:
        return (None, path)  # Multiple agents, can't determine default
    
    # 3. Fallback to embedded robutler agent
    from webagents.agents.builtin import get_robutler_path
    embedded = get_robutler_path()
    if embedded.exists():
        return (embedded, path)  # Use cwd as working dir, not embedded location
    
    return (None, path)


def _resolve_agent(agent_id: Optional[str]) -> tuple[Optional[Path], Path]:
    """Resolve agent from name or path.
    
    Returns:
        Tuple of (agent_path, working_dir). The working_dir is the directory
        where the agent should operate (cwd for local agents).
    """
    cwd = Path.cwd()
    
    if agent_id is None:
        return _find_default_agent(cwd)
    
    # Check if it's a path
    path = Path(agent_id)
    if path.exists():
        return (path, path.parent if path.is_file() else path)
    
    # Check if it's relative to current directory
    cwd_path = cwd / agent_id
    if cwd_path.exists():
        return (cwd_path, cwd)
    
    # Try as AGENT-<name>.md
    agent_file = cwd / f"AGENT-{agent_id}.md"
    if agent_file.exists():
        return (agent_file, cwd)
    
    # Check if requesting robutler specifically
    if agent_id.lower() == "robutler":
        from webagents.agents.builtin import get_robutler_path
        return (get_robutler_path(), cwd)
    
    # TODO: Look up in registry
    return (None, cwd)


@app.command("run")
def run(
    agent: Optional[str] = typer.Argument(None, help="Agent name or path"),
    prompt: Optional[str] = typer.Option(None, "-p", "--prompt", help="Single prompt, exit after"),
    all_agents: bool = typer.Option(False, "--all", help="Run all agents in directory"),
):
    """Run an agent (headless execution)."""
    import asyncio
    
    if all_agents:
        agent_files = list(Path.cwd().glob("AGENT*.md"))
        if not agent_files:
            console.print("[red]No AGENT*.md files found in current directory[/red]")
            raise typer.Exit(1)
        
        console.print(f"[cyan]Running {len(agent_files)} agent(s)...[/cyan]")
        for af in agent_files:
            console.print(f"  - {af.name}")
        # TODO: Actually run agents
        return
    
    agent_path, working_dir = _resolve_agent(agent)
    if agent_path is None:
        if agent:
            console.print(f"[red]Agent not found: {agent}[/red]")
        else:
            console.print("[red]No AGENT.md or AGENT-*.md found in current directory[/red]")
            console.print("[dim]Use 'webagents init' to create one, or specify an agent path.[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Running agent: {agent_path.name}[/cyan]")
    
    if prompt:
        # Single prompt mode - run synchronously
        asyncio.run(_run_single_prompt(agent_path, prompt, working_dir))
    else:
        # Headless mode
        console.print("[yellow]Headless execution not yet implemented[/yellow]")
        console.print("[dim]Use -p/--prompt to run with a single prompt[/dim]")


async def _run_single_prompt(agent_path: Path, prompt: str, working_dir: Optional[Path] = None):
    """Execute a single prompt against an agent.
    
    Args:
        agent_path: Path to the agent definition file
        prompt: The prompt to send to the agent
        working_dir: Working directory for the agent (defaults to agent's parent dir)
    """
    if working_dir is None:
        working_dir = agent_path.parent
    from ..loader.hierarchy import load_agent
    from ..daemon.manager import AgentManager
    from webagents.agents.core.base_agent import BaseAgent
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.spinner import Spinner
    import importlib
    
    try:
        # Load merged agent (sync)
        merged = load_agent(agent_path)
        console.print(f"[dim]Agent: {merged.name}[/dim]")
        console.print(f"[dim]Model: {merged.metadata.model}[/dim]")
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        console.print()
        
        # Get skills from metadata
        skills_list = merged.metadata.skills or []
        if not skills_list:
            skills_list = ["completions"]  # Minimal default
        
        # Skill classes mapping (subset for CLI use)
        skill_classes = {
            "testrunner": "webagents.agents.skills.local.testrunner.skill.TestRunnerSkill",
            "filesystem": "webagents.agents.skills.local.filesystem.skill.FilesystemSkill",
            "shell": "webagents.agents.skills.local.shell.skill.ShellSkill",
            "completions": "webagents.agents.skills.core.transport.completions.skill.CompletionsTransportSkill",
            "litellm": "webagents.agents.skills.core.llm.litellm.skill.LiteLLMSkill",
            "google": "webagents.agents.skills.core.llm.google.skill.GoogleAISkill",
        }
        
        # Load skills
        skills = {}
        for item in skills_list:
            skill_name = None
            config = {}
            
            if isinstance(item, str):
                skill_name = item
            elif isinstance(item, dict) and len(item) == 1:
                skill_name = list(item.keys())[0]
                config = item[skill_name] or {}
            
            if skill_name and skill_name in skill_classes:
                try:
                    module_path, class_name = skill_classes[skill_name].rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    skill_class = getattr(module, class_name)
                    
                    # Try config dict, kwargs, or no args
                    try:
                        skills[skill_name] = skill_class(config=config)
                    except TypeError:
                        try:
                            skills[skill_name] = skill_class(**config)
                        except TypeError:
                            skills[skill_name] = skill_class()
                    
                    console.print(f"[dim]Loaded skill: {skill_name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to load skill {skill_name}: {e}[/yellow]")
        
        # Always add Google LLM skill for handoff if not already present
        if "llm" not in skills and "google" not in skills and "litellm" not in skills and "primary_llm" not in skills:
            try:
                from webagents.agents.skills.core.llm.google.skill import GoogleAISkill
                skills["llm"] = GoogleAISkill()
                console.print(f"[dim]Loaded default LLM skill: GoogleAI[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load LLM skill: {e}[/yellow]")
        
        # Create BaseAgent
        agent = BaseAgent(
            name=merged.name,
            instructions=merged.instructions,
            skills=skills,
            scopes=merged.metadata.scopes or ["all"],
            model=merged.metadata.model or "openai/gpt-4o-mini",
        )
        
        # Initialize async skills
        await agent._ensure_skills_initialized()
        
        console.print(f"[dim]Skills loaded: {list(skills.keys())}[/dim]")
        console.print()
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Show spinner while processing
        with Live(Spinner("dots", text="Processing..."), console=console, refresh_per_second=10) as live:
            # Run the agent
            response = await agent.run(messages)
        
        # Extract and display response
        if response and "choices" in response:
            content = response["choices"][0].get("message", {}).get("content", "")
            if content:
                console.print(Markdown(content))
            else:
                console.print("[dim]No response content[/dim]")
        else:
            console.print(f"[dim]Response: {response}[/dim]")
        
        # Show tool calls if any
        if response and "choices" in response:
            tool_calls = response["choices"][0].get("message", {}).get("tool_calls", [])
            if tool_calls:
                console.print()
                console.print(f"[dim]Tool calls: {len(tool_calls)}[/dim]")
                for tc in tool_calls:
                    console.print(f"  [cyan]{tc.get('function', {}).get('name', 'unknown')}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command("stop")
def stop(
    agent: str = typer.Argument(..., help="Agent to stop"),
):
    """Stop a running agent."""
    console.print(f"[cyan]Stopping agent: {agent}[/cyan]")
    # TODO: Implement stop via daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("restart")
def restart(
    agent: str = typer.Argument(..., help="Agent to restart"),
):
    """Restart an agent."""
    console.print(f"[cyan]Restarting agent: {agent}[/cyan]")
    # TODO: Implement restart via daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("logs")
def logs(
    agent: str = typer.Argument(..., help="Agent name"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream logs"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines"),
):
    """View agent logs."""
    console.print(f"[cyan]Logs for agent: {agent}[/cyan]")
    if follow:
        console.print("[dim]Following logs... (Ctrl+C to exit)[/dim]")
    # TODO: Read logs from .webagents/logs/<agent>/
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("debug")
def debug(
    agent: str = typer.Argument(..., help="Agent to debug"),
):
    """Start interactive debug mode for an agent."""
    console.print(f"[cyan]Debug mode for agent: {agent}[/cyan]")
    # TODO: Launch debug REPL
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("info")
def info(
    agent: str = typer.Argument(..., help="Agent name or path"),
):
    """Show detailed agent information."""
    agent_path, working_dir = _resolve_agent(agent)
    if agent_path is None:
        console.print(f"[red]Agent not found: {agent}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold]Agent: {agent_path.stem}[/bold]\n"
        f"Path: {agent_path}\n"
        f"Working Dir: {working_dir}\n"
        f"[dim]Use 'webagents connect {agent}' to start a session[/dim]",
        title="Agent Info",
        border_style="cyan"
    ))
    # TODO: Parse AGENT.md and show full info


# Command functions used by main.py
def connect_command(agent: Optional[str] = None, use_tui: bool = True):
    """Start interactive REPL session."""
    import os
    
    agent_path, working_dir = _resolve_agent(agent)
    
    if agent_path is None and agent:
        console.print(f"[red]Agent not found: {agent}[/red]")
        raise typer.Exit(1)
    
    # Set working directory (important for embedded agents like robutler)
    os.chdir(working_dir)
    
    if use_tui:
        # Use the new Textual TUI
        import asyncio
        from ..repl.tui import run_tui
        from ..client.daemon_client import DaemonClient
        
        # Extract agent name from path
        if agent_path:
            stem = agent_path.stem
            if stem == "ROBUTLER":
                agent_name = "robutler"
            elif stem.startswith("AGENT-"):
                agent_name = stem.replace("AGENT-", "")
            elif stem == "AGENT":
                agent_name = "default"
            else:
                agent_name = stem
        else:
            agent_name = "assistant"
        
        daemon_client = DaemonClient(working_dir=str(working_dir))
        asyncio.run(run_tui(
            agent_name=agent_name,
            agent_path=agent_path,
            daemon_client=daemon_client,
            use_daemon=True
        ))
    else:
        # Use the legacy Rich/prompt_toolkit REPL
        from ..repl.session import start_repl
        start_repl(agent_path=agent_path)


def list_command(
    local: bool = False,
    remote: bool = False,
    running: bool = False,
    namespace: Optional[str] = None
):
    """List registered agents."""
    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Namespace", style="dim")
    table.add_column("Status", style="green")
    table.add_column("Source", style="dim")
    
    # Scan current directory for agents
    cwd = Path.cwd()
    agent_files = list(cwd.glob("AGENT.md")) + list(cwd.glob("AGENT-*.md"))
    
    for af in agent_files:
        name = af.stem if af.stem != "AGENT" else "default"
        table.add_row(
            name,
            "local",
            "idle",
            str(af.relative_to(cwd))
        )
    
    if len(agent_files) == 0:
        console.print("[dim]No agents found in current directory.[/dim]")
        console.print("[dim]Use 'webagents init' to create one.[/dim]")
    else:
        console.print(table)


def init_command(
    name: Optional[str] = None,
    template_name: Optional[str] = None,
    context: bool = False
):
    """Initialize a new agent."""
    if context:
        # Create AGENTS.md context file
        agents_md = Path.cwd() / "AGENTS.md"
        if agents_md.exists():
            console.print("[yellow]AGENTS.md already exists[/yellow]")
            raise typer.Exit(1)
        
        content = """---
# Context inherited by all AGENT*.md files in this directory
namespace: local
---

# Project Context

This file provides context for all agents in this directory.

## Guidelines

- Be helpful and concise
- Follow best practices
- Ask for clarification when needed
"""
        agents_md.write_text(content)
        console.print(f"[green]Created {agents_md}[/green]")
        return
    
    # Create AGENT.md or AGENT-<name>.md
    if name:
        filename = f"AGENT-{name}.md"
    else:
        filename = "AGENT.md"
    
    agent_path = Path.cwd() / filename
    if agent_path.exists():
        console.print(f"[yellow]{filename} already exists[/yellow]")
        raise typer.Exit(1)
    
    # Use template if specified
    if template_name:
        console.print(f"[cyan]Using template: {template_name}[/cyan]")
        # TODO: Load and apply template
    
    content = f"""---
name: {name or 'assistant'}
description: A helpful AI assistant
namespace: local
model: openai/gpt-4o-mini
intents:
  - answer questions
  - help with tasks
skills: []
visibility: local
---

# {name.title() if name else 'Assistant'} Agent

You are a helpful AI assistant.

## Capabilities

- Answer questions
- Help with various tasks
- Provide information

## Guidelines

- Be helpful and concise
- Ask for clarification when needed
"""
    agent_path.write_text(content)
    console.print(f"[green]Created {agent_path}[/green]")
    console.print(f"[dim]Edit the file to customize your agent.[/dim]")
    console.print(f"[dim]Run 'webagents connect' to start a session.[/dim]")
