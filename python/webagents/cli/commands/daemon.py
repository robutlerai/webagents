"""
Daemon management commands.

webagents daemon start, stop, status, install, expose
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Manage webagentsd daemon")
console = Console()


@app.command("start")
def start(
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    dev: bool = typer.Option(False, "--dev", help="Run in development mode (auto-reload)"),
    port: int = typer.Option(8765, "--port", "-p", help="Daemon port"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    watch: Optional[List[str]] = typer.Option(None, "--watch", "-w", help="Directories to watch"),
):
    """Start the webagentsd daemon."""
    console.print(Panel(
        f"[bold cyan]Starting webagentsd[/bold cyan]\n\n"
        f"Port: {port}\n"
        f"Mode: {'background' if background else ('dev' if dev else 'foreground')}\n"
        f"Watch: {', '.join(watch) if watch else 'current directory'}",
        title="webagentsd",
        border_style="cyan"
    ))
    
    if background:
        # Re-launch current process in background (simplistic approach)
        import subprocess
        import sys
        
        args = [sys.executable, "-m", "webagents", "daemon", "start", "--port", str(port)]
        if watch:
            for w in watch:
                args.extend(["--watch", w])
        
        # Detach process
        subprocess.Popen(args, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print("[green]Daemon started in background[/green]")
        return

    # Foreground mode
    import uvicorn
    if dev:
        import os
        if watch:
            os.environ["WEBAGENTS_WATCH_DIRS"] = ",".join(watch)
        
        # Enable debug logs for dev mode
        os.environ["WEBAGENTS_LOG_LEVEL"] = "DEBUG"
        
        import logging as _logging
        _logging.getLogger("watchfiles.main").setLevel(_logging.WARNING)
        
        console.print("[yellow]Running in development mode with auto-reload[/yellow]")
        console.print("[dim]Debug logging enabled[/dim]")
        from pathlib import Path
        import webagents
        webagents_dir = Path(webagents.__file__).parent
        reload_dirs = [str(webagents_dir)]
        if watch:
            reload_dirs.extend(watch)
        else:
            reload_dirs.append(str(Path.cwd()))
        console.print(f"[dim]Watching: {', '.join(reload_dirs)}[/dim]")
        uvicorn.run(
            "webagents.server.dev_entry:app",
            host=host,
            port=port,
            log_level="debug",
            reload=True,
            reload_dirs=reload_dirs,
            reload_includes=["*.py"],
            reload_excludes=[".*", "__pycache__", "*.pyc", "node_modules", ".git"],
        )
        return

    from webagents.server.core.app import create_server
    from pathlib import Path
    
    watch_dirs = [Path(w) for w in watch] if watch else [Path.cwd()]
    
    # Create server instance with daemon features enabled
    server = create_server(
        title="WebAgents Daemon",
        description="Local agent daemon",
        version="0.2.3",
        url_prefix="/agents",  # Match daemon client expectations
        enable_file_watching=True,
        watch_dirs=watch_dirs,
        enable_cron=True,
        storage_backend="json" # Local storage for daemon
    )
    
    # Run uvicorn
    uvicorn.run(server.app, host=host, port=port, log_level="info")


@app.command("stop")
def stop():
    """Stop the webagentsd daemon."""
    console.print("[cyan]Stopping webagentsd...[/cyan]")
    # TODO: Send stop signal to daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("restart")
def restart():
    """Restart the webagentsd daemon."""
    console.print("[cyan]Restarting webagentsd...[/cyan]")
    # TODO: Restart daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("status")
def status():
    """Show daemon status."""
    console.print(Panel(
        "[dim]webagentsd status[/dim]\n\n"
        "Status: [yellow]not running[/yellow]\n"
        "Port: -\n"
        "Agents: 0\n"
        "Uptime: -",
        title="Daemon Status",
        border_style="yellow"
    ))
    # TODO: Check actual daemon status


@app.command("install")
def install(
    systemd: bool = typer.Option(False, "--systemd", help="Install as systemd service (Linux)"),
    launchd: bool = typer.Option(False, "--launchd", help="Install as launchd service (macOS)"),
):
    """Install webagentsd as a system service."""
    import platform
    
    if not systemd and not launchd:
        # Auto-detect
        system = platform.system()
        if system == "Linux":
            systemd = True
        elif system == "Darwin":
            launchd = True
        else:
            console.print(f"[yellow]Auto-install not supported on {system}[/yellow]")
            console.print("[dim]Use --systemd or --launchd explicitly[/dim]")
            raise typer.Exit(1)
    
    if systemd:
        console.print("[cyan]Installing systemd service...[/cyan]")
        # TODO: Generate and install systemd unit file
        console.print("[yellow]Not yet implemented[/yellow]")
    elif launchd:
        console.print("[cyan]Installing launchd service...[/cyan]")
        # TODO: Generate and install launchd plist
        console.print("[yellow]Not yet implemented[/yellow]")


@app.command("uninstall")
def uninstall():
    """Uninstall webagentsd system service."""
    console.print("[cyan]Uninstalling service...[/cyan]")
    # TODO: Remove system service
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("expose")
def expose(
    agent: str = typer.Argument(..., help="Agent to expose"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Custom port"),
):
    """Expose agent via HTTP endpoint."""
    console.print(f"[cyan]Exposing agent: {agent}[/cyan]")
    if port:
        console.print(f"[dim]Port: {port}[/dim]")
    # TODO: Register agent exposure with daemon
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("hide")
def hide(
    agent: str = typer.Argument(..., help="Agent to hide"),
):
    """Stop exposing agent."""
    console.print(f"[cyan]Hiding agent: {agent}[/cyan]")
    # TODO: Unregister agent exposure
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("endpoints")
def endpoints():
    """List exposed agent endpoints."""
    table = Table(title="Exposed Endpoints")
    table.add_column("Agent", style="cyan")
    table.add_column("Endpoint", style="green")
    table.add_column("Port")
    
    # TODO: Get endpoints from daemon
    console.print("[dim]No endpoints exposed[/dim]")


@app.command("logs")
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream logs"),
):
    """View daemon logs."""
    console.print("[cyan]Daemon logs[/cyan]")
    if follow:
        console.print("[dim]Following... (Ctrl+C to exit)[/dim]")
    # TODO: Read daemon logs
    console.print("[yellow]Not yet implemented[/yellow]")
