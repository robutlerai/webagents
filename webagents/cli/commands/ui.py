"""
WebUI CLI command.

webagents ui - Development and build commands for the React web UI.
"""

import typer
import subprocess
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Web UI development and build")
console = Console()

# Path to the webui directory
UI_DIR = Path(__file__).parent.parent / "webui"


def check_pnpm() -> bool:
    """Check if pnpm is installed."""
    try:
        subprocess.run(
            ["pnpm", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_node_modules() -> bool:
    """Check if node_modules exists."""
    return (UI_DIR / "node_modules").exists()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    port: int = typer.Option(5173, "--port", "-p", help="Development server port"),
    build: bool = typer.Option(False, "--build", "-b", help="Build for production"),
    install: bool = typer.Option(False, "--install", "-i", help="Install dependencies"),
):
    """WebAgents Web UI development server.
    
    Start the Vite development server for the React web UI,
    or build for production deployment.
    
    Examples:
        webagents ui              # Start dev server on port 5173
        webagents ui --port 3000  # Start on custom port
        webagents ui --build      # Build for production
        webagents ui --install    # Install dependencies
    """
    # If a subcommand was invoked, don't run the main callback
    if ctx.invoked_subcommand is not None:
        return
    
    # Check UI directory exists
    if not UI_DIR.exists():
        console.print(f"[red]Error:[/red] Web UI directory not found: {UI_DIR}")
        console.print("[dim]Expected webagents/cli/webui/ to exist[/dim]")
        raise typer.Exit(1)
    
    # Check for pnpm
    if not check_pnpm():
        console.print("[red]Error:[/red] pnpm is not installed")
        console.print("[dim]Install with: npm install -g pnpm[/dim]")
        raise typer.Exit(1)
    
    # Install dependencies if requested or missing
    if install or not check_node_modules():
        console.print("[cyan]Installing dependencies...[/cyan]")
        try:
            subprocess.run(
                ["pnpm", "install"],
                cwd=UI_DIR,
                check=True
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install dependencies:[/red] {e}")
            raise typer.Exit(1)
    
    if build:
        # Production build
        console.print(Panel(
            "[bold cyan]Building WebAgents Web UI[/bold cyan]\n\n"
            f"Output: {UI_DIR / 'dist'}",
            title="Build",
            border_style="cyan"
        ))
        
        try:
            subprocess.run(
                ["pnpm", "build"],
                cwd=UI_DIR,
                check=True
            )
            console.print(f"\n[green]Build complete![/green]")
            console.print(f"[dim]Output: {UI_DIR / 'dist'}[/dim]")
            
            # Show build contents
            dist_dir = UI_DIR / "dist"
            if dist_dir.exists():
                console.print("\n[dim]Build contents:[/dim]")
                for item in sorted(dist_dir.iterdir()):
                    if item.is_dir():
                        console.print(f"  [blue]{item.name}/[/blue]")
                    else:
                        size = item.stat().st_size
                        console.print(f"  {item.name} ({size:,} bytes)")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Build failed:[/red] {e}")
            raise typer.Exit(1)
    else:
        # Development server
        console.print(Panel(
            f"[bold cyan]Starting WebAgents Web UI Dev Server[/bold cyan]\n\n"
            f"Local:   http://localhost:{port}/ui\n"
            f"API:     http://localhost:8765 (proxy)\n\n"
            f"[dim]Press Ctrl+C to stop[/dim]",
            title="Development Server",
            border_style="cyan"
        ))
        
        try:
            subprocess.run(
                ["pnpm", "dev", "--port", str(port)],
                cwd=UI_DIR
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Development server stopped[/yellow]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Development server failed:[/red] {e}")
            raise typer.Exit(1)


@app.command("build")
def build_cmd():
    """Build the Web UI for production.
    
    Compiles the React application to static files in dist/.
    These files are served by the WebUISkill.
    """
    # Delegate to main with --build flag
    main(ctx=typer.Context(main), build=True, port=5173, install=False)


@app.command("dev")
def dev_cmd(
    port: int = typer.Option(5173, "--port", "-p", help="Development server port"),
):
    """Start the development server with hot reload.
    
    The dev server proxies API requests to the daemon at localhost:8765.
    """
    main(ctx=typer.Context(main), port=port, build=False, install=False)


@app.command("install")
def install_cmd():
    """Install Web UI dependencies.
    
    Runs pnpm install in the webui directory.
    """
    main(ctx=typer.Context(main), port=5173, build=False, install=True)


@app.command("status")
def status():
    """Show Web UI build status."""
    console.print(Panel(
        "[bold]Web UI Status[/bold]",
        border_style="cyan"
    ))
    
    # Check paths
    console.print(f"\n[bold]Paths[/bold]")
    console.print(f"  Source:  {UI_DIR}")
    console.print(f"  Dist:    {UI_DIR / 'dist'}")
    
    # Check source
    console.print(f"\n[bold]Source[/bold]")
    if UI_DIR.exists():
        console.print("  [green]✓[/green] Source directory exists")
        
        if (UI_DIR / "package.json").exists():
            console.print("  [green]✓[/green] package.json found")
        else:
            console.print("  [red]✗[/red] package.json missing")
            
        if check_node_modules():
            console.print("  [green]✓[/green] Dependencies installed")
        else:
            console.print("  [yellow]![/yellow] Dependencies not installed (run: webagents ui --install)")
    else:
        console.print("  [red]✗[/red] Source directory missing")
    
    # Check build
    console.print(f"\n[bold]Build[/bold]")
    dist_dir = UI_DIR / "dist"
    if dist_dir.exists():
        console.print("  [green]✓[/green] Dist directory exists")
        
        index_file = dist_dir / "index.html"
        if index_file.exists():
            console.print("  [green]✓[/green] index.html found")
            
            # Show build time
            from datetime import datetime
            mtime = datetime.fromtimestamp(index_file.stat().st_mtime)
            console.print(f"  [dim]Built: {mtime.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        else:
            console.print("  [red]✗[/red] index.html missing")
            
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            asset_count = len(list(assets_dir.iterdir()))
            console.print(f"  [green]✓[/green] Assets: {asset_count} files")
        else:
            console.print("  [yellow]![/yellow] Assets directory missing")
    else:
        console.print("  [yellow]![/yellow] Not built (run: webagents ui --build)")
    
    # Check pnpm
    console.print(f"\n[bold]Tools[/bold]")
    if check_pnpm():
        try:
            result = subprocess.run(
                ["pnpm", "--version"],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip()
            console.print(f"  [green]✓[/green] pnpm {version}")
        except Exception:
            console.print("  [green]✓[/green] pnpm installed")
    else:
        console.print("  [red]✗[/red] pnpm not found (npm install -g pnpm)")
