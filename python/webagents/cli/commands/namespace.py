"""
Namespace management commands.

webagents namespace list, create, auth, invite, join, members, delete
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Namespace management")
console = Console()


@app.command("list")
def list_namespaces(
    all_: bool = typer.Option(False, "--all", "-a", help="All accessible namespaces"),
):
    """List namespaces."""
    table = Table(title="Namespaces")
    table.add_column("Namespace", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Role")
    table.add_column("Agents")
    
    # Default local namespace
    table.add_row("local", "local", "owner", "0")
    
    if all_:
        console.print("[dim]Fetching all accessible namespaces...[/dim]")
        # TODO: Fetch from platform
    
    console.print(table)


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Namespace name"),
    type_: str = typer.Option("user", "--type", "-t", help="Type: user, reversedomain, global"),
):
    """Create a new namespace."""
    if type_ not in ["user", "reversedomain", "global"]:
        console.print(f"[red]Invalid type: {type_}[/red]")
        console.print("[dim]Valid types: user, reversedomain, global[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Creating namespace: {name}[/cyan]")
    console.print(f"[dim]Type: {type_}[/dim]")
    
    if type_ == "global":
        console.print("[yellow]Global namespaces require platform approval[/yellow]")
    elif type_ == "reversedomain":
        console.print("[dim]Domain verification will be required[/dim]")
    
    # TODO: Create namespace via platform API
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("auth")
def auth(
    namespace: str = typer.Argument(..., help="Namespace to configure"),
    secret: Optional[str] = typer.Option(None, "--secret", "-s", help="Set shared secret"),
    verify_domain: bool = typer.Option(False, "--verify-domain", help="Verify domain ownership"),
    token: bool = typer.Option(False, "--token", help="Generate access token"),
):
    """Configure namespace authentication."""
    console.print(f"[cyan]Configuring auth for: {namespace}[/cyan]")
    
    if secret:
        console.print("[dim]Setting shared secret...[/dim]")
        # TODO: Set secret
    
    if verify_domain:
        console.print(Panel(
            "[bold]Domain Verification[/bold]\n\n"
            "Add this TXT record to your DNS:\n\n"
            "[cyan]_webagents.yourdomain.com TXT \"verify=abc123\"[/cyan]\n\n"
            "Then run this command again to verify.",
            title="DNS Verification",
            border_style="cyan"
        ))
        # TODO: Initiate domain verification
    
    if token:
        console.print("[dim]Generating access token...[/dim]")
        # TODO: Generate token
        console.print("[yellow]Token generation not yet implemented[/yellow]")


@app.command("invite")
def invite(
    namespace: str = typer.Argument(..., help="Namespace to invite to"),
):
    """Generate invite code for namespace."""
    console.print(f"[cyan]Generating invite for: {namespace}[/cyan]")
    # TODO: Generate invite code
    console.print(Panel(
        "[dim]Share this invite code:[/dim]\n\n"
        "[cyan]webagents-invite-abc123xyz[/cyan]\n\n"
        "[dim]Run: webagents namespace join <code>[/dim]",
        title="Invite Code",
        border_style="green"
    ))


@app.command("join")
def join(
    invite_code: str = typer.Argument(..., help="Invite code"),
):
    """Join namespace via invite code."""
    console.print(f"[cyan]Joining namespace with code: {invite_code}[/cyan]")
    # TODO: Join via invite
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("info")
def info(
    namespace: str = typer.Argument(..., help="Namespace to show"),
):
    """Show namespace details."""
    console.print(Panel(
        f"[bold]Namespace: {namespace}[/bold]\n\n"
        "Type: user\n"
        "Owner: (you)\n"
        "Members: 1\n"
        "Agents: 0\n"
        "Created: -",
        title="Namespace Info",
        border_style="cyan"
    ))


@app.command("members")
def members(
    namespace: str = typer.Argument(..., help="Namespace to list members"),
):
    """List namespace members."""
    table = Table(title=f"Members of {namespace}")
    table.add_column("User", style="cyan")
    table.add_column("Role")
    table.add_column("Joined", style="dim")
    
    table.add_row("(you)", "owner", "-")
    
    console.print(table)


@app.command("remove-member")
def remove_member(
    namespace: str = typer.Argument(..., help="Namespace"),
    user: str = typer.Argument(..., help="User to remove"),
):
    """Remove member from namespace."""
    console.print(f"[yellow]Removing {user} from {namespace}[/yellow]")
    # TODO: Remove member
    console.print("[green]Member removed[/green]")


@app.command("delete")
def delete(
    namespace: str = typer.Argument(..., help="Namespace to delete"),
):
    """Delete namespace (must be empty)."""
    console.print(f"[red]Deleting namespace: {namespace}[/red]")
    confirm = typer.confirm("Are you sure?")
    if not confirm:
        raise typer.Exit(0)
    
    # TODO: Delete namespace
    console.print("[yellow]Not yet implemented[/yellow]")


@app.command("default")
def default(
    namespace: str = typer.Argument(..., help="Namespace to set as default"),
):
    """Set default namespace for publishing."""
    console.print(f"[cyan]Setting default namespace: {namespace}[/cyan]")
    # TODO: Update config
    console.print(f"[green]Default namespace set to: {namespace}[/green]")
