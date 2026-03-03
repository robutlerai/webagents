#!/usr/bin/env python3
"""
WebAgents Daemon Entry Point (webagentsd)

Directly runs the daemon start command.
"""
import typer
from webagents.cli.commands.daemon import start

def main():
    """Entry point for webagentsd."""
    typer.run(start)

if __name__ == "__main__":
    main()
