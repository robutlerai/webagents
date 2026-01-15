#!/usr/bin/env python3
"""
WebAgents CLI Entry Point

Premium command-line interface for AI agents.
Build, run, and discover agents from your terminal.
"""


def main():
    """Main entry point for the webagents CLI."""
    from webagents.cli.main import cli
    cli()


if __name__ == "__main__":
    main()
