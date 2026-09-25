#!/usr/bin/env python3
"""
WebAgents Daemon Entry Point (webagentsd)

`webagentsd` is `webagents daemon`: the same options, in the foreground.
"""

from typing import Optional

import typer


def _daemon(
    port: Optional[int] = typer.Option(None, "-p", "--port", help="Port (default: daemon.port, 8765)"),
    host: Optional[str] = typer.Option(None, "--host", help="Interface to listen on (default: daemon.host, 127.0.0.1)"),
    watch: Optional[str] = typer.Option(None, "-w", "--watch", help="Watch directory"),
    no_cron: bool = typer.Option(False, "--no-cron", help="Disable cron"),
) -> None:
    """Start the WebAgents daemon."""
    from webagents.cli.commands.daemon import run_daemon

    run_daemon(port=port, host=host, watch=watch, cron=not no_cron)


def main() -> None:
    """Entry point for webagentsd."""
    typer.run(_daemon)


if __name__ == "__main__":
    main()
