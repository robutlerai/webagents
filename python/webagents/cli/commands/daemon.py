"""
`webagents daemon`: serve every agent in a folder, in the foreground
(2026-09-24).

ONE COMMAND, AS IN THE TYPESCRIPT CLI (`typescript/src/cli/index.ts`):
`-p/--port`, `--host`, `-w/--watch` and `--no-cron`, running until Ctrl+C. It
was a group of six (`start`, `stop`, `restart`, `status`, `logs`,
`endpoints`) with a background mode and PID files, which the TypeScript CLI
never had; the chat no longer needs a daemon at all, since both chats build
their agent in their own process. A daemon that should outlive the terminal is
the job of whatever runs services on the machine (launchd, systemd, a
container).

WHICH ADDRESS. `--port`/`--host`, then `daemon.port`/`daemon.host` from the
config store, then 127.0.0.1:8765 (`cli/daemon_address.py`). A host from config
must be loopback; only `--host` binds anywhere else (S-218).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from webagents import __version__

from ..daemon_address import DaemonAddress, DaemonAddressError, resolve_daemon_address


def _address(port: Optional[int], host: Optional[str] = None) -> DaemonAddress:
    """The address these flags mean; exits 1, saying why, when the config cannot be used."""
    try:
        return resolve_daemon_address(port=port, host=host)
    except DaemonAddressError as exc:
        print(str(exc), file=sys.stderr)
        if exc.fix:
            print(f"Fix: {exc.fix}", file=sys.stderr)
        raise SystemExit(1)


def daemon_server(watch: Optional[str] = None, cron: bool = True, error_detail: bool = True) -> Any:
    """The daemon's server: the agents under `watch`, else under this folder,
    as the TypeScript daemon serves them without `-w` too."""
    from webagents.server.core.app import create_server

    folder = Path(watch) if watch else Path.cwd()
    return create_server(
        title="WebAgents Daemon",
        description="Local agent daemon",
        version=__version__,
        url_prefix="/agents",
        enable_file_watching=True,
        watch_dirs=[folder],
        enable_cron=cron,
        storage_backend="json",
        error_detail=error_detail,
        quiet=True,
    )


def run_daemon(port: Optional[int] = None, host: Optional[str] = None, watch: Optional[str] = None, cron: bool = True) -> None:
    """Serve the agents in `watch` (this folder by default), reloading them as their files change."""
    import uvicorn

    address = _address(port, host)
    # A FAILED RUN'S OWN TEXT, FOR THE DEVELOPER'S TERMINAL (S-228), and only
    # on loopback: a daemon bound to every interface has remote callers and
    # answers them with a fixed message and a reference.
    server = daemon_server(watch=watch, cron=cron, error_detail=address.is_loopback)
    # The TypeScript daemon's words (`daemon/server.ts`).
    print(f"WebAgents daemon starting on {address.base_url}")
    server.app.add_event_handler("startup", lambda: print("WebAgents daemon started", flush=True))
    try:
        uvicorn.run(server.app, host=address.host, port=address.port, log_level="warning")
    finally:
        print("WebAgents daemon stopped")
