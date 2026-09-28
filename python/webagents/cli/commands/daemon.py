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

import os
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


def daemon_server(
    watch: Optional[str] = None,
    cron: bool = True,
    error_detail: bool = True,
    loopback: bool = False,
    public_url: Optional[str] = None,
) -> Any:
    """The daemon's server: the agents under `watch`, else under this folder,
    as the TypeScript daemon serves them without `-w` too.

    `loopback` says the daemon is bound to this machine only (S-306): its
    register and unregister routes then take no credential, as the local CLI
    sends none; anywhere else they require one, as the TypeScript daemon's
    do (S-284). Default False, so a caller that does not say is guarded.

    `public_url` is the address the daemon publishes for its agents
    (2026-09-27, `daemon/identity.py`): each served agent signs its `cron:`
    webhooks as `{public_url}/agents/{name}` and its key set is served at
    `/agents/{name}/.well-known/jwks.json`. `run_daemon` passes
    `WEBAGENTS_PUBLIC_URL`, else the bind address, which the signer refuses
    (loopback, plain http), so webhooks then go out unsigned and the run's
    record says why."""
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
        loopback=loopback,
        public_url=public_url,
    )


def run_daemon(port: Optional[int] = None, host: Optional[str] = None, watch: Optional[str] = None, cron: bool = True) -> None:
    """Serve the agents in `watch` (this folder by default), reloading them as their files change."""
    import uvicorn

    from webagents.agents.skills.local.secrets.keychain_ux import forbid_dialogs

    from ..daemon.identity import daemon_public_url

    # A daemon never waits on a macOS keychain dialog: nobody may be there to
    # answer, and it would be on the screen at a random later moment. A read
    # that may ask is refused with the one sentence instead (keychain-ux).
    forbid_dialogs("daemon")
    # A QUIETER DAEMON (2026-09-28, the e2e pass): it printed every skill's
    # INFO line (agents created, skills initialised, prompt sizes) to the
    # terminal. Warnings and errors now, on stderr as for every command; its
    # own lines (starting, started, each schedule's outcome) as the
    # TypeScript daemon prints them. WEBAGENTS_LOG_LEVEL still overrides.
    from webagents.utils.logging import setup_logging

    setup_logging(level="WARNING", stream=sys.stderr, tracebacks=bool(os.environ.get("WEBAGENTS_DEBUG")))
    address = _address(port, host)
    # A FAILED RUN'S OWN TEXT, FOR THE DEVELOPER'S TERMINAL (S-228), and only
    # on loopback: a daemon bound to every interface has remote callers and
    # answers them with a fixed message and a reference. The same bind decides
    # whether the registry routes ask for a credential (S-306).
    server = daemon_server(
        watch=watch,
        cron=cron,
        error_detail=address.is_loopback,
        loopback=address.is_loopback,
        public_url=daemon_public_url(address.host, address.port),
    )
    # Each schedule's outcome, `[daemon] <agent>/<schedule>: <outcome> (...)`,
    # as the TypeScript daemon says it; the module logger keeps them too.
    cron = getattr(server, "cron", None)
    if cron is not None and getattr(cron, "_log", None) is None:
        cron._log = lambda line: print(f"[daemon] {line}", flush=True)
    # A busy port is one sentence and exit 1 before the address line (`listen.py`).
    from ..listen import bind_or_refuse

    bind_or_refuse(address.host, address.port)
    # The TypeScript daemon's words (`daemon/server.ts`).
    print(f"WebAgents daemon starting on {address.base_url}")
    server.app.add_event_handler("startup", lambda: print("WebAgents daemon started", flush=True))
    try:
        uvicorn.run(server.app, host=address.host, port=address.port, log_level="warning")
    finally:
        print("WebAgents daemon stopped")
