"""
A port that is already in use, said in one sentence before anything else
claims to be listening (2026-09-26, the new-developer e2e run): `webagents
serve --port <busy>` printed `[webagents] my-agent on http://127.0.0.1:<port>`
and THEN uvicorn's `[Errno 48] error while attempting to bind on address`.
`bind_or_refuse` binds a probe socket first, so a busy port is one sentence
on stderr and exit 1 with the address line never printed, and a free one is
released for uvicorn to bind a moment later. The TypeScript CLI prints the
same sentence (`server/listen-error.ts`), pinned by
`tests/fixtures/cli/listen.json`; `serve`, `daemon` and `mcp serve --http`
all go through here.
"""

from __future__ import annotations

import errno
import socket
import sys

#: The sentence, with `{port}` and `{host}` (the fixture's `port_in_use`).
PORT_IN_USE = "Port {port} on {host} is already in use. Stop the program using it, or pass --port with a free one."

_IN_USE = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE)}


def port_in_use_sentence(host: str, port: int) -> str:
    return PORT_IN_USE.format(port=port, host=host)


def port_is_free(host: str, port: int) -> bool:
    """Whether a listener could bind `host:port` now: a probe socket, bound
    with the same reuse option uvicorn sets, then closed."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError as error:
        if error.errno in _IN_USE:
            return False
        raise
    finally:
        probe.close()


def bind_or_refuse(host: str, port: int) -> None:
    """Say the sentence on stderr and exit 1 when `host:port` is taken (module docstring)."""
    if not port_is_free(host, port):
        print(port_in_use_sentence(host, port), file=sys.stderr, flush=True)
        raise SystemExit(1)
