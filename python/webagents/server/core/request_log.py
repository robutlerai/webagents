"""
One line when a request arrives and one when it is answered (2026-09-25).

The same lines in both SDKs (`typescript/src/server/request-log.ts`):

    <-- POST /chat/completions
    --> POST /chat/completions 200 15ms

This server printed nothing for a request, and the TypeScript one printed these
through Hono's logger, so `webagents serve` looked different in each. The status
is coloured only on a terminal, and never with NO_COLOR set.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

_STATUS_COLOURS = {2: 32, 3: 36, 4: 33, 5: 31}


def _coloured() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def status_text(status: int) -> str:
    """The status, coloured by class on a terminal."""
    code = _STATUS_COLOURS.get(status // 100)
    return f"\x1b[{code}m{status}\x1b[0m" if code and _coloured() else str(status)


def elapsed_text(ms: int) -> str:
    """Milliseconds under a second, whole seconds from one."""
    return f"{ms}ms" if ms < 1000 else f"{round(ms / 1000)}s"


def _print(line: str) -> None:
    print(line, flush=True)


class RequestLog:
    """ASGI middleware: the two lines per HTTP request (module docstring)."""

    def __init__(self, app: Any, print_line: Callable[[str], None] = _print) -> None:
        self.app = app
        self.print_line = print_line

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        raw = scope.get("raw_path") or scope.get("path", "/").encode()
        query = scope.get("query_string") or b""
        path = raw.decode("latin-1") + (("?" + query.decode("latin-1")) if query else "")
        self.print_line(f"<-- {method} {path}")
        started = time.monotonic()
        status = 500

        async def send_and_note(message: Any) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_and_note)
        finally:
            ms = int((time.monotonic() - started) * 1000)
            self.print_line(f"--> {method} {path} {status_text(status)} {elapsed_text(ms)}")
