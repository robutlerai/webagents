"""
`webagents mcp serve [path]` (plan item 1.8, 2026-09-26): the agent's tools to
an MCP client, the TypeScript CLI's command (`typescript/src/cli/mcp-serve-action.ts`):
the agent at `path` (an AGENT.md, or the folder holding one), over stdio by
default, or over Streamable HTTP with `--http <port>` (and `--host`).

THE SAME AGENT AS `serve` (`serve.load_served_agent`): its skills and model,
the stored keys, the folder, the access block. Over stdio the caller is the
person at this terminal, the owner; over HTTP the caller is whoever the
request verifies as, by `serve`'s rules, and the bind is `serve`'s too:
loopback unless the agent is meant to be reached (`serve.bind_host`).

STDOUT IS THE WIRE over stdio, so it is reserved before the agent file is even
read (`load_served_agent` prints a line when there is no file).
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional


def mcp_serve_command(path: str, http_port: Optional[int], host: Optional[str]) -> None:
    from webagents.server.mcp_server import reserve_stdout, serve_http, serve_stdio

    from .serve import bind_host, load_served_agent

    if http_port is not None and not 0 <= http_port <= 65535:
        print(f'--http takes a port number from 0 to 65535, not "{http_port}".', file=sys.stderr)
        raise SystemExit(1)

    real_stdout = reserve_stdout() if http_port is None else None
    # Over stdio the caller is the owner's own MCP client, so the owner's
    # turns; over HTTP, whoever reaches the port: never the sign-in (S-327).
    built = load_served_agent(path, for_callers=http_port is not None)
    if http_port is None:
        asyncio.run(serve_stdio(built.agent, real_stdout))
        return
    serve_http(built.agent, host=bind_host(built, host), port=http_port)
