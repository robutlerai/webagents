"""
The MCP server the secrets tests start (S-292, 2026-09-26;
`probe_server_mcpsecrets.json` holds its tools, `typescript/tests/fixtures/
mcp-probe-server-mcpsecrets.mjs` is the same server in TypeScript). Over stdio
it answers what its own environment holds, over Streamable HTTP which
Authorization header the request carried, so a test can prove what a server
actually received without any of it being printed. Over stdio by default;
with `--http <port>`, stateless Streamable HTTP at /mcp on 127.0.0.1.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

FIXTURE = json.loads((Path(__file__).resolve().parent / "probe_server_mcpsecrets.json").read_text())

_last_authorization = FIXTURE["none"]


def make_server() -> Server:
    server = Server("probe", version="0.0.1")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(**tool) for tool in FIXTURE["tools"]]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "env":
            return [types.TextContent(type="text", text=os.environ.get(str(arguments["name"]), FIXTURE["unset"]))]
        if name == "authorization":
            return [types.TextContent(type="text", text=_last_authorization)]
        raise ValueError(f"unknown tool {name}")

    return server


async def run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    server = make_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def run_http(port: int) -> None:
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Route

    manager = StreamableHTTPSessionManager(app=make_server(), json_response=True, stateless=True, security_settings=None)

    class Endpoint:  # a Route, not a Mount: a mount answers the bare path with a redirect
        async def __call__(self, scope, receive, send):
            global _last_authorization
            found = FIXTURE["none"]
            for key, value in scope.get("headers", []):
                if key == b"authorization":
                    found = value.decode("latin-1")
            _last_authorization = found
            await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Route(FIXTURE["http_path"], endpoint=Endpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)
    print(f"probe server on http://127.0.0.1:{port}{FIXTURE['http_path']}", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http(int(sys.argv[sys.argv.index("--http") + 1]))
    else:
        anyio.run(run_stdio)
