"""
The small MCP server the agent-file mcp client is tested against
(`echo_server.json` holds its tools and answers; `typescript/tests/fixtures/
mcp-echo-server.mjs` is the same server in TypeScript). Over stdio by default;
with `--http <port>`, stateless Streamable HTTP at /mcp on 127.0.0.1.

Kept to the SDK's low-level Server so the tool schemas are served exactly as
the fixture writes them.
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

FIXTURE = json.loads((Path(__file__).resolve().parent / "echo_server.json").read_text())


def make_server() -> Server:
    server = Server("echo", version="0.0.1")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(**tool) for tool in FIXTURE["tools"]]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "echo":
            return [types.TextContent(type="text", text=f"echo: {arguments['message']}")]
        if name == "add":
            total = arguments["a"] + arguments["b"]
            return [types.TextContent(type="text", text=str(int(total) if float(total).is_integer() else total))]
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
            await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Route(FIXTURE["http_path"], endpoint=Endpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)
    print(f"echo server on http://127.0.0.1:{port}{FIXTURE['http_path']}", file=sys.stderr, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http(int(sys.argv[sys.argv.index("--http") + 1]))
    else:
        anyio.run(run_stdio)
