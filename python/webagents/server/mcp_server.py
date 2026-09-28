"""
An agent's tools, served to an MCP client (plan item 1.8, 2026-09-26).

`webagents mcp serve` puts the agent behind the Model Context Protocol, so
Claude Code, Codex, OpenCode and any other MCP client call its tools the way
they call any MCP server's: over stdio (the client starts this process), or
over stateless Streamable HTTP at `/mcp` with `--http <port>`. The TypeScript
twin is `typescript/src/server/mcp.ts`; what a client sees from either is
pinned by `tests/fixtures/mcp_tool/serve.json`.

WHO IS CALLING decides what is listed and what runs, as it does for a chat
turn:
  - over stdio the caller is the person at this terminal, the agent's owner,
    as in the local chat (`access.caller.LOCAL_OWNER`);
  - over HTTP the rules are `serve`'s: a request with nothing to authenticate
    is refused by the credential floor before its body is read
    (`core/credential_floor.py`); the agent's auth skills and access block
    then say who it is (`BaseAgent.identify_caller`, through the endpoint
    gate), and a refusal they raise keeps its status and body; a caller they
    place in no group is `everyone`, and a bearer nothing verifies names no one.
The tools listed are `get_tools_for_scopes` for that caller's scopes, and a
call runs the way the agentic loop runs one: the `before_toolcall` hooks (which
is where pricing and payment live), `_execute_single_tool`, the
`after_toolcall` hooks. A tool the caller may not use is refused by name with a
JSON-RPC invalid-params error; a tool that fails answers an `isError` result,
as the protocol wants.

SCHEMAS ARE SERVED VERBATIM: the SDK's low-level `Server`, so `tools/list`
carries each tool's definition as the skill declared it, the same bytes the
TypeScript server sends for the same skill.

STDOUT IS THE WIRE over stdio: `reserve_stdout()` points `sys.stdout` at
stderr before the agent is built, because a skill that prints one line while
starting would corrupt the protocol stream; the transport gets the real one.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from webagents.server.context.context_vars import Context, create_context, set_context

#: Where the Streamable HTTP server answers.
MCP_HTTP_PATH = "/mcp"

#: Where the HTTP handler leaves the identified context for the tool handlers (the ASGI scope).
CONTEXT_SCOPE_KEY = "webagents.mcp_context"


def tool_not_open(name: str) -> str:
    """The refusal for a tool that exists and is not the caller's to use (fixture `refusals.not_open`)."""
    return f'Tool "{name}" is not open to this caller.'


def unknown_tool(name: str) -> str:
    """The refusal for a name no tool has (fixture `refusals.unknown`)."""
    return f"Unknown tool: {name}"


def reserve_stdout():
    """Keep the process's stdout for the MCP transport: everything printed from
    now on goes to stderr. Returns the real stdout, for the transport."""
    original = sys.stdout
    sys.stdout = sys.stderr
    return original


def owner_context(agent: Any) -> Context:
    """A context whose caller is the local owner, for one call over stdio."""
    from webagents.access.caller import LOCAL_OWNER

    context = create_context(messages=[], stream=False, agent=agent)
    # A copy per call, as `run_as_local_owner` makes one per turn.
    context.auth = replace(LOCAL_OWNER, groups=[], principals=[])
    return context


def mcp_tools_of(tools: List[Dict[str, Any]]) -> list:
    """Tools as MCP lists them: name, description and the JSON Schema the
    tool declares, sorted by name."""
    import mcp.types as types

    out = []
    for config in tools:
        function = (config.get("definition") or {}).get("function") or {}
        params = function.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(params.get("type"), str):
            params = {"type": "object", "properties": {}, **params}
        out.append(
            types.Tool(
                name=config["name"],
                description=function.get("description") or config.get("description") or "",
                inputSchema=params,
            )
        )
    return sorted(out, key=lambda tool: tool.name)


def build_mcp_server(agent: Any, context_for: Callable[[Any], Context]):
    """The protocol server. `context_for(request_context)` names the caller of
    each request: the local owner over stdio, the identified caller over HTTP."""
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.shared.exceptions import McpError

    from webagents import __version__

    server = Server(agent.name, version=__version__)

    def visible(context: Context) -> List[Dict[str, Any]]:
        return agent.get_tools_for_scopes(list(context.auth_scopes))

    @server.list_tools()
    async def list_tools() -> list:
        context = context_for(server.request_context)
        set_context(context)
        return mcp_tools_of(visible(context))

    # The request handler itself, not `@server.call_tool()`: that decorator
    # turns EVERY exception its function raises into an `isError` result, the
    # refusal included, and a tool the caller may not use must be a JSON-RPC
    # invalid-params error, as the TypeScript server answers it (the fixture's
    # `refusals`). `Server._handle_request` answers a raised `McpError` with
    # its error, which is the path this takes.
    async def call_tool(request: types.CallToolRequest) -> types.ServerResult:
        name = request.params.name
        arguments = request.params.arguments or {}
        context = context_for(server.request_context)
        set_context(context)
        if not any(tool.get("name") == name for tool in visible(context)):
            exists = any(tool.get("name") == name for tool in agent.get_all_tools())
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=tool_not_open(name) if exists else unknown_tool(name)))
        tool_call = {
            "id": f"mcp_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        try:
            context.set("tool_call", tool_call)
            context = await agent._execute_hooks("before_toolcall", context)
            tool_call = context.get("tool_call", tool_call)
            result = await agent._execute_single_tool(tool_call)
            context.set("tool_result", result)
            await agent._execute_hooks("after_toolcall", context)
        except McpError:
            raise
        except Exception as error:  # noqa: BLE001 - a hook's refusal (a payment error) or a failure is answered, not crashed on
            status = getattr(error, "status_code", None)
            if isinstance(status, int) and not isinstance(status, bool):
                raise McpError(types.ErrorData(code=types.INTERNAL_ERROR, message=str(error))) from error
            return types.ServerResult(
                types.CallToolResult(content=[types.TextContent(type="text", text=str(error))], isError=True)
            )
        text = result.get("content", "") if isinstance(result, dict) else str(result)
        text = text if isinstance(text, str) else json.dumps(text)
        is_error = text.startswith("Tool execution error:") or text.startswith("Error parsing tool arguments:")
        return types.ServerResult(
            types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)
        )

    server.request_handlers[types.CallToolRequest] = call_tool
    return server


async def serve_stdio(agent: Any, stdout: Any = None) -> None:
    """Serve the agent over stdio, as the owner, until the client closes the
    stream. `stdout` is the real one `reserve_stdout()` returned; without it
    the current `sys.stdout` is used. Starts the skills itself, as `serve` does."""
    from io import TextIOWrapper

    import anyio
    from mcp.server.stdio import stdio_server

    await agent._ensure_skills_initialized()
    server = build_mcp_server(agent, lambda _request_context: owner_context(agent))
    wire = anyio.wrap_file(TextIOWrapper((stdout or sys.stdout).buffer, encoding="utf-8")) if hasattr(stdout or sys.stdout, "buffer") else None
    print(f"[webagents] {agent.name}: MCP over stdio", file=sys.stderr, flush=True)
    async with stdio_server(stdout=wire) as (read, write):
        await server.run(read, write, server.create_initialization_options())


def http_app(agent: Any):
    """The ASGI app for Streamable HTTP at `/mcp`: the floor, the caller, then
    the SDK's stateless session manager, whose `run()` the app's lifespan holds."""
    from contextlib import asynccontextmanager

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from .core import endpoint_gate
    from .core.credential_floor import has_credential, unauthorized_response

    server = build_mcp_server(agent, lambda request_context: request_context.request.scope[CONTEXT_SCOPE_KEY])
    # Stateless and JSON: every request stands alone, as the TypeScript server
    # answers them. The SDK's own host check is off because the floor and the
    # loopback bind below are the guard, as they are for `serve`.
    manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True, security_settings=None)

    class Endpoint:
        """The ASGI app at `/mcp`. A class, and a `Route` rather than a
        `Mount`: a mount at `/mcp` answers the bare path with a 307 to `/mcp/`,
        which the SDK clients follow and a plain request does not, so the
        floor's 401 was a redirect to anyone probing it."""

        async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
            request = Request(scope, receive)
            if not has_credential(request):
                await unauthorized_response()(scope, receive, send)
                return
            # The body once, for the signature check and then for the transport:
            # a request body is a stream, and the identity skills read it first.
            body = await request.body()
            context = create_context(messages=[], stream=False, agent=agent, request=request)
            set_context(context)
            # `user` is a scope that is not open, so the gate identifies the caller;
            # anonymous is an answer here, not a refusal, so `identify`, not `admit`.
            context, refusal = await endpoint_gate.identify(agent, "user", context)
            if refusal is not None:
                await JSONResponse(status_code=refusal[0], content=refusal[1])(scope, receive, send)
                return
            scope[CONTEXT_SCOPE_KEY] = context
            replayed = False

            async def replay() -> Dict[str, Any]:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await manager.handle_request(scope, replay, send)

    @asynccontextmanager
    async def lifespan(_app):
        await agent._ensure_skills_initialized()
        async with manager.run():
            yield

    return Starlette(routes=[Route(MCP_HTTP_PATH, endpoint=Endpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)


def serve_http(agent: Any, *, host: str, port: int) -> None:
    """Serve the agent over stateless Streamable HTTP at `/mcp` on `host:port`."""
    import uvicorn

    # A server never waits on a macOS keychain dialog (keychain-ux, 2026-09-27).
    from webagents.agents.skills.local.secrets.keychain_ux import forbid_dialogs

    forbid_dialogs("serve")
    app = http_app(agent)
    # A busy port is one sentence and exit 1 before the address line (`cli/listen.py`).
    from webagents.cli.listen import bind_or_refuse

    bind_or_refuse(host, port)
    print(f"[webagents] {agent.name}: MCP on http://{host}:{port}{MCP_HTTP_PATH}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
