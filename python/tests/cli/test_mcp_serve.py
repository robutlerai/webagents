"""
`webagents mcp serve` (plan item 1.8, 2026-09-26): the command's words, and the
real thing: a real MCP client starts the CLI over stdio and connects to it over
Streamable HTTP, and sees what `tests/fixtures/mcp_tool/serve.json` says a
client sees (the TypeScript CLI runs the same cases in
`tests/unit/cli/mcp-serve.test.ts` and `tests/unit/server/mcp-server.test.ts`).
The CLI runs as `python -m webagents` with HOME in a temporary folder, so
nothing stored on this machine reaches it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import click
import pytest
import typer

from webagents.cli.main import app
from webagents.server.mcp_server import MCP_HTTP_PATH, tool_not_open, unknown_tool

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SERVE = json.loads((FIXTURES / "mcp_tool" / "serve.json").read_text())


def expected_tools(names: list) -> list:
    """The definitions the fixture says each tool has, as MCP lists them."""
    todo = json.loads((FIXTURES / "todo_tool" / "definitions.json").read_text())["definitions"]
    web = json.loads((FIXTURES / "web_tool" / "definition.json").read_text())
    by_name = {}
    for definition in [*todo, web]:
        function = definition["function"]
        by_name[function["name"]] = {
            "name": function["name"],
            "description": function["description"],
            "inputSchema": function["parameters"],
        }
    return [by_name[name] for name in names]


def listed(tools) -> list:
    return [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "agent"
    folder.mkdir()
    (folder / "AGENT.md").write_text(SERVE["agent_file"])
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN")}
    env.update({"HOME": str(home), "PYTHONUNBUFFERED": "1"})
    return folder, env


# -- the words ------------------------------------------------------------------------------------


def test_the_words_are_the_fixture_s():
    from webagents.cli.help_format import _description, option_term

    root = typer.main.get_command(app)
    group = root.commands[SERVE["cli"]["group"]["name"]]
    assert _description(group) == SERVE["cli"]["group"]["description"]
    serve = group.commands["serve"]
    assert _description(serve) == SERVE["cli"]["serve"]["description"]
    options = [
        [option_term(p), p.help or "", None if p.default is None else str(p.default)]
        for p in serve.params
        if isinstance(p, click.Option) and not (set(p.opts) & {"-h", "--help"})
    ]
    assert options == SERVE["cli"]["serve"]["options"]
    arguments = [[f"[{p.name}]" if not p.required else f"<{p.name}>", p.help or "", str(p.default)] for p in serve.params if isinstance(p, click.Argument)]
    assert arguments == SERVE["cli"]["serve"]["arguments"]


def test_the_fixture_pins_what_both_sdks_list():
    assert [t["name"] for t in expected_tools(SERVE["tools"]["owner"])] == sorted(SERVE["tools"]["owner"])
    assert set(SERVE["tools"]["everyone"]) <= set(SERVE["tools"]["owner"])
    assert MCP_HTTP_PATH == SERVE["http_path"]
    assert unknown_tool("no_such_tool") == SERVE["refusals"]["unknown"]["message"]
    assert tool_not_open(SERVE["call"]["name"]) == SERVE["refusals"]["not_open"]["message"]


# -- a real MCP client and the CLI -------------------------------------------------------------


async def test_over_stdio_the_owner_lists_every_tool_and_calls_one(project):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError

    folder, env = project
    params = StdioServerParameters(command=sys.executable, args=["-m", "webagents", "mcp", "serve", str(folder)], env=env)
    # `errlog` named: its default binds `sys.stderr` at the module's first
    # import, which a CliRunner test earlier in the session can make a stream
    # with no fileno (2026-09-26).
    async with stdio_client(params, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            assert listed(tools) == expected_tools(SERVE["tools"]["owner"])

            result = await session.call_tool(SERVE["call"]["name"], SERVE["call"]["arguments"])
            assert not result.isError
            assert SERVE["call"]["contains"] in "".join(c.text for c in result.content if c.type == "text")

            with pytest.raises(McpError) as refused:
                await session.call_tool("no_such_tool", {})
            assert refused.value.error.code == SERVE["refusals"]["unknown"]["code"]
            assert SERVE["refusals"]["unknown"]["message"] in refused.value.error.message


async def test_over_streamable_http_a_bearer_nothing_verifies_is_everyone(project):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.shared.exceptions import McpError

    folder, env = project
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "webagents", "mcp", "serve", str(folder), "--http", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"the CLI exited with {process.returncode}: {process.stdout.read()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError(f"nothing listened on {port}")
        url = f"http://127.0.0.1:{port}{SERVE['http_path']}"

        async with streamablehttp_client(url, headers={"Authorization": "Bearer anything"}) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert listed(tools) == expected_tools(SERVE["tools"]["everyone"])
                with pytest.raises(McpError) as refused:
                    await session.call_tool(SERVE["call"]["name"], SERVE["call"]["arguments"])
                assert refused.value.error.code == SERVE["refusals"]["not_open"]["code"]
                assert refused.value.error.message == SERVE["refusals"]["not_open"]["message"]

        # No credential: the floor answers 401 before anything reads the body.
        import httpx

        response = httpx.post(url, content="{", headers={"Content-Type": "application/json"})
        assert response.status_code == SERVE["refusals"]["no_credential"]["status"]
        with pytest.raises(Exception) as failure:
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
        assert str(SERVE["refusals"]["no_credential"]["status"]) in str(failure.value) or "401" in repr(failure.value)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
