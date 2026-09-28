"""
The chat with a stdio MCP server (interactive-mode part 2, 2026-09-26), against
the probe server `tests/fixtures/mcp_tool/probe_server_mcpsecrets.py`.

THE CRASH THIS PINS. Rebuilding an agent that has a stdio server (`/reload`,
an `/agent` switch, `/model`, `/keys`) died with `CancelledError: Cancelled
via cancel scope ... by Task-1` at the next prompt: `stdio_client` and
`ClientSession` are anyio task groups, entered on the MCP skill's exit stack
in the chat's main task, and the new agent's servers were entered there too
before the old ones were closed, so the old scopes were exited out of order.
anyio then kept delivering a cancellation to the main task. Every server now
lives in its own task (`LocalMcpSkill._serve_connections`), so the whole
sequence below runs in one loop, with the main task's later awaits
(`asyncio.sleep`) as the place a stray cancellation would surface.

The TypeScript twin is `tests/unit/cli/chat-mcp-exit-interactive2.test.ts`.
"""

import asyncio
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from webagents.cli.repl.session import WebAgentsSession

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_tool"
PROBE_SERVER = FIXTURES / "probe_server_mcpsecrets.py"
QUALIFIED = ["probe__authorization", "probe__env"]

AGENT = f"""---
name: mcp-agent
skills:
  - openai
  - mcp:
      probe:
        command: {sys.executable}
        args: ["{PROBE_SERVER}"]
---
Read your server's environment.
"""
OTHER = "---\nname: plain\nskills:\n  - openai\n---\nPlain.\n"


@pytest.fixture(autouse=True)
def newcomer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    for var in ("ANTHROPIC_API_KEY", "WEBAGENTS_TOKEN", "WEBAGENTS_PROFILE", "ROBUTLER_LLM_PROXY_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    from webagents.cli import credentials

    credentials.set_flag_token(None)
    return project


def _mcp(session: WebAgentsSession):
    return session.built.agent.skills.get("mcp")


def test_reload_switch_model_and_keys_survive_with_a_stdio_server_and_exit_closes_it(newcomer):
    Path("AGENT.md").write_text(AGENT)
    Path("AGENT-plain.md").write_text(OTHER)
    session = WebAgentsSession(agent_path=None, interactive=True)
    session.console = Console(file=StringIO(), width=160, force_terminal=False, color_system=None, record=True)

    async def settle() -> None:
        # Where a stray cancellation from a mis-exited scope would land.
        for _ in range(5):
            await asyncio.sleep(0.05)

    def printed_since(at: int) -> str:
        return session.console.export_text(clear=False)[at:]

    def mark() -> int:
        return len(session.console.export_text(clear=False))

    async def main() -> None:
        await session.initialize()
        report = _mcp(session).server_report()
        assert report and report[0]["name"] == "probe" and report[0]["connected"] and report[0]["tools"] == QUALIFIED

        at = mark()
        await session.handle_input("/mcp")
        out = printed_since(at)
        assert "MCP servers (from AGENT.md)" in out
        assert f"● probe  stdio  2 tools: {', '.join(QUALIFIED)}" in out
        assert "To use this agent from an MCP client: webagents mcp serve (stdio) or webagents mcp serve --http <port>." in out

        # /reload, twice: each rebuild opens a new server and closes the old one.
        for change in ("Read more.", "Read even more."):
            Path("AGENT.md").write_text(AGENT.replace("Read your server's environment.", change))
            at = mark()
            await session.handle_input("/reload")
            assert "✓ Reloaded mcp-agent from AGENT.md." in printed_since(at)
            await settle()
            assert _mcp(session).server_report()[0]["connected"]

        # An agent switch away and back.
        at = mark()
        await session.handle_input("/agent plain")
        assert "✓ Now talking to plain." in printed_since(at)
        await settle()
        at = mark()
        await session.handle_input("/agent mcp-agent")
        assert "✓ Now talking to mcp-agent." in printed_since(at)
        await settle()
        assert _mcp(session).server_report()[0]["connected"]

        # /model and /keys rebuild too.
        at = mark()
        await session.handle_input("/model openai/gpt-4o")
        assert "✓ Model set to openai/gpt-4o" in printed_since(at)
        await settle()
        at = mark()
        await session.handle_input("/keys unset ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY was not stored." in printed_since(at)
        await settle()
        skill = _mcp(session)
        assert skill.server_report()[0]["connected"]
        runner = skill._runner
        assert runner is not None and not runner.done()

        # /exit: the server is closed by the task that opened it, and nothing is left running.
        await session._cleanup_agent(session.built)
        assert skill.sessions == {} and skill._runner is None
        assert runner.done()
        await settle()

    asyncio.run(main())
