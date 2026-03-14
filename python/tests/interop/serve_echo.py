#!/usr/bin/env python3
"""Minimal echo agent server for cross-language interop tests.

Usage:
    python serve_echo.py [PORT]

Starts a WebAgents server with a single echo agent on the given port.
The agent echoes input text back and exposes a priced tool.
"""

import sys

try:
    import uvicorn
    from webagents.agents.core.base_agent import BaseAgent
    from webagents.agents.skills.base import Skill
    from webagents.agents.tools.decorators import tool
    from webagents.server.core.app import WebAgentsServer
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install webagents in dev mode: pip install -e '.[dev]'", file=sys.stderr)
    sys.exit(1)


class EchoSkill(Skill):
    @tool(name="echo", description="Echo the input message back")
    async def echo_tool(self, message: str) -> str:
        return f"echo: {message}"

    @tool(name="priced_echo", description="Echo with pricing")
    async def priced_echo(self, text: str) -> str:
        return f"[priced] {text}"


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100

    agent = BaseAgent(
        name="py-echo",
        instructions="You are a Python echo agent for interop tests.",
        model="openai/gpt-4o",
        skills={"echo": EchoSkill()},
    )

    server = WebAgentsServer(
        agents=[agent],
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
    )

    uvicorn.run(server.app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
