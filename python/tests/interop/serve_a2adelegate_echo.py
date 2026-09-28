#!/usr/bin/env python3
"""The Python half of the delegate-over-A2A cross-SDK test
(`typescript/tests/interop/a2adelegate-cross-sdk.test.ts`, 2026-09-27, the
a2a-delegate lane of the gap-closure build).

Started by that test with the TypeScript agent already listening. It first
DELEGATES to the TypeScript agent through the NLI skill's `nli_tool`, with the
TypeScript agent configured as an `a2a` peer of the Python agent (so the hop
goes over A2A v1.0 with the peer's bearer, the card verified against the key
set the TypeScript server publishes, and the unpaid note appended), then
SERVES a Python echo agent with the A2A transport on a loopback port of its
own, at the root as `webagents serve` serves it, and prints ONE JSON line on
stdout:

    {"ready": true, "port": <port>, "agent": "py-echo", "delegate": "<the tool's text>"}

The test reads that line, delegates to the Python agent the other way and
checks the reply and the note. The process serves until it is killed.

Usage:
    serve_a2adelegate_echo.py --ts-url http://127.0.0.1:PORT/agents/ts-echo --ts-token TOKEN
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import tempfile
from typing import Any, Dict, List

try:
    import uvicorn

    from webagents.agents.core.base_agent import BaseAgent
    from webagents.agents.skills.base import Skill
    from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill
    from webagents.agents.skills.robutler.nli import NLISkill
    from webagents.agents.tools.decorators import handoff
    from webagents.server.core.app import WebAgentsServer
    from webagents.server.core.root_mount import RootMount
except ImportError as error:  # pragma: no cover - the test skips with the reason
    print(json.dumps({"ready": False, "error": f"missing dependency: {error}"}), flush=True)
    sys.exit(1)

AGENT_NAME = "py-echo"
REPLY_PREFIX = "echo: "


class EchoLLM(Skill):
    """Answers `echo: ` plus the user texts."""

    @handoff(name="echo-llm")
    async def echo(self, messages: List[Dict[str, Any]], tools=None, **kwargs):
        texts: List[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
        reply = REPLY_PREFIX + " ".join(texts)
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


async def delegate_to_typescript(agent: BaseAgent, ts_url: str) -> Dict[str, Any]:
    """`nli_tool` against the TypeScript agent, a configured peer: the text it answers, or the error."""
    await agent._ensure_skills_initialized()
    nli = agent.skills["nli"]
    try:
        text = await nli.nli_tool(agent=ts_url, message="hello from python")
    except Exception as error:  # noqa: BLE001 - reported to the test, which fails on it
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        await nli.cleanup()
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-url", required=True, help="the TypeScript agent URL (its card is at /.well-known/agent-card.json)")
    parser.add_argument("--ts-token", required=True, help="the bearer the TypeScript agent gets, configured as the peer's token")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    public_url = f"http://127.0.0.1:{port}"

    agent = BaseAgent(
        name=AGENT_NAME,
        instructions="Python echo agent for the delegate-over-A2A cross-SDK test",
        skills={
            "echo-llm": EchoLLM(),
            "a2a": A2ATransportSkill({"peers": {args.ts_url: {"token": args.ts_token}}}),
            "nli": NLISkill({"transport": "http", "agent_base_url": "https://robutler.ai", "timeout": 30.0, "max_retries": 0}),
        },
    )
    agent.description = "Python echo agent for the delegate-over-A2A cross-SDK test"
    delegated = asyncio.run(delegate_to_typescript(agent, args.ts_url))

    server = WebAgentsServer(
        agents=[agent],
        public_url=public_url,
        root_agent=AGENT_NAME,
        keys_dir=tempfile.mkdtemp(prefix="a2adelegate-cross-sdk-keys-"),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
    )

    print(json.dumps({"ready": True, "port": port, "agent": AGENT_NAME, "delegate": delegated}), flush=True)
    config = uvicorn.Config(RootMount(server.app, AGENT_NAME), host="127.0.0.1", port=port, log_level="warning")
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()
