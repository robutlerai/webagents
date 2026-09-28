#!/usr/bin/env python3
"""The Python half of the A2A v1.0 cross-SDK test
(`typescript/tests/interop/a2a-cross-sdk.test.ts`, plan item 1.3, 2026-09-26).

Started by that test with the TypeScript agent already listening. It first
CALLS the TypeScript agent over A2A as a client (`a2a_client.call_agent`:
the card, its signature verified against the key set the TypeScript server
publishes, one `SendMessage`, the reply), then SERVES a Python echo agent
with the A2A transport on a loopback port of its own, signed with the key
set `WebAgentsServer` mints for it, and prints ONE JSON line on stdout:

    {"ready": true, "port": <port>, "ts_call": {"reply": ..., "verified": {...}}}

The test reads that line, calls the Python agent over A2A the other way and
verifies the Python card. The process serves until it is killed.

Usage:
    serve_a2a_echo.py --ts-url http://127.0.0.1:PORT/agents/ts-echo --ts-token TOKEN
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
    from webagents.agents.skills.core.transport.a2a.a2a_client import A2AClientError, call_agent
    from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill
    from webagents.agents.tools.decorators import handoff, tool
    from webagents.server.core.app import WebAgentsServer
    from webagents.server.core.root_mount import RootMount
except ImportError as error:  # pragma: no cover - the test skips with the reason
    print(json.dumps({"ready": False, "error": f"missing dependency: {error}"}), flush=True)
    sys.exit(1)

AGENT_NAME = "py-echo"
REPLY_PREFIX = "echo: "


class EchoLLM(Skill):
    """Answers `echo: ` plus the user texts, like the fixture agent of the unit tests."""

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


class ToolsSkill(Skill):
    @tool(name="public_lookup", description="Anyone may call this")
    async def public_lookup(self, q: str) -> str:
        return "ok"

    @tool(name="owner_only_admin", description="Only the owner may call this", scope="owner")
    async def owner_only_admin(self) -> str:
        return "ok"


async def call_typescript(ts_url: str, token: str) -> Dict[str, Any]:
    try:
        result = await call_agent(ts_url, "hello from python", token=token, verify_card=True, verify={"allow_http": True}, timeout=30.0)
    except A2AClientError as error:
        return {"error": str(error), "code": error.code, "status": error.status}
    verified = result["verified"]
    task = result.get("task") or {}
    return {
        "reply": result["reply"],
        "card_url": result["card_url"],
        "rpc_url": result["rpc_url"],
        "state": (task.get("status") or {}).get("state"),
        "verified": {"ok": verified.ok, "kid": verified.kid, "alg": verified.alg, "checked": verified.checked, "reason": verified.reason},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-url", required=True, help="the TypeScript agent URL (its card is at /.well-known/agent-card.json)")
    parser.add_argument("--ts-token", required=True, help="the bearer the TypeScript agent gets")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    public_url = f"http://127.0.0.1:{port}"

    ts_call = asyncio.run(call_typescript(args.ts_url, args.ts_token))

    agent = BaseAgent(
        name=AGENT_NAME,
        instructions="Python echo agent for the A2A cross-SDK test",
        skills={"echo-llm": EchoLLM(), "tools": ToolsSkill(), "a2a": A2ATransportSkill({"peers": {args.ts_url: {"token": args.ts_token}}})},
    )
    agent.description = "Python echo agent for the A2A cross-SDK test"
    asyncio.run(agent._ensure_skills_initialized())

    # AT THE ROOT, as `webagents serve` serves it (2026-09-26): the agent is
    # the server's `root_agent`, so its principal is the origin, its cards
    # name `{origin}/a2a` and a `jku` at the origin key set, and `RootMount`
    # answers `/a2a` and `/.well-known/agent-card.json` there; the named
    # paths under `/py-echo/` keep working.
    server = WebAgentsServer(
        agents=[agent],
        public_url=public_url,
        root_agent=AGENT_NAME,
        keys_dir=tempfile.mkdtemp(prefix="a2a-cross-sdk-keys-"),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
    )

    print(json.dumps({"ready": True, "port": port, "agent": AGENT_NAME, "ts_call": ts_call}), flush=True)
    config = uvicorn.Config(RootMount(server.app, AGENT_NAME), host="127.0.0.1", port=port, log_level="warning")
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()
