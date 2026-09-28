"""
One agent at the root (2026-09-26, the new-developer e2e run): `webagents
serve` answered `POST /a2a` at the root with 405, served the v1.0 card only
under `/<name>/` with RELATIVE interface URLs and a relative `jku`, and its
registration card's URLs were relative too, so a TypeScript peer could not
call a served Python agent. `RootMount` (`server/core/root_mount.py`) answers
the agent's routes at the root and the server's `root_agent` makes the base
URL the agent's principal, so every card names absolute root URLs, as the
TypeScript `serve()` does. The cross-SDK proof is
`typescript/tests/interop/a2a-cross-sdk.test.ts`, which calls the Python
agent at its root.
"""

from __future__ import annotations

import asyncio
import base64
import json

from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill
from webagents.agents.tools.decorators import handoff
from webagents.server.core.app import WebAgentsServer, create_server
from webagents.server.core.root_mount import ROOT_PATHS, RootMount, is_root_path

ORIGIN = "http://localhost:18812"
AUTH = {"Authorization": "Bearer caller-one"}


class EchoLLM(Skill):
    @handoff(name="echo-llm")
    async def echo(self, messages, tools=None, **kwargs):
        texts = [m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
        reply = "echo: " + " ".join(texts)
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def _served(tmp_path, **options):
    agent = BaseAgent(name="a2a-agent", instructions="Answers over A2A.", skills={"echo-llm": EchoLLM(), "a2a": A2ATransportSkill({})})
    agent.description = "Answers over A2A"
    asyncio.run(agent._ensure_skills_initialized())
    server = create_server(
        agents=[agent],
        public_url=ORIGIN,
        root_agent="a2a-agent",
        keys_dir=str(tmp_path / "keys"),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
        **options,
    )
    return TestClient(RootMount(server.app, "a2a-agent"))


def test_the_root_paths_are_the_agents_and_nothing_else():
    assert "/chat/completions" in ROOT_PATHS and "/a2a" in ROOT_PATHS
    assert is_root_path("/a2a/message:send") and is_root_path("/.well-known/agent-card.json")
    assert not is_root_path("/health") and not is_root_path("/a2a-agent/a2a") and not is_root_path("/")

    seen = []

    async def app(scope, receive, send):
        seen.append(scope["path"])

    wrapped = RootMount(app, "helper")
    for path in ("/chat/completions", "/a2a", "/a2a/tasks/x", "/.well-known/agent.json", "/helper/a2a", "/health"):
        asyncio.run(wrapped({"type": "http", "path": path}, None, None))
    assert seen == ["/helper/chat/completions", "/helper/a2a", "/helper/a2a/tasks/x", "/helper/.well-known/agent.json", "/helper/a2a", "/health"]


def test_the_v1_card_at_the_root_names_absolute_interfaces_and_an_absolute_jku(tmp_path):
    client = _served(tmp_path)
    card = client.get("/.well-known/agent-card.json")
    assert card.status_code == 200
    body = card.json()
    assert [i["url"] for i in body["supportedInterfaces"]] == [f"{ORIGIN}/a2a", f"{ORIGIN}/a2a"]
    protected = body["signatures"][0]["protected"]
    header = json.loads(base64.urlsafe_b64decode(protected + "=" * (-len(protected) % 4)))
    assert header["jku"] == f"{ORIGIN}/.well-known/jwks.json"
    # The same card under the agent's name, naming the root.
    named = client.get("/a2a-agent/.well-known/agent-card.json").json()
    assert named["supportedInterfaces"][0]["url"] == f"{ORIGIN}/a2a"
    # The key set the jku names is served at the origin.
    assert client.get("/.well-known/jwks.json").status_code == 200


def test_the_registration_card_at_the_root_self_names_the_root(tmp_path):
    client = _served(tmp_path)
    card = client.get("/.well-known/agent.json").json()
    assert card["client_id"] == f"{ORIGIN}/.well-known/agent.json"
    assert card["url"] == ORIGIN
    assert card["jwks_uri"] == f"{ORIGIN}/.well-known/jwks.json"
    assert card["description"] == "Answers over A2A"


def test_post_a2a_at_the_root_is_accepted_and_still_needs_a_credential(tmp_path):
    client = _served(tmp_path)
    body = {"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": {"message": {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"text": "hello over a2a"}]}}}
    refused = client.post("/a2a", json=body, headers={"A2A-Version": "1.0"})
    assert refused.status_code == 401
    answered = client.post("/a2a", json=body, headers={"A2A-Version": "1.0", **AUTH})
    assert answered.status_code == 200, answered.text
    result = answered.json()["result"]
    assert result["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert result["task"]["artifacts"][0]["parts"][0]["text"] == "echo: hello over a2a"
    rest = client.post("/a2a/message:send", json={"message": {"messageId": "m-3", "role": "ROLE_USER", "parts": [{"text": "rest binding"}]}}, headers={"A2A-Version": "1.0", **AUTH})
    assert rest.status_code == 200, rest.text
    assert rest.json()["task"]["artifacts"][0]["parts"][0]["text"] == "echo: rest binding"


def test_without_a_root_agent_a_multi_agent_server_keeps_every_agent_under_its_name(tmp_path):
    agent = BaseAgent(name="one", instructions="x", skills={"echo-llm": EchoLLM(), "a2a": A2ATransportSkill({})})
    asyncio.run(agent._ensure_skills_initialized())
    server = WebAgentsServer(agents=[agent], public_url=ORIGIN, keys_dir=str(tmp_path / "keys"), enable_monitoring=False, enable_prometheus=False, enable_rate_limiting=False, enable_request_logging=False, heartbeat=False, quiet=True)
    client = TestClient(server.app)
    assert client.get("/.well-known/agent-card.json").status_code == 404
    card = client.get("/one/.well-known/agent-card.json").json()
    assert card["supportedInterfaces"][0]["url"] == f"{ORIGIN}/one/a2a"
