"""
`nli_tool` reaches a peer over A2A v1.0 (2026-09-27, the a2a-delegate lane):
the route decided by `robutler/nli/a2a_target.py` from the shared fixture
(`tests/fixtures/a2a/delegate_routing.json`), a configured peer called on
loopback with its bearer and nothing of the caller's, an https URL probed for
a card, a card verified when its signature names a `jku`, a redirect refused,
and the platform path left as it was. The TypeScript twin is
`typescript/tests/unit/skills/nli-a2adelegate.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.core.transport.a2a import a2a_client
from webagents.agents.skills.core.transport.a2a.card import ed25519_card_signer, sign_agent_card
from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill
from webagents.agents.skills.robutler.nli import NLISkill
from webagents.agents.skills.robutler.nli.a2a_target import (
    A2A_ATTACHMENTS_REFUSED,
    A2A_EMPTY_REPLY,
    A2A_FAILED,
    A2A_PROBE_TIMEOUT_SECONDS,
    A2A_UNPAID_NOTE,
    classify_delegate_target,
    is_loopback_url,
)
from webagents.agents.tools.decorators import handoff, hook
from webagents.crypto.http_signature import SigningKey
from webagents.server.core.app import WebAgentsServer

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "a2a" / "delegate_routing.json").read_text(encoding="utf-8"))
PEER_NAME = "peer"
URL_PREFIX = "/agents"
PEER_PATH = f"{URL_PREFIX}/{PEER_NAME}"
SECRETS = ["parent-payment-token", "caller-auth-token", "platform-key-never-forwarded"]


# ---------------------------------------------------------------------------
# A peer agent, served on loopback by uvicorn
# ---------------------------------------------------------------------------


class EchoLLM(Skill):
    def __init__(self):
        super().__init__({})
        self.reply_text: Optional[str] = None

    @handoff(name="echo-llm")
    async def echo(self, messages, tools=None, **kwargs):
        texts = [m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
        reply = self.reply_text if self.reply_text is not None else "echo: " + " ".join(texts)
        yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class HeaderRecorder(Skill):
    """Keeps the headers of every request the peer admits (the A2A skill identifies each caller)."""

    identifies_caller = True

    def __init__(self):
        super().__init__({})
        self.received: List[Dict[str, str]] = []

    @hook("on_connection", priority=1)
    async def record(self, context):
        request = getattr(context, "request", None)
        if request is not None:
            self.received.append({k.lower(): v for k, v in request.headers.items()})
        return context


def build_peer(origin: str, keys_dir: Path, *, peers: Optional[Dict[str, Any]] = None):
    llm = EchoLLM()
    recorder = HeaderRecorder()
    agent = BaseAgent(name=PEER_NAME, instructions="Echo peer", skills={"echo-llm": llm, "recorder": recorder, "a2a": A2ATransportSkill({"peers": peers or {}})})
    agent.description = "A2A peer for the delegate test"
    asyncio.run(agent._ensure_skills_initialized())
    server = WebAgentsServer(
        agents=[agent],
        url_prefix=URL_PREFIX,
        public_url=origin,
        keys_dir=str(keys_dir),
        enable_monitoring=False,
        enable_prometheus=False,
        enable_rate_limiting=False,
        enable_request_logging=False,
        heartbeat=False,
        quiet=True,
    )
    return agent, llm, recorder, server


class Loopback:
    """A bound loopback socket first (so the origin is known before the server is built), then uvicorn on it."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.origin = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.server = None
        self.thread = None

    def start(self, app) -> None:
        import uvicorn

        config = uvicorn.Config(app, host="127.0.0.1", port=self.sock.getsockname()[1], log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=lambda: self.server.run(sockets=[self.sock]), daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline:
                raise AssertionError("uvicorn did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=10)


def build_caller(peers: Dict[str, Any], **nli_config: Any):
    """The calling agent: an `a2a` skill with `peers` beside the NLI skill, so `nli_tool` reads them."""
    nli = NLISkill({"transport": "http", "agent_base_url": FIXTURE["platform_base"], "timeout": 30.0, "max_retries": 0, **nli_config})
    agent = BaseAgent(name="caller", instructions="Caller", skills={"a2a": A2ATransportSkill({"peers": peers}), "nli": nli})
    return agent, nli


def context_with_secrets():
    """A run context carrying everything a peer must never see, as the platform's caller gives it."""
    ctx = MagicMock()
    ctx.auth = MagicMock(user_id="user-1")
    ctx.payments = MagicMock(payment_token="parent-payment-token")
    ctx.payment_token = "parent-payment-token"
    ctx.request = MagicMock()
    ctx.request.headers = {"X-Payment-Token": "parent-payment-token", "Authorization": "Bearer caller-auth-token"}
    ctx.get = MagicMock(return_value=None)
    return ctx


async def delegate(nli: NLISkill, caller: BaseAgent, target: str, message: str, monkeypatch) -> str:
    monkeypatch.setattr("webagents.server.context.context_vars.get_context", lambda: context_with_secrets())
    await caller._ensure_skills_initialized()
    nli._auth_token = "platform-key-never-forwarded"
    nli._resolve_agent_id = AsyncMock(return_value=None)
    nli._mint_owner_assertion = AsyncMock(return_value=None)
    try:
        return await nli.nli_tool(agent=target, message=message)
    finally:
        await nli.cleanup()


# ---------------------------------------------------------------------------
# The words and the routing, shared with TypeScript
# ---------------------------------------------------------------------------


def test_the_sentences_are_the_fixture_s():
    assert A2A_UNPAID_NOTE == FIXTURE["unpaid_note"]
    assert A2A_FAILED == FIXTURE["a2a_failed"]
    assert A2A_ATTACHMENTS_REFUSED == FIXTURE["attachments_refused"]
    assert A2A_EMPTY_REPLY == FIXTURE["result"]["empty_reply"]
    assert A2A_PROBE_TIMEOUT_SECONDS == FIXTURE["probe"]["timeout_seconds"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_routing_cases(case):
    assert classify_delegate_target(case["target"], FIXTURE["peers"], FIXTURE["platform_base"]) == case["expect"]


def test_knows_a_loopback_url():
    for url in ["http://127.0.0.1:8765/agents/x", "http://localhost/x", "https://[::1]:9/x", "http://a.localhost/x"]:
        assert is_loopback_url(url), url
    for url in ["https://peer.example/x", "http://10.0.0.1/x", "not a url"]:
        assert not is_loopback_url(url), url


# ---------------------------------------------------------------------------
# A configured peer, on loopback
# ---------------------------------------------------------------------------


def test_calls_a_configured_peer_with_its_bearer_and_nothing_of_the_caller_s(tmp_path, monkeypatch):
    box = Loopback()
    _agent, _llm, recorder, server = build_peer(box.origin, tmp_path / "keys")
    box.start(server.app)
    try:
        peer_url = f"{box.origin}{PEER_PATH}"
        caller, nli = build_caller({peer_url: {"token": "peer-token"}})
        result = asyncio.run(delegate(nli, caller, f"{peer_url}/", "hello peer", monkeypatch))
        assert result == "echo: hello peer\n" + FIXTURE["unpaid_note"].format(url=peer_url)
        # Every request the peer admitted carries only its own bearer.
        assert recorder.received, "the peer admitted nothing"
        for headers in recorder.received:
            for name in FIXTURE["never_sent_headers"]:
                assert name not in headers, f"{name} reached the peer"
            for secret in SECRETS:
                assert secret not in "\n".join(headers.values()), f"{secret} reached the peer"
        send = recorder.received[0]
        assert send["authorization"] == "Bearer peer-token"
        assert send["a2a-version"] == "1.0"
        assert nli.communication_history[-1].cost_usd == 0.0
        assert nli.communication_history[-1].target_url == peer_url
    finally:
        box.stop()


def test_an_unsigned_card_is_called_unverified_and_an_empty_reply_reads_as_no_response(tmp_path, monkeypatch):
    box = Loopback()
    _agent, llm, _recorder, server = build_peer(box.origin, tmp_path / "keys")
    llm.reply_text = ""
    box.start(server.app)
    try:
        peer_url = f"{box.origin}{PEER_PATH}"
        original = a2a_client.fetch_agent_card

        # The card the peer serves, stripped of its signature: what OpenClaw serves.
        async def unsigned(base_url, **options):
            card, card_url = await original(base_url, **options)
            card.pop("signatures", None)
            return card, card_url

        monkeypatch.setattr(a2a_client, "fetch_agent_card", unsigned)
        caller, nli = build_caller({peer_url: {"token": "peer-token"}})
        result = asyncio.run(delegate(nli, caller, peer_url, "anything", monkeypatch))
        assert result == A2A_EMPTY_REPLY + "\n" + FIXTURE["unpaid_note"].format(url=peer_url)
    finally:
        box.stop()


def test_refuses_a_card_whose_jku_does_not_verify_and_a_redirect(tmp_path, monkeypatch):
    box = Loopback()
    _agent, _llm, recorder, server = build_peer(box.origin, tmp_path / "keys")
    box.start(server.app)
    try:
        peer_url = f"{box.origin}{PEER_PATH}"
        card_url = f"{peer_url}{FIXTURE['probe']['card_paths'][0]}"
        original = a2a_client.fetch_agent_card
        from cryptography.hazmat.primitives.asymmetric import ed25519

        stranger = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())

        async def signed_by_a_stranger(base_url, **options):
            card, fetched_url = await original(base_url, **options)
            card.pop("signatures", None)
            signer = ed25519_card_signer(stranger.thumbprint, stranger.private_key, jku=f"{peer_url}/.well-known/jwks.json")
            return sign_agent_card(card, signer), fetched_url

        monkeypatch.setattr(a2a_client, "fetch_agent_card", signed_by_a_stranger)
        caller, nli = build_caller({peer_url: {"token": "peer-token"}})
        refused = asyncio.run(delegate(nli, caller, peer_url, "hello", monkeypatch))
        assert refused.startswith("❌ " + FIXTURE["a2a_failed"].format(url=peer_url, reason=""))
        assert FIXTURE["verification"]["refused"].format(card_url=card_url, reason="") in refused
        assert "does not verify" in refused
        assert recorder.received == [], "the send went out although the card did not verify"

        async def redirecting(base_url, **options):
            raise a2a_client.A2AClientError(f"{base_url} answered a redirect (302); an A2A endpoint is called where it is", status=302)

        monkeypatch.setattr(a2a_client, "fetch_agent_card", redirecting)
        caller, nli = build_caller({peer_url: {"token": "peer-token"}})
        moved = asyncio.run(delegate(nli, caller, peer_url, "hello", monkeypatch))
        assert moved.startswith("❌ " + FIXTURE["a2a_failed"].format(url=peer_url, reason=""))
        assert "redirect" in moved
    finally:
        box.stop()


def test_a_real_redirect_is_refused_by_the_client(monkeypatch):
    """The client itself, not a stub: a peer answering 3xx to the card is refused."""

    def redirecting(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://elsewhere.example/"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(redirecting)) as client:
            return await a2a_client.call_agent("https://moved.example", "hi", client=client)

    with pytest.raises(a2a_client.A2AClientError, match="redirect"):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# An https URL: probed for a card, and left on the platform path without one
# ---------------------------------------------------------------------------


def _route_every_url_to(app, monkeypatch):
    """Every httpx client the A2A client makes answers from `app`, whatever the host: the way an https peer is reached in a test."""

    def make(client, timeout):
        if client is not None:
            return client, False
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), timeout=timeout, follow_redirects=False), True

    monkeypatch.setattr(a2a_client, "_client", make)


def test_an_https_url_that_serves_a_card_goes_over_a2a_with_no_bearer(tmp_path, monkeypatch):
    origin = "https://remote.example"
    agent, _llm, recorder, server = build_peer(origin, tmp_path / "keys")
    _route_every_url_to(server.app, monkeypatch)

    async def fetch_json(url: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as client:
            return (await client.get(url)).json()

    from webagents.agents.skills.core.transport.a2a import card as card_module

    monkeypatch.setattr(card_module, "_default_fetch_json", fetch_json)
    caller, nli = build_caller({})
    result = asyncio.run(delegate(nli, caller, f"{origin}{PEER_PATH}", "hi remote", monkeypatch))
    # The floor refuses an anonymous POST to /a2a: the honest answer for a peer that demands a credential.
    assert result.startswith("❌ " + FIXTURE["a2a_failed"].format(url=f"{origin}{PEER_PATH}", reason=""))
    assert "Authentication required" in result
    # Nothing of the caller's went out, and no bearer at all.
    assert recorder.received == []


def test_stays_on_the_platform_path_when_the_url_serves_no_card(monkeypatch):
    asked: List[str] = []

    def no_card(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(404, text="nope")

    def make(client, timeout):
        return httpx.AsyncClient(transport=httpx.MockTransport(no_card), timeout=timeout), True

    monkeypatch.setattr(a2a_client, "_client", make)
    caller, nli = build_caller({})

    async def run():
        monkeypatch.setattr("webagents.server.context.context_vars.get_context", lambda: None)
        await caller._ensure_skills_initialized()
        nli._resolve_agent_id = AsyncMock(return_value=None)
        nli._mint_owner_assertion = AsyncMock(return_value=None)
        answered = MagicMock(status_code=403, text="no")
        nli.http_client = AsyncMock()
        nli.http_client.post = AsyncMock(return_value=answered)
        try:
            return await nli.nli_tool(agent="https://nocard.example/agents/x", message="hi")
        finally:
            nli.http_client = None

    result = asyncio.run(run())
    assert asked == [f"https://nocard.example/agents/x{p}" for p in FIXTURE["probe"]["card_paths"]]
    # The legacy path posted to the completions URL, as before.
    assert nli.http_client is None
    assert "Failed to communicate" in result


def test_never_probes_a_platform_agent_or_a_loopback_url_the_model_names(monkeypatch):
    def never(client, timeout):
        raise AssertionError("a card was probed")

    monkeypatch.setattr(a2a_client, "_client", never)
    caller, nli = build_caller({})

    async def run():
        monkeypatch.setattr("webagents.server.context.context_vars.get_context", lambda: None)
        await caller._ensure_skills_initialized()
        nli._resolve_agent_id = AsyncMock(return_value=None)
        nli._mint_owner_assertion = AsyncMock(return_value=None)
        nli.http_client = AsyncMock()
        nli.http_client.post = AsyncMock(return_value=MagicMock(status_code=403, text="no"))
        platform = await nli.nli_tool(agent="@bob", message="hi")
        local = await nli.nli_tool(agent="http://127.0.0.1:9999/agents/x", message="hi")
        nli.http_client = None
        return platform, local

    platform, local = asyncio.run(run())
    assert "Failed to communicate" in platform
    assert "Internal URLs are not allowed" in local
