"""The bridge signs the ``/ws`` handshake when it has an identity and no token
(2026-09-23).

Until this day ``PortalConnectSkill`` refused at start unless a per-agent
token was configured, although the very same agent proves its identity to
the platform's HTTP surface by signing (RFC 9421, the Web Bot Auth profile,
``webagents/crypto/http_signature.py``). The platform's ``/ws`` upgrade now
accepts that signature too (portal ``lib/ws/signed-upgrade.ts``), so:

  * with an identity (the held Ed25519 keys and the agent URL, exactly what
    ``WebBotAuth`` and ``MppBuyer`` take) and no token, the handshake carries
    the three signature headers, no ``?token=``, and ``session.create``
    carries no token;
  * every connection attempt is signed afresh (single-use nonce, sixty-second
    window);
  * a token, when configured, is used exactly as before and wins over an
    identity beside it;
  * with NEITHER, the skill refuses before any socket is opened, naming both
    ways in; a loopback agent URL is refused at start too, with the fix;
  * the server hands each static agent's ``PortalConnectSkill`` the identity
    it serves the key set for (``adopt_identity``), so the stock
    ``create_server`` setup signs with no token configured at all.

Against a REAL stub portal on loopback (websockets' asyncio server) that
records each handshake's request path and headers, which is exactly what the
platform reads, and answers ``session.create`` with ``session.created``.
"""

import asyncio
import json
import re

import pytest
import websockets
from cryptography.hazmat.primitives.asymmetric import ed25519

from webagents.agents.skills.robutler.portal_connect import (
    PortalConnectSkill,
    PortalCredentialError,
)
from webagents.crypto.http_signature import SigningKey

AGENT_URL = "https://agent.example/agents/mini"
KEY_SET_URL = f"{AGENT_URL}/.well-known/jwks.json"

SIGNATURE_INPUT_RE = re.compile(
    r'^sig1=\("@method" "@authority" "@path" "@query" "signature-agent";key="sig1"\);'
    r'created=\d+;expires=\d+;keyid="[^"]+";alg="ed25519";nonce="[^"]+";tag="web-bot-auth"$'
)


def fresh_key() -> SigningKey:
    return SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())


def fake_agent_token(agent_id: str = "agent-1") -> str:
    import base64

    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    claims = {"sub": "owner-1"}
    if agent_id:
        claims["agent_id"] = agent_id
    return f"{seg({'alg': 'none'})}.{seg(claims)}.x"


class StubAgent:
    def __init__(self, name="mini"):
        self.name = name

    async def run_streaming(self, messages, tools=None):
        yield {"choices": [{"delta": {"content": "hello from mini"}}]}


class StubPortal:
    """Records every handshake (path and headers) and every frame; answers session.create."""

    def __init__(self):
        self.frames: "asyncio.Queue[dict]" = asyncio.Queue()
        self.handshakes: list = []
        self.server = None
        self.socket = None

    async def start(self):
        self.server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, socket):
        self.socket = socket
        request = socket.request
        self.handshakes.append({"path": request.path, "headers": dict(request.headers.raw_items())})
        try:
            async for raw in socket:
                frame = json.loads(raw)
                await self.frames.put(frame)
                if frame.get("type") == "session.create":
                    await socket.send(json.dumps({
                        "type": "session.created",
                        "session_id": f"sess_{len(self.handshakes)}",
                        "session": {"agent": frame["session"]["agent"]},
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def drop_socket(self):
        if self.socket is not None:
            await self.socket.close()

    async def next_frame(self, of_type=None, timeout=5.0):
        async def _pull():
            while True:
                frame = await self.frames.get()
                if of_type is None or frame.get("type") == of_type:
                    return frame
        return await asyncio.wait_for(_pull(), timeout=timeout)

    async def wait_for_handshakes(self, count, timeout=5.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.handshakes) < count:
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError(f"expected {count} handshakes, saw {len(self.handshakes)}")
            await asyncio.sleep(0.02)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()


def header(handshake, name):
    for key, value in handshake["headers"].items():
        if key.lower() == name.lower():
            return value
    return None


def assert_signed(handshake):
    assert SIGNATURE_INPUT_RE.match(header(handshake, "Signature-Input") or ""), header(handshake, "Signature-Input")
    assert re.match(r"^sig1=:[A-Za-z0-9+/]+=*:$", header(handshake, "Signature") or "")
    assert header(handshake, "Signature-Agent") == f'sig1="{KEY_SET_URL}";type=jwks_uri'
    assert header(handshake, "Content-Digest") is None


def nonce_of(handshake):
    return re.search(r'nonce="([^"]+)"', header(handshake, "Signature-Input")).group(1)


@pytest.fixture(autouse=True)
def _no_token_in_env(monkeypatch):
    monkeypatch.delenv("WEBAGENTS_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("WEBAGENTS_ALLOW_UNBOUND_TOKEN", raising=False)


async def test_identity_and_no_token_signs_the_handshake_and_every_reconnect():
    portal = StubPortal()
    port = await portal.start()
    skill = PortalConnectSkill({
        "portal_ws_url": f"ws://127.0.0.1:{port}/ws",
        "signing_keys": [fresh_key()],
        "agent_url": AGENT_URL,
        "reconnect_delay": 0,
    })
    try:
        await skill.initialize(StubAgent())

        created = await portal.next_frame("session.create")
        assert created["session"] == {"agent": "mini"}
        assert portal.handshakes[0]["path"] == "/ws"
        assert_signed(portal.handshakes[0])
        assert header(portal.handshakes[0], "Authorization") is None

        # The socket drops; the reconnect carries a NEW signature.
        await portal.drop_socket()
        await portal.wait_for_handshakes(2)
        await portal.next_frame("session.create")
        assert_signed(portal.handshakes[1])
        assert nonce_of(portal.handshakes[1]) != nonce_of(portal.handshakes[0])
        assert header(portal.handshakes[1], "Signature") != header(portal.handshakes[0], "Signature")
    finally:
        await skill.disconnect()
        await portal.stop()


async def test_a_configured_token_is_used_as_before_and_wins_over_the_identity():
    portal = StubPortal()
    port = await portal.start()
    token = fake_agent_token()
    skill = PortalConnectSkill({
        "portal_ws_url": f"ws://127.0.0.1:{port}/ws",
        "agents": [{"name": "mini", "token": token}],
        "signing_keys": [fresh_key()],
        "agent_url": AGENT_URL,
        "auto_reconnect": False,
    })
    try:
        await skill.initialize(StubAgent())
        created = await portal.next_frame("session.create")
        assert created["session"] == {"agent": "mini", "token": token}
        assert portal.handshakes[0]["path"] == f"/ws?token={token}"
        assert header(portal.handshakes[0], "Signature-Input") is None
        assert header(portal.handshakes[0], "Signature") is None
    finally:
        await skill.disconnect()
        await portal.stop()


async def test_neither_token_nor_identity_is_refused_before_any_socket_naming_both_ways_in():
    portal = StubPortal()
    port = await portal.start()
    skill = PortalConnectSkill({"portal_ws_url": f"ws://127.0.0.1:{port}/ws", "auto_reconnect": False})
    try:
        with pytest.raises(PortalCredentialError) as refused:
            await skill.initialize(StubAgent())
        assert "WEBAGENTS_AGENT_TOKEN" in str(refused.value)
        assert "sign" in str(refused.value).lower()
        assert skill._connection_task is None
        await asyncio.sleep(0.05)
        assert portal.handshakes == []
    finally:
        await skill.disconnect()
        await portal.stop()


def test_check_portal_credential_rules():
    from webagents.agents.skills.robutler.portal_connect import check_portal_credential

    key = fresh_key()
    # A public https identity signs; a bound token passes; an unbound token is refused as before.
    check_portal_credential("", identity=([key], AGENT_URL))
    check_portal_credential(fake_agent_token(), identity=None)
    with pytest.raises(PortalCredentialError, match="api-key"):
        check_portal_credential(fake_agent_token(agent_id=""), identity=None)
    # A loopback agent URL cannot be fetched from by the platform: refused with the fix.
    with pytest.raises(PortalCredentialError) as loopback:
        check_portal_credential("", identity=([key], "http://localhost:8000/agents/mini"))
    assert "WEBAGENTS_PUBLIC_URL" in str(loopback.value)
    assert "WEBAGENTS_AGENT_TOKEN" in str(loopback.value)
    # An identity with no key cannot sign anything.
    with pytest.raises(PortalCredentialError):
        check_portal_credential("", identity=([], AGENT_URL))
    # Neither.
    with pytest.raises(PortalCredentialError, match="WEBAGENTS_AGENT_TOKEN"):
        check_portal_credential("", identity=None)


async def test_adopt_identity_then_start_signs_unless_one_was_configured():
    portal = StubPortal()
    port = await portal.start()
    skill = PortalConnectSkill({"portal_ws_url": f"ws://127.0.0.1:{port}/ws", "auto_reconnect": False, "autostart": False})
    try:
        await skill.initialize(StubAgent())
        assert skill.signing_identity is None
        skill.adopt_identity([fresh_key()], AGENT_URL)
        assert skill.signing_identity is not None
        await skill.start()
        created = await portal.next_frame("session.create")
        assert created["session"] == {"agent": "mini"}
        assert_signed(portal.handshakes[0])
    finally:
        await skill.disconnect()
        await portal.stop()

    configured_key = fresh_key()
    configured = PortalConnectSkill({
        "portal_ws_url": "ws://127.0.0.1:1/ws",
        "signing_keys": [configured_key],
        "agent_url": "https://configured.example/agents/mini",
        "autostart": False,
    })
    configured.adopt_identity([fresh_key()], "https://served.example/agents/mini")
    keys, agent_url = configured.signing_identity
    assert keys == [configured_key]
    assert agent_url == "https://configured.example/agents/mini"


async def test_the_server_hands_its_identity_to_the_skill_so_the_stock_setup_signs(tmp_path):
    from webagents import BaseAgent, create_server

    portal = StubPortal()
    port = await portal.start()
    skill = PortalConnectSkill({"portal_ws_url": f"ws://127.0.0.1:{port}/ws", "auto_reconnect": False, "autostart": False})
    agent = BaseAgent(name="mini", instructions="You are helpful.", skills={"portal": skill})
    server = create_server(
        agents=[agent],
        public_url="https://agent.example",
        keys_dir=str(tmp_path),
        heartbeat=False,
    )
    try:
        # The startup hook that starts every attached PortalConnectSkill.
        await server._start_portal_connect_skills()
        created = await portal.next_frame("session.create")
        assert created["session"] == {"agent": "mini"}
        handshake = portal.handshakes[0]
        # The signature names the key set THIS server serves for the agent: the
        # principal `{public_url}{url_prefix}/{agent}`.
        assert header(handshake, "Signature-Agent") == 'sig1="https://agent.example/mini/.well-known/jwks.json";type=jwks_uri'
        keys, agent_url = skill.signing_identity
        assert agent_url == "https://agent.example/mini"
        assert re.search(r'keyid="([^"]+)"', header(handshake, "Signature-Input")).group(1) == keys[0].thumbprint
    finally:
        await skill.disconnect()
        await portal.stop()
