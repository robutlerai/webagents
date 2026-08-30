"""PortalConnectSkill lifecycle: initialize() must OPEN THE SOCKET.

The failure this file exists to prevent has no observable other than silence:
a skill that is constructed and initialized but never `start()`ed leaves the
process up, /health at 200, no error in any log — and receives not one turn.
Every stock setup (the documented 27-line agent, a daemon that only builds the
skill, anything relying on BaseAgent's lazy skill init) has no hook other than
initialize(), so initialize() is what has to connect.

These run against a REAL stub portal on loopback: a websockets server that
accepts the socket, answers `session.create` with `session.created`, then
pushes an `input.text` turn and reads back the deltas.
"""

import asyncio
import json

import pytest
import websockets

from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill


class StubAgent:
    """The smallest thing PortalConnectSkill can drive: a name and a
    streaming run. Deliberately not a BaseAgent — this pins the skill's own
    behaviour, not the agent's."""

    def __init__(self, name="mini"):
        self.name = name
        self.seen_messages = []

    async def run_streaming(self, messages, tools=None):
        self.seen_messages.append(messages)
        yield {"choices": [{"delta": {"content": "hello from mini"}}]}


class StubPortal:
    """Accepts one agent socket and records every frame it sends."""

    def __init__(self):
        self.frames: "asyncio.Queue[dict]" = asyncio.Queue()
        self.server = None
        self.socket = None
        self._pushed = False

    async def start(self):
        self.server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, socket):
        self.socket = socket
        try:
            async for raw in socket:
                frame = json.loads(raw)
                await self.frames.put(frame)
                if frame.get("type") == "session.create":
                    await socket.send(json.dumps({
                        "type": "session.created",
                        "session_id": "sess_1",
                        "session": {"agent": frame["session"]["agent"]},
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def push_turn(self):
        await self.socket.send(json.dumps({
            "type": "input.text",
            "session_id": "sess_1",
            "text": "Hi",
            "messages": [
                {"role": "user", "content": "earlier turn"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "Hi"},
            ],
            "payment_token": "pt_test",
        }))

    async def next_frame(self, of_type=None, timeout=5.0):
        async def _pull():
            while True:
                frame = await self.frames.get()
                if of_type is None or frame.get("type") == of_type:
                    return frame
        return await asyncio.wait_for(_pull(), timeout=timeout)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()


def make_skill(port, **overrides):
    config = {
        "portal_ws_url": f"ws://127.0.0.1:{port}/ws",
        "agents": [{"name": "mini", "token": "jwt-mini"}],
        "auto_reconnect": False,
    }
    config.update(overrides)
    return PortalConnectSkill(config)


@pytest.mark.asyncio
async def test_initialize_alone_connects_and_serves_a_turn():
    portal = StubPortal()
    port = await portal.start()
    agent = StubAgent()
    skill = make_skill(port)
    try:
        # The ONLY lifecycle call. No start(), no server, no CLI.
        await skill.initialize(agent)

        created = await portal.next_frame("session.create")
        assert created["session"]["agent"] == "mini"
        assert created["session"]["token"] == "jwt-mini"
        assert skill.is_started is True

        await portal.push_turn()
        delta = await portal.next_frame("response.delta")
        assert delta["delta"]["text"] == "hello from mini"
        await portal.next_frame("response.done")

        # The full history reached the run, not a synthesized single turn.
        assert len(agent.seen_messages[0]) == 3
    finally:
        await skill.disconnect()
        await portal.stop()


@pytest.mark.asyncio
async def test_start_after_initialize_is_idempotent():
    """`initialize()` already starts the connection, and server startup calls
    `start()` explicitly on top of it (`_start_portal_connect_skills`). That
    must not open a second socket — the platform treats a second connection as
    a takeover."""
    portal = StubPortal()
    port = await portal.start()
    skill = make_skill(port)
    try:
        await skill.initialize(StubAgent())
        first_task = skill._connection_task
        await skill.start()
        await skill.start()
        assert skill._connection_task is first_task
        await portal.next_frame("session.create")
    finally:
        await skill.disconnect()
        await portal.stop()


@pytest.mark.asyncio
async def test_autostart_false_warns_loudly(monkeypatch):
    """Opting out is allowed, but it may never be silent.

    (The SDK logger does not propagate to the root handler, so this asserts
    on the logger call itself rather than through caplog.)"""
    from webagents.agents.skills.robutler.portal_connect import skill as skill_module

    warnings: list = []
    monkeypatch.setattr(
        skill_module.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg)
    )

    portal = StubPortal()
    port = await portal.start()
    skill = make_skill(port, autostart=False)
    try:
        await skill.initialize(StubAgent())
        assert skill.is_started is False
        assert any("NOT started" in w for w in warnings), warnings
        assert portal.frames.empty()
    finally:
        await skill.disconnect()
        await portal.stop()
