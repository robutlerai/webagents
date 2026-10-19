"""Announcing a published set: on the attached control session, with the
manifest inlined and the relay as the default location; waiting for a session
that is not up yet; a refusal reaching the caller; a closed connection being
no session; and the portal-connect skill attaching each agent's session and
routing ``artifact.error`` to the relay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from unittest.mock import AsyncMock

import pytest

from webagents.artifacts import (
    ArtifactAnnounceError,
    ArtifactProducer,
    ArtifactRegistry,
    ArtifactRelay,
    canonical_manifest_bytes,
    files,
    parse_artifact_set,
)

from .test_relay import Wire, until


def _published(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "a.bin").write_bytes(os.urandom(4096))
    registry = ArtifactRegistry()
    published = files.publish(folder, producer=ArtifactProducer(agent="https://agent.example/agents/flow", task="task-1"), registry=registry)
    return published, ArtifactRelay(registry)


def _announcements(wire):
    return [f for f in wire.json if f["type"] == "artifact.manifest"]


@pytest.mark.asyncio
async def test_announces_on_the_attached_session(tmp_path):
    published, relay = _published(tmp_path)
    wire = Wire()
    relay.attach_control(wire.send, "sess_1", "flow")

    result = await relay.announce_set(published, task_id="task-1", progress={"done": 1, "total": 1})
    assert result == {"sha256": published.sha256, "session_id": "sess_1"}
    [frame] = _announcements(wire)
    assert frame["session_id"] == "sess_1"
    assert frame["task_id"] == "task-1"
    assert frame["sha256"] == published.sha256
    assert frame["size"] == len(published.canonical)
    assert frame["locations"] == [{"kind": "relay", "agent": "flow"}]
    assert frame["progress"] == {"done": 1, "total": 1}
    # The inlined manifest canonicalises to exactly the bytes the hash names.
    inlined = canonical_manifest_bytes(parse_artifact_set(frame["manifest"]))
    assert hashlib.sha256(inlined).hexdigest() == published.sha256

    await relay.announce_set(published, task_id="task-1", inline=False, locations=[{"kind": "https", "url": "https://agent.example/agents/flow/artifacts/blobs/sha256:00"}])
    second = _announcements(wire)[1]
    assert "manifest" not in second
    assert second["locations"] == [{"kind": "https", "url": "https://agent.example/agents/flow/artifacts/blobs/sha256:00"}]


@pytest.mark.asyncio
async def test_waits_for_a_session_and_refuses_as_not_connected_after_the_wait(tmp_path):
    published, relay = _published(tmp_path)
    with pytest.raises(ArtifactAnnounceError) as refused:
        await relay.announce_set(published, task_id="task-1", connect_timeout=0.05)
    assert refused.value.code == "not_connected"

    wire = Wire()
    pending = asyncio.create_task(relay.announce_set(published, task_id="task-1"))
    await asyncio.sleep(0.05)
    assert not pending.done() and _announcements(wire) == []
    relay.attach_control(wire.send, "sess_7", "flow")
    assert await asyncio.wait_for(pending, 2) == {"sha256": published.sha256, "session_id": "sess_7"}
    assert len(_announcements(wire)) == 1


@pytest.mark.asyncio
async def test_a_refusal_reaches_the_listener_and_the_confirm_window(tmp_path):
    published, relay = _published(tmp_path)
    wire = Wire()
    relay.attach_control(wire.send, "sess_1", "flow")
    heard = []
    stop = relay.on_announce_error(heard.append)
    refusal = {"type": "artifact.error", "sha256": published.sha256, "error": {"code": "forbidden", "message": "The manifest names another producer"}}

    pending = asyncio.create_task(relay.announce_set(published, task_id="task-1", confirm_within=2.0))
    await until(lambda: len(_announcements(wire)) == 1, "artifact.manifest")
    assert await relay.handle_frame(wire.send, refusal) is True
    with pytest.raises(ArtifactAnnounceError) as refused:
        await asyncio.wait_for(pending, 2)
    assert (refused.value.code, refused.value.sha256, str(refused.value)) == ("forbidden", published.sha256, "The manifest names another producer")
    assert [e.code for e in heard] == ["forbidden"]

    # A refusal for another set does not end this one's window; silence for the window is acceptance.
    other = dict(refusal, sha256="00" * 32)
    pending = asyncio.create_task(relay.announce_set(published, task_id="task-1", confirm_within=0.15))
    await until(lambda: len(_announcements(wire)) == 2, "second artifact.manifest")
    await relay.handle_frame(wire.send, other)
    assert (await asyncio.wait_for(pending, 2))["sha256"] == published.sha256
    stop()
    await relay.handle_frame(wire.send, refusal)
    assert len(heard) == 2


@pytest.mark.asyncio
async def test_a_closed_connection_is_no_session_and_several_agents_must_be_named(tmp_path):
    published, relay = _published(tmp_path)
    wire_a, wire_b = Wire(), Wire()
    # One sender object per connection, as the transport holds it (a bound
    # method is a new object at every attribute access).
    send_a, send_b = wire_a.send, wire_b.send
    relay.attach_control(send_a, "sess_a", "alpha")
    relay.attach_control(send_b, "sess_b", "beta")
    with pytest.raises(ValueError):
        await relay.announce_set(published, task_id="task-1", connect_timeout=0.05)
    assert (await relay.announce_set(published, task_id="task-1", agent="beta"))["session_id"] == "sess_b"
    assert len(_announcements(wire_b)) == 1 and _announcements(wire_a) == []

    relay.connection_closed(send_a)
    assert relay.control_sessions == {"beta": "sess_b"}
    with pytest.raises(ArtifactAnnounceError) as refused:
        await relay.announce_set(published, task_id="task-1", agent="alpha", connect_timeout=0.05)
    assert refused.value.code == "not_connected"


@pytest.mark.asyncio
async def test_the_skill_attaches_each_session_and_routes_artifact_error(tmp_path):
    from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill

    published, relay = _published(tmp_path)
    skill = PortalConnectSkill({
        "portal_ws_url": "wss://robutler.test/ws",
        "agents": [{"name": "flow", "token": "jwt-a"}],
        "auto_reconnect": False,
        "max_reconnect_attempts": 0,
        "artifact_relay": relay,
    })
    ws = AsyncMock()
    skill._ws = ws
    skill._connected = True
    skill._relay_sender = skill._make_relay_sender(ws)
    heard = []
    relay.on_announce_error(heard.append)

    await skill._handle_message(json.dumps({"type": "session.created", "session_id": "sess_9", "session": {"agent": "flow"}}))
    assert relay.control_sessions == {"flow": "sess_9"}
    await relay.announce_set(published, task_id="task-1")
    sent = [json.loads(c.args[0]) for c in ws.send.call_args_list]
    assert [f["type"] for f in sent][-1] == "artifact.manifest"
    assert sent[-1]["session_id"] == "sess_9"

    await skill._handle_message(json.dumps({"type": "artifact.error", "sha256": published.sha256, "error": {"code": "announce_invalid", "message": "size must be an integer"}}))
    assert [e.code for e in heard] == ["announce_invalid"]

    skill._relay_connection_closed()
    assert relay.control_sessions == {}


@pytest.mark.asyncio
async def test_announces_on_the_relayed_turn_when_named(tmp_path):
    published, relay = _published(tmp_path)
    wire = Wire()
    relay.attach_control(wire.send, "sess_1", "flow")
    result = await relay.announce_set(published, task_id="task-1", session_id="turn-7")
    assert result == {"sha256": published.sha256, "session_id": "turn-7"}
    assert _announcements(wire)[-1]["session_id"] == "turn-7"
    # Without a turn it falls back to the control session.
    await relay.announce_set(published, task_id="task-1")
    assert _announcements(wire)[-1]["session_id"] == "sess_1"


def test_a_relayed_turn_keeps_its_session_on_the_run_context():
    from unittest.mock import MagicMock

    from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill, portal_turn_of
    from webagents.server.context.context_vars import get_context

    agent = MagicMock()
    PortalConnectSkill._run_as_portal_caller(agent, None, "turn-9")
    assert portal_turn_of(get_context()) == {"session_id": "turn-9"}
    PortalConnectSkill._run_as_portal_caller(agent, None)
    assert portal_turn_of(get_context()) is None
