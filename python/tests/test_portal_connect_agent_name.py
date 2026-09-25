"""
A per-agent key names its agent; the session is opened under THAT name
(2026-09-24).

Found walking the quickstart as a first-time developer. A per-agent key carries
`agent_name`, the platform username `<owner>.<name>`, while the quickstart's
code (and every AGENT.md) calls the agent `<name>`. The skill sent the local
name, the platform answered `session.error agent_not_found`, and the skill kept
reporting itself connected: a live socket that never received a turn. Measured
against a local cluster with a key minted by `webagents deploy`.
"""

import base64
import json

import pytest

from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill


def _token(claims):
    """An unsigned JWT-shaped token: the skill only ever reads its own claims."""
    def part(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))


class _Agent:
    def __init__(self, name):
        self.name = name


def _skill(name, token):
    skill = PortalConnectSkill(
        {"portal_ws_url": "ws://portal.test/ws", "agents": [{"name": name, "token": token}], "autostart": False}
    )
    skill._ws = _FakeSocket()
    skill.agent = _Agent(name)
    return skill


@pytest.mark.asyncio
async def test_the_session_is_opened_under_the_name_the_key_was_issued_for():
    token = _token({"agent_id": "a-1", "agent_name": "owner.mini"})
    skill = _skill("mini", token)

    await skill._send_session_create("mini", token)

    sent = skill._ws.sent[0]
    assert sent["type"] == "session.create"
    # THE BUG: this was "mini", which the platform does not know.
    assert sent["session"]["agent"] == "owner.mini"


@pytest.mark.asyncio
async def test_turns_for_the_platform_name_still_reach_the_local_agent():
    token = _token({"agent_id": "a-1", "agent_name": "owner.mini"})
    skill = _skill("mini", token)
    await skill._send_session_create("mini", token)

    # The platform echoes the name the session was opened under; turns carry it.
    resolved = await skill._resolve_agent_by_name("owner.mini")
    assert resolved is skill.agent


@pytest.mark.asyncio
async def test_a_name_that_already_matches_the_key_is_left_alone():
    token = _token({"agent_id": "a-1", "agent_name": "owner.mini"})
    skill = _skill("owner.mini", token)

    await skill._send_session_create("owner.mini", token)

    assert skill._ws.sent[0]["session"]["agent"] == "owner.mini"
    assert skill._local_name_for == {}


@pytest.mark.asyncio
async def test_a_key_without_the_claim_keeps_the_configured_name():
    token = _token({"agent_id": "a-1"})
    skill = _skill("mini", token)

    await skill._send_session_create("mini", token)

    assert skill._ws.sent[0]["session"]["agent"] == "mini"


@pytest.mark.asyncio
async def test_an_unreadable_token_keeps_the_configured_name():
    skill = _skill("mini", "not-a-jwt")

    await skill._send_session_create("mini", "not-a-jwt")

    assert skill._ws.sent[0]["session"]["agent"] == "mini"
