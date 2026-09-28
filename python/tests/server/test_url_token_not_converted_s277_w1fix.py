"""
S-277 (2026-09-26): the Python agent server turned a `?token=` query parameter
into an `Authorization: Bearer` header when the request's Host named localhost,
and the Host is whatever the client's header says. The conversion is deleted:
a URL names a caller only through the auth skills now.

This exercises the exact condition the old code triggered on (a localhost
Host, which `TestClient(base_url="http://localhost")` sends): a `?token=` no
longer becomes an `Authorization` header the handler or the auth skills can
see, so a scoped endpoint stays refused and a real header still works. The
WebSocket `?token` (browsers cannot set a handshake header) is a separate,
kept reader and is not touched here.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from webagents.access.caller import CallerAuth
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http, hook
from webagents.server.core.app import WebAgentsServer


class Echo(Skill):
    """An open endpoint that reports the Authorization header it received, and
    an owner-only one, plus an identity that trusts a real Bearer header."""

    identifies_caller = True

    @hook("on_connection", priority=0, scope="all")
    async def who(self, context):
        headers = getattr(getattr(context, "request", None), "headers", None) or {}
        if headers.get("authorization") == "Bearer owner-token":
            context.auth = CallerAuth(scope="owner", provider="test")
        return context

    @http("/headers", method="get")
    async def headers(self) -> dict:
        request = getattr(self.get_context(), "request", None)
        got = (getattr(request, "headers", None) or {}).get("authorization")
        return {"authorization": got}

    @http("/mine", method="get", scope="owner")
    async def mine(self) -> dict:
        return {"mine": True}


def _agent() -> BaseAgent:
    agent = BaseAgent(name="probe", instructions="Probe.", skills={"echo": Echo()})
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _dynamic() -> TestClient:
    agent = _agent()  # built once, outside the request loop, as the scopes suite does

    def resolve(name, **_):
        return agent if name == "probe" else None

    # base_url localhost: exactly the Host the deleted conversion keyed on.
    return TestClient(WebAgentsServer(agents=[], dynamic_agents=resolve, quiet=True).app, base_url="http://localhost")


def test_a_url_token_is_not_injected_as_an_authorization_header():
    client = _dynamic()
    response = client.get("/probe/headers?token=sekret")
    assert response.status_code == 200
    # Before the fix this was "Bearer sekret"; now the URL token reaches no header.
    assert response.json()["authorization"] is None


def test_a_url_token_does_not_authenticate_a_scoped_endpoint():
    client = _dynamic()
    assert client.get("/probe/mine?token=owner-token").status_code == 401


def test_a_real_authorization_header_still_reaches_the_handler():
    client = _dynamic()
    response = client.get("/probe/headers", headers={"authorization": "Bearer real"})
    assert response.status_code == 200
    assert response.json()["authorization"] == "Bearer real"


def test_a_real_owner_header_still_opens_the_scoped_endpoint():
    client = _dynamic()
    assert client.get("/probe/mine", headers={"authorization": "Bearer owner-token"}).status_code == 200
