"""
Every 401 this SDK's servers answer carries the bearer challenge (2026-09-29),
against the shared fixture `tests/fixtures/credential_floor/www_authenticate.json`,
which the TypeScript suite reads too
(`tests/unit/server/every-401-challenges-noticed3.test.ts`).

`mcp serve --http` got its `WWW-Authenticate` in the morning; that afternoon
every other 401 was found bare: the floor's own refusal on `chat/completions`
and the `command` paths, the gate's `no_caller` on a scoped endpoint, an auth
hook's refusal on `chat/completions`, the scoped websocket refusal, the
daemons' registry and command routes. RFC 7235 makes the header a MUST on
every 401, and an MCP or OAuth-shaped client reads the scheme from it.

Pinned here, door by door:

  * the values are the fixture's, and the MCP fixture's `refusals` say the
    same (one rule, two fixtures, no drift);
  * the variant follows the ONE rule: `bad_credential` when the request
    carried a credential (`has_credential`), whoever refused it, and
    `no_credential` when it carried none;
  * every door the fixture says this SDK serves is probed, and every probe is
    a door the fixture names, so a new 401 fails here until it is listed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Dict

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from webagents.access.caller import CallerAuth
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.auth.skill import AuthenticationError
from webagents.agents.tools.decorators import command, hook, http, websocket
from webagents.cli.daemon.server import WebAgentsDaemon
from webagents.server.core import endpoint_gate
from webagents.server.core.app import WebAgentsServer, create_server
from webagents.server.core.credential_floor import (
    BEARER_REALM,
    bearer_challenge,
    challenge_for,
    challenge_headers,
    refusal_headers,
    unauthorized_response,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHALLENGE = json.loads((FIXTURES / "credential_floor" / "www_authenticate.json").read_text())
SERVE = json.loads((FIXTURES / "mcp_tool" / "serve.json").read_text())
HEADER = CHALLENGE["header"]
NONE = CHALLENGE["no_credential"]
BAD = CHALLENGE["bad_credential"]
DOORS = {door["name"]: door for door in CHALLENGE["doors"]}
BODY = {"messages": [{"role": "user", "content": "hi"}]}


class Identity(Skill):
    """Stands in for an auth skill: `Bearer owner-token` is the owner,
    `Bearer bad-token` is refused with the auth skill's own error, anything
    else verifies no one."""

    identifies_caller = True

    @hook("on_connection", priority=0, scope="all")
    async def who(self, context):
        headers = getattr(getattr(context, "request", None), "headers", None) or {}
        bearer = headers.get("authorization")
        if bearer == "Bearer bad-token":
            raise AuthenticationError("this bearer is refused")
        if bearer == "Bearer owner-token":
            context.auth = CallerAuth(scope="owner", provider="test")
        return context


class Doors(Skill):
    @http("/mine", method="get", scope="owner")
    async def mine(self) -> dict:
        return {"mine": True}

    @websocket("/live", scope="owner")
    async def live(self, ws) -> None:
        await ws.accept()
        await ws.send_json({"live": True})
        await ws.close()

    @command("/any/thing", description="Anyone's command", scope="all")
    async def any_thing(self) -> str:
        return "ran"


def _agent() -> BaseAgent:
    agent = BaseAgent(name="guarded", instructions="Guarded.", skills={"doors": Doors(), "identity": Identity()})
    asyncio.run(agent._ensure_skills_initialized())
    return agent


def _static() -> TestClient:
    return TestClient(WebAgentsServer(agents=[_agent()], quiet=True).app)


def _dynamic() -> TestClient:
    agent = _agent()
    return TestClient(WebAgentsServer(agents=[], dynamic_agents=lambda name, **_: agent if name == "guarded" else None, quiet=True).app)


def _expect(response, challenge: str) -> None:
    assert response.status_code == 401, response.text
    assert response.headers[HEADER] == challenge


# -- the values and the rule -------------------------------------------------------------------


def test_the_values_are_the_fixtures_and_the_mcp_fixture_agrees():
    assert BEARER_REALM == CHALLENGE["realm"]
    assert bearer_challenge() == NONE
    assert bearer_challenge(refused=True) == BAD
    assert SERVE["refusals"]["no_credential"]["www_authenticate"] == NONE
    assert SERVE["refusals"]["bad_credential"]["www_authenticate"] == BAD
    assert challenge_headers() == {HEADER: NONE} and challenge_headers(True) == {HEADER: BAD}


def test_the_one_rule_reads_the_request():
    assert challenge_for({}) == NONE
    assert challenge_for({"authorization": "Bearer"}) == NONE  # a bare Bearer is not a credential
    assert challenge_for({"authorization": "Bearer x"}) == BAD
    assert challenge_for({"x-api-key": "k"}) == BAD
    assert challenge_for({"signature-input": "sig1=()"}) == BAD
    assert refusal_headers({}, 401) == {HEADER: NONE}
    assert refusal_headers({"authorization": "Bearer x"}, 401) == {HEADER: BAD}
    assert refusal_headers({"authorization": "Bearer x"}, 403) is None
    # The floor's own response carries the plain challenge without being asked.
    assert unauthorized_response().headers[HEADER] == NONE
    # A gate refusal answered for a request carries the rule's variant; a 403 carries none.
    refused = endpoint_gate.refusal_response({"authorization": "Bearer x"}, (401, {"error": {}}))
    assert refused.status_code == 401 and refused.headers[HEADER] == BAD
    forbidden = endpoint_gate.refusal_response({"authorization": "Bearer x"}, (403, {"error": {}}))
    assert forbidden.status_code == 403 and HEADER.lower() not in forbidden.headers


# -- the doors --------------------------------------------------------------------------------


def probe_floor_billable() -> None:
    for client in (_static(), _dynamic()):
        _expect(client.post("/guarded/chat/completions", json=BODY), NONE)
    # The daemon's floor is the same middleware, off loopback.
    server = create_server(url_prefix="/agents", enable_file_watching=False, quiet=True, loopback=False)
    _expect(TestClient(server.app).post("/agents/any/chat/completions", json=BODY), NONE)


def probe_floor_credentialed() -> None:
    for client in (_static(), _dynamic()):
        _expect(client.post("/guarded/command/any/thing", json={}), NONE)
        _expect(client.get("/guarded/command"), NONE)


def _legacy_daemon() -> TestClient:
    """The legacy daemon class, off loopback, serving the guarded agent."""
    daemon = WebAgentsDaemon(port=0, host="0.0.0.0")
    daemon.manager._loaded_agents["guarded"] = _agent()
    return TestClient(daemon.app, raise_server_exceptions=False)


def probe_completions_refused_credential() -> None:
    for stream in (False, True):
        response = _static().post(
            "/guarded/chat/completions", json={**BODY, "stream": stream}, headers={"Authorization": "Bearer bad-token"}
        )
        _expect(response, BAD)
        assert response.json()["error"]["code"] == "unauthorized"
        # The legacy daemon class too (S-345, 2026-09-29): its route ran the
        # agent with no request on the context, so the hook saw no bearer and
        # refused nothing, and a refusal that did surface was a 500.
        response = _legacy_daemon().post(
            "/agents/guarded/chat/completions", json={**BODY, "stream": stream}, headers={"Authorization": "Bearer bad-token"}
        )
        _expect(response, BAD)
        assert response.json()["error"]["code"] == "unauthorized"


def probe_scoped_endpoint_anonymous() -> None:
    for client in (_static(), _dynamic()):
        response = client.get("/guarded/mine")
        _expect(response, NONE)
        assert response.json()["error"]["message"] == endpoint_gate.NEEDS_CALLER


def probe_scoped_endpoint_made_up_credential() -> None:
    for client in (_static(), _dynamic()):
        response = client.get("/guarded/mine", headers={"Authorization": "Bearer made-up"})
        _expect(response, BAD)
        assert response.json()["error"]["message"] == endpoint_gate.NEEDS_CALLER
        # An api key nobody verifies is a presented credential too.
        _expect(client.get("/guarded/mine", headers={"X-Api-Key": "made-up"}), BAD)


def probe_scoped_endpoint_refused_credential() -> None:
    for client in (_static(), _dynamic()):
        response = client.get("/guarded/mine", headers={"Authorization": "Bearer bad-token"})
        _expect(response, BAD)
        assert response.json()["error"]["message"] == "this bearer is refused"
        # And the owner still gets in: the challenge changed nothing else.
        assert client.get("/guarded/mine", headers={"Authorization": "Bearer owner-token"}).json() == {"mine": True}


def probe_scoped_websocket_anonymous() -> None:
    with pytest.raises(WebSocketDenialResponse) as refused:
        with _dynamic().websocket_connect("/guarded/live"):
            pass
    assert refused.value.status_code == 401
    assert refused.value.headers[HEADER] == NONE
    # A `?token` a browser sends is a presented credential: the same rule.
    with pytest.raises(WebSocketDenialResponse) as refused:
        with _dynamic().websocket_connect("/guarded/live?token=made-up"):
            pass
    assert refused.value.status_code == 401
    assert refused.value.headers[HEADER] == BAD


def probe_daemon_registry_anonymous(tmp_path: Path) -> None:
    server = create_server(url_prefix="/agents", enable_file_watching=True, watch_dirs=[tmp_path], quiet=True, loopback=False)
    _expect(TestClient(server.app).post("/agents/", json={"path": str(tmp_path / "AGENT.md")}), NONE)
    _expect(TestClient(server.app).delete("/agents/reg-probe"), NONE)
    legacy = TestClient(WebAgentsDaemon(port=0, host="0.0.0.0").app)
    _expect(legacy.post("/agents/", json={"path": str(tmp_path / "AGENT.md")}), NONE)
    _expect(legacy.post("/scan?path=."), NONE)


def probe_commands_anonymous() -> None:
    # On loopback the legacy daemon installs no floor, so its command route's own 401 answers.
    _expect(TestClient(WebAgentsDaemon(port=0).app).post("/agents/any/command/any/thing", json={}), NONE)


def probe_mcp_anonymous() -> None:
    # Pinned by tests/cli/test_mcp_serve.py against mcp_tool/serve.json; the values agree.
    assert SERVE["refusals"]["no_credential"] == {"status": 401, "www_authenticate": NONE}


def probe_mcp_refused_credential() -> None:
    assert SERVE["refusals"]["bad_credential"] == {"status": 401, "www_authenticate": BAD}


PROBES: Dict[str, Callable] = {
    "floor_billable": probe_floor_billable,
    "floor_credentialed": probe_floor_credentialed,
    "completions_refused_credential": probe_completions_refused_credential,
    "scoped_endpoint_anonymous": probe_scoped_endpoint_anonymous,
    "scoped_endpoint_made_up_credential": probe_scoped_endpoint_made_up_credential,
    "scoped_endpoint_refused_credential": probe_scoped_endpoint_refused_credential,
    "scoped_websocket_anonymous": probe_scoped_websocket_anonymous,
    "daemon_registry_anonymous": probe_daemon_registry_anonymous,
    "commands_anonymous": probe_commands_anonymous,
    "mcp_anonymous": probe_mcp_anonymous,
    "mcp_refused_credential": probe_mcp_refused_credential,
}


def test_every_door_this_sdk_serves_is_probed_and_every_probe_is_a_door():
    served = {name for name, door in DOORS.items() if door["python"] is not None}
    assert set(PROBES) == served
    for name, door in DOORS.items():
        if door["python"] is None:
            assert door["python_why"], name
        assert door["challenge"] in ("no_credential", "bad_credential"), name


@pytest.mark.parametrize("name", sorted(DOORS))
def test_the_door_answers_its_challenge(name, tmp_path):
    door = DOORS[name]
    if door["python"] is None:
        pytest.skip(door["python_why"])
    probe = PROBES[name]
    # Each probe asserts the fixture's variant for its door in its own body;
    # the door's `challenge` says which, and the two must not disagree.
    expected = NONE if door["challenge"] == "no_credential" else BAD
    assert ("NONE" if expected == NONE else "BAD") in _names_used(probe), f"{name} probes the other variant"
    if name == "daemon_registry_anonymous":
        probe(tmp_path)
    else:
        probe()


def _names_used(probe: Callable) -> set:
    return set(probe.__code__.co_names)
