"""
S-224: a server that cannot verify its callers answers loopback origins only (2026-09-24).

Measured before the fix against the daemon `webagents connect` starts: an
unrelated origin's preflight for `POST .../chat/completions` came back approved,
that origin echoed with `access-control-allow-credentials: true`, so any web page
could drive a local agent with `Authorization: Bearer <anything>`. WebSocket
routes had no origin check at all.
"""

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from webagents import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.server.core.app import create_server
from webagents.server.core.origin_policy import (
    cors_middleware_kwargs,
    is_loopback_origin,
    origin_allowed,
)

EVIL = "https://unrelated.example"
LOCAL_UI = "http://localhost:5173"


class AuthSkill(Skill):
    """Stands in for the platform AuthSkill: the rule keys on the class name."""


def _client(skills=None, **server_kwargs):
    agent = BaseAgent(name="mini", instructions="x", model="openai/gpt-4o-mini", skills=skills or {})
    server = create_server(agents=[agent], heartbeat=False, **server_kwargs)
    return TestClient(server.app)


def _preflight(client, origin):
    return client.options(
        "/mini/chat/completions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


def test_loopback_origins_are_recognised():
    for origin in ("http://localhost:3000", "http://127.0.0.1:8080", "http://[::1]:9000"):
        assert is_loopback_origin(origin)
    for origin in (EVIL, "http://localhost.evil.example", "http://127.0.0.1.nip.io"):
        assert not is_loopback_origin(origin)


def test_the_rule_and_its_overrides():
    assert "allow_origin_regex" in cors_middleware_kwargs(None, False)
    assert cors_middleware_kwargs(None, True)["allow_origins"] == ["*"]
    assert cors_middleware_kwargs("*", False)["allow_origins"] == ["*"]
    assert cors_middleware_kwargs([EVIL], False)["allow_origins"] == [EVIL]
    assert cors_middleware_kwargs(False, True) is None
    assert not origin_allowed(None, False, EVIL)
    assert origin_allowed(None, False, LOCAL_UI)
    assert origin_allowed(None, False, None)  # not a browser: left to the floor


def test_an_unauthenticated_server_does_not_approve_an_unrelated_origin():
    response = _preflight(_client(), EVIL)
    # THE BUG: the origin came back echoed, with credentials allowed.
    assert "access-control-allow-origin" not in response.headers


def test_a_local_page_on_another_port_still_works():
    response = _preflight(_client(), LOCAL_UI)
    assert response.headers.get("access-control-allow-origin") == LOCAL_UI


def test_a_server_whose_agents_verify_callers_stays_permissive():
    response = _preflight(_client(skills={"auth": AuthSkill()}), EVIL)
    assert response.headers.get("access-control-allow-origin") in ("*", EVIL)


def test_an_explicit_setting_wins():
    response = _preflight(_client(cors_origins=[EVIL]), EVIL)
    assert response.headers.get("access-control-allow-origin") == EVIL


def _handshake_close_code(client, origin):
    try:
        with client.websocket_connect("/mini/uamp?token=anything", headers={"origin": origin}):
            return None
    except WebSocketDisconnect as closed:
        return closed.code


def test_a_cross_origin_websocket_handshake_is_refused_by_the_guard():
    client = _client()
    # 1008 is the guard's own refusal, before any route or the floor runs.
    # THE BUG: CORS never covered sockets, so nothing refused this page.
    assert _handshake_close_code(client, EVIL) == 1008
    # A local page gets past the guard (the route then does whatever it does).
    assert _handshake_close_code(client, LOCAL_UI) != 1008
