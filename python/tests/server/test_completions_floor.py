"""The credential floor stands in front of every door to the billable endpoint.

This file is the SPECIFIC-DOOR half of the floor's coverage; the general half is
``test_billable_routes.py``, which discovers the doors instead of listing them.
Both exist on purpose: enumeration catches the door nobody thought of, and these
named cases stay readable about which door is which and why it was open.

There are THREE ways a request can reach the model on a served Python agent:

1. the dedicated ``POST /{agent}/chat/completions`` route (static agents),
2. the dynamic-agent catch-all ``dynamic_agent_http_dispatch``,
3. the per-skill static mount in ``_register_agent_http_handlers``, which turns
   every ``@http`` handler into its own FastAPI route.

They used to be closed one at a time, each with its own copy of the same ``if``,
and each new one was found only after the previous fix shipped. They are not
closed separately any more: ``CredentialFloorMiddleware`` runs above routing, so
from the floor's point of view all three are the same door — see
``webagents/server/core/credential_floor.py``.

DOOR 3 AND WHAT THE EVIDENCE ACTUALLY SHOWED. Door 3 is how
``CompletionsTransportSkill``'s ``@http("/uamp/completions")`` stayed open. It
runs the model through ``agent.process_uamp`` — same provider, same owner's
credit, just a different wire format — and nothing gated it on any door.

Being exact about the proof, because the first write-up of this overstated it:
an anonymous request did NOT execute the handler body and reach
``process_uamp``. ``uamp_completions`` is an ``async def ... yield`` — an async
GENERATOR function — so calling it returns a generator whose body has not run,
and FastAPI then fails in ``serialize_response -> jsonable_encoder`` with
``TypeError: 'async_generator' object is not iterable``. The route accepted the
anonymous request and returned the handler's generator; nothing gated it, and
the only thing standing between an anonymous caller and ``process_uamp`` was
that FastAPI never iterates that generator — which the obvious fix for the
serialization bug (wrapping it in a ``StreamingResponse``) removes. A floor that
depends on a bug elsewhere staying broken is not a floor, which is why these
tests assert the status is exactly 401: a 500 does not count as "closed".

Door 1 is also asserted to refuse BEFORE the body is parsed, matching the
TypeScript half. An anonymous caller must not be able to make the server parse
an arbitrary body, and the observable must be 401 on both SDKs rather than 401
on one and a parser error on the other. That property is now structural: the
floor is an ASGI middleware, so there is no route, no dependency and no
body-reading between the socket and the refusal.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.core.transport import CompletionsTransportSkill
from webagents.server.core.app import COMPLETIONS_PATHS, WebAgentsServer


ANON_BODY = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}


def _transport_http_handlers(skill) -> list:
    """The ``@http`` handlers a transport skill contributes, in the shape
    ``_register_agent_http_handlers`` consumes."""
    handlers = []
    for attr_name in dir(skill):
        attr = getattr(skill, attr_name, None)
        if callable(attr) and hasattr(attr, "_http_subpath"):
            handlers.append(
                {
                    "subpath": attr._http_subpath,
                    "method": getattr(attr, "_http_method", "get"),
                    "function": attr,
                    "scope": getattr(attr, "_http_scope", "all"),
                    "description": getattr(attr, "_http_description", ""),
                    "source": "completions_transport",
                }
            )
    return handlers


@pytest.fixture
def static_server():
    """A static agent whose transport-skill ``@http`` handlers really are
    mounted as FastAPI routes.

    ``_registered_http_handlers`` is REPLACED rather than appended to, because
    ``BaseAgent`` ships built-in ``/command/{path:path}`` handlers and
    ``_register_agent_http_handlers`` cannot build a signature for a
    ``{path:path}`` parameter — it aborts the whole registration loop on the
    first one, so nothing after it is mounted at all. That is a separate,
    pre-existing bug; the point here is the floor on the mount, so this fixture
    hands the mount the skill handlers on their own.
    """
    agent = BaseAgent(name="floor-agent", instructions="Test agent", scopes=["all"])

    async def fake_run(messages, stream=False, tools=None):
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}

    async def fake_run_streaming(messages, tools=None, **kwargs):
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    agent.run = fake_run
    agent.run_streaming = fake_run_streaming

    transport = CompletionsTransportSkill()
    agent.skills["completions_transport"] = transport
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(transport.initialize(agent))
    finally:
        loop.close()

    agent._registered_http_handlers = _transport_http_handlers(transport)
    return WebAgentsServer(agents=[agent])


@pytest.fixture
def anonymous(static_server):
    return TestClient(static_server.app)


class TestCompletionsFloorCoversEveryDoor:
    def test_uamp_completions_is_in_the_guarded_set(self):
        """``uamp/completions`` runs the model, so it is a completions path
        wherever it is reached from — both doors read this tuple."""
        assert "uamp/completions" in COMPLETIONS_PATHS

    def test_the_static_mount_really_mounted_the_skill_handlers(self, static_server):
        """Guard the guard: if the mount silently failed, every assertion below
        would be a 404 dressed up as a pass."""
        mounted = {getattr(r, "path", None) for r in static_server.app.routes}
        assert "/floor-agent/uamp/completions" in mounted
        assert "/floor-agent/capabilities" in mounted

    def test_static_uamp_completions_refuses_an_anonymous_caller(self, anonymous):
        """Door 3. Must be 401 — NOT the 500 the unrelated UAMPEvent bug
        produces, which is what made this look closed while it was open."""
        response = anonymous.post("/floor-agent/uamp/completions", json=ANON_BODY)
        assert response.status_code == 401, response.text
        assert "Authentication required" in response.text

    def test_static_uamp_completions_is_reachable_with_a_credential(self, static_server):
        """The 401 above is the floor talking, not a missing route: presenting
        a credential gets past it.

        What is behind the floor is currently broken for an unrelated reason
        (``UAMPEvent`` is constructed without its required ``id``), which is
        exactly why the floor had to be added on its own merits rather than
        left to that bug — hence ``raise_server_exceptions=False`` and an
        assertion on what the status is NOT."""
        client = TestClient(static_server.app, raise_server_exceptions=False)
        response = client.post(
            "/floor-agent/uamp/completions",
            json=ANON_BODY,
            headers={"Authorization": "Bearer test-service-token"},
        )
        assert response.status_code not in (401, 404), response.status_code

    def test_static_chat_completions_refuses_an_anonymous_caller(self, anonymous):
        """Door 1, for completeness — the floor the other two are measured
        against."""
        response = anonymous.post("/floor-agent/chat/completions", json=ANON_BODY)
        assert response.status_code == 401, response.text

    def test_anonymous_malformed_body_is_401_not_a_parser_error(self, anonymous):
        """Cross-SDK parity: the floor runs BEFORE the body is parsed, so an
        anonymous caller cannot make the server parse arbitrary bytes and does
        not get a parser error instead of the refusal. The TypeScript half
        asserts the same in tests/unit/server/completions-auth.test.ts."""
        for path in ("chat/completions", "uamp/completions"):
            response = anonymous.post(
                f"/floor-agent/{path}",
                content=b"{ this is not json",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 401, (path, response.status_code, response.text)

    def test_the_floor_is_scoped_and_does_not_gate_public_handlers(self, anonymous):
        """The other side of the same coin: an agent's own ``@http`` handlers
        may be public by design, so the static mount must gate only the
        completions subpaths. ``/capabilities`` and ``/models`` come through
        the SAME mount and must stay anonymous."""
        for subpath in ("capabilities", "models"):
            response = anonymous.get(f"/floor-agent/{subpath}")
            assert response.status_code == 200, (subpath, response.text)
