"""THE ENUMERATING FLOOR TEST — Python.

Every previous round of this bug was missed the same way: the tests checked the
doors somebody remembered. ``POST /{agent}/chat/completions`` was pinned, so the
next door found was the dynamic catch-all; that got pinned, so the next was the
per-skill ``@http`` static mount; that got pinned, so the next two were in
TypeScript. Then this file was written to DISCOVER routes instead of listing
them — and the round after that found three more open doors anyway, because of
the specific way it discovered.

WHAT WENT WRONG WITH THE PREVIOUS VERSION, because it is the whole design of
this one. It walked ``app.routes``, and the entire WebSocket surface of the
server is a SINGLE wildcard route, ``/{agent_name}/{ws_path:path}``. A wildcard
cannot be listed, so the test expanded the billable set THROUGH it — it probed
``/dyn/uamp`` because ``uamp`` was already in ``BILLABLE_WS_PATHS``. A socket was
therefore only ever probed if it was already known to be billable, which is the
exact inversion of what a discovery test is for. The reviewer proved it by
adding a new model-reaching handler to the very skill this fixture uses::

    @websocket("/live")
    async def probe_live(self, ws):
        await ws.accept()
        async for chunk in self.execute_handoff([...]):
            await ws.send_json(chunk)

``pytest tests/server/test_billable_routes.py`` returned 9 passed while an
anonymous socket to ``/dyn/live`` recorded ``execute_handoff: 1``. Two shipped
handlers, ``@websocket("/realtime")`` and ``@websocket("/acp/stream")``, were
open for exactly that reason. A test that can only confirm the list it was given
is a memo.

The same blindness had a second form: the fixture built its agent from ONE
transport skill, so ``A2ATransportSkill``, ``ACPTransportSkill`` and
``RealtimeTransportSkill`` — all shipped, all mounting billable routes — were
invisible. A hard-coded route list had been traded for a hard-coded skill list.

SO THIS VERSION DISCOVERS FROM THE REGISTRIES, NOT FROM THE ROUTES.

* The fixture agent is built from every transport skill the SDK EXPORTS,
  enumerated out of ``webagents.agents.skills.core.transport`` rather than
  named here. Adding a skill to the SDK is itself the thing that trips the wire.
* ``agent.get_all_http_handlers()`` and ``agent.get_all_websocket_handlers()``
  are the real registries the server dispatches from — the HTTP one at
  ``app.py:1312``, the WebSocket one at ``app.py:738``. Walking them finds a
  handler because it EXISTS, not because a path set already named it, and it
  finds it whether or not the route it should have been mounted at got mounted.
* ``app.routes`` is walked too, for the framework-level surface the registries
  know nothing about (``/health``, ``/docs``, the catch-alls themselves).

Three assertions run against what is discovered:

1. CLASSIFICATION, and it FAILS CLOSED. Every discovered handler — HTTP or
   WebSocket — must be in :data:`BILLABLE_PATHS` / :data:`BILLABLE_WS_PATHS`, or
   in the shipped public allow-list :data:`PUBLIC_SUBPATHS` /
   :data:`PUBLIC_WS_SUBPATHS`, or (for framework chrome) in
   :data:`FRAMEWORK_PUBLIC_SUBPATHS` below. A handler in none of them is a test
   failure. That is the property this file exists for; everything else is a
   consequence of it.
2. REFUSAL. Every billable route answers exactly 401 to an anonymous caller and
   every billable socket is closed at the handshake with 4401. Exactly 401, not
   "not 200": a 500 from an unrelated bug upstream of the provider is not a
   security property, and treating it as one is precisely how
   ``uamp/completions`` looked closed while it was open.
3. THE MODEL IS NOT REACHED. Counted, not inferred from a status code, on the
   socket surface as well as the HTTP one — ``agent.run``,
   ``agent.run_streaming``, ``agent.process_uamp`` and ``Skill.execute_handoff``
   all increment a counter, and every counter must be zero after the anonymous
   sweep. ``execute_handoff`` is in that list because it is how the two open
   sockets reached the provider: ``/realtime`` never touches ``agent.run``.
"""

import ast
import asyncio
import inspect
import pathlib

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import webagents.agents.skills.core.transport as transport_module
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.server.core.app import WebAgentsServer
from webagents.server.core.credential_floor import (
    BILLABLE_PATHS,
    BILLABLE_WS_PATHS,
    PUBLIC_SUBPATHS,
    PUBLIC_WS_SUBPATHS,
    is_billable_path,
    is_billable_ws_path,
)


AGENT = "og"
#: A name that is served ONLY by the dynamic catch-all. Probing the catch-all
#: under the STATIC agent's name would prove nothing: FastAPI matches the
#: concrete `/og/chat/completions` route first, so door 2 would never run. This
#: name has no concrete route, so a request to it can only arrive through
#: `dynamic_agent_http_dispatch` — and the WebSocket route only exists at all on
#: a server with `dynamic_agents` configured, so every socket probe uses it.
DYNAMIC_AGENT = "dyn"
ANON_BODY = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}

#: Server-level FastAPI chrome: not agent surface, not part of the cross-SDK
#: contract, and different in every framework — so it is allow-listed here
#: rather than in the shipped `PUBLIC_SUBPATHS`.
#:
#: `""` and `{name}` are the server's own agent listing and its DELETE; `docs`,
#: `redoc` and `openapi.json` are FastAPI's; the rest are probes a load balancer
#: reaches with no credential. None of them dispatch to a skill handler.
FRAMEWORK_PUBLIC_SUBPATHS = frozenset(
    {
        "",
        "{name}",
        "health",
        "health/detailed",
        "ready",
        "live",
        "metrics",
        "info",
        "stats",
        "cron",
        ".well-known/agent.json",
        ".well-known/jwks.json",
        ".well-known/openid-configuration",
        "docs",
        "docs/oauth2-redirect",
        "redoc",
        "openapi.json",
    }
)


def _norm(path: str) -> str:
    return path.strip("/")


def transport_skill_classes() -> dict:
    """Every transport skill the SDK EXPORTS, enumerated from the module.

    Not a list of class names. The previous fixture named one class, and the
    three skills it did not name shipped four billable routes that no test ever
    looked at. Reading ``__all__`` means adding ``FooTransportSkill`` to the SDK
    puts its ``@http`` and ``@websocket`` handlers in front of the classification
    assertion on the same commit, with nobody having to remember this file.
    """
    classes = {}
    for name in getattr(transport_module, "__all__", []):
        candidate = getattr(transport_module, name, None)
        if inspect.isclass(candidate) and issubclass(candidate, Skill):
            classes[name] = candidate
    return classes


class Counters:
    """How many times the model was actually reached — counted, not inferred
    from a status code.

    ``execute_handoff`` is counted because it is the entry point the two open
    sockets used. ``RealtimeTransportSkill`` never calls ``agent.run`` or
    ``agent.process_uamp``; it calls ``self.execute_handoff`` on the skill base
    class, so a counter set that watched only the agent methods would have
    reported zero while the provider was being billed.
    """

    def __init__(self) -> None:
        self.run = 0
        self.process_uamp = 0
        self.execute_handoff = 0

    def as_dict(self) -> dict:
        return {
            "run": self.run,
            "process_uamp": self.process_uamp,
            "execute_handoff": self.execute_handoff,
        }

    def is_zero(self) -> bool:
        return not any(self.as_dict().values())


@pytest.fixture
def server_and_counters(monkeypatch):
    """A server with EVERY shipped transport skill loaded and all the doors
    really open for business:

    * the dedicated static ``POST /og/chat/completions`` route,
    * the per-skill static mount (``@http("/uamp/completions")``, ``@http("/acp")``,
      ``@http("/tasks")`` and friends, really registered as their own FastAPI
      routes — see the route-table assertion below, which fails loudly if the
      mount silently did nothing),
    * the dynamic-agent catch-all, which only exists when ``dynamic_agents`` is
      configured,
    * the WebSocket catch-all, which likewise only exists on a dynamic server —
      a static-only Python server registers no ``APIWebSocketRoute`` at all, so
      the socket doors are reachable in the daemon/CLI shape and nowhere else.

    The skills are constructed and attached through the normal ``BaseAgent``
    constructor, so ``_register_skill_capabilities`` does the registering — the
    test discovers what the SDK really registers, not what a hand-rolled helper
    thought it would.
    """
    counters = Counters()

    async def counting_run(messages, stream=False, tools=None):
        counters.run += 1
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}

    async def counting_run_streaming(messages, tools=None, **kwargs):
        counters.run += 1
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    async def counting_process_uamp(events, **kwargs):
        counters.process_uamp += 1
        return
        yield  # pragma: no cover - makes this an async generator, as callers expect

    async def counting_execute_handoff(self, messages, tools=None, handoff_name=None, **kwargs):
        counters.execute_handoff += 1
        yield {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}

    # Patched on the CLASS, so every transport skill's `self.execute_handoff`
    # resolves to the counter — including a skill added tomorrow.
    monkeypatch.setattr(Skill, "execute_handoff", counting_execute_handoff, raising=True)

    skills = {name: cls() for name, cls in transport_skill_classes().items()}
    agent = BaseAgent(name=AGENT, instructions="Test agent", scopes=["all"], skills=skills)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(agent._ensure_skills_initialized())
    finally:
        loop.close()

    agent.run = counting_run
    agent.run_streaming = counting_run_streaming
    agent.process_uamp = counting_process_uamp

    # `_register_agent_http_handlers` builds a FastAPI signature per path
    # parameter and cannot build one for `{path:path}` — it raises on the first
    # such handler and, because the whole loop sits in one `try`, mounts NOTHING
    # after it. That is a separate pre-existing bug; hiding the built-in
    # `/command/{path:path}` handlers from the MOUNT keeps it from silently
    # emptying the static route table and turning every assertion below into a
    # 404 dressed up as a pass. The registry is restored immediately afterwards,
    # so discovery and the dynamic dispatch both still see the real thing.
    registered = agent.get_all_http_handlers()
    agent._registered_http_handlers = [h for h in registered if ":" not in h["subpath"]]

    def _resolve_dynamic(name, **kwargs):
        return agent

    server = WebAgentsServer(agents=[agent], dynamic_agents=_resolve_dynamic)
    agent._registered_http_handlers = registered

    return server, agent, counters


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_registry_handlers(agent):
    """Walk the agent's REAL handler registries.

    These are the same two lists the server dispatches from — ``app.py:1312``
    for HTTP, ``app.py:738`` for WebSocket — so a handler is found here because
    it was registered, full stop. No path set is consulted, which is what makes
    this discovery rather than confirmation.

    Returns ``(http, websockets)``: ``http`` is ``[(method, subpath, source)]``,
    ``websockets`` is ``[(subpath, source)]``.
    """
    http = [
        (h["method"].upper(), _norm(h["subpath"]), h.get("source", "?"))
        for h in agent.get_all_http_handlers()
    ]
    websockets = [
        (_norm(h["path"]), h.get("source", "?")) for h in agent.get_all_websocket_handlers()
    ]
    return sorted(set(http)), sorted(set(websockets))


def discover_routes(app):
    """Walk the REAL routing table, for the surface the registries cannot see.

    Returns ``(concrete, http_catch_all_heads, ws_catch_all_heads)``:

    * ``concrete`` — every listable route as ``(methods, path)``, including
      routes whose subpath merely CONTAINS a parameter (``tasks/{task_id}``).
      Those are classifiable by their literal template and are classified.
    * ``*_catch_all_heads`` — the mount prefixes of routes whose whole subpath is
      a parameter (``/{agent_name}/{request_path:path}``,
      ``/{agent_name}/{ws_path:path}``). Nothing can be listed through those, so
      the known sets are expanded through them instead. Note what this is NOT:
      it is no longer how sockets are discovered, only how the declared sets are
      re-probed on the door that serves them.
    """
    concrete = []
    http_heads = set()
    ws_heads = set()

    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        is_websocket = type(route).__name__ == "APIWebSocketRoute"
        # An agent-name parameter is not a catch-all — bind it and carry on.
        bound = path.replace("{agent_name}", AGENT).replace("{name}", AGENT)
        subpath = _subpath_of(bound)

        if subpath.startswith("{"):
            dynamic = path.replace("{agent_name}", DYNAMIC_AGENT).replace(
                "{name}", DYNAMIC_AGENT
            )
            head = dynamic[: dynamic.index("{")].rstrip("/")
            (ws_heads if is_websocket else http_heads).add(head)
            continue

        methods = getattr(route, "methods", set()) or set()
        concrete.append((methods, bound))

    return concrete, sorted(http_heads), sorted(ws_heads)


def _subpath_of(path: str) -> str:
    """Drop the agent mount prefix so a route can be classified by subpath."""
    normalized = _norm(path)
    if normalized == AGENT:
        return ""
    if normalized.startswith(f"{AGENT}/"):
        return normalized[len(AGENT) + 1 :]
    return normalized


def classify_http(subpath: str):
    """``"billable"``, ``"public"``, or ``None`` — and ``None`` is a failure.

    Classification is method-agnostic on purpose: a subpath is billable if the
    floor would gate its POST, whatever verb this particular handler uses. The
    floor itself is POST-only, which is what lets ``POST /tasks`` be billable
    while ``GET /tasks/{task_id}`` is a declared-public status read.
    """
    if is_billable_path(subpath):
        return "billable"
    if subpath in PUBLIC_SUBPATHS or subpath in FRAMEWORK_PUBLIC_SUBPATHS:
        return "public"
    return None


def classify_ws(subpath: str):
    """The socket half. Deliberately does NOT fall back to the HTTP allow-list:
    an HTTP path being safe says nothing about a socket at the same name, and
    ``live`` — a bare liveness probe over HTTP — is exactly the name the
    reviewer's proof-of-blindness handler used."""
    if is_billable_ws_path(subpath):
        return "billable"
    if subpath in PUBLIC_WS_SUBPATHS:
        return "public"
    return None


UNCLASSIFIED_HELP = (
    "handler(s) discovered that are in neither the billable set nor a public "
    "allow-list. If they can reach the model, add them to BILLABLE_PATHS / "
    "BILLABLE_WS_PATHS in webagents/server/core/credential_floor.py (and its "
    "TypeScript twin, or the parity test fails). If they are safe to serve "
    "anonymously, add them to PUBLIC_SUBPATHS / PUBLIC_WS_SUBPATHS there — "
    "deliberately, not to make the test pass."
)


class TestTheFixtureReallyOpensEveryDoor:
    """Guard the guard. If the mount silently failed, or a transport skill
    stopped being enumerated, every assertion below would be a 404 dressed up as
    a pass."""

    def test_every_exported_transport_skill_is_loaded(self, server_and_counters):
        _, agent, _ = server_and_counters
        exported = set(transport_skill_classes())
        assert exported, "no transport skills were discovered — the walk is broken"
        loaded = {type(s).__name__ for s in agent.skills.values()}
        assert exported <= loaded, exported - loaded

    def test_the_registries_really_hold_the_shipped_handlers(self, server_and_counters):
        _, agent, _ = server_and_counters
        http, websockets = discover_registry_handlers(agent)
        http_subpaths = {sub for _method, sub, _src in http}
        ws_subpaths = {sub for sub, _src in websockets}
        # One from each shipped transport, so a skill quietly dropping out of the
        # fixture is a failure here rather than a silent loss of coverage.
        assert {"chat/completions", "uamp/completions", "tasks", "acp"} <= http_subpaths
        assert {"uamp", "realtime", "acp/stream"} <= ws_subpaths

    def test_all_the_mount_doors_exist(self, server_and_counters):
        server, _, _ = server_and_counters
        mounted = {getattr(r, "path", None) for r in server.app.routes}
        # Door 1 — the dedicated static route.
        assert f"/{AGENT}/chat/completions" in mounted
        # Door 3 — the per-skill static mount, for three different skills.
        assert f"/{AGENT}/uamp/completions" in mounted
        assert f"/{AGENT}/tasks" in mounted
        assert f"/{AGENT}/acp" in mounted
        # Door 2 — the dynamic catch-all, and its WebSocket twin.
        assert "/{agent_name}/{request_path:path}" in mounted
        assert "/{agent_name}/{ws_path:path}" in mounted


class TestEveryDiscoveredHandlerIsClassified:
    """THE TRIPWIRE. A handler that is neither billable nor explicitly public
    fails here, so a new one — HTTP or WebSocket — cannot be added without
    someone deciding which it is."""

    def test_every_registered_http_handler_is_classified(self, server_and_counters):
        _, agent, _ = server_and_counters
        http, _ = discover_registry_handlers(agent)
        unclassified = sorted(
            {
                f"{method} {sub} (from {src})"
                for method, sub, src in http
                if classify_http(sub) is None
            }
        )
        assert unclassified == [], f"{UNCLASSIFIED_HELP} Offenders: {unclassified}"

    def test_every_registered_websocket_handler_is_classified(self, server_and_counters):
        """The assertion that did not exist, and the reason ``/realtime`` and
        ``/acp/stream`` shipped as anonymous billable endpoints. It reads the
        registry, so a brand-new ``@websocket`` handler is in front of it the
        moment it is written."""
        _, agent, _ = server_and_counters
        _, websockets = discover_registry_handlers(agent)
        assert websockets, "discovery found no WebSocket handlers — the walk is broken"
        unclassified = sorted(
            {f"{sub} (from {src})" for sub, src in websockets if classify_ws(sub) is None}
        )
        assert unclassified == [], f"{UNCLASSIFIED_HELP} Offenders: {unclassified}"

    def test_every_mounted_route_is_classified(self, server_and_counters):
        """The framework surface, which the registries know nothing about."""
        server, _, _ = server_and_counters
        concrete, _, _ = discover_routes(server.app)
        unclassified = sorted(
            {_subpath_of(path) for _methods, path in concrete if classify_http(_subpath_of(path)) is None}
        )
        assert unclassified == [], f"{UNCLASSIFIED_HELP} Offenders: {unclassified}"

    def test_the_declared_sets_are_not_dead_weight(self, server_and_counters):
        """Every WebSocket path the floor declares billable corresponds to a
        real registered handler, or is declared for cross-SDK parity.

        Without this, ``BILLABLE_WS_PATHS`` could rot into a list of paths that
        no longer exist while the handlers that replaced them go unclassified —
        green, and meaningless.
        """
        _, agent, _ = server_and_counters
        _, websockets = discover_registry_handlers(agent)
        ws_subpaths = {sub for sub, _src in websockets}
        assert set(BILLABLE_WS_PATHS) <= ws_subpaths, (
            "a path is declared billable-over-WebSocket but no handler registers "
            f"it: {sorted(set(BILLABLE_WS_PATHS) - ws_subpaths)}"
        )


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


def billable_http_targets(server, agent):
    """Every URL that must answer 401 to an anonymous POST.

    Three independent sources, unioned so that no single one of them going quiet
    can shrink the sweep: the registry (each billable handler at both its static
    mount and the dynamic catch-all), the concrete route table, and the declared
    set expanded through each HTTP catch-all.
    """
    targets = set()

    http, _ = discover_registry_handlers(agent)
    for _method, sub, _src in http:
        if classify_http(sub) == "billable":
            targets.add(f"/{AGENT}/{sub}")
            targets.add(f"/{DYNAMIC_AGENT}/{sub}")

    concrete, http_heads, _ = discover_routes(server.app)
    for _methods, path in concrete:
        if is_billable_path(path):
            targets.add(path)
    for head in http_heads:
        for candidate in BILLABLE_PATHS:
            targets.add(f"{head}/{candidate}")

    return sorted(targets)


def billable_ws_targets(server, agent):
    """Every socket URL that must be refused at the handshake.

    Registry-first: each registered handler classified billable, at the dynamic
    mount (the only place a Python WebSocket route exists). The declared set is
    expanded through the catch-all as well, so a path in the set with no handler
    behind it is still probed.
    """
    targets = set()

    _, websockets = discover_registry_handlers(agent)
    for sub, _src in websockets:
        if classify_ws(sub) == "billable":
            targets.add(f"/{DYNAMIC_AGENT}/{sub}")

    _, _, ws_heads = discover_routes(server.app)
    for head in ws_heads:
        for candidate in BILLABLE_WS_PATHS:
            targets.add(f"{head}/{candidate}")

    return sorted(targets)


class TestEveryBillableRouteRefusesAnonymousCallers:
    def test_every_discovered_billable_route_answers_exactly_401(self, server_and_counters):
        """The refusal, on every door at once.

        Failures are COLLECTED rather than asserted one at a time: the point of
        an enumerating test is to hand you the whole list of open doors, not the
        first one.
        """
        server, agent, counters = server_and_counters
        targets = billable_http_targets(server, agent)
        assert targets, "discovery found no billable routes — the walk is broken"
        # The routes the previous round proved open, named so a regression that
        # merely shrinks the sweep cannot pass quietly.
        for expected in (f"/{AGENT}/tasks", f"/{AGENT}/acp", f"/{DYNAMIC_AGENT}/a2a"):
            assert expected in targets, (expected, targets)

        client = TestClient(server.app, raise_server_exceptions=False)
        open_doors = []
        for path in targets:
            response = client.post(path, json=ANON_BODY)
            if response.status_code != 401:
                open_doors.append(f"POST {path} -> {response.status_code}")

        assert open_doors == [], f"anonymous callers were not refused: {open_doors}"
        assert counters.is_zero(), f"the model was reached anonymously: {counters.as_dict()}"

    def test_the_401_is_the_floor_talking_not_a_missing_route(self, server_and_counters):
        """With a credential the same paths get THROUGH the floor. Without this,
        a typo'd path would pass the test above forever."""
        server, _, _ = server_and_counters
        client = TestClient(server.app, raise_server_exceptions=False)
        for subpath in ("chat/completions", "uamp/completions", "tasks", "acp"):
            response = client.post(
                f"/{AGENT}/{subpath}",
                json=ANON_BODY,
                headers={"Authorization": "Bearer test-service-token"},
            )
            assert response.status_code not in (401, 404), (subpath, response.status_code)

    def test_anonymous_malformed_body_is_401_not_a_parser_error(self, server_and_counters):
        """The floor runs BEFORE the body is parsed — it is an ASGI middleware,
        so there is no route, no dependency and no body-reading between the
        socket and the refusal. An anonymous caller cannot make the server parse
        arbitrary bytes, and the observable is 401 on both SDKs rather than 401
        on one and a parser error on the other. The TypeScript half asserts the
        same in tests/unit/server/completions-auth.test.ts.
        """
        server, _, _ = server_and_counters
        client = TestClient(server.app, raise_server_exceptions=False)
        for subpath in ("chat/completions", "uamp/completions", "tasks", "acp"):
            response = client.post(
                f"/{AGENT}/{subpath}",
                content=b"{ this is not json",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 401, (subpath, response.status_code, response.text)

    def test_the_floor_does_not_gate_the_routes_meant_to_be_public(self, server_and_counters):
        """The other side of the same coin: an agent's own ``@http`` handlers may
        be public by design, and ``/capabilities`` and ``/models`` come through
        the SAME per-skill mount as ``/uamp/completions``."""
        server, _, _ = server_and_counters
        client = TestClient(server.app)
        for subpath in ("capabilities", "models", "health"):
            response = client.get(f"/{AGENT}/{subpath}")
            assert response.status_code == 200, (subpath, response.text)

    def test_a_get_on_a_billable_looking_path_is_left_alone(self, server_and_counters):
        """Only POST is gated, which is what keeps the suffix match from
        swallowing the info page of an agent that happens to be named ``uamp``
        — and what lets ``GET /tasks/{task_id}`` stay a public status read next
        to a billable ``POST /tasks``."""
        server, _, _ = server_and_counters
        client = TestClient(server.app, raise_server_exceptions=False)
        for path in (f"/{AGENT}/health", f"/{AGENT}/tasks/does-not-exist"):
            response = client.get(path)
            assert response.status_code != 401, (path, response.status_code)


class TestEveryBillableWebSocketRefusesAnonymousHandshakes:
    """The socket surface, discovered from ``get_all_websocket_handlers()``.

    ``UAMPTransportSkill``'s ``@websocket("/uamp")`` was found two rounds ago;
    ``RealtimeTransportSkill``'s ``/realtime`` and ``ACPTransportSkill``'s
    ``/acp/stream`` were not, because nothing walked the registry. All three run
    the owner's model on an accepted anonymous socket. Same money, different
    protocol; same floor, same path set.
    """

    def test_every_registered_billable_socket_is_probed(self, server_and_counters):
        """Guard the sweep itself. If a handler is classified billable it MUST
        end up in the probe list — the failure mode being defended against is a
        discovery step that quietly stops finding things."""
        server, agent, _ = server_and_counters
        _, websockets = discover_registry_handlers(agent)
        expected = {
            f"/{DYNAMIC_AGENT}/{sub}"
            for sub, _src in websockets
            if classify_ws(sub) == "billable"
        }
        assert expected, "no billable WebSocket handler was discovered"
        assert expected <= set(billable_ws_targets(server, agent))
        # Named explicitly: these three are the ones that shipped open.
        assert {
            f"/{DYNAMIC_AGENT}/uamp",
            f"/{DYNAMIC_AGENT}/realtime",
            f"/{DYNAMIC_AGENT}/acp/stream",
        } <= expected

    def test_an_anonymous_billable_websocket_handshake_is_refused(self, server_and_counters):
        server, agent, counters = server_and_counters
        targets = billable_ws_targets(server, agent)
        assert targets, "discovery found no billable sockets — the walk is broken"

        client = TestClient(server.app)
        open_doors = []
        for path in targets:
            try:
                with client.websocket_connect(path):
                    open_doors.append(f"WS {path} -> ACCEPTED")
            except WebSocketDisconnect as exc:
                if exc.code != 4401:
                    open_doors.append(f"WS {path} -> closed {exc.code}, not the floor")

        assert open_doors == [], f"anonymous sockets were not refused: {open_doors}"
        assert counters.is_zero(), (
            f"the model was reached over an anonymous socket: {counters.as_dict()}"
        )

    def test_a_credential_gets_the_handshake_past_the_floor(self, server_and_counters):
        """Guard the guard, and pin the query-parameter form: a browser cannot
        set headers on a WebSocket handshake, so ``?token=`` is accepted exactly
        as the existing upgrade path already reads it."""
        server, _, _ = server_and_counters
        client = TestClient(server.app)
        for suffix, headers in (
            ("", {"Authorization": "Bearer test-service-token"}),
            ("?token=test-service-token", {}),
        ):
            path = f"/{DYNAMIC_AGENT}/uamp{suffix}"
            try:
                with client.websocket_connect(path, headers=headers):
                    pass
            except WebSocketDisconnect as exc:
                # Whatever happens past the floor (the socket handler may close
                # for its own reasons on a bare agent) it must not be OUR
                # refusal — that is what proves the 4401 above is the floor and
                # not a route that was never there.
                assert exc.code != 4401, (path, exc.code)


class TestTheCommandSurfaceStaysNonBillable:
    """``command`` and ``command/{path:path}`` are in the shipped public
    allow-list, and this is what makes that a checked claim rather than a
    remembered one.

    The slash-command surface cannot be classified by path: ONE route,
    ``POST /{agent}/command/{path:path}``, dispatches to every ``@command``
    handler in every skill an agent happens to load. Path classification says
    nothing about what is behind it. So the guarantee is enforced from the other
    end — no shipped ``@command`` handler may reach the model — by walking the
    package's ASTs.

    If this fails, the answer is not to soften it: a command that reaches the
    model is a billable endpoint served by a route the floor lets through
    anonymously, and it needs its own path in ``BILLABLE_PATHS``.
    """

    #: How a handler reaches the provider. ``agent.run(`` rather than ``run(``
    #: because ``asyncio.run`` and friends are everywhere and mean nothing here.
    MODEL_ENTRY_POINTS = ("execute_handoff", "process_uamp", "agent.run(", "run_streaming")

    def _command_functions(self):
        package_root = pathlib.Path(
            __import__("webagents").__file__
        ).resolve().parent
        for source_file in sorted(package_root.rglob("*.py")):
            try:
                source = source_file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (OSError, SyntaxError):  # pragma: no cover - unreadable file
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = []
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    decorators.append(getattr(target, "id", getattr(target, "attr", "")))
                if "command" not in decorators:
                    continue
                yield source_file, node.name, ast.get_source_segment(source, node) or ""

    def test_the_scan_actually_finds_the_command_handlers(self):
        """Guard the guard: a scan that finds nothing would pass forever."""
        found = list(self._command_functions())
        assert len(found) > 50, f"the @command scan found only {len(found)} handlers"

    def test_no_shipped_command_handler_reaches_the_model(self):
        offenders = []
        for source_file, name, body in self._command_functions():
            reached = [needle for needle in self.MODEL_ENTRY_POINTS if needle in body]
            if reached:
                offenders.append(f"{source_file.name}::{name} uses {reached}")
        assert offenders == [], (
            "a @command handler reaches the model, but the command surface is "
            "declared public in PUBLIC_SUBPATHS and is served by a route the "
            "floor does not gate. Give it its own path and add that path to "
            f"BILLABLE_PATHS in both SDKs. Offenders: {offenders}"
        )
