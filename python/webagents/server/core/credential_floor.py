"""The credential floor — ONE predicate, ONE path set, ONE guard, for the
whole Python server.

WHY THIS MODULE EXISTS AT ALL. The floor used to be a per-route ``if`` in the
one branch someone remembered. Four separate doors to the same billable
endpoint were found that way, in three rounds, across two SDKs:

1. ``POST /{agent}/chat/completions`` (the dedicated static route) — guarded.
2. the dynamic-agent catch-all ``dynamic_agent_http_dispatch`` — found later,
   guarded later.
3. the per-skill static mount in ``_register_agent_http_handlers``, which turns
   every ``@http`` handler into its own FastAPI route and therefore went
   through neither of the above.
4. the TypeScript ``/uamp``, ``/uamp/stream`` and every billable route on
   ``WebAgentsServer`` — anonymous 200s, proven with invocation counters.

The recurrence is the finding. A per-route check means every new route
silently reintroduces the hole, and the tests only ever cover the doors
somebody thought of. So the decision — "is this request allowed to reach the
model?" — lives here, is made from the request line alone (method + path, no
body), and is applied at exactly ONE point: an ASGI middleware wrapping the
whole app (:func:`install_credential_floor`), upstream of routing.

That placement is what makes it structural rather than remembered. Doors 1, 2
and 3 are all *routes*; the middleware runs before FastAPI has decided which
one a request belongs to, so all three are the same door from up here, and so
is any route added tomorrow.

WHAT STILL HAS TO BE MAINTAINED BY HAND: :data:`BILLABLE_PATHS`. Nothing can
infer that a NEW path reaches the model. That is what the enumerating test in
``tests/server/test_billable_routes.py`` is for — it walks the real
``app.routes`` table and fails on any mounted route that is neither in this set
nor in an explicit public allow-list, so a new route cannot be added without
someone classifying it.

WHY AN ASGI MIDDLEWARE AND NOT ``@app.middleware("http")`` OR A ROUTER
DEPENDENCY:

* ``@app.middleware("http")`` is ``BaseHTTPMiddleware``, which wraps every
  response in a task group and has a long history of interfering with
  ``StreamingResponse`` — and the endpoint being guarded is the SSE one.
* a router-level ``Depends`` runs AFTER FastAPI has read and parsed the request
  body (``get_request_handler`` reads the body before ``solve_dependencies``),
  which loses the "refuse before parsing arbitrary bytes" property and the
  cross-SDK observable that goes with it.

A bare ASGI middleware touches neither the body nor the response.
"""

from types import SimpleNamespace
from typing import Iterable
from urllib.parse import parse_qs

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


#: Header names a caller can present a credential in.
#:
#: Kept byte-identical to ``CREDENTIAL_HEADERS`` in
#: ``typescript/src/server/credential-floor.ts``; the two SDKs serve the same
#: endpoint and must not disagree about what counts as "authenticated enough to
#: reach the model". ``tests/server/test_floor_parity.py`` reads the TypeScript
#: file and asserts the two lists are equal.
CREDENTIAL_HEADERS = ("authorization", "x-api-key", "x-owner-assertion")

#: Sub-paths that ARE a billable model endpoint, whichever door they are
#: reached through: the dedicated static route, the dynamic catch-all, or a
#: transport skill's ``@http`` handler mounted at the same subpath.
#:
#: ``uamp/completions`` is here because ``CompletionsTransportSkill`` declares
#: ``@http("/uamp/completions", method="post")``, which runs the model through
#: ``agent.process_uamp`` — the same provider, the same owner's credit, just a
#: different wire format.
#:
#: ``uamp`` and ``uamp/stream`` have no Python HTTP route today; they are the
#: TypeScript handler's UAMP endpoints, which call ``agent.processUAMP``.  They
#: are listed here anyway so the two SDKs cannot drift apart on the set, and so
#: that a Python transport skill mounting ``@http("/uamp")`` tomorrow is
#: covered on the day it is written rather than on the day someone notices.
#:
#: ``a2a``, ``tasks`` and ``acp`` were found by the enumerating test once it
#: started walking the handler REGISTRIES instead of the paths the floor
#: already knew:
#:
#: * ``a2a`` — TypeScript ``A2ATransportSkill`` ``@http({path: '/a2a', method:
#:   'POST'})`` (``src/skills/transport/a2a/skill.ts``), a JSON-RPC envelope
#:   whose ``tasks/send`` runs ``agent.processUAMP``. It answered an anonymous
#:   200 with the model reached.
#: * ``tasks`` — Python ``A2ATransportSkill`` ``@http("/tasks", method="post")``
#:   (``skills/core/transport/a2a/skill.py``), which calls
#:   ``agent.process_uamp`` directly. Only the POST is billable; the ``GET
#:   /tasks/{task_id}`` status reads are in :data:`PUBLIC_SUBPATHS` and are not
#:   gated, which is why the floor is POST-only.
#: * ``acp`` — Python ``ACPTransportSkill`` ``@http("/acp", method="post")``,
#:   whose ``session/prompt`` method reaches ``process_uamp``. The ACP spec has
#:   its own ``authenticate`` RPC, but nothing forces a client to call it before
#:   ``session/prompt``, so "it has its own auth story" was never a reason to
#:   leave it open.
#:
#: Kept identical to ``BILLABLE_PATHS`` in
#: ``typescript/src/server/credential-floor.ts`` — asserted by
#: ``tests/server/test_floor_parity.py``.
BILLABLE_PATHS = (
    "chat/completions",
    "v1/chat/completions",
    "uamp",
    "uamp/stream",
    "uamp/completions",
    "a2a",
    "tasks",
    "acp",
)

#: The floor applies to POST only.
#:
#: Every billable path is POST-served, and restricting the method is what keeps
#: the SUFFIX match below from swallowing an unrelated route: ``GET /uamp`` is
#: the agent-info page of an agent that happens to be named ``uamp``, not a
#: model call. An OPTIONS preflight is likewise never billable and must not be
#: refused, or a browser can never reach the endpoint at all.
BILLABLE_METHODS = ("POST",)

#: WebSocket sub-paths that reach the model. ``UAMPTransportSkill`` declares
#: ``@websocket("/uamp")`` with ``scope="all"``, and the WebSocket route in
#: ``app.py`` only runs an auth check when the handler's scope is NOT ``all`` —
#: so the model was reachable over an anonymous socket the whole time the HTTP
#: door next to it was being argued about. Same money, different protocol.
#:
#: ``realtime`` and ``acp/stream`` are the two sockets the previous round left
#: open, and they were left open for a structural reason worth writing down: the
#: enumerating test expanded the billable set THROUGH the WebSocket catch-all
#: instead of walking ``agent.get_all_websocket_handlers()``, so a socket was
#: only ever probed if it was already in this tuple. A tuple that can only
#: confirm itself is a memo, not a test. The test now walks the registry, so a
#: new ``@websocket`` handler fails classification on the day it is written.
#:
#: * ``realtime`` — ``RealtimeTransportSkill.realtime_session`` accepts the
#:   socket with no credential and runs ``execute_handoff`` on
#:   ``response.create``.
#: * ``acp/stream`` — ``ACPTransportSkill.acp_websocket`` accepts, then
#:   dispatches ``session/prompt`` to ``process_uamp``, whether or not the ACP
#:   ``authenticate`` method was ever called.
BILLABLE_WS_PATHS = ("uamp", "realtime", "acp/stream")

#: THE OTHER HALF OF THE CLASSIFICATION. Agent-surface sub-paths that are
#: deliberately anonymous.
#:
#: A path set on its own cannot make a new route fail by default — it can only
#: confirm the routes already in it. What makes the enumerating test a tripwire
#: is that every handler discovered in the agent's registries must land in
#: EXACTLY ONE of two declared sets: :data:`BILLABLE_PATHS` or this one. A
#: handler in neither is a test failure, so the author of the next transport
#: skill has to say which it is before the suite goes green.
#:
#: That is why this list lives in the shipped module next to the billable set
#: rather than in the test file: it is a security declaration, it must not drift
#: between the two SDKs, and ``tests/server/test_floor_parity.py`` asserts it
#: does not. Adding a line here is a decision that a route is safe to serve to
#: anyone who can reach the port. It should feel like one.
#:
#: The reasons, in order:
#:
#: * the empty string — the agent info page at the mount root; static metadata.
#: * capabilities, models, v1/models, info — static descriptions of the agent.
#:   They are also how a client discovers the billable endpoints, so gating them
#:   would break discovery without protecting anything.
#: * health, metrics — liveness and counters, deliberately reachable by a load
#:   balancer that has no credential.
#: * the three .well-known documents — agent card, JWKS and OIDC discovery, all
#:   of which are useless unless they are public.
#: * command and command/{path:path} — the slash-command surface. It dispatches
#:   through agent.execute_command, and no shipped @command handler reaches
#:   execute_handoff, process_uamp or run. A future one that does is billable
#:   and belongs in BILLABLE_PATHS, not here.
#: * tasks/{task_id} and tasks/{task_id}/artifacts — A2A task status reads and a
#:   cancel. They serve results already stored by the billable POST /tasks and
#:   never call the model themselves.
#:
#: Framework chrome that is not agent surface — FastAPI docs, openapi.json,
#: the server-level readiness probes, the Hono agents listing — is allow-listed
#: in the test files instead, because it differs per framework and is not part
#: of the cross-SDK contract this tuple encodes.
#:
#: Kept identical to ``PUBLIC_SUBPATHS`` in
#: ``typescript/src/server/credential-floor.ts``.
PUBLIC_SUBPATHS = (
    "",
    "capabilities",
    "models",
    "v1/models",
    "info",
    "health",
    "metrics",
    ".well-known/agent.json",
    ".well-known/jwks.json",
    ".well-known/openid-configuration",
    "command",
    "command/{path:path}",
    "tasks/{task_id}",
    "tasks/{task_id}/artifacts",
)

#: The WebSocket half of the same declaration, and it is EMPTY on purpose.
#:
#: Every ``@websocket`` handler either of these SDKs ships today reaches the
#: model, so every one of them is in :data:`BILLABLE_WS_PATHS`. This tuple is
#: where a socket that genuinely does not — a presence feed, a log tail — would
#: be declared. It exists so that the classification has somewhere to put such a
#: handler other than silence, which is what ``/realtime`` and ``/acp/stream``
#: got for two rounds.
#:
#: Note that it is deliberately SEPARATE from :data:`PUBLIC_SUBPATHS`: an HTTP
#: path being safe says nothing about a socket at the same name. The reviewer
#: probe that proved the old test blind was ``@websocket``/live, and ``live`` is
#: exactly the kind of name that is an innocuous HTTP probe.
#:
#: Kept identical to ``PUBLIC_WS_SUBPATHS`` in
#: ``typescript/src/server/credential-floor.ts``.
PUBLIC_WS_SUBPATHS = ()

#: Query parameters a WebSocket client can present a credential in. A browser
#: cannot set headers on a WebSocket handshake, which is why the existing
#: upgrade path already reads ``?token=``; the floor accepts the same thing
#: rather than locking out the callers the server documents.
WS_CREDENTIAL_QUERY_PARAMS = ("token", "access_token", "api_key")

UNAUTHORIZED_MESSAGE = (
    "Authentication required: send the platform service token or "
    "an api key in the Authorization header."
)


def has_credential(request) -> bool:
    """True when the request carries SOMETHING that can be authenticated.

    This is a FLOOR, not the authentication itself: ``AuthSkill`` (when the
    agent has one) verifies the credential inside the run's ``on_connection``
    hook and raises when it does not check out. The floor exists so that an
    agent with no AuthSkill — the quickstart shape — is not an anonymous,
    BILLABLE model endpoint for anyone who can reach the port.

    Accepts anything with a ``.headers`` mapping: a Starlette ``Request``, a
    ``WebSocket``, or a bare header dict. A lone ``Bearer`` with nothing after
    it is not a credential.
    """
    headers = getattr(request, "headers", request)
    for name in CREDENTIAL_HEADERS:
        try:
            value = headers.get(name)
        except AttributeError:
            return False
        if value and value.strip() and value.strip().lower() != "bearer":
            return True
    return False


def _normalize(path: str) -> str:
    return path.strip("/")


def _matches_suffix(path: str, candidates: Iterable[str]) -> bool:
    normalized = _normalize(path)
    for candidate in candidates:
        if normalized == candidate or normalized.endswith("/" + candidate):
            return True
    return False


def is_billable_path(path: str) -> bool:
    """True when ``path`` ends in one of :data:`BILLABLE_PATHS`.

    Suffix matching rather than equality because the same subpath is mounted
    under every prefix the server supports and the floor must not care which:
    ``/og/chat/completions``, ``/agents/og/chat/completions`` (a ``url_prefix``
    server), ``/og/uamp/completions`` from the per-skill mount. Matching the
    tail is what makes ONE check cover all of them.
    """
    return _matches_suffix(path, BILLABLE_PATHS)


def is_billable_ws_path(path: str) -> bool:
    """True when ``path`` ends in a WebSocket sub-path that reaches the model."""
    return _matches_suffix(path, BILLABLE_WS_PATHS)


def is_billable_request(method: str, path: str) -> bool:
    """The whole floor decision, from the request line alone. No body is read."""
    return method.upper() in BILLABLE_METHODS and is_billable_path(path)


def unauthorized_response() -> JSONResponse:
    """The 401 body. ``{"detail": ...}`` because that is what ``HTTPException``
    produced from the per-route checks this replaced — the observable does not
    change just because the enforcement point moved."""
    return JSONResponse(status_code=401, content={"detail": UNAUTHORIZED_MESSAGE})


def websocket_upgrade_is_refused(scope: Scope) -> bool:
    """True when a WebSocket handshake must be refused by the floor.

    Reads the raw ASGI scope rather than wrapping it in a Starlette object: a
    ``Request`` asserts ``scope["type"] == "http"``, and a ``WebSocket`` would
    pull in a receive channel this decision has no business touching.
    """
    path = scope.get("path", "")
    if not is_billable_ws_path(path):
        return False

    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    if has_credential(SimpleNamespace(headers=headers)):
        return False

    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    for param in WS_CREDENTIAL_QUERY_PARAMS:
        for value in query.get(param, []):
            if value and value.strip():
                return False
    return True


class CredentialFloorMiddleware:
    """THE chokepoint. Pure ASGI: reads the request line and the headers, never
    the body, never the response.

    Refuses an anonymous request to a billable path with 401 before FastAPI has
    resolved a route, which is why it covers the dedicated route, the dynamic
    catch-all, the per-skill ``@http`` mount and anything added later in one
    place. A billable WebSocket handshake with no credential is closed at the
    handshake rather than after the socket is live.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope.get("type")

        if scope_type == "http":
            if is_billable_request(scope.get("method", ""), scope.get("path", "")):
                request = Request(scope, receive)
                if not has_credential(request):
                    response = unauthorized_response()
                    await response(scope, receive, send)
                    return

        elif scope_type == "websocket":
            if websocket_upgrade_is_refused(scope):
                # Refuse the handshake: consume the client's `websocket.connect`
                # (the protocol requires it before anything may be sent) and
                # close WITHOUT accepting, so the socket is never established.
                # 4401 is the conventional WebSocket analogue of HTTP 401 — a
                # close code is the only way a client learns WHY on a socket.
                await receive()
                await send(
                    {
                        "type": "websocket.close",
                        "code": 4401,
                        "reason": "Authentication required",
                    }
                )
                return

        await self.app(scope, receive, send)


def install_credential_floor(app, *, _installed_attr: str = "_webagents_credential_floor") -> None:
    """Wrap ``app`` in the floor. Idempotent, so a server that is built twice
    (or a test that reuses an app) does not stack the middleware."""
    if getattr(app, _installed_attr, False):
        return
    app.add_middleware(CredentialFloorMiddleware)
    setattr(app, _installed_attr, True)


__all__ = [
    "BILLABLE_METHODS",
    "BILLABLE_PATHS",
    "BILLABLE_WS_PATHS",
    "CREDENTIAL_HEADERS",
    "CredentialFloorMiddleware",
    "PUBLIC_SUBPATHS",
    "PUBLIC_WS_SUBPATHS",
    "UNAUTHORIZED_MESSAGE",
    "WS_CREDENTIAL_QUERY_PARAMS",
    "has_credential",
    "install_credential_floor",
    "is_billable_path",
    "is_billable_request",
    "is_billable_ws_path",
    "unauthorized_response",
    "websocket_upgrade_is_refused",
]
