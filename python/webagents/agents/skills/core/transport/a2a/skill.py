"""
A2A v1.0 transport (a2aproject/A2A v1.0.1), the Python half of the TypeScript
`A2ATransportSkill` (`typescript/src/skills/transport/a2a/skill.ts`): the same
shape in both SDKs, pinned by one fixture both suites replay
(`tests/fixtures/a2a/vectors.json`, `config_shapes.json`). Plan item 1.3,
2026-09-26.

WHAT IT SERVES, under the agent's prefix:
  * `POST /a2a`: JSON-RPC, the v1.0 PascalCase methods and the v0.3 dotted
    aliases (`protocol.py` has the table and the version rules that keep
    OpenClaw, which sends no `A2A-Version`, and Hermes, which sends 1.0,
    both talking to us);
  * the HTTP+JSON binding at the same prefix: `POST /a2a/message:send`,
    `POST /a2a/message:stream`, `GET /a2a/tasks`, `GET /a2a/tasks/{id}`,
    `POST /a2a/tasks/{id}:cancel`, `GET|POST /a2a/tasks/{id}:subscribe`
    (the spec's prose says POST for subscribe and its proto says GET, so
    both are served);
  * `GET /.well-known/agent-card.json`: the v1.0 card, signed with the
    agent's Ed25519 identity when the server gave it one
    (`agent.signing_identity`, see `card.py`). The registration card at
    `/.well-known/agent.json` is the server's (`registration.py`) and is not
    touched: the platform reads it and refuses one that does not name itself.

THE RUN IS THE AGENT'S NORMAL RUN, UNDER THE CALLER'S VERIFIED IDENTITY.
Every request is identified the way a scoped endpoint is (`endpoint_gate.py`:
`BaseAgent.identify_caller`, the auth skills and the access block, on the
request's own context), refused with the gate's own 401/403 when that fails,
and the send then goes through `agent.run` / `agent.run_streaming` on that
same context, so the run's hooks read the same request headers
`/chat/completions` gives them. `access:` applies, payment applies, and an
A2A caller can do nothing a chat caller could not. `contextId` is the
conversation; parts become content items; the output becomes the task's
status message and one artifact.

TASKS ARE PER CALLER (`tasks.py`): a caller reads, lists, subscribes to and
cancels only what it created, and another caller's task id is not found,
never forbidden. `returnImmediately` answers the task WORKING at once and
keeps working; the default blocks up to `blocking_timeout_seconds` and
answers whatever state the task is in by then, which is what OpenClaw's 120 s
poll and Hermes' 120 s wait expect. Push notifications and the extended card
are declined with the spec's own codes.

HOW THE HANDLERS READ THE REQUEST. Both server doors (the static mount and
the dynamic route in `server/core/app.py`) parse a JSON body and hand a
handler the fields its signature names. A JSON-RPC envelope and a REST body
are read whole instead, from the request on the current context: the body
bytes Starlette cached when the door parsed them, the headers (the
`A2A-Version`, the credential the caller key hashes) and the query. Each
handler returns a Starlette `Response`, which both doors pass through as it
is; a path parameter (`{id}`) is the one thing taken from the door.

WHAT WENT AWAY (0.4.0): the 0.2.1 REST binding (`POST /tasks` with its own
SSE event names, `GET /tasks/{task_id}`, `DELETE /tasks/{task_id}`,
`GET /tasks/{task_id}/artifacts`), its UAMP adapter, and the
`/.well-known/agent.json` handler this skill mounted. The latter is the one
to know about: the static mount registers a skill's route beside the
server's, so this skill's old card SHADOWED the self-naming registration
card on any agent that attached it, and that agent could not register.
Nothing else spoke the 0.2.1 shapes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from starlette.responses import JSONResponse, Response, StreamingResponse

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http
from webagents.server.context.context_vars import create_context, get_context, set_context
from webagents.server.core import endpoint_gate
from webagents.server.core.error_reply import reply_text as safe_reply_text

from .a2a_client import call_agent
from .card import (
    A2A_RPC_SUBPATH,
    AGENT_CARD_WELL_KNOWN_SUFFIX,
    build_a2a_agent_card,
    identity_card_signer,
    sign_agent_card,
    skills_from_tools,
)
from .protocol import (
    caller_key,
    check_version,
    message_to_run_message,
    now_iso,
    output_to_parts,
    read_configuration,
    read_message,
    requested_version,
    resolve_method,
)
from .tasks import TaskRecord, TaskStore
from .types import A2AError, is_settled, task_view

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

__all__ = [
    "A2ATransportSkill",
    "A2A_RPC_SUBPATH",
    "AGENT_CARD_WELL_KNOWN_SUFFIX",
    "DEFAULT_BLOCKING_TIMEOUT_SECONDS",
    "DEFAULT_TASK_TTL_SECONDS",
    "peer_token_for",
    "resolve_a2a_settings",
]

# ---------------------------------------------------------------------------
# Configuration: the agent file's `a2a` entry, the same keys as TypeScript
# (`tests/fixtures/a2a/config_shapes.json`)
# ---------------------------------------------------------------------------

DEFAULT_TASK_TTL_SECONDS = 3600
DEFAULT_BLOCKING_TIMEOUT_SECONDS = 60


def _positive_number(*candidates: Any) -> Optional[float]:
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if value > 0 and value != float("inf"):
                return value
            continue
        if isinstance(value, str) and value.strip():
            try:
                number = float(value)
            except ValueError:
                continue
            if number > 0 and number != float("inf"):
                return int(number) if number.is_integer() else number
    return None


def _optional_string(*candidates: Any) -> Optional[str]:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return None


def resolve_a2a_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The `a2a` entry's settings, read back as `skill.settings`; unknown keys
    and the loaders' own (`agent_name`, `agent_path`) are ignored. The
    TypeScript `resolveA2ASettings`."""
    config = config or {}
    peers: Dict[str, Dict[str, str]] = {}
    raw_peers = config.get("peers")
    if isinstance(raw_peers, dict):
        for url, entry in raw_peers.items():
            if not isinstance(url, str) or not url:
                continue
            token = entry.get("token") if isinstance(entry, dict) else None
            peers[url] = {"token": token} if isinstance(token, str) and token else {}
    provider = config.get("provider")
    valid_provider = (
        {"organization": provider["organization"], "url": provider["url"]}
        if isinstance(provider, dict)
        and isinstance(provider.get("organization"), str)
        and isinstance(provider.get("url"), str)
        else None
    )
    return {
        "task_ttl_seconds": _positive_number(config.get("task_ttl_seconds"), config.get("taskTtlSeconds"))
        or DEFAULT_TASK_TTL_SECONDS,
        "blocking_timeout_seconds": _positive_number(
            config.get("blocking_timeout_seconds"), config.get("blockingTimeoutSeconds")
        )
        or DEFAULT_BLOCKING_TIMEOUT_SECONDS,
        "peers": peers,
        "version": _optional_string(config.get("version")) or "1.0.0",
        "provider": valid_provider,
        "documentation_url": _optional_string(config.get("documentation_url"), config.get("documentationUrl")),
        "icon_url": _optional_string(config.get("icon_url"), config.get("iconUrl")),
        # Carry this agent's signed TrustFlow record in the card (plan item
        # 2.7): fetched from the platform as this agent, as the extension
        # `trustflow.trust_record` defines, and covered by the card signature.
        "trust_record": config.get("trust_record") is True or config.get("trustRecord") is True,
    }


def peer_token_for(url: str, peers: Dict[str, Dict[str, str]]) -> Optional[str]:
    """The bearer for `url`: the token of the longest configured peer that is
    the URL itself or a path prefix of it (`https://peer.example` matches
    `https://peer.example/agents/x/a2a` and not `https://peer.example.evil`)."""
    target = (url or "").rstrip("/")
    best: Optional[Tuple[int, str]] = None
    for peer, entry in (peers or {}).items():
        prefix = (peer or "").rstrip("/")
        token = entry.get("token") if isinstance(entry, dict) else None
        if not prefix or not token:
            continue
        if target != prefix and not target.startswith(prefix + "/"):
            continue
        if best is None or len(prefix) > best[0]:
            best = (len(prefix), token)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# The skill
# ---------------------------------------------------------------------------

JSON_RPC = "2.0"
REST_CONTENT_TYPE = "application/a2a+json"
CARD_CACHE_CONTROL = "public, max-age=300"
_VERBS = ("cancel", "subscribe")

#: `(owner, context)` for an admitted request: the task-store key and the
#: request context naming the caller.
Admitted = Tuple[str, Any]


class A2ATransportSkill(Skill):
    """The A2A v1.0 transport: the JSON-RPC and HTTP+JSON bindings and the v1.0 card."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        super().__init__(config, scope="all")
        self.settings = resolve_a2a_settings(config)
        self._store = TaskStore(float(self.settings["task_ttl_seconds"]))

    async def initialize(self, agent: "BaseAgent") -> None:
        self.agent = agent

    def _agent(self) -> Any:
        """The served agent: the one `initialize` gave this skill, else the
        one on the request's context (a static agent's skills are initialised
        on its first run, which may come after its first A2A request)."""
        if self.agent is not None:
            return self.agent
        context = get_context()
        return getattr(context, "agent", None) if context is not None else None

    def peer_token_for(self, url: str) -> Optional[str]:
        """The bearer a client sends `url`, from the configured peers."""
        return peer_token_for(url, self.settings["peers"])

    async def call_peer(self, url: str, message_or_text: Union[str, Dict[str, Any]], **options: Any) -> Dict[str, Any]:
        """Call a peer over A2A v1.0 as this agent (`a2a_client.py`: its card,
        the first JSON-RPC 1.0 interface, one `SendMessage`, the task polled
        to a settled state), with the bearer the agent file configures for it
        (`peers`); `token=` overrides it."""
        token = options.pop("token", None) or self.peer_token_for(url)
        return await call_agent(url, message_or_text, token=token, **options)

    @property
    def task_count(self) -> int:
        """How many tasks the store holds (tests)."""
        return len(self._store)

    async def cleanup(self) -> None:
        self._store.clear()

    # =======================================================================
    # The card
    # =======================================================================

    @http(AGENT_CARD_WELL_KNOWN_SUFFIX, method="get")
    async def a2a_agent_card(self) -> Response:
        """The A2A v1.0 agent card, beside the registration card."""
        agent = self._agent()
        if agent is None:
            return JSONResponse(status_code=503, content={"error": {"code": "not_ready", "message": "No agent attached"}})
        request = _current_request()
        card = await self.build_card(agent, request)
        body = json.dumps(card, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
        headers = {"Cache-Control": CARD_CACHE_CONTROL, "ETag": etag}
        if _header(request, "if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(content=body, media_type="application/json", headers=headers)

    async def build_card(self, agent: Any, request: Any = None) -> Dict[str, Any]:
        """The v1.0 card for `agent`, signed when the agent holds an identity."""
        principal = self.principal_for(agent, request)
        card = build_a2a_agent_card(
            agent.name,
            getattr(agent, "description", None) or "",
            principal=principal,
            skills=skills_from_tools(self.public_tools(agent)),
            version=self.settings["version"],
            provider=self.settings["provider"],
            documentation_url=self.settings["documentation_url"],
            icon_url=self.settings["icon_url"],
        )
        if self.settings["trust_record"]:
            record = await self.own_trust_record(agent, principal)
            if record:
                from webagents.trustflow.trust_record import with_trust_record_extension

                card = with_trust_record_extension(card, record)
        identity = getattr(agent, "signing_identity", None)
        if identity is not None and callable(getattr(identity, "held_keys", None)):
            try:
                return sign_agent_card(card, identity_card_signer(identity))
            except Exception:  # noqa: BLE001 - an identity without a key signs nothing; the card is still served
                pass
        return card

    #: The platform lookup the record comes from (`trustflow.trust_lookup.TrustLookup`); a test sets a stub.
    trust_lookup: Any = None
    _held_trust_record: Optional[Dict[str, Any]] = None
    _trust_record_retry_at: float = 0.0

    async def own_trust_record(self, agent: Any, principal: str) -> Optional[str]:
        """This agent's signed TrustFlow record from the platform (plan item
        2.7), held until an hour before it expires and at most a day, so a
        fresh one follows the batch; when the platform cannot answer the card
        is served without it and the ask is repeated a minute later, not per
        request."""
        import time

        now = time.time()
        held = self._held_trust_record
        if held is not None and held["expires"] > now:
            return held["record"]
        if self._trust_record_retry_at > now:
            return None
        try:
            if self.trust_lookup is None:
                from webagents.trustflow.trust_lookup import TrustLookup, platform_credential_for

                def credential():
                    return platform_credential_for(
                        getattr(agent, "signing_identity", None),
                        os.environ.get("WEBAGENTS_AGENT_TOKEN") or os.environ.get("WEBAGENTS_API_KEY"),
                    )

                self.trust_lookup = TrustLookup(credential=credential)
            answer = await self.trust_lookup.record(principal)
            record = answer["record"]
            exp = answer.get("payload", {}).get("exp")
            expires_at = float(exp) if isinstance(exp, (int, float)) and not isinstance(exp, bool) else now + 3600
            self._held_trust_record = {"record": record, "expires": min(expires_at - 3600, now + 86400)}
            return record
        except Exception:  # noqa: BLE001 - the card is served without the record; retried in a minute
            self._trust_record_retry_at = now + 60
            return None

    def principal_for(self, agent: Any, request: Any = None) -> str:
        """The agent URL the card's interfaces are built on: the identity's
        issuer (what `WebAgentsServer` composes from its public URL, the
        prefix and the agent name), else the configured or environment public
        URL plus the served prefix, else the request's own origin plus that
        prefix. The last is a Host-header guess the registration card refuses
        to make; it is made here because this card's readers (OpenClaw,
        Hermes) post to the interface URL as written and cannot resolve a
        relative one, and the card was fetched from exactly that origin a
        moment ago."""
        issuer = getattr(getattr(agent, "signing_identity", None), "issuer", None)
        if isinstance(issuer, str) and issuer:
            return issuer.rstrip("/")
        origin = ""
        base_path = ""
        url = getattr(request, "url", None) if request is not None else None
        if url is not None:
            scheme = getattr(url, "scheme", "") or ""
            netloc = getattr(url, "netloc", "") or ""
            if scheme and netloc:
                origin = f"{scheme}://{netloc}"
            path = getattr(url, "path", "") or ""
            base_path = (
                path[: -len(AGENT_CARD_WELL_KNOWN_SUFFIX)] if path.endswith(AGENT_CARD_WELL_KNOWN_SUFFIX) else path
            )
        configured = _optional_string(
            self.config.get("public_url"), self.config.get("publicUrl"), os.environ.get("WEBAGENTS_PUBLIC_URL")
        )
        base = (configured or origin).rstrip("/")
        return f"{base}{base_path}".rstrip("/") or base_path or "/"

    def public_tools(self, agent: Any) -> List[Dict[str, Any]]:
        """The tools a default-group caller may use, never owner-only ones:
        the registry filtered for an anonymous caller in the access block's
        default group (when the block names one), the way an MCP listing is
        scoped."""
        scopes: List[str] = []
        skills = getattr(agent, "skills", None) or {}
        for skill in skills.values() if isinstance(skills, dict) else skills:
            default = getattr(getattr(skill, "policy", None), "default", None)
            if isinstance(default, str) and default:
                scopes.append(f"group:{default}")
                break
        lister = getattr(agent, "get_tools_for_scopes", None)
        if callable(lister):
            try:
                return list(lister(scopes))
            except Exception:  # noqa: BLE001 - fall through to the unscoped listing
                pass
        return list(getattr(agent, "get_all_tools", lambda: [])())

    # =======================================================================
    # JSON-RPC
    # =======================================================================

    @http(A2A_RPC_SUBPATH, method="post")
    async def a2a_jsonrpc(self) -> Response:
        """A2A v1.0 JSON-RPC: the PascalCase methods and the v0.3 dotted aliases."""
        request = _current_request()
        raw = await _raw_body(request)
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            return _rpc_response(None, error=A2AError("PARSE_ERROR").json_rpc())
        if not isinstance(body, dict):
            return _rpc_response(None, error=A2AError("INVALID_REQUEST", "Request must be a JSON object").json_rpc())
        raw_id = body.get("id")
        rpc_id = raw_id if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else None
        method = body.get("method")
        if not isinstance(method, str) or not method:
            return _rpc_response(rpc_id, error=A2AError("INVALID_REQUEST", "Request has no method").json_rpc())
        resolved = resolve_method(method)
        if resolved is None:
            return _rpc_response(rpc_id, error=A2AError("METHOD_NOT_FOUND", f"Method not found: {method}").json_rpc())
        try:
            check_version(requested_version(_headers(request), _query(request)))
        except A2AError as error:
            return _rpc_response(rpc_id, error=error.json_rpc())
        op, _dotted = resolved
        if op == "push_config":
            return _rpc_response(rpc_id, error=A2AError("PUSH_NOTIFICATION_NOT_SUPPORTED").json_rpc())
        if op == "extended_card":
            return _rpc_response(rpc_id, error=A2AError("EXTENDED_AGENT_CARD_NOT_CONFIGURED").json_rpc())

        admitted, refusal = await self._admit(request)
        if refusal is not None or admitted is None:
            return refusal if refusal is not None else _rpc_response(rpc_id, error=A2AError("INTERNAL_ERROR").json_rpc())
        params = body.get("params") if isinstance(body.get("params"), dict) else {}

        def wrap(event: Dict[str, Any]) -> Dict[str, Any]:
            return {"jsonrpc": JSON_RPC, "id": rpc_id, "result": event}

        try:
            if op == "send":
                return _rpc_response(rpc_id, result={"task": await self._send(admitted, params)})
            if op == "stream":
                record = await self._start_send(admitted, params, streaming=True)
                return _sse_response(self._events(record), wrap)
            if op == "get":
                record = self._require_task(admitted[0], params.get("id"))
                view = task_view(record.task, history_length=_history_length(params.get("historyLength")))
                return _rpc_response(rpc_id, result={"task": view})
            if op == "list":
                return _rpc_response(rpc_id, result=self._store.list(admitted[0], **_list_filter(params)))
            if op == "cancel":
                record = self._require_task(admitted[0], params.get("id"))
                return _rpc_response(rpc_id, result={"task": self._cancel(record)})
            if op == "subscribe":
                record = self._require_task(admitted[0], params.get("id"))
                return _sse_response(self._events(record), wrap)
            return _rpc_response(rpc_id, error=A2AError("UNSUPPORTED_OPERATION").json_rpc())
        except A2AError as error:
            return _rpc_response(rpc_id, error=error.json_rpc())

    # =======================================================================
    # HTTP+JSON
    # =======================================================================

    @http(f"{A2A_RPC_SUBPATH}/message:send", method="post")
    async def a2a_rest_send(self) -> Response:
        """HTTP+JSON `SendMessage`."""
        return await self._rest("send")

    @http(f"{A2A_RPC_SUBPATH}/message:stream", method="post")
    async def a2a_rest_stream(self) -> Response:
        """HTTP+JSON `SendStreamingMessage` (SSE)."""
        return await self._rest("stream")

    @http(f"{A2A_RPC_SUBPATH}/tasks", method="get")
    async def a2a_rest_list(self) -> Response:
        """HTTP+JSON `ListTasks`: the caller's own tasks."""
        return await self._rest("list")

    @http(f"{A2A_RPC_SUBPATH}/tasks/{{id}}", method="get")
    async def a2a_rest_get(self, id: str = "") -> Response:
        """HTTP+JSON `GetTask`; `{id}` is one segment, so this route also
        matches `{id}:subscribe` when it is the one a door dispatched to, and
        the suffix decides."""
        task_id, verb = _split_verb(id)
        return await self._rest("subscribe" if verb == "subscribe" else "get", task_id)

    @http(f"{A2A_RPC_SUBPATH}/tasks/{{id}}:cancel", method="post")
    async def a2a_rest_cancel(self, id: str = "") -> Response:
        """HTTP+JSON `CancelTask`."""
        return await self._rest("cancel", _split_verb(id)[0])

    @http(f"{A2A_RPC_SUBPATH}/tasks/{{id}}:subscribe", method="post")
    async def a2a_rest_subscribe(self, id: str = "") -> Response:
        """HTTP+JSON `SubscribeToTask` (the spec's prose verb)."""
        return await self._rest("subscribe", _split_verb(id)[0])

    @http(f"{A2A_RPC_SUBPATH}/tasks/{{id}}:subscribe", method="get")
    async def a2a_rest_subscribe_get(self, id: str = "") -> Response:
        """HTTP+JSON `SubscribeToTask` (the proto's verb)."""
        return await self._rest("subscribe", _split_verb(id)[0])

    async def _rest(self, op: str, task_id: Optional[str] = None) -> Response:
        request = _current_request()
        raw = await _raw_body(request)
        try:
            check_version(requested_version(_headers(request), _query(request)))
        except A2AError as error:
            return _rest_error(error)
        admitted, refusal = await self._admit(request)
        if refusal is not None or admitted is None:
            return refusal if refusal is not None else _rest_error(A2AError("INTERNAL_ERROR"))
        owner = admitted[0]
        try:
            if op == "send":
                return _rest_json({"task": await self._send(admitted, _parse_rest_body(raw))})
            if op == "stream":
                record = await self._start_send(admitted, _parse_rest_body(raw), streaming=True)
                return _sse_response(self._events(record), lambda event: event)
            if op == "get":
                record = self._require_task(owner, task_id)
                history = _history_length(_query(request).get("historyLength"))
                return _rest_json(task_view(record.task, history_length=history))
            if op == "list":
                return _rest_json(self._store.list(owner, **_list_filter(dict(_query(request).items()))))
            if op == "cancel":
                record = self._require_task(owner, task_id)
                return _rest_json(self._cancel(record))
            if op == "subscribe":
                record = self._require_task(owner, task_id)
                return _sse_response(self._events(record), lambda event: event)
            return _rest_error(A2AError("UNSUPPORTED_OPERATION"))
        except A2AError as error:
            return _rest_error(error)

    # =======================================================================
    # Who is calling
    # =======================================================================

    async def _admit(self, request: Any) -> Tuple[Optional[Admitted], Optional[Response]]:
        """Identify the caller the way a scoped endpoint does
        (`endpoint_gate.py`), or answer the gate's own refusal. The owner key
        for the task store comes from the result."""
        agent = self._agent()
        context = get_context()
        if context is None:
            context = create_context(messages=[], stream=False, agent=agent, request=request)
            set_context(context)
        # Anonymous until an identity skill says otherwise, as the TypeScript
        # `identificationContext` makes it.
        context.auth = None
        if agent is not None and hasattr(agent, "identify_caller"):
            try:
                identified = await agent.identify_caller(context)
            except Exception as error:  # noqa: BLE001 - only a refusal is answered here
                refusal = endpoint_gate.refusal_of(error)
                if refusal is None:
                    raise
                # A 401 carries the bearer challenge (2026-09-29, `refusal_response`).
                return None, endpoint_gate.refusal_response(request, refusal)
            if identified is not None:
                context = identified
                set_context(context)
        return (caller_key(getattr(context, "auth", None), _headers(request)), context), None

    # =======================================================================
    # Tasks
    # =======================================================================

    def _require_task(self, owner: str, task_id: Any) -> TaskRecord:
        if not isinstance(task_id, str) or not task_id:
            raise A2AError("INVALID_PARAMS", "A task id is required")
        record = self._store.get(owner, task_id)
        if record is None:
            raise A2AError("TASK_NOT_FOUND", f"Task not found: {task_id}", {"taskId": task_id})
        return record

    async def _send(self, admitted: Admitted, params: Dict[str, Any]) -> Dict[str, Any]:
        """A blocking or immediate send: the task as it stands when the answer goes out."""
        configuration = read_configuration(params.get("configuration"))
        record = await self._start_send(admitted, params, streaming=False, configuration=configuration)
        if not configuration["return_immediately"] and not is_settled(record.task["status"]["state"]):
            try:
                await asyncio.wait_for(record.settled.wait(), timeout=float(self.settings["blocking_timeout_seconds"]))
            except asyncio.TimeoutError:
                pass
        return task_view(record.task, history_length=configuration["history_length"])

    async def _start_send(
        self,
        admitted: Admitted,
        params: Dict[str, Any],
        *,
        streaming: bool,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        """Create the task, record its first event and start the run."""
        agent = self._agent()
        if agent is None:
            raise A2AError("INTERNAL_ERROR", "No agent attached")
        if configuration is None:
            # Push delivery is refused before any task exists.
            read_configuration(params.get("configuration"))
        message = read_message(params.get("message"))
        owner, context = admitted
        if message.get("taskId"):
            existing = self._store.get(owner, message["taskId"])
            if existing is None:
                raise A2AError("TASK_NOT_FOUND", f"Task not found: {message['taskId']}", {"taskId": message["taskId"]})
            raise A2AError(
                "INVALID_PARAMS",
                "That task is finished; send a new message with the same contextId to continue the conversation"
                if is_settled(existing.task["status"]["state"])
                else "That task is still running",
            )
        context_id = message.get("contextId") or str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        inbound = {**message, "contextId": context_id, "taskId": task_id}
        task: Dict[str, Any] = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "TASK_STATE_SUBMITTED", "timestamp": now_iso()},
            "artifacts": [],
            "history": [inbound],
        }
        record = self._store.create(owner, task)
        self._store.emit(record, {"task": task_view(task)})
        self._store.set_status(record, {"state": "TASK_STATE_WORKING", "timestamp": now_iso()})
        record.runner = asyncio.create_task(self._execute(agent, record, context, streaming))
        return record

    async def _execute(self, agent: Any, record: TaskRecord, context: Any, streaming: bool) -> None:
        """The run, under the caller's identity; the task's terminal state is
        its outcome. A streaming send runs `run_streaming` and records each
        delta as an artifact chunk; a blocking one runs `run` and records the
        whole artifact at once, media included."""
        set_context(context)
        messages = [message_to_run_message(m) for m in record.task["history"]]
        artifact_id = str(uuid.uuid4())
        try:
            if streaming:
                text = ""
                first = True
                async for chunk in agent.run_streaming(messages):
                    delta = _delta_text(chunk)
                    if delta:
                        text += delta
                        self._store.add_artifact_chunk(
                            record,
                            {"artifactId": artifact_id, "name": "response", "parts": [{"text": delta}]},
                            append=not first,
                            last_chunk=False,
                        )
                        first = False
                parts = output_to_parts(text)
                artifact = next((a for a in record.task["artifacts"] if a["artifactId"] == artifact_id), None)
                if artifact is not None:
                    artifact["parts"] = parts
                else:
                    record.task["artifacts"].append({"artifactId": artifact_id, "name": "response", "parts": parts})
                self._complete(record, parts)
                return
            result = await agent.run(messages, stream=False)
            parts = output_to_parts(_completion_content(result))
            self._store.add_artifact_chunk(
                record, {"artifactId": artifact_id, "name": "response", "parts": parts}, append=False, last_chunk=True
            )
            self._complete(record, parts)
        except asyncio.CancelledError:
            # `_cancel` set the state before cancelling the run.
            raise
        except Exception as error:  # noqa: BLE001 - the task's terminal state is the outcome
            if is_settled(record.task["status"]["state"]):
                return
            status = getattr(error, "status_code", None)
            http_status = status if isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 600 else None
            code = getattr(error, "error_code", None) if http_status in (401, 403) else None
            if not isinstance(code, str) or not code:
                code = "unauthorized" if http_status == 401 else "forbidden" if http_status == 403 else "run_failed"
            # The fixed text unless the error was written to be shown (S-228).
            text = safe_reply_text(error, where=f"{agent.name} a2a")
            reply = {
                "messageId": str(uuid.uuid4()),
                "contextId": record.task["contextId"],
                "taskId": record.task["id"],
                "role": "ROLE_AGENT",
                "parts": [{"text": text}],
            }
            detail: Dict[str, Any] = {"code": code, "message": text}
            if http_status:
                detail["httpStatus"] = http_status
            self._store.set_status(
                record, {"state": "TASK_STATE_FAILED", "message": reply, "timestamp": now_iso()}, {"error": detail}
            )

    def _complete(self, record: TaskRecord, parts: List[Dict[str, Any]]) -> None:
        reply = {
            "messageId": str(uuid.uuid4()),
            "contextId": record.task["contextId"],
            "taskId": record.task["id"],
            "role": "ROLE_AGENT",
            "parts": parts,
        }
        self._store.add_history(record, reply)
        self._store.set_status(record, {"state": "TASK_STATE_COMPLETED", "message": reply, "timestamp": now_iso()})

    def _cancel(self, record: TaskRecord) -> Dict[str, Any]:
        state = record.task["status"]["state"]
        if is_settled(state):
            raise A2AError("TASK_NOT_CANCELABLE", f"Task {record.task['id']} is {state}", {"taskId": record.task["id"], "state": state})
        runner = record.runner
        self._store.set_status(record, {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()})
        if runner is not None and not runner.done():
            runner.cancel()
        return task_view(record.task)

    async def _events(self, record: TaskRecord) -> AsyncGenerator[Dict[str, Any], None]:
        """Every event of `record` so far and then live, until a settled state."""
        index = 0
        while True:
            while index < len(record.events):
                event = record.events[index]
                index += 1
                yield event
                if _ends_stream(event):
                    return
            if is_settled(record.task["status"]["state"]):
                return
            await record.changed.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_request() -> Any:
    context = get_context()
    return getattr(context, "request", None) if context is not None else None


async def _raw_body(request: Any) -> bytes:
    """The request bytes: what Starlette cached when the door parsed the body."""
    reader = getattr(request, "body", None) if request is not None else None
    if not callable(reader):
        return b""
    try:
        data = await reader()
    except Exception:  # noqa: BLE001 - a stream that cannot be read again reads as empty
        return b""
    return bytes(data) if isinstance(data, (bytes, bytearray)) else b""


def _headers(request: Any) -> Any:
    headers = getattr(request, "headers", None) if request is not None else None
    return headers if headers is not None else {}


def _header(request: Any, name: str) -> Optional[str]:
    value = _headers(request).get(name)
    return value if isinstance(value, str) else None


def _query(request: Any) -> Any:
    params = getattr(request, "query_params", None) if request is not None else None
    return params if params is not None else {}


def _delta_text(chunk: Any) -> str:
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return ""
    content = (choices[0].get("delta") or {}).get("content")
    return content if isinstance(content, str) else ""


def _completion_content(result: Any) -> Any:
    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") if isinstance(message, dict) else ""


def _history_length(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return int(number) if number >= 0 else None
    return None


def _list_filter(params: Dict[str, Any]) -> Dict[str, Any]:
    """`TaskStore.list` keywords from JSON-RPC params or a REST query."""
    page_size = params.get("pageSize")
    if isinstance(page_size, str):
        try:
            page_size = int(float(page_size))
        except ValueError:
            page_size = None
    status = params.get("status")
    include = params.get("includeArtifacts")
    out: Dict[str, Any] = {}
    if isinstance(params.get("contextId"), str) and params["contextId"]:
        out["context_id"] = params["contextId"]
    if isinstance(status, str) and status.startswith("TASK_STATE_"):
        out["status"] = status
    if isinstance(page_size, (int, float)) and not isinstance(page_size, bool) and page_size > 0:
        out["page_size"] = int(page_size)
    if isinstance(params.get("pageToken"), str) and params["pageToken"]:
        out["page_token"] = params["pageToken"]
    history = _history_length(params.get("historyLength"))
    if history is not None:
        out["history_length"] = history
    if include is False or include == "false":
        out["include_artifacts"] = False
    return out


def _split_verb(raw: Any) -> Tuple[str, Optional[str]]:
    """`("abc", "subscribe")` for `abc:subscribe`; the id alone otherwise."""
    text = raw if isinstance(raw, str) else ""
    colon = text.rfind(":")
    if colon != -1 and text[colon + 1 :] in _VERBS:
        return text[:colon], text[colon + 1 :]
    return text, None


def _parse_rest_body(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise A2AError("INVALID_PARAMS", "Body is not valid JSON") from None
    if not isinstance(body, dict):
        raise A2AError("INVALID_PARAMS", "Body must be a JSON object")
    return body


def _ends_stream(event: Dict[str, Any]) -> bool:
    update = event.get("statusUpdate")
    return isinstance(update, dict) and is_settled(update.get("status", {}).get("state", ""))


def _rpc_response(rpc_id: Any, result: Any = None, error: Any = None) -> JSONResponse:
    body: Dict[str, Any] = {"jsonrpc": JSON_RPC, "id": rpc_id}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result
    return JSONResponse(content=body)


def _rest_json(body: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content=body, media_type=REST_CONTENT_TYPE)


def _rest_error(error: A2AError) -> JSONResponse:
    return _rest_json(error.rest(), error.http)


def _sse_response(
    events: AsyncGenerator[Dict[str, Any], None], wrap: Callable[[Dict[str, Any]], Any]
) -> StreamingResponse:
    async def stream() -> AsyncGenerator[str, None]:
        try:
            async for event in events:
                yield f"data: {json.dumps(wrap(event), ensure_ascii=False)}\n\n"
        finally:
            await events.aclose()

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
