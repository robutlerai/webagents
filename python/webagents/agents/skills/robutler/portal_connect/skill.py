"""
Portal Connect Skill.

Connect local Python agents to the platform (Robutler/Robutler) via UAMP WebSocket.
One WS connection, one session per agent; per-session AOAuth tokens. Replaces
the legacy PortalWSSkill custom protocol with standard UAMP session multiplexing.
"""

import asyncio
import base64
import inspect
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from webagents.agents.skills.base import Skill

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent

from webagents.uamp.events import (
    generate_event_id,
    current_timestamp,
)
from webagents.uamp.types import ContentDelta

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any  # type: ignore

logger = logging.getLogger(__name__)

PING_INTERVAL_S = 55
DEFAULT_WS_PATH = "/ws"
DEFAULT_RECONNECT_DELAY_S = 5
DEFAULT_MAX_RECONNECT_ATTEMPTS = 10

# Roles an OpenAI-shaped chat/completions API accepts on a plain turn.
_PLAIN_ROLES = ("system", "user", "assistant", "developer")


class PortalConnectConfigError(ValueError):
    """Raised when the skill is configured with an unusable credential."""


class PortalCredentialError(PortalConnectConfigError):
    """Refused before any I/O: the configured token cannot possibly work."""


def _decode_jwt_claims(token: str) -> Dict[str, Any]:
    """Decode OUR OWN token's payload. No verification — this is the
    credential we are about to present, not one we received."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def check_agent_token(token: str) -> None:
    """Refuse, at start and with the fix in the message, the one credential
    shape that costs people days: an owner-subject key with no agent binding.

    ``POST /api/agents/{id}/api-key`` mints keys whose ``sub`` is the OWNER
    and whose ``agent_id`` claim names the agent — that binding is what lets
    the platform key the socket's session on the AGENT. A generic owner key
    carries no ``agent_id`` at all; the socket then registers under the owner,
    the router never finds the agent's session, and the process idles forever
    with every observable looking healthy (F-045).

    Set ``WEBAGENTS_ALLOW_UNBOUND_TOKEN=1`` to bypass (custom deployments).
    """
    if not token:
        raise PortalCredentialError(
            "No agent token configured. Set WEBAGENTS_AGENT_TOKEN to a per-agent "
            "API key minted with POST /api/agents/{id}/api-key."
        )
    if os.getenv("WEBAGENTS_ALLOW_UNBOUND_TOKEN") == "1":
        return
    claims = _decode_jwt_claims(token)
    if not claims:
        # Not a JWT we can read; let the platform be the judge.
        return
    if not claims.get("agent_id"):
        raise PortalCredentialError(
            "The configured token is not bound to an agent: its subject is "
            f"'{claims.get('sub', '?')}' and it carries no agent_id claim. "
            "This is an owner/account key — the platform will accept the "
            "connection and then never route a single message to this agent. "
            "Fix: mint a per-agent key with POST /api/agents/{id}/api-key "
            "(the returned JWT carries agent_id) and put THAT in "
            "WEBAGENTS_AGENT_TOKEN."
        )


def resolve_portal_ws_url(configured: Optional[str] = None) -> str:
    """Resolve the portal WS URL from config or environment.

    Accepts http(s) URLs (converted to ws(s)) and appends the `/ws` path when
    the URL has no path. Order: explicit config, WEBAGENTS_PORTAL_URL,
    PORTAL_WS_URL (documented compat), then the public default.
    """
    raw = (
        configured
        or os.getenv("WEBAGENTS_PORTAL_URL")
        or os.getenv("PORTAL_WS_URL")
        or "wss://robutler.ai/ws"
    ).strip()
    if raw.startswith("https://"):
        raw = "wss://" + raw[len("https://"):]
    elif raw.startswith("http://"):
        raw = "ws://" + raw[len("http://"):]
    # Append the WS path when the URL is a bare origin.
    scheme_end = raw.find("://")
    rest = raw[scheme_end + 3:] if scheme_end != -1 else raw
    if "/" not in rest:
        raw = raw.rstrip("/") + "/ws"
    return raw


def sanitize_portal_messages(raw: Any) -> List[Dict[str, Any]]:
    """
    Reduce the platform's conversation history to plain OpenAI-shaped turns.

    The portal projects its stored history with ``chatHistoryToOpenAIMessages``,
    which emits two things a stock chat/completions API rejects with a 400:

      * a non-standard ``content_items`` key (signed media descriptors), and
      * ``role: "tool"`` rows paired with ``assistant.tool_calls``.

    The tool pairing is all-or-nothing: keeping the assistant ``tool_calls``
    while dropping the ``tool`` replies is *also* a 400 ("an assistant message
    with tool_calls must be followed by tool messages"), so both halves go.
    What survives is the conversation as text, which is what a fresh local run
    can actually act on.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in _PLAIN_ROLES:
            # Drops role="tool" rows (and anything else unexpected).
            continue
        content = item.get("content")
        if isinstance(content, list):
            # Multimodal array form: keep the text parts, drop the rest.
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "".join(part for part in parts if part)
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            # An assistant row that carried only tool_calls or only media has
            # nothing left to say once those are stripped.
            continue
        out.append({"role": role, "content": content})
    return out


class PortalConnectSkill(Skill):
    """
    Connect agents to the platform via UAMP WebSocket with session multiplexing.

    - One WS connection to `portal_ws_url` (e.g. wss://robutler.ai/ws or wss://robutler.ai/ws).
    - Connection auth: `?token=<jwt>` (first agent's AOAuth or daemon token).
    - One `session.create` per agent with `session: { agent: "<name>", token: "<aoauth-jwt>" }`.
    - Handles `input.text` → runs agent → sends `response.delta` / `response.done`.
    - Sends UAMP `ping` periodically; handles `pong`, `session.updated`, `session.end`.

    Config (every field has an environment fallback — attaching the skill is
    the whole configuration):
        portal_ws_url: Full WS URL (e.g. wss://robutler.ai/ws). Falls back to
            WEBAGENTS_PORTAL_URL, then PORTAL_WS_URL.
        agents: List of { name: str, token: str } for each agent to register.
            Defaults to the attached agent with WEBAGENTS_AGENT_TOKEN.
        auto_reconnect: Whether to reconnect on disconnect (default True).
        reconnect_delay: Seconds between reconnect attempts.
        max_reconnect_attempts: Max attempts before giving up.
        autostart: Open the connection from initialize() (default True).
            False keeps registration-only behaviour and WARNS.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        # URL resolution: explicit config, then WEBAGENTS_PORTAL_URL (http(s)
        # accepted and converted, `/ws` appended when missing), then the
        # long-documented PORTAL_WS_URL for compatibility. The docs promised
        # an env fallback for years while the skill read only its config.
        self.portal_ws_url = resolve_portal_ws_url(cfg.get("portal_ws_url"))
        self.agents: List[Dict[str, str]] = cfg.get("agents", [])  # [{"name": "alice", "token": "jwt..."}]
        self.auto_reconnect = cfg.get("auto_reconnect", True)
        self.reconnect_delay = cfg.get("reconnect_delay", DEFAULT_RECONNECT_DELAY_S)
        self.max_reconnect_attempts = cfg.get("max_reconnect_attempts", DEFAULT_MAX_RECONNECT_ATTEMPTS)

        # Start-on-initialize (see initialize()). Opt out with
        # {"autostart": False} when something else owns the lifecycle.
        self.autostart = cfg.get("autostart", True)
        self._started = False
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._shutdown = False
        self._reconnect_attempts = 0
        self._connection_task: Optional[asyncio.Task] = None
        self._session_by_id: Dict[str, str] = {}  # session_id -> agent_name
        self._agent_resolver: Optional[Callable[[str], Any]] = None  # agent_name -> BaseAgent
        # One in-flight turn per session_id. A model turn used to be awaited
        # inline in the read loop, so a slow turn stalled EVERY multiplexed
        # agent on this socket — including its own `session.end`. Running each
        # turn as its own task also gives it an isolated ContextVar copy, which
        # is what makes a per-turn payment token safe.
        self._runs: Dict[str, "asyncio.Task[None]"] = {}

    def set_agent_resolver(self, resolver: Callable[[str], Any]) -> None:
        """Set a callable that returns the agent instance by name (used by daemon to resolve agents).

        The resolver may be sync or async (`WebAgentsServer.resolve_agent` is
        a coroutine function; several callers pass a plain lambda).
        """
        self._agent_resolver = resolver

    async def _resolve_agent_by_name(self, agent_name: str) -> Any:
        """Resolve via the configured resolver (awaiting when needed), else
        fall back to this skill's own agent when the name matches."""
        if self._agent_resolver:
            result = self._agent_resolver(agent_name)
            if inspect.isawaitable(result):
                result = await result
            return result
        own = getattr(self, "agent", None)
        if own is not None and getattr(own, "name", None) == agent_name:
            return own
        return None

    async def initialize(self, agent: "BaseAgent") -> None:  # noqa: F821
        """Initialize; if single-agent mode, use this agent and its token from config.

        Reads its own credentials from the environment when config omits
        them: ``WEBAGENTS_AGENT_TOKEN`` for the per-agent key (here) and
        ``WEBAGENTS_PORTAL_URL`` / ``PORTAL_WS_URL`` for the endpoint (in
        ``__init__``, via ``resolve_portal_ws_url``). Attaching the skill is
        therefore the WHOLE configuration — there is nothing for a caller to
        thread through, which is why no wrapper needs to exist.

        OPENS THE CONNECTION. `start()` is the explicit entry point
        (`WebAgentsServer` startup calls it), but initialize() calls it too,
        because every stock setup — a daemon that only constructs the skill,
        anything that relies on `BaseAgent`'s lazy skill init — has no other
        hook to call. Registration-only initialization meant those connected
        NEVER, silently, while the agent looked healthy.

        There is no deadlock in doing this: `start()` only schedules
        `_connect_and_serve()` as a task and returns; it never awaits the
        socket. `start()` is idempotent, so an explicit call after this one
        is a no-op.

        Pass `config={"autostart": False}` to keep the old registration-only
        behaviour — that path WARNS, because a skill that is initialized and
        never started is indistinguishable from a healthy one from the
        outside.
        """
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not installed. Portal Connect skill disabled.")
            return
        await super().initialize(agent)
        if not self.agents and agent:
            token = (self.config or {}).get("token") or os.getenv("WEBAGENTS_AGENT_TOKEN", "")
            self.agents = [{"name": agent.name, "token": token}]
        if self.autostart:
            await self.start()
        else:
            logger.warning(
                "Portal Connect initialized for %s with autostart=False and NOT started: "
                "no socket is open and no turn can arrive until start() is called.",
                ", ".join(a.get("name", "?") for a in self.agents) or "<no agents>",
            )

    async def start(self) -> None:
        """Open the portal connection (idempotent). Called by initialize().

        Refuses HERE, before any socket is opened, when a configured token
        cannot work (see :func:`check_agent_token`). That check used to live
        in a `connect()` wrapper, so it protected exactly one entry point and
        nothing else; on the skill it protects every one of them.
        """
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not installed. Portal Connect skill disabled.")
            return
        if self._connection_task and not self._connection_task.done():
            return
        for entry in self.agents:
            check_agent_token(entry.get("token") or "")
        self._started = True
        self._connection_task = asyncio.create_task(self._connect_and_serve())

    async def _connect_and_serve(self) -> None:
        """Connect to portal and handle messages; reconnect when needed."""
        while not self._shutdown:
            try:
                token = (self.agents[0].get("token") or "") if self.agents else ""
                url = f"{self.portal_ws_url}?token={token}" if token else self.portal_ws_url
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_attempts = 0
                    self._session_by_id.clear()
                    for ag in self.agents:
                        await self._send_session_create(ag.get("name", ""), ag.get("token", ""))
                    ping_task = asyncio.create_task(self._run_ping_loop())
                    try:
                        async for message in ws:
                            await self._handle_message(message)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                self._ws = None
                self._session_by_id.clear()
                if self._shutdown:
                    break
                if self.auto_reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info("Portal Connect reconnecting (attempt %s)...", self._reconnect_attempts)
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Portal Connect max reconnect attempts reached")
                    break
            except Exception as e:
                logger.error("Portal Connect error: %s", e)
                self._connected = False
                self._ws = None
                if self._shutdown:
                    break
                if self.auto_reconnect:
                    await asyncio.sleep(self.reconnect_delay)

    async def _run_ping_loop(self) -> None:
        """Send UAMP ping periodically (runs until connection closed)."""
        try:
            while self._ws and self._connected and not self._shutdown:
                await asyncio.sleep(PING_INTERVAL_S)
                if self._ws and self._connected:
                    await self._send_raw({"type": "ping", "event_id": generate_event_id(), "timestamp": current_timestamp()})
        except asyncio.CancelledError:
            pass

    async def _send_session_create(self, agent_name: str, token: str) -> None:
        """Send session.create for one agent with per-agent AOAuth token."""
        # An empty token used to return silently: the socket opened, no
        # `session.create` was ever sent, and nothing was logged — a typo'd
        # env var was indistinguishable from success. Fail loudly instead.
        if agent_name and not token:
            raise PortalConnectConfigError(
                f"Portal Connect: no token configured for agent '{agent_name}'. "
                "Set the per-agent token (its subject must be the agent, or its "
                "owner holding a per-agent API key) before connecting."
            )
        if not self._ws or not agent_name:
            return
        event = {
            "type": "session.create",
            "event_id": generate_event_id(),
            "timestamp": current_timestamp(),
            "uamp_version": "1.0",
            "session": {"agent": agent_name, "token": token},
        }
        await self._ws.send(json.dumps(event))
        logger.info("Portal Connect sent session.create for agent '%s'", agent_name)

    @staticmethod
    def _normalize(data: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Classify one inbound frame. The socket carries two unrelated shapes.

          * COMMAND   — a top-level ``type``. Addressed to this daemon; the
                        only shape that may start a run.
          * BROADCAST — an ``event`` key with NO top-level ``type``: the
                        wrapped chat envelope ``{event, chatId, _origin,
                        _portal}`` that the platform fans out to everyone on
                        the user's channel. It is a copy of what happened, not
                        a request. It must NEVER start a run.

        Both shapes arrive for the same turn, which is why ``message.created``
        is deliberately NOT mapped onto the ``input.text`` path: doing so runs
        the model twice per turn and bills twice.

        Anything else is logged with its keys so an unrecognised shape is
        visible rather than silently swallowed.
        """
        if not isinstance(data, dict):
            return "unknown", {}
        if isinstance(data.get("type"), str):
            return "command", data
        if "event" in data:
            return "broadcast", data
        logger.warning("Portal Connect unrecognised frame keys=%s", sorted(data.keys()))
        return "unknown", data

    async def _handle_message(self, message: str) -> None:
        """Handle incoming UAMP event from platform."""
        try:
            data = json.loads(message)
            kind, data = self._normalize(data)
            if kind != "command":
                if kind == "broadcast":
                    inner = data.get("event") or {}
                    logger.debug(
                        "Portal Connect broadcast %s (chat=%s) — not a command",
                        (inner or {}).get("type") if isinstance(inner, dict) else None,
                        data.get("chatId"),
                    )
                return

            event_type = data.get("type")
            session_id = data.get("session_id")

            if event_type == "session.created":
                sid = data.get("session_id")
                session = data.get("session") or {}
                # Key on the agent string the PLATFORM echoed, which is the
                # string this daemon sent. Never on `agent_username`: a daemon
                # that connected by UUID would then never match its own turns.
                agent = session.get("agent") or data.get("agent")
                if sid and agent:
                    self._session_by_id[sid] = agent
                logger.info(
                    "Portal Connect session.created session_id=%s agent=%s agent_id=%s",
                    sid, agent, session.get("agent_id"),
                )

            elif event_type == "session.updated":
                pass

            elif event_type == "session.error":
                # The primary observable for a refused connect credential.
                # Before this arm existed the frame fell through in silence and
                # a rejected token looked exactly like a healthy idle socket.
                error = data.get("error") or {}
                logger.error(
                    "Portal Connect session.error code=%s: %s",
                    error.get("code"), error.get("message"),
                )

            elif event_type == "session.end":
                if session_id:
                    self._session_by_id.pop(session_id, None)
                    self._cancel_run(session_id)

            elif event_type == "pong":
                pass

            elif event_type == "input.text":
                self._start_run(data, session_id)

            elif event_type == "response.cancel":
                await self._handle_response_cancel(session_id)

            elif event_type == "payment.submit":
                payment = data.get("payment") or {}
                logger.info(
                    "Portal Connect payment.submit session=%s scheme=%s amount=%s",
                    session_id, payment.get("scheme"), payment.get("amount"),
                )

            elif event_type == "payment.error":
                logger.error(
                    "Portal Connect payment.error session=%s code=%s can_retry=%s: %s",
                    session_id, data.get("code"), data.get("can_retry"), data.get("message"),
                )

            elif event_type == "response.error":
                logger.warning("Portal Connect response.error: %s", data.get("error"))

            elif event_type == "extension.message":
                # A namespaced sub-protocol frame. The portal sends
                # `workspace.terminal` envelopes onto this socket
                # (lib/terminal/backend-webagentsd.ts). The PYTHON SDK
                # IMPLEMENTS NO SUB-PROTOCOL: there is no PTY surface here and
                # never has been (the terminal router is TypeScript-only, in
                # src/transport/terminal). This arm exists so the frame is
                # REFUSED OUT LOUD instead of falling through in silence — the
                # portal converts the missing `ready` into a `peer_offline`
                # close, and this log is what connects that to a cause.
                logger.warning(
                    "Portal Connect received extension.message namespace=%s "
                    "(v%s), which this SDK does not implement; the portal will "
                    "time the session out as peer_offline. Use the TypeScript "
                    "SDK's new PortalConnectSkill({ terminal: true }) for "
                    "workspace.terminal.",
                    data.get("namespace"),
                    data.get("extension_version"),
                )

        except Exception as e:
            logger.error("Portal Connect handle message error: %s", e)

    def _start_run(self, data: Dict[str, Any], session_id: Optional[str]) -> None:
        """Run one turn in its own task so it cannot block the read loop."""
        if not session_id:
            return
        self._cancel_run(session_id)
        task = asyncio.ensure_future(self._handle_input_text(data, session_id))
        self._runs[session_id] = task

        def _reap(t: "asyncio.Task[None]") -> None:
            if self._runs.get(session_id) is t:
                self._runs.pop(session_id, None)

        task.add_done_callback(_reap)

    def _cancel_run(self, session_id: str) -> bool:
        """Cancel the in-flight turn for a session. True if one was running."""
        task = self._runs.pop(session_id, None)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _handle_response_cancel(self, session_id: Optional[str]) -> None:
        """
        Stop the run for this session and acknowledge.

        This is the ONLY cancel signal a daemon gets — the platform sends it
        when the user hits Stop and from the disconnect-grace fan-out. The
        acknowledgement (`response.cancelled`) is already handled portal-side.
        """
        if not session_id:
            return
        cancelled = self._cancel_run(session_id)
        logger.info(
            "Portal Connect response.cancel session=%s (run_active=%s)", session_id, cancelled
        )
        await self._send_raw({
            "type": "response.cancelled",
            "event_id": generate_event_id(),
            "timestamp": current_timestamp(),
            "session_id": session_id,
        })

    async def _handle_input_text(self, data: Dict[str, Any], session_id: Optional[str]) -> None:
        """Run the agent for this session and send response.delta / response.done."""
        text = data.get("text", "").strip()
        if not text or not session_id:
            return
        agent_name = self._session_by_id.get(session_id)
        if not agent_name:
            # Fall back to the frame's own `agent` field. Multi-pod routing
            # dispatches with a per-request session id that never appeared in
            # a local `session.created`, so a sid-only lookup would silently
            # drop every cross-pod turn (R9). The old behaviour — return and
            # send nothing — is exactly what the M5 dispatch hop cannot
            # tolerate.
            #
            # INERT TODAY: no portal emitter sets `agent` on the frame yet
            # (`sendInputToAgentSession` omits it, and `dispatchInputToAgent`
            # puts `agentId` on the Redis WRAPPER, not on the frame). This is
            # the parser half of the M5 prerequisite, ready and tested; until
            # the emitters follow, an unmapped sid still drops — with the
            # error log below instead of silence.
            agent_name = data.get("agent")
            if agent_name:
                logger.info(
                    "Portal Connect input.text for unmapped session_id=%s; using frame agent=%s",
                    session_id, agent_name,
                )
                self._session_by_id[session_id] = agent_name
        if not agent_name:
            logger.error(
                "Portal Connect input.text unknown session_id=%s and no agent field on the frame; dropping",
                session_id,
            )
            return
        agent = await self._resolve_agent_by_name(agent_name)
        if not agent:
            logger.warning("Portal Connect no agent for name=%s", agent_name)
            return
        response_id = f"resp_{int(time.time() * 1000)}_{generate_event_id()}"
        # The platform has been sending the FULL conversation on every
        # `input.text` all along; this used to discard it and synthesize a
        # single user turn, which made every hosted agent memoryless per turn.
        messages = sanitize_portal_messages(data.get("messages"))
        if not messages:
            messages = [{"role": "user", "content": text}]
        self._apply_payment_token(data.get("payment_token"))
        try:
            async for chunk in agent.run_streaming(messages, tools=None):
                if not isinstance(chunk, dict):
                    continue
                # Handle tool_call / tool_result / tool_progress events
                chunk_type = chunk.get("type")
                if chunk_type in ("tool_call", "tool_result", "tool_progress"):
                    await self._send_raw({
                        "type": "response.delta",
                        "event_id": generate_event_id(),
                        "timestamp": current_timestamp(),
                        "session_id": session_id,
                        "response_id": response_id,
                        "delta": chunk,
                    })
                    continue
                # Handle OpenAI-compatible streaming chunks
                choices = chunk.get("choices", [])
                if choices:
                    delta_content = choices[0].get("delta", {}).get("content")
                    if delta_content:
                        delta = ContentDelta(type="text", text=delta_content)
                        await self._send_response_delta(session_id, response_id, delta)
                elif chunk.get("content"):
                    delta = ContentDelta(type="text", text=chunk["content"])
                    await self._send_response_delta(session_id, response_id, delta)
            await self._send_response_done(session_id, response_id)
        except asyncio.CancelledError:
            # A `response.cancel` (or session.end) killed this turn. The
            # acknowledgement is sent by the cancel handler; do not turn a
            # deliberate stop into a `response.error`.
            logger.info("Portal Connect run cancelled for %s (session=%s)", agent_name, session_id)
            raise
        except Exception as e:
            logger.exception("Portal Connect run error for %s: %s", agent_name, e)
            await self._send_response_error(session_id, response_id, str(e))

    def _apply_payment_token(self, payment_token: Optional[str]) -> None:
        """
        Put the turn's payment token where `PaymentsSkill` looks for it.

        `_extract_payment_token` has read `context.payment_token` since it was
        written and nothing had ever set it, so every hosted agent ran unfunded
        and any paid sub-agent call it tried failed. Each turn runs in its own
        task, so the ContextVar write here is that turn's alone.
        """
        if not payment_token:
            return
        try:
            from webagents.server.context.context_vars import (
                create_context,
                get_context,
                set_context,
            )
            context = get_context()
            if context is None:
                context = create_context()
                set_context(context)
            setattr(context, "payment_token", payment_token)
        except Exception as e:  # pragma: no cover - context module is optional
            logger.warning("Portal Connect could not attach payment token: %s", e)

    async def _send_raw(self, obj: Dict[str, Any]) -> None:
        if self._ws and self._connected:
            await self._ws.send(json.dumps(obj))

    async def _send_response_delta(self, session_id: str, response_id: str, delta: ContentDelta) -> None:
        event = {
            "type": "response.delta",
            "event_id": generate_event_id(),
            "timestamp": current_timestamp(),
            "session_id": session_id,
            "response_id": response_id,
            "delta": {"type": delta.type, "text": getattr(delta, "text", None)},
        }
        await self._send_raw(event)

    async def _send_response_done(self, session_id: str, response_id: str) -> None:
        event = {
            "type": "response.done",
            "event_id": generate_event_id(),
            "timestamp": current_timestamp(),
            "session_id": session_id,
            "response_id": response_id,
        }
        await self._send_raw(event)

    async def _send_response_error(self, session_id: str, response_id: str, message: str) -> None:
        event = {
            "type": "response.error",
            "event_id": generate_event_id(),
            "timestamp": current_timestamp(),
            "session_id": session_id,
            "response_id": response_id,
            "error": {"code": "agent_error", "message": message},
        }
        await self._send_raw(event)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_started(self) -> bool:
        """True once start() has scheduled the connection task. A skill that
        is initialized but not started never receives a single turn, so this
        is what a health check should look at — `is_connected` alone cannot
        tell "never started" from "reconnecting"."""
        return self._started

    async def stop(self) -> None:
        """Close the socket and unwind the bridge loop.

        The cross-SDK name: TypeScript's `PortalConnectSkill.stop()` does the
        same thing, and the two socket-only examples sit in the same section of
        the same doc page — they must not use different verbs for one
        lifecycle. `disconnect()` is the original name and stays.
        """
        await self.disconnect()

    async def disconnect(self) -> None:
        self._shutdown = True
        self.auto_reconnect = False
        for sid in list(self._runs):
            self._cancel_run(sid)
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        self._session_by_id.clear()
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
