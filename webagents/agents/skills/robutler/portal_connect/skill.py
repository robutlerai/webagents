"""
Portal Connect Skill.

Connect local Python agents to the platform (Roborum/Robutler) via UAMP WebSocket.
One WS connection, one session per agent; per-session AOAuth tokens. Replaces
the legacy PortalWSSkill custom protocol with standard UAMP session multiplexing.
"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

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


class PortalConnectSkill(Skill):
    """
    Connect agents to the platform via UAMP WebSocket with session multiplexing.

    - One WS connection to `portal_ws_url` (e.g. wss://roborum.ai/ws or wss://robutler.ai/ws).
    - Connection auth: `?token=<jwt>` (first agent's AOAuth or daemon token).
    - One `session.create` per agent with `session: { agent: "<name>", token: "<aoauth-jwt>" }`.
    - Handles `input.text` → runs agent → sends `response.delta` / `response.done`.
    - Sends UAMP `ping` periodically; handles `pong`, `session.updated`, `session.end`.

    Config:
        portal_ws_url: Full WS URL (e.g. wss://roborum.ai/ws).
        agents: List of { name: str, token: str } for each agent to register.
        auto_reconnect: Whether to reconnect on disconnect (default True).
        reconnect_delay: Seconds between reconnect attempts.
        max_reconnect_attempts: Max attempts before giving up.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        self.portal_ws_url = cfg.get("portal_ws_url", "wss://roborum.ai/ws")
        self.agents: List[Dict[str, str]] = cfg.get("agents", [])  # [{"name": "alice", "token": "jwt..."}]
        self.auto_reconnect = cfg.get("auto_reconnect", True)
        self.reconnect_delay = cfg.get("reconnect_delay", DEFAULT_RECONNECT_DELAY_S)
        self.max_reconnect_attempts = cfg.get("max_reconnect_attempts", DEFAULT_MAX_RECONNECT_ATTEMPTS)

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._shutdown = False
        self._reconnect_attempts = 0
        self._connection_task: Optional[asyncio.Task] = None
        self._session_by_id: Dict[str, str] = {}  # session_id -> agent_name
        self._agent_resolver: Optional[Callable[[str], Any]] = None  # agent_name -> BaseAgent

    def set_agent_resolver(self, resolver: Callable[[str], Any]) -> None:
        """Set a callable that returns the agent instance by name (used by daemon to resolve agents)."""
        self._agent_resolver = resolver

    async def initialize(self, agent: "BaseAgent") -> None:  # noqa: F821
        """Initialize; if single-agent mode, use this agent and its token from config."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not installed. Portal Connect skill disabled.")
            return
        await super().initialize(agent)
        if not self.agents and agent:
            self.agents = [{"name": agent.name, "token": (self.config or {}).get("token", "")}]
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
        if not self._ws or not agent_name or not token:
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

    async def _handle_message(self, message: str) -> None:
        """Handle incoming UAMP event from platform."""
        try:
            data = json.loads(message)
            event_type = data.get("type")
            session_id = data.get("session_id")

            if event_type == "session.created":
                sid = data.get("session_id")
                agent = (data.get("session") or {}).get("agent") or (data.get("agent"))
                if sid and agent:
                    self._session_by_id[sid] = agent
                logger.info("Portal Connect session.created session_id=%s agent=%s", sid, agent)

            elif event_type == "session.updated":
                pass

            elif event_type == "session.end":
                if session_id:
                    self._session_by_id.pop(session_id, None)

            elif event_type == "pong":
                pass

            elif event_type == "input.text":
                await self._handle_input_text(data, session_id)

            elif event_type == "response.error":
                logger.warning("Portal Connect response.error: %s", data.get("error"))

        except Exception as e:
            logger.error("Portal Connect handle message error: %s", e)

    async def _handle_input_text(self, data: Dict[str, Any], session_id: Optional[str]) -> None:
        """Run the agent for this session and send response.delta / response.done."""
        text = data.get("text", "").strip()
        if not text or not session_id:
            return
        agent_name = self._session_by_id.get(session_id)
        if not agent_name:
            logger.warning("Portal Connect input.text unknown session_id=%s", session_id)
            return
        agent = self._agent_resolver(agent_name) if self._agent_resolver else (self.agent if getattr(self, "agent", None) and getattr(self.agent, "name", None) == agent_name else None)
        if not agent:
            logger.warning("Portal Connect no agent for name=%s", agent_name)
            return
        response_id = f"resp_{int(time.time() * 1000)}_{generate_event_id()}"
        try:
            messages = [{"role": "user", "content": text}]
            async for chunk in agent.run_streaming(messages, tools=None):
                if isinstance(chunk, dict) and chunk.get("content"):
                    delta = ContentDelta(type="text", text=chunk["content"])
                    await self._send_response_delta(session_id, response_id, delta)
            await self._send_response_done(session_id, response_id)
        except Exception as e:
            logger.exception("Portal Connect run error for %s: %s", agent_name, e)
            await self._send_response_error(session_id, response_id, str(e))

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

    async def disconnect(self) -> None:
        self._shutdown = True
        self.auto_reconnect = False
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
