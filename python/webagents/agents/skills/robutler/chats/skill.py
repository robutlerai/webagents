"""
ChatsSkill - Robutler chat metadata enrichment + unreads management.

Fetches the agent's chats from Robutler on initialize and adds them to the
agent's metadata so agent info responses can include chat links. Also provides
tools for checking unread messages and polling for new conversations.

External callers can then connect to /chats/{uuid}/completions or /chats/{uuid}/uamp.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution


class ChatsSkill(Skill):
    """
    Enriches agent metadata with active Robutler chats and transport links.
    Exposes tools for querying unread messages and polling for new conversations.

    On initialize, fetches GET /api/messages from Robutler (using the agent's
    API key) and sets agent.metadata['chats'] with each chat's id, url, and
    transports (completions, uamp).

    Configuration:
    - robutler_url: Robutler API base (default from ROBUTLER_API_URL or config)
    - api_key: Optional; otherwise uses agent.api_key or WEBAGENTS_API_KEY
    - poll_unreads: If True, start a background task polling unreads (default False)
    - poll_interval: Seconds between unreads polls (default 60)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {}, scope="all")
        self.config = self.config or {}
        self.robutler_url = (
            self.config.get("robutler_url")
            or os.getenv("ROBUTLER_API_URL")
            or os.getenv("ROBUTLER_INTERNAL_API_URL")
            or "https://robutler.ai"
        ).rstrip("/")
        self.api_key = self.config.get("api_key")
        self._poll_unreads = self.config.get("poll_unreads", False)
        self._poll_interval = self.config.get("poll_interval", 60)
        self._poll_task: Optional[asyncio.Task] = None
        self._cached_unreads: Optional[List[Dict[str, Any]]] = None
        self.logger = None

    async def initialize(self, agent: Any) -> None:
        self.agent = agent
        self.logger = get_logger("skill.chats", getattr(agent, "name", "unknown"))
        api_key = (
            self.api_key
            or getattr(agent, "api_key", None)
            or os.getenv("WEBAGENTS_API_KEY")
            or os.getenv("SERVICE_TOKEN")
        )
        if not api_key:
            return

        self._resolved_api_key = api_key

        try:
            chats = await self._fetch_chats(api_key)
            if not hasattr(agent, "metadata") or agent.metadata is None:
                agent.metadata = {}
            agent.metadata["chats"] = chats
            agent.chats = chats
            log_skill_event(getattr(agent, "name", "unknown"), "chats", "initialized", {"count": len(chats)})
        except Exception as e:
            if self.logger:
                self.logger.warning("ChatsSkill: could not fetch chats: %s", e)

        # Fetch initial unreads
        try:
            self._cached_unreads = await self._fetch_unreads(api_key)
        except Exception as e:
            if self.logger:
                self.logger.warning("ChatsSkill: could not fetch unreads: %s", e)

        # Optionally start background poll
        if self._poll_unreads:
            self._poll_task = asyncio.create_task(self._unreads_poll_loop(api_key))

    async def _fetch_chats(self, api_key: str) -> List[Dict[str, Any]]:
        import aiohttp

        out: List[Dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.robutler_url}/api/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if not resp.ok:
                    raise RuntimeError(f"Robutler API error: {resp.status}")
                data = await resp.json()
        raw_chats = data.get("chats") or []
        for c in raw_chats:
            chat_id = c.get("id")
            if not chat_id:
                continue
            base = self.robutler_url
            ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
            out.append({
                "id": chat_id,
                "type": c.get("type", "dm"),
                "name": c.get("name"),
                "url": f"{base}/chats/{chat_id}",
                "transports": {
                    "completions": f"{base}/api/chats/{chat_id}/completions",
                    "uamp": f"{ws_base}/chats/{chat_id}/uamp",
                },
                "participants": [
                    p.get("username") for p in (c.get("participants") or []) if p.get("username")
                ],
                "last_message_at": c.get("lastMessageAt"),
            })
        return out

    async def _fetch_unreads(self, api_key: str) -> List[Dict[str, Any]]:
        """Fetch unreads from GET /api/agents/unreads."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.robutler_url}/api/agents/unreads",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if not resp.ok:
                    raise RuntimeError(f"Unreads API error: {resp.status}")
                data = await resp.json()
        return data.get("unreads") or []

    async def _unreads_poll_loop(self, api_key: str) -> None:
        """Background loop that periodically refreshes cached unreads."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                self._cached_unreads = await self._fetch_unreads(api_key)
            except Exception as e:
                if self.logger:
                    self.logger.debug("ChatsSkill: unreads poll error: %s", e)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @prompt(priority=50, scope=["owner", "all"])
    def get_chats_prompt(self) -> str:
        return (
            "You can use get_unreads() to check for unread messages across your chats. "
            "Use refresh_chats() to reload your chat list from the platform."
        )

    @tool(
        name="get_unreads",
        description=(
            "Get a list of chats with unread messages for this agent. "
            "Returns chat_id, unread_count, chat_type, last_message_at for each chat with unreads."
        ),
        scope="all",
    )
    async def get_unreads(self, refresh: bool = False) -> str:
        """Get chats with unread messages."""
        api_key = getattr(self, "_resolved_api_key", None)
        if not api_key:
            return "No API key configured"

        try:
            if refresh or self._cached_unreads is None:
                self._cached_unreads = await self._fetch_unreads(api_key)

            if not self._cached_unreads:
                return "No unread messages."

            lines = []
            for u in self._cached_unreads:
                lines.append(
                    f"- Chat {u['chat_id']} ({u.get('chat_type', 'dm')}): "
                    f"{u['unread_count']} unread, last message at {u.get('last_message_at', 'unknown')}"
                )
            log_tool_execution(
                getattr(self.agent, "name", "unknown"),
                "chats.get_unreads",
                "success",
                {"count": len(self._cached_unreads)},
            )
            return "\n".join(lines)
        except Exception as e:
            log_tool_execution(
                getattr(self.agent, "name", "unknown"),
                "chats.get_unreads",
                "failure",
                {"error": str(e)},
            )
            return f"Error fetching unreads: {e}"

    @tool(
        name="refresh_chats",
        description="Reload the agent's chat list from the platform and return a summary.",
        scope="all",
    )
    async def refresh_chats(self) -> str:
        """Refresh the agent's chat list."""
        api_key = getattr(self, "_resolved_api_key", None)
        if not api_key:
            return "No API key configured"

        try:
            chats = await self._fetch_chats(api_key)
            if hasattr(self.agent, "metadata") and self.agent.metadata is not None:
                self.agent.metadata["chats"] = chats
            self.agent.chats = chats

            if not chats:
                return "No chats found."

            lines = []
            for c in chats:
                participants = ", ".join(c.get("participants") or [])
                lines.append(f"- {c['id']} ({c['type']}): {c.get('name') or 'unnamed'} [{participants}]")

            log_tool_execution(
                getattr(self.agent, "name", "unknown"),
                "chats.refresh_chats",
                "success",
                {"count": len(chats)},
            )
            return f"{len(chats)} chats:\n" + "\n".join(lines)
        except Exception as e:
            log_tool_execution(
                getattr(self.agent, "name", "unknown"),
                "chats.refresh_chats",
                "failure",
                {"error": str(e)},
            )
            return f"Error refreshing chats: {e}"

    async def cleanup(self) -> None:
        """Cancel background poll task on shutdown."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
