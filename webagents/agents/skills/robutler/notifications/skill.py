import os
from typing import Any, Dict, Optional, List

import httpx

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, prompt
from webagents.server.context.context_vars import get_context
from webagents.utils.logging import get_logger, log_skill_event, log_tool_execution


def _resolve_portal_url() -> str:
    return os.getenv("ROBUTLER_INTERNAL_API_URL") or os.getenv("ROBUTLER_API_URL", "http://localhost:3000")


class NotificationsSkill(Skill):
    """Send push notifications to the owner of this agent (owner-only).

    This skill exposes a single owner-scoped tool that sends a push notification via the
    portal notifications API (POST /api/notifications/send). The notification will be
    targeted to this agent's owner user ID.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Align with other skills (e.g., VectorMemorySkill) for consistent SDK behavior
        super().__init__(config or {}, scope="all")
        self.config = config or {}
        self.logger = None

    async def initialize(self, agent) -> None:
        self.agent = agent
        self.logger = get_logger('skill.notifications', agent.name)
        # Log level is configured globally via setup_logging()
        log_skill_event(agent.name, 'notifications', 'initialized', {})
        # Cache agent API key for later use
        try:
            self.agent_api_key: Optional[str] = getattr(agent, 'api_key', None)
        except Exception:
            self.agent_api_key = None

    @prompt(priority=60, scope=["owner"])  # Owner sees how to use it
    def get_notifications_prompt(self) -> str:
        return (
            "You can use send_notification(title, body, tag?, type?, priority?, requireInteraction?, silent?, ttl?) to send a push notifications to the owner of this agent. Use it only when explicitly asked to do so."
        )

    def _get_owner_user_id(self) -> Optional[str]:
        # Prefer agent metadata set by factory
        try:
            owner_id = getattr(self.agent, 'owner_user_id', None)
            if owner_id:
                return owner_id
        except Exception:
            pass
        # Fallback to auth context if available
        try:
            ctx = get_context()
            auth_ctx = getattr(ctx, 'auth', None) or (ctx and ctx.get('auth'))
            if auth_ctx and getattr(auth_ctx, 'scope', '').upper() == 'OWNER':
                # Not strictly the owner ID, but if missing we skip
                return getattr(auth_ctx, 'user_id', None)
        except Exception:
            pass
        return None

    async def _post_notification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        base_url = _resolve_portal_url()
        url = f"{base_url}/api/notifications/send"
        headers = {
            'Content-Type': 'application/json'
        }
        # Use agent API key only (no service tokens here)
        api_key = getattr(self, 'agent_api_key', None) or getattr(self.agent, 'api_key', None)
        if not api_key:
            raise RuntimeError("Agent API key unavailable for notifications")
        headers['X-API-Key'] = api_key

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {'status': resp.status_code, 'text': resp.text}
            if resp.status_code >= 400:
                raise RuntimeError(f"Notification send failed: {resp.status_code} {data}")
            return data

    @tool(
        name="send_notification",
        description=(
            "Send a push notification. "
            "Parameters: title (string), body (string). Optional: tag (string), type (chat_message|agent_update|system_announcement|marketing), "
            "priority (low|normal|high|urgent), requireInteraction (bool), silent (bool), ttl (seconds)."
        ),
        scope="owner",
    )
    async def send_notification(
        self,
        title: str,
        body: str,
        tag: Optional[str] = None,
        type: Optional[str] = "agent_update",
        priority: Optional[str] = "normal",
        requireInteraction: Optional[bool] = False,
        silent: Optional[bool] = False,
        ttl: Optional[int] = 86400,
        context: Any = None,
    ) -> str:
        owner_id = self._get_owner_user_id()
        if not owner_id:
            return "❌ Cannot resolve agent owner user ID"

        payload: Dict[str, Any] = {
            'title': title,
            'body': body,
            'type': type or 'agent_update',
            'priority': priority or 'normal',
            'userIds': [owner_id],
            'requireInteraction': bool(requireInteraction),
            'silent': bool(silent),
            'ttl': int(ttl or 86400),
        }
        if tag:
            payload['tag'] = tag

        # Call API
        try:
            result = await self._post_notification(payload)
            log_tool_execution(self.agent.name, 'notifications.send_notification', 'success', {
                'owner_id': owner_id,
                'title': title[:50]
            })
            return f"✅ Notification queued: {result.get('message', 'ok')}"
        except Exception as e:
            log_tool_execution(self.agent.name, 'notifications.send_notification', 'failure', {
                'owner_id': owner_id,
                'error': str(e)
            })
            return f"❌ Failed to send notification: {e}"


