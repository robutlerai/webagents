import os
from typing import Any, Dict, Optional

import httpx

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.server.context.context_vars import get_context


class KVSkill(Skill):
    """Simple per-agent key-value storage via portal /api/kv.

    Scope: owner-only tools to ensure only the agent owner can read/write.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config or {}, scope="all")
        self.portal_url = os.getenv('ROBUTLER_INTERNAL_API_URL') or os.getenv('ROBUTLER_API_URL', 'http://localhost:3000')

    async def _resolve_agent_and_user(self) -> Optional[Dict[str, str]]:
        try:
            ctx = get_context()
            agent = getattr(ctx, 'agent', None)
            auth = getattr(ctx, 'auth', None) or (ctx and ctx.get('auth'))
            agent_id = getattr(agent, 'id', None)
            user_id = getattr(auth, 'user_id', None)
            if agent_id and user_id:
                return {"agent_id": agent_id, "user_id": user_id}
        except Exception:
            pass
        return None

    async def _headers(self) -> Dict[str, str]:
        # Prefer agent API key for auth
        api_key = getattr(self.agent, 'api_key', None)
        return {"X-API-Key": api_key} if api_key else {}

    @tool(description="Set a key to a string value (owner-only)", scope="owner")
    async def kv_set(self, key: str, value: str, namespace: Optional[str] = None) -> str:
        ids = await self._resolve_agent_and_user()
        if not ids:
            return "❌ Missing agent/user context"
        body = {"agentId": ids["agent_id"], "key": key, "value": value, "namespace": namespace}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{self.portal_url}/api/kv", json=body, headers=await self._headers())
            if resp.status_code >= 400:
                return f"❌ KV set failed: {resp.text}"
        return "✅ Saved"

    @tool(description="Get a string value by key (owner-only)", scope="owner")
    async def kv_get(self, key: str, namespace: Optional[str] = None) -> str:
        ids = await self._resolve_agent_and_user()
        if not ids:
            return ""
        params = {"agentId": ids["agent_id"], "key": key}
        if namespace:
            params["namespace"] = namespace
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self.portal_url}/api/kv", params=params, headers=await self._headers())
            if resp.status_code >= 400:
                return ""
            data = resp.json()
            return data.get('value') or ""

    @tool(description="Delete a key (owner-only)", scope="owner")
    async def kv_delete(self, key: str, namespace: Optional[str] = None) -> str:
        ids = await self._resolve_agent_and_user()
        if not ids:
            return ""
        params = {"agentId": ids["agent_id"], "key": key}
        if namespace:
            params["namespace"] = namespace
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(f"{self.portal_url}/api/kv", params=params, headers=await self._headers())
            if resp.status_code >= 400:
                return ""
        return "🗑️ Deleted"


