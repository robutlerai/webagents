"""
RobutlerKVSkill - Simple key/value storage backed by the portal /api/kv endpoint.

Used for storing small secrets and tokens like OAuth credentials per user/agent.
"""

from typing import Any, Dict, Optional
import os

from ....base import Skill
from robutler.api.client import RobutlerClient
from webagents.agents.tools.decorators import tool


class RobutlerKVSkill(Skill):
    """KV storage skill using the WebAgents portal API"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.portal_url = config.get('portal_url') if config else os.getenv('ROBUTLER_API_URL', 'http://localhost:3000')
        self.api_key = config.get('api_key') if config else os.getenv('WEBAGENTS_API_KEY')
        self.client: Optional[RobutlerClient] = None
        self.agent = None

    async def initialize(self, agent):
        await super().initialize(agent)
        self.agent = agent
        # Prefer agent.api_key over env
        api_key = getattr(agent, 'api_key', None) or self.api_key
        self.client = RobutlerClient(api_key=api_key, base_url=self.portal_url)

    async def cleanup(self):
        if self.client:
            await self.client.close()

    # Internal helpers
    def _subject(self) -> Dict[str, Optional[str]]:
        # Use new subject addressing with type/id
        agent_id = getattr(self.agent, 'id', None)
        if agent_id:
            return {"subjectType": "agent", "subjectId": agent_id}
        # Fallback legacy agentId if missing (shouldn't happen)
        return {"agentId": agent_id}

    # Programmatic API
    async def kv_set(self, key: str, value: str, namespace: Optional[str] = None) -> bool:
        if not self.client:
            raise RuntimeError("KV client not initialized")
        body = {"key": key, "value": value}
        body.update(self._subject())
        if namespace is not None:
            body["namespace"] = namespace
        res = await self.client._make_request('POST', '/kv', data=body)
        return bool(res.success)

    async def kv_get(self, key: str, namespace: Optional[str] = None) -> Optional[str]:
        if not self.client:
            raise RuntimeError("KV client not initialized")
        params = {"key": key}
        params.update(self._subject())
        if namespace is not None:
            params["namespace"] = namespace
        res = await self.client._make_request('GET', '/kv', params=params)
        if not res.success:
            return None
        data = res.data or {}
        if data.get('found'):
            return data.get('value')
        return None

    # Expose minimal tools for manual ops (owner scope)
    @tool(
        description="Store a small piece of data (like credentials or settings) by key. Use sparingly - not for large data or temporary storage.",
        scope="owner"
    )
    async def kv_store(self, key: str, value: str, namespace: Optional[str] = None) -> str:
        ok = await self.kv_set(key, value, namespace)
        return "stored" if ok else "failed"

    @tool(
        description="Retrieve a previously stored piece of data by key. Returns empty string if not found.",
        scope="owner"
    )
    async def kv_read(self, key: str, namespace: Optional[str] = None) -> str:
        val = await self.kv_get(key, namespace)
        return val or ""


