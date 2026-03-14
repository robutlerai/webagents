"""
Agent Memory Skill (Portal-backed)

Store-based key-value storage with grants, search, and sharing.
Replaces the old KVSkill with a richer interface aligned to
/api/storage/memory endpoints.
"""

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool
from webagents.server.context.context_vars import get_context


class MemorySkill(Skill):
    """Store-based agent memory via portal /api/storage/memory.

    Supports multiple stores (self, chat, user, shared), text search,
    grants for inter-agent sharing, and internal API for skill-only data.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        context_stores: Optional[List[Dict[str, str]]] = None,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        super().__init__(config or {}, scope="all")
        self.portal_url = (
            os.getenv("ROBUTLER_INTERNAL_API_URL")
            or os.getenv("ROBUTLER_API_URL", "http://localhost:3000")
        )
        self._agent_id = agent_id
        self._context_stores = context_stores or []
        self._chat_id = chat_id
        self._user_id = user_id

    def initialize(self, agent: Any) -> None:
        super().initialize(agent)
        self._register_memory_tool()

    def _register_memory_tool(self) -> None:
        store_lines: List[str] = []
        if self._agent_id:
            store_lines.append(f"- {self._agent_id} (self): Your persistent memory")
        for s in self._context_stores:
            store_lines.append(f"- {s['storeId']} ({s['label']}): {s['label']} memory")

        stores_section = ""
        if store_lines:
            stores_section = (
                "\nAvailable stores:\n"
                + "\n".join(store_lines)
                + "\nUse 'stores' action to discover additional stores shared with you.\n"
            )

        description = (
            "Persistent memory. Store and retrieve information across conversations."
            + stores_section
            + "\nActions:\n"
            "- get(store, key): retrieve a stored value\n"
            "- set(store, key, value, ttl?): store a value\n"
            "- delete(store, key): remove a key (own entries only)\n"
            "- list(store, prefix?): list keys in a store\n"
            "- search(query, store?): full-text search\n"
            "- share(store, agent, level?): grant another agent access\n"
            "- unshare(store, agent): revoke a grant\n"
            "- stores(): list all stores you can access"
        )

        default_store = self._agent_id or "self"

        async def memory_handler(
            action: str,
            store: Optional[str] = None,
            key: Optional[str] = None,
            value: Any = None,
            ttl: Optional[int] = None,
            prefix: Optional[str] = None,
            query: Optional[str] = None,
            agent: Optional[str] = None,
            level: Optional[str] = None,
        ) -> Any:
            return await self._handle_memory(
                action=action, store=store, key=key, value=value,
                ttl=ttl, prefix=prefix, query=query, agent=agent, level=level,
            )

        memory_handler.__name__ = "memory"
        memory_handler.__doc__ = description
        memory_handler._webagents_tool_definition = {
            "type": "function",
            "function": {
                "name": "memory",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["get", "set", "delete", "list", "search", "share", "unshare", "stores"],
                            "description": "Operation to perform",
                        },
                        "store": {
                            "type": "string",
                            "description": f"Store UUID. Default: {default_store}",
                        },
                        "key": {"type": "string", "description": "Storage key"},
                        "value": {"description": "Value to store (any JSON-serializable)"},
                        "ttl": {"type": "number", "description": "TTL in seconds"},
                        "prefix": {"type": "string", "description": "Key prefix (for list)"},
                        "query": {"type": "string", "description": "Search query (for search)"},
                        "agent": {"type": "string", "description": "Agent UUID (for share/unshare)"},
                        "level": {
                            "type": "string",
                            "enum": ["search", "read", "readwrite"],
                            "description": "Access level (for share)",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

        self.register_tool(memory_handler, scope="all")

    async def _resolve_agent_id(self) -> Optional[str]:
        if self._agent_id:
            return self._agent_id
        try:
            ctx = get_context()
            agent = getattr(ctx, "agent", None)
            return getattr(agent, "id", None)
        except Exception:
            return None

    async def _headers(self) -> Dict[str, str]:
        api_key = getattr(self.agent, "api_key", None) if self.agent else None
        return {"X-API-Key": api_key} if api_key else {}

    async def _handle_memory(self, **kwargs: Any) -> Any:
        action = kwargs.get("action")
        agent_id = await self._resolve_agent_id()
        store = kwargs.get("store") or agent_id
        headers = await self._headers()
        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=15) as client:
            if action == "get":
                key = kwargs.get("key")
                if not key:
                    return {"error": "key is required"}
                params = {"agentId": agent_id, "store": store}
                if self._chat_id:
                    params["chatId"] = self._chat_id
                if self._user_id:
                    params["userId"] = self._user_id
                resp = await client.get(
                    f"{self.portal_url}/api/storage/memory/{key}",
                    params=params, headers=headers,
                )
                if resp.status_code == 404:
                    return None
                if resp.status_code == 403:
                    return {"error": "Access denied"}
                if resp.status_code >= 400:
                    return {"error": f"get failed: {resp.status_code}"}
                return resp.json()

            elif action == "set":
                key = kwargs.get("key")
                if not key:
                    return {"error": "key is required"}
                if kwargs.get("value") is None:
                    return {"error": "value is required"}
                body = {
                    "value": kwargs["value"],
                    "ttl": kwargs.get("ttl", 0),
                    "agentId": agent_id,
                    "store": store,
                }
                if self._chat_id:
                    body["chatId"] = self._chat_id
                if self._user_id:
                    body["userId"] = self._user_id
                resp = await client.put(
                    f"{self.portal_url}/api/storage/memory/{key}",
                    json=body, headers=headers,
                )
                if resp.status_code == 403:
                    return {"error": "Access denied"}
                if resp.status_code >= 400:
                    return {"error": f"set failed: {resp.status_code}"}
                return "OK"

            elif action == "delete":
                key = kwargs.get("key")
                if not key:
                    return {"error": "key is required"}
                params = {"agentId": agent_id, "store": store}
                if self._chat_id:
                    params["chatId"] = self._chat_id
                resp = await client.delete(
                    f"{self.portal_url}/api/storage/memory/{key}",
                    params=params, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"delete failed: {resp.status_code}"}
                return "OK"

            elif action == "list":
                params = {"agentId": agent_id, "store": store, "inContext": "true"}
                if kwargs.get("prefix"):
                    params["prefix"] = kwargs["prefix"]
                if self._chat_id:
                    params["chatId"] = self._chat_id
                resp = await client.get(
                    f"{self.portal_url}/api/storage/memory",
                    params=params, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"list failed: {resp.status_code}"}
                return resp.json()

            elif action == "search":
                query = kwargs.get("query")
                if not query:
                    return {"error": "query is required"}
                params = {"action": "search", "agentId": agent_id, "q": query}
                if store and store != agent_id:
                    params["store"] = store
                if self._chat_id:
                    params["chatId"] = self._chat_id
                resp = await client.get(
                    f"{self.portal_url}/api/storage/memory",
                    params=params, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"search failed: {resp.status_code}"}
                return resp.json()

            elif action == "share":
                if not kwargs.get("agent"):
                    return {"error": "agent is required"}
                body = {
                    "agentId": agent_id,
                    "store": store,
                    "grantee": kwargs["agent"],
                    "level": kwargs.get("level", "read"),
                }
                resp = await client.post(
                    f"{self.portal_url}/api/storage/memory",
                    params={"action": "share"},
                    json=body, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"share failed: {resp.status_code}"}
                return "OK"

            elif action == "unshare":
                if not kwargs.get("agent"):
                    return {"error": "agent is required"}
                params = {
                    "action": "share",
                    "agentId": agent_id,
                    "store": store,
                    "grantee": kwargs["agent"],
                }
                resp = await client.delete(
                    f"{self.portal_url}/api/storage/memory",
                    params=params, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"unshare failed: {resp.status_code}"}
                return "OK"

            elif action == "stores":
                params = {"action": "stores", "agentId": agent_id}
                if self._chat_id:
                    params["chatId"] = self._chat_id
                if self._user_id:
                    params["userId"] = self._user_id
                resp = await client.get(
                    f"{self.portal_url}/api/storage/memory",
                    params=params, headers=headers,
                )
                if resp.status_code >= 400:
                    return {"error": f"stores failed: {resp.status_code}"}
                return resp.json()

            else:
                return {"error": f"Unknown action: {action}"}

    async def get_internal(self, store: str, key: str) -> Any:
        """Internal API: get a value (including inContext=false entries)"""
        agent_id = await self._resolve_agent_id()
        headers = await self._headers()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.portal_url}/api/storage/memory/{key}",
                params={"agentId": agent_id, "store": store},
                headers=headers,
            )
            if resp.status_code >= 400:
                return None
            return resp.json()

    async def set_internal(
        self, store: str, key: str, value: Any,
        encrypted: bool = False, ttl: int = 0,
    ) -> bool:
        """Internal API: set a value not exposed to LLM"""
        agent_id = await self._resolve_agent_id()
        headers = await self._headers()
        headers["Content-Type"] = "application/json"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(
                f"{self.portal_url}/api/storage/memory/{key}",
                json={
                    "value": value,
                    "agentId": agent_id,
                    "store": store,
                    "inContext": False,
                    "encrypted": encrypted,
                    "ttl": ttl,
                },
                headers=headers,
            )
            return resp.status_code < 400


# Backward compatibility
KVSkill = MemorySkill
