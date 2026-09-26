"""
RobutlerJSONSkill - JSON documents an agent keeps on the platform.

THE ROUTES AN AGENT'S KEY CAN USE (2026-09-25). A document is uploaded with
`POST /api/content/upload`, then found, read and removed through the agent's
own content routes, `/api/agents/{agent_id}/content[/{id}]`, all through the
platform API client (`../../api/client.py`). The skill used the client's older
content methods, which called routes that take only a browser session or do
not exist, so reading, updating and deleting never worked. It also decided the
caller's scope from a context key nothing sets, so every document was stored
public, readable by anyone with its link (S-255). Documents are private now:
every tool here is the owner's.

WHOSE FILES. The key is the agent's own (`resolve_agent_token`: an `api_key`
on the agent, `WEBAGENTS_AGENT_TOKEN`, or the key `webagents publish` stored),
then `WEBAGENTS_API_KEY`, and the agent's platform id is that key's `agent_id`
claim; `api_key` and `agent_id` in the config come first. An owner's key files
documents in the owner's library, where the agent routes do not look.

A NAME IS NOT UNIQUE on the platform. Storing a name that exists adds another
file, and reading takes the newest; `update_json_data` stores the new document
and then removes the older ones. Removing takes away the agent's link: a
document the agent stored is also linked into its owner's library and stays
there.
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from webagents.agents.skills.base import Skill
from webagents.agents.skills.robutler.api.client import RobutlerClient
from webagents.agents.skills.robutler.platform_url import resolve_platform_url
from webagents.agents.tools.decorators import tool


def _json_name(filename: str) -> str:
    """The stored name: `filename`, with `.json` added when it has none."""
    name = str(filename or "").strip()
    return name if name.endswith(".json") else f"{name}.json"


def _agent_id_in(token: Optional[str]) -> Optional[str]:
    """The `agent_id` claim of an agent's own platform key, read without
    verifying it: it only names whose files to ask for, and the platform checks
    the key itself on every call."""
    try:
        payload = str(token).split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        value = claims.get("agent_id") if isinstance(claims, dict) else None
        return value if isinstance(value, str) and value else None
    except Exception:
        return None


def _failure(message: str, **extra: Any) -> str:
    return json.dumps({"success": False, "error": message, **extra})


class RobutlerJSONSkill(Skill):
    """
    JSON documents for an agent's long-term memory, private on the platform.

    Tools (owner scope only):
    - store_json_data: store a document under a name
    - retrieve_json_data: read the newest document with that name
    - update_json_data: store a new version and remove the older ones
    - delete_json_file: remove every document with that name
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        # The SDK's one platform lookup (`../../platform_url.py`), after this
        # skill's own `portal_url`.
        self.portal_url = cfg.get('portal_url') or resolve_platform_url(cfg)
        # No placeholder key: a fake one produced a silently unauthenticated
        # client whose first visible failure was never the real one (F-041).
        self.api_key: Optional[str] = cfg.get('api_key')
        self.agent_id: Optional[str] = cfg.get('agent_id')
        self.client: Optional[RobutlerClient] = None
        self.agent = None

    async def initialize(self, agent_reference):
        """Find the agent's key and id (module docstring) and open the client."""
        await super().initialize(agent_reference)
        self.agent = agent_reference
        if not self.api_key:
            from webagents.server.core.registration import resolve_agent_token

            self.api_key = resolve_agent_token(agent_reference) or os.getenv('WEBAGENTS_API_KEY')
        if not self.agent_id:
            self.agent_id = _agent_id_in(self.api_key)
        if self.api_key:
            self.client = RobutlerClient(api_key=self.api_key, base_url=self.portal_url)

    async def cleanup(self):
        """Close the client's session."""
        if self.client:
            await self.client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ready(self) -> Tuple[RobutlerClient, str]:
        """The client and the agent's id, or a ValueError naming what is missing."""
        if not self.client:
            raise ValueError(
                "No platform key for this agent. Publish it with `webagents publish`, "
                "or set `api_key` in the skill's config."
            )
        if not self.agent_id:
            raise ValueError(
                "This agent's platform id is not known: its key names none. "
                "Set `agent_id` in the skill's config."
            )
        return self.client, self.agent_id

    async def _named(self, name: str) -> List[Dict[str, Any]]:
        """The agent's documents called exactly `name`, newest first."""
        client, agent_id = self._ready()
        response = await client.list_agent_content(agent_id, query=name)
        if not response.success:
            raise ValueError(f"Could not list this agent's documents: {response.message}")
        items = [i for i in (response.data or {}).get('items', []) if i.get('displayName') == name]
        return sorted(items, key=lambda i: str(i.get('createdAt') or ''), reverse=True)

    async def _json_names(self) -> List[str]:
        """The names of the agent's JSON documents, for a helpful miss."""
        client, agent_id = self._ready()
        response = await client.list_agent_content(agent_id)
        items = (response.data or {}).get('items', []) if response.success else []
        return sorted({str(i.get('displayName')) for i in items if str(i.get('displayName') or '').endswith('.json')})

    async def _store(self, name: str, data: Any) -> Dict[str, Any]:
        client, _ = self._ready()
        payload = json.dumps(data, indent=2).encode('utf-8')
        response = await client.upload_file(name, payload, content_type='application/json', visibility='private')
        if not response.success:
            raise ValueError(f"Could not store {name}: {response.message}")
        return response.data or {}

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(scope="owner")
    async def store_json_data(self, filename: str, data: Dict[str, Any]) -> str:
        """
        Store JSON data for long-term memory.

        Args:
            filename: Name of the document (.json is added when missing)
            data: JSON-serializable data to store

        Returns:
            JSON string with the stored document's id, name and size
        """
        try:
            name = _json_name(filename)
            info = await self._store(name, data)
            return json.dumps({
                "success": True,
                "file_id": info.get('id'),
                "filename": info.get('displayName') or name,
                "size": info.get('size'),
            }, indent=2)
        except Exception as e:
            return _failure(f"Failed to store JSON data: {e}")

    @tool(scope="owner")
    async def retrieve_json_data(self, filename: str) -> str:
        """
        Retrieve JSON data from long-term memory.

        Args:
            filename: Name of the document to read

        Returns:
            JSON string with the document's data, or the names that exist
        """
        try:
            name = _json_name(filename)
            found = await self._named(name)
            if not found:
                return _failure(f"No JSON document named {name}", available_json_files=await self._json_names())
            client, agent_id = self._ready()
            newest = found[0]
            response = await client.read_agent_content(agent_id, newest['id'], format='text')
            if not response.success:
                return _failure(f"Could not read {name}: {response.message}")
            text = (response.data or {}).get('text')
            return json.dumps({
                "success": True,
                "filename": name,
                "data": json.loads(text) if text else None,
                "metadata": {"file_id": newest.get('id'), "size": newest.get('size'), "created_at": newest.get('createdAt')},
            }, indent=2)
        except Exception as e:
            return _failure(f"Failed to retrieve JSON data: {e}")

    @tool(scope="owner")
    async def update_json_data(self, filename: str, data: Dict[str, Any]) -> str:
        """
        Update existing JSON data in long-term memory.

        Args:
            filename: Name of the document to update
            data: The new JSON data

        Returns:
            JSON string with the new document's id, name and size
        """
        try:
            name = _json_name(filename)
            older = await self._named(name)
            info = await self._store(name, data)
            # Only once the new version is safely stored.
            client, agent_id = self._ready()
            for item in older:
                if item.get('id') != info.get('id'):
                    await client.delete_agent_content(agent_id, item['id'])
            return json.dumps({
                "success": True,
                "file_id": info.get('id'),
                "filename": info.get('displayName') or name,
                "size": info.get('size'),
                "replaced": len(older),
            }, indent=2)
        except Exception as e:
            return _failure(f"Failed to update JSON data: {e}")

    @tool(scope="owner")
    async def delete_json_file(self, filename: str) -> str:
        """
        Delete a JSON document from long-term memory.

        Args:
            filename: Name of the document to delete

        Returns:
            JSON string with the result
        """
        try:
            name = _json_name(filename)
            found = await self._named(name)
            if not found:
                return _failure(f"No JSON document named {name}")
            client, agent_id = self._ready()
            for item in found:
                response = await client.delete_agent_content(agent_id, item['id'])
                if not response.success:
                    return _failure(f"Could not delete {name}: {response.message}")
            return json.dumps({"success": True, "message": f"JSON document '{name}' deleted", "deleted": len(found)})
        except Exception as e:
            return _failure(f"Failed to delete JSON file: {e}")

    def get_skill_info(self) -> Dict[str, Any]:
        """Skill information; `tools` is derived from the live registry by the
        base class so the advertised and registered surfaces cannot drift."""
        info = super().get_skill_info()
        info.update({
            "name": "RobutlerJSONSkill",
            "description": "Private JSON documents for an agent's long-term memory",
            "capabilities": [
                "Store, read, update and delete JSON documents (owner scope only)",
                "Documents are private to the agent and its owner",
                "The agent is the one its own platform key names",
            ],
            "config": {
                "portal_url": self.portal_url,
                "api_key_configured": bool(self.api_key),
                "agent_id": self.agent_id,
            },
        })
        return info
