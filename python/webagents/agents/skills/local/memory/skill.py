"""
LocalMemorySkill - SQLite-backed memory for standalone agents.

Same tool interface as MemorySkill (portal-backed) but stores data
locally via aiosqlite. No portal dependency required.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from webagents.agents.skills.base import Skill


class LocalMemorySkill(Skill):
    """SQLite-backed agent memory for standalone deployment."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        agent_id: str = "default",
        storage_path: str = "./.webagents/memory.db",
        context_stores: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        super().__init__(config or {}, scope="all")
        self._agent_id = agent_id
        self._storage_path = storage_path
        self._context_stores = context_stores or []
        self._db = None

    def initialize(self, agent: Any) -> None:
        super().initialize(agent)
        self._register_memory_tool()

    async def _ensure_db(self):
        if self._db is not None:
            return
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "LocalMemorySkill requires aiosqlite. Install it: pip install aiosqlite"
            )
        os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._storage_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                store_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT '',
                key TEXT NOT NULL,
                value TEXT,
                in_context INTEGER NOT NULL DEFAULT 1,
                encrypted INTEGER NOT NULL DEFAULT 0,
                ttl INTEGER DEFAULT 0,
                expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(store_id, owner_id, namespace, key)
            );
            CREATE INDEX IF NOT EXISTS memory_store_idx ON memory(store_id);

            CREATE TABLE IF NOT EXISTS memory_grants (
                store_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'read',
                granted_by TEXT NOT NULL,
                expires_at TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(store_id, agent_id)
            );
        """)

    def _register_memory_tool(self) -> None:
        store_lines = [f"- {self._agent_id} (self): Your persistent memory"]
        for s in self._context_stores:
            store_lines.append(f"- {s['storeId']} ({s['label']}): {s['label']} memory")

        description = (
            "Persistent memory. Store and retrieve information across conversations.\n"
            "\nAvailable stores:\n" + "\n".join(store_lines) + "\n"
            "\nActions:\n"
            "- get(store, key): retrieve a stored value\n"
            "- set(store, key, value, ttl?): store a value\n"
            "- delete(store, key): remove a key\n"
            "- list(store, prefix?): list keys\n"
            "- search(query, store?): full-text search\n"
            "- share(store, agent, level?): grant access\n"
            "- unshare(store, agent): revoke grant\n"
            "- stores(): list accessible stores"
        )

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
            return await self._handle(
                action=action, store=store or self._agent_id,
                key=key, value=value, ttl=ttl or 0,
                prefix=prefix, query=query, agent_target=agent, level=level,
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
                            "enum": ["get", "set", "delete", "list", "search",
                                     "share", "unshare", "stores"],
                        },
                        "store": {"type": "string"},
                        "key": {"type": "string"},
                        "value": {},
                        "ttl": {"type": "number"},
                        "prefix": {"type": "string"},
                        "query": {"type": "string"},
                        "agent": {"type": "string"},
                        "level": {"type": "string", "enum": ["search", "read", "readwrite"]},
                    },
                    "required": ["action"],
                },
            },
        }
        self.register_tool(memory_handler, scope="all")

    def _can_access(self, agent_id: str, store_id: str) -> Dict[str, Any]:
        if store_id == agent_id:
            return {"allowed": True, "level": "readwrite"}
        return {"allowed": False, "level": "search"}

    async def _can_access_async(self, agent_id: str, store_id: str) -> Dict[str, Any]:
        if store_id == agent_id:
            return {"allowed": True, "level": "readwrite"}
        await self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        async with self._db.execute(
            "SELECT level FROM memory_grants WHERE store_id=? AND agent_id=? AND (expires_at IS NULL OR expires_at>?)",
            (store_id, agent_id, now),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return {"allowed": True, "level": row[0]}
        return {"allowed": False, "level": "search"}

    def _has_level(self, actual: str, required: str) -> bool:
        rank = {"search": 0, "read": 1, "readwrite": 2}
        return rank.get(actual, -1) >= rank.get(required, 99)

    async def _handle(self, **kw: Any) -> Any:
        await self._ensure_db()
        action = kw["action"]
        store = kw["store"]
        agent_id = self._agent_id
        now = datetime.now(timezone.utc).isoformat()

        if action == "get":
            if not kw.get("key"):
                return {"error": "key is required"}
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "read"):
                return {"error": "Access denied"}
            async with self._db.execute(
                "SELECT key, value, namespace, owner_id, in_context FROM memory WHERE store_id=? AND key=? AND (expires_at IS NULL OR expires_at>?)",
                (store, kw["key"], now),
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return None
            return {
                "key": row[0], "value": json.loads(row[1]) if row[1] else None,
                "namespace": row[2], "ownerId": row[3], "inContext": bool(row[4]),
            }

        elif action == "set":
            if not kw.get("key"):
                return {"error": "key is required"}
            if kw.get("value") is None:
                return {"error": "value is required"}
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "readwrite"):
                return {"error": "Access denied"}
            ttl = kw.get("ttl", 0)
            expires_at = (
                datetime.fromtimestamp(
                    datetime.now(timezone.utc).timestamp() + ttl, tz=timezone.utc
                ).isoformat()
                if ttl > 0
                else None
            )
            import uuid as _uuid
            await self._db.execute(
                """INSERT INTO memory (id, store_id, owner_id, key, value, ttl, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(store_id, owner_id, namespace, key) DO UPDATE SET
                     value=excluded.value, ttl=excluded.ttl, expires_at=excluded.expires_at,
                     updated_at=datetime('now')""",
                (str(_uuid.uuid4()), store, agent_id, kw["key"], json.dumps(kw["value"]), ttl, expires_at),
            )
            await self._db.commit()
            return "OK"

        elif action == "delete":
            if not kw.get("key"):
                return {"error": "key is required"}
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "readwrite"):
                return {"error": "Access denied"}
            await self._db.execute(
                "DELETE FROM memory WHERE store_id=? AND owner_id=? AND key=?",
                (store, agent_id, kw["key"]),
            )
            await self._db.commit()
            return "OK"

        elif action == "list":
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "read"):
                return {"error": "Access denied"}
            q = "SELECT key, value, namespace, owner_id, in_context FROM memory WHERE store_id=? AND in_context=1 AND (expires_at IS NULL OR expires_at>?)"
            args: list = [store, now]
            if kw.get("prefix"):
                q += " AND key LIKE ?"
                args.append(f"{kw['prefix']}%")
            q += " LIMIT 1000"
            async with self._db.execute(q, args) as cur:
                rows = await cur.fetchall()
            return {
                "entries": [
                    {"key": r[0], "value": json.loads(r[1]) if r[1] else None,
                     "namespace": r[2], "ownerId": r[3], "inContext": bool(r[4])}
                    for r in rows
                ]
            }

        elif action == "search":
            if not kw.get("query"):
                return {"error": "query is required"}
            async with self._db.execute(
                "SELECT store_id, key, value, namespace, owner_id FROM memory WHERE (key LIKE ? OR value LIKE ?) AND encrypted=0 AND (expires_at IS NULL OR expires_at>?) LIMIT 50",
                (f"%{kw['query']}%", f"%{kw['query']}%", now),
            ) as cur:
                rows = await cur.fetchall()
            results = []
            for r in rows:
                acc = await self._can_access_async(agent_id, r[0])
                if acc["allowed"]:
                    results.append({
                        "storeId": r[0], "key": r[1],
                        "value": json.loads(r[2]) if r[2] else None,
                        "namespace": r[3], "ownerId": r[4],
                    })
            return {"entries": results}

        elif action == "share":
            if not kw.get("agent_target"):
                return {"error": "agent is required"}
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "readwrite"):
                return {"error": "Access denied"}
            lvl = kw.get("level", "read")
            await self._db.execute(
                """INSERT INTO memory_grants (store_id, agent_id, level, granted_by)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(store_id, agent_id) DO UPDATE SET
                     level=excluded.level, granted_by=excluded.granted_by""",
                (store, kw["agent_target"], lvl, agent_id),
            )
            await self._db.commit()
            return "OK"

        elif action == "unshare":
            if not kw.get("agent_target"):
                return {"error": "agent is required"}
            access = await self._can_access_async(agent_id, store)
            if not access["allowed"] or not self._has_level(access["level"], "readwrite"):
                return {"error": "Access denied"}
            await self._db.execute(
                "DELETE FROM memory_grants WHERE store_id=? AND agent_id=?",
                (store, kw["agent_target"]),
            )
            await self._db.commit()
            return "OK"

        elif action == "stores":
            stores = [{"storeId": agent_id, "level": "readwrite", "source": "self"}]
            async with self._db.execute(
                "SELECT store_id, level FROM memory_grants WHERE agent_id=? AND (expires_at IS NULL OR expires_at>?)",
                (agent_id, now),
            ) as cur:
                for row in await cur.fetchall():
                    stores.append({"storeId": row[0], "level": row[1], "source": "grant"})
            for s in self._context_stores:
                stores.append({"storeId": s["storeId"], "level": "readwrite", "source": s["label"]})
            return {"stores": stores}

        return {"error": f"Unknown action: {action}"}

    async def get_internal(self, store: str, key: str) -> Any:
        await self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        async with self._db.execute(
            "SELECT value FROM memory WHERE store_id=? AND key=? AND owner_id=? AND (expires_at IS NULL OR expires_at>?)",
            (store, key, self._agent_id, now),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return json.loads(row[0]) if row[0] else None

    async def set_internal(self, store: str, key: str, value: Any,
                           encrypted: bool = False, ttl: int = 0) -> bool:
        await self._ensure_db()
        import uuid as _uuid
        expires_at = (
            datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + ttl, tz=timezone.utc
            ).isoformat()
            if ttl > 0
            else None
        )
        await self._db.execute(
            """INSERT INTO memory (id, store_id, owner_id, key, value, in_context, encrypted, ttl, expires_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
               ON CONFLICT(store_id, owner_id, namespace, key) DO UPDATE SET
                 value=excluded.value, encrypted=excluded.encrypted, ttl=excluded.ttl,
                 expires_at=excluded.expires_at, updated_at=datetime('now')""",
            (str(_uuid.uuid4()), store, self._agent_id, key, json.dumps(value), int(encrypted), ttl, expires_at),
        )
        await self._db.commit()
        return True

    async def cleanup(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
