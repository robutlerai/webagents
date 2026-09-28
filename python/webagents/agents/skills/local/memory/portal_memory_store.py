"""
The portal memory tier, as the SDK reaches it (gap-closure plan item 2.1,
principle 5, 2026-09-26): the platform's memory store with per-caller
namespaces, semantic search on its embedding and Milvus stack, and the
sync. One route, ``/api/storage/memory-scoped`` (portal
``lib/storage/memory-scoped-service.ts``), authenticated as the AGENT with its
own key (``WEBAGENTS_AGENT_TOKEN``, or the key ``publish`` stored), never as a
caller: the platform trusts the agent about which of its callers an entry
belongs to, because only the agent verified them, and never trusts one agent
about another agent's store (S-257).

Every method raises a ``MemoryPortalError`` whose message starts with
``memory:`` when the platform cannot be reached or refuses; the skill turns
that into a tool result and keeps the local tier working. The TypeScript twin
is ``typescript/src/skills/memory/portal-store.ts``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

import httpx

from .local_memory_store import MemoryEntry

TokenSource = Callable[[], Union[Optional[str], Awaitable[Optional[str]]]]


class MemoryPortalError(Exception):
    """The platform could not be reached, or refused."""


def _only_within(entries: List[MemoryEntry], namespaces: Optional[Sequence[str]]) -> List[MemoryEntry]:
    """Only entries of the namespaces asked for (S-298): what the platform answers is checked, not trusted."""
    if namespaces is None:
        return entries
    allowed = set(namespaces)
    return [e for e in entries if e.namespace in allowed]


def _from_wire(e: Dict[str, Any]) -> MemoryEntry:
    return MemoryEntry(
        id=str(e.get("id", "")),
        namespace=str(e.get("namespace", "")),
        key=str(e.get("key", "")),
        content=str(e.get("content", "")),
        source=str(e.get("source") or "tool"),
        created_at=str(e.get("created_at", "")),
        updated_at=str(e.get("updated_at", "")),
    )


class PortalMemoryStore:
    def __init__(
        self,
        portal_url: str,
        token: TokenSource,
        agent_id: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = 20.0,
    ) -> None:
        self.portal_url = portal_url.rstrip("/")
        self._token = token
        self._agent_id = agent_id
        self._transport = transport
        self._timeout = timeout

    async def _request(self, method: str, query: Dict[str, Union[Optional[str], Sequence[str]]], body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """A list value is sent as a REPEATED parameter, one value each, never
        comma-joined (S-298, 2026-09-26): a caller principal may carry a
        comma, and a joined list would be split on the portal into a
        namespace the caller was never given."""
        token = self._token()
        if hasattr(token, "__await__"):
            token = await token  # type: ignore[misc]
        if not token:
            raise MemoryPortalError("memory: no platform credential for this agent (set WEBAGENTS_AGENT_TOKEN or publish the agent).")
        params: List[Tuple[str, str]] = []
        for k, v in query.items():
            if not v:
                continue
            if isinstance(v, str):
                params.append((k, v))
            else:
                params.extend((k, item) for item in v)
        if self._agent_id:
            params.append(("agentId", self._agent_id))
        payload = None
        if body is not None:
            payload = {**({"agentId": self._agent_id} if self._agent_id else {}), **body}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.request(
                    method,
                    f"{self.portal_url}/api/storage/memory-scoped",
                    params=params,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as exc:
            raise MemoryPortalError(f"memory: the platform could not be reached ({exc}).") from None
        if response.status_code >= 400:
            detail = ""
            try:
                detail = str(response.json().get("error", ""))
            except ValueError:
                pass
            raise MemoryPortalError(f"memory: the platform answered {response.status_code}" + (f" ({detail})" if detail else "") + ".")
        try:
            return response.json()
        except ValueError:
            raise MemoryPortalError("memory: the platform answered with something that is not JSON.") from None

    async def get(self, namespace: str, key: str) -> Optional[MemoryEntry]:
        data = await self._request("GET", {"action": "get", "namespace": namespace, "key": key})
        return _from_wire(data["entry"]) if data.get("entry") else None

    async def list(
        self,
        namespaces: Optional[Sequence[str]],
        prefix: Optional[str] = None,
        limit: int = 50,
        exclude_sources: Sequence[str] = (),
    ) -> List[MemoryEntry]:
        data = await self._request(
            "GET",
            {
                "action": "list",
                "namespace": list(namespaces) if namespaces is not None else None,
                "prefix": prefix,
                "limit": str(limit),
                "excludeSource": list(exclude_sources) if exclude_sources else None,
            },
        )
        return _only_within([_from_wire(e) for e in data.get("entries", [])], namespaces)

    async def search(self, query: str, namespaces: Optional[Sequence[str]], limit: int = 10) -> List[MemoryEntry]:
        data = await self._request(
            "GET",
            {"action": "search", "q": query, "namespace": list(namespaces) if namespaces is not None else None, "limit": str(limit)},
        )
        return _only_within([_from_wire(e) for e in data.get("entries", [])], namespaces)

    async def put(self, namespace: str, key: str, content: str, source: str = "tool", at: Optional[str] = None) -> MemoryEntry:
        data = await self._request("PUT", {}, {"namespace": namespace, "key": key, "content": content, "source": source, "at": at})
        return _from_wire(data["entry"])

    async def forget(self, namespace: str, key: str) -> bool:
        data = await self._request("DELETE", {"namespace": namespace, "key": key})
        return int(data.get("forgotten", 0)) > 0

    async def pull(self, namespace: str, since: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """One page of what changed on the platform in a namespace after ``since``
        (None: from the start), the cursor to ask from next, and whether another
        page waits. The cursor is the platform's own string, kept and sent back
        as it came (S-307, 2026-09-27): it is never compared or computed here."""
        data = await self._request("GET", {"action": "sync", "namespace": namespace, "since": since})
        lines = data.get("lines")
        cursor = data.get("cursor")
        return (
            list(lines) if isinstance(lines, list) else [],
            cursor if isinstance(cursor, str) and cursor else None,
            data.get("more") is True,
        )

    async def push(self, namespace: str, lines: Sequence[Dict[str, Any]]) -> int:
        """This machine's changes for a namespace, merged on the platform by
        entry id; answers how many landed. No cursor comes back: a push never
        moves a pull cursor (a push answering one was S-307, which stopped
        every later pull)."""
        wire = [
            {k: l.get(k) for k in ("op", "id", "namespace", "key", "content", "source", "at") if l.get(k) is not None}
            for l in lines
        ]
        data = await self._request("POST", {"action": "sync"}, {"namespace": namespace, "lines": wire})
        return int(data.get("applied", 0))
