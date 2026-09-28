"""A stand-in for the platform's ``/api/storage/memory-scoped`` route (portal
``lib/storage/memory-scoped-service.ts``), for the memory tests: the same
actions, an in-memory store per (namespace, key), word-match search, and the
sync as the platform serves it since 2026-09-27 (S-307; the shared fixture
``tests/fixtures/memory_tool/migrations_trim_sync.json``):

- every change stamps the row with the SERVER's clock (microseconds, only
  growing), apart from the change's own ``at`` that the merge compares;
- a forget leaves a tombstone (no content) that the pull serves as a delete;
- a pull answers the rows stamped after ``since``, oldest first, a page at a
  time, with the last row's stamp as an opaque string cursor; anything that
  is not such a cursor pulls from the start;
- a push answers ``{applied}`` and no cursor.

It used to mint its own cursor (the log length) on a push, which is why the
SDK suites never saw the platform's S-307 cursor. Every request is recorded in
``seen``; ``rows`` holds the live entries only. The TypeScript twin is
``typescript/tests/unit/skills/memory/fake-portal-w2mem.ts``."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from webagents.agents.skills.local.memory.memory_namespace import entry_id_for

AGENT_ID = "7f3c2a10-5b6e-4c8d-9e1f-0a2b3c4d5e6f"
CURSOR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _stamp_text(us: int) -> str:
    return (EPOCH + timedelta(microseconds=us)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stamp_of(text: str) -> int:
    m = re.match(r"^(.{19})(?:\.(\d{1,6}))?Z$", text)
    assert m is not None
    whole = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return int((whole - EPOCH).total_seconds()) * 1_000_000 + int((m.group(2) or "").ljust(6, "0"))


class FakePortal:
    def __init__(self, agent_id: str = AGENT_ID, page_size: int = 500) -> None:
        self.agent_id = agent_id
        self.page_size = page_size
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.store: Dict[str, Dict[str, Any]] = {}
        self.seen: List[httpx.Request] = []
        self._clock = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        self._server_us = _stamp_of("2023-11-14T22:13:20Z")
        self.transport = httpx.MockTransport(self._handle)

    def _stamp(self) -> str:
        self._clock += timedelta(seconds=1)
        return self._clock.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _next_server_us(self) -> int:
        self._server_us += 1_000
        return self._server_us

    @staticmethod
    def _row_key(namespace: str, key: str) -> str:
        return f"{namespace}\n{key}"

    def put(self, namespace: str, key: str, content: str, source: str = "tool", at: Optional[str] = None) -> Dict[str, Any]:
        """A change made on the platform (the owner's panel, another machine)."""
        existing = self.store.get(self._row_key(namespace, key))
        when = at or self._stamp()
        row = {
            "id": entry_id_for(self.agent_id, namespace, key),
            "namespace": namespace,
            "key": key,
            "content": content,
            "source": source,
            "created_at": existing["row"]["created_at"] if existing and not existing["deleted"] else when,
            "updated_at": when,
        }
        self.store[self._row_key(namespace, key)] = {"row": row, "deleted": False, "stamp_us": self._next_server_us()}
        self.rows[self._row_key(namespace, key)] = row
        return row

    def delete(self, namespace: str, key: str, at: Optional[str] = None) -> int:
        existing = self.store.get(self._row_key(namespace, key))
        if existing is None or existing["deleted"]:
            return 0
        when = at or self._stamp()
        self.store[self._row_key(namespace, key)] = {"row": {**existing["row"], "content": "", "updated_at": when}, "deleted": True, "stamp_us": self._next_server_us()}
        self.rows.pop(self._row_key(namespace, key), None)
        return 1

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if request.headers.get("authorization") != "Bearer agent-key":
            return httpx.Response(401, json={"error": "Unauthorized"})
        q = dict(request.url.params)
        # Lists are repeated parameters (S-298); the retired comma-joined form is refused as the route refuses it.
        if "namespaces" in q or "excludeSources" in q:
            return httpx.Response(400, json={"error": "namespaces is not accepted; repeat namespace=<value> once per value"})
        namespaces = request.url.params.get_list("namespace") or None

        def within(row: Dict[str, Any]) -> bool:
            return namespaces is None or row["namespace"] in namespaces

        action = q.get("action")
        method = request.method
        body = json.loads(request.content) if request.content else {}
        if method == "GET" and action == "search":
            words = re.findall(r"[a-z0-9]+", (q.get("q") or "").lower())
            hits = [r for r in self.rows.values() if within(r) and any(w in f"{r['key']} {r['content']}".lower() for w in words)]
            return httpx.Response(200, json={"entries": hits[: int(q.get("limit", 10))]})
        if method == "GET" and action == "list":
            excluded = request.url.params.get_list("excludeSource")
            prefix = q.get("prefix") or ""
            entries = [r for r in self.rows.values() if within(r) and r["key"].startswith(prefix) and r["source"] not in excluded]
            entries.sort(key=lambda r: r["updated_at"], reverse=True)
            return httpx.Response(200, json={"entries": entries[: int(q.get("limit", 50))]})
        if method == "GET" and action == "get":
            return httpx.Response(200, json={"entry": self.rows.get(self._row_key(q["namespace"], q["key"]))})
        if method == "GET" and action == "sync":
            raw = q.get("since")
            since = raw if raw and CURSOR_RE.match(raw) and _stamp_of(raw) <= self._server_us else None
            changed = sorted(
                (s for s in self.store.values() if s["row"]["namespace"] == q.get("namespace") and (since is None or s["stamp_us"] > _stamp_of(since))),
                key=lambda s: s["stamp_us"],
            )
            page = changed[: self.page_size]
            lines = []
            for s in page:
                r = s["row"]
                if s["deleted"]:
                    lines.append({"op": "delete", "id": r["id"], "namespace": r["namespace"], "key": r["key"], "at": r["updated_at"]})
                else:
                    lines.append({"op": "put", "id": r["id"], "namespace": r["namespace"], "key": r["key"], "content": r["content"], "source": r["source"], "at": r["updated_at"]})
            cursor = _stamp_text(page[-1]["stamp_us"]) if page else since
            return httpx.Response(200, json={"lines": lines, "cursor": cursor, "more": len(changed) > self.page_size})
        if method == "PUT":
            return httpx.Response(200, json={"entry": self.put(body["namespace"], body["key"], body["content"], body.get("source") or "tool", body.get("at"))})
        if method == "DELETE":
            return httpx.Response(200, json={"forgotten": self.delete(q["namespace"], q["key"])})
        if method == "POST" and action == "sync":
            applied = 0
            for line in body["lines"]:
                existing = self.store.get(self._row_key(line["namespace"], line["key"]))
                if existing is not None and existing["row"]["updated_at"] >= line["at"]:
                    continue
                if line["op"] == "put":
                    source = line.get("source") or "tool"
                    self.put(line["namespace"], line["key"], line.get("content") or "", "tool" if source == "sync" else source, line["at"])
                    applied += 1
                else:
                    applied += self.delete(line["namespace"], line["key"], line["at"])
            return httpx.Response(200, json={"applied": applied})
        return httpx.Response(400, json={"error": "Unknown action"})
