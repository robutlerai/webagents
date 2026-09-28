"""
The A2A task store: every task a served agent has answered or is working on,
per caller, with a TTL (plan item 1.3, 2026-09-26). The Python half of
`a2a/tasks.ts`.

PER CALLER, AS A RULE OF THE STORE. A caller reads, lists, subscribes to and
cancels only the tasks it created. The old store was one dict keyed by task
id, so any caller who learned an id could read another caller's conversation
with this agent, or cancel it. The owner key is part of every lookup, and a
miss for the wrong owner is a miss, never a 403: a 403 would confirm the id
exists.

Events are kept per task so `SubscribeToTask` can replay what a streaming
caller missed before attaching; they go with the task when it expires.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import is_settled, task_view

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


@dataclass
class TaskRecord:
    owner: str
    task: Dict[str, Any]
    created_at: float
    expires_at: float
    events: List[Dict[str, Any]] = field(default_factory=list)
    #: Wakes every subscriber when an event lands.
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    #: Set at a terminal or interrupted state.
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    #: The run in flight, when there is one.
    runner: Optional[asyncio.Task] = None


class TaskStore:
    def __init__(self, ttl_seconds: float, now=time.time):
        self._ttl = ttl_seconds
        self._now = now
        self._records: Dict[str, TaskRecord] = {}

    def __len__(self) -> int:
        self.sweep()
        return len(self._records)

    def create(self, owner: str, task: Dict[str, Any]) -> TaskRecord:
        self.sweep()
        now = self._now()
        record = TaskRecord(owner=owner, task=task, created_at=now, expires_at=now + self._ttl)
        self._records[task["id"]] = record
        return record

    def get(self, owner: str, task_id: str) -> Optional[TaskRecord]:
        """The record for `task_id` when `owner` created it; None otherwise."""
        self.sweep()
        record = self._records.get(task_id)
        return record if record is not None and record.owner == owner else None

    def list(
        self,
        owner: str,
        *,
        context_id: Optional[str] = None,
        status: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        history_length: Optional[int] = None,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """`owner`'s tasks, newest first, filtered and paged."""
        self.sweep()
        size = max(1, min(page_size or DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE))
        matching = [
            r
            for r in self._records.values()
            if r.owner == owner
            and (not context_id or r.task["contextId"] == context_id)
            and (not status or r.task["status"]["state"] == status)
        ]
        matching.sort(key=lambda r: r.created_at, reverse=True)
        try:
            offset = int(page_token) if page_token else 0
        except ValueError:
            offset = 0
        page = matching[offset : offset + size]
        next_token = str(offset + size) if offset + size < len(matching) else ""
        return {
            "tasks": [task_view(r.task, history_length=history_length, include_artifacts=include_artifacts) for r in page],
            "nextPageToken": next_token,
            "pageSize": size,
            "totalSize": len(matching),
        }

    def set_status(self, record: TaskRecord, status: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update the status, record the event, wake listeners; settles at a terminal or interrupted state."""
        record.task["status"] = status
        if metadata:
            record.task["metadata"] = {**(record.task.get("metadata") or {}), **metadata}
        self.emit(record, {"statusUpdate": {"taskId": record.task["id"], "contextId": record.task["contextId"], "status": status}})
        if is_settled(status["state"]):
            record.runner = None
            record.settled.set()

    def add_artifact_chunk(self, record: TaskRecord, artifact: Dict[str, Any], append: bool, last_chunk: bool) -> None:
        """Append an artifact chunk and record the event."""
        existing = next((a for a in record.task["artifacts"] if a["artifactId"] == artifact["artifactId"]), None)
        if existing is not None and append:
            existing["parts"].extend(artifact["parts"])
        elif existing is not None:
            existing["parts"] = list(artifact["parts"])
        else:
            record.task["artifacts"].append({**artifact, "parts": list(artifact["parts"])})
        self.emit(
            record,
            {
                "artifactUpdate": {
                    "taskId": record.task["id"],
                    "contextId": record.task["contextId"],
                    "artifact": artifact,
                    "append": append,
                    "lastChunk": last_chunk,
                }
            },
        )

    def add_history(self, record: TaskRecord, message: Dict[str, Any]) -> None:
        record.task["history"].append(message)

    def emit(self, record: TaskRecord, event: Dict[str, Any]) -> None:
        record.events.append(event)
        record.changed.set()
        record.changed = asyncio.Event()

    def sweep(self) -> None:
        """Drop every record past its TTL."""
        now = self._now()
        for task_id in [k for k, r in self._records.items() if r.expires_at <= now]:
            del self._records[task_id]

    def clear(self) -> None:
        self._records.clear()
