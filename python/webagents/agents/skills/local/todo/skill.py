"""
Todo Skill

Task tracking for multi-step work: `todo_add`, `todo_list`, `todo_update` and
`todo_delete`, kept in `.webagents/todos.json` in the agent's folder.

THE TYPESCRIPT SKILL, TOOL FOR TOOL (2026-09-25). This skill offered one
`write_todos(todos)` tool that replaced the whole list in memory, and the
TypeScript one four tools over a saved file, so an agent file naming `todo`
worked differently under each CLI. The TypeScript design is the reference
(`typescript/src/skills/todo/skill.ts`): the same tools, parameters, answers,
file and guide, checked against `tests/fixtures/todo_tool/definitions.json`.

THE LIST IS THE CALLER'S (S-294, 2026-09-26). It was one file for everyone:
every credentialed caller of a served agent read and changed the owner's
todos, and in the e2e run a stranger with an arbitrary bearer over
`webagents mcp serve --http` listed the todo the owner had made over stdio.
Whose list a turn sees is now decided the way the memory skill decides whose
notes it reads (`memory/memory_namespace.py`, `namespace_of`): the owner
keeps `.webagents/todos.json`; a verified caller (`user:`, `agent:`, `key:`,
`channel:`) has its own file under `.webagents/todos/callers/`, named by the
same caller key the memory and session skills use; a caller nothing verified
has no list, and every todo tool tells it so. Chosen over owner-only tools
because the list is a caller's working plan for the multi-step work it asked
for, which is as useful to a verified caller as to the owner, and because one
identity rule for memory, sessions and todos is one rule to get right. Pinned
by `todo_tool/definitions.json` (`caller_scope`), which both suites read.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...base import Skill
from ..memory.memory_namespace import OWNER_NAMESPACE, local_dir_of, namespace_of
from webagents.agents.tools.decorators import prompt, tool

#: The refusal every todo tool answers a caller nothing verified (the fixture's `caller_scope.refusal`).
TODO_NO_CALLER = "todo: nothing is kept for a caller nothing verified."

STATUSES = ("pending", "in_progress", "completed", "cancelled")
PRIORITIES = ("low", "medium", "high", "critical")

#: What the model is told: the TypeScript definitions, word for word.
DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "todo_add": {
        "type": "function",
        "function": {
            "name": "todo_add",
            "description": "Add a new todo item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "low",
                            "medium",
                            "high",
                            "critical"
                        ],
                        "description": "Priority (default: medium)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional tags"
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "IDs of tasks this depends on"
                    }
                },
                "required": [
                    "content"
                ]
            }
        }
    },
    "todo_list": {
        "type": "function",
        "function": {
            "name": "todo_list",
            "description": "List todo items, optionally filtered by status or tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "pending",
                            "in_progress",
                            "completed",
                            "cancelled"
                        ]
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag"
                    }
                }
            }
        }
    },
    "todo_update": {
        "type": "function",
        "function": {
            "name": "todo_update",
            "description": "Update a todo item (status, content, priority, tags).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Todo ID"
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "pending",
                            "in_progress",
                            "completed",
                            "cancelled"
                        ]
                    },
                    "content": {
                        "type": "string"
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "low",
                            "medium",
                            "high",
                            "critical"
                        ]
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "id"
                ]
            }
        }
    },
    "todo_delete": {
        "type": "function",
        "function": {
            "name": "todo_delete",
            "description": "Delete a todo item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Todo ID to delete"
                    }
                },
                "required": [
                    "id"
                ]
            }
        }
    }
}

GUIDE = "## Todo skill\n\nUse the `todo_*` tools for **multi-step work where tracking progress visibly helps the user** — refactors touching many files, multi-feature implementation plans, long-running investigations, anything where you'd otherwise lose context across iterations or tool calls.\n\n### When to use\n- 3+ distinct steps with dependencies between them.\n- Work that spans many tool calls and the user benefits from seeing what's done vs pending.\n- After receiving new instructions mid-task — capture the new requirements as todos so nothing is dropped.\n- When the task is complex enough that you might forget a step.\n\n### When NOT to use\n- Single-step tasks (just do them).\n- Trivial tasks completable in 1-2 obvious tool calls.\n- Purely conversational requests.\n- Don't add a \"test the change\" todo unless the user explicitly asked — it shifts focus toward testing over implementation.\n\n### Discipline\n- Update status in real time. Mark `in_progress` when you start, `completed` IMMEDIATELY after finishing — not in batches.\n- Only ONE task `in_progress` at a time. Finish it before starting another.\n- Break complex tasks into specific, actionable items. Vague todos like \"improve performance\" are noise.\n- Persisted to `.webagents/todos.json` in the working directory; survives across runs."


def _now() -> str:
    """An ISO timestamp as JavaScript's `toISOString` writes it."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TodoSkill(Skill):
    """Task tracking for multi-step work (module docstring)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, session: Any = None):
        # The loader passes the entry's config first; an older caller passed a
        # session there, which this skill no longer uses.
        if not isinstance(config, dict):
            config = None
        super().__init__(config or {})
        settings = config or {}
        folder = settings.get("agent_path") or os.getcwd()
        #: The owner's file; a caller's sits beside it under `todos/callers/`.
        self.file_path = Path(settings.get("filePath") or settings.get("file_path") or Path(folder) / ".webagents" / "todos.json")
        #: Each list that has been read, by namespace (module docstring).
        self.lists: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def items(self) -> List[Dict[str, Any]]:
        """The OWNER's list as it stands, for a transport that shows it as a
        plan (the ACP `plan` update; the editor's user is the owner)."""
        return self._load(OWNER_NAMESPACE)

    def _caller(self) -> Optional[str]:
        """Whose list this turn is: `owner`, `caller:<principal>`, or None for a caller nothing verified."""
        context = self.get_context()
        return namespace_of(getattr(context, "auth", None) if context is not None else None)

    def _file_for(self, namespace: str) -> Path:
        """Where a namespace's list lives: the owner's file, or `todos/callers/<key>.json` beside it."""
        if namespace == OWNER_NAMESPACE:
            return self.file_path
        return self.file_path.parent / "todos" / f"{local_dir_of(namespace)}.json"

    def _load(self, namespace: str) -> List[Dict[str, Any]]:
        cached = self.lists.get(namespace)
        if cached is not None:
            return cached
        try:
            parsed = json.loads(self._file_for(namespace).read_text("utf-8"))
            items = parsed if isinstance(parsed, list) else []
        except Exception:  # noqa: BLE001 - a missing or unreadable file is an empty list, as in TypeScript
            items = []
        self.lists[namespace] = items
        return items

    def _save(self, namespace: str, items: List[Dict[str, Any]]) -> None:
        file = self._file_for(namespace)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    @staticmethod
    def _next_id(items: List[Dict[str, Any]]) -> str:
        highest = 0
        for item in items:
            try:
                highest = max(highest, int(str(item.get("id", "")).replace("todo-", "")))
            except ValueError:
                continue
        return f"todo-{highest + 1}"

    @prompt(priority=50, scope="all")
    def todo_guide(self, context: Any = None) -> str:
        return GUIDE

    @tool(name="todo_add", description=DEFINITIONS["todo_add"]["function"]["description"])
    async def todo_add(self, content: str, priority: Optional[str] = None, tags: Optional[List[str]] = None,
                       depends_on: Optional[List[str]] = None) -> Dict[str, Any]:
        namespace = self._caller()
        if namespace is None:
            return {"error": TODO_NO_CALLER}
        items = self._load(namespace)
        now = _now()
        item = {
            "id": self._next_id(items),
            "content": content,
            "status": "pending",
            "priority": priority or "medium",
            "tags": tags or [],
            "dependsOn": depends_on or [],
            "createdAt": now,
            "updatedAt": now,
        }
        items.append(item)
        self._save(namespace, items)
        return item

    todo_add._webagents_tool_definition = DEFINITIONS["todo_add"]

    @tool(name="todo_list", description=DEFINITIONS["todo_list"]["function"]["description"])
    async def todo_list(self, status: Optional[str] = None, tag: Optional[str] = None) -> Any:
        namespace = self._caller()
        if namespace is None:
            return {"error": TODO_NO_CALLER}
        result = list(self._load(namespace))
        if status:
            result = [item for item in result if item.get("status") == status]
        if tag:
            result = [item for item in result if tag in (item.get("tags") or [])]
        return result

    todo_list._webagents_tool_definition = DEFINITIONS["todo_list"]

    @tool(name="todo_update", description=DEFINITIONS["todo_update"]["function"]["description"])
    async def todo_update(self, id: str, status: Optional[str] = None, content: Optional[str] = None,
                          priority: Optional[str] = None, tags: Optional[List[str]] = None) -> Any:
        namespace = self._caller()
        if namespace is None:
            return {"error": TODO_NO_CALLER}
        items = self._load(namespace)
        item = next((i for i in items if i.get("id") == id), None)
        if item is None:
            return f"Todo {id} not found"
        if status:
            item["status"] = status
        if content:
            item["content"] = content
        if priority:
            item["priority"] = priority
        if tags:
            item["tags"] = tags
        item["updatedAt"] = _now()
        if status == "completed":
            item["completedAt"] = item["updatedAt"]
        self._save(namespace, items)
        return item

    todo_update._webagents_tool_definition = DEFINITIONS["todo_update"]

    @tool(name="todo_delete", description=DEFINITIONS["todo_delete"]["function"]["description"])
    async def todo_delete(self, id: str) -> Any:
        namespace = self._caller()
        if namespace is None:
            return {"error": TODO_NO_CALLER}
        items = self._load(namespace)
        index = next((n for n, i in enumerate(items) if i.get("id") == id), None)
        if index is None:
            return f"Todo {id} not found"
        del items[index]
        self._save(namespace, items)
        return "OK"

    todo_delete._webagents_tool_definition = DEFINITIONS["todo_delete"]
