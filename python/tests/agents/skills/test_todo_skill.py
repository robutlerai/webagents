"""
The todo skill (2026-09-25): the TypeScript design, tool for tool, checked
against the fixture both SDKs run (`tests/fixtures/todo_tool/definitions.json`;
TypeScript: `tests/unit/skills/todo.test.ts`).
"""

import asyncio
import json
from pathlib import Path

from webagents.agents.skills.local.todo.skill import TodoSkill

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "todo_tool" / "definitions.json").read_text())


def test_the_definitions_and_guide_both_sdks_share():
    skill = TodoSkill({"agent_path": "/nonexistent"})
    for definition in FIXTURE["definitions"]:
        name = definition["function"]["name"]
        assert getattr(skill, name)._webagents_tool_definition == definition
    assert skill.todo_guide() == FIXTURE["prompt"]


def test_add_list_update_delete_in_the_agents_folder(tmp_path):
    skill = TodoSkill({"agent_path": str(tmp_path)})
    first = asyncio.run(skill.todo_add("Write the parser", priority="high", tags=["core"]))
    second = asyncio.run(skill.todo_add("Test it"))
    assert (first["id"], second["id"]) == ("todo-1", "todo-2")
    assert first["status"] == "pending" and second["priority"] == "medium"
    assert list(first) == ["id", "content", "status", "priority", "tags", "dependsOn", "createdAt", "updatedAt"]

    done = asyncio.run(skill.todo_update("todo-1", status="completed"))
    assert done["status"] == "completed" and done["completedAt"] == done["updatedAt"]
    assert [i["id"] for i in asyncio.run(skill.todo_list(status="pending"))] == ["todo-2"]
    assert [i["id"] for i in asyncio.run(skill.todo_list(tag="core"))] == ["todo-1"]
    assert asyncio.run(skill.todo_update("todo-9", status="completed")) == "Todo todo-9 not found"
    assert asyncio.run(skill.todo_delete("todo-2")) == "OK"
    assert asyncio.run(skill.todo_delete("todo-2")) == "Todo todo-2 not found"

    saved = json.loads((tmp_path / ".webagents" / "todos.json").read_text())
    assert [i["id"] for i in saved] == ["todo-1"]
    # A new skill reads the same list back, and numbering continues from it.
    again = TodoSkill({"agent_path": str(tmp_path)})
    assert asyncio.run(again.todo_add("Ship it"))["id"] == "todo-2"
