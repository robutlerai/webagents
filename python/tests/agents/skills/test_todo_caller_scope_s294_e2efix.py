"""
S-294 (2026-09-26): the todo list is the CALLER's. It was one file for
everyone, so a stranger with an arbitrary bearer over `mcp serve --http`
listed the todo the owner had made over stdio. Whose list a turn sees now
follows the memory skill's namespace rule, pinned by `caller_scope` in
`tests/fixtures/todo_tool/definitions.json`, which the TypeScript suite runs
too (`tests/unit/skills/todo-caller-scope-s294-e2efix.test.ts`): two verified
callers and the owner each keep their own list, a caller nothing verified gets
the refusal from every tool, and the ACP plan reads the owner's list.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from webagents.agents.skills.local.memory.memory_namespace import caller_key, namespace_of
from webagents.agents.skills.local.todo.skill import TODO_NO_CALLER, TodoSkill
from webagents.server.context.context_vars import create_context, set_context

SCOPE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "todo_tool" / "definitions.json").read_text())["caller_scope"]


def _auth(raw: Optional[Dict[str, Any]]) -> Any:
    return SimpleNamespace(**raw) if isinstance(raw, dict) else None


def _by_case(name: str) -> Dict[str, Any]:
    return next(c for c in SCOPE["callers"] if c["case"] == name)


def _as(raw: Optional[Dict[str, Any]]) -> None:
    """Make the current turn this caller's, as a served request or the local chat would."""
    context = create_context(messages=[], stream=False, agent=None)
    context.auth = _auth(raw)
    set_context(context)


def _run(coro):
    return asyncio.run(coro)


def test_the_refusal_and_the_namespace_of_every_fixture_caller_match():
    assert TODO_NO_CALLER == SCOPE["refusal"]
    for case in SCOPE["callers"]:
        assert namespace_of(_auth(case["auth"])) == case["namespace"], case["case"]
        if case.get("caller_key"):
            assert caller_key(case["namespace"][len("caller:"):]) == case["caller_key"]


def test_two_verified_callers_and_the_owner_each_keep_their_own_list(tmp_path):
    skill = TodoSkill({"file_path": str(tmp_path / SCOPE["owner_file"])})
    owner, alice, bob = _by_case("the owner"), _by_case("a platform credential's user"), _by_case("another platform user")

    _as(owner["auth"])
    _run(skill.todo_add("owner: pay the rent"))
    _as(alice["auth"])
    alice_first = _run(skill.todo_add("alice: buy milk"))
    _as(bob["auth"])
    _run(skill.todo_add("bob: call mum"))

    def contents(who):
        _as(who["auth"])
        return [i["content"] for i in _run(skill.todo_list())]

    assert contents(owner) == ["owner: pay the rent"]
    assert contents(alice) == ["alice: buy milk"]
    assert contents(bob) == ["bob: call mum"]
    # Ids are per list: alice's first is todo-1 too.
    assert alice_first["id"] == "todo-1"
    # Bob cannot reach alice's item by id, and cannot touch the owner's.
    _as(bob["auth"])
    assert _run(skill.todo_delete("todo-1")) == "OK"
    assert contents(alice) == ["alice: buy milk"]
    assert contents(owner) == ["owner: pay the rent"]

    assert (tmp_path / SCOPE["owner_file"]).exists()
    alice_file = tmp_path / SCOPE["caller_file"].replace("{caller_key}", alice["caller_key"])
    assert [i["content"] for i in json.loads(alice_file.read_text())] == ["alice: buy milk"]
    assert len(json.loads((tmp_path / SCOPE["owner_file"]).read_text())) == 1


def test_a_caller_nothing_verified_gets_the_refusal_from_every_tool(tmp_path):
    skill = TodoSkill({"file_path": str(tmp_path / SCOPE["owner_file"])})
    for name in ("a bearer nothing verified", "anonymous"):
        _as(_by_case(name)["auth"])
        assert _run(skill.todo_add("x")) == {"error": SCOPE["refusal"]}
        assert _run(skill.todo_list()) == {"error": SCOPE["refusal"]}
        assert _run(skill.todo_update("todo-1", status="completed")) == {"error": SCOPE["refusal"]}
        assert _run(skill.todo_delete("todo-1")) == {"error": SCOPE["refusal"]}
    assert not (tmp_path / ".webagents").exists()


def test_the_plan_a_transport_shows_is_the_owners_list(tmp_path):
    skill = TodoSkill({"file_path": str(tmp_path / SCOPE["owner_file"])})
    _as(_by_case("a platform credential's user")["auth"])
    _run(skill.todo_add("alice: buy milk"))
    _as(_by_case("the owner")["auth"])
    _run(skill.todo_add("owner: pay the rent"))
    assert [i["content"] for i in skill.items] == ["owner: pay the rent"]
    # A fresh skill over the same folder reads the owner's file for the plan.
    again = TodoSkill({"file_path": str(tmp_path / SCOPE["owner_file"])})
    assert [i["content"] for i in again.items] == ["owner: pay the rent"]
