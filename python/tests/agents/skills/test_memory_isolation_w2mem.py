"""THE RELEASE GATE for the memory skill (plan item 2.1, principle 1): one
caller's `memory_search` never returns another caller's entries. Pinned
through the skill's own tools over contexts shaped like served turns, on the
local tier and on the portal tier (a fake platform behind `httpx`), and
through a real `BaseAgent.execute_tool` so the tool sees the caller the
context carries. The TypeScript twin is
`typescript/tests/unit/skills/memory/memory-isolation-w2mem.test.ts`."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from webagents.agents.skills.local.memory.caller_scoped import NO_CALLER, NOT_YOURS, MemorySkill
from webagents.server.context.context_vars import create_context, set_context

from fake_portal_w2mem import FakePortal

OWNER = SimpleNamespace(authenticated=True, provider="platform", scope="owner", user_id="owner-1")
ALICE = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="alice")
BOB = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="bob")
FINDER = SimpleNamespace(authenticated=True, provider="signature", scope="user", principals=["agent:https://a.example/finder", "key:k1"])
NOBODY = SimpleNamespace(authenticated=False)


class FakeAgent:
    name = "helper"
    skills = {}

    def register_tool(self, fn, source=None, scope=None):
        pass


def as_caller(auth, metadata=None):
    context = create_context(messages=[])
    context.auth = auth
    context.metadata = metadata or {}
    set_context(context)
    return context


def run(coro):
    return asyncio.run(coro)


def keys(result):
    return sorted(f"{e['namespace']}/{e['key']}" for e in result["entries"])


def seeded(skill):
    run(skill.initialize(FakeAgent()))
    as_caller(ALICE)
    assert run(skill.memory_write(key="preferences", content="Alice likes short answers about launches."))["namespace"] == "caller:user:alice"
    as_caller(BOB)
    assert run(skill.memory_write(key="preferences", content="Bob wants long answers about launches."))["namespace"] == "caller:user:bob"
    as_caller(FINDER)
    assert run(skill.memory_write(key="venue", content="The finder agent booked the Forum for the launch."))["namespace"] == "caller:agent:https://a.example/finder"
    as_caller(OWNER)
    assert run(skill.memory_write(key="plan", content="Owner plan: launch in October."))["namespace"] == "owner"
    assert run(skill.memory_write(key="office-hours", content="Launch office hours: 9 to 5.", namespace="shared"))["namespace"] == "shared"


def local_skill(tmp_path, **extra):
    return MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", **extra})


def portal_skill(portal):
    return MemorySkill(
        {"local": False, "portal": True, "agent_id": portal.agent_id, "robutler_api_url": "https://portal.test", "api_key": "agent-key", "transport": portal.transport}
    )


@pytest.fixture(params=["local", "plain", "portal"])
def make(request, tmp_path):
    if request.param == "local":
        return lambda: local_skill(tmp_path)
    if request.param == "plain":
        return lambda: local_skill(tmp_path, plain_index=True)
    portal = FakePortal()
    return lambda: portal_skill(portal)


def test_a_callers_search_never_returns_anothers_entries(make):
    skill = make()
    seeded(skill)
    as_caller(ALICE)
    assert keys(run(skill.memory_search(query="launch"))) == ["caller:user:alice/preferences", "shared/office-hours"]
    as_caller(BOB)
    assert keys(run(skill.memory_search(query="launch"))) == ["caller:user:bob/preferences", "shared/office-hours"]
    as_caller(FINDER)
    assert keys(run(skill.memory_search(query="launch"))) == ["caller:agent:https://a.example/finder/venue", "shared/office-hours"]
    as_caller(NOBODY)
    assert keys(run(skill.memory_search(query="launch"))) == ["shared/office-hours"]


def test_the_owner_sees_everything_and_may_narrow(make):
    skill = make()
    seeded(skill)
    as_caller(OWNER)
    assert keys(run(skill.memory_search(query="launch"))) == [
        "caller:agent:https://a.example/finder/venue",
        "caller:user:alice/preferences",
        "caller:user:bob/preferences",
        "owner/plan",
        "shared/office-hours",
    ]
    assert keys(run(skill.memory_search(query="launch", namespace="caller:user:bob"))) == ["caller:user:bob/preferences"]
    assert len(run(skill.memory_list())["entries"]) == 5


def test_a_caller_naming_a_namespace_gets_its_own_or_a_refusal(make):
    skill = make()
    seeded(skill)
    as_caller(ALICE)
    assert run(skill.memory_search(query="launch", namespace="caller:user:bob")) == {"error": NOT_YOURS}
    assert run(skill.memory_search(query="launch", namespace="owner")) == {"error": NOT_YOURS}
    assert keys(run(skill.memory_search(query="launch", namespace="shared"))) == ["shared/office-hours"]
    assert keys(run(skill.memory_list())) == ["caller:user:alice/preferences", "shared/office-hours"]
    assert run(skill.memory_list(namespace="caller:user:bob")) == {"error": NOT_YOURS}


def test_a_callers_write_lands_in_its_own_namespace_never_the_owners(make):
    skill = make()
    seeded(skill)
    as_caller(BOB)
    assert run(skill.memory_write(key="plan", content="Bob says: cancel the launch.", namespace="owner"))["namespace"] == "caller:user:bob"
    assert run(skill.memory_write(key="office-hours", content="Bob says: closed.", namespace="shared"))["namespace"] == "caller:user:bob"
    as_caller(OWNER)
    assert [e["content"] for e in run(skill.memory_search(query="plan", namespace="owner"))["entries"]] == ["Owner plan: launch in October."]
    as_caller(ALICE)
    assert [e["content"] for e in run(skill.memory_search(query="office hours", namespace="shared"))["entries"]] == ["Launch office hours: 9 to 5."]


def test_forgetting_is_scoped_the_same_way(make):
    skill = make()
    seeded(skill)
    as_caller(BOB)
    assert run(skill.memory_forget(key="preferences", namespace="caller:user:alice")) == {"error": NOT_YOURS}
    assert run(skill.memory_forget(key="preferences")) == {"ok": True, "forgotten": 1}
    as_caller(ALICE)
    assert keys(run(skill.memory_list())) == ["caller:user:alice/preferences", "shared/office-hours"]
    as_caller(OWNER)
    assert run(skill.memory_forget(key="preferences", namespace="caller:user:alice")) == {"ok": True, "forgotten": 1}
    as_caller(ALICE)
    assert keys(run(skill.memory_list())) == ["shared/office-hours"]


def test_nobody_verified_reads_shared_and_writes_nothing(make):
    skill = make()
    seeded(skill)
    as_caller(NOBODY)
    assert run(skill.memory_write(key="x", content="y")) == {"error": NO_CALLER}
    assert keys(run(skill.memory_list())) == ["shared/office-hours"]


def test_each_caller_in_a_folder_of_its_own(tmp_path):
    skill = local_skill(tmp_path)
    seeded(skill)
    root = tmp_path / ".webagents" / "memory"
    assert (root / "owner" / "plan.md").exists() and (root / "shared" / "office-hours.md").exists()
    callers = sorted(p for p in (root / "callers").iterdir() if p.is_dir())
    assert len(callers) == 3
    assert sorted(json.loads((c / "caller.json").read_text())["principal"] for c in callers) == ["agent:https://a.example/finder", "user:alice", "user:bob"]
    text = (root / "owner" / "plan.md").read_text()
    assert text.startswith("---\nid: ") and "\nkey: plan\nnamespace: owner\nsource: tool\n" in text
    assert text.endswith("Owner plan: launch in October.\n")
    assert oct(os.stat(root / "owner" / "plan.md").st_mode & 0o777) == "0o600"
    assert skill._local.index_kind == "sqlite"


def test_the_portal_tier_sends_the_namespace_it_derived_and_the_agent_key():
    portal = FakePortal()
    skill = portal_skill(portal)
    run(skill.initialize(FakeAgent()))
    as_caller(ALICE)
    run(skill.memory_write(key="preferences", content="short"))
    run(skill.memory_search(query="short"))
    assert all(r.headers.get("authorization") == "Bearer agent-key" for r in portal.seen)
    put = next(r for r in portal.seen if r.method == "PUT")
    assert json.loads(put.content) == {"agentId": portal.agent_id, "namespace": "caller:user:alice", "key": "preferences", "content": "short", "source": "tool", "at": None}
    search = next(r for r in portal.seen if r.url.params.get("action") == "search")
    # One namespace per repeated parameter, never comma-joined (S-298).
    assert search.url.params.get_list("namespace") == ["caller:user:alice", "shared"]
    assert "namespaces" not in search.url.params


def test_through_the_agent(tmp_path):
    from webagents.agents.core.base_agent import BaseAgent

    skill = local_skill(tmp_path)
    agent = BaseAgent(name="helper", instructions="x", skills={"memory": skill})
    run(agent._ensure_skills_initialized())
    names = sorted(t["name"] for t in agent.get_all_tools() if t["name"].startswith("memory_"))
    assert names == ["memory_forget", "memory_list", "memory_read", "memory_search", "memory_write"]
    as_caller(ALICE)
    assert run(agent.execute_tool("memory_write", {"key": "preferences", "content": "short"}))["namespace"] == "caller:user:alice"
    as_caller(OWNER)
    assert run(agent.execute_tool("memory_write", {"key": "plan", "content": "October"}))["namespace"] == "owner"
    as_caller(BOB)
    assert keys(run(agent.execute_tool("memory_search", {"query": "short October"}))) == []
    as_caller(ALICE)
    assert keys(run(agent.execute_tool("memory_search", {"query": "short October"}))) == ["caller:user:alice/preferences"]
    as_caller(OWNER)
    assert keys(run(agent.execute_tool("memory_search", {"query": "short October"}))) == ["caller:user:alice/preferences", "owner/plan"]
