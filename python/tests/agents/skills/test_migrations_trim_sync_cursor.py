"""S-307 (logged 2026-09-27, fixed by the migrations-trim lane): the portal
answered a push with ``Number.MAX_SAFE_INTEGER`` as the cursor, this skill kept
it with ``max``, and every later pull asked for "after the end": the owner's
edits and forgets on Robutler never reached the agent's local notes. The suites
missed it because the fake portal minted its own cursor. Pinned here against
the fake portal that now follows the platform's contract, with the SHARED
fixture the portal and the TypeScript suite read too
(``tests/fixtures/memory_tool/migrations_trim_sync.json``):

- a second pull after a push sees the owner's later edit and forget;
- a push sends no cursor back and none is kept; a pull keeps the platform's
  string as it came, and sends it back as ``since``;
- a numeric cursor an older state file holds (S-307's stuck one) is dropped,
  and that namespace pulls from the start;
- a pull takes every page (``more``) in one go.

The TypeScript twin is ``typescript/tests/unit/skills/memory/migrations-trim-sync-cursor.test.ts``.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from webagents.agents.skills.local.memory.caller_scoped import MAX_PULL_PAGES, MemorySkill
from webagents.agents.skills.local.memory.portal_memory_store import PortalMemoryStore
from webagents.server.context.context_vars import create_context, set_context

from fake_portal_w2mem import FakePortal

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "memory_tool" / "migrations_trim_sync.json").read_text())
CURSOR = re.compile(FIXTURE["cursor"]["pattern"])
OWNER = SimpleNamespace(authenticated=True, provider="platform", scope="owner", user_id="owner-1")


class FakeAgent:
    name = "helper"
    skills = {}

    def register_tool(self, fn, source=None, scope=None):
        pass


def as_caller(auth):
    context = create_context(messages=[])
    context.auth = auth
    context.metadata = {}
    set_context(context)
    return context


def run(coro):
    return asyncio.run(coro)


def skill_for(portal: FakePortal, path: Path) -> MemorySkill:
    skill = MemorySkill(
        {
            "local": True,
            "portal": True,
            "agent_path": str(path),
            "agent_id": portal.agent_id,
            "robutler_api_url": "https://portal.test",
            "api_key": "agent-key",
            "transport": portal.transport,
            "plain_index": True,
            "now": lambda: datetime(2026, 9, 27, 9, 0, 0, tzinfo=timezone.utc),
        }
    )
    run(skill.initialize(FakeAgent()))
    return skill


def state_of(path: Path) -> dict:
    return json.loads((path / ".webagents" / "memory" / "sync-state.json").read_text())


def sync_gets(portal: FakePortal):
    return [r for r in portal.seen if r.method == "GET" and r.url.params.get("action") == "sync"]


def test_a_second_pull_after_a_push_sees_the_owners_later_edit_and_forget(tmp_path):
    s = FIXTURE["scenario"]
    portal = FakePortal()
    skill = skill_for(portal, tmp_path)
    as_caller(OWNER)
    for w in s["agent_writes"]:
        assert run(skill.memory_write(key=w["key"], content=w["content"]))["ok"] is True
    assert sorted(f"{r['key']}={r['content']}" for r in portal.rows.values()) == sorted(f"{w['key']}={w['content']}" for w in s["agent_writes"])
    # The pushes answered no cursor, and none was kept.
    assert state_of(tmp_path)["cursors"] == {}

    run(skill.pull([s["namespace"]]))
    first = state_of(tmp_path)["cursors"][s["namespace"]]
    assert CURSOR.match(first)

    # On Robutler, after the agent wrote: the owner edits one note and forgets the other.
    portal.put(s["namespace"], s["owner_edit"]["key"], s["owner_edit"]["content"], "owner", "2026-09-27T10:00:00.000Z")
    portal.delete(s["namespace"], s["owner_forget"]["key"], "2026-09-27T10:00:01.000Z")

    run(skill.pull([s["namespace"]]))
    assert sync_gets(portal)[-1].url.params.get("since") == first
    folder = tmp_path / ".webagents" / "memory" / s["namespace"]
    for key, content in s["local_after"].items():
        assert content in (folder / f"{key}.md").read_text()
    for key in s["local_gone"]:
        assert not (folder / f"{key}.md").exists()
    as_caller(OWNER)
    assert sorted(e["key"] for e in run(skill.memory_list(namespace=s["namespace"]))["entries"]) == sorted(s["local_after"])
    second = state_of(tmp_path)["cursors"][s["namespace"]]
    assert CURSOR.match(second) and second > first
    # What was pulled is not pushed back.
    assert len([r for r in portal.seen if r.method == "POST"]) == len(s["agent_writes"])


def test_the_store_sends_since_only_when_it_has_one_and_a_push_answers_no_cursor():
    portal = FakePortal()
    store = PortalMemoryStore("https://portal.test", lambda: "agent-key", agent_id=portal.agent_id, transport=portal.transport)
    assert run(store.pull("owner", None)) == ([], None, False)
    assert "since" not in sync_gets(portal)[0].url.params
    applied = run(store.push("owner", [{"seq": 1, "op": "put", "id": "x", "namespace": "owner", "key": "plan", "content": "October", "source": "tool", "at": "2026-09-27T09:00:00.000Z"}]))
    assert applied == 1
    posted = [r for r in portal.seen if r.method == "POST"][0]
    assert sorted(json.loads(posted.content)) == sorted(["agentId"] + FIXTURE["wire"]["push"]["body_keys"])
    lines, cursor, more = run(store.pull("owner", None))
    assert [l["key"] for l in lines] == ["plan"] and more is False
    assert CURSOR.match(cursor)
    run(store.pull("owner", cursor))
    assert sync_gets(portal)[-1].url.params.get("since") == cursor


def test_numeric_cursors_from_an_older_state_file_are_dropped(tmp_path):
    portal = FakePortal()
    memory = tmp_path / ".webagents" / "memory"
    memory.mkdir(parents=True)
    (memory / "sync-state.json").write_text(json.dumps(FIXTURE["cursor"]["retired_state_file"]))
    portal.put("owner", "plan", "November", "owner", "2026-09-27T10:00:00.000Z")
    skill = skill_for(portal, tmp_path)
    run(skill.pull(["owner", "shared"]))
    for get in sync_gets(portal):
        assert "since" not in get.url.params
    assert "November" in (memory / "owner" / "plan.md").read_text()
    cursors = state_of(tmp_path)["cursors"]
    assert CURSOR.match(cursors["owner"])
    assert "shared" not in cursors
    assert FIXTURE["cursor"]["retired_state_file_keeps"] == {}


def test_a_pull_takes_every_page_in_one_go(tmp_path):
    portal = FakePortal(page_size=2)
    for i in range(5):
        portal.put("shared", f"k{i}", f"note {i}", "owner", "2026-09-27T10:00:00.000Z")
    skill = skill_for(portal, tmp_path)
    run(skill.pull(["shared"]))
    assert len(sync_gets(portal)) == 3
    for i in range(5):
        assert (tmp_path / ".webagents" / "memory" / "shared" / f"k{i}.md").exists()
    assert MAX_PULL_PAGES == FIXTURE["wire"]["max_pages_per_pull"]
