"""Compaction through the skill's own hook (plan items 2.1 and 2.4): past the
threshold the working conversation is rewritten in place, the summary becomes
an episode in the CALLER's namespace, a summarizing run is left alone, the
frozen notes stay frozen for the session, and the two tiers sync. Deterministic
under a stub model; the TypeScript twin is
`typescript/tests/unit/skills/memory/memory-compaction-w2mem.test.ts` and
`memory-local-store-w2mem.test.ts`."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from webagents.agents.skills.local.memory.caller_scoped import MemorySkill
from webagents.agents.skills.local.memory.local_memory_store import LocalMemoryStore, parse_entry_file, render_entry_file
from webagents.agents.skills.local.memory.memory_compaction import COMPACTION_PREFIX
from webagents.server.context.context_vars import create_context, set_context

from fake_portal_w2mem import FakePortal

OWNER = SimpleNamespace(authenticated=True, provider="platform", scope="owner", user_id="owner-1")
ALICE = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="alice")
BOB = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="bob")


class FakeAgent:
    name = "helper"
    skills = {}

    def register_tool(self, fn, source=None, scope=None):
        pass


def as_caller(auth, metadata=None, messages=None, flags=None):
    context = create_context(messages=messages if messages is not None else [])
    context.auth = auth
    context.metadata = metadata or {}
    for key, value in (flags or {}).items():
        context.set(key, value)
    set_context(context)
    return context


def run(coro):
    return asyncio.run(coro)


async def stub(transcript, instructions):
    assert instructions.startswith("Summarize the conversation below")
    return f"[stub summary of {len(transcript.split(chr(10)))} lines]"


def long_conversation():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Plan the launch."},
        {"role": "assistant", "content": "Which week?"},
        {"role": "user", "content": "The first of October."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Book a venue."},
        {"role": "assistant", "content": "Done."},
    ]


def test_rewrites_the_conversation_in_place_and_keeps_the_callers_episode(tmp_path):
    calls = []

    async def counting(transcript, instructions):
        calls.append(1)
        return await stub(transcript, instructions)

    skill = MemorySkill(
        {
            "agent_path": str(tmp_path),
            "agent_name": "helper",
            "compaction": {"threshold": 20, "keep": 2},
            "summarizer": counting,
            "now": lambda: datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc),
        }
    )
    run(skill.initialize(FakeAgent()))
    conversation = long_conversation()
    context = as_caller(ALICE, messages=conversation)
    run(skill.compact_before_call(context))
    assert calls == [1]
    assert context.messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": f"{COMPACTION_PREFIX}[stub summary of 4 lines]"},
        {"role": "user", "content": "Book a venue."},
        {"role": "assistant", "content": "Done."},
    ]
    assert conversation is context.messages
    assert skill.compactions == 1
    as_caller(ALICE)
    assert run(skill.memory_list())["entries"] == [
        {"key": "episode-2026-09-26T10-00-00-000Z", "namespace": "caller:user:alice", "updated_at": "2026-09-26T10:00:00.000Z"}
    ]
    assert [e["content"] for e in run(skill.memory_search(query="stub summary"))["entries"]] == ["[stub summary of 4 lines]"]
    as_caller(OWNER)
    assert run(skill.memory_list(namespace="owner"))["entries"] == []
    skill.unfreeze_notes()
    assert run(skill.frozen_notes(context=as_caller(ALICE))) == ""


def test_does_nothing_under_the_threshold_and_stays_out_of_its_own_run(tmp_path):
    calls = []

    async def counting(transcript, instructions):
        calls.append(1)
        return await stub(transcript, instructions)

    skill = MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", "compaction": {"threshold": 20, "keep": 2}, "summarizer": counting})
    run(skill.initialize(FakeAgent()))
    short = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}]
    run(skill.compact_before_call(as_caller(ALICE, messages=short)))
    assert len(short) == 2
    nested = as_caller(ALICE, messages=long_conversation(), flags={"memory_compaction": True})
    run(skill.compact_before_call(nested))
    assert len(nested.messages) == 7
    assert run(skill.frozen_notes(context=nested)) == ""
    assert calls == []


def test_an_empty_summary_leaves_the_conversation_as_it_was(tmp_path):
    async def nothing(_t, _i):
        return ""

    skill = MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", "compaction": {"threshold": 20, "keep": 2}, "summarizer": nothing})
    run(skill.initialize(FakeAgent()))
    context = as_caller(ALICE, messages=long_conversation())
    run(skill.compact_before_call(context))
    assert len(context.messages) == 7 and skill.compactions == 0


def test_the_default_summarizer_is_the_agents_model(tmp_path):
    class Model:
        def __init__(self):
            self.seen = []

        async def chat_completion(self, messages, **_):
            self.seen.append(messages)
            return {"choices": [{"message": {"content": "  the summary  "}}]}

    model = Model()
    agent = FakeAgent()
    agent.skills = {"primary_llm": model}
    skill = MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", "compaction": {"threshold": 20, "keep": 2}})
    run(skill.initialize(agent))
    context = as_caller(ALICE, messages=long_conversation())
    run(skill.compact_before_call(context))
    assert context.messages[1]["content"] == f"{COMPACTION_PREFIX}the summary"
    assert model.seen[0][0]["role"] == "system" and "user: Plan the launch." in model.seen[0][1]["content"]


def test_frozen_notes_are_computed_once_per_session(tmp_path):
    skill = MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", "notes_budget": 400})
    run(skill.initialize(FakeAgent()))
    as_caller(OWNER)
    run(skill.memory_write(key="office-hours", content="9 to 5.", namespace="shared"))
    as_caller(ALICE)
    run(skill.memory_write(key="name", content="Ada."))
    session = {"session_id": "sess-1"}
    first = run(skill.frozen_notes(context=as_caller(ALICE, session)))
    assert first == "## Memory\nShared notes:\n- office-hours: 9 to 5.\nNotes about this caller:\n- name: Ada."
    as_caller(ALICE)
    run(skill.memory_write(key="tone", content="Formal."))
    assert run(skill.frozen_notes(context=as_caller(ALICE, session))) == first
    assert "- tone: Formal." in run(skill.frozen_notes(context=as_caller(ALICE, {"session_id": "sess-2"})))
    assert run(skill.frozen_notes(context=as_caller(OWNER, session))) == "## Memory\nShared notes:\n- office-hours: 9 to 5."
    assert run(skill.frozen_notes(context=as_caller(BOB, session))) == "## Memory\nShared notes:\n- office-hours: 9 to 5."


def test_the_local_store_finds_hand_written_and_hand_edited_notes(tmp_path):
    for plain in (False, True):
        root = tmp_path / ("plain" if plain else "sqlite") / "memory"
        store = LocalMemoryStore(root, "helper", plain_index=plain)
        store.open()
        assert store.index_kind == ("plain" if plain else "sqlite")
        store.put("owner", "preferences", "Short answers.")
        store.close()
        (root / "owner" / "venue.md").write_text(
            render_entry_file(store.get("owner", "preferences").__class__(id="x", namespace="owner", key="venue", content="The Forum, on the river.", source="owner", created_at="2026-01-01T00:00:00.000Z", updated_at="2026-01-01T00:00:00.000Z"))
        )
        (root / "owner" / "preferences.md").write_text((root / "owner" / "preferences.md").read_text().replace("Short answers.", "Long answers."))
        again = LocalMemoryStore(root, "helper", plain_index=plain)
        again.open()
        assert [e.key for e in again.search("river", None)] == ["venue"]
        assert [e.content for e in again.search("answers", ["owner"])] == ["Long answers."]
        assert again.search("answers", ["shared"]) == []
        again.close()
    # A plain file under a known namespace is a note written by hand (2026-09-27,
    # `tests/agents/skills/test_chat_fixes_hand_written_note.py`); one under a
    # caller folder that names no caller is still nobody's.
    hand = parse_entry_file("just text", "owner", "a", "s")
    assert (hand.namespace, hand.key, hand.content, hand.source) == ("owner", "a", "just text", "owner")
    assert parse_entry_file("just text", "", "a", "s") is None
    assert parse_entry_file("---\nkey: a\n---\nbody\n", "", "a", "s") is None
    parsed = parse_entry_file("---\nkey: a\nnamespace: caller:user:bob\nsource: compaction\n---\nbody\n", "", "a", "s")
    assert (parsed.namespace, parsed.source, parsed.content) == ("caller:user:bob", "compaction", "body")


def test_the_log_and_the_other_tiers_changes(tmp_path):
    store = LocalMemoryStore(tmp_path / "memory", "helper", plain_index=True, now=lambda: datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc))
    store.open()
    store.put("owner", "a", "one")
    store.put("owner", "a", "two")
    store.forget("owner", "a")
    assert [(l["seq"], l["op"], l["key"], l.get("content")) for l in store.log_since(0)] == [(1, "put", "a", "one"), (2, "put", "a", "two"), (3, "delete", "a", None)]
    assert [l["seq"] for l in store.log_since(2)] == [3]
    applied = store.apply(
        [
            {"seq": 9, "op": "put", "id": "x", "namespace": "shared", "key": "hours", "content": "9 to 5", "source": "tool", "at": "2026-09-26T11:00:00.000Z"},
            {"seq": 10, "op": "put", "id": "x", "namespace": "shared", "key": "hours", "content": "older", "source": "tool", "at": "2026-09-26T09:00:00.000Z"},
            {"seq": 11, "op": "delete", "id": "y", "namespace": "shared", "key": "missing", "at": "2026-09-26T11:00:00.000Z"},
        ]
    )
    assert applied == 1
    assert (store.get("shared", "hours").content, store.get("shared", "hours").source) == ("9 to 5", "sync")
    assert len(store.log_since(0)) == 3


def test_local_and_portal_together_sync_both_ways(tmp_path):
    portal = FakePortal()
    skill = MemorySkill(
        {
            "local": True,
            "portal": True,
            "agent_path": str(tmp_path),
            "agent_id": portal.agent_id,
            "robutler_api_url": "https://portal.test",
            "api_key": "agent-key",
            "transport": portal.transport,
            "plain_index": True,
        }
    )
    run(skill.initialize(FakeAgent()))
    as_caller(OWNER)
    run(skill.memory_write(key="plan", content="October"))
    as_caller(ALICE)
    run(skill.memory_write(key="name", content="Ada"))
    assert sorted(f"{r['namespace']}/{r['key']}={r['content']}" for r in portal.rows.values()) == ["caller:user:alice/name=Ada", "owner/plan=October"]
    assert json.loads((tmp_path / ".webagents" / "memory" / "sync-state.json").read_text())["pushedSeq"] == 2

    portal.put("owner", "plan", "November", "owner", "2999-01-01T00:00:00.000Z")
    portal.put("shared", "hours", "9 to 5", "owner", "2999-01-01T00:00:00.000Z")
    run(skill.pull_for_caller(as_caller(OWNER)))
    assert [e["content"] for e in run(skill.memory_search(query="November"))["entries"]] == ["November"]
    assert "9 to 5" in (tmp_path / ".webagents" / "memory" / "shared" / "hours.md").read_text()
    assert len([r for r in portal.seen if r.method == "POST"]) == 2
    as_caller(ALICE)
    assert [e["key"] for e in run(skill.memory_search(query="Ada"))["entries"]] == ["name"]
