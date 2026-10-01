"""
The memory index (2026-09-29, the owner: "is it index based like in claude?"),
the same in the TypeScript skill (`typescript/tests/unit/skills/memory/memory-index.test.ts`).

The prompt's notes are now an index, as Claude Code's `MEMORY.md` is: one line
per note, its key and a one-line description, and `memory_read` gives a note
in full. Pinned here: `memory_write` keeps a description (in the note file's
front matter, one line); `memory_list`, `memory_search` and `memory_read` give
it back; search finds a note by it; it survives the store reopening and the
sync; the index shows it, or the note's first line without one; and
`memory_read` reads only what the caller may.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from webagents.agents.skills.local.memory.caller_scoped import NOT_YOURS, MemorySkill
from webagents.agents.skills.local.memory.local_memory_store import LocalMemoryStore
from webagents.agents.skills.local.memory.memory_notes import NOTES_GUIDE, NOTES_HEADING
from webagents.server.context.context_vars import create_context, set_context

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "memory_tool" / "definition.json").read_text())
OWNER = SimpleNamespace(authenticated=True, provider="platform", scope="owner", user_id="owner-1")
ALICE = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="alice")
BOB = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="bob")
NOW = lambda: datetime(2026, 9, 29, 10, 0, 0, tzinfo=timezone.utc)  # noqa: E731


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


def _skill(tmp_path):
    skill = MemorySkill({"agent_path": str(tmp_path), "agent_name": "helper", "now": NOW})
    run(skill.initialize(FakeAgent()))
    return skill


def test_the_index_words_are_the_fixtures():
    assert NOTES_HEADING == FIXTURE["notes"]["heading"]
    assert NOTES_GUIDE == FIXTURE["notes"]["guide"]


def test_a_description_is_kept_given_back_and_found(tmp_path):
    skill = _skill(tmp_path)
    as_caller(OWNER)
    wrote = run(skill.memory_write(key="launch", content="The launch is on 1 October.\nVenue: the Forum.", description="When and where\nthe launch is"))
    assert wrote["ok"] is True
    text = (tmp_path / ".webagents" / "memory" / "owner" / "launch.md").read_text()
    assert "\ndescription: When and where the launch is\n" in text
    assert run(skill.memory_list())["entries"][0]["description"] == "When and where the launch is"
    read = run(skill.memory_read(key="launch"))
    assert (read["key"], read["namespace"], read["description"]) == ("launch", "owner", "When and where the launch is")
    assert read["content"] == "The launch is on 1 October.\nVenue: the Forum."
    assert [e["key"] for e in run(skill.memory_search(query="where"))["entries"]] == ["launch"]
    assert run(skill.memory_read(key="nope")) == {"error": "memory: no note called nope."}


def test_the_index_shows_descriptions_and_first_lines(tmp_path):
    skill = _skill(tmp_path)
    as_caller(OWNER)
    run(skill.memory_write(key="launch", content="The launch is on 1 October.", description="When the launch is"))
    run(skill.memory_write(key="tone", content="# Tone\nFormal, no emoji."))
    notes = run(skill.frozen_notes(context=as_caller(OWNER, {"session_id": "s1"})))
    assert notes.startswith(f"{NOTES_HEADING}\n{NOTES_GUIDE}\nYour notes (owner):\n")
    assert "- launch: When the launch is" in notes and "- tone: Tone" in notes
    assert "The launch is on 1 October." not in notes


def test_a_caller_reads_only_its_own_and_the_shared_notes(tmp_path):
    skill = _skill(tmp_path)
    as_caller(OWNER)
    run(skill.memory_write(key="office-hours", content="9 to 5.", namespace="shared"))
    as_caller(ALICE)
    run(skill.memory_write(key="name", content="Ada.", description="What to call her"))
    assert run(skill.memory_read(key="office-hours"))["namespace"] == "shared"
    assert run(skill.memory_read(key="name"))["description"] == "What to call her"
    as_caller(BOB)
    assert run(skill.memory_read(key="name")) == {"error": "memory: no note called name."}
    assert run(skill.memory_read(key="name", namespace="caller:user:alice")) == {"error": NOT_YOURS}
    as_caller(OWNER)
    assert run(skill.memory_read(key="name", namespace="caller:user:alice"))["content"] == "Ada."


def test_the_description_survives_reopening_and_the_sync(tmp_path):
    root = tmp_path / "memory"
    store = LocalMemoryStore(root, "helper")
    store.open()
    store.put("owner", "launch", "On 1 October.", description="When the launch is")
    line = store.log_since(0)[-1]
    store.close()
    again = LocalMemoryStore(root, "helper")
    again.open()
    assert again.get("owner", "launch").description == "When the launch is"
    assert [e.key for e in again.search("launch is", ["owner"])] == ["launch"]
    other = LocalMemoryStore(tmp_path / "other", "helper")
    other.open()
    other.apply([{"op": "put", "namespace": "owner", "key": "venue", "content": "The Forum.", "description": "Where it is", "source": "tool", "at": "2026-09-29T10:00:00.000Z"}])
    assert other.get("owner", "venue").description == "Where it is"
    assert line["description"] == "When the launch is"
