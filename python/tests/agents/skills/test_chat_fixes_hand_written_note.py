"""
A memory note written by hand is loaded (2026-09-27, the chat-fixes lane),
against the shared fixture `tests/fixtures/memory_tool/chat_fixes_hand_written_note.json`,
which the TypeScript suite runs too (`chat-fixes-hand-written-note.test.ts`).
The docs said the files could be edited by hand; a plain `.md` with no front
matter was ignored without a word.
"""

import json
from pathlib import Path

import pytest

from webagents.agents.skills.local.memory.local_memory_store import LocalMemoryStore, parse_entry_file

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "memory_tool" / "chat_fixes_hand_written_note.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_the_cases_both_stores_read(case):
    entry = parse_entry_file(case["text"], case["namespace"], case["key"], "helper")
    if case["entry"] is None:
        assert entry is None
        return
    assert entry is not None
    assert {"namespace": entry.namespace, "key": entry.key, "content": entry.content, "source": entry.source} == case["entry"]


def test_a_note_dropped_into_owner_is_found_by_the_store(tmp_path):
    root = tmp_path / "memory"
    (root / "owner").mkdir(parents=True)
    (root / "owner" / "preferences.md").write_text("Prefers short answers.\n")
    store = LocalMemoryStore(root, "helper", plain_index=True)
    store.open()
    try:
        found = store.get("owner", "preferences")
        assert found is not None and found.content == "Prefers short answers." and found.source == "owner"
        assert [e.key for e in store.search("short", ["owner"])] == ["preferences"]
    finally:
        store.close()
