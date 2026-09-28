"""
The /undo snapshot in a large folder (2026-09-28), the same in both SDKs
(`typescript/tests/unit/cli/checkpoints-partial.test.ts` runs these cases too).

The walk used to go on past its caps, listing every later path as skipped:
under ~/dev/portal (331,695 files) that was a 9 s walk before every message,
before the chat's spinner had started. It now stops at the cap and says the
snapshot is partial, and a restore of a partial snapshot removes nothing,
because it cannot tell a file made since from one past the cap.
"""

import json

from webagents.cli import checkpoints as cp


def _folder(tmp_path, names):
    root = tmp_path / "project"
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{name}\n")
    return root, tmp_path / "store"


def test_the_walk_stops_at_the_cap_and_the_snapshot_says_partial(tmp_path):
    root, store = _folder(tmp_path, ["a.txt", "b/c.txt", "b/d.txt", "e.txt", "f.txt"])
    manifest = cp.take_snapshot(root, "before", store=store, limits={"max_files": 3})
    assert manifest["partial"] is True
    # Walk order: names sorted at each level, directories entered in place.
    assert sorted(manifest["files"]) == ["a.txt", "b/c.txt", "b/d.txt"]
    assert manifest["skipped"] == {}


def test_a_folder_under_the_cap_is_not_partial_and_has_no_partial_key(tmp_path):
    root, store = _folder(tmp_path, ["a.txt", "b.txt"])
    manifest = cp.take_snapshot(root, "before", store=store, limits={"max_files": 3})
    assert "partial" not in manifest
    assert "partial" not in json.loads((store / f"{manifest['id']}.json").read_text())


def test_the_byte_cap_stops_it_too(tmp_path):
    root, store = _folder(tmp_path, ["a.txt", "b.txt", "c.txt"])
    state = cp.scan_folder(root, store, None, {"max_total_bytes": 12})
    assert state["partial"] is True and list(state["files"]) == ["a.txt", "b.txt"]


def test_a_restore_of_a_partial_snapshot_puts_back_what_it_recorded_and_removes_nothing(tmp_path):
    root, store = _folder(tmp_path, ["a.txt", "b.txt", "c.txt", "d.txt"])
    target = cp.take_snapshot(root, "before", store=store, limits={"max_files": 2})
    (root / "a.txt").write_text("changed by the agent\n")
    (root / "new.txt").write_text("made since\n")
    plan = cp.plan_restore(target, cp.scan_folder(root, store))
    assert plan.write == ["a.txt"] and plan.remove == []
    result = cp.restore_snapshot(root, target["id"], "before /undo", store=store)
    assert result.removed == []
    assert (root / "a.txt").read_text() == "a.txt\n"
    # Past the cap when the snapshot was taken, or made since: either way, left.
    assert (root / "new.txt").exists() and (root / "d.txt").exists()


def test_the_chats_say_why_nothing_is_removed():
    assert cp.PARTIAL_NOTE == (
        "This folder is larger than a snapshot holds (20000 files, 200 MB), so files made since it are left in place."
    )


def _write(store, id_, created_at):
    store.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "id": id_, "created_at": created_at, "label": id_, "files": {}, "links": {}, "skipped": {}}
    (store / f"{id_}.json").write_text(json.dumps(manifest))


def test_the_newest_is_found_breaking_a_same_second_tie_by_the_time_inside_it(tmp_path):
    store = tmp_path / "store"
    _write(store, "cp_20260928T120000Z_ffffffff", "2026-09-28T12:00:00.100Z")
    _write(store, "cp_20260928T120001Z_00000001", "2026-09-28T12:00:01.900Z")
    _write(store, "cp_20260928T120001Z_ffffffff", "2026-09-28T12:00:01.200Z")
    assert cp.latest_checkpoint(store)["id"] == "cp_20260928T120001Z_00000001"


def test_the_newest_skips_a_manifest_that_cannot_be_read(tmp_path):
    store = tmp_path / "store"
    _write(store, "cp_20260928T120000Z_aaaaaaaa", "2026-09-28T12:00:00.000Z")
    (store / "cp_20260928T120005Z_bbbbbbbb.json").write_text("{not json")
    assert cp.latest_checkpoint(store)["id"] == "cp_20260928T120000Z_aaaaaaaa"


def test_there_is_no_newest_in_an_empty_store(tmp_path):
    assert cp.latest_checkpoint(tmp_path / "missing") is None
