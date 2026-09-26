"""
Snapshots of an agent's folder for /undo and /rewind (2026-09-25), the same in
both SDKs: `tests/fixtures/checkpoints/checkpoints.json`, which the TypeScript
suite runs too (`typescript/tests/unit/cli/checkpoints-fixture.test.ts`).

What a snapshot of a folder holds, what a restore does and leaves the folder
as, the words both chats say; and beside the fixture, what keeps a restore
safe: nothing written through a symlinked folder, owner-only files kept under
the profile and never in the folder, and only the newest `KEEP` snapshots.
"""

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from webagents.cli import checkpoints as cp

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "checkpoints" / "checkpoints.json").read_text())


def build(root: Path, tree) -> None:
    for f in tree["files"]:
        p = root / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["text"])
        os.chmod(p, f["mode"])
    for link in tree["links"]:
        (root / link["path"]).symlink_to(link["target"])
    for big in tree["large"]:
        p = root / big["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as handle:
            handle.truncate(big["size"])


def change(root: Path, changes) -> None:
    for c in changes:
        if "write" in c:
            p = root / c["write"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c["text"])
        elif "remove" in c:
            (root / c["remove"]).unlink()
        elif "chmod" in c:
            os.chmod(root / c["chmod"], c["mode"])
        elif "relink" in c:
            (root / c["relink"]).unlink()
            (root / c["relink"]).symlink_to(c["target"])


def held(manifest):
    return {
        "files": {k: {kk: v[kk] for kk in ("sha256", "size", "mode")} for k, v in manifest["files"].items()},
        "links": manifest["links"],
        "skipped": manifest["skipped"],
    }


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "agent"
    root.mkdir()
    build(root, FIXTURE["tree"])
    return root


def test_a_snapshot_holds_what_the_fixture_says(folder, tmp_path):
    manifest = cp.take_snapshot(folder, 'before "plan the launch"', store=tmp_path / "store")
    assert held(manifest) == FIXTURE["snapshot"]
    assert cp.is_checkpoint_id(manifest["id"]) and manifest["label"] == 'before "plan the launch"'
    assert manifest["created_at"].endswith("Z") and len(manifest["created_at"]) == len("2026-09-25T19:05:12.345Z")


def test_a_restore_does_what_the_fixture_says_and_leaves_the_folder_as_it_was(folder, tmp_path):
    store = tmp_path / "store"
    manifest = cp.take_snapshot(folder, "before", store=store)
    change(folder, FIXTURE["changes"])

    plan = cp.plan_restore(manifest, cp.scan_folder(folder, store, manifest))
    assert {"write": plan.write, "remove": plan.remove, "link": plan.link, "keep": plan.keep} == FIXTURE["plan"]

    result = cp.restore_snapshot(folder, manifest["id"], "before /undo", store=store)
    assert {"written": result.written, "removed": result.removed, "kept": result.kept} == FIXTURE["restored"]
    assert result.failed == []
    assert held(cp.scan_folder(folder, store, None)) == FIXTURE["snapshot"]
    # The restore snapshotted the folder first, so it can be undone too.
    assert result.before["label"] == "before /undo" and result.before["id"] != manifest["id"]
    assert (folder / "node_modules/pkg/index.js").read_text() == "changed\n"


def test_nothing_changed_is_not_kept_twice(folder, tmp_path):
    store = tmp_path / "store"
    first = cp.take_snapshot(folder, "one", store=store)
    second = cp.take_snapshot(folder, "two", store=store)
    assert second["id"] == first["id"]
    assert len(cp.list_checkpoints(store)) == 1


def test_a_link_made_since_is_removed_and_the_folder_put_back_without_touching_its_target(tmp_path):
    root, outside, store = tmp_path / "agent", tmp_path / "outside", tmp_path / "store"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "file.txt").write_text("mine\n")
    outside.mkdir()
    manifest = cp.take_snapshot(root, "before", store=store)
    # The folder is swapped for a link to somewhere else.
    (root / "sub" / "file.txt").unlink()
    (root / "sub").rmdir()
    (root / "sub").symlink_to(outside)

    result = cp.restore_snapshot(root, manifest["id"], "before /undo", store=store)

    assert list(outside.iterdir()) == []
    assert "sub" in result.removed and not (root / "sub").is_symlink()
    assert (root / "sub" / "file.txt").read_text() == "mine\n"


def test_nothing_is_written_through_a_symlinked_folder(tmp_path):
    """A manifest that names a file under a folder that is a link (written by
    hand, or the link made between the plan and the write) is refused for
    that file: the way to it is not a folder of its own."""
    root, outside, store = tmp_path / "agent", tmp_path / "outside", tmp_path / "store"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "file.txt").write_text("mine\n")
    outside.mkdir()
    manifest = cp.take_snapshot(root, "before", store=store)
    (root / "sub" / "file.txt").unlink()
    (root / "sub").rmdir()
    (root / "sub").symlink_to(outside)
    tampered = dict(manifest, links={"sub": str(outside)})
    (store / f"{manifest['id']}.json").write_text(json.dumps(tampered))

    result = cp.restore_snapshot(root, manifest["id"], "before /undo", store=store)

    assert list(outside.iterdir()) == []
    assert [f["path"] for f in result.failed] == ["sub/file.txt"]


def test_the_store_is_owner_only_and_outside_the_folder(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    manifest = cp.take_snapshot(folder, "before")
    store = cp.checkpoints_dir(folder)
    assert store.is_relative_to(tmp_path / "home" / ".webagents" / "checkpoints")
    assert not (folder / ".webagents").exists()
    assert stat.S_IMODE((store / f"{manifest['id']}.json").stat().st_mode) == 0o600
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (store / "objects").iterdir())


def test_only_the_newest_are_kept_with_the_objects_they_use(folder, tmp_path):
    store = tmp_path / "store"
    start = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    for i in range(cp.KEEP + 2):
        (folder / "notes" / "plan.md").write_text(f"version {i}\n")
        cp.take_snapshot(folder, f"v{i}", store=store, now=start + timedelta(seconds=i))
    kept = cp.list_checkpoints(store)
    assert len(kept) == cp.KEEP and kept[0]["label"] == f"v{cp.KEEP + 1}" and kept[-1]["label"] == "v2"
    used = {f["sha256"] for m in kept for f in m["files"].values()}
    assert {p.name for p in (store / "objects").iterdir()} == used


@pytest.mark.parametrize("case", FIXTURE["ids"], ids=[c["id"] or "empty" for c in FIXTURE["ids"]])
def test_ids(case):
    assert cp.is_checkpoint_id(case["id"]) == case["valid"]


def test_a_new_id_is_one():
    assert cp.is_checkpoint_id(cp.new_checkpoint_id())


@pytest.mark.parametrize("case", FIXTURE["labels"], ids=[c["message"][:20] for c in FIXTURE["labels"]])
def test_labels(case):
    assert cp.turn_label(case["message"]) == case["label"]


@pytest.mark.parametrize("case", FIXTURE["plan_lines"])
def test_plan_lines(case):
    assert cp.plan_lines(cp.RestorePlan(**case["plan"])) == case["lines"]


@pytest.mark.parametrize("case", FIXTURE["restored_sentences"])
def test_restored_sentences(case):
    assert cp.restored_sentence(case["written"], case["removed"]) == case["sentence"]


@pytest.mark.parametrize("case", FIXTURE["off"], ids=[c["case"] for c in FIXTURE["off"]])
def test_undo_is_off_in_the_home_folder_and_above(case, tmp_path):
    (tmp_path / "home" / "project").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    reason = cp.snapshots_off_reason(tmp_path / case["folder"], home=tmp_path / "home")
    assert (reason == cp.UNDO_OFF_HERE) if case["off"] else reason is None


def test_the_root_is_off():
    assert cp.snapshots_off_reason(Path("/")) == cp.UNDO_OFF_HERE


def test_the_words():
    said = FIXTURE["words"]
    assert said == {
        "nothing_to_undo": cp.NOTHING_TO_UNDO,
        "nothing_changed": cp.NOTHING_CHANGED,
        "undo_header": cp.UNDO_HEADER,
        "confirm": cp.CONFIRM,
        "left_as_is": cp.LEFT_AS_IS,
        "no_snapshots": cp.NO_SNAPSHOTS,
        "rewind_title": cp.REWIND_TITLE,
        "rewind_hint": cp.REWIND_HINT,
        "rewind_same": cp.REWIND_SAME,
        "rewind_missing": cp.rewind_missing("7"),
        "rewind_missing_hint": cp.REWIND_MISSING_HINT,
        "failed": cp.failed_sentence("a.txt", "Permission denied"),
        "snapshot_failed": cp.snapshot_failed("Permission denied"),
        "off_here": cp.UNDO_OFF_HERE,
        "rewind_header": cp.rewind_header("2 min ago", 'before "plan the launch"'),
    }
