"""
Snapshots of an agent's folder, for ``/undo`` and ``/rewind`` (2026-09-25).

WHY. An agent with ``filesystem`` or ``shell`` edits the folder it works in,
and a person had no way to take an edit back. Both SDKs shipped a
``checkpoint`` skill; neither worked: this one could not be reached from the
chat, and its restore brought deleted files back and wrote through symlinks;
the TypeScript one could not be named in an agent file, and handed every
caller of a served agent unscoped tools that restored any readable folder over
the working one (S-249). Both are gone. The chat takes the snapshot itself,
before each message to an agent that can change files, and only the person at
the terminal restores one.

WHERE. Under the profile, never in the folder, so file tools cannot reach them
and a repository cannot commit them:
``~/.webagents[-profile]/checkpoints/<folder>/`` (the folder named as
``sessions.py`` names it), holding ``objects/<sha256>`` (each content once) and
one ``<id>.json`` manifest per snapshot. The last ``KEEP`` are kept.

WHAT. Every regular file under the folder, by path, less ``.git``,
``.webagents``, ``node_modules``, ``.venv``, ``venv`` and ``__pycache__`` at any
depth. A symlink is recorded as a link and never followed. A file over
``MAX_FILE_BYTES`` is listed as skipped, and a restore leaves it alone. Only files whose size or modification
time changed since the last snapshot are read again, except one changed within
``RACY_MS`` of the scan: that one is kept without its time, so the next scan
reads it again (two writes of the same size inside one tick of the file
system's clock look the same, and HFS+ keeps whole seconds). A snapshot identical to
the last one is not kept twice.

PARTIAL (2026-09-28). The walk STOPS at ``MAX_FILES`` / ``MAX_TOTAL_BYTES`` and
the manifest says ``"partial": true``. It used to go on, listing every later
path as skipped: in a large repository (331,695 files under ~/dev/portal) that
was a 9 s walk before every message, before the chat's spinner had even
started, and a manifest the size of the whole tree. A partial snapshot cannot
tell a file made since from one past its cap, so a restore of one removes
nothing.

RESTORE puts every file back as the snapshot has it, removes files the
snapshot does not have (made after it; never for a partial one), and first takes a snapshot of how
things are, so a restore can itself be undone. A file is written to a
temporary name and renamed into place; nothing is written through a symlink,
or outside the folder.

ONE FORMAT IN BOTH SDKS: ``typescript/src/cli/checkpoints.ts`` reads and writes
the same files, and ``tests/fixtures/checkpoints/checkpoints.json`` holds the
cases both run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat as stat_mod
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sessions import slug_for

EXCLUDED_DIRS = (".git", ".webagents", "node_modules", ".venv", "venv", "__pycache__")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES = 20_000
MAX_TOTAL_BYTES = 200 * 1024 * 1024
#: Snapshots kept per folder.
KEEP = 50
#: A file changed this close to a scan is read again by the next one.
RACY_MS = 2000

SKIPPED_LARGE = "larger than 10 MB"
#: The UTC-second prefix of a checkpoint id: ``cp_20260925T190512Z``.
_ID_SECOND_LENGTH = len("cp_20260925T190512Z")

_ID = re.compile(r"^cp_\d{8}T\d{6}Z_[0-9a-f]{8}$")


def checkpoints_dir(folder: Path, profile: Optional[str] = None) -> Path:
    """Where ``folder``'s snapshots are kept (module docstring)."""
    from .config_store import global_dir, profile_name

    return global_dir(profile_name(profile)) / "checkpoints" / slug_for(str(Path(folder).resolve()))


#: Why snapshots are off for a folder: never for the home folder or one above it.
UNDO_OFF_HERE = "/undo is off in your home folder and above: start the chat in a project folder to use it."


def snapshots_off_reason(folder: Path, home: Optional[Path] = None) -> Optional[str]:
    """Why snapshots are off for ``folder``, or None when they are on."""
    here = Path(folder).resolve()
    mine = (home or Path.home()).resolve()
    if here == Path(here.anchor):
        return UNDO_OFF_HERE
    if mine == here or here in mine.parents:
        return UNDO_OFF_HERE
    return None


def _iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_checkpoint_id(now: Optional[datetime] = None) -> str:
    """``cp_<UTC time>_<8 hex>``."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"cp_{moment.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"


def is_checkpoint_id(value: str) -> bool:
    return bool(_ID.match(value or ""))


def _write_private(path: Path, data: bytes) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(temp, path)


def scan_folder(
    folder: Path, store: Path, previous: Optional[Dict[str, Any]] = None, limits: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """The folder's files, links and skipped paths (module docstring, WHAT and
    PARTIAL), re-reading only what changed since ``previous``. ``limits``
    (``max_files``, ``max_total_bytes``) is a test seam for the caps."""
    files: Dict[str, Any] = {}
    links: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    objects = store / "objects"
    before = (previous or {}).get("files") or {}
    counted = {"count": 0, "total": 0}
    started = time.time_ns() // 1_000_000
    max_files = (limits or {}).get("max_files", MAX_FILES)
    max_total = (limits or {}).get("max_total_bytes", MAX_TOTAL_BYTES)
    partial = {"hit": False}

    def walk(directory: Path, prefix: str) -> bool:
        """False once a cap has stopped the walk."""
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return True
        for name in names:
            full = directory / name
            rel = f"{prefix}/{name}" if prefix else name
            try:
                info = os.lstat(full)
            except OSError:
                continue
            if stat_mod.S_ISLNK(info.st_mode):
                try:
                    links[rel] = os.readlink(full)
                except OSError:
                    pass  # An unreadable link is not recorded.
            elif stat_mod.S_ISDIR(info.st_mode):
                if name not in EXCLUDED_DIRS and not walk(full, rel):
                    return False
            elif stat_mod.S_ISREG(info.st_mode):
                if info.st_size > MAX_FILE_BYTES:
                    skipped[rel] = SKIPPED_LARGE
                    continue
                if counted["count"] + 1 > max_files or counted["total"] + info.st_size > max_total:
                    partial["hit"] = True
                    return False
                mtime = info.st_mtime_ns // 1_000_000
                old = before.get(rel)
                if old and old.get("mtime_ms") and old.get("size") == info.st_size and old.get("mtime_ms") == mtime and (objects / old["sha256"]).exists():
                    sha256 = old["sha256"]
                else:
                    try:
                        data = full.read_bytes()
                    except OSError:
                        continue
                    sha256 = hashlib.sha256(data).hexdigest()
                    target = objects / sha256
                    if not target.exists():
                        objects.mkdir(parents=True, exist_ok=True, mode=0o700)
                        _write_private(target, data)
                kept_time = mtime if mtime < started - RACY_MS else 0
                files[rel] = {"sha256": sha256, "size": info.st_size, "mode": info.st_mode & 0o777, "mtime_ms": kept_time}
                counted["count"] += 1
                counted["total"] += info.st_size
        return True

    walk(Path(folder), "")
    state: Dict[str, Any] = {"files": files, "links": links, "skipped": skipped}
    if partial["hit"]:
        state["partial"] = True
    return state


def _same_state(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    def content(m: Dict[str, Any]) -> Any:
        return (
            sorted((k, v["sha256"], v["mode"]) for k, v in m["files"].items()),
            sorted(m["links"].items()),
            sorted(m["skipped"].items()),
            m.get("partial") is True,
        )

    return content(a) == content(b)


def _manifest_names(store: Path) -> List[str]:
    try:
        return [n for n in os.listdir(store) if n.endswith(".json") and is_checkpoint_id(n[: -len(".json")])]
    except OSError:
        return []


def _read_manifest(store: Path, name: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((store / name).read_text())
    except (OSError, ValueError):
        return None  # A manifest that cannot be read is not a snapshot to offer.
    if isinstance(data, dict) and data.get("version") == 1 and is_checkpoint_id(str(data.get("id"))):
        return data
    return None


def _newest_first(manifests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(manifests, key=lambda m: (m.get("created_at") or "", m["id"]), reverse=True)


def list_checkpoints(store: Path) -> List[Dict[str, Any]]:
    """Every snapshot of the folder, newest first."""
    return _newest_first([m for m in (_read_manifest(store, n) for n in _manifest_names(store)) if m is not None])


def latest_checkpoint(store: Path) -> Optional[Dict[str, Any]]:
    """The newest snapshot, reading only the manifests that could be it: ids
    sort by their UTC second, so only the newest second's (usually one) are
    read. Taking a snapshot used to read every kept manifest (up to ``KEEP``)."""
    candidates: List[Dict[str, Any]] = []
    second: Optional[str] = None
    for name in sorted(_manifest_names(store), reverse=True):
        if second is not None and name[:_ID_SECOND_LENGTH] != second:
            break
        manifest = _read_manifest(store, name)
        if manifest is None:
            continue
        candidates.append(manifest)
        second = name[:_ID_SECOND_LENGTH]
    ordered = _newest_first(candidates)
    return ordered[0] if ordered else None


def _prune(store: Path) -> None:
    """Keep the newest ``KEEP`` manifests, and the objects they use."""
    if len(_manifest_names(store)) <= KEEP:
        return  # Nothing to drop: no manifest need be read.
    everything = list_checkpoints(store)
    for old in everything[KEEP:]:
        try:
            (store / f"{old['id']}.json").unlink()
        except OSError:
            pass
    used = {f["sha256"] for m in everything[:KEEP] for f in m["files"].values()}
    objects = store / "objects"
    try:
        names = os.listdir(objects)
    except OSError:
        return
    for name in names:
        if name not in used:
            try:
                (objects / name).unlink()
            except OSError:
                pass


def take_snapshot(
    folder: Path,
    label: str,
    store: Optional[Path] = None,
    now: Optional[datetime] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Take a snapshot of ``folder``, labelled ``label``. Answers the snapshot,
    or the newest one when nothing changed since it (module docstring)."""
    store = store or checkpoints_dir(folder)
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    latest = latest_checkpoint(store)
    state = scan_folder(folder, store, latest, limits)
    if latest is not None and _same_state(latest, state):
        return latest
    moment = now or datetime.now(timezone.utc)
    manifest = {"version": 1, "id": new_checkpoint_id(moment), "created_at": _iso(moment), "label": label, **state}
    _write_private(store / f"{manifest['id']}.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    _prune(store)
    return manifest


@dataclass
class RestorePlan:
    """What a restore would do: files it writes back, files it removes, files it leaves alone."""

    write: List[str] = field(default_factory=list)
    remove: List[str] = field(default_factory=list)
    link: List[str] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)


def plan_restore(target: Dict[str, Any], now: Dict[str, Any]) -> RestorePlan:
    """The difference between how the folder is (``now``) and a snapshot (``target``), as a restore would settle it."""
    tfiles, nfiles = target["files"], now["files"]
    write = sorted(
        rel
        for rel, entry in tfiles.items()
        if (nfiles.get(rel) or {}).get("sha256") != entry["sha256"]
        or (nfiles.get(rel) or {}).get("mode") != entry["mode"]
        or rel in now["links"]
    )
    link = sorted(rel for rel, value in target["links"].items() if now["links"].get(rel) != value)
    # A partial snapshot cannot tell a file made since from one past its cap
    # (module docstring, PARTIAL): it removes nothing.
    remove = (
        []
        if target.get("partial") is True
        else sorted(
            rel
            for rel in [*nfiles.keys(), *now["links"].keys()]
            if rel not in tfiles and rel not in target["links"] and rel not in target["skipped"]
        )
    )
    return RestorePlan(write=write, remove=remove, link=link, keep=sorted(target["skipped"]))


def _safe_relative(rel: str) -> bool:
    if not rel or rel.startswith("/") or "\\" in rel:
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def _inside_folder(folder: Path, rel: str) -> bool:
    """Whether writing at ``rel`` stays inside ``folder``: no part of the way there is a symlink."""
    at = Path(folder)
    for part in rel.split("/")[:-1]:
        at = at / part
        try:
            info = os.lstat(at)
        except OSError:
            return True  # Not there yet: it will be made as a directory.
        if stat_mod.S_ISLNK(info.st_mode) or not stat_mod.S_ISDIR(info.st_mode):
            return False
    return True


@dataclass
class RestoreResult:
    written: List[str]
    removed: List[str]
    kept: List[str]
    #: Paths it could not put back, and why.
    failed: List[Dict[str, str]]
    #: The snapshot of how things were, taken first.
    before: Dict[str, Any]


_NOT_ITS_OWN = "the way to it is not a folder of its own"


def restore_snapshot(folder: Path, checkpoint_id: str, before_label: str, store: Optional[Path] = None) -> RestoreResult:
    """Put ``folder`` back as snapshot ``checkpoint_id`` has it (module docstring,
    RESTORE), after taking a snapshot labelled ``before_label``."""
    if not is_checkpoint_id(checkpoint_id):
        raise ValueError(f"There is no snapshot {checkpoint_id}.")
    store = store or checkpoints_dir(folder)
    target = json.loads((store / f"{checkpoint_id}.json").read_text())
    before = take_snapshot(folder, before_label, store=store)
    plan = plan_restore(target, before)
    failed: List[Dict[str, str]] = []
    written: List[str] = []
    removed: List[str] = []
    folder = Path(folder)

    for rel in plan.remove:
        if not _safe_relative(rel) or not _inside_folder(folder, rel):
            continue
        try:
            os.unlink(folder.joinpath(*rel.split("/")))
            removed.append(rel)
        except OSError as error:
            failed.append({"path": rel, "reason": str(error)})
    for rel in plan.write:
        if not _safe_relative(rel) or not _inside_folder(folder, rel):
            failed.append({"path": rel, "reason": _NOT_ITS_OWN})
            continue
        target_file = folder.joinpath(*rel.split("/"))
        entry = target["files"][rel]
        try:
            data = (store / "objects" / entry["sha256"]).read_bytes()
            target_file.parent.mkdir(parents=True, exist_ok=True)
            temp = target_file.with_name(f"{target_file.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
            fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, entry["mode"])
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.chmod(temp, entry["mode"])
            # A link where the file belongs is removed, never written through.
            if target_file.is_symlink():
                target_file.unlink()
            os.replace(temp, target_file)
            written.append(rel)
        except OSError as error:
            failed.append({"path": rel, "reason": str(error)})
    for rel in plan.link:
        if not _safe_relative(rel) or not _inside_folder(folder, rel):
            failed.append({"path": rel, "reason": _NOT_ITS_OWN})
            continue
        target_file = folder.joinpath(*rel.split("/"))
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if target_file.is_symlink() or target_file.exists():
                target_file.unlink()
            os.symlink(target["links"][rel], target_file)
            written.append(rel)
        except OSError as error:
            failed.append({"path": rel, "reason": str(error)})
    return RestoreResult(written=sorted(written), removed=removed, kept=plan.keep, failed=failed, before=before)


# ============================================================================
# The words (`tests/fixtures/checkpoints/checkpoints.json`)
# ============================================================================

NOTHING_TO_UNDO = "Nothing to undo in this conversation."
NOTHING_CHANGED = "Nothing to undo: the folder is as it was before your last message."
UNDO_HEADER = "Undo your last message's changes to this folder:"
CONFIRM = "Put these back? [y/N] "
LEFT_AS_IS = "Left as it is."
NO_SNAPSHOTS = "No snapshots of this folder yet."
REWIND_TITLE = "Snapshots of this folder"
REWIND_HINT = "Put the folder back with /rewind <number>."
REWIND_SAME = "Nothing to put back: the folder is as that snapshot has it."
REWIND_MISSING_HINT = "Type /rewind to see the list."
PARTIAL_NOTE = "This folder is larger than a snapshot holds (20000 files, 200 MB), so files made since it are left in place."


def rewind_missing(pick: str) -> str:
    return f"There is no snapshot {pick}."


def failed_sentence(rel: str, reason: str) -> str:
    return f"Could not put back {rel}: {reason}."


def snapshot_failed(reason: str) -> str:
    return f"Snapshots are off for this conversation: {reason}."


def turn_label(message: str) -> str:
    """What a snapshot taken before a message is called: ``before "<the message, cut to 50>"``."""
    line = " ".join(message.split())
    return f'before "{line[:49] + "…" if len(line) > 50 else line}"'


def rewind_header(when: str, label: str) -> str:
    """The header of a ``/rewind <n>`` confirmation."""
    return f"Put the folder back as it was {when} ({label}):"


def plan_lines(plan: RestorePlan, limit: int = 12) -> List[str]:
    """The lines a confirmation lists: what goes back, what goes, at most ``limit``."""
    rows = [f"  restore  {rel}" for rel in sorted([*plan.write, *plan.link])] + [f"  remove   {rel}" for rel in plan.remove]
    return rows[:limit] + [f"  and {len(rows) - limit} more"] if len(rows) > limit else rows


def restored_sentence(written: int, removed: int) -> str:
    """``Put back 2 files and removed 1 file made since.``"""

    def files(n: int) -> str:
        return f"{n} {'file' if n == 1 else 'files'}"

    parts = ([f"put back {files(written)}"] if written else []) + ([f"removed {files(removed)} made since"] if removed else [])
    text = " and ".join(parts)
    return f"{text[:1].upper()}{text[1:]}."


def plan_changes_anything(plan: RestorePlan) -> bool:
    return bool(plan.write or plan.remove or plan.link)
