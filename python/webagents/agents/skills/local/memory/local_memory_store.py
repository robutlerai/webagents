"""
The local memory tier (gap-closure plan item 2.1, principle 3, 2026-09-26):
Markdown files a person can read and edit, indexed for search, with an
append-only log the portal sync reads.

LAYOUT, under ``<agent dir>/.webagents/memory/``::

    owner/<key>.md                 the owner's notes
    shared/<key>.md                notes the owner shares with every caller
    callers/<hash>/<key>.md        one folder per verified caller (memory_namespace.py)
    callers/<hash>/caller.json     which caller: {"principal": "user:..."}
    index.db                       the full-text index (sqlite3, FTS5)
    log.jsonl                      every local change, in order, for the sync

THE FILES ARE THE TRUTH. Each file carries its entry's front matter (id, key,
namespace, source, created_at, updated_at) and its content. The index is
rebuilt from the files every time the store opens, so a note edited or dropped
in by hand is found on the next start, and a lost or corrupt index costs
nothing. The index is the standard library's ``sqlite3`` with FTS5; where the
interpreter's SQLite has no FTS5, a plain in-memory index ranks by matched
words instead. Embeddings are the portal tier's, never a local dependency.

The TypeScript twin is ``typescript/src/skills/memory/local-store.ts``,
writing the same files, so an agent folder can move between the CLIs.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .memory_namespace import CALLER_PREFIX, NAMESPACE_RE, entry_id_for, is_valid_key, key_refusal, local_dir_of

SOURCES = ("tool", "compaction", "owner", "sync")

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_FRONT_MATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", re.DOTALL)


def query_tokens(query: str) -> List[str]:
    """The words of a query, lower-cased, without repeats; ``[]`` for a query with none."""
    seen: List[str] = []
    for token in _TOKEN.findall(query.lower()):
        if token not in seen:
            seen.append(token)
    return seen


def _source_of(value: Any) -> str:
    return value if isinstance(value, str) and value in SOURCES else "tool"


def _now_iso(now: Optional[Callable[[], datetime]] = None) -> str:
    stamp = (now() if now else datetime.now(timezone.utc)).astimezone(timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


@dataclass
class MemoryEntry:
    id: str
    namespace: str
    key: str
    content: str
    source: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def render_entry_file(entry: MemoryEntry) -> str:
    return "\n".join(
        [
            "---",
            f"id: {entry.id}",
            f"key: {entry.key}",
            f"namespace: {entry.namespace}",
            f"source: {entry.source}",
            f"created_at: {entry.created_at}",
            f"updated_at: {entry.updated_at}",
            "---",
            entry.content,
            "",
        ]
    )


def parse_entry_file(text: str, namespace: str, key: str, store: str) -> Optional[MemoryEntry]:
    """The entry in a file: its front matter's, or a note written by hand."""
    m = _FRONT_MATTER.match(text)
    if not m:
        # A NOTE WRITTEN BY HAND (2026-09-27). The docs say the files can be
        # edited by hand, and a plain `.md` with no front matter was ignored
        # without a word. Under a folder whose namespace is known (`owner/`,
        # `shared/`, a caller folder with its caller.json) it is that
        # namespace's note, keyed by its file name and marked `owner`; it gets
        # front matter the next time the store writes it. A file in a caller
        # folder that names no caller stays nobody's, as below. The
        # TypeScript store reads it the same way
        # (`tests/fixtures/memory_tool/chat_fixes_hand_written_note.json`).
        if not NAMESPACE_RE.match(namespace or "") or not text.strip():
            return None
        now = _now_iso()
        return MemoryEntry(
            id=entry_id_for(store, namespace, key),
            namespace=namespace,
            key=key,
            content=text.rstrip("\n"),
            source="owner",
            created_at=now,
            updated_at=now,
        )
    fields: Dict[str, str] = {}
    for line in re.split(r"\r?\n", m.group(1)):
        colon = line.find(":")
        if colon <= 0:
            continue
        fields[line[:colon].strip()] = line[colon + 1 :].strip()
    real_key = fields.get("key") if is_valid_key(fields.get("key")) else key
    named = fields.get("namespace", "")
    if named and NAMESPACE_RE.match(named):
        real_namespace = named
    elif NAMESPACE_RE.match(namespace or ""):
        real_namespace = namespace
    else:
        # A file in a caller folder that names no caller (no caller.json, no
        # namespace line) is nobody's: it is left alone rather than misfiled.
        return None
    content = m.group(2)
    if content.endswith("\n"):
        content = content[:-1]
    now = _now_iso()
    return MemoryEntry(
        id=entry_id_for(store, real_namespace, real_key),
        namespace=real_namespace,
        key=real_key,
        content=content,
        source=_source_of(fields.get("source")),
        created_at=fields.get("created_at") or now,
        updated_at=fields.get("updated_at") or now,
    )


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


class PlainMemoryIndex:
    """Ranks by matched words: a word in the key counts double. Used where FTS5 is not."""

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}

    def reset(self, entries: Iterable[MemoryEntry]) -> None:
        self._entries = {e.id: e for e in entries}

    def upsert(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry

    def remove(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def search(self, query: str, namespaces: Optional[Sequence[str]], limit: int) -> List[str]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        scored = []
        for e in self._entries.values():
            if namespaces is not None and e.namespace not in namespaces:
                continue
            key, content = e.key.lower(), e.content.lower()
            score = sum((2 if t in key else 0) + (1 if t in content else 0) for t in tokens)
            if score > 0:
                scored.append((-score, e.updated_at, e.id))
        # Highest score first, then the newest.
        scored.sort(key=lambda s: (s[0], _reverse(s[1])))
        return [s[2] for s in scored[:limit]]

    def close(self) -> None:
        self._entries.clear()


def _reverse(text: str) -> str:
    """A sort key that orders ISO timestamps newest first."""
    return "".join(chr(0x10FFFF - ord(c)) for c in text)


class SqliteMemoryIndex:
    """FTS5 in the standard library's SQLite. ``open`` answers None where FTS5 is not compiled in."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    @classmethod
    def open(cls, file: Path) -> Optional["SqliteMemoryIndex"]:
        try:
            connection = sqlite3.connect(str(file))
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5("
                "id UNINDEXED, namespace UNINDEXED, key, content, tokenize='unicode61')"
            )
            connection.commit()
        except sqlite3.Error:
            return None
        return cls(connection)

    def reset(self, entries: Iterable[MemoryEntry]) -> None:
        self._db.execute("DELETE FROM entries")
        self._db.executemany(
            "INSERT INTO entries (id, namespace, key, content) VALUES (?, ?, ?, ?)",
            [(e.id, e.namespace, e.key, e.content) for e in entries],
        )
        self._db.commit()

    def upsert(self, entry: MemoryEntry) -> None:
        self._db.execute("DELETE FROM entries WHERE id = ?", (entry.id,))
        self._db.execute(
            "INSERT INTO entries (id, namespace, key, content) VALUES (?, ?, ?, ?)",
            (entry.id, entry.namespace, entry.key, entry.content),
        )
        self._db.commit()

    def remove(self, entry_id: str) -> None:
        self._db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._db.commit()

    def search(self, query: str, namespaces: Optional[Sequence[str]], limit: int) -> List[str]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        # Each word as a quoted prefix term, any of them: a search is a lookup,
        # not a boolean expression, and quoting keeps FTS5 syntax out of it.
        match = " OR ".join('"' + t.replace('"', '""') + '"*' for t in tokens)
        params: List[Any] = [match]
        where = "entries MATCH ?"
        if namespaces is not None:
            if not namespaces:
                return []
            where += " AND namespace IN (" + ", ".join("?" for _ in namespaces) + ")"
            params.extend(namespaces)
        params.append(limit)
        rows = self._db.execute(f"SELECT id FROM entries WHERE {where} ORDER BY bm25(entries) LIMIT ?", params).fetchall()
        return [str(r[0]) for r in rows]

    def close(self) -> None:
        try:
            self._db.close()
        except sqlite3.Error:
            pass


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def _write_private(file: Path, text: str) -> None:
    temp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(str(temp), str(file))


class LocalMemoryStore:
    def __init__(
        self,
        root: Path,
        store: str,
        plain_index: bool = False,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self._now = now
        self._force_plain = plain_index
        self._entries: Dict[str, MemoryEntry] = {}
        self._index: Any = PlainMemoryIndex()
        self._last_seq = 0
        self._opened = False
        #: Which index serves: ``sqlite`` or ``plain``, for ``doctor`` and the tests.
        self.index_kind = "plain"

    @property
    def log_file(self) -> Path:
        return self.root / "log.jsonl"

    def open(self) -> None:
        """Opens once: makes the folders, reads the files, rebuilds the index, finds the log's last seq."""
        if self._opened:
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._entries = {e.id: e for e in self._read_all_files()}
        if not self._force_plain:
            sqlite_index = SqliteMemoryIndex.open(self.root / "index.db")
            if sqlite_index is not None:
                self._index = sqlite_index
                self.index_kind = "sqlite"
        self._index.reset(list(self._entries.values()))
        self._last_seq = max((line["seq"] for line in self._read_log()), default=0)
        self._opened = True

    def close(self) -> None:
        self._index.close()
        self._opened = False

    # -- files -----------------------------------------------------------

    def _dir_for(self, namespace: str) -> Path:
        return self.root / local_dir_of(namespace)

    def _file_for(self, namespace: str, key: str) -> Path:
        return self._dir_for(namespace) / f"{key}.md"

    def _read_all_files(self) -> Iterable[MemoryEntry]:
        dirs: List[tuple] = [(self.root / name, name) for name in ("owner", "shared")]
        callers = self.root / "callers"
        if callers.is_dir():
            for hashed in sorted(p for p in callers.iterdir() if p.is_dir()):
                namespace = ""
                try:
                    meta = json.loads((hashed / "caller.json").read_text(encoding="utf-8"))
                    if isinstance(meta.get("principal"), str):
                        namespace = f"{CALLER_PREFIX}{meta['principal']}"
                except (OSError, ValueError):
                    pass  # no caller.json: the files' own front matter names the namespace
                dirs.append((hashed, namespace))
        for directory, namespace in dirs:
            if not directory.is_dir():
                continue
            for file in sorted(directory.iterdir()):
                if file.suffix != ".md":
                    continue
                key = file.name[:-3]
                if not is_valid_key(key):
                    continue
                try:
                    text = file.read_text(encoding="utf-8")
                except OSError:
                    continue
                entry = parse_entry_file(text, namespace, key, self.store)
                if entry is not None and (entry.namespace == namespace if namespace else True):
                    yield entry

    def _write_file(self, entry: MemoryEntry) -> None:
        directory = self._dir_for(entry.namespace)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if entry.namespace.startswith(CALLER_PREFIX):
            meta = directory / "caller.json"
            if not meta.exists():
                _write_private(meta, json.dumps({"principal": entry.namespace[len(CALLER_PREFIX):]}) + "\n")
        _write_private(self._file_for(entry.namespace, entry.key), render_entry_file(entry))

    # -- log -------------------------------------------------------------

    def _read_log(self) -> List[Dict[str, Any]]:
        try:
            text = self.log_file.read_text(encoding="utf-8")
        except OSError:
            return []
        lines: List[Dict[str, Any]] = []
        for raw in text.split("\n"):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except ValueError:
                continue  # a torn last line: the next append starts a clean one
            if isinstance(line, dict) and isinstance(line.get("seq"), int) and line.get("op") in ("put", "delete"):
                lines.append(line)
        return lines

    def _append_log(self, line: Dict[str, Any]) -> Dict[str, Any]:
        full = {"seq": self._last_seq + 1, **line}
        fd = os.open(str(self.log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(full) + "\n")
        self._last_seq = full["seq"]
        return full

    def log_since(self, seq: int, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """The local changes after ``seq``, oldest first, for one namespace or all."""
        return [l for l in self._read_log() if l["seq"] > seq and (namespace is None or l.get("namespace") == namespace)]

    @property
    def last_log_seq(self) -> int:
        return self._last_seq

    # -- reads -----------------------------------------------------------

    def get(self, namespace: str, key: str) -> Optional[MemoryEntry]:
        self.open()
        return self._entries.get(entry_id_for(self.store, namespace, key))

    def list(
        self,
        namespaces: Optional[Sequence[str]],
        prefix: Optional[str] = None,
        limit: int = 50,
        exclude_sources: Sequence[str] = (),
    ) -> List[MemoryEntry]:
        """Entries in ``namespaces`` (all when None), newest first."""
        self.open()
        out = [
            e
            for e in self._entries.values()
            if (namespaces is None or e.namespace in namespaces)
            and (not prefix or e.key.startswith(prefix))
            and e.source not in exclude_sources
        ]
        out.sort(key=lambda e: (_reverse(e.updated_at), e.key))
        return out[:limit]

    def search(self, query: str, namespaces: Optional[Sequence[str]], limit: int = 10) -> List[MemoryEntry]:
        self.open()
        out: List[MemoryEntry] = []
        for entry_id in self._index.search(query, namespaces, limit):
            e = self._entries.get(entry_id)
            # The index is checked against the namespaces again here: an index
            # is a ranking, never the access decision.
            if e is not None and (namespaces is None or e.namespace in namespaces):
                out.append(e)
        return out

    def namespaces(self) -> List[str]:
        """Every namespace with at least one entry."""
        self.open()
        return sorted({e.namespace for e in self._entries.values()})

    # -- writes ----------------------------------------------------------

    def put(self, namespace: str, key: str, content: str, source: str = "tool", at: Optional[str] = None) -> MemoryEntry:
        self.open()
        if not is_valid_key(key):
            raise ValueError(key_refusal(key))
        if not NAMESPACE_RE.match(namespace):
            raise ValueError(f"memory: not a namespace: {json.dumps(namespace)}")
        entry = self._put_quietly(namespace, key, content, source, at or _now_iso(self._now))
        self._append_log(
            {"op": "put", "id": entry.id, "namespace": namespace, "key": key, "content": content, "source": source, "at": entry.updated_at}
        )
        return entry

    def _put_quietly(self, namespace: str, key: str, content: str, source: str, at: str) -> MemoryEntry:
        entry_id = entry_id_for(self.store, namespace, key)
        existing = self._entries.get(entry_id)
        entry = MemoryEntry(
            id=entry_id,
            namespace=namespace,
            key=key,
            content=content,
            source=source,
            created_at=existing.created_at if existing else at,
            updated_at=at,
        )
        self._write_file(entry)
        self._entries[entry_id] = entry
        self._index.upsert(entry)
        return entry

    def forget(self, namespace: str, key: str) -> bool:
        self.open()
        removed = self._forget_quietly(namespace, key)
        if removed is not None:
            self._append_log({"op": "delete", "id": removed.id, "namespace": namespace, "key": key, "at": _now_iso(self._now)})
        return removed is not None

    def _forget_quietly(self, namespace: str, key: str) -> Optional[MemoryEntry]:
        entry_id = entry_id_for(self.store, namespace, key)
        existing = self._entries.get(entry_id)
        if existing is None:
            return None
        try:
            self._file_for(namespace, key).unlink()
        except OSError:
            pass  # already gone
        del self._entries[entry_id]
        self._index.remove(entry_id)
        return existing

    def apply(self, lines: Sequence[Dict[str, Any]]) -> int:
        """Changes from the other tier, applied without logging (they are not
        this machine's to push back). Last writer wins by time; the count is
        what was newer than what was here."""
        self.open()
        applied = 0
        for line in lines:
            key, namespace, at = line.get("key"), line.get("namespace"), line.get("at")
            if not is_valid_key(key) or not isinstance(namespace, str) or not NAMESPACE_RE.match(namespace) or not isinstance(at, str):
                continue
            entry_id = entry_id_for(self.store, namespace, key)
            existing = self._entries.get(entry_id)
            if existing is not None and existing.updated_at >= at:
                continue
            if line.get("op") == "put":
                source = _source_of(line.get("source"))
                content = line.get("content") if isinstance(line.get("content"), str) else ""
                self._put_quietly(namespace, key, content, "sync" if source == "tool" else source, at)
                applied += 1
            elif existing is not None:
                self._forget_quietly(namespace, key)
                applied += 1
        return applied
