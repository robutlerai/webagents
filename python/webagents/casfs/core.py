"""CASFS core: the rules every CASFS reader and writer shares.

A CASFS project is a Merkle tree of ``casfs.dir/1`` directory manifests over
content-addressed blobs. A manifest is canonical JSON (sorted keys, no
whitespace, integers only) named by the SHA-256 of its bytes. Entries are
files, folders, sealed folders and mounts. Everything read from anywhere is
hostile input and is refused by name with a ``Cas:*`` code, never repaired.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

DIR_KIND = "casfs.dir/1"
RUN_KIND = "casfs.run/1"
POLICIES = ("cas", "both", "local-served", "local-only")
CHUNK_UNIT = 256 * 1024
CHUNK_DEFAULT = 4 * 1024 * 1024
CHUNK_MAX = 64 * 1024 * 1024
INLINE_CHUNK_HASHES = 1024
MAX_NAME_BYTES = 255
MAX_DEPTH = 64
MAX_ENTRIES = 100_000
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
#: Mounts nest at most this deep (a mount inside a mounted source counts one more).
MAX_MOUNT_DEPTH = 4

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PKG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_KEY_ID = re.compile(r"^[0-9a-f]{16,64}$")
_BAD_NAME = re.compile(r"[/\\\x00-\x1f\x7f]")


class CasError(Exception):
    """A named refusal: ``code`` is one of the ``Cas:*`` codes, ``args_`` its details."""

    def __init__(self, code: str, message: str, args: Optional[Dict[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details: Dict[str, Any] = args or {}


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha(s: Any) -> bool:
    return isinstance(s, str) and bool(_SHA.match(s))


def is_pkg_name(s: Any) -> bool:
    return isinstance(s, str) and bool(_PKG.match(s))


def canon(v: Any) -> str:
    if v is None or isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, int):
        if abs(v) > 2**53 - 1:
            raise CasError("Cas:ManifestInvalid", f"number {v} is not a safe integer")
        return str(v)
    if isinstance(v, float):
        raise CasError("Cas:ManifestInvalid", f"number {v} is not a safe integer")
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = sorted(v.keys(), key=_utf16_key)
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canon(v[k]) for k in keys) + "}"
    raise CasError("Cas:ManifestInvalid", f"value of type {type(v).__name__} cannot be canonical")


def _utf16_key(k: str) -> List[int]:
    b = k.encode("utf-16-be")
    return [int.from_bytes(b[i : i + 2], "big") for i in range(0, len(b), 2)]


def canon_bytes(v: Any) -> bytes:
    return canon(v).encode("utf-8")


def check_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise CasError("Cas:PathInvalid", "empty name", {"name": name})
    if name != unicodedata.normalize("NFC", name):
        raise CasError("Cas:PathInvalid", f"name is not NFC: {name!r}", {"name": name})
    if name in (".", ".."):
        raise CasError("Cas:PathInvalid", f"name {name} is reserved", {"name": name})
    if _BAD_NAME.search(name):
        raise CasError("Cas:PathInvalid", f"name has a separator or control character: {name!r}", {"name": name})
    if len(name.encode("utf-8")) > MAX_NAME_BYTES:
        raise CasError("Cas:PathInvalid", f"name longer than {MAX_NAME_BYTES} bytes", {"name": name})
    return name


def split_path(p: Any) -> List[str]:
    if not isinstance(p, str):
        raise CasError("Cas:PathInvalid", "path is not a string")
    s = p.lstrip("/")
    if s == "":
        return []
    parts = s.split("/")
    if len(parts) > MAX_DEPTH:
        raise CasError("Cas:PathInvalid", f"path deeper than {MAX_DEPTH}", {"path": p})
    for part in parts:
        check_name(part)
    return parts


def check_policy(p: Any) -> Optional[str]:
    if p is None:
        return None
    if p not in POLICIES:
        raise CasError("Cas:PolicyInvalid", f"unknown policy {p!r}", {"policy": p})
    return p


def check_chunk_size(n: Any) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n < CHUNK_UNIT or n > CHUNK_MAX or n % CHUNK_UNIT:
        raise CasError("Cas:ManifestInvalid", f"chunk size {n} must be a multiple of {CHUNK_UNIT} between {CHUNK_UNIT} and {CHUNK_MAX}")
    return n


def chunk_count(size: int, cs: int) -> int:
    return 0 if size == 0 else -(-size // cs)


def check_mount_source(s: Any) -> Dict[str, str]:
    """A mount's source as hostile input; returns exactly the source fields."""
    if not isinstance(s, dict):
        raise CasError("Cas:ManifestInvalid", "a mount source is an object")
    if not isinstance(s.get("id"), str) or not _UUID.match(s["id"]):
        raise CasError("Cas:ManifestInvalid", "a mount source names its project id (lower-case uuid)")
    if s.get("kind") == "project":
        if set(s) - {"kind", "id"}:
            raise CasError("Cas:ManifestInvalid", "a project mount source is { kind, id }")
        return {"kind": "project", "id": s["id"]}
    if s.get("kind") == "pdk":
        if set(s) - {"kind", "id", "pkg", "version"}:
            raise CasError("Cas:ManifestInvalid", "a pdk mount source is { kind, id, pkg, version }")
        if not is_pkg_name(s.get("pkg")) or not is_pkg_name(s.get("version")):
            raise CasError("Cas:ManifestInvalid", "pkg and version: 1 to 64 of A-Z a-z 0-9 . _ + -, a letter or digit first")
        return {"kind": "pdk", "id": s["id"], "pkg": s["pkg"], "version": s["version"]}
    raise CasError("Cas:ManifestInvalid", "a mount source kind is 'project' or 'pdk'")


def pdk_mount_path(pkg: str, version: str) -> str:
    """The canonical path of a PDK mount: one per package version, shared by every cell that uses it."""
    return f"pdk/{pkg}@{version}"


def same_mount(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.get("root") == b.get("root") and canon(a.get("source")) == canon(b.get("source"))


def check_dir_manifest(m: Any) -> Dict[str, Any]:
    if not isinstance(m, dict) or m.get("kind") != DIR_KIND or not isinstance(m.get("entries"), dict) or set(m) - {"kind", "entries"}:
        raise CasError("Cas:ManifestInvalid", "not a casfs.dir/1 manifest")
    entries = m["entries"]
    if len(entries) > MAX_ENTRIES:
        raise CasError("Cas:ManifestInvalid", f"more than {MAX_ENTRIES} entries")
    for n, e in entries.items():
        check_name(n)
        if not isinstance(e, dict):
            raise CasError("Cas:ManifestInvalid", f"entry {n} is not an object")
        t = e.get("type")
        if t == "sealed":
            if not (isinstance(e.get("keyId"), str) and _KEY_ID.match(e["keyId"]) and is_sha(e.get("seal"))) or set(e) - {"type", "keyId", "seal"}:
                raise CasError("Cas:ManifestInvalid", f"sealed entry {n} carries only keyId and seal")
            continue
        if t == "mount":
            if set(e) - {"type", "readOnly", "root", "source"}:
                raise CasError("Cas:ManifestInvalid", f"mount {n} carries more than type, readOnly, root and source")
            if e.get("readOnly") is not True:
                raise CasError("Cas:ManifestInvalid", f"mount {n}: readOnly is true")
            if not is_sha(e.get("root")):
                raise CasError("Cas:ManifestInvalid", f"mount {n} has no root")
            check_mount_source(e.get("source"))
            continue
        if t not in ("file", "dir"):
            raise CasError("Cas:ManifestInvalid", f"entry {n} has no valid type")
        check_policy(e.get("policy"))
        if t == "dir":
            if not is_sha(e.get("sha256")):
                raise CasError("Cas:ManifestInvalid", f"entry {n} has no sha256")
            continue
        if not is_sha(e.get("sha256")) and not e.get("growing"):
            raise CasError("Cas:ManifestInvalid", f"entry {n} has no sha256")
        size = e.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CasError("Cas:ManifestInvalid", f"entry {n} has a bad size")
        if "chunks" in e:
            c = e["chunks"]
            k = chunk_count(size, check_chunk_size((c or {}).get("size")))
            if "hashes" in c:
                hs = c["hashes"]
                if not isinstance(hs, list) or len(hs) != k or len(hs) > INLINE_CHUNK_HASHES or not all(is_sha(h) for h in hs):
                    raise CasError("Cas:ManifestInvalid", f"entry {n}: {k} chunk hashes expected")
            elif not is_sha(c.get("index")):
                raise CasError("Cas:ManifestInvalid", f"entry {n}: chunks need hashes or an index")
    return m


def parse_dir_manifest(sha: str, data: bytes) -> Dict[str, Any]:
    if len(data) > MAX_MANIFEST_BYTES:
        raise CasError("Cas:ManifestInvalid", f"manifest {sha} is too large")
    if sha256_hex(data) != sha:
        raise CasError("Cas:HashMismatch", f"directory manifest does not hash to {sha}", {"sha": sha})
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise CasError("Cas:ManifestInvalid", f"manifest {sha} is not JSON")
    check_dir_manifest(m)
    if canon(m).encode("utf-8") != data:
        raise CasError("Cas:ManifestInvalid", f"manifest {sha} is not in canonical form")
    return m
