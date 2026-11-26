"""Where CASFS blobs come from, and the platform API the commands use.

A backend stores blobs by hash and answers ranged reads; it never interprets a
manifest, and nothing it returns is trusted (``project.py`` verifies every
manifest, chunk and small file it reads).

* ``PlatformBackend``: ``/api/casfs/projects/<id>/blobs/<sha>`` on a Robutler
  portal, under a bearer token and, optionally, a CASFS grant for that project
  (``X-Casfs-Grant``).
* ``LocalDirBackend``: ``<dir>/cas/<ab>/<sha>``, the store's own layout.
* ``MemoryBackend``: tests and scratch.

The machine's chunk cache (``DirChunkCache``) is content addressed and shared
by every project and mount on the machine; bytes are cached exactly as
stored, so an encrypted or sealed source is cached as ciphertext.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import httpx

from .core import CasError, is_sha, sha256_hex


class CasBackend(Protocol):
    name: str

    async def has(self, sha: str) -> bool: ...

    async def read_range(self, sha: str, start: int, end: int) -> bytes: ...

    async def put(self, data: bytes, sha: str) -> None: ...


class ChunkCache(Protocol):
    async def get(self, sha: str) -> Optional[bytes]: ...

    async def put(self, sha: str, data: bytes) -> None: ...


class MemoryChunkCache:
    def __init__(self, limit_bytes: int = 256 * 1024 * 1024):
        self._map: Dict[str, bytes] = {}
        self._used = 0
        self._limit = limit_bytes

    async def get(self, sha: str) -> Optional[bytes]:
        return self._map.get(sha)

    async def put(self, sha: str, data: bytes) -> None:
        if sha in self._map or len(data) > self._limit:
            return
        self._map[sha] = data
        self._used += len(data)
        while self._used > self._limit and self._map:
            k = next(iter(self._map))
            self._used -= len(self._map.pop(k))


class MemoryBackend:
    def __init__(self, name: str = "memory"):
        self.name = name
        self.blobs: Dict[str, bytes] = {}
        self.bytes_read = 0

    async def has(self, sha: str) -> bool:
        return sha in self.blobs

    async def read_range(self, sha: str, start: int, end: int) -> bytes:
        b = self.blobs.get(sha)
        if b is None:
            raise CasError("Cas:NotFound", f"blob {sha} is not in {self.name}", {"sha": sha})
        out = b[start:end]
        self.bytes_read += len(out)
        return out

    async def put(self, data: bytes, sha: str) -> None:
        self.blobs.setdefault(sha, bytes(data))


def _blob_path(root: Path, sha: str) -> Path:
    return root / "cas" / sha[:2] / sha


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class LocalDirBackend:
    """A local CASFS store: ``<dir>/cas/<ab>/<sha256>`` and ``<dir>/heads/<projectId>``."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.dir = Path(directory)
        self.name = f"local:{self.dir}"

    async def has(self, sha: str) -> bool:
        return is_sha(sha) and _blob_path(self.dir, sha).exists()

    async def read_range(self, sha: str, start: int, end: int) -> bytes:
        if not is_sha(sha):
            raise CasError("Cas:NotFound", "not a sha256")
        p = _blob_path(self.dir, sha)
        try:
            with open(p, "rb") as f:
                f.seek(start)
                return f.read(max(0, end - start))
        except FileNotFoundError:
            raise CasError("Cas:NotFound", f"blob {sha} is not in {self.dir}", {"sha": sha})

    async def put(self, data: bytes, sha: str) -> None:
        if sha256_hex(data) != sha:
            raise CasError("Cas:HashMismatch", "the bytes do not hash to the name", {"expected": sha})
        p = _blob_path(self.dir, sha)
        if not p.exists():
            _atomic_write(p, data)


class LocalHeads:
    """The head of a project kept in a local directory, moved by compare-and-swap."""

    def __init__(self, directory: str | os.PathLike[str], project_id: str):
        self.path = Path(directory) / "heads" / project_id

    async def get(self) -> Optional[str]:
        try:
            s = self.path.read_text().strip()
        except FileNotFoundError:
            return None
        return s if is_sha(s) else None

    async def swap(self, base: Optional[str], nxt: str) -> Dict[str, Any]:
        cur = await self.get()
        if cur != base:
            return {"ok": False, "current": cur}
        _atomic_write(self.path, (nxt + "\n").encode())
        return {"ok": True}


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/robutler/casfs``, else ``~/.cache/robutler/casfs``."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "robutler" / "casfs"


class DirChunkCache:
    """Content-addressed chunk cache on disk; every read is checked against its name."""

    def __init__(self, directory: str | os.PathLike[str] | None = None):
        self.dir = Path(directory) if directory else default_cache_dir()

    async def get(self, sha: str) -> Optional[bytes]:
        if not is_sha(sha):
            return None
        try:
            b = _blob_path(self.dir, sha).read_bytes()
        except FileNotFoundError:
            return None
        return b if sha256_hex(b) == sha else None

    async def put(self, sha: str, data: bytes) -> None:
        if not is_sha(sha) or sha256_hex(data) != sha:
            return
        p = _blob_path(self.dir, sha)
        if not p.exists():
            _atomic_write(p, data)


def local_package_backend(packages_dir: Optional[str], source_id: str) -> Optional[LocalDirBackend]:
    """A local package store for one source project (``<packages_dir>/<sourceId>/cas``), if this machine holds one."""
    if not packages_dir:
        return None
    d = Path(packages_dir) / source_id
    return LocalDirBackend(d) if (d / "cas").is_dir() else None


class PlatformApi:
    """The CASFS HTTP API of a portal: blobs, heads, commits, history and mounts."""

    def __init__(self, base_url: str, token: Optional[str] = None, grants: Optional[Dict[str, str]] = None, client: Optional[httpx.AsyncClient] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.grants = grants or {}
        self.client = client or httpx.AsyncClient(timeout=60.0)

    def headers(self, project_id: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = dict(extra or {})
        if self.token:
            h["authorization"] = f"Bearer {self.token}"
        g = self.grants.get(project_id)
        if g:
            h["x-casfs-grant"] = g
        return h

    def url(self, project_id: str, rest: str = "") -> str:
        return f"{self.base_url}/api/casfs/projects/{project_id}{rest}"

    async def raw(self, method: str, project_id: str, rest: str, **kw: Any) -> httpx.Response:
        headers = self.headers(project_id, kw.pop("headers", None))
        return await self.client.request(method, self.url(project_id, rest), headers=headers, **kw)

    async def json(self, method: str, project_id: str, rest: str, **kw: Any) -> Dict[str, Any]:
        r = await self.raw(method, project_id, rest, **kw)
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 400 or not isinstance(body, dict) or body.get("ok") is False:
            err = (body or {}).get("error") or {}
            default = "Cas:NotFound" if r.status_code == 404 else "Cas:Forbidden" if r.status_code == 401 else "Cas:SourceOffline"
            msg = str(err.get("message") or f"HTTP {r.status_code}")
            if ": " in msg and msg.startswith("Cas:"):
                msg = msg.split(": ", 1)[1]
            raise CasError(err.get("code") or default, msg, err.get("args") or {})
        return body

    async def head(self, project_id: str) -> Dict[str, Any]:
        return await self.json("GET", project_id, "")

    async def commit(self, project_id: str, base: Optional[str], root: str, message: Optional[str] = None) -> Dict[str, Any]:
        return await self.json("POST", project_id, "/commit", json={"base": base, "root": root, "message": message})

    async def history(self, project_id: str, limit: int = 50) -> Dict[str, Any]:
        return await self.json("GET", project_id, f"/versions?limit={limit}")

    async def mounts(self, project_id: str, root: Optional[str] = None) -> Dict[str, Any]:
        return await self.json("GET", project_id, f"/mounts?root={root}" if root else "/mounts")

    async def mount(self, project_id: str, path: str, source: Dict[str, Any], root: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"path": path, "source": source}
        if root:
            body["root"] = root
        return await self.json("POST", project_id, "/mounts", json=body)

    async def ensure_pdk_mount(self, project_id: str, pdk: Dict[str, str], commit: bool) -> Dict[str, Any]:
        return await self.json("POST", project_id, "/mounts", json={"pdk": pdk, "commit": commit})

    async def update_mount(self, project_id: str, path: str, root: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"path": path}
        if root:
            body["root"] = root
        return await self.json("PATCH", project_id, "/mounts", json=body)

    async def unmount(self, project_id: str, path: str) -> Dict[str, Any]:
        return await self.json("DELETE", project_id, "/mounts", params={"path": path})


class PlatformBackend:
    """One project's blobs on the platform."""

    def __init__(self, api: PlatformApi, project_id: str):
        self.api = api
        self.project_id = project_id
        self.name = f"platform:{project_id[:8]}"

    async def has(self, sha: str) -> bool:
        r = await self.api.raw("HEAD", self.project_id, f"/blobs/{sha}")
        return r.status_code == 200

    async def read_range(self, sha: str, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        r = await self.api.raw("GET", self.project_id, f"/blobs/{sha}", headers={"range": f"bytes={start}-{end - 1}"})
        if r.status_code in (403, 404):
            try:
                err = (r.json() or {}).get("error") or {}
            except ValueError:
                err = {}
            raise CasError(err.get("code") or ("Cas:NotFound" if r.status_code == 404 else "Cas:Forbidden"), str(err.get("message") or f"blob {sha} is not readable here"), {"sha": sha})
        if r.status_code not in (200, 206):
            raise CasError("Cas:SourceOffline", f"HTTP {r.status_code} reading {sha}", {"sha": sha})
        return r.content[start:end] if r.status_code == 200 else r.content

    async def put(self, data: bytes, sha: str) -> None:
        await self.api.json("PUT", self.project_id, f"/blobs/{sha}", content=data, headers={"content-type": "application/octet-stream"})
