"""``CasFs``: one interface to CASFS projects on the platform or in a local directory.

The same operations, names and ``Cas:*`` refusals as an app's ``host.cas``, the
TypeScript ``CasFs`` and the portal's HTTP API::

    fs = CasFs.platform("https://robutler.ai", token)
    p = await fs.open(project_id)
    await p.ensure_pdk_mount("sky130A", commit=True)
    data = await p.read("pdk/sky130A@2026.10/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib")

Mount commands on the platform run on the server, which checks that the caller
may read the source, refuses cycles and over-deep chains, and commits. In a
local directory, ``mount`` places the pointer and commits locally; reading
through it uses a local package store or the platform.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .backends import DirChunkCache, MemoryChunkCache, PlatformApi, PlatformBackend, local_package_backend
from .core import CasError
from .project import CasProject


class _Opener:
    def __init__(self, fs: "CasFs"):
        self.fs = fs

    async def open(self, path: str, source: Dict[str, str], root: str) -> Tuple[Any, None]:
        local = local_package_backend(self.fs.packages_dir, source["id"]) if self.fs.packages_dir else None
        if local is not None:
            return local, None
        if self.fs.api is None:
            raise CasError("Cas:SourceOffline", f"{path}: no local copy of its source and no platform to read it from", {"path": path})
        return PlatformBackend(self.fs.api, source["id"]), None


class _PlatformHeads:
    def __init__(self, api: PlatformApi, project_id: str):
        self.api = api
        self.project_id = project_id

    async def swap(self, base: Optional[str], nxt: str) -> Dict[str, Any]:
        try:
            await self.api.commit(self.project_id, base, nxt)
            return {"ok": True}
        except CasError as e:
            if e.code == "Cas:HeadMoved":
                return {"ok": False, "current": e.details.get("head")}
            raise


class CasFs:
    def __init__(self, api: Optional[PlatformApi] = None, cache: Any = None, packages_dir: Optional[str] = None):
        self.api = api
        self.cache = cache
        self.packages_dir = packages_dir

    @classmethod
    def platform(cls, base_url: str, token: Optional[str] = None, grants: Optional[Dict[str, str]] = None, cache: Any = None, packages_dir: Optional[str] = None, client: Any = None) -> "CasFs":
        return cls(PlatformApi(base_url, token, grants, client), cache if cache is not None else MemoryChunkCache(), packages_dir)

    def opener(self) -> _Opener:
        return _Opener(self)

    async def open(self, project_id: str) -> "CasHandle":
        if self.api is None:
            raise CasError("Cas:SourceOffline", "no platform configured")
        h = CasHandle(self, project_id, PlatformBackend(self.api, project_id), _PlatformHeads(self.api, project_id), self.api)
        await h.reload()
        return h

    async def open_local(self, project_id: str, backend: Any, heads: Any) -> "CasHandle":
        h = CasHandle(self, project_id, backend, heads, None)
        await h.reload()
        return h


class CasHandle:
    def __init__(self, fs: CasFs, project_id: str, backend: Any, heads: Any, api: Optional[PlatformApi]):
        self.fs = fs
        self.project_id = project_id
        self.backend = backend
        self.heads = heads
        self.api = api
        self.head: Optional[str] = None
        self.project: CasProject

    async def reload(self) -> None:
        self.head = (await self.api.head(self.project_id)).get("head") if self.api else await self.heads.get()
        self.project = await CasProject.open(self.backend, self.head, heads=self.heads, mounts=self.fs.opener(), cache=self.fs.cache)

    async def list(self, path: str = "") -> List[Dict[str, Any]]:
        return await self.project.list(path)

    async def stat(self, path: str) -> Dict[str, Any]:
        return await self.project.stat(path)

    async def read(self, path: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        return await self.project.read(path, offset, length)

    def read_stream(self, path: str, offset: int = 0, length: Optional[int] = None):
        return self.project.read_stream(path, offset, length)

    async def write(self, path: str, data: bytes | str, media_type: Optional[str] = None, policy: Optional[str] = None) -> Dict[str, Any]:
        return await self.project.write(path, data, media_type, policy)

    async def mkdir(self, path: str) -> None:
        await self.project.mkdir(path)

    async def delete(self, path: str) -> None:
        await self.project.delete(path)

    async def commit(self) -> str:
        self.head = await self.project.commit()
        return self.head

    async def history(self, limit: int = 50) -> Dict[str, Any]:
        if self.api is None:
            raise CasError("Cas:SourceOffline", "history is kept on the platform")
        return await self.api.history(self.project_id, limit)

    # ── mounts ──

    async def mounts(self, root: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.api is not None:
            return (await self.api.mounts(self.project_id, root))["mounts"]
        out: List[Dict[str, Any]] = []

        async def walk(path: str) -> None:
            for e in await self.project.list(path):
                p = f"{path}/{e['name']}" if path else e["name"]
                if e["type"] == "mount":
                    out.append({"path": p, "source": e["source"], "root": e["root"]})
                elif e["type"] == "dir":
                    await walk(p)

        await walk("")
        return out

    async def mount(self, path: str, source: Dict[str, Any], root: Optional[str] = None) -> Dict[str, Any]:
        """Mount another project or a published PDK at ``path``, pinned by hash. One commit."""
        if self.api is not None:
            out = await self.api.mount(self.project_id, path, source, root)
            await self.reload()
            return out
        if not root or not source.get("id") or (source.get("kind") == "pdk" and not source.get("version")):
            raise CasError("Cas:MountSource", "a local mount names the source id, the version of a PDK, and the root to pin")
        src = {"kind": "pdk", "id": source["id"], "pkg": source["pkg"], "version": source["version"]} if source.get("kind") == "pdk" else {"kind": "project", "id": source["id"]}
        try:
            await self.project.stat(path)
            raise CasError("Cas:Exists", f"{path} exists", {"path": path})
        except CasError as e:
            if e.code != "Cas:NotFound":
                raise
        entry = await self.project.place_mount(path, {"type": "mount", "readOnly": True, "root": root, "source": src})
        return {"head": await self.commit(), "path": path, "entry": entry}

    async def unmount(self, path: str) -> Dict[str, Any]:
        if self.api is not None:
            out = await self.api.unmount(self.project_id, path)
            await self.reload()
            return out
        await self.project.unmount(path)
        return {"head": await self.commit(), "path": path}

    async def update_mount(self, path: str, root: Optional[str] = None) -> Dict[str, Any]:
        """Move a project mount's pin (default: the source's current head). A PDK version never moves."""
        if self.api is None:
            raise CasError("Cas:SourceOffline", "moving a pin needs the platform to read the source head")
        out = await self.api.update_mount(self.project_id, path, root)
        await self.reload()
        return out

    async def ensure_pdk_mount(self, pkg: str, version: Optional[str] = None, id: Optional[str] = None, commit: bool = False) -> Dict[str, Any]:
        """The PDK mount at ``pdk/<pkg>@<version>``: reused when present; without ``commit``, the entry to ``place_mount``."""
        if self.api is None:
            raise CasError("Cas:SourceOffline", "resolving a PDK by name needs the platform")
        pdk = {"pkg": pkg, **({"version": version} if version else {}), **({"id": id} if id else {})}
        out = await self.api.ensure_pdk_mount(self.project_id, pdk, commit)
        if commit:
            await self.reload()
        return out

    async def place_mount(self, path: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        return await self.project.place_mount(path, entry)


__all__ = ["CasFs", "CasHandle", "DirChunkCache"]
