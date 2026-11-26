"""A CASFS project: read by path and verified by hash, written by path and committed by compare-and-swap.

Reads verify a manifest against its name and canonical form, a chunk against
its hash (``Cas:ChunkMismatch``) and a small file against its own hash
(``Cas:HashMismatch``). Verified bytes go to the chunk cache.

Mounts: a ``{"type": "mount", "readOnly": true, "root", "source"}`` entry shows
another project, or a published PDK, as a read-only folder pinned to one root
of the source. Reads resolve through it to a sub-project over the source's own
backend (``mounts.open``), under the caller's rights on the source; anything
that would change the inside of a mount is ``Cas:MountReadOnly``. Chains deeper
than ``MAX_MOUNT_DEPTH`` are ``Cas:MountDepth``; a chain that returns to a
source already in it is ``Cas:MountCycle``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

from .core import (
    CHUNK_DEFAULT,
    DIR_KIND,
    INLINE_CHUNK_HASHES,
    MAX_MOUNT_DEPTH,
    CasError,
    canon,
    canon_bytes,
    check_chunk_size,
    check_mount_source,
    check_policy,
    chunk_count,
    is_sha,
    parse_dir_manifest,
    same_mount,
    sha256_hex,
    split_path,
)


class Heads(Protocol):
    async def swap(self, base: Optional[str], nxt: str) -> Dict[str, Any]: ...


class MountOpener(Protocol):
    async def open(self, path: str, source: Dict[str, str], root: str) -> Tuple[Any, Optional["MountOpener"]]: ...


class _Node:
    __slots__ = ("entries", "dirty")

    def __init__(self, entries: Optional[Dict[str, Dict[str, Any]]] = None, dirty: bool = True):
        self.entries: Dict[str, Dict[str, Any]] = entries if entries is not None else {}
        self.dirty = dirty


def _read_only(path: str) -> CasError:
    return CasError("Cas:MountReadOnly", f"{path} is a mount: nothing inside it can change", {"path": path})


def _plain(e: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in e.items() if k != "_node"}


class CasProject:
    def __init__(
        self,
        backend: Any,
        root: Optional[str] = None,
        heads: Optional[Heads] = None,
        mounts: Optional[MountOpener] = None,
        cache: Any = None,
        chunk_size: int = CHUNK_DEFAULT,
        root_policy: str = "cas",
        _depth: int = 0,
        _chain: Optional[List[str]] = None,
        _source_id: Optional[str] = None,
        _prefix: str = "",
        _opened: Optional[List[Dict[str, Any]]] = None,
    ):
        self.backend = backend
        self.base_root = root
        self.heads = heads
        self.mounts = mounts
        self.cache = cache
        self.chunk_size = check_chunk_size(chunk_size)
        self.root_policy = root_policy
        self._depth = _depth
        self._chain = _chain or []
        self._source_id = _source_id
        self._prefix = _prefix
        self.opened: List[Dict[str, Any]] = _opened if _opened is not None else []
        self._root = _Node()
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._subs: Dict[str, Tuple["CasProject", Dict[str, Any]]] = {}

    @classmethod
    async def open(cls, backend: Any, root: Optional[str] = None, **kw: Any) -> "CasProject":
        p = cls(backend, root, **kw)
        if root:
            p._root = await p._node_for(root)
        return p

    # ── blobs, verified and cached ──

    async def _fetch_verified(self, sha: str, start: int, end: int, want: str) -> bytes:
        if self.cache is not None:
            c = await self.cache.get(want)
            if c is not None and sha256_hex(c) == want:
                return c
        b = await self.backend.read_range(sha, start, end)
        if sha256_hex(b) != want:
            code = "Cas:HashMismatch" if (start == 0 and want == sha) else "Cas:ChunkMismatch"
            raise CasError(code, f"bytes of {sha[:12]} [{start}, {end}) do not hash to {want}", {"sha": sha, "expected": want})
        if self.cache is not None:
            await self.cache.put(want, b)
        return b

    async def _manifest(self, sha: str) -> Dict[str, Any]:
        if sha in self._manifests:
            return self._manifests[sha]
        cached = await self.cache.get(sha) if self.cache is not None else None
        data = cached if cached is not None else await self.backend.read_range(sha, 0, 32 * 1024 * 1024)
        m = parse_dir_manifest(sha, data)
        if cached is None and self.cache is not None:
            await self.cache.put(sha, data)
        self._manifests[sha] = m
        return m

    async def _node_for(self, sha: str) -> _Node:
        m = await self._manifest(sha)
        return _Node({k: dict(v) for k, v in m["entries"].items()}, dirty=False)

    # ── mounts ──

    def _full(self, path: str) -> str:
        if not self._prefix:
            return path
        return f"{self._prefix}/{path}" if path else self._prefix

    async def _mount_project(self, path: str, entry: Dict[str, Any]) -> "CasProject":
        have = self._subs.get(path)
        if have and same_mount(have[1], entry):
            return have[0]
        source = check_mount_source(entry.get("source"))
        full = self._full(path)
        depth = self._depth + 1
        if depth > MAX_MOUNT_DEPTH:
            raise CasError("Cas:MountDepth", f"mounts nest more than {MAX_MOUNT_DEPTH} deep at {full}", {"path": full, "limit": MAX_MOUNT_DEPTH})
        chain = self._chain + ([self._source_id] if self._source_id else [])
        if source["id"] in chain:
            raise CasError("Cas:MountCycle", f"{full} leads back to a project already in the chain", {"path": full})
        if self.mounts is None:
            raise CasError("Cas:SourceOffline", f"{full} is a mount and this reader has no way to open mounts", {"path": full})
        backend, nested = await self.mounts.open(full, source, entry["root"])
        sub = await CasProject.open(
            backend, entry["root"], mounts=nested or self.mounts, cache=self.cache, chunk_size=self.chunk_size,
            _depth=depth, _chain=chain, _source_id=source["id"], _prefix=full, _opened=self.opened,
        )
        self._subs[path] = (sub, entry)
        return sub

    async def _resolve(self, segs: List[str], into: bool) -> Tuple["CasProject", List[str], Optional[Dict[str, Any]]]:
        node = self._root
        for i, s in enumerate(segs):
            e = node.entries.get(s)
            if e is None:
                break
            if e["type"] == "mount":
                if i == len(segs) - 1 and not into:
                    return self, segs, e
                sub = await self._mount_project("/".join(segs[: i + 1]), e)
                return await sub._resolve(segs[i + 1 :], into)
            if e["type"] != "dir" or i == len(segs) - 1:
                break
            if "_node" not in e:
                e["_node"] = await self._node_for(e["sha256"])
            node = e["_node"]
        return self, segs, None

    def mounts_opened(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path, (sub, entry) in self._subs.items():
            out.append({"at": self._full(path), "root": entry["root"], "source": dict(entry["source"])})
            out.extend(sub.mounts_opened())
        return out

    # ── reading ──

    async def _dir_chain(self, segs: List[str], create: bool = False) -> List[_Node]:
        chain = [self._root]
        for i, s in enumerate(segs):
            node = chain[-1]
            e = node.entries.get(s)
            sub = "/".join(segs[: i + 1])
            if e is not None and e["type"] == "mount":
                raise _read_only(sub)
            if e is None:
                if not create:
                    raise CasError("Cas:NotFound", f"no directory {sub}", {"path": sub})
                e = {"type": "dir", "sha256": "", "_node": _Node()}
                node.entries[s] = e
                node.dirty = True
            if e["type"] != "dir":
                raise CasError("Cas:NotADirectory", f"{sub} is not a folder", {"path": sub})
            if "_node" not in e:
                e["_node"] = await self._node_for(e["sha256"])
            chain.append(e["_node"])
        return chain

    def _policy_at(self, segs: List[str], chain: List[_Node], e: Optional[Dict[str, Any]]) -> str:
        p = self.root_policy
        for i in range(len(segs) - 1):
            d = chain[i].entries.get(segs[i])
            if d and d["type"] == "dir" and d.get("policy"):
                p = d["policy"]
        if e and e["type"] in ("file", "dir") and e.get("policy"):
            p = e["policy"]
        return p

    @staticmethod
    def _view(name: str, e: Dict[str, Any], policy: str) -> Dict[str, Any]:
        t = e["type"]
        if t == "mount":
            return {"name": name, "type": "mount", "root": e["root"], "source": dict(e["source"]), "policy": "cas"}
        if t == "sealed":
            return {"name": name, "type": "sealed"}
        if t == "dir":
            return {"name": name, "type": "dir", "sha256": e.get("sha256") or None, "policy": policy}
        return {"name": name, "type": "file", "size": e["size"], "sha256": e.get("sha256"), "id": e.get("id"), "policy": policy, "mediaType": e.get("mediaType")}

    async def stat(self, path: str) -> Dict[str, Any]:
        segs = split_path(path)
        proj, rest, mount = await self._resolve(segs, False)
        if proj is not self:
            return await proj.stat("/".join(rest))
        if mount is not None:
            return {"name": segs[-1], "path": "/".join(segs), "type": "mount", "root": mount["root"], "source": dict(mount["source"]), "policy": "cas"}
        if not segs:
            return {"name": "", "path": "", "type": "dir", "policy": self.root_policy}
        chain = await self._dir_chain(segs[:-1])
        e = chain[-1].entries.get(segs[-1])
        if e is None:
            raise CasError("Cas:NotFound", f"no such file or directory: {path}", {"path": path})
        return {**self._view(segs[-1], e, self._policy_at(segs, chain, e)), "path": "/".join(segs)}

    async def list(self, path: str = "") -> List[Dict[str, Any]]:
        proj, segs, _ = await self._resolve(split_path(path), True)
        if proj is not self:
            return await proj.list("/".join(segs))
        chain = await self._dir_chain(segs)
        return [self._view(n, e, self._policy_at(segs + [n], chain, e)) for n, e in sorted(chain[-1].entries.items())]

    async def _hashes(self, path: str, e: Dict[str, Any]) -> List[str]:
        c = e["chunks"]
        if "hashes" in c:
            return c["hashes"]
        k = chunk_count(e["size"], c["size"])
        raw = await self._fetch_verified(c["index"], 0, k * 32, c["index"])
        if len(raw) != k * 32:
            raise CasError("Cas:HashMismatch", f"chunk index of {path} has the wrong length", {"path": path})
        return [raw[i * 32 : i * 32 + 32].hex() for i in range(k)]

    async def read(self, path: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        segs = split_path(path)
        proj, rest, mount = await self._resolve(segs, False)
        if proj is not self:
            return await proj.read("/".join(rest), offset, length)
        if mount is not None:
            raise CasError("Cas:IsADirectory", f"{path} is a mount (a folder)", {"path": path})
        st = await self.stat(path)
        if st["type"] != "file":
            raise CasError("Cas:IsADirectory", f"{path} is not a file", {"path": path})
        chain = await self._dir_chain(segs[:-1])
        e = chain[-1].entries[segs[-1]]
        if e.get("growing") or not e.get("sha256"):
            raise CasError("Cas:NotFound", f"{path} is still being written", {"path": path})
        size = e["size"]
        end = min(size, size if length is None else offset + length)
        if offset < 0 or offset > size or end < offset:
            raise CasError("Cas:RangeInvalid", f"range outside {path}", {"path": path})
        full = self._full("/".join(segs))
        if not any(o["path"] == full for o in self.opened):
            self.opened.append({"path": full, "sha256": e["sha256"], "size": size, "policy": st.get("policy", "cas")})
        if end == offset:
            return b""
        if "chunks" not in e:
            return (await self._fetch_verified(e["sha256"], 0, size, e["sha256"]))[offset:end]
        hs = await self._hashes(path, e)
        cs = e["chunks"]["size"]
        parts = []
        i = offset // cs
        while i * cs < end:
            s = i * cs
            b = await self._fetch_verified(e["sha256"], s, min(size, s + cs), hs[i])
            parts.append(b[max(0, offset - s) : min(len(b), end - s)])
            i += 1
        return b"".join(parts)

    async def read_stream(self, path: str, offset: int = 0, length: Optional[int] = None, step: int = 4 * 1024 * 1024) -> AsyncIterator[bytes]:
        st = await self.stat(path)
        size = st.get("size") or 0
        end = min(size, size if length is None else offset + length)
        o = offset
        while o < end:
            yield await self.read(path, o, min(step, end - o))
            o += step

    # ── writing (staged until commit) ──

    async def write(self, path: str, data: bytes | str, media_type: Optional[str] = None, policy: Optional[str] = None) -> Dict[str, Any]:
        check_policy(policy)
        b = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        segs = split_path(path)
        if not segs:
            raise CasError("Cas:PathInvalid", "cannot write the root")
        chain = await self._dir_chain(segs[:-1], create=True)
        d = chain[-1]
        old = d.entries.get(segs[-1])
        if old is not None and old["type"] == "mount":
            raise _read_only("/".join(segs))
        if old is not None and old["type"] != "file":
            raise CasError("Cas:IsADirectory", f"{path} is a folder", {"path": path})
        sha = sha256_hex(b)
        e: Dict[str, Any] = {"type": "file", "size": len(b), "sha256": sha}
        if media_type:
            e["mediaType"] = media_type
        if old and old.get("id"):
            e["id"] = old["id"]
        pol = policy or (old or {}).get("policy")
        if pol:
            e["policy"] = pol
        if len(b) > self.chunk_size:
            hashes = [sha256_hex(b[o : o + self.chunk_size]) for o in range(0, len(b), self.chunk_size)]
            if len(hashes) <= INLINE_CHUNK_HASHES:
                e["chunks"] = {"size": self.chunk_size, "hashes": hashes}
            else:
                raw = b"".join(bytes.fromhex(h) for h in hashes)
                idx = sha256_hex(raw)
                await self.backend.put(raw, idx)
                e["chunks"] = {"size": self.chunk_size, "index": idx}
        if pol not in ("local-only", "local-served"):
            await self.backend.put(b, sha)
        d.entries[segs[-1]] = e
        for n in chain:
            n.dirty = True
        return e

    async def mkdir(self, path: str) -> None:
        for n in await self._dir_chain(split_path(path), create=True):
            n.dirty = True

    async def delete(self, path: str) -> None:
        segs = split_path(path)
        chain = await self._dir_chain(segs[:-1])
        if chain[-1].entries.pop(segs[-1], None) is None:
            raise CasError("Cas:NotFound", f"no such file or directory: {path}", {"path": path})
        for n in chain:
            n.dirty = True
        self._subs.pop("/".join(segs), None)

    async def place_mount(self, path: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Place a mount entry (from ``ensure_pdk_mount`` or the mount API) so it commits with other changes."""
        if not isinstance(entry, dict) or entry.get("type") != "mount" or entry.get("readOnly") is not True or not is_sha(entry.get("root")):
            raise CasError("Cas:ManifestInvalid", "not a mount entry")
        e = {"type": "mount", "readOnly": True, "root": entry["root"], "source": check_mount_source(entry.get("source"))}
        segs = split_path(path)
        if not segs:
            raise CasError("Cas:PathInvalid", "a mount needs a path below the root")
        chain = await self._dir_chain(segs[:-1], create=True)
        old = chain[-1].entries.get(segs[-1])
        if old is not None:
            if old["type"] == "mount" and same_mount(old, e):
                return old
            raise CasError("Cas:Exists", f"{'/'.join(segs)} exists", {"path": "/".join(segs)})
        chain[-1].entries[segs[-1]] = e
        for n in chain:
            n.dirty = True
        return e

    async def unmount(self, path: str) -> None:
        segs = split_path(path)
        chain = await self._dir_chain(segs[:-1])
        e = chain[-1].entries.get(segs[-1])
        if e is None:
            raise CasError("Cas:NotFound", f"no such path: {path}", {"path": path})
        if e["type"] != "mount":
            raise CasError("Cas:NotAMount", f"{path} is not a mount", {"path": path})
        await self.delete(path)

    async def _write_node(self, n: _Node) -> str:
        entries: Dict[str, Any] = {}
        for name, e in n.entries.items():
            if e["type"] == "dir" and "_node" in e and (e["_node"].dirty or not e["sha256"]):
                e["sha256"] = await self._write_node(e["_node"])
            entries[name] = _plain(e)
        m = {"kind": DIR_KIND, "entries": entries}
        data = canon_bytes(m)
        sha = sha256_hex(data)
        await self.backend.put(data, sha)
        self._manifests[sha] = m
        n.dirty = False
        return sha

    async def write_tree(self) -> str:
        return await self._write_node(self._root)

    async def flatten(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        async def walk(n: _Node, pre: str) -> None:
            for name, e in n.entries.items():
                p = f"{pre}/{name}" if pre else name
                if e["type"] == "dir":
                    if "_node" not in e:
                        e["_node"] = await self._node_for(e["sha256"])
                    out[p] = {"type": "dir", **({"policy": e["policy"]} if e.get("policy") else {})}
                    await walk(e["_node"], p)
                else:
                    out[p] = _plain(e)

        await walk(self._root, "")
        return out

    @staticmethod
    def _from_flat(flat: Dict[str, Dict[str, Any]]) -> _Node:
        root = _Node()
        for p in sorted(flat):
            segs = p.split("/")
            node = root
            for s in segs[:-1]:
                d = node.entries.get(s)
                if d is None:
                    d = {"type": "dir", "sha256": "", "_node": _Node()}
                    node.entries[s] = d
                node = d["_node"]
            v = flat[p]
            if v["type"] == "dir":
                node.entries.setdefault(segs[-1], {"type": "dir", "sha256": "", **({"policy": v["policy"]} if v.get("policy") else {}), "_node": _Node()})
            else:
                node.entries[segs[-1]] = dict(v)
        return root

    async def commit(self) -> str:
        """Compare-and-swap commit; on a moved head, merge by path, and refuse a path both sides changed (``Cas:PathConflict``)."""
        if self.heads is None:
            raise CasError("Cas:Forbidden", "this project was opened without a way to move its head")
        mine = await self.write_tree()
        for _ in range(5):
            r = await self.heads.swap(self.base_root, mine)
            if r.get("ok"):
                self.base_root = mine
                return mine
            cur = r.get("current")
            b = await (await CasProject.open(self.backend, self.base_root, cache=self.cache)).flatten() if self.base_root else {}
            t = await (await CasProject.open(self.backend, cur, cache=self.cache)).flatten() if cur else {}
            o = await self.flatten()

            def same(x: Any, y: Any) -> bool:
                return (x is None and y is None) or (x is not None and y is not None and canon(x) == canon(y))

            merged: Dict[str, Dict[str, Any]] = {}
            for p in set(b) | set(o) | set(t):
                bv, ov, tv = b.get(p), o.get(p), t.get(p)
                if same(ov, bv):
                    v = tv
                elif same(tv, bv) or same(ov, tv):
                    v = ov
                else:
                    raise CasError("Cas:PathConflict", f"{p} changed on both sides", {"path": p, "head": cur})
                if v is not None:
                    merged[p] = v
            self._root = self._from_flat(merged)
            self.base_root = cur
            mine = await self.write_tree()
        raise CasError("Cas:HeadMoved", "the head kept moving; try again")

    def run_record(self, run_id: str, tool: Any, argv: List[str], outputs: List[Any], exit_code: int, mounts: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> Dict[str, Any]:
        given = mounts or []
        opened = [m for m in self.mounts_opened() if not any(x["at"] == m["at"] and x["root"] == m["root"] for x in given)]
        return {
            "kind": "casfs.run/1", "runId": run_id, "tool": tool, "argv": argv, "env": extra.get("env", []),
            "mounts": given + opened, "inputs": sorted(self.opened, key=lambda x: x["path"]), "outputs": outputs,
            "exitCode": exit_code, **{k: v for k, v in extra.items() if k in ("startedAt", "finishedAt", "host")},
        }
