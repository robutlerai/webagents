"""CASFS in Python: the same rules as the TypeScript SDK and the portal.

Canonical manifests (byte for byte with the other cores), mount entries,
reading through a mount from the source's backend, read-only refusals,
cycles and depth, merge by path, the local directory store with a mount
resolved from a local package store, the machine cache, the platform API's
requests and named refusals, and the CLI's parsing.
"""

import hashlib
import json

import httpx
import pytest

from webagents.casfs import (
    MAX_MOUNT_DEPTH,
    CasError,
    CasFs,
    CasProject,
    DirChunkCache,
    LocalDirBackend,
    LocalHeads,
    MemoryBackend,
    MemoryChunkCache,
    PlatformApi,
    canon,
    check_dir_manifest,
    sha256_hex,
)
from webagents.cli.cas_command import parse_source, parse_target, run_cas

A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"
K = "66666666-6666-4666-8666-666666666666"


class MemHeads:
    def __init__(self):
        self.head = None

    async def get(self):
        return self.head

    async def swap(self, base, nxt):
        if self.head != base:
            return {"ok": False, "current": self.head}
        self.head = nxt
        return {"ok": True}


class Opener:
    def __init__(self, backends):
        self.backends = backends
        self.log = []

    async def open(self, path, source, root):
        self.log.append(source["id"])
        b = self.backends.get(source["id"])
        if b is None:
            raise CasError("Cas:MountSource", "not readable")
        return b, None


def mount_of(pid, root):
    return {"type": "mount", "readOnly": True, "root": root, "source": {"kind": "project", "id": pid}}


async def source_project(files, backend=None):
    backend = backend or MemoryBackend("src")
    p = await CasProject.open(backend, None, heads=MemHeads())
    for path, text in files.items():
        await p.write(path, text)
    return backend, await p.commit()


async def code(coro):
    try:
        await coro
        return "ok"
    except CasError as e:
        return e.code


def test_canonical_json_matches_the_other_cores():
    assert canon({"b": 1, "a": [True, None, "x"], "é": "ü"}) == '{"a":[true,null,"x"],"b":1,"é":"ü"}'
    with pytest.raises(CasError):
        canon({"a": 1.5})
    empty = canon({"kind": "casfs.dir/1", "entries": {}}).encode()
    # The empty directory's hash is the same in every core.
    assert sha256_hex(empty) == hashlib.sha256(b'{"entries":{},"kind":"casfs.dir/1"}').hexdigest()


def test_mount_entries_are_checked_like_the_portal():
    sha = "a" * 64
    ok = {"type": "mount", "readOnly": True, "root": sha, "source": {"kind": "pdk", "id": K, "pkg": "sky130A", "version": "2026.10"}}
    check_dir_manifest({"kind": "casfs.dir/1", "entries": {"m": ok}})
    for bad in (
        {**ok, "readOnly": False},
        {**ok, "extra": 1},
        {**ok, "source": {"kind": "pdk", "id": K, "pkg": "sky130A"}},
        {**ok, "source": {"kind": "url", "id": K}},
    ):
        with pytest.raises(CasError) as e:
            check_dir_manifest({"kind": "casfs.dir/1", "entries": {"m": bad}})
        assert e.value.code == "Cas:ManifestInvalid"


async def test_chunked_write_and_range_read_move_only_their_chunks():
    backend = MemoryBackend()
    p = await CasProject.open(backend, None, heads=MemHeads(), chunk_size=256 * 1024)
    big = bytes(i & 255 for i in range(3 * 256 * 1024 + 7))
    await p.write("data/big.bin", big)
    root = await p.commit()
    q = await CasProject.open(backend, root)
    backend.bytes_read = 0
    assert await q.read("data/big.bin", 256 * 1024 + 5, 10) == big[256 * 1024 + 5 : 256 * 1024 + 15]
    assert backend.bytes_read <= 256 * 1024 + 4096
    backend.blobs[sha256_hex(b"truth")] = b"lies!"
    await p.write("t.txt", "truth")
    backend.blobs[sha256_hex(b"truth")] = b"lies!"
    r2 = await p.commit()
    assert await code((await CasProject.open(backend, r2)).read("t.txt")) == "Cas:HashMismatch"


async def test_reads_through_a_mount_and_names_it_in_the_run_record():
    src, sroot = await source_project({"lib/x.lib": "liberty"})
    mine = MemoryBackend("mine")
    p = await CasProject.open(mine, None, heads=MemHeads(), mounts=Opener({B: src}))
    await p.place_mount("pdk/kit", mount_of(B, sroot))
    root = await p.commit()
    assert sha256_hex(b"liberty") not in mine.blobs
    q = await CasProject.open(mine, root, mounts=Opener({B: src}))
    assert [e["type"] for e in await q.list("pdk")] == ["mount"]
    assert await q.read("pdk/kit/lib/x.lib") == b"liberty"
    rec = q.run_record("r", {}, [], [], 0)
    assert [i["path"] for i in rec["inputs"]] == ["pdk/kit/lib/x.lib"]
    assert rec["mounts"] == [{"at": "pdk/kit", "root": sroot, "source": {"kind": "project", "id": B}}]


async def test_mounts_are_read_only_and_cycles_and_depth_are_refused():
    src, sroot = await source_project({"x": "x"})
    p = await CasProject.open(MemoryBackend(), None, heads=MemHeads(), mounts=Opener({B: src}), _source_id=A)
    await p.place_mount("m", mount_of(B, sroot))
    assert await code(p.write("m/new", "no")) == "Cas:MountReadOnly"
    assert await code(p.write("m", "no")) == "Cas:MountReadOnly"
    assert await code(p.mkdir("m/d")) == "Cas:MountReadOnly"
    back = await CasProject.open(src, None, heads=MemHeads())
    await back.place_mount("back", mount_of(A, sroot))
    back_root = await back.commit()
    c = await CasProject.open(MemoryBackend(), None, heads=MemHeads(), mounts=Opener({B: src}), _source_id=A)
    await c.place_mount("b", mount_of(B, back_root))
    assert await code(c.list("b/back")) == "Cas:MountCycle"

    shared = MemoryBackend("shared")
    _, prev = await source_project({"leaf": "x"}, shared)
    for i in range(MAX_MOUNT_DEPTH + 1):
        s = await CasProject.open(shared, None, heads=MemHeads())
        await s.place_mount("n", mount_of(f"7777777{i}-7777-4777-8777-777777777777", prev))
        prev = await s.commit()

    class All:
        async def open(self, path, source, root):
            return shared, None

    d = await CasProject.open(shared, prev, mounts=All())
    assert await code(d.list("n/n/n/n")) == "ok"
    assert await code(d.list("n/n/n/n/n")) == "Cas:MountDepth"


async def test_merge_by_path_and_conflict():
    backend = MemoryBackend()
    heads = MemHeads()
    p0 = await CasProject.open(backend, None, heads=heads)
    await p0.write("a", "1")
    base = await p0.commit()
    x = await CasProject.open(backend, base, heads=heads)
    y = await CasProject.open(backend, base, heads=heads)
    await x.write("b", "x")
    await y.write("c", "y")
    await x.commit()
    merged = await y.commit()
    assert [e["name"] for e in await (await CasProject.open(backend, merged)).list()] == ["a", "b", "c"]
    u = await CasProject.open(backend, merged, heads=heads)
    v = await CasProject.open(backend, merged, heads=heads)
    await u.write("a", "u")
    await v.write("a", "v")
    await u.commit()
    assert await code(v.commit()) == "Cas:PathConflict"


async def test_local_store_keeps_a_pointer_and_resolves_it_from_a_package_store(tmp_path):
    proj_dir, pkgs, cache_dir = tmp_path / "p", tmp_path / "pkgs", tmp_path / "cache"
    kit = LocalDirBackend(pkgs / K)
    _, kit_root = await source_project({"libs/a.lib": "kit bytes"}, kit)
    fs = CasFs(None, DirChunkCache(cache_dir), str(pkgs))
    h = await fs.open_local(A, LocalDirBackend(proj_dir), LocalHeads(proj_dir, A))
    await h.write("cells/c.json", "{}")
    await h.commit()
    await h.mount("pdk/sky130A@2026.10", {"kind": "pdk", "id": K, "pkg": "sky130A", "version": "2026.10"}, kit_root)
    assert not await LocalDirBackend(proj_dir).has(sha256_hex(b"kit bytes"))
    assert await h.read("pdk/sky130A@2026.10/libs/a.lib") == b"kit bytes"
    assert await DirChunkCache(cache_dir).get(sha256_hex(b"kit bytes")) == b"kit bytes"
    assert [m["path"] for m in await h.mounts()] == ["pdk/sky130A@2026.10"]
    await h.unmount("pdk/sky130A@2026.10")
    assert await h.mounts() == []


async def test_platform_api_requests_and_named_refusals():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST" and request.url.path.endswith("/mounts"):
            return httpx.Response(400, json={"ok": False, "error": {"code": "Cas:MountSource", "message": "Cas:MountSource: the package sky130A is not a source you can read"}})
        return httpx.Response(200, json={"ok": True, "head": None, "counter": 0})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fs = CasFs.platform("https://portal.example/", "tok", grants={A: "grant-a"}, client=client, cache=MemoryChunkCache())
    h = await fs.open(A)
    with pytest.raises(CasError) as e:
        await h.ensure_pdk_mount("sky130A", commit=True)
    assert e.value.code == "Cas:MountSource"
    assert str(e.value) == "Cas:MountSource: the package sky130A is not a source you can read"
    post = [r for r in seen if r.method == "POST"][0]
    assert str(post.url) == f"https://portal.example/api/casfs/projects/{A}/mounts"
    assert post.headers["authorization"] == "Bearer tok"
    assert post.headers["x-casfs-grant"] == "grant-a"
    assert json.loads(post.content) == {"pdk": {"pkg": "sky130A"}, "commit": True}


async def test_cli_parsing():
    assert parse_target(f"{A}:pdk/kit") == {"project": A, "path": "pdk/kit"}
    assert parse_source("pdk:sky130A@2026.10") == {"kind": "pdk", "pkg": "sky130A", "version": "2026.10"}
    assert parse_source(f"project:{B}") == {"kind": "project", "id": B}
    with pytest.raises(CasError):
        parse_source("http://x")
    with pytest.raises(CasError):
        parse_target("nope")
    assert await code(run_cas(CasFs(None), "frobnicate", [], None, None, print, print)) == "Cas:PathInvalid"
