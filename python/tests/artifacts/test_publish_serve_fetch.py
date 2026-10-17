"""Publish a folder, serve it from the producer endpoint, fetch it back with
the client: ranges, digests, authorisation by caller, resume after a cut, a
wrong chunk refused and not kept."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from starlette.responses import StreamingResponse

from webagents.artifacts import (
    ARTIFACT_TASK_HEADER,
    ArtifactFetchError,
    ArtifactLocation,
    ArtifactProducer,
    ArtifactRegistry,
    ArtifactSet,
    ArtifactsSkill,
    FetchOptions,
    fetch_artifact_set,
    fetch_manifest,
    files,
    hash_bytes_iter,
    hash_file,
    read_artifact_set,
    repr_digest,
)

CHUNK = 256 * 1024
BASE = "https://producer.example/agents/flow"
CALLER = "agent:https://consumer.example/agents/reda"


def context_for(principals):
    auth = SimpleNamespace(authenticated=True, principals=principals) if principals is not None else SimpleNamespace(authenticated=False, principals=[])
    return SimpleNamespace(auth=auth, request=None)


async def starlette_to_httpx(response) -> httpx.Response:
    if isinstance(response, StreamingResponse):
        chunks = []
        async for piece in response.body_iterator:
            chunks.append(piece if isinstance(piece, bytes) else piece.encode())
        body = b"".join(chunks)
    else:
        body = response.body
    return httpx.Response(response.status_code, headers=dict(response.headers), content=body)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    folder = root / "out"
    (folder / "gds").mkdir(parents=True)
    (folder / "reports").mkdir()
    data = {
        "gds/top.gds": os.urandom(3 * CHUNK + 1234),
        "reports/drc.txt": b"clean\n",
        "reports/empty.log": b"",
        "summary.json": json.dumps({"ok": True}).encode(),
    }
    for rel, b in data.items():
        (folder / rel).write_bytes(b)
    os.symlink("/etc/hostname", folder / "escape")
    registry = ArtifactRegistry()
    skill = ArtifactsSkill(registry=registry, max_concurrent_transfers=8)
    published = files.publish(
        folder,
        producer=ArtifactProducer(agent=BASE, task="task-42"),
        registry=registry,
        base_url=BASE,
        chunk_size=CHUNK,
        audience=["https://consumer.example/agents/reda"],
        created_at=datetime(2026, 10, 17, 10, 0, 0, tzinfo=timezone.utc),
    )
    return SimpleNamespace(root=root, folder=folder, data=data, registry=registry, skill=skill, published=published)


def client_as(world, principals, *, corrupt=None, fail_once=None):
    state = {"failed": False}

    async def handler(request: httpx.Request) -> httpx.Response:
        if fail_once and not state["failed"] and fail_once(str(request.url), request.headers.get("range")):
            state["failed"] = True
            raise httpx.ReadError("connection reset", request=request)
        digest = str(request.url.path).rsplit("/", 1)[-1]
        response = await world.skill.serve(digest, head_only=request.method == "HEAD", context=context_for(principals), headers=dict(request.headers))
        out = await starlette_to_httpx(response)
        if corrupt is not None:
            body = corrupt(str(request.url), out.content)
            headers = dict(out.headers)
            headers.pop("content-length", None)
            out = httpx.Response(out.status_code, headers=headers, content=body)
        return out

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_publish_writes_the_manifest_and_registers_the_blobs(world):
    p = world.published
    assert p.skipped == [{"path": "escape", "reason": "symlink"}]
    assert [e.path for e in p.artifact_set.entries] == ["gds/top.gds", "reports/drc.txt", "reports/empty.log", "summary.json"]
    gds = p.artifact_set.entries[0]
    big = world.data["gds/top.gds"]
    assert gds.size == len(big)
    assert gds.sha256 == hashlib.sha256(big).hexdigest()
    assert gds.chunks.chunk_size == CHUNK
    assert len(gds.chunks.hashes) == 4
    assert gds.chunks.hashes[1] == hashlib.sha256(big[CHUNK:2 * CHUNK]).hexdigest()
    assert gds.locations[0].to_json() == {"kind": "https", "url": f"{BASE}/artifacts/blobs/sha256:{gds.sha256}"}
    assert p.artifact_set.entries[1].chunks is None
    assert p.manifest_url == f"{BASE}/artifacts/blobs/sha256:{p.sha256}"
    assert world.registry.get_blob(gds.sha256).path == str(world.folder / "gds/top.gds")
    assert world.registry.access(gds.sha256, [CALLER], "task-42").ok
    assert world.registry.access(gds.sha256, ["agent:https://other.example/agents/x"], "task-42").reason == "forbidden"
    assert world.registry.access(gds.sha256, [CALLER], None).reason == "task_required"
    assert world.registry.access(gds.sha256, [CALLER], "task-1").reason == "task_mismatch"
    assert world.registry.access("00" * 32, [CALLER], "task-42").reason == "unknown"
    assert hash_bytes_iter([big], CHUNK) == hash_file(world.folder / "gds/top.gds", CHUNK)


async def test_endpoint_serves_manifest_and_ranges(world):
    p = world.published
    ctx = context_for([CALLER])
    headers = {ARTIFACT_TASK_HEADER: "task-42"}
    m = await world.skill.serve(f"sha256:{p.sha256}", head_only=False, context=ctx, headers=headers)
    assert m.status_code == 200
    assert m.headers["repr-digest"] == repr_digest(p.sha256)
    assert read_artifact_set(m.body)[1] == p.sha256

    gds = p.artifact_set.entries[0]
    big = world.data["gds/top.gds"]
    whole = await starlette_to_httpx(await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=ctx, headers=headers))
    assert whole.status_code == 200
    assert whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["content-length"] == str(gds.size)
    assert whole.content == big

    part = await starlette_to_httpx(await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=ctx, headers={**headers, "range": f"bytes={CHUNK}-{2 * CHUNK - 1}"}))
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes {CHUNK}-{2 * CHUNK - 1}/{gds.size}"
    assert part.headers["repr-digest"] == repr_digest(gds.sha256)
    assert part.content == big[CHUNK:2 * CHUNK]

    past = await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=ctx, headers={**headers, "range": f"bytes={gds.size}-"})
    assert past.status_code == 416
    assert past.headers["content-range"] == f"bytes */{gds.size}"

    head = await world.skill.serve(f"sha256:{gds.sha256}", head_only=True, context=ctx, headers={**headers, "range": "bytes=0-9"})
    assert head.status_code == 206
    assert head.headers["content-length"] == "10"
    assert head.body == b""

    empty = await world.skill.serve(f"sha256:{p.artifact_set.entries[2].sha256}", head_only=False, context=ctx, headers=headers)
    assert empty.status_code == 200
    assert empty.headers["content-length"] == "0"


async def test_endpoint_refusals_are_named(world):
    p = world.published
    gds = p.artifact_set.entries[0]
    task = {ARTIFACT_TASK_HEADER: "task-42"}
    assert (await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for(None), headers=task)).status_code == 401
    assert (await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for(["agent:https://other.example/agents/x"]), headers=task)).status_code == 404
    no_task = await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for([CALLER]), headers={})
    assert no_task.status_code == 403
    assert json.loads(no_task.body)["error"]["code"] == "task_required"
    wrong = await world.skill.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for([CALLER]), headers={ARTIFACT_TASK_HEADER: "other"})
    assert json.loads(wrong.body)["error"]["code"] == "task_mismatch"
    assert (await world.skill.serve("sha256:" + "00" * 32, head_only=False, context=context_for([CALLER]), headers=task)).status_code == 404
    assert (await world.skill.serve("md5:abc", head_only=False, context=context_for([CALLER]), headers=task)).status_code == 404

    tight = ArtifactsSkill(registry=world.registry, max_concurrent_transfers=1)
    first = await tight.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for([CALLER]), headers=task)
    assert first.status_code == 200
    second = await tight.serve(f"sha256:{gds.sha256}", head_only=False, context=context_for([CALLER]), headers=task)
    assert second.status_code == 503
    assert json.loads(second.body)["error"]["code"] == "too_many_transfers"
    await starlette_to_httpx(first)  # drains and releases
    assert tight.active_transfers == 0


async def test_client_fetches_everything_verified(world, tmp_path):
    p = world.published
    async with client_as(world, [CALLER]) as client:
        options = FetchOptions(dest=str(tmp_path / "fetched"), client=client, task="task-42", concurrency=3)
        artifact_set, sha, _ = await fetch_manifest(p.manifest_url, options, expected_sha256=p.sha256)
        assert sha == p.sha256
        result = await fetch_artifact_set(artifact_set, options)
    assert [f.path for f in result.files] == [e.path for e in p.artifact_set.entries]
    for rel, data in world.data.items():
        assert (tmp_path / "fetched" / rel).read_bytes() == data
    assert result.bytes_fetched == sum(len(b) for b in world.data.values())
    assert not (tmp_path / "fetched" / "gds/top.gds.part").exists()


async def test_client_resumes_after_a_cut(world, tmp_path):
    p = world.published
    gds = p.artifact_set.entries[0]
    only = ArtifactSet(created_at=p.artifact_set.created_at, expires_at=None, producer=p.artifact_set.producer, entries=[gds])
    dest = tmp_path / "fetched"
    async with client_as(world, [CALLER], fail_once=lambda url, rng: gds.sha256 in url and (rng or "").startswith(f"bytes={2 * CHUNK}-")) as client:
        with pytest.raises(ArtifactFetchError) as caught:
            await fetch_artifact_set(only, FetchOptions(dest=str(dest), client=client, task="task-42", concurrency=1))
    assert caught.value.code == "no_reachable_location"
    state = json.loads((dest / "gds/top.gds.part.json").read_text())
    assert state["done"] == [0, 1]
    async with client_as(world, [CALLER]) as client:
        result = await fetch_artifact_set(only, FetchOptions(dest=str(dest), client=client, task="task-42", concurrency=2))
    assert result.files[0].bytes_fetched == gds.size - 2 * CHUNK
    assert (dest / "gds/top.gds").read_bytes() == world.data["gds/top.gds"]
    assert not (dest / "gds/top.gds.part").exists()


async def test_client_refuses_a_wrong_chunk_by_name(world, tmp_path):
    p = world.published
    gds = p.artifact_set.entries[0]
    only = ArtifactSet(created_at=p.artifact_set.created_at, expires_at=None, producer=p.artifact_set.producer, entries=[gds])

    def corrupt(url, body):
        if gds.sha256 not in url:
            return body
        b = bytearray(body)
        b[5] ^= 0xFF
        return bytes(b)

    async with client_as(world, [CALLER], corrupt=corrupt) as client:
        with pytest.raises(ArtifactFetchError) as caught:
            await fetch_artifact_set(only, FetchOptions(dest=str(tmp_path / "f"), client=client, task="task-42", concurrency=1))
    assert caught.value.code == "no_reachable_location"
    assert "chunk_hash_mismatch" in json.dumps(caught.value.detail)
    assert not (tmp_path / "f" / "gds/top.gds").exists()


async def test_client_refuses_unsupported_location_kinds_and_path_escapes(world, tmp_path):
    p = world.published
    drc = p.artifact_set.entries[1]
    oci = ArtifactSet(created_at=p.artifact_set.created_at, expires_at=None, producer=p.artifact_set.producer,
                      entries=[type(drc)(path=drc.path, size=drc.size, sha256=drc.sha256, media_type=drc.media_type,
                                        locations=[ArtifactLocation(kind="oci", ref="ghcr.io/x/y"), ArtifactLocation(kind="relay", agent="flow")])])
    async with client_as(world, [CALLER]) as client:
        with pytest.raises(ArtifactFetchError) as caught:
            await fetch_artifact_set(oci, FetchOptions(dest=str(tmp_path / "f"), client=client, task="task-42"))
        assert "oci: location_kind_unsupported" in str(caught.value)
        assert "relay: location_kind_unsupported" in str(caught.value)
        escaping = ArtifactSet(created_at=p.artifact_set.created_at, expires_at=None, producer=p.artifact_set.producer,
                               entries=[type(drc)(path="ok/../../escape.txt", size=drc.size, sha256=drc.sha256, media_type=drc.media_type, locations=drc.locations)])
        with pytest.raises(ArtifactFetchError) as escaped:
            await fetch_artifact_set(escaping, FetchOptions(dest=str(tmp_path / "f"), client=client, task="task-42"))
        assert escaped.value.code == "path_escape"
