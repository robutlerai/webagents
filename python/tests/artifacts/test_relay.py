"""The artifacts relay: the credit window is never exceeded, a cancel stops
the read where it is, concurrent reads interleave, refusals are named, and
the binary frame round-trips."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import pytest

from webagents.artifacts import (
    BLOB_FRAME_MAX_BYTES,
    ArtifactProducer,
    ArtifactRegistry,
    ArtifactRelay,
    decode_blob_data_frame,
    encode_blob_data_frame,
    files,
)


class Wire:
    """The platform's end of one connection: collects what the agent sends."""

    def __init__(self):
        self.json = []
        self.binary = []

    async def send(self, payload):
        if isinstance(payload, (bytes, bytearray)):
            self.binary.append(decode_blob_data_frame(bytes(payload)))
        else:
            self.json.append(json.loads(payload))

    def frames(self, request_id):
        return [b for b in self.binary if b["request_id"] == request_id]

    def ended(self, request_id):
        return any(f["type"] == "blob.end" and f["request_id"] == request_id for f in self.json)

    def error(self, request_id):
        return next((f["error"]["code"] for f in self.json if f["type"] == "blob.error" and f["request_id"] == request_id), None)


async def until(test, what, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not test():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timeout waiting for {what}")
        await asyncio.sleep(0.005)


def test_blob_data_frame_round_trips():
    payload = os.urandom(1000)
    frame = encode_blob_data_frame("req-1", 2**40 + 7, payload)
    back = decode_blob_data_frame(frame)
    assert back == {"request_id": "req-1", "offset": 2**40 + 7, "payload": payload}
    assert decode_blob_data_frame(bytes([2, 1, 0, 1, 65])) is None
    with pytest.raises(ValueError):
        encode_blob_data_frame("r", 0, bytes(BLOB_FRAME_MAX_BYTES + 1))
    with pytest.raises(ValueError):
        encode_blob_data_frame("x" * 65, 0, b"1")


async def test_relay_credit_interleave_cancel_and_refusals(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    big = os.urandom(5 * BLOB_FRAME_MAX_BYTES + 777)
    small = os.urandom(3 * 1024)
    (folder / "big.bin").write_bytes(big)
    (folder / "small.bin").write_bytes(small)
    registry = ArtifactRegistry()
    published = files.publish(folder, producer=ArtifactProducer(agent="https://agent.example/agents/flow"), registry=registry, chunk_size=262144)
    big_sha = next(e.sha256 for e in published.artifact_set.entries if e.path == "big.bin")
    small_sha = next(e.sha256 for e in published.artifact_set.entries if e.path == "small.bin")
    relay = ArtifactRelay(registry)
    wire = Wire()
    send = wire.send

    # 1. Credit is never exceeded.
    assert await relay.handle_frame(send, {"type": "blob.read", "request_id": "r1", "sha256": big_sha, "offset": 0, "length": len(big), "credit": 2})
    await until(lambda: len(wire.frames("r1")) >= 2, "two frames")
    await asyncio.sleep(0.05)
    assert len(wire.frames("r1")) == 2
    assert not wire.ended("r1")
    await relay.handle_frame(send, {"type": "blob.credit", "request_id": "r1", "frames": 2})
    await until(lambda: len(wire.frames("r1")) >= 4, "four frames")
    await asyncio.sleep(0.05)
    assert len(wire.frames("r1")) == 4
    await relay.handle_frame(send, {"type": "blob.credit", "request_id": "r1", "frames": 10})
    await until(lambda: wire.ended("r1"), "r1 end")
    frames = sorted(wire.frames("r1"), key=lambda b: b["offset"])
    assert len(frames) == 6
    assert b"".join(b["payload"] for b in frames) == big
    assert next(f for f in wire.json if f["type"] == "blob.end" and f["request_id"] == "r1")["bytes"] == len(big)

    # 2. Concurrent reads interleave: the small one is not starved by the big one.
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "r2", "sha256": big_sha, "offset": 0, "length": len(big), "credit": 1000})
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "r3", "sha256": small_sha, "offset": 0, "length": len(small), "credit": 1000})
    await until(lambda: wire.ended("r2") and wire.ended("r3"), "r2 and r3")
    order = [b["request_id"] for b in wire.binary if b["request_id"] in ("r2", "r3")]
    assert order.index("r3") < len(order) - 1 - order[::-1].index("r2")
    assert wire.frames("r3")[0]["payload"] == small

    # 3. Cancel mid-transfer: one frame of credit, then cancel; nothing more.
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "r4", "sha256": big_sha, "offset": 0, "length": len(big), "credit": 1})
    await until(lambda: len(wire.frames("r4")) >= 1, "r4 first frame")
    await relay.handle_frame(send, {"type": "blob.cancel", "request_id": "r4"})
    await relay.handle_frame(send, {"type": "blob.credit", "request_id": "r4", "frames": 100})
    await asyncio.sleep(0.1)
    assert len(wire.frames("r4")) == 1
    assert not wire.ended("r4") and wire.error("r4") is None
    assert relay.active_reads == 0

    # 4. Refusals by name.
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "r5", "sha256": "00" * 32, "offset": 0, "length": 10})
    assert wire.error("r5") == "unknown_blob"
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "r6", "sha256": big_sha, "offset": len(big) - 5, "length": 10})
    assert wire.error("r6") == "range_invalid"
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "", "sha256": big_sha, "offset": 0, "length": 10})
    assert wire.error("?") == "bad_request"
    tight = ArtifactRelay(registry, max_concurrent_reads=0)
    await tight.handle_frame(send, {"type": "blob.read", "request_id": "r7", "sha256": big_sha, "offset": 0, "length": 10})
    assert wire.error("r7") == "too_many_reads"
    assert not await relay.handle_frame(send, {"type": "input.text"})

    # 5. A closed connection ends its reads; a read from a second connection is answered there.
    other = Wire()
    other_send = other.send  # one sender per connection: the relay tells connections apart by identity
    await relay.handle_frame(other_send, {"type": "blob.read", "request_id": "r8", "sha256": big_sha, "offset": 0, "length": len(big), "credit": 1})
    await until(lambda: len(other.frames("r8")) >= 1, "r8 first frame")
    relay.connection_closed(other_send)
    await asyncio.sleep(0.05)
    assert relay.active_reads == 0
    assert wire.frames("r8") == []

    # 6. The manifest by its hash, and the announce frame.
    await relay.handle_frame(send, {"type": "blob.read", "request_id": "m1", "sha256": published.sha256, "offset": 0, "length": len(published.canonical)})
    await until(lambda: wire.ended("m1"), "m1 end")
    assert hashlib.sha256(b"".join(b["payload"] for b in wire.frames("m1"))).hexdigest() == published.sha256
    frame = relay.manifest_frame(task_id="t1", sha256=published.sha256, size=len(published.canonical), locations=[{"kind": "relay", "agent": "flow"}], progress={"done": 2, "total": 2})
    assert frame["type"] == "artifact.manifest" and frame["task_id"] == "t1" and frame["progress"] == {"done": 2, "total": 2}
