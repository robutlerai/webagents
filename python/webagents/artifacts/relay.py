"""The agent side of the artifact relay: serving a platform's ``blob.read``
requests over the agent's own outbound connection, under a credit window.

Frames (documented under protocols/artifacts):

    platform -> agent   blob.read   { request_id, sha256, offset, length, credit? }
    platform -> agent   blob.credit { request_id, frames }
    platform -> agent   blob.cancel { request_id }
    agent -> platform   blob.data   BINARY: version, kind, request id, offset, payload
    agent -> platform   blob.end    { request_id, offset, length, bytes }
    agent -> platform   blob.error  { request_id, error: { code, message } }
    agent -> platform   artifact.manifest { task_id, sha256, size, locations, manifest?, progress? }

A ``blob.data`` frame carries at most 256 KiB. The agent sends no more frames
than the credit it holds; frames of concurrent reads interleave by request
id; ``blob.cancel`` stops a read where it is; a dropped connection ends every
read on it."""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .registry import ArtifactRegistry

BLOB_FRAME_VERSION = 1
BLOB_FRAME_KIND_DATA = 1
BLOB_FRAME_MAX_BYTES = 256 * 1024
DEFAULT_CREDIT_FRAMES = 4
MAX_CREDIT_FRAMES = 1024
MAX_CONCURRENT_READS = 16
MAX_READ_BYTES = 64 * 1024 * 1024
MAX_REQUEST_ID_BYTES = 64

Sender = Callable[[Union[str, bytes]], Awaitable[None]]


def encode_blob_data_frame(request_id: str, offset: int, payload: bytes) -> bytes:
    rid = request_id.encode("utf-8")
    if not 1 <= len(rid) <= MAX_REQUEST_ID_BYTES:
        raise ValueError("request id must be 1 to 64 bytes")
    if len(payload) > BLOB_FRAME_MAX_BYTES:
        raise ValueError(f"a blob.data payload is at most {BLOB_FRAME_MAX_BYTES} bytes")
    return struct.pack(">BBH", BLOB_FRAME_VERSION, BLOB_FRAME_KIND_DATA, len(rid)) + rid + struct.pack(">Q", offset) + payload


def decode_blob_data_frame(frame: bytes) -> Optional[Dict[str, Any]]:
    if len(frame) < 12 or frame[0] != BLOB_FRAME_VERSION or frame[1] != BLOB_FRAME_KIND_DATA:
        return None
    (rid_len,) = struct.unpack(">H", frame[2:4])
    if rid_len == 0 or rid_len > MAX_REQUEST_ID_BYTES or len(frame) < 12 + rid_len:
        return None
    try:
        request_id = frame[4:4 + rid_len].decode("utf-8")
    except UnicodeDecodeError:
        return None
    (offset,) = struct.unpack(">Q", frame[4 + rid_len:12 + rid_len])
    payload = frame[12 + rid_len:]
    if len(payload) > BLOB_FRAME_MAX_BYTES:
        return None
    return {"request_id": request_id, "offset": offset, "payload": payload}


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class _Read:
    request_id: str
    send: Sender
    credit: int
    cancelled: bool = False
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None


class ArtifactRelay:
    def __init__(self, registry: ArtifactRegistry, *, max_concurrent_reads: int = MAX_CONCURRENT_READS, frame_bytes: int = BLOB_FRAME_MAX_BYTES) -> None:
        self.registry = registry
        self.max_reads = max_concurrent_reads
        self.frame_bytes = min(BLOB_FRAME_MAX_BYTES, frame_bytes)
        self._reads: Dict[str, _Read] = {}

    @property
    def active_reads(self) -> int:
        return len(self._reads)

    def manifest_frame(self, *, task_id: str, sha256: str, size: int, locations: List[Dict[str, Any]], manifest: Any = None, progress: Optional[Dict[str, int]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        frame: Dict[str, Any] = {
            "type": "artifact.manifest",
            "event_id": _event_id(),
            "timestamp": _now_ms(),
            "task_id": task_id,
            "sha256": sha256,
            "size": size,
            "locations": locations,
        }
        if session_id:
            frame["session_id"] = session_id
        if manifest is not None:
            frame["manifest"] = manifest
        if progress is not None:
            frame["progress"] = progress
        return frame

    async def announce(self, send: Sender, **kwargs: Any) -> None:
        await send(json.dumps(self.manifest_frame(**kwargs)))

    async def handle_frame(self, send: Sender, frame: Dict[str, Any]) -> bool:
        """Handle one JSON frame from the platform. True when it was the relay's."""
        kind = frame.get("type")
        if kind == "blob.read":
            await self._start_read(send, frame)
            return True
        if kind == "blob.credit":
            read = self._reads.get(str(frame.get("request_id")))
            frames = frame.get("frames")
            if read is not None and read.send is send and isinstance(frames, int) and not isinstance(frames, bool) and frames > 0:
                read.credit = min(MAX_CREDIT_FRAMES, read.credit + frames)
                read.wake.set()
            return True
        if kind == "blob.cancel":
            read = self._reads.get(str(frame.get("request_id")))
            if read is not None and read.send is send:
                read.cancelled = True
                read.wake.set()
            return True
        return False

    def connection_closed(self, send: Sender) -> None:
        """A connection dropped: every read on it is over."""
        for read in list(self._reads.values()):
            if read.send is send:
                read.cancelled = True
                read.wake.set()

    async def _error(self, send: Sender, request_id: str, code: str, message: str) -> None:
        await send(json.dumps({"type": "blob.error", "event_id": _event_id(), "timestamp": _now_ms(), "request_id": request_id, "error": {"code": code, "message": message}}))

    async def _start_read(self, send: Sender, frame: Dict[str, Any]) -> None:
        request_id = frame.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id.encode("utf-8")) > MAX_REQUEST_ID_BYTES:
            await self._error(send, request_id if isinstance(request_id, str) and request_id else "?", "bad_request", "request_id must be a string of 1 to 64 bytes")
            return
        if request_id in self._reads:
            await self._error(send, request_id, "duplicate_request", "a read with this request_id is in progress")
            return
        sha256 = frame.get("sha256")
        offset = frame.get("offset")
        length = frame.get("length")
        is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
        if not (isinstance(sha256, str) and len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256)) or not is_int(offset) or not is_int(length) or offset < 0 or length <= 0:
            await self._error(send, request_id, "bad_request", "blob.read needs sha256 (hex), offset (>= 0) and length (> 0)")
            return
        if length > MAX_READ_BYTES:
            await self._error(send, request_id, "range_invalid", f"one read is at most {MAX_READ_BYTES} bytes")
            return
        blob = self.registry.get_blob(sha256)
        manifest = None if blob else self.registry.get_manifest(sha256)
        if blob is None and manifest is None:
            await self._error(send, request_id, "unknown_blob", "this agent has not published that hash")
            return
        size = blob.size if blob else len(manifest.canonical)  # type: ignore[union-attr]
        if offset >= size or offset + length > size:
            await self._error(send, request_id, "range_invalid", f"the blob is {size} bytes; {offset}+{length} is past its end")
            return
        if len(self._reads) >= self.max_reads:
            await self._error(send, request_id, "too_many_reads", f"this agent serves at most {self.max_reads} reads at once")
            return
        credit = frame.get("credit")
        credit = min(MAX_CREDIT_FRAMES, credit) if is_int(credit) and credit > 0 else DEFAULT_CREDIT_FRAMES
        read = _Read(request_id=request_id, send=send, credit=credit)
        self._reads[request_id] = read
        read.task = asyncio.create_task(self._run_read(read, blob, manifest, offset, length))

    async def _run_read(self, read: _Read, blob: Any, manifest: Any, offset: int, length: int) -> None:
        sent = 0
        position = offset
        try:
            async def emit(payload: bytes) -> bool:
                nonlocal sent, position
                for off in range(0, len(payload), self.frame_bytes):
                    while read.credit <= 0 and not read.cancelled:
                        read.wake.clear()
                        await read.wake.wait()
                    if read.cancelled:
                        return False
                    piece = payload[off:off + self.frame_bytes]
                    await read.send(encode_blob_data_frame(read.request_id, position, piece))
                    position += len(piece)
                    sent += len(piece)
                    read.credit -= 1
                    # One frame per turn so concurrent reads interleave.
                    await asyncio.sleep(0)
                return True

            if manifest is not None:
                if not await emit(bytes(manifest.canonical[offset:offset + length])):
                    return
            else:
                loop = asyncio.get_running_loop()
                with open(blob.path, "rb") as f:
                    f.seek(offset)
                    remaining = length
                    while remaining > 0:
                        piece = await loop.run_in_executor(None, f.read, min(self.frame_bytes, remaining))
                        if not piece:
                            raise OSError("the file ended before the range did")
                        remaining -= len(piece)
                        if not await emit(piece):
                            return
            if read.cancelled:
                return
            await read.send(json.dumps({"type": "blob.end", "event_id": _event_id(), "timestamp": _now_ms(), "request_id": read.request_id, "offset": offset, "length": length, "bytes": sent}))
        except Exception as error:  # noqa: BLE001 - the read's failure is reported to the reader
            if not read.cancelled:
                try:
                    await self._error(read.send, read.request_id, "read_failed", str(error))
                except Exception:  # noqa: BLE001 - the connection is gone
                    pass
        finally:
            self._reads.pop(read.request_id, None)


__all__ = [
    "ArtifactRelay",
    "encode_blob_data_frame",
    "decode_blob_data_frame",
    "BLOB_FRAME_MAX_BYTES",
    "BLOB_FRAME_VERSION",
    "BLOB_FRAME_KIND_DATA",
    "DEFAULT_CREDIT_FRAMES",
    "MAX_CREDIT_FRAMES",
    "MAX_CONCURRENT_READS",
    "MAX_READ_BYTES",
    "MAX_REQUEST_ID_BYTES",
]
