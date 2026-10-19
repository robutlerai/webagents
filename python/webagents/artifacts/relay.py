"""The agent side of the artifact relay: serving a platform's ``blob.read``
requests over the agent's own outbound connection, under a credit window.

Frames (documented under protocols/artifacts):

    platform -> agent   blob.read   { request_id, sha256, offset, length, credit? }
    platform -> agent   blob.credit { request_id, frames }
    platform -> agent   blob.cancel { request_id }
    agent -> platform   blob.data   BINARY: version, kind, request id, offset, payload
    agent -> platform   blob.end    { request_id, offset, length, bytes }
    agent -> platform   blob.error  { request_id, error: { code, message } }
    agent -> platform   artifact.manifest { task_id, sha256, size, locations, manifest?, progress?, session_id? }
    platform -> agent   artifact.error { sha256, error: { code, message } }   (a refused announcement)

A ``blob.data`` frame carries at most 256 KiB. The agent sends no more frames
than the credit it holds; frames of concurrent reads interleave by request
id; ``blob.cancel`` stops a read where it is; a dropped connection ends every
read on it.

Announcing: the transport that carries the relay (``PortalConnectSkill``)
tells it which control session each agent is on with ``attach_control``, and
``connection_closed`` forgets them. ``announce_set`` sends
``artifact.manifest`` on the producing agent's session, waiting for one when
the connection is not up yet. The platform answers only a refusal
(``artifact.error``); ``on_announce_error`` hears every refusal, and
``announce_set(confirm_within=...)`` waits that long for one before taking
silence as acceptance. Nothing is re-announced on its own after a reconnect:
the platform keeps what it was told."""

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
ANNOUNCE_CONNECT_TIMEOUT_S = 30.0

Sender = Callable[[Union[str, bytes]], Awaitable[None]]


class ArtifactAnnounceError(Exception):
    """A refused announcement: the platform's ``artifact.error``, or no
    session to send it on (``not_connected``)."""

    def __init__(self, code: str, message: str, sha256: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.sha256 = sha256


@dataclass
class _ControlSession:
    send: Sender
    session_id: str
    agent: str


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
        self._sessions: Dict[str, _ControlSession] = {}
        self._session_changed = asyncio.Event()
        self._refusal_listeners: List[Callable[[ArtifactAnnounceError], None]] = []

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

    def attach_control(self, send: Sender, session_id: str, agent: str) -> None:
        """The control session ``agent`` is on: called by the transport when the
        platform confirms it (``session.created``). ``connection_closed(send)``
        forgets every session on that connection."""
        self._sessions[agent] = _ControlSession(send=send, session_id=session_id, agent=agent)
        self._wake_session_waiters()

    @property
    def control_sessions(self) -> Dict[str, str]:
        """Agent name to session id, for the sessions attached now."""
        return {agent: s.session_id for agent, s in self._sessions.items()}

    def on_announce_error(self, listener: Callable[[ArtifactAnnounceError], None]) -> Callable[[], None]:
        """Hear every refused announcement (``artifact.error``), whichever call
        sent it. Returns a function that stops listening."""
        self._refusal_listeners.append(listener)

        def stop() -> None:
            if listener in self._refusal_listeners:
                self._refusal_listeners.remove(listener)

        return stop

    async def announce_set(
        self,
        published: Any,
        *,
        task_id: str,
        agent: Optional[str] = None,
        session_id: Optional[str] = None,
        locations: Optional[List[Dict[str, Any]]] = None,
        inline: bool = True,
        progress: Optional[Dict[str, int]] = None,
        connect_timeout: float = ANNOUNCE_CONNECT_TIMEOUT_S,
        confirm_within: float = 0.0,
    ) -> Dict[str, str]:
        """Announce a set ``files.publish`` returned on the control session of
        ``agent`` (optional when one session is attached). Waits up to
        ``connect_timeout`` seconds for that session, then raises
        ``not_connected``. Returns once the frame is sent, or, with
        ``confirm_within``, once that many seconds have passed without the
        platform refusing this set; a refusal in the window raises it.
        ``locations`` defaults to this relay under the session's agent.
        ``session_id`` names the relayed turn that produced the set
        (``portal_turn_of(context)["session_id"]`` in a tool), so the platform
        binds the set to the chat that turn came from; by default the agent's
        control session, which is bound to no chat. The frame always goes out
        on the control connection."""
        session = await self._wait_for_session(agent, connect_timeout)
        sha256 = published.sha256
        named_session = session_id or session.session_id
        frame = self.manifest_frame(
            task_id=task_id,
            session_id=named_session,
            sha256=sha256,
            size=len(published.canonical),
            locations=locations if locations is not None else [{"kind": "relay", "agent": session.agent}],
            manifest=json.loads(published.canonical) if inline else None,
            progress=progress,
        )
        result = {"sha256": sha256, "session_id": named_session}
        if confirm_within <= 0:
            await session.send(json.dumps(frame))
            return result
        loop = asyncio.get_running_loop()
        refused: asyncio.Future = loop.create_future()

        def listener(err: ArtifactAnnounceError) -> None:
            if err.sha256 is not None and err.sha256 != sha256:
                return
            if not refused.done():
                refused.set_result(err)

        stop = self.on_announce_error(listener)
        try:
            await session.send(json.dumps(frame))
            done, _ = await asyncio.wait({refused}, timeout=confirm_within)
            if not done:
                return result
            raise refused.result()
        finally:
            stop()

    async def _wait_for_session(self, agent: Optional[str], timeout: float) -> _ControlSession:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if agent is not None and agent in self._sessions:
                return self._sessions[agent]
            if agent is None and len(self._sessions) == 1:
                return next(iter(self._sessions.values()))
            if agent is None and len(self._sessions) > 1:
                raise ValueError(f"several agents are connected ({', '.join(sorted(self._sessions))}); name the one announcing")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ArtifactAnnounceError("not_connected", f"no control session within {timeout} s to announce on", None)
            self._session_changed.clear()
            try:
                await asyncio.wait_for(self._session_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    def _wake_session_waiters(self) -> None:
        self._session_changed.set()

    def _announce_refused(self, frame: Dict[str, Any]) -> None:
        error = frame.get("error") if isinstance(frame.get("error"), dict) else {}
        code = error.get("code") if isinstance(error.get("code"), str) and error.get("code") else "announce_failed"
        message = error.get("message") if isinstance(error.get("message"), str) and error.get("message") else "the platform refused the announcement"
        sha = frame.get("sha256")
        sha256 = sha if isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha) else None
        err = ArtifactAnnounceError(code, message, sha256)
        for listener in list(self._refusal_listeners):
            try:
                listener(err)
            except Exception:
                pass

    async def handle_frame(self, send: Sender, frame: Dict[str, Any]) -> bool:
        """Handle one JSON frame from the platform. True when it was the relay's
        (``blob.read``, ``blob.credit``, ``blob.cancel``, ``artifact.error``)."""
        kind = frame.get("type")
        if kind == "artifact.error":
            self._announce_refused(frame)
            return True
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
        """A connection dropped: every read on it is over, and the control
        sessions on it are gone."""
        for agent in [a for a, s in self._sessions.items() if s.send is send]:
            del self._sessions[agent]
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
