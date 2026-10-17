"""Fetching a set's files from its locations: ranged, parallel, resumable,
every chunk and every file verified.

For each entry the client takes the locations in order (the entry's own,
else the set's). Version 1 fetches ``https`` locations; every other kind is
refused by name as unsupported by this client, and the next location is
tried. A file with a chunk list is fetched one range per chunk, each chunk
hashed as it streams into the file at its offset and refused on a mismatch; a
file without one is fetched in fixed ranges and verified by the whole-file
hash at the end, which every file gets. Progress lives beside the file
(``<name>.part`` and ``<name>.part.json``), so a cut transfer resumes with
the chunks it already verified. Requests are signed with the fetching agent's
keys when given, which is what a producer endpoint authorises on."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from .http_range import parse_content_range, parse_repr_digest
from .manifest import (
    CHUNK_SIZE_DEFAULT,
    LIMITS,
    ArtifactEntry,
    ArtifactLocation,
    ArtifactSet,
    chunk_count,
    chunk_range,
    parse_artifact_set,
)
from .skill import ARTIFACT_TASK_HEADER

Identity = Tuple[Sequence[Any], str]  # (signing keys, agent URL), as the portal connection carries them

READ_SIZE = 1024 * 1024


class ArtifactFetchError(Exception):
    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


@dataclass
class FetchedFile:
    path: str
    absolute_path: str
    size: int
    sha256: str
    bytes_fetched: int


@dataclass
class FetchResult:
    files: List[FetchedFile] = field(default_factory=list)
    bytes_fetched: int = 0


@dataclass
class FetchOptions:
    dest: str
    client: httpx.AsyncClient
    identity: Optional[Identity] = None
    concurrency: int = 4
    range_size: int = CHUNK_SIZE_DEFAULT
    allow_http: bool = False
    task: Optional[str] = None
    location_kinds: Sequence[str] = ("https",)
    depth: int = 0


def _signed_headers(options: FetchOptions, url: str, headers: Dict[str, str]) -> Dict[str, str]:
    out = dict(headers)
    if options.task:
        out[ARTIFACT_TASK_HEADER] = options.task
    if options.identity is not None:
        from webagents.crypto.http_signature import sign_request

        keys, agent_url = options.identity
        signed = sign_request(list(keys), agent_url, "GET", url, allow_http=options.allow_http)
        out.update(signed.headers)
    return out


async def fetch_manifest(url: str, options: FetchOptions, *, expected_sha256: Optional[str] = None) -> Tuple[ArtifactSet, str, bytes]:
    """Fetch and parse a manifest by URL; returns ``(set, sha256, bytes)``."""
    response = await options.client.get(url, headers=_signed_headers(options, url, {}))
    if response.status_code != 200:
        raise ArtifactFetchError("bad_response", f"{url} answered {response.status_code}", status=response.status_code, url=url)
    raw = response.content
    if len(raw) > LIMITS["max_manifest_bytes"]:
        raise ArtifactFetchError("manifest_too_large", f"{url} sent more than {LIMITS['max_manifest_bytes']} bytes for a manifest", url=url)
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and expected_sha256 != sha256:
        raise ArtifactFetchError("file_hash_mismatch", f"the manifest at {url} hashes to {sha256}, not {expected_sha256}", url=url, expected=expected_sha256, actual=sha256)
    declared = parse_repr_digest(response.headers.get("repr-digest"))
    if declared and declared != sha256:
        raise ArtifactFetchError("bad_response", f"{url}: Repr-Digest names {declared}, the bytes hash to {sha256}", url=url)
    return parse_artifact_set(raw, depth=options.depth, allow_http=options.allow_http), sha256, raw


async def fetch_artifact_set(artifact_set: ArtifactSet, options: FetchOptions) -> FetchResult:
    """Fetch every entry of ``artifact_set`` into ``options.dest``."""
    if options.depth > LIMITS["max_depth"]:
        raise ArtifactFetchError("nesting_too_deep", f"manifests nest deeper than {LIMITS['max_depth']}", depth=options.depth)
    result = FetchResult()
    for entry in artifact_set.entries:
        if entry.kind == "manifest":
            if entry.size > LIMITS["max_manifest_bytes"]:
                raise ArtifactFetchError("manifest_too_large", f"{entry.path} declares {entry.size} bytes for a manifest", path=entry.path)
            manifest_file = await fetch_entry(artifact_set, entry, options)
            result.files.append(manifest_file)
            result.bytes_fetched += manifest_file.bytes_fetched
            with open(manifest_file.absolute_path, "rb") as f:
                nested = parse_artifact_set(f.read(), depth=options.depth + 1, allow_http=options.allow_http)
            nested_options = FetchOptions(**{**options.__dict__, "dest": os.path.join(options.dest, os.path.dirname(entry.path)), "depth": options.depth + 1})
            nested_result = await fetch_artifact_set(nested, nested_options)
            result.files.extend(nested_result.files)
            result.bytes_fetched += nested_result.bytes_fetched
            continue
        fetched = await fetch_entry(artifact_set, entry, options)
        result.files.append(fetched)
        result.bytes_fetched += fetched.bytes_fetched
    return result


def _target_for(dest: str, path: str) -> str:
    root = os.path.realpath(dest)
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ArtifactFetchError("path_escape", f"{path} resolves outside the destination", path=path)
    return target


def _load_state(state_path: str, sha256: str, size: int, piece_size: int) -> Optional[List[int]]:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("sha256") != sha256 or raw.get("size") != size or raw.get("pieceSize") != piece_size:
        return None
    done = raw.get("done")
    if not isinstance(done, list):
        return None
    return [n for n in done if isinstance(n, int) and n >= 0]


def _candidate_locations(artifact_set: ArtifactSet, entry: ArtifactEntry) -> List[ArtifactLocation]:
    if entry.locations:
        return list(entry.locations)
    return list(artifact_set.locations or [])


async def fetch_entry(artifact_set: ArtifactSet, entry: ArtifactEntry, options: FetchOptions) -> FetchedFile:
    """Fetch one entry; resumes a ``.part`` left by an earlier run."""
    target = _target_for(options.dest, entry.path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part_path = f"{target}.part"
    state_path = f"{target}.part.json"
    piece_size = entry.chunks.chunk_size if entry.chunks else options.range_size
    pieces = chunk_count(entry.size, piece_size)
    done_list = _load_state(state_path, entry.sha256, entry.size, piece_size)
    resumed = done_list is not None and os.path.isfile(part_path) and os.path.getsize(part_path) == entry.size
    done: set = set(done_list or []) if resumed else set()
    pending = [i for i in range(pieces) if i not in done]

    flags = os.O_RDWR | (0 if resumed else os.O_CREAT | os.O_TRUNC)
    fd = os.open(part_path, flags, 0o644)
    bytes_fetched = 0
    lock = asyncio.Lock()
    try:
        if not resumed and entry.size > 0:
            os.ftruncate(fd, entry.size)
        locations = _candidate_locations(artifact_set, entry)

        def persist() -> None:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"sha256": entry.sha256, "size": entry.size, "pieceSize": piece_size, "done": sorted(done)}, f)

        # A small pool pulling from one queue, stopping at the first failure:
        # what was verified before the failure is kept for the resume, nothing
        # after it is started.
        queue = list(pending)
        failure: List[BaseException] = []

        async def worker() -> None:
            nonlocal bytes_fetched
            while queue and not failure:
                index = queue.pop(0)
                start, end = chunk_range(index, entry.size, piece_size)
                expected = entry.chunks.hashes[index] if entry.chunks else None
                try:
                    got = await _fetch_range(locations, entry, start, end, expected, fd, options)
                except BaseException as error:  # noqa: BLE001 - recorded, raised below
                    failure.append(error)
                    return
                async with lock:
                    bytes_fetched += got
                    done.add(index)
                    persist()

        await asyncio.gather(*(worker() for _ in range(min(max(1, options.concurrency), max(1, len(pending))))))
        if failure:
            raise failure[0]
    finally:
        os.close(fd)

    actual = _hash_path(part_path)
    if actual != entry.sha256:
        for p in (part_path, state_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise ArtifactFetchError("file_hash_mismatch", f"{entry.path} hashed to {actual}, the manifest says {entry.sha256}", path=entry.path, expected=entry.sha256, actual=actual)
    os.replace(part_path, target)
    try:
        os.remove(state_path)
    except OSError:
        pass
    return FetchedFile(path=entry.path, absolute_path=target, size=entry.size, sha256=entry.sha256, bytes_fetched=bytes_fetched)


async def _fetch_range(locations: List[ArtifactLocation], entry: ArtifactEntry, start: int, end: int, expected_sha256: Optional[str], fd: int, options: FetchOptions) -> int:
    allowed = set(options.location_kinds)
    refusals: List[Dict[str, Any]] = []
    for location in locations:
        if location.kind not in allowed or location.kind != "https" or not location.url:
            refusals.append({"location": location.to_json(), "code": "location_kind_unsupported", "message": f"location kind {location.kind} is not supported by this client"})
            continue
        if not options.allow_http and not location.url.startswith("https://"):
            refusals.append({"location": location.to_json(), "code": "location_kind_unsupported", "message": "only https locations are fetched"})
            continue
        try:
            return await _fetch_range_from(location.url, entry, start, end, expected_sha256, fd, options)
        except ArtifactFetchError as error:
            refusals.append({"location": location.to_json(), "code": error.code, "message": str(error)})
        except (httpx.HTTPError, OSError) as error:
            refusals.append({"location": location.to_json(), "code": "bad_response", "message": str(error) or error.__class__.__name__})
    summary = "; ".join(f"{r['location']['kind']}: {r['code']}" for r in refusals)
    raise ArtifactFetchError("no_reachable_location", f"no location served {entry.path} bytes {start}-{end}: {summary}", path=entry.path, start=start, end=end, refusals=refusals)


async def _fetch_range_from(url: str, entry: ArtifactEntry, start: int, end: int, expected_sha256: Optional[str], fd: int, options: FetchOptions) -> int:
    want_length = end - start + 1
    whole = start == 0 and end == entry.size - 1
    headers = _signed_headers(options, url, {"Range": f"bytes={start}-{end}"})
    async with options.client.stream("GET", url, headers=headers) as response:
        status = response.status_code
        if status in (401, 403, 404, 416) or status >= 500:
            raise ArtifactFetchError("bad_response", f"{url} answered {status}", status=status)
        body_start, body_length = start, want_length
        if status == 206:
            cr = parse_content_range(response.headers.get("content-range"))
            if cr is None or cr[0] != start or cr[1] != end or (cr[2] != -1 and cr[2] != entry.size):
                raise ArtifactFetchError("bad_response", f"{url} answered a range other than {start}-{end}/{entry.size}: {response.headers.get('content-range')}")
        elif status == 200:
            if not whole:
                raise ArtifactFetchError("bad_response", f"{url} does not serve ranges")
            body_start, body_length = 0, entry.size
        else:
            raise ArtifactFetchError("bad_response", f"{url} answered {status}", status=status)
        declared = parse_repr_digest(response.headers.get("repr-digest"))
        if declared and declared != entry.sha256:
            raise ArtifactFetchError("bad_response", f"{url}: Repr-Digest names {declared}, the manifest says {entry.sha256}")
        hasher = hashlib.sha256() if expected_sha256 else None
        written = 0
        position = body_start
        async for piece in response.aiter_bytes(READ_SIZE):
            if written + len(piece) > body_length:
                raise ArtifactFetchError("size_mismatch", f"{url} sent more than the {body_length} bytes asked for", url=url)
            if hasher is not None:
                hasher.update(piece)
            os.pwrite(fd, piece, position)
            position += len(piece)
            written += len(piece)
    if written != body_length:
        raise ArtifactFetchError("size_mismatch", f"{url} sent {written} bytes of the {body_length} asked for", url=url, expected=body_length, actual=written)
    if hasher is not None:
        actual = hasher.hexdigest()
        if actual != expected_sha256:
            raise ArtifactFetchError("chunk_hash_mismatch", f"{entry.path} bytes {start}-{end} hashed to {actual}, the chunk list says {expected_sha256}", path=entry.path, start=start, end=end, expected=expected_sha256, actual=actual)
    return written


def _hash_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            piece = f.read(READ_SIZE)
            if not piece:
                break
            h.update(piece)
    return h.hexdigest()


__all__ = ["ArtifactFetchError", "FetchOptions", "FetchResult", "FetchedFile", "fetch_manifest", "fetch_artifact_set", "fetch_entry"]
