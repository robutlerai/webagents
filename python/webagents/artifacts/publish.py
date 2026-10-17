"""``files.publish(folder)``: walk a folder, hash every regular file once
(streaming), write the manifest, register the blobs for serving, and return
the manifest with its hash. Symbolic links are refused rather than followed,
so a published folder can never reach outside itself."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Union

from .hashing import hash_file
from .manifest import (
    CHUNK_SIZE_DEFAULT,
    LIMITS,
    ArtifactEntry,
    ArtifactLocation,
    ArtifactProducer,
    ArtifactSet,
    canonical_manifest_bytes,
    digest_name,
    is_valid_chunk_size,
    manifest_hash,
    parse_artifact_set,
    validate_entry_path,
)
from .registry import ArtifactRegistry

_DEFAULT_MEDIA_TYPES = {
    ".json": "application/json",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gz": "application/gzip",
    ".zip": "application/zip",
    ".gds": "application/octet-stream",
    ".def": "text/plain",
}


def default_media_type(relative_path: str) -> str:
    dot = relative_path.rfind(".")
    ext = relative_path[dot:].lower() if dot >= 0 else ""
    return _DEFAULT_MEDIA_TYPES.get(ext, "application/octet-stream")


def blob_url(base_url: str, sha256: str) -> str:
    """``https`` location for a hash under an agent base URL."""
    return f"{base_url.rstrip('/')}/artifacts/blobs/{digest_name(sha256)}"


@dataclass
class PublishedSet:
    artifact_set: ArtifactSet
    sha256: str
    canonical: bytes
    manifest_url: Optional[str] = None
    skipped: List[dict] = field(default_factory=list)


def _iso(value: Union[datetime, str, None], fallback: Optional[str]) -> Optional[str]:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return value


def _walk(root: str, skipped: List[dict]) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames + [d for d in list(dirnames) if os.path.islink(os.path.join(dirpath, d))]):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                skipped.append({"path": rel, "reason": "symlink"})
                if name in dirnames:
                    dirnames.remove(name)
                continue
            if not os.path.isfile(full):
                skipped.append({"path": rel, "reason": "not_a_file"})
                continue
            out.append(full)
            if len(out) > LIMITS["max_entries"]:
                raise ValueError(f"the folder has more than {LIMITS['max_entries']} files; a set is limited to that many entries")
    out.sort(key=lambda p: os.path.relpath(p, root).replace(os.sep, "/"))
    return out


def publish_folder(
    folder: Union[str, os.PathLike],
    *,
    producer: ArtifactProducer,
    registry: ArtifactRegistry,
    base_url: Optional[str] = None,
    locations: Optional[Iterable[ArtifactLocation]] = None,
    audience: Optional[Union[str, Iterable[str]]] = None,
    expires_at: Union[datetime, str, None] = None,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    media_type: Optional[Callable[[str], Optional[str]]] = None,
    created_at: Union[datetime, str, None] = None,
    allow_http: bool = False,
) -> PublishedSet:
    """Publish a folder. See the module docstring."""
    if not is_valid_chunk_size(chunk_size):
        raise ValueError(f"invalid chunk size {chunk_size}")
    root = os.fspath(folder)
    skipped: List[dict] = []
    files = _walk(root, skipped)
    extra = list(locations or [])

    entries: List[ArtifactEntry] = []
    hashed: List[tuple] = []
    for full in files:
        rel = validate_entry_path(os.path.relpath(full, root).replace(os.sep, "/"), len(entries))
        h = hash_file(full, chunk_size)
        mt = (media_type(rel) if media_type else None) or default_media_type(rel)
        entry = ArtifactEntry(path=rel, size=h.size, sha256=h.sha256, media_type=mt, chunks=h.chunks)
        locs: List[ArtifactLocation] = []
        if base_url:
            locs.append(ArtifactLocation(kind="https", url=blob_url(base_url, h.sha256)))
        locs.extend(extra)
        if locs:
            entry.locations = locs
        entries.append(entry)
        hashed.append((full, h, mt))

    artifact_set = ArtifactSet(
        created_at=_iso(created_at, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),  # type: ignore[arg-type]
        expires_at=_iso(expires_at, None),
        producer=producer,
        entries=entries,
    )
    # The writer holds itself to the reader's rules.
    parse_artifact_set(artifact_set.to_json(), allow_http=allow_http)
    canonical = canonical_manifest_bytes(artifact_set)
    sha256 = manifest_hash(artifact_set)

    for full, h, mt in hashed:
        registry.add_blob(sha256=h.sha256, path=full, size=h.size, media_type=mt, chunks=h.chunks)
    registry.add_manifest(sha256=sha256, canonical=canonical, artifact_set=artifact_set, audience=audience, task=producer.task)

    return PublishedSet(
        artifact_set=artifact_set,
        sha256=sha256,
        canonical=canonical,
        manifest_url=blob_url(base_url, sha256) if base_url else None,
        skipped=skipped,
    )


class _Files:
    """The ``files`` namespace: ``files.publish(folder, ...)``."""

    publish = staticmethod(publish_folder)


files = _Files()

__all__ = ["publish_folder", "files", "blob_url", "default_media_type", "PublishedSet"]
