"""`webagents.artifact-set/v1`: a folder of large files, by reference.

A manifest lists files by the SHA-256 of their bytes; the manifest's own
identity is the SHA-256 of its canonical form (RFC 8785, the same
canonicalisation an A2A agent card is signed over). A manifest may name
another manifest as an entry (``kind: "manifest"``), so a tree is a tree of
manifests. Every file above the chunk size also carries a chunk list
(fixed-size chunks, each with its own SHA-256) for transport; the file's
identity stays the plain hash of its bytes. Locations are ordered hints; a
reader tries them in order and verifies what it receives.

The reader is strict. A manifest is input from another party and is refused
by name (``ArtifactSetError.code``) when it breaks a rule. The TypeScript SDK
implements the same rules; ``tests/fixtures/artifact_set/vectors.json`` pins
the canonical bytes, the identities and the refusal codes for both.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from webagents.agents.skills.core.transport.a2a.jcs import canonical_bytes, canonicalize

ARTIFACT_SET_KIND = "webagents.artifact-set/v1"
ARTIFACT_SET_MEDIA_TYPE = "application/vnd.webagents.artifact-set+json"

CHUNK_SIZE_UNIT = 256 * 1024
CHUNK_SIZE_MIN = CHUNK_SIZE_UNIT
CHUNK_SIZE_MAX = 64 * 1024 * 1024
CHUNK_SIZE_DEFAULT = 8 * 1024 * 1024

LIMITS = {
    "max_manifest_bytes": 16 * 1024 * 1024,
    "max_entries": 10_000,
    "max_path_bytes": 1024,
    "max_path_segments": 64,
    "max_segment_bytes": 255,
    "max_depth": 8,
    "max_locations": 16,
    "max_url_bytes": 2048,
    "max_media_type_bytes": 255,
    "max_producer_bytes": 2048,
    "max_file_size": 2**53 - 1,
}

LOCATION_KINDS = ("https", "oci", "s3", "robutler", "relay")

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]{1,127}/[A-Za-z0-9!#$&^_.+-]{1,127}"
    r"(?:\s*;\s*[A-Za-z0-9!#$&^_.+-]+=(?:[A-Za-z0-9!#$&^_.+-]+|\"[^\"\\\\]*\"))*$"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")
_CONTROL = re.compile(r"[\u0000-\u001f\u007f]")
_TOP_FIELDS = {"kind", "createdAt", "expiresAt", "producer", "entries", "locations"}
_ENTRY_FIELDS = {"path", "size", "sha256", "mediaType", "kind", "chunks", "locations"}
_PRODUCER_FIELDS = {"agent", "task", "name"}
_LOCATION_FIELDS = {"kind", "url", "ref", "agent"}


class ArtifactSetError(ValueError):
    """A manifest refused by name. ``code`` is stable; ``detail`` names the place."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


@dataclass
class ChunkList:
    chunk_size: int
    hashes: List[str]

    def to_json(self) -> Dict[str, Any]:
        return {"chunkSize": self.chunk_size, "hashes": list(self.hashes)}


@dataclass
class ArtifactLocation:
    kind: str
    url: Optional[str] = None
    ref: Optional[str] = None
    agent: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.url is not None:
            out["url"] = self.url
        if self.ref is not None:
            out["ref"] = self.ref
        if self.agent is not None:
            out["agent"] = self.agent
        return out


@dataclass
class ArtifactEntry:
    path: str
    size: int
    sha256: str
    media_type: str
    kind: Optional[str] = None
    chunks: Optional[ChunkList] = None
    locations: Optional[List[ArtifactLocation]] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path, "size": self.size, "sha256": self.sha256, "mediaType": self.media_type}
        if self.kind is not None:
            out["kind"] = self.kind
        if self.chunks is not None:
            out["chunks"] = self.chunks.to_json()
        if self.locations is not None:
            out["locations"] = [loc.to_json() for loc in self.locations]
        return out


@dataclass
class ArtifactProducer:
    agent: str
    task: Optional[str] = None
    name: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"agent": self.agent}
        if self.task is not None:
            out["task"] = self.task
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass
class ArtifactSet:
    created_at: str
    expires_at: Optional[str]
    producer: ArtifactProducer
    entries: List[ArtifactEntry] = field(default_factory=list)
    locations: Optional[List[ArtifactLocation]] = None
    kind: str = ARTIFACT_SET_KIND

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
            "producer": self.producer.to_json(),
            "entries": [e.to_json() for e in self.entries],
        }
        if self.locations is not None:
            out["locations"] = [loc.to_json() for loc in self.locations]
        return out


def _byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _refuse_unknown(record: Dict[str, Any], allowed: set, where: str) -> None:
    for key in record:
        if key not in allowed:
            raise ArtifactSetError("unknown_field", f"{where} carries an unknown field {json.dumps(key)}", where=where, field=key)


def is_valid_chunk_size(value: Any) -> bool:
    return _is_int(value) and CHUNK_SIZE_MIN <= value <= CHUNK_SIZE_MAX and value % CHUNK_SIZE_UNIT == 0


def chunk_count(size: int, chunk_size: int) -> int:
    """How many chunks a file of ``size`` bytes has at ``chunk_size``."""
    return 0 if size == 0 else -(-size // chunk_size)


def chunk_range(index: int, size: int, chunk_size: int) -> tuple[int, int]:
    """Inclusive byte range of chunk ``index``."""
    start = index * chunk_size
    return start, min(size, start + chunk_size) - 1


def _is_loopback(hostname: str) -> bool:
    return hostname in ("localhost", "127.0.0.1", "::1", "[::1]")


def validate_entry_path(path: Any, index: int) -> str:
    where = f"entries[{index}].path"
    if not isinstance(path, str) or path == "":
        raise ArtifactSetError("path_invalid", f"{where} must be a non-empty string", index=index)
    if _byte_length(path) > LIMITS["max_path_bytes"]:
        raise ArtifactSetError("path_too_long", f"{where} is longer than {LIMITS['max_path_bytes']} bytes", index=index, limit=LIMITS["max_path_bytes"])
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path) or path.startswith("\\"):
        raise ArtifactSetError("path_absolute", f"{where} must be relative", index=index, path=path)
    if "\\" in path:
        raise ArtifactSetError("path_invalid", f"{where} must use '/' as the separator", index=index, path=path)
    if _CONTROL.search(path):
        raise ArtifactSetError("path_invalid", f"{where} carries a control character", index=index)
    segments = path.split("/")
    if len(segments) > LIMITS["max_path_segments"]:
        raise ArtifactSetError("path_too_many_segments", f"{where} has more than {LIMITS['max_path_segments']} segments", index=index, limit=LIMITS["max_path_segments"])
    for segment in segments:
        if segment == "..":
            raise ArtifactSetError("path_parent_segment", f"{where} escapes with '..'", index=index, path=path)
        if segment in ("", "."):
            raise ArtifactSetError("path_invalid", f"{where} has an empty or '.' segment", index=index, path=path)
        if _byte_length(segment) > LIMITS["max_segment_bytes"]:
            raise ArtifactSetError("path_too_long", f"{where} has a segment longer than {LIMITS['max_segment_bytes']} bytes", index=index, limit=LIMITS["max_segment_bytes"])
    return path


def validate_chunk_list(raw: Any, size: int, where: str) -> ChunkList:
    if not isinstance(raw, dict):
        raise ArtifactSetError("chunk_size_invalid", f"{where}.chunks must be an object", where=where)
    _refuse_unknown(raw, {"chunkSize", "hashes"}, f"{where}.chunks")
    chunk_size = raw.get("chunkSize")
    hashes = raw.get("hashes")
    if not is_valid_chunk_size(chunk_size):
        raise ArtifactSetError(
            "chunk_size_invalid",
            f"{where}.chunks.chunkSize must be a multiple of {CHUNK_SIZE_UNIT} between {CHUNK_SIZE_MIN} and {CHUNK_SIZE_MAX}",
            where=where, min=CHUNK_SIZE_MIN, max=CHUNK_SIZE_MAX, unit=CHUNK_SIZE_UNIT,
        )
    if not isinstance(hashes, list):
        raise ArtifactSetError("chunk_hash_invalid", f"{where}.chunks.hashes must be an array", where=where, index=-1)
    expected = chunk_count(size, chunk_size)
    if len(hashes) != expected:
        raise ArtifactSetError(
            "chunk_count_mismatch",
            f"{where}: a {size}-byte file in {chunk_size}-byte chunks has {expected} chunks, the list has {len(hashes)}",
            where=where, expected=expected, received=len(hashes),
        )
    out: List[str] = []
    for i, h in enumerate(hashes):
        if not isinstance(h, str) or not _SHA256_HEX.match(h):
            raise ArtifactSetError("chunk_hash_invalid", f"{where}.chunks.hashes[{i}] is not a lower-case hex SHA-256", where=where, index=i)
        out.append(h)
    return ChunkList(chunk_size=chunk_size, hashes=out)


def validate_locations(raw: Any, where: str, *, allow_http: bool = False) -> List[ArtifactLocation]:
    from urllib.parse import urlsplit

    if not isinstance(raw, list):
        raise ArtifactSetError("locations_invalid", f"{where} must be an array", where=where)
    if len(raw) > LIMITS["max_locations"]:
        raise ArtifactSetError("too_many_locations", f"{where} has more than {LIMITS['max_locations']} locations", where=where, limit=LIMITS["max_locations"])
    out: List[ArtifactLocation] = []
    for i, item in enumerate(raw):
        w = f"{where}[{i}]"
        if not isinstance(item, dict):
            raise ArtifactSetError("locations_invalid", f"{w} must be an object", where=w)
        _refuse_unknown(item, _LOCATION_FIELDS, w)
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in LOCATION_KINDS:
            raise ArtifactSetError("location_kind_invalid", f"{w}.kind must be one of {', '.join(LOCATION_KINDS)}", where=w, received=str(kind))
        location = ArtifactLocation(kind=kind)
        url = item.get("url")
        if url is not None:
            if not isinstance(url, str) or _byte_length(url) > LIMITS["max_url_bytes"]:
                raise ArtifactSetError("location_url_invalid", f"{w}.url must be a string of at most {LIMITS['max_url_bytes']} bytes", where=w)
            parts = urlsplit(url)
            if not parts.scheme or not parts.netloc:
                raise ArtifactSetError("location_url_invalid", f"{w}.url is not a URL", where=w)
            if kind in ("https", "robutler") and parts.scheme != "https" and not (parts.scheme == "http" and (allow_http or _is_loopback(parts.hostname or ""))):
                raise ArtifactSetError("location_url_invalid", f"{w}.url must be https", where=w, received=parts.scheme + ":")
            if kind == "s3" and parts.scheme != "https" and not (allow_http and parts.scheme == "http"):
                raise ArtifactSetError("location_url_invalid", f"{w}.url must be https", where=w, received=parts.scheme + ":")
            location.url = url
        ref = item.get("ref")
        if ref is not None:
            if not isinstance(ref, str) or ref == "" or _byte_length(ref) > LIMITS["max_url_bytes"]:
                raise ArtifactSetError("location_url_invalid", f"{w}.ref must be a non-empty string", where=w)
            location.ref = ref
        agent = item.get("agent")
        if agent is not None:
            if not isinstance(agent, str) or agent == "" or _byte_length(agent) > LIMITS["max_url_bytes"]:
                raise ArtifactSetError("location_url_invalid", f"{w}.agent must be a non-empty string", where=w)
            location.agent = agent
        if kind in ("https", "s3", "robutler") and location.url is None:
            raise ArtifactSetError("location_url_invalid", f"{w} ({kind}) needs a url", where=w, kind=kind)
        if kind == "oci" and location.ref is None:
            raise ArtifactSetError("location_url_invalid", f"{w} (oci) needs a ref", where=w, kind=kind)
        out.append(location)
    return out


def _validate_entry(raw: Any, index: int, *, allow_http: bool) -> ArtifactEntry:
    where = f"entries[{index}]"
    if not isinstance(raw, dict):
        raise ArtifactSetError("entry_invalid", f"{where} must be an object", index=index)
    _refuse_unknown(raw, _ENTRY_FIELDS, where)
    path = validate_entry_path(raw.get("path"), index)
    size = raw.get("size")
    if not _is_int(size) or size < 0 or size > LIMITS["max_file_size"]:
        raise ArtifactSetError("size_invalid", f"{where}.size must be a non-negative integer", index=index)
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX.match(sha256):
        raise ArtifactSetError("sha256_invalid", f"{where}.sha256 must be a lower-case hex SHA-256", index=index)
    media_type = raw.get("mediaType")
    if not isinstance(media_type, str) or _byte_length(media_type) > LIMITS["max_media_type_bytes"] or not _MEDIA_TYPE.match(media_type):
        raise ArtifactSetError("media_type_invalid", f"{where}.mediaType must be a media type", index=index)
    entry = ArtifactEntry(path=path, size=size, sha256=sha256, media_type=media_type)
    if "kind" in raw:
        if raw["kind"] not in ("file", "manifest"):
            raise ArtifactSetError("kind_invalid", f'{where}.kind must be "file" or "manifest"', index=index)
        entry.kind = raw["kind"]
    if "chunks" in raw:
        entry.chunks = validate_chunk_list(raw["chunks"], size, where)
    if "locations" in raw:
        entry.locations = validate_locations(raw["locations"], f"{where}.locations", allow_http=allow_http)
    return entry


def _validate_producer(raw: Any) -> ArtifactProducer:
    if not isinstance(raw, dict):
        raise ArtifactSetError("producer_invalid", "producer must be an object")
    _refuse_unknown(raw, _PRODUCER_FIELDS, "producer")
    agent = raw.get("agent")
    if not isinstance(agent, str) or agent == "" or _byte_length(agent) > LIMITS["max_producer_bytes"]:
        raise ArtifactSetError("producer_invalid", "producer.agent must be a non-empty string")
    producer = ArtifactProducer(agent=agent)
    for key in ("task", "name"):
        if key in raw:
            value = raw[key]
            if not isinstance(value, str) or _byte_length(value) > LIMITS["max_producer_bytes"]:
                raise ArtifactSetError("producer_invalid", f"producer.{key} must be a string", field=key)
            setattr(producer, key, value)
    return producer


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def parse_artifact_set(data: Union[bytes, bytearray, str, Dict[str, Any], Any], *, depth: int = 0, allow_http: bool = False) -> ArtifactSet:
    """Parse and validate a manifest from bytes, text or its object form.

    ``depth`` is how many manifests deep the caller already is; past
    ``LIMITS["max_depth"]`` the manifest is refused as ``nesting_too_deep``.
    ``allow_http`` accepts plaintext ``http`` locations beyond loopback (a
    private network in development); off by default.
    """
    if depth > LIMITS["max_depth"]:
        raise ArtifactSetError("nesting_too_deep", f"manifests nest deeper than {LIMITS['max_depth']}", limit=LIMITS["max_depth"], depth=depth)
    value: Any = data
    if isinstance(data, (bytes, bytearray, str)):
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        if len(raw) > LIMITS["max_manifest_bytes"]:
            raise ArtifactSetError("manifest_too_large", f"the manifest is larger than {LIMITS['max_manifest_bytes']} bytes", limit=LIMITS["max_manifest_bytes"], received=len(raw))
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ArtifactSetError("not_json", "the manifest is not UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ArtifactSetError("not_an_object", "the manifest must be a JSON object")
    if value.get("kind") != ARTIFACT_SET_KIND:
        raise ArtifactSetError("wrong_kind", f"kind must be {ARTIFACT_SET_KIND}", received=str(value.get("kind")))
    _refuse_unknown(value, _TOP_FIELDS, "the manifest")
    if not _valid_timestamp(value.get("createdAt")):
        raise ArtifactSetError("created_at_invalid", "createdAt must be an ISO 8601 UTC timestamp ending in Z")
    expires = value.get("expiresAt")
    if expires is not None and not _valid_timestamp(expires):
        raise ArtifactSetError("expires_at_invalid", "expiresAt must be null or an ISO 8601 UTC timestamp ending in Z")
    producer = _validate_producer(value.get("producer"))
    entries_raw = value.get("entries")
    if not isinstance(entries_raw, list):
        raise ArtifactSetError("entries_invalid", "entries must be an array")
    if len(entries_raw) > LIMITS["max_entries"]:
        raise ArtifactSetError("too_many_entries", f"more than {LIMITS['max_entries']} entries", limit=LIMITS["max_entries"], received=len(entries_raw))
    entries = [_validate_entry(raw, i, allow_http=allow_http) for i, raw in enumerate(entries_raw)]
    seen: set = set()
    dirs: set = set()
    for i, entry in enumerate(entries):
        if entry.path in seen:
            raise ArtifactSetError("path_duplicate", f"entries[{i}].path repeats {json.dumps(entry.path)}", index=i, path=entry.path)
        seen.add(entry.path)
        segments = entry.path.split("/")
        for d in range(1, len(segments)):
            dirs.add("/".join(segments[:d]))
    for p in seen:
        if p in dirs:
            raise ArtifactSetError("path_conflict", f"{json.dumps(p)} is both a file and a directory", path=p)
    out = ArtifactSet(created_at=value["createdAt"], expires_at=expires, producer=producer, entries=entries)
    if "locations" in value:
        out.locations = validate_locations(value["locations"], "locations", allow_http=allow_http)
    return out


def canonical_manifest(artifact_set: ArtifactSet) -> str:
    """The canonical (RFC 8785) text of a manifest: what its identity hashes."""
    return canonicalize(artifact_set.to_json())


def canonical_manifest_bytes(artifact_set: ArtifactSet) -> bytes:
    return canonical_bytes(artifact_set.to_json())


def manifest_hash(artifact_set: ArtifactSet) -> str:
    """Lower-case hex SHA-256 of the canonical bytes: the manifest's identity."""
    return hashlib.sha256(canonical_manifest_bytes(artifact_set)).hexdigest()


def read_artifact_set(raw: bytes, *, depth: int = 0, allow_http: bool = False) -> tuple[ArtifactSet, str]:
    """Parse bytes received as a manifest and name them: ``(set, sha256)``."""
    return parse_artifact_set(raw, depth=depth, allow_http=allow_http), hashlib.sha256(raw).hexdigest()


def digest_name(sha256: str) -> str:
    return f"sha256:{sha256}"


_DIGEST_NAME = re.compile(r"^sha256:([0-9a-f]{64})$")


def parse_digest_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    m = _DIGEST_NAME.match(name)
    return m.group(1) if m else None


__all__ = [
    "ARTIFACT_SET_KIND",
    "ARTIFACT_SET_MEDIA_TYPE",
    "CHUNK_SIZE_UNIT",
    "CHUNK_SIZE_MIN",
    "CHUNK_SIZE_MAX",
    "CHUNK_SIZE_DEFAULT",
    "LIMITS",
    "LOCATION_KINDS",
    "ArtifactSetError",
    "ArtifactSet",
    "ArtifactEntry",
    "ArtifactLocation",
    "ArtifactProducer",
    "ChunkList",
    "parse_artifact_set",
    "read_artifact_set",
    "canonical_manifest",
    "canonical_manifest_bytes",
    "manifest_hash",
    "validate_entry_path",
    "validate_chunk_list",
    "validate_locations",
    "is_valid_chunk_size",
    "chunk_count",
    "chunk_range",
    "digest_name",
    "parse_digest_name",
]
