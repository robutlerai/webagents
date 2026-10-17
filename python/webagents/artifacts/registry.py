"""What this agent serves by hash: the blobs it published (a path on disk)
and the manifests that name them (their canonical bytes), with who may read
them.

Access is by caller, never by link: a manifest is published to an audience of
principals (``agent:<url>``, the form the access skill records for a caller
whose signature it verified), or to every verified caller (``"*"``). A blob
is readable by the union of the audiences of the manifests that list it. A
manifest published for a task may require the reader to name it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from .manifest import ArtifactSet, ChunkList

Audience = Union[str, Set[str]]  # "*" or a set of principals


def agent_principal(value: str) -> str:
    return value if value.startswith("agent:") else f"agent:{value}"


@dataclass
class RegisteredBlob:
    sha256: str
    path: str
    size: int
    media_type: str
    chunks: Optional[ChunkList] = None
    manifests: Set[str] = field(default_factory=set)


@dataclass
class RegisteredManifest:
    sha256: str
    canonical: bytes
    artifact_set: ArtifactSet
    audience: Audience = "*"
    task: Optional[str] = None


@dataclass
class AccessVerdict:
    ok: bool
    reason: Optional[str] = None  # unknown | forbidden | task_required | task_mismatch


class ArtifactRegistry:
    def __init__(self) -> None:
        self._blobs: Dict[str, RegisteredBlob] = {}
        self._manifests: Dict[str, RegisteredManifest] = {}

    def add_blob(self, *, sha256: str, path: str, size: int, media_type: str, chunks: Optional[ChunkList] = None) -> RegisteredBlob:
        existing = self._blobs.get(sha256)
        if existing is not None:
            return existing
        blob = RegisteredBlob(sha256=sha256, path=path, size=size, media_type=media_type, chunks=chunks)
        self._blobs[sha256] = blob
        return blob

    def add_manifest(
        self,
        *,
        sha256: str,
        canonical: bytes,
        artifact_set: ArtifactSet,
        audience: Optional[Union[str, Iterable[str]]] = None,
        task: Optional[str] = None,
    ) -> RegisteredManifest:
        resolved: Audience = "*" if audience is None or audience == "*" else {agent_principal(p) for p in audience}
        manifest = RegisteredManifest(sha256=sha256, canonical=canonical, artifact_set=artifact_set, audience=resolved, task=task)
        self._manifests[sha256] = manifest
        for entry in artifact_set.entries:
            blob = self._blobs.get(entry.sha256)
            if blob is not None:
                blob.manifests.add(sha256)
        return manifest

    def get_blob(self, sha256: str) -> Optional[RegisteredBlob]:
        return self._blobs.get(sha256)

    def get_manifest(self, sha256: str) -> Optional[RegisteredManifest]:
        return self._manifests.get(sha256)

    def grant(self, manifest_sha256: str, principal: str) -> None:
        manifest = self._manifests.get(manifest_sha256)
        if manifest is None:
            raise KeyError(manifest_sha256)
        if manifest.audience == "*":
            return
        assert isinstance(manifest.audience, set)
        manifest.audience.add(agent_principal(principal))

    def remove(self, manifest_sha256: str) -> None:
        manifest = self._manifests.pop(manifest_sha256, None)
        if manifest is None:
            return
        for entry in manifest.artifact_set.entries:
            blob = self._blobs.get(entry.sha256)
            if blob is None:
                continue
            blob.manifests.discard(manifest_sha256)
            if not blob.manifests:
                del self._blobs[entry.sha256]

    def access(self, sha256: str, principals: Sequence[str], task: Optional[str]) -> AccessVerdict:
        """Whether a caller holding ``principals`` may read ``sha256``."""
        manifest = self._manifests.get(sha256)
        candidates: List[RegisteredManifest] = [manifest] if manifest else []
        if manifest is None:
            blob = self._blobs.get(sha256)
            if blob is None:
                return AccessVerdict(False, "unknown")
            candidates = [m for h in blob.manifests if (m := self._manifests.get(h)) is not None]
            if not candidates:
                return AccessVerdict(False, "unknown")
        reason = "forbidden"
        for m in candidates:
            in_audience = m.audience == "*" or any(p in m.audience for p in principals)  # type: ignore[operator]
            if not in_audience:
                continue
            if m.task is not None:
                if task is None:
                    reason = "task_required"
                    continue
                if task != m.task:
                    reason = "task_mismatch"
                    continue
            return AccessVerdict(True)
        return AccessVerdict(False, reason)

    def list_manifests(self) -> List[RegisteredManifest]:
        return list(self._manifests.values())

    def list_blobs(self) -> List[RegisteredBlob]:
        return list(self._blobs.values())


__all__ = ["ArtifactRegistry", "RegisteredBlob", "RegisteredManifest", "AccessVerdict", "agent_principal", "Audience"]
