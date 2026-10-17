"""How a published set rides in the protocols an agent already speaks: A2A
parts (a data part for the manifest, file parts per entry) and MCP resource
links."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .manifest import ARTIFACT_SET_KIND, ARTIFACT_SET_MEDIA_TYPE, ArtifactEntry, ArtifactSet
from .publish import blob_url


def preferred_url(artifact_set: ArtifactSet, entry: ArtifactEntry, base_url: Optional[str] = None) -> Optional[str]:
    for loc in entry.locations or []:
        if loc.kind == "https" and loc.url:
            return loc.url
    for loc in artifact_set.locations or []:
        if loc.kind == "https" and loc.url:
            return loc.url
    return blob_url(base_url, entry.sha256) if base_url else None


def artifact_set_to_a2a_parts(artifact_set: ArtifactSet, sha256: str, *, base_url: Optional[str] = None, manifest_url: Optional[str] = None) -> List[Dict[str, Any]]:
    metadata: Dict[str, Any] = {"sha256": sha256}
    if manifest_url:
        metadata["url"] = manifest_url
    parts: List[Dict[str, Any]] = [{"data": artifact_set.to_json(), "mediaType": ARTIFACT_SET_MEDIA_TYPE, "metadata": metadata}]
    for entry in artifact_set.entries:
        url = preferred_url(artifact_set, entry, base_url)
        if not url:
            continue
        parts.append({"url": url, "mediaType": entry.media_type, "filename": entry.path, "metadata": {"sha256": entry.sha256, "size": entry.size}})
    return parts


def artifact_set_to_mcp_resource_links(artifact_set: ArtifactSet, sha256: str, *, base_url: Optional[str] = None, manifest_url: Optional[str] = None) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    manifest_uri = manifest_url or (blob_url(base_url, sha256) if base_url else None)
    if manifest_uri:
        links.append({
            "type": "resource_link",
            "uri": manifest_uri,
            "name": f"artifact-set {sha256[:12]}",
            "mimeType": ARTIFACT_SET_MEDIA_TYPE,
            "description": f"{len(artifact_set.entries)} files from {artifact_set.producer.agent}",
        })
    for entry in artifact_set.entries:
        uri = preferred_url(artifact_set, entry, base_url)
        if not uri:
            continue
        links.append({"type": "resource_link", "uri": uri, "name": entry.path, "mimeType": entry.media_type, "size": entry.size})
    return links


def artifact_set_from_a2a_parts(parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The manifest a list of A2A parts carries, if one does: ``{"data", "sha256"?}``."""
    for part in parts:
        data = part.get("data") if isinstance(part, dict) else None
        if isinstance(data, dict) and data.get("kind") == ARTIFACT_SET_KIND:
            out: Dict[str, Any] = {"data": data}
            meta = part.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("sha256"), str):
                out["sha256"] = meta["sha256"]
            return out
    return None


__all__ = ["preferred_url", "artifact_set_to_a2a_parts", "artifact_set_to_mcp_resource_links", "artifact_set_from_a2a_parts"]
