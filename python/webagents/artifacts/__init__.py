"""Large artifacts by reference: the ``webagents.artifact-set/v1`` manifest
and its strict reader, streaming hashes and chunk lists, ``files.publish``,
the ranged producer endpoint (``ArtifactsSkill``), the resuming client, the
A2A and MCP mapping, and the relay an agent serves over its outbound
connection."""

from .client import ArtifactFetchError, FetchedFile, FetchOptions, FetchResult, fetch_artifact_set, fetch_entry, fetch_manifest
from .hashing import HashedBytes, hash_bytes_aiter, hash_bytes_iter, hash_file, sha256_hex
from .http_range import parse_content_range, parse_range_header, parse_repr_digest, repr_digest
from .manifest import (
    ARTIFACT_SET_KIND,
    ARTIFACT_SET_MEDIA_TYPE,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_UNIT,
    LIMITS,
    LOCATION_KINDS,
    ArtifactEntry,
    ArtifactLocation,
    ArtifactProducer,
    ArtifactSet,
    ArtifactSetError,
    ChunkList,
    canonical_manifest,
    canonical_manifest_bytes,
    chunk_count,
    chunk_range,
    digest_name,
    is_valid_chunk_size,
    manifest_hash,
    parse_artifact_set,
    parse_digest_name,
    read_artifact_set,
    validate_chunk_list,
    validate_entry_path,
    validate_locations,
)
from .mapping import artifact_set_from_a2a_parts, artifact_set_to_a2a_parts, artifact_set_to_mcp_resource_links, preferred_url
from .publish import PublishedSet, blob_url, default_media_type, files, publish_folder
from .registry import AccessVerdict, ArtifactRegistry, RegisteredBlob, RegisteredManifest, agent_principal
from .relay import (
    ANNOUNCE_CONNECT_TIMEOUT_S,
    BLOB_FRAME_KIND_DATA,
    BLOB_FRAME_MAX_BYTES,
    BLOB_FRAME_VERSION,
    DEFAULT_CREDIT_FRAMES,
    MAX_CONCURRENT_READS,
    MAX_CREDIT_FRAMES,
    MAX_READ_BYTES,
    MAX_REQUEST_ID_BYTES,
    ArtifactAnnounceError,
    ArtifactRelay,
    decode_blob_data_frame,
    encode_blob_data_frame,
)
from .skill import ARTIFACT_BLOB_SUBPATH, ARTIFACT_TASK_HEADER, DEFAULT_MAX_CONCURRENT_TRANSFERS, ArtifactsSkill

__all__ = [
    "ARTIFACT_SET_KIND", "ARTIFACT_SET_MEDIA_TYPE", "CHUNK_SIZE_DEFAULT", "CHUNK_SIZE_MAX", "CHUNK_SIZE_MIN", "CHUNK_SIZE_UNIT",
    "LIMITS", "LOCATION_KINDS", "ArtifactEntry", "ArtifactLocation", "ArtifactProducer", "ArtifactSet", "ArtifactSetError", "ChunkList",
    "canonical_manifest", "canonical_manifest_bytes", "chunk_count", "chunk_range", "digest_name", "is_valid_chunk_size", "manifest_hash",
    "parse_artifact_set", "parse_digest_name", "read_artifact_set", "validate_chunk_list", "validate_entry_path", "validate_locations",
    "HashedBytes", "hash_bytes_aiter", "hash_bytes_iter", "hash_file", "sha256_hex",
    "parse_content_range", "parse_range_header", "parse_repr_digest", "repr_digest",
    "artifact_set_from_a2a_parts", "artifact_set_to_a2a_parts", "artifact_set_to_mcp_resource_links", "preferred_url",
    "PublishedSet", "blob_url", "default_media_type", "files", "publish_folder",
    "AccessVerdict", "ArtifactRegistry", "RegisteredBlob", "RegisteredManifest", "agent_principal",
    "BLOB_FRAME_KIND_DATA", "BLOB_FRAME_MAX_BYTES", "BLOB_FRAME_VERSION", "DEFAULT_CREDIT_FRAMES", "MAX_CONCURRENT_READS", "MAX_CREDIT_FRAMES",
    "MAX_READ_BYTES", "MAX_REQUEST_ID_BYTES", "ArtifactRelay", "decode_blob_data_frame", "encode_blob_data_frame",
    "ANNOUNCE_CONNECT_TIMEOUT_S", "ArtifactAnnounceError",
    "ARTIFACT_BLOB_SUBPATH", "ARTIFACT_TASK_HEADER", "DEFAULT_MAX_CONCURRENT_TRANSFERS", "ArtifactsSkill",
    "ArtifactFetchError", "FetchedFile", "FetchOptions", "FetchResult", "fetch_artifact_set", "fetch_entry", "fetch_manifest",
]
