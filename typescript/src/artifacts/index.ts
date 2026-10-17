/**
 * Large artifacts by reference: the `webagents.artifact-set/v1` manifest, its
 * strict reader, streaming hashes and chunk lists, `files.publish`, the ranged
 * producer endpoint (`ArtifactsSkill`), the resuming client, the A2A and MCP
 * mapping, and the relay an agent serves over its outbound connection.
 */

export {
  ARTIFACT_SET_KIND,
  ARTIFACT_SET_MEDIA_TYPE,
  ARTIFACT_SET_LIMITS,
  CHUNK_SIZE_UNIT,
  CHUNK_SIZE_MIN,
  CHUNK_SIZE_MAX,
  CHUNK_SIZE_DEFAULT,
  LOCATION_KINDS,
  ArtifactSetError,
  parseArtifactSet,
  readArtifactSet,
  canonicalManifest,
  canonicalManifestBytes,
  manifestHash,
  validateEntryPath,
  validateChunkList,
  validateLocations,
  isValidChunkSize,
  chunkCount,
  chunkRange,
  digestName,
  parseDigestName,
} from './manifest';
export type {
  ArtifactSet,
  ArtifactEntry,
  ArtifactLocation,
  ArtifactProducer,
  ArtifactSetErrorCode,
  ChunkList,
  LocationKind,
  ReadOptions,
} from './manifest';
export { hashFile, hashStream, sha256Hex, HASH_READ_HIGH_WATER_MARK } from './hash';
export type { HashedBytes } from './hash';
export { ArtifactRegistry, agentPrincipal } from './registry';
export type { Audience, RegisteredBlob, RegisteredManifest } from './registry';
export { publishFolder, files, blobUrl, defaultMediaType } from './publish';
export type { PublishOptions, PublishedSet } from './publish';
export { parseRangeHeader, parseContentRange, reprDigest, parseReprDigest } from './http-range';
export type { ParsedRange } from './http-range';
export { ArtifactsSkill, ARTIFACT_BLOB_SUBPATH, ARTIFACT_TASK_HEADER, DEFAULT_MAX_CONCURRENT_TRANSFERS } from './skill';
export type { ArtifactsSkillConfig } from './skill';
export { fetchArtifactSet, fetchEntry, fetchManifest, ArtifactFetchError } from './client';
export type { FetchArtifactOptions, FetchResult, FetchedFile, ArtifactFetchErrorCode } from './client';
export { artifactSetToA2AParts, artifactSetToMcpResourceLinks, artifactSetFromA2AParts, preferredUrl } from './mapping';
export type { McpResourceLink } from './mapping';
export {
  ArtifactRelay,
  encodeBlobDataFrame,
  decodeBlobDataFrame,
  BLOB_FRAME_MAX_BYTES,
  BLOB_FRAME_VERSION,
  BLOB_FRAME_KIND_DATA,
  DEFAULT_CREDIT_FRAMES,
  MAX_CREDIT_FRAMES,
  MAX_CONCURRENT_READS,
  MAX_READ_BYTES,
  MAX_REQUEST_ID_BYTES,
} from './relay';
export type { RelaySocket, BlobReadRequest, RelayErrorCode, ArtifactRelayOptions } from './relay';
