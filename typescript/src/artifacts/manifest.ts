/**
 * `webagents.artifact-set/v1`: a folder of large files, by reference.
 *
 * A manifest lists files by the SHA-256 of their bytes; the manifest's own
 * identity is the SHA-256 of its canonical form (RFC 8785 JSON Canonicalization
 * Scheme, the same `canonicalize` an A2A agent card is signed over). A manifest
 * may name another manifest as an entry (`kind: "manifest"`), so a tree is a
 * tree of manifests. Every file above the chunk size also carries a chunk list
 * (fixed-size chunks, each with its own SHA-256) for transport: chunks are what
 * moves, resumes and is verified on arrival; the file's identity stays the
 * plain hash of its bytes. Locations are ordered hints; a reader tries them in
 * order and verifies what it receives, so none has to be trusted.
 *
 * The reader is strict. A manifest is input from another party and is refused
 * by name (`ArtifactSetError.code`) when it breaks a rule: a path that is
 * absolute, escapes (`..`), is too long, repeats or collides with a directory
 * of the same name; too many entries; too deep a nesting; a size, hash, chunk
 * size or chunk count that does not add up; an unknown field.
 */

import { createHash } from 'node:crypto';

import { canonicalize } from '../skills/transport/a2a/jcs';

export const ARTIFACT_SET_KIND = 'webagents.artifact-set/v1';
export const ARTIFACT_SET_MEDIA_TYPE = 'application/vnd.webagents.artifact-set+json';

/** Transport chunking: a multiple of 256 KiB between 256 KiB and 64 MiB; 8 MiB unless the producer says otherwise. */
export const CHUNK_SIZE_UNIT = 256 * 1024;
export const CHUNK_SIZE_MIN = CHUNK_SIZE_UNIT;
export const CHUNK_SIZE_MAX = 64 * 1024 * 1024;
export const CHUNK_SIZE_DEFAULT = 8 * 1024 * 1024;

export const ARTIFACT_SET_LIMITS = {
  maxManifestBytes: 16 * 1024 * 1024,
  maxEntries: 10_000,
  maxPathBytes: 1024,
  maxPathSegments: 64,
  maxSegmentBytes: 255,
  maxDepth: 8,
  maxLocations: 16,
  maxUrlBytes: 2048,
  maxMediaTypeBytes: 255,
  maxProducerBytes: 2048,
  maxFileSize: Number.MAX_SAFE_INTEGER,
} as const;

export type LocationKind = 'https' | 'oci' | 's3' | 'robutler' | 'relay';
export const LOCATION_KINDS: readonly LocationKind[] = ['https', 'oci', 's3', 'robutler', 'relay'];

/**
 * Where bytes may be fetched from, in order of preference.
 *   https     the producer's own endpoint: `GET <url>` serves the blob by hash
 *             (`<base>/artifacts/blobs/sha256:<hex>`), ranged, with `Repr-Digest`.
 *   oci       an OCI registry reference (`ref`), the blob as an OCI artifact.
 *   s3        an S3-compatible object through a presigned `url`, short-lived.
 *   robutler  a platform content store: `GET <url>` is the blob by hash.
 *   relay     the platform fetches it over the producer's own outbound
 *             connection; `agent` names the producer on the platform.
 */
export interface ArtifactLocation {
  kind: LocationKind;
  url?: string;
  ref?: string;
  agent?: string;
}

export interface ChunkList {
  chunkSize: number;
  hashes: string[];
}

export interface ArtifactEntry {
  /** Relative, `/`-separated, normalised: no empty, `.` or `..` segments. */
  path: string;
  size: number;
  /** Lower-case hex SHA-256 of the entry's bytes. */
  sha256: string;
  mediaType: string;
  /** `file` unless the bytes are themselves a manifest. */
  kind?: 'file' | 'manifest';
  chunks?: ChunkList;
  locations?: ArtifactLocation[];
}

export interface ArtifactProducer {
  /** The producing agent's principal (its agent URL). */
  agent: string;
  /** The task the set answers, when there is one. */
  task?: string;
  name?: string;
}

export interface ArtifactSet {
  kind: typeof ARTIFACT_SET_KIND;
  createdAt: string;
  expiresAt: string | null;
  producer: ArtifactProducer;
  entries: ArtifactEntry[];
  /** Locations that apply to every entry without its own. */
  locations?: ArtifactLocation[];
}

export type ArtifactSetErrorCode =
  | 'not_json'
  | 'not_an_object'
  | 'wrong_kind'
  | 'manifest_too_large'
  | 'unknown_field'
  | 'entries_invalid'
  | 'too_many_entries'
  | 'entry_invalid'
  | 'path_invalid'
  | 'path_absolute'
  | 'path_parent_segment'
  | 'path_too_long'
  | 'path_too_many_segments'
  | 'path_duplicate'
  | 'path_conflict'
  | 'size_invalid'
  | 'sha256_invalid'
  | 'media_type_invalid'
  | 'kind_invalid'
  | 'chunk_size_invalid'
  | 'chunk_count_mismatch'
  | 'chunk_hash_invalid'
  | 'locations_invalid'
  | 'too_many_locations'
  | 'location_kind_invalid'
  | 'location_url_invalid'
  | 'created_at_invalid'
  | 'expires_at_invalid'
  | 'producer_invalid'
  | 'nesting_too_deep';

export class ArtifactSetError extends Error {
  constructor(
    public readonly code: ArtifactSetErrorCode,
    message: string,
    public readonly detail: Record<string, string | number | boolean | null> = {},
  ) {
    super(message);
    this.name = 'ArtifactSetError';
  }
}

const SHA256_HEX = /^[0-9a-f]{64}$/;
const MEDIA_TYPE = /^[A-Za-z0-9!#$&^_.+-]{1,127}\/[A-Za-z0-9!#$&^_.+-]{1,127}(?:\s*;\s*[A-Za-z0-9!#$&^_.+-]+=(?:[A-Za-z0-9!#$&^_.+-]+|"[^"\\]*"))*$/;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$/;
const TOP_FIELDS = new Set(['kind', 'createdAt', 'expiresAt', 'producer', 'entries', 'locations']);
const ENTRY_FIELDS = new Set(['path', 'size', 'sha256', 'mediaType', 'kind', 'chunks', 'locations']);
const PRODUCER_FIELDS = new Set(['agent', 'task', 'name']);
const LOCATION_FIELDS = new Set(['kind', 'url', 'ref', 'agent']);

const utf8 = new TextEncoder();
const byteLength = (s: string): number => utf8.encode(s).length;

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function refuseUnknown(record: Record<string, unknown>, allowed: Set<string>, where: string): void {
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) throw new ArtifactSetError('unknown_field', `${where} carries an unknown field ${JSON.stringify(key)}`, { where, field: key });
  }
}

export function isValidChunkSize(n: unknown): n is number {
  return typeof n === 'number' && Number.isSafeInteger(n) && n >= CHUNK_SIZE_MIN && n <= CHUNK_SIZE_MAX && n % CHUNK_SIZE_UNIT === 0;
}

/** How many chunks a file of `size` bytes has at `chunkSize`. */
export function chunkCount(size: number, chunkSize: number): number {
  return size === 0 ? 0 : Math.ceil(size / chunkSize);
}

/** Inclusive byte range of chunk `index` in a file of `size` bytes. */
export function chunkRange(index: number, size: number, chunkSize: number): { start: number; end: number } {
  const start = index * chunkSize;
  return { start, end: Math.min(size, start + chunkSize) - 1 };
}

/**
 * Validate and normalise an entry path. Refuses absolute paths, drive
 * letters, backslashes, `.` and `..` segments, empty segments, control
 * characters, and over-long paths and segments. Returns the path unchanged
 * when it is already normal (the only accepted form).
 */
export function validateEntryPath(path: unknown, index: number): string {
  const where = `entries[${index}].path`;
  if (typeof path !== 'string' || path.length === 0) throw new ArtifactSetError('path_invalid', `${where} must be a non-empty string`, { index });
  if (byteLength(path) > ARTIFACT_SET_LIMITS.maxPathBytes) {
    throw new ArtifactSetError('path_too_long', `${where} is longer than ${ARTIFACT_SET_LIMITS.maxPathBytes} bytes`, { index, limit: ARTIFACT_SET_LIMITS.maxPathBytes });
  }
  if (path.startsWith('/') || /^[A-Za-z]:/.test(path) || path.startsWith('\\')) {
    throw new ArtifactSetError('path_absolute', `${where} must be relative`, { index, path });
  }
  if (path.includes('\\')) throw new ArtifactSetError('path_invalid', `${where} must use '/' as the separator`, { index, path });
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(path)) throw new ArtifactSetError('path_invalid', `${where} carries a control character`, { index });
  const segments = path.split('/');
  if (segments.length > ARTIFACT_SET_LIMITS.maxPathSegments) {
    throw new ArtifactSetError('path_too_many_segments', `${where} has more than ${ARTIFACT_SET_LIMITS.maxPathSegments} segments`, { index, limit: ARTIFACT_SET_LIMITS.maxPathSegments });
  }
  for (const segment of segments) {
    if (segment === '..') throw new ArtifactSetError('path_parent_segment', `${where} escapes with '..'`, { index, path });
    if (segment === '' || segment === '.') throw new ArtifactSetError('path_invalid', `${where} has an empty or '.' segment`, { index, path });
    if (byteLength(segment) > ARTIFACT_SET_LIMITS.maxSegmentBytes) {
      throw new ArtifactSetError('path_too_long', `${where} has a segment longer than ${ARTIFACT_SET_LIMITS.maxSegmentBytes} bytes`, { index, limit: ARTIFACT_SET_LIMITS.maxSegmentBytes });
    }
  }
  return path;
}

export function validateChunkList(input: unknown, size: number, where: string): ChunkList {
  if (!isRecord(input)) throw new ArtifactSetError('chunk_size_invalid', `${where}.chunks must be an object`, { where });
  refuseUnknown(input, new Set(['chunkSize', 'hashes']), `${where}.chunks`);
  const { chunkSize, hashes } = input;
  if (!isValidChunkSize(chunkSize)) {
    throw new ArtifactSetError('chunk_size_invalid',
      `${where}.chunks.chunkSize must be a multiple of ${CHUNK_SIZE_UNIT} between ${CHUNK_SIZE_MIN} and ${CHUNK_SIZE_MAX}`,
      { where, min: CHUNK_SIZE_MIN, max: CHUNK_SIZE_MAX, unit: CHUNK_SIZE_UNIT, received: typeof chunkSize === 'number' ? chunkSize : String(chunkSize) });
  }
  if (!Array.isArray(hashes)) throw new ArtifactSetError('chunk_hash_invalid', `${where}.chunks.hashes must be an array`, { where, index: -1 });
  const expected = chunkCount(size, chunkSize);
  if (hashes.length !== expected) {
    throw new ArtifactSetError('chunk_count_mismatch', `${where}: a ${size}-byte file in ${chunkSize}-byte chunks has ${expected} chunks, the list has ${hashes.length}`, { where, expected, received: hashes.length });
  }
  const out: string[] = [];
  for (let i = 0; i < hashes.length; i++) {
    const h = hashes[i];
    if (typeof h !== 'string' || !SHA256_HEX.test(h)) throw new ArtifactSetError('chunk_hash_invalid', `${where}.chunks.hashes[${i}] is not a lower-case hex SHA-256`, { where, index: i });
    out.push(h);
  }
  return { chunkSize, hashes: out };
}

export interface ReadOptions {
  /** How many manifests deep the caller already is (0 for a root). */
  depth?: number;
  /**
   * Accept plaintext `http` locations beyond loopback (a private network in
   * development). Off by default: a manifest from the open internet names
   * `https` locations or none.
   */
  allowHttp?: boolean;
}

export function validateLocations(input: unknown, where: string, options: { allowHttp?: boolean } = {}): ArtifactLocation[] {
  if (!Array.isArray(input)) throw new ArtifactSetError('locations_invalid', `${where} must be an array`, { where });
  if (input.length > ARTIFACT_SET_LIMITS.maxLocations) {
    throw new ArtifactSetError('too_many_locations', `${where} has more than ${ARTIFACT_SET_LIMITS.maxLocations} locations`, { where, limit: ARTIFACT_SET_LIMITS.maxLocations });
  }
  const out: ArtifactLocation[] = [];
  input.forEach((raw, i) => {
    const w = `${where}[${i}]`;
    if (!isRecord(raw)) throw new ArtifactSetError('locations_invalid', `${w} must be an object`, { where: w });
    refuseUnknown(raw, LOCATION_FIELDS, w);
    const kind = raw.kind;
    if (typeof kind !== 'string' || !(LOCATION_KINDS as readonly string[]).includes(kind)) {
      throw new ArtifactSetError('location_kind_invalid', `${w}.kind must be one of ${LOCATION_KINDS.join(', ')}`, { where: w, received: String(kind) });
    }
    const location: ArtifactLocation = { kind: kind as LocationKind };
    const url = raw.url;
    if (url !== undefined) {
      if (typeof url !== 'string' || byteLength(url) > ARTIFACT_SET_LIMITS.maxUrlBytes) throw new ArtifactSetError('location_url_invalid', `${w}.url must be a string of at most ${ARTIFACT_SET_LIMITS.maxUrlBytes} bytes`, { where: w });
      let parsed: URL;
      try { parsed = new URL(url); } catch { throw new ArtifactSetError('location_url_invalid', `${w}.url is not a URL`, { where: w }); }
      if ((kind === 'https' || kind === 'robutler') && parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && (options.allowHttp || isLoopback(parsed.hostname)))) {
        throw new ArtifactSetError('location_url_invalid', `${w}.url must be https`, { where: w, received: parsed.protocol });
      }
      if (kind === 's3' && parsed.protocol !== 'https:' && !(options.allowHttp && parsed.protocol === 'http:')) throw new ArtifactSetError('location_url_invalid', `${w}.url must be https`, { where: w, received: parsed.protocol });
      location.url = url;
    }
    if (raw.ref !== undefined) {
      if (typeof raw.ref !== 'string' || raw.ref.length === 0 || byteLength(raw.ref) > ARTIFACT_SET_LIMITS.maxUrlBytes) throw new ArtifactSetError('location_url_invalid', `${w}.ref must be a non-empty string`, { where: w });
      location.ref = raw.ref;
    }
    if (raw.agent !== undefined) {
      if (typeof raw.agent !== 'string' || raw.agent.length === 0 || byteLength(raw.agent) > ARTIFACT_SET_LIMITS.maxUrlBytes) throw new ArtifactSetError('location_url_invalid', `${w}.agent must be a non-empty string`, { where: w });
      location.agent = raw.agent;
    }
    if ((kind === 'https' || kind === 's3' || kind === 'robutler') && !location.url) throw new ArtifactSetError('location_url_invalid', `${w} (${kind}) needs a url`, { where: w, kind });
    if (kind === 'oci' && !location.ref) throw new ArtifactSetError('location_url_invalid', `${w} (oci) needs a ref`, { where: w, kind });
    out.push(location);
  });
  return out;
}

function isLoopback(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]' || hostname === '::1';
}

function validateEntry(raw: unknown, index: number, options: { allowHttp?: boolean }): ArtifactEntry {
  const where = `entries[${index}]`;
  if (!isRecord(raw)) throw new ArtifactSetError('entry_invalid', `${where} must be an object`, { index });
  refuseUnknown(raw, ENTRY_FIELDS, where);
  const path = validateEntryPath(raw.path, index);
  const size = raw.size;
  if (typeof size !== 'number' || !Number.isSafeInteger(size) || size < 0 || size > ARTIFACT_SET_LIMITS.maxFileSize) {
    throw new ArtifactSetError('size_invalid', `${where}.size must be a non-negative integer`, { index });
  }
  if (typeof raw.sha256 !== 'string' || !SHA256_HEX.test(raw.sha256)) throw new ArtifactSetError('sha256_invalid', `${where}.sha256 must be a lower-case hex SHA-256`, { index });
  if (typeof raw.mediaType !== 'string' || byteLength(raw.mediaType) > ARTIFACT_SET_LIMITS.maxMediaTypeBytes || !MEDIA_TYPE.test(raw.mediaType)) {
    throw new ArtifactSetError('media_type_invalid', `${where}.mediaType must be a media type`, { index });
  }
  const entry: ArtifactEntry = { path, size, sha256: raw.sha256, mediaType: raw.mediaType };
  if (raw.kind !== undefined) {
    if (raw.kind !== 'file' && raw.kind !== 'manifest') throw new ArtifactSetError('kind_invalid', `${where}.kind must be "file" or "manifest"`, { index });
    entry.kind = raw.kind;
  }
  if (raw.chunks !== undefined) entry.chunks = validateChunkList(raw.chunks, size, where);
  if (raw.locations !== undefined) entry.locations = validateLocations(raw.locations, `${where}.locations`, options);
  return entry;
}

function validateProducer(raw: unknown): ArtifactProducer {
  if (!isRecord(raw)) throw new ArtifactSetError('producer_invalid', 'producer must be an object', {});
  refuseUnknown(raw, PRODUCER_FIELDS, 'producer');
  if (typeof raw.agent !== 'string' || raw.agent.length === 0 || byteLength(raw.agent) > ARTIFACT_SET_LIMITS.maxProducerBytes) {
    throw new ArtifactSetError('producer_invalid', 'producer.agent must be a non-empty string', {});
  }
  const producer: ArtifactProducer = { agent: raw.agent };
  for (const key of ['task', 'name'] as const) {
    const v = raw[key];
    if (v === undefined) continue;
    if (typeof v !== 'string' || byteLength(v) > ARTIFACT_SET_LIMITS.maxProducerBytes) throw new ArtifactSetError('producer_invalid', `producer.${key} must be a string`, { field: key });
    producer[key] = v;
  }
  return producer;
}

/**
 * Parse and validate a manifest from its object form, JSON text or bytes.
 * `depth` is how many manifests deep the caller already is (0 for a root);
 * past `maxDepth` the manifest is refused as `nesting_too_deep`.
 */
export function parseArtifactSet(input: unknown, options: ReadOptions = {}): ArtifactSet {
  const depth = options.depth ?? 0;
  if (depth > ARTIFACT_SET_LIMITS.maxDepth) {
    throw new ArtifactSetError('nesting_too_deep', `manifests nest deeper than ${ARTIFACT_SET_LIMITS.maxDepth}`, { limit: ARTIFACT_SET_LIMITS.maxDepth, depth });
  }
  let value: unknown = input;
  if (input instanceof Uint8Array || typeof input === 'string') {
    const bytes = typeof input === 'string' ? utf8.encode(input) : input;
    if (bytes.length > ARTIFACT_SET_LIMITS.maxManifestBytes) {
      throw new ArtifactSetError('manifest_too_large', `the manifest is larger than ${ARTIFACT_SET_LIMITS.maxManifestBytes} bytes`, { limit: ARTIFACT_SET_LIMITS.maxManifestBytes, received: bytes.length });
    }
    try {
      value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    } catch {
      throw new ArtifactSetError('not_json', 'the manifest is not UTF-8 JSON', {});
    }
  }
  if (!isRecord(value)) throw new ArtifactSetError('not_an_object', 'the manifest must be a JSON object', {});
  if (value.kind !== ARTIFACT_SET_KIND) throw new ArtifactSetError('wrong_kind', `kind must be ${ARTIFACT_SET_KIND}`, { received: String(value.kind) });
  refuseUnknown(value, TOP_FIELDS, 'the manifest');
  if (typeof value.createdAt !== 'string' || !ISO_DATE.test(value.createdAt) || Number.isNaN(Date.parse(value.createdAt))) {
    throw new ArtifactSetError('created_at_invalid', 'createdAt must be an ISO 8601 UTC timestamp ending in Z', {});
  }
  if (value.expiresAt !== null && (typeof value.expiresAt !== 'string' || !ISO_DATE.test(value.expiresAt) || Number.isNaN(Date.parse(value.expiresAt)))) {
    throw new ArtifactSetError('expires_at_invalid', 'expiresAt must be null or an ISO 8601 UTC timestamp ending in Z', {});
  }
  const producer = validateProducer(value.producer);
  if (!Array.isArray(value.entries)) throw new ArtifactSetError('entries_invalid', 'entries must be an array', {});
  if (value.entries.length > ARTIFACT_SET_LIMITS.maxEntries) {
    throw new ArtifactSetError('too_many_entries', `more than ${ARTIFACT_SET_LIMITS.maxEntries} entries`, { limit: ARTIFACT_SET_LIMITS.maxEntries, received: value.entries.length });
  }
  const entries = value.entries.map((raw, i) => validateEntry(raw, i, options));
  const seen = new Set<string>();
  const dirs = new Set<string>();
  entries.forEach((entry, i) => {
    if (seen.has(entry.path)) throw new ArtifactSetError('path_duplicate', `entries[${i}].path repeats ${JSON.stringify(entry.path)}`, { index: i, path: entry.path });
    seen.add(entry.path);
    const segments = entry.path.split('/');
    for (let d = 1; d < segments.length; d++) dirs.add(segments.slice(0, d).join('/'));
  });
  for (const p of seen) {
    if (dirs.has(p)) throw new ArtifactSetError('path_conflict', `${JSON.stringify(p)} is both a file and a directory`, { path: p });
  }
  const set: ArtifactSet = {
    kind: ARTIFACT_SET_KIND,
    createdAt: value.createdAt,
    expiresAt: value.expiresAt as string | null,
    producer,
    entries,
  };
  if (value.locations !== undefined) set.locations = validateLocations(value.locations, 'locations', options);
  return set;
}

/** The canonical (RFC 8785) text of a manifest: what its identity hashes. */
export function canonicalManifest(set: ArtifactSet): string {
  return canonicalize(set);
}

/** The canonical bytes. */
export function canonicalManifestBytes(set: ArtifactSet): Uint8Array {
  return utf8.encode(canonicalManifest(set));
}

/** Lower-case hex SHA-256 of the canonical bytes: the manifest's identity. */
export function manifestHash(set: ArtifactSet): string {
  return createHash('sha256').update(canonicalManifestBytes(set)).digest('hex');
}

/** Parse bytes received as a manifest and name them: `{ set, sha256 }`. */
export function readArtifactSet(bytes: Uint8Array, options: ReadOptions = {}): { set: ArtifactSet; sha256: string } {
  const set = parseArtifactSet(bytes, options);
  return { set, sha256: createHash('sha256').update(bytes).digest('hex') };
}

/** The `sha256:<hex>` digest spelling used in URLs, and its parse. */
export function digestName(sha256: string): string {
  return `sha256:${sha256}`;
}

export function parseDigestName(name: unknown): string | null {
  if (typeof name !== 'string') return null;
  const m = /^sha256:([0-9a-f]{64})$/.exec(name);
  return m ? m[1] : null;
}
