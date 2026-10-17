/**
 * Fetching a set's files from its locations: ranged, parallel, resumable,
 * every chunk and every file verified.
 *
 * For each entry the client takes the locations in order (the entry's own,
 * else the set's). Version 1 fetches `https` locations; every other kind is
 * refused by name as unsupported by this client, and the next location is
 * tried. A file with a chunk list is fetched one range per chunk, each chunk
 * hashed as it streams into the file at its offset and refused on a mismatch;
 * a file without one is fetched in fixed ranges and verified by the whole-file
 * hash at the end, which every file gets. Progress lives beside the file
 * (`<name>.part` and `<name>.part.json`), so a cut transfer resumes with the
 * chunks it already verified and re-fetches only the rest. Requests are signed
 * with the fetching agent's identity when one is given, which is what a
 * producer endpoint authorises on.
 *
 * Memory is one range buffer per worker: bytes stream from the response into
 * the file and are never gathered.
 */

import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';

import type { SigningIdentity } from '../crypto/http-signature';
import { signRequest } from '../crypto/http-signature';
import { ARTIFACT_TASK_HEADER } from './skill';
import { parseContentRange, parseReprDigest } from './http-range';
import {
  ARTIFACT_SET_LIMITS,
  CHUNK_SIZE_DEFAULT,
  chunkCount,
  chunkRange,
  parseArtifactSet,
  type ArtifactEntry,
  type ArtifactLocation,
  type ArtifactSet,
  type LocationKind,
} from './manifest';

export type ArtifactFetchErrorCode =
  | 'location_kind_unsupported'
  | 'no_reachable_location'
  | 'bad_response'
  | 'chunk_hash_mismatch'
  | 'file_hash_mismatch'
  | 'size_mismatch'
  | 'manifest_too_large'
  | 'nesting_too_deep'
  | 'path_escape'
  | 'aborted';

export class ArtifactFetchError extends Error {
  constructor(
    public readonly code: ArtifactFetchErrorCode,
    message: string,
    public readonly detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = 'ArtifactFetchError';
  }
}

export interface FetchArtifactOptions {
  /** The folder the entries are written into, by their paths. */
  dest: string;
  /** Signs every request; required by a producer that serves verified callers only. */
  identity?: SigningIdentity;
  fetch?: typeof fetch;
  /** Parallel ranges per file. Default 4. */
  concurrency?: number;
  /** Range size for a file without a chunk list. Default 8 MiB. */
  rangeSize?: number;
  /** Allow `http://` loopback locations and signing to them (tests). */
  allowHttp?: boolean;
  /** Sent as `Artifact-Task` when the set was published for a task. */
  task?: string;
  /** Location kinds this fetch may use. Default `['https']`. */
  locationKinds?: LocationKind[];
  onProgress?: (progress: { path: string; bytes: number; size: number; resumed: boolean }) => void;
  signal?: AbortSignal;
  /** Nesting depth of the set being fetched; callers leave it alone. */
  depth?: number;
}

export interface FetchedFile {
  path: string;
  absolutePath: string;
  size: number;
  sha256: string;
  /** Bytes fetched this run (less than `size` after a resume). */
  bytesFetched: number;
}

export interface FetchResult {
  files: FetchedFile[];
  bytesFetched: number;
}

interface PartState {
  sha256: string;
  size: number;
  pieceSize: number;
  done: number[];
}


/**
 * Fetch one manifest by URL (or confirm one passed by hash) and return it
 * parsed with its hash.
 */
export async function fetchManifest(url: string, options: Omit<FetchArtifactOptions, 'dest'> & { expectedSha256?: string } = {}): Promise<{ set: ArtifactSet; sha256: string; bytes: Uint8Array }> {
  const res = await signedGet(url, options, {});
  if (!res.ok) throw new ArtifactFetchError('bad_response', `${url} answered ${res.status}`, { status: res.status, url });
  const length = Number(res.headers.get('content-length') ?? '0');
  if (length > ARTIFACT_SET_LIMITS.maxManifestBytes) throw new ArtifactFetchError('manifest_too_large', `${url} declares ${length} bytes, a manifest is at most ${ARTIFACT_SET_LIMITS.maxManifestBytes}`, { url, length });
  const bytes = await readBounded(res, ARTIFACT_SET_LIMITS.maxManifestBytes, url);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  if (options.expectedSha256 && options.expectedSha256 !== sha256) {
    throw new ArtifactFetchError('file_hash_mismatch', `the manifest at ${url} hashes to ${sha256}, not ${options.expectedSha256}`, { url, expected: options.expectedSha256, actual: sha256 });
  }
  const declared = parseReprDigest(res.headers.get('repr-digest'));
  if (declared && declared !== sha256) throw new ArtifactFetchError('bad_response', `${url}: Repr-Digest names ${declared}, the bytes hash to ${sha256}`, { url });
  const set = parseArtifactSet(bytes, { depth: options.depth ?? 0, allowHttp: options.allowHttp });
  return { set, sha256, bytes };
}

/** Fetch every entry of `set` into `options.dest`. */
export async function fetchArtifactSet(set: ArtifactSet, options: FetchArtifactOptions): Promise<FetchResult> {
  const depth = options.depth ?? 0;
  if (depth > ARTIFACT_SET_LIMITS.maxDepth) throw new ArtifactFetchError('nesting_too_deep', `manifests nest deeper than ${ARTIFACT_SET_LIMITS.maxDepth}`, { depth });
  const files: FetchedFile[] = [];
  let bytesFetched = 0;
  for (const entry of set.entries) {
    if (entry.kind === 'manifest') {
      const nested = await fetchNested(set, entry, options, depth);
      files.push(nested.manifest);
      bytesFetched += nested.manifest.bytesFetched;
      files.push(...nested.result.files);
      bytesFetched += nested.result.bytesFetched;
      continue;
    }
    const file = await fetchEntry(set, entry, options);
    files.push(file);
    bytesFetched += file.bytesFetched;
  }
  return { files, bytesFetched };
}

async function fetchNested(set: ArtifactSet, entry: ArtifactEntry, options: FetchArtifactOptions, depth: number) {
  if (entry.size > ARTIFACT_SET_LIMITS.maxManifestBytes) throw new ArtifactFetchError('manifest_too_large', `${entry.path} declares ${entry.size} bytes for a manifest`, { path: entry.path, size: entry.size });
  const manifest = await fetchEntry(set, entry, options);
  const bytes = new Uint8Array(await readFile(manifest.absolutePath));
  const nestedSet = parseArtifactSet(bytes, { depth: depth + 1, allowHttp: options.allowHttp });
  const nestedDest = join(options.dest, dirname(entry.path));
  const result = await fetchArtifactSet(nestedSet, { ...options, dest: nestedDest, depth: depth + 1 });
  return { manifest, result };
}

function targetFor(dest: string, path: string): string {
  const root = resolve(dest);
  const target = resolve(root, path);
  if (target !== root && !target.startsWith(root + sep)) {
    throw new ArtifactFetchError('path_escape', `${path} resolves outside the destination`, { path });
  }
  return target;
}

async function loadState(statePath: string, want: Omit<PartState, 'done'>): Promise<PartState | null> {
  try {
    const raw = JSON.parse(await readFile(statePath, 'utf8')) as PartState;
    if (raw && raw.sha256 === want.sha256 && raw.size === want.size && raw.pieceSize === want.pieceSize && Array.isArray(raw.done)) {
      return { ...want, done: raw.done.filter((n) => Number.isSafeInteger(n) && n >= 0) };
    }
  } catch {
    /* no usable state */
  }
  return null;
}

/** Fetch one entry; resumes a `.part` left by an earlier run. */
export async function fetchEntry(set: ArtifactSet, entry: ArtifactEntry, options: FetchArtifactOptions): Promise<FetchedFile> {
  const target = targetFor(options.dest, entry.path);
  await mkdir(dirname(target), { recursive: true });
  const partPath = `${target}.part`;
  const statePath = `${target}.part.json`;
  const pieceSize = entry.chunks?.chunkSize ?? options.rangeSize ?? CHUNK_SIZE_DEFAULT;
  const pieces = entry.size === 0 ? 0 : chunkCount(entry.size, pieceSize);
  const want = { sha256: entry.sha256, size: entry.size, pieceSize };
  let state = await loadState(statePath, want);
  let resumed = false;
  if (state) {
    // The part on disk has to be the length the state says it reached.
    try {
      const st = await stat(partPath);
      if (st.size !== entry.size) state = null;
    } catch {
      state = null;
    }
    resumed = state !== null;
  }
  if (!state) state = { ...want, done: [] };
  const done = new Set(state.done);
  const pending = Array.from({ length: pieces }, (_, i) => i).filter((i) => !done.has(i));

  const fh = await open(partPath, resumed ? 'r+' : 'w+');
  let bytesFetched = 0;
  try {
    if (!resumed && entry.size > 0) await fh.truncate(entry.size);
    const locations = candidateLocations(set, entry);
    const concurrency = Math.max(1, options.concurrency ?? 4);
    let next = 0;
    let failure: unknown = null;
    const persist = async () => writeFile(statePath, JSON.stringify({ ...want, done: Array.from(done).sort((a, b) => a - b) }));
    const worker = async () => {
      while (next < pending.length && !failure) {
        const index = pending[next++];
        if (options.signal?.aborted) throw new ArtifactFetchError('aborted', 'fetch aborted', {});
        const { start, end } = chunkRange(index, entry.size, pieceSize);
        const expected = entry.chunks?.hashes[index];
        const got = await fetchRange(locations, entry, start, end, expected, fh, options);
        bytesFetched += got;
        done.add(index);
        await persist();
        options.onProgress?.({ path: entry.path, bytes: done.size === pieces ? entry.size : Math.min(entry.size, done.size * pieceSize), size: entry.size, resumed });
      }
    };
    await Promise.all(Array.from({ length: Math.min(concurrency, pending.length || 1) }, () => worker().catch((err) => { failure ??= err; })));
    if (failure) throw failure;
  } finally {
    await fh.close();
  }

  // The whole-file hash decides, chunk list or not.
  const actual = await hashPath(partPath);
  if (actual !== entry.sha256) {
    await rm(partPath, { force: true });
    await rm(statePath, { force: true });
    throw new ArtifactFetchError('file_hash_mismatch', `${entry.path} hashed to ${actual}, the manifest says ${entry.sha256}`, { path: entry.path, expected: entry.sha256, actual });
  }
  await rename(partPath, target);
  await rm(statePath, { force: true });
  return { path: entry.path, absolutePath: target, size: entry.size, sha256: entry.sha256, bytesFetched };
}

function candidateLocations(set: ArtifactSet, entry: ArtifactEntry): ArtifactLocation[] {
  return entry.locations && entry.locations.length ? entry.locations : set.locations ?? [];
}

async function signedGet(url: string, options: Omit<FetchArtifactOptions, 'dest'>, headers: Record<string, string>): Promise<Response> {
  const doFetch = options.fetch ?? fetch;
  const h = new Headers(headers);
  if (options.task) h.set(ARTIFACT_TASK_HEADER, options.task);
  let request = new Request(url, { method: 'GET', headers: h, signal: options.signal });
  if (options.identity) request = await signRequest(options.identity, request, { allowHttp: options.allowHttp });
  return doFetch(request);
}

async function readBounded(res: Response, max: number, url: string): Promise<Uint8Array> {
  if (!res.body) return new Uint8Array();
  const parts: Uint8Array[] = [];
  let total = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > max) {
      await reader.cancel().catch(() => {});
      throw new ArtifactFetchError('manifest_too_large', `${url} sent more than ${max} bytes for a manifest`, { url, limit: max });
    }
    parts.push(value);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.byteLength; }
  return out;
}

/**
 * Fetch `[start, end]` of `entry` from the first location that serves it,
 * streaming into the file at `start`. Returns the bytes written. Refusals
 * are collected per location and thrown together when none serves.
 */
async function fetchRange(
  locations: ArtifactLocation[],
  entry: ArtifactEntry,
  start: number,
  end: number,
  expectedSha256: string | undefined,
  fh: import('node:fs/promises').FileHandle,
  options: FetchArtifactOptions,
): Promise<number> {
  const allowed = new Set<LocationKind>(options.locationKinds ?? ['https']);
  const refusals: Array<{ location: ArtifactLocation; code: string; message: string }> = [];
  for (const location of locations) {
    if (!allowed.has(location.kind) || location.kind !== 'https' || !location.url) {
      refusals.push({ location, code: 'location_kind_unsupported', message: `location kind ${location.kind} is not supported by this client` });
      continue;
    }
    if (!options.allowHttp && !location.url.startsWith('https://')) {
      refusals.push({ location, code: 'location_kind_unsupported', message: 'only https locations are fetched' });
      continue;
    }
    try {
      return await fetchRangeFrom(location.url, entry, start, end, expectedSha256, fh, options);
    } catch (err) {
      if (err instanceof ArtifactFetchError && err.code === 'aborted') throw err;
      refusals.push({ location, code: err instanceof ArtifactFetchError ? err.code : 'bad_response', message: err instanceof Error ? err.message : String(err) });
    }
  }
  throw new ArtifactFetchError('no_reachable_location', `no location served ${entry.path} bytes ${start}-${end}: ${refusals.map((r) => `${r.location.kind}: ${r.code}`).join('; ')}`, { path: entry.path, start, end, refusals });
}

async function fetchRangeFrom(
  url: string,
  entry: ArtifactEntry,
  start: number,
  end: number,
  expectedSha256: string | undefined,
  fh: import('node:fs/promises').FileHandle,
  options: FetchArtifactOptions,
): Promise<number> {
  const wantLength = end - start + 1;
  const whole = start === 0 && end === entry.size - 1;
  const res = await signedGet(url, options, { Range: `bytes=${start}-${end}` });
  if (res.status === 416 || res.status === 404 || res.status === 401 || res.status === 403 || res.status >= 500) {
    throw new ArtifactFetchError('bad_response', `${url} answered ${res.status}`, { status: res.status });
  }
  let bodyStart = start;
  let bodyLength = wantLength;
  if (res.status === 206) {
    const cr = parseContentRange(res.headers.get('content-range'));
    if (!cr || cr.start !== start || cr.end !== end || (cr.size !== -1 && cr.size !== entry.size)) {
      throw new ArtifactFetchError('bad_response', `${url} answered a range other than ${start}-${end}/${entry.size}: ${res.headers.get('content-range')}`, {});
    }
  } else if (res.status === 200) {
    // The server ignored the range: it sends the whole file. Take it only when
    // that is what was asked for; otherwise the stream would be the wrong bytes.
    if (!whole) {
      await res.body?.cancel().catch(() => {});
      throw new ArtifactFetchError('bad_response', `${url} does not serve ranges`, {});
    }
    bodyStart = 0;
    bodyLength = entry.size;
  } else {
    throw new ArtifactFetchError('bad_response', `${url} answered ${res.status}`, { status: res.status });
  }
  const declaredDigest = parseReprDigest(res.headers.get('repr-digest'));
  if (declaredDigest && declaredDigest !== entry.sha256) {
    await res.body?.cancel().catch(() => {});
    throw new ArtifactFetchError('bad_response', `${url}: Repr-Digest names ${declaredDigest}, the manifest says ${entry.sha256}`, {});
  }
  if (!res.body) throw new ArtifactFetchError('bad_response', `${url} sent no body`, {});

  const hasher = expectedSha256 ? createHash('sha256') : null;
  const reader = res.body.getReader();
  let written = 0;
  let position = bodyStart;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (written + value.byteLength > bodyLength) {
        throw new ArtifactFetchError('size_mismatch', `${url} sent more than the ${bodyLength} bytes asked for`, { url });
      }
      hasher?.update(value);
      await fh.write(value, 0, value.byteLength, position);
      position += value.byteLength;
      written += value.byteLength;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  if (written !== bodyLength) {
    throw new ArtifactFetchError('size_mismatch', `${url} sent ${written} bytes of the ${bodyLength} asked for`, { url, expected: bodyLength, actual: written });
  }
  if (hasher) {
    const actual = hasher.digest('hex');
    if (actual !== expectedSha256) {
      throw new ArtifactFetchError('chunk_hash_mismatch', `${entry.path} bytes ${start}-${end} hashed to ${actual}, the chunk list says ${expectedSha256}`, { path: entry.path, start, end, expected: expectedSha256, actual });
    }
  }
  return written;
}

async function hashPath(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path, { highWaterMark: 1024 * 1024 })) hash.update(chunk as Buffer);
  return hash.digest('hex');
}
