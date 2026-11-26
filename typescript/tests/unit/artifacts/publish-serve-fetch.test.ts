/**
 * Publish a folder, serve it from the producer endpoint, fetch it back with
 * the client: ranges, digests, authorisation by caller, resume after a cut,
 * a wrong chunk refused and not kept, memory never holding a file whole.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync, readdirSync, existsSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { randomBytes, createHash } from 'node:crypto';

import {
  ArtifactRegistry,
  ArtifactsSkill,
  ArtifactFetchError,
  fetchArtifactSet,
  fetchManifest,
  files,
  hashFile,
  hashStream,
  readArtifactSet,
  reprDigest,
  ARTIFACT_TASK_HEADER,
  canonicalManifestBytes,
  type ArtifactSet,
} from '../../../src/artifacts';
import type { Context } from '../../../src/core/types';

const CHUNK = 256 * 1024;
const BASE = 'https://producer.example/agents/flow';
const CALLER = 'agent:https://consumer.example/agents/reda';

let root: string;
let folder: string;
let registry: ArtifactRegistry;
let skill: ArtifactsSkill;
let published: Awaited<ReturnType<typeof files.publish>>;
const bytes: Record<string, Buffer> = {};

function requireStateTemps(directory: string): string[] {
  return readdirSync(directory).filter((name) => name.includes('.part.json.') && name.endsWith('.tmp'));
}

function contextFor(principals: string[] | null): Context {
  return {
    auth: principals ? { authenticated: true, principals } : { authenticated: false },
    metadata: {},
    get: () => undefined,
    set: () => undefined,
  } as unknown as Context;
}

/** A fetch that answers from the skill as the given caller. */
function fetchAs(principals: string[] | null, options: { corrupt?: (url: string, body: Uint8Array) => Uint8Array; failOnce?: (url: string, range: string | null) => boolean } = {}) {
  let failed = false;
  return async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : new Request(input, init);
    if (options.failOnce && !failed && options.failOnce(request.url, request.headers.get('range'))) {
      failed = true;
      throw new Error('connection reset');
    }
    const res = request.method === 'HEAD' ? await skill.headBlob(request, contextFor(principals)) : await skill.getBlob(request, contextFor(principals));
    if (!options.corrupt || !res.body) return res;
    const body = new Uint8Array(await res.arrayBuffer());
    const headers = new Headers(res.headers);
    return new Response(options.corrupt(request.url, body) as unknown as BodyInit, { status: res.status, headers });
  };
}

beforeAll(async () => {
  root = mkdtempSync(path.join(tmpdir(), 'artifacts-publish-'));
  folder = path.join(root, 'out');
  mkdirSync(path.join(folder, 'gds'), { recursive: true });
  mkdirSync(path.join(folder, 'reports'), { recursive: true });
  bytes['gds/top.gds'] = randomBytes(3 * CHUNK + 1234); // 4 chunks
  bytes['reports/drc.txt'] = Buffer.from('clean\n');
  bytes['reports/empty.log'] = Buffer.alloc(0);
  bytes['summary.json'] = Buffer.from(JSON.stringify({ ok: true }));
  for (const [rel, data] of Object.entries(bytes)) writeFileSync(path.join(folder, rel), data);
  symlinkSync('/etc/hostname', path.join(folder, 'escape'));

  registry = new ArtifactRegistry();
  skill = new ArtifactsSkill({ registry, maxConcurrentTransfers: 8 });
  published = await files.publish(folder, {
    producer: { agent: BASE, task: 'task-42' },
    registry,
    baseUrl: BASE,
    chunkSize: CHUNK,
    audience: ['https://consumer.example/agents/reda'],
    createdAt: new Date('2026-10-17T10:00:00Z'),
  });
});
afterAll(() => rmSync(root, { recursive: true, force: true }));

describe('files.publish', () => {
  it('writes a manifest with chunk lists above the chunk size, https locations, and refuses the symlink', () => {
    expect(published.skipped).toEqual([{ path: 'escape', reason: 'symlink' }]);
    expect(published.set.entries.map((e) => e.path)).toEqual(['gds/top.gds', 'reports/drc.txt', 'reports/empty.log', 'summary.json']);
    const gds = published.set.entries[0];
    expect(gds.size).toBe(bytes['gds/top.gds'].length);
    expect(gds.sha256).toBe(createHash('sha256').update(bytes['gds/top.gds']).digest('hex'));
    expect(gds.chunks?.chunkSize).toBe(CHUNK);
    expect(gds.chunks?.hashes).toHaveLength(4);
    expect(gds.chunks?.hashes[1]).toBe(createHash('sha256').update(bytes['gds/top.gds'].subarray(CHUNK, 2 * CHUNK)).digest('hex'));
    expect(gds.locations?.[0]).toEqual({ kind: 'https', url: `${BASE}/artifacts/blobs/sha256:${gds.sha256}` });
    expect(published.set.entries[1].chunks).toBeUndefined();
    expect(published.set.entries[1].mediaType).toBe('text/plain');
    expect(published.set.producer).toEqual({ agent: BASE, task: 'task-42' });
    expect(published.manifestUrl).toBe(`${BASE}/artifacts/blobs/sha256:${published.sha256}`);
    // The registry knows the blobs and the manifest; the audience is the consumer.
    expect(registry.getBlob(gds.sha256)?.path).toBe(path.join(folder, 'gds/top.gds'));
    expect(registry.getManifest(published.sha256)?.task).toBe('task-42');
    expect(registry.access(gds.sha256, [CALLER], 'task-42')).toEqual({ ok: true });
    expect(registry.access(gds.sha256, ['agent:https://other.example/agents/x'], 'task-42')).toEqual({ ok: false, reason: 'forbidden' });
    expect(registry.access(gds.sha256, [CALLER], null)).toEqual({ ok: false, reason: 'task_required' });
    expect(registry.access(gds.sha256, [CALLER], 'task-1')).toEqual({ ok: false, reason: 'task_mismatch' });
    expect(registry.access('00'.repeat(32), [CALLER], 'task-42')).toEqual({ ok: false, reason: 'unknown' });
  });

  it('hashes a stream as the file is hashed', async () => {
    const fromFile = await hashFile(path.join(folder, 'gds/top.gds'), CHUNK);
    const fromStream = await hashStream((async function* () { yield bytes['gds/top.gds']; })(), CHUNK);
    expect(fromStream).toEqual(fromFile);
    const small = await hashStream((async function* () { yield Buffer.from('x'); })(), CHUNK);
    expect(small.chunks).toBeUndefined();
  });
});

describe('the producer endpoint', () => {
  const url = (sha: string) => `${BASE}/artifacts/blobs/sha256:${sha}`;
  const req = (sha: string, headers: Record<string, string> = {}, method = 'GET') =>
    new Request(url(sha), { method, headers: { [ARTIFACT_TASK_HEADER]: 'task-42', ...headers } });

  it('serves the manifest by its hash with its digest, and 200 / 206 / 416 for a blob', async () => {
    const m = await skill.getBlob(req(published.sha256), contextFor([CALLER]));
    expect(m.status).toBe(200);
    expect(m.headers.get('content-type')).toBe('application/vnd.webagents.artifact-set+json');
    expect(m.headers.get('repr-digest')).toBe(reprDigest(published.sha256));
    const text = new Uint8Array(await m.arrayBuffer());
    expect(readArtifactSet(text).sha256).toBe(published.sha256);

    const gds = published.set.entries[0];
    const whole = await skill.getBlob(req(gds.sha256), contextFor([CALLER]));
    expect(whole.status).toBe(200);
    expect(whole.headers.get('accept-ranges')).toBe('bytes');
    expect(whole.headers.get('content-length')).toBe(String(gds.size));
    expect(whole.headers.get('repr-digest')).toBe(reprDigest(gds.sha256));
    expect(Buffer.from(await whole.arrayBuffer()).equals(bytes['gds/top.gds'])).toBe(true);

    const part = await skill.getBlob(req(gds.sha256, { Range: `bytes=${CHUNK}-${2 * CHUNK - 1}` }), contextFor([CALLER]));
    expect(part.status).toBe(206);
    expect(part.headers.get('content-range')).toBe(`bytes ${CHUNK}-${2 * CHUNK - 1}/${gds.size}`);
    expect(part.headers.get('repr-digest')).toBe(reprDigest(gds.sha256));
    expect(Buffer.from(await part.arrayBuffer()).equals(bytes['gds/top.gds'].subarray(CHUNK, 2 * CHUNK))).toBe(true);

    const past = await skill.getBlob(req(gds.sha256, { Range: `bytes=${gds.size}-` }), contextFor([CALLER]));
    expect(past.status).toBe(416);
    expect(past.headers.get('content-range')).toBe(`bytes */${gds.size}`);

    const head = await skill.headBlob(req(gds.sha256, { Range: 'bytes=0-9' }, 'HEAD'), contextFor([CALLER]));
    expect(head.status).toBe(206);
    expect(head.headers.get('content-length')).toBe('10');
    expect(head.body).toBeNull();

    const empty = await skill.getBlob(req(published.set.entries[2].sha256), contextFor([CALLER]));
    expect(empty.status).toBe(200);
    expect(empty.headers.get('content-length')).toBe('0');
  });

  it('a link alone grants nothing: anonymous 401, a stranger 404, a wrong task 403, an unknown hash 404', async () => {
    const gds = published.set.entries[0];
    expect((await skill.getBlob(req(gds.sha256), contextFor(null))).status).toBe(401);
    expect((await skill.getBlob(req(gds.sha256), contextFor(['agent:https://other.example/agents/x']))).status).toBe(404);
    const noTask = await skill.getBlob(new Request(url(gds.sha256)), contextFor([CALLER]));
    expect(noTask.status).toBe(403);
    expect((await noTask.json()).error.code).toBe('task_required');
    const wrongTask = await skill.getBlob(req(gds.sha256, { [ARTIFACT_TASK_HEADER]: 'other' }), contextFor([CALLER]));
    expect((await wrongTask.json()).error.code).toBe('task_mismatch');
    expect((await skill.getBlob(req('00'.repeat(32)), contextFor([CALLER]))).status).toBe(404);
    expect((await skill.getBlob(new Request(`${BASE}/artifacts/blobs/md5:abc`), contextFor([CALLER]))).status).toBe(404);
  });

  it('refuses past its transfer cap by name', async () => {
    const tight = new ArtifactsSkill({ registry, maxConcurrentTransfers: 1 });
    const gds = published.set.entries[0];
    const first = await tight.getBlob(req(gds.sha256), contextFor([CALLER]));
    expect(first.status).toBe(200);
    const second = await tight.getBlob(req(gds.sha256), contextFor([CALLER]));
    expect(second.status).toBe(503);
    expect((await second.json()).error.code).toBe('too_many_transfers');
    await first.arrayBuffer(); // drains and releases
    await new Promise((r) => setTimeout(r, 20));
    expect(tight.activeTransfers).toBe(0);
  });
});

describe('the client', () => {
  it('fetches the manifest and every file, verifying chunks and files, through ranged requests', async () => {
    const dest = path.join(root, 'fetched-1');
    const fetchImpl = fetchAs([CALLER]);
    const manifest = await fetchManifest(published.manifestUrl!, { fetch: fetchImpl, task: 'task-42', expectedSha256: published.sha256 });
    expect(manifest.sha256).toBe(published.sha256);
    const result = await fetchArtifactSet(manifest.set, { dest, fetch: fetchImpl, task: 'task-42', concurrency: 3 });
    expect(result.files.map((f) => f.path)).toEqual(published.set.entries.map((e) => e.path));
    for (const [rel, data] of Object.entries(bytes)) {
      expect(readFileSync(path.join(dest, rel)).equals(data), rel).toBe(true);
    }
    expect(result.bytesFetched).toBe(Object.values(bytes).reduce((s, b) => s + b.length, 0));
    expect(existsSync(path.join(dest, 'gds/top.gds.part'))).toBe(false);
  });

  it('resumes after a cut with only the missing chunks, and the file still hashes right', async () => {
    const dest = path.join(root, 'fetched-2');
    const gds = published.set.entries[0];
    // The third chunk of the big file dies once; the first run fails there with two chunks kept.
    const first = fetchAs([CALLER], { failOnce: (u, range) => u.includes(gds.sha256) && (range ?? '').startsWith(`bytes=${2 * CHUNK}-`) });
    const only: ArtifactSet = { ...published.set, entries: [gds] };
    await expect(fetchArtifactSet(only, { dest, fetch: first, task: 'task-42', concurrency: 1 })).rejects.toMatchObject({ code: 'no_reachable_location' });
    expect(existsSync(path.join(dest, 'gds/top.gds.part'))).toBe(true);
    expect(existsSync(path.join(dest, 'gds/top.gds.part.json'))).toBe(true);
    const state = JSON.parse(readFileSync(path.join(dest, 'gds/top.gds.part.json'), 'utf8')) as { done: number[] };
    expect(state.done).toEqual([0, 1]);

    const resumed = await fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42', concurrency: 2 });
    expect(resumed.files[0].bytesFetched).toBe(gds.size - state.done.length * CHUNK);
    expect(readFileSync(path.join(dest, 'gds/top.gds')).equals(bytes['gds/top.gds'])).toBe(true);
    expect(existsSync(path.join(dest, 'gds/top.gds.part'))).toBe(false);
  });

  it('a wrong chunk is refused by name and nothing wrong is kept', async () => {
    const dest = path.join(root, 'fetched-3');
    const gds = published.set.entries[0];
    const corrupting = fetchAs([CALLER], {
      corrupt: (u, body) => {
        if (!u.includes(gds.sha256)) return body;
        const copy = new Uint8Array(body);
        copy[5] ^= 0xff;
        return copy;
      },
    });
    const only: ArtifactSet = { ...published.set, entries: [gds] };
    const err = await fetchArtifactSet(only, { dest, fetch: corrupting, task: 'task-42', concurrency: 1 }).catch((e) => e);
    expect(err).toBeInstanceOf(ArtifactFetchError);
    expect(err.code).toBe('no_reachable_location');
    expect(JSON.stringify(err.detail)).toContain('chunk_hash_mismatch');
    expect(existsSync(path.join(dest, 'gds/top.gds'))).toBe(false);
  });

  it('a server whose Repr-Digest disagrees with the manifest is refused before a byte is written', async () => {
    const dest = path.join(root, 'fetched-4');
    const drc = published.set.entries[1];
    const lying: ArtifactSet = { ...published.set, entries: [{ ...drc, sha256: '11'.repeat(32) }] };
    registry.addManifest({ sha256: '22'.repeat(32), canonical: new Uint8Array(), set: lying, audience: '*' });
    // The endpoint still serves the real blob by its real hash; make the entry point at it.
    const err = await fetchArtifactSet({ ...lying, entries: [{ ...lying.entries[0], locations: [{ kind: 'https', url: `${BASE}/artifacts/blobs/sha256:${drc.sha256}` }] }] }, { dest, fetch: fetchAs([CALLER]), task: 'task-42' }).catch((e) => e);
    expect(err.code).toBe('no_reachable_location');
    expect(JSON.stringify(err.detail)).toContain('Repr-Digest');
    expect(existsSync(path.join(dest, 'reports/drc.txt'))).toBe(false);
  });

  it('refuses location kinds it does not speak by name and reports them', async () => {
    const dest = path.join(root, 'fetched-5');
    const drc = published.set.entries[1];
    const set: ArtifactSet = { ...published.set, entries: [{ ...drc, locations: [{ kind: 'oci', ref: 'ghcr.io/x/y@sha256:' + drc.sha256 }, { kind: 'relay', agent: 'flow' }] }] };
    const err = await fetchArtifactSet(set, { dest, fetch: fetchAs([CALLER]), task: 'task-42' }).catch((e) => e);
    expect(err.code).toBe('no_reachable_location');
    expect(err.message).toContain('oci: location_kind_unsupported');
    expect(err.message).toContain('relay: location_kind_unsupported');
  });

  it('never writes outside the destination', async () => {
    const dest = path.join(root, 'fetched-6');
    const set = { ...published.set, entries: [{ ...published.set.entries[1], path: 'ok/../../escape.txt' }] } as ArtifactSet;
    await expect(fetchArtifactSet(set, { dest, fetch: fetchAs([CALLER]), task: 'task-42' })).rejects.toMatchObject({ code: 'path_escape' });
  });

  it('refuses a directory symlink that points outside the destination', async () => {
    const dest = path.join(root, 'symlink-dest');
    const outside = path.join(root, 'symlink-outside');
    mkdirSync(dest);
    mkdirSync(outside);
    symlinkSync(outside, path.join(dest, 'reports'));
    const only: ArtifactSet = { ...published.set, entries: [published.set.entries[1]] };
    await expect(fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42' })).rejects.toMatchObject({ code: 'path_escape' });
    expect(existsSync(path.join(outside, 'drc.txt'))).toBe(false);
  });

  it('refuses a symlink at the destination root', async () => {
    const outside = path.join(root, 'root-link-outside');
    const dest = path.join(root, 'root-link');
    mkdirSync(outside);
    symlinkSync(outside, dest);
    const only: ArtifactSet = { ...published.set, entries: [published.set.entries[1]] };
    await expect(fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42' })).rejects.toMatchObject({ code: 'path_escape' });
    expect(existsSync(path.join(outside, 'reports/drc.txt'))).toBe(false);
  });

  it.each(['.part', '.part.json'])('refuses a %s symlink without changing its target', async (suffix) => {
    const dest = path.join(root, `sidecar-${suffix.slice(1)}`);
    const entry = published.set.entries[1];
    const target = path.join(dest, entry.path);
    const outside = path.join(root, `outside-${suffix.slice(1)}`);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(outside, 'keep me');
    symlinkSync(outside, `${target}${suffix}`);
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    await expect(fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42' })).rejects.toMatchObject({ code: 'path_escape' });
    expect(readFileSync(outside, 'utf8')).toBe('keep me');
  });

  it('keeps nested manifest downloads inside the original destination', async () => {
    const inner: ArtifactSet = { ...published.set, entries: [published.set.entries[1]] };
    const canonical = canonicalManifestBytes(inner);
    const sha256 = createHash('sha256').update(canonical).digest('hex');
    registry.addManifest({ sha256, canonical, set: inner, audience: [CALLER], task: 'task-42' });
    const outer: ArtifactSet = {
      ...published.set,
      entries: [{ path: 'nested/index.json', kind: 'manifest', size: canonical.length, sha256, mediaType: 'application/vnd.webagents.artifact-set+json', locations: [{ kind: 'https', url: `${BASE}/artifacts/blobs/sha256:${sha256}` }] }],
    };
    const dest = path.join(root, 'nested-symlink-dest');
    const outside = path.join(root, 'nested-symlink-outside');
    mkdirSync(path.join(dest, 'nested'), { recursive: true });
    mkdirSync(outside);
    symlinkSync(outside, path.join(dest, 'nested/reports'));
    await expect(fetchArtifactSet(outer, { dest, fetch: fetchAs([CALLER]), task: 'task-42' })).rejects.toMatchObject({ code: 'path_escape' });
    expect(existsSync(path.join(outside, 'drc.txt'))).toBe(false);
  });

  it('re-fetches a completed chunk when the resume file has been corrupted', async () => {
    const dest = path.join(root, 'corrupt-resume');
    const entry = published.set.entries[0];
    const target = path.join(dest, entry.path);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(`${target}.part`, Buffer.alloc(entry.size));
    writeFileSync(`${target}.part.json`, JSON.stringify({ sha256: entry.sha256, size: entry.size, pieceSize: CHUNK, done: [0] }));
    const ranges: string[] = [];
    const serving = fetchAs([CALLER]);
    const fetchImpl: typeof fetch = async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      ranges.push(request.headers.get('range') ?? '');
      return serving(request);
    };
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    await fetchArtifactSet(only, { dest, fetch: fetchImpl, task: 'task-42', concurrency: 1 });
    expect(ranges).toContain(`bytes=0-${CHUNK - 1}`);
    expect(readFileSync(target).equals(bytes[entry.path])).toBe(true);
  });

  it('requires a full retry when saved progress has no chunk hashes', async () => {
    const dest = path.join(root, 'no-chunk-resume');
    const entry = published.set.entries[1];
    const target = path.join(dest, entry.path);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(`${target}.part`, Buffer.alloc(entry.size));
    writeFileSync(`${target}.part.json`, JSON.stringify({ sha256: entry.sha256, size: entry.size, pieceSize: CHUNK, done: [0] }));
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    await expect(fetchArtifactSet(only, { dest, fetch: () => { throw new Error('unexpected fetch'); }, task: 'task-42', rangeSize: CHUNK })).rejects.toMatchObject({ code: 'file_hash_mismatch' });
    expect(existsSync(`${target}.part`)).toBe(false);
    expect(existsSync(`${target}.part.json`)).toBe(false);
    await fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42', rangeSize: CHUNK });
    expect(readFileSync(target).equals(bytes[entry.path])).toBe(true);
  });

  it('starts fresh when resume state contains malformed JSON', async () => {
    const dest = path.join(root, 'malformed-resume');
    const entry = published.set.entries[1];
    const target = path.join(dest, entry.path);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(`${target}.part`, Buffer.alloc(entry.size));
    writeFileSync(`${target}.part.json`, '{broken');
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    const result = await fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42' });
    expect(result.bytesFetched).toBe(entry.size);
    expect(readFileSync(target).equals(bytes[entry.path])).toBe(true);
  });

  it('propagates a filesystem error while reading resume state', async () => {
    const dest = path.join(root, 'unreadable-resume');
    const entry = published.set.entries[1];
    const target = path.join(dest, entry.path);
    mkdirSync(path.dirname(target), { recursive: true });
    mkdirSync(`${target}.part.json`);
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    await expect(fetchArtifactSet(only, { dest, fetch: () => { throw new Error('unexpected fetch'); }, task: 'task-42' })).rejects.toMatchObject({ code: 'EISDIR' });
    expect(existsSync(`${target}.part`)).toBe(false);
  });

  it('settles queued persistence failures and cleans temporary state files', async () => {
    const dest = path.join(root, 'state-write-failure');
    const entry = published.set.entries[0];
    const target = path.join(dest, entry.path);
    mkdirSync(path.dirname(target), { recursive: true });
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    let started = 0;
    let release!: () => void;
    const allStarted = new Promise<void>((resolve) => { release = resolve; });
    const serving = fetchAs([CALLER]);
    const fetchImpl: typeof fetch = async (input, init) => {
      if (started === 0) mkdirSync(`${target}.part.json`);
      if (++started === 4) release();
      await allStarted;
      return serving(input, init);
    };
    await expect(fetchArtifactSet(only, { dest, fetch: fetchImpl, task: 'task-42', concurrency: 4 })).rejects.toBeTruthy();
    expect(started).toBe(4);
    expect(existsSync(`${target}.part`)).toBe(true);
    expect(existsSync(target)).toBe(false);
    expect(requireStateTemps(path.dirname(target))).toEqual([]);
    rmSync(`${target}.part.json`, { recursive: true });
    await fetchArtifactSet(only, { dest, fetch: fetchAs([CALLER]), task: 'task-42' });
    expect(readFileSync(target).equals(bytes[entry.path])).toBe(true);
  });

  it('persists concurrent completions and resumes only the interrupted range', async () => {
    const dest = path.join(root, 'parallel-resume');
    const entry = published.set.entries[0];
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    const target = path.join(dest, entry.path);
    let failRange: (() => void) | undefined;
    const interrupted = new Promise<Response>((_resolve, reject) => { failRange = () => reject(new Error('connection reset')); });
    const serving = fetchAs([CALLER]);
    const fetchImpl: typeof fetch = (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      return request.headers.get('range') === `bytes=${2 * CHUNK}-${3 * CHUNK - 1}` ? interrupted : serving(request);
    };
    let completed = 0;
    await expect(fetchArtifactSet(only, {
      dest, fetch: fetchImpl, task: 'task-42', concurrency: 4,
      onProgress: () => { if (++completed === 3) failRange?.(); },
    })).rejects.toMatchObject({ code: 'no_reachable_location' });
    const state = JSON.parse(readFileSync(`${target}.part.json`, 'utf8')) as { done: number[] };
    expect(state.done).toEqual([0, 1, 3]);
    expect(requireStateTemps(path.dirname(target))).toEqual([]);
    const ranges: string[] = [];
    const resumedFetch: typeof fetch = (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      ranges.push(request.headers.get('range') ?? '');
      return serving(request);
    };
    const result = await fetchArtifactSet(only, { dest, fetch: resumedFetch, task: 'task-42', concurrency: 4 });
    expect(ranges).toEqual([`bytes=${2 * CHUNK}-${3 * CHUNK - 1}`]);
    expect(result.bytesFetched).toBe(CHUNK);
    expect(readFileSync(target).equals(bytes[entry.path])).toBe(true);
  });

  it('finishes an empty file without a transfer or leftover sidecars', async () => {
    const dest = path.join(root, 'empty-fetch');
    const entry = published.set.entries[2];
    const target = path.join(dest, entry.path);
    const only: ArtifactSet = { ...published.set, entries: [entry] };
    const result = await fetchArtifactSet(only, { dest, fetch: () => { throw new Error('unexpected fetch'); }, task: 'task-42' });
    expect(result.bytesFetched).toBe(0);
    expect(readFileSync(target).length).toBe(0);
    expect(existsSync(`${target}.part`)).toBe(false);
    expect(existsSync(`${target}.part.json`)).toBe(false);
  });
});
