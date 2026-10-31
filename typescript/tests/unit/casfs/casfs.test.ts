import { createHash } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import {
  CasError,
  CasFs,
  CasProject,
  DIR_KIND,
  MAX_MOUNT_DEPTH,
  MemoryBackend,
  MemoryChunkCache,
  PlatformApi,
  checkDirManifest,
  sha256Hex,
  type CasBackend,
  type Heads,
  type MountEntry,
  type MountOpener,
} from '../../../src/casfs/index.js';
import { DirChunkCache, LocalDirBackend, LocalHeads, localPackageBackend } from '../../../src/casfs/local.js';
import { parseSource, parseTarget, runCas } from '../../../src/cli/cas-command.js';

const A = '11111111-1111-4111-8111-111111111111';
const B = '22222222-2222-4222-8222-222222222222';
const K = '66666666-6666-4666-8666-666666666666';
const code = (p: Promise<unknown>) => p.then(() => 'ok', (e: CasError) => e.code);

function memHeads(): Heads & { head: string | null } {
  const h = {
    head: null as string | null,
    async swap(base: string | null, next: string) {
      if (h.head !== base) return { ok: false as const, current: h.head };
      h.head = next;
      return { ok: true as const };
    },
  };
  return h;
}

async function sourceProject(files: Record<string, string>, backend = new MemoryBackend('src')) {
  const heads = memHeads();
  const p = await CasProject.open({ backend, heads });
  for (const [path, text] of Object.entries(files)) await p.write(path, text);
  const root = await p.commit();
  return { backend, root };
}

const mountOf = (id: string, root: string): MountEntry => ({ type: 'mount', readOnly: true, root, source: { kind: 'project', id } });
const openerFor = (backends: Record<string, CasBackend>, log: string[] = []): MountOpener => ({
  open: async ({ source }) => {
    log.push(source.id);
    const b = backends[source.id];
    if (!b) throw new CasError('Cas:MountSource', 'not readable');
    return { backend: b };
  },
});

const dirs: string[] = [];
afterEach(async () => {
  while (dirs.length) await rm(dirs.pop()!, { recursive: true, force: true });
});
const tmp = async () => {
  const d = await mkdtemp(join(tmpdir(), 'casfs-'));
  dirs.push(d);
  return d;
};

describe('CASFS core', () => {
  it('the synchronous SHA-256 equals node:crypto', () => {
    for (const n of [0, 55, 56, 64, 1000, 70_001]) {
      const b = new Uint8Array(n).map((_, i) => (i * 31) & 255);
      expect(sha256Hex(b)).toBe(createHash('sha256').update(b).digest('hex'));
    }
  });

  it('accepts exactly the mount entries the portal accepts', () => {
    const sha = 'a'.repeat(64);
    const ok = { type: 'mount', readOnly: true, root: sha, source: { kind: 'pdk', id: K, pkg: 'sky130A', version: '2026.10' } };
    expect(() => checkDirManifest({ kind: DIR_KIND, entries: { m: ok } })).not.toThrow();
    for (const bad of [{ ...ok, readOnly: false }, { ...ok, extra: 1 }, { ...ok, source: { kind: 'pdk', id: K, pkg: 'sky130A' } }, { ...ok, source: { kind: 'url', id: K } }]) {
      expect(() => checkDirManifest({ kind: DIR_KIND, entries: { m: bad } })).toThrow(/Cas:ManifestInvalid/);
    }
  });
});

describe('CasProject', () => {
  it('writes, commits and reads back; a large file is chunked and a range moves only its chunks', async () => {
    const backend = new MemoryBackend();
    const heads = memHeads();
    const p = await CasProject.open({ backend, heads, chunkSize: 256 * 1024 });
    const big = new Uint8Array(3 * 256 * 1024 + 7).map((_, i) => i & 255);
    await p.write('data/big.bin', big);
    await p.write('a.txt', 'hello');
    const root = await p.commit();
    const q = await CasProject.open({ backend, root });
    backend.stats.bytesRead = 0;
    const r = await q.read('data/big.bin', { offset: 256 * 1024 + 5, length: 10 });
    expect([...r]).toEqual([...big.subarray(256 * 1024 + 5, 256 * 1024 + 15)]);
    expect(backend.stats.bytesRead).toBeLessThanOrEqual(256 * 1024 + 4096);
    expect(new TextDecoder().decode(await q.read('a.txt'))).toBe('hello');
  });

  it('a lying store is refused by name', async () => {
    const backend = new MemoryBackend();
    const p = await CasProject.open({ backend, heads: memHeads() });
    await p.write('a.txt', 'truth');
    const root = await p.commit();
    backend.blobs.set(sha256Hex('truth'), new TextEncoder().encode('lies!'));
    expect(await code((await CasProject.open({ backend, root })).read('a.txt'))).toBe('Cas:HashMismatch');
  });

  it('reads through a mount from the SOURCE backend, records full paths, and names the mount in a run record', async () => {
    const src = await sourceProject({ 'lib/x.lib': 'liberty' });
    const backend = new MemoryBackend('mine');
    const p = await CasProject.open({ backend, heads: memHeads(), mounts: openerFor({ [B]: src.backend }) });
    await p.write('cells/a.json', '{}');
    await p.placeMount('pdk/kit', mountOf(B, src.root));
    const root = await p.commit();
    expect(backend.blobs.has(sha256Hex('liberty'))).toBe(false);
    const q = await CasProject.open({ backend, root, mounts: openerFor({ [B]: src.backend }) });
    expect((await q.list('pdk')).map((e) => e.type)).toEqual(['mount']);
    expect(new TextDecoder().decode(await q.read('pdk/kit/lib/x.lib'))).toBe('liberty');
    const rec = q.runRecord({ runId: 'r', tool: {}, argv: [], outputs: [], exitCode: 0 });
    expect(rec.inputs.map((i) => i.path)).toEqual(['pdk/kit/lib/x.lib']);
    expect(rec.mounts).toEqual([{ at: 'pdk/kit', root: src.root, source: { kind: 'project', id: B } }]);
  });

  it('nothing inside a mount changes; cycles and over-deep chains are refused by name', async () => {
    const src = await sourceProject({ 'x': 'x' });
    const p = await CasProject.open({ backend: new MemoryBackend(), heads: memHeads(), mounts: openerFor({ [B]: src.backend }), sourceId: A });
    await p.placeMount('m', mountOf(B, src.root));
    expect(await code(p.write('m/new', 'no'))).toBe('Cas:MountReadOnly');
    expect(await code(p.write('m', 'no'))).toBe('Cas:MountReadOnly');
    expect(await code(p.mkdir('m/d'))).toBe('Cas:MountReadOnly');
    expect(await code(p.unmount('nothing'))).toBe('Cas:NotFound');
    // A source that mounts the reader's own project back.
    const back = await CasProject.open({ backend: src.backend, heads: memHeads() });
    await back.placeMount('back', mountOf(A, src.root));
    const backRoot = await back.commit();
    const c = await CasProject.open({ backend: new MemoryBackend(), heads: memHeads(), mounts: openerFor({ [B]: src.backend }), sourceId: A });
    await c.placeMount('b', mountOf(B, backRoot));
    expect(await code(c.list('b/back'))).toBe('Cas:MountCycle');
    // MAX_MOUNT_DEPTH + 1 distinct sources in a chain.
    const shared = new MemoryBackend('shared');
    const ids = Array.from({ length: MAX_MOUNT_DEPTH + 1 }, (_, i) => `7777777${i}-7777-4777-8777-777777777777`);
    let prev = (await sourceProject({ leaf: 'x' }, shared)).root;
    for (const id of ids) {
      const s = await CasProject.open({ backend: shared, heads: memHeads() });
      await s.placeMount('n', mountOf(id, prev));
      prev = await s.commit();
    }
    const all: MountOpener = { open: async () => ({ backend: shared }) };
    const d = await CasProject.open({ backend: shared, root: prev, mounts: all });
    expect(await code(d.list('n/n/n/n'))).toBe('ok');
    expect(await code(d.list('n/n/n/n/n'))).toBe('Cas:MountDepth');
  });

  it('merges by path when the head moved; the same path changed twice is Cas:PathConflict', async () => {
    const backend = new MemoryBackend();
    const heads = memHeads();
    const p0 = await CasProject.open({ backend, heads });
    await p0.write('a', '1');
    const base = await p0.commit();
    const x = await CasProject.open({ backend, heads, root: base });
    const y = await CasProject.open({ backend, heads, root: base });
    await x.write('b', 'x');
    await y.write('c', 'y');
    await x.commit();
    const merged = await y.commit();
    const z = await CasProject.open({ backend, root: merged });
    expect((await z.list()).map((e) => e.name)).toEqual(['a', 'b', 'c']);
    const u = await CasProject.open({ backend, heads, root: merged });
    const v = await CasProject.open({ backend, heads, root: merged });
    await u.write('a', 'u');
    await v.write('a', 'v');
    await u.commit();
    expect(await code(v.commit())).toBe('Cas:PathConflict');
  });
});

describe('local directory, the machine cache and the CLI', () => {
  it('a local project keeps a mount as a pointer and resolves it from a local package store; the cache serves a second reader', async () => {
    const dir = await tmp();
    const pkgs = await tmp();
    const cacheDir = await tmp();
    const kit = new LocalDirBackend(join(pkgs, K));
    const { root: kitRoot } = await sourceProject({ 'libs/a.lib': 'kit bytes' }, kit as unknown as MemoryBackend);
    const fs = new CasFs({ api: null, cache: new DirChunkCache(cacheDir), packagesDir: pkgs, localPackage: localPackageBackend });
    const h = await fs.openLocal(A, new LocalDirBackend(dir), new LocalHeads(dir, A));
    await h.write('cells/c.json', '{}');
    await h.commit();
    await h.mount('pdk/sky130A@2026.10', { kind: 'pdk', id: K, pkg: 'sky130A', version: '2026.10' }, kitRoot);
    expect(await new LocalDirBackend(dir).has(sha256Hex('kit bytes'))).toBe(false);
    expect(new TextDecoder().decode(await h.read('pdk/sky130A@2026.10/libs/a.lib'))).toBe('kit bytes');
    expect(await new DirChunkCache(cacheDir).get(sha256Hex('kit bytes'))).not.toBeNull();
    expect((await h.mounts()).map((m) => m.path)).toEqual(['pdk/sky130A@2026.10']);
    // With the package store gone, the cache still answers the bytes (the manifests are cached too).
    await rm(join(pkgs, K), { recursive: true, force: true });
    const cachedOnly = new CasFs({ api: null, cache: new DirChunkCache(cacheDir), localPackage: async () => new MemoryBackend('empty') });
    const h2 = await cachedOnly.openLocal(A, new LocalDirBackend(dir), new LocalHeads(dir, A));
    expect(new TextDecoder().decode(await h2.read('pdk/sky130A@2026.10/libs/a.lib'))).toBe('kit bytes');
    await h.unmount('pdk/sky130A@2026.10');
    expect(await h.mounts()).toEqual([]);
  });

  it('the platform API carries the token, the grant and the named refusals', async () => {
    const seen: Array<{ url: string; method: string; headers: Record<string, string>; body?: string }> = [];
    const fetchStub = (async (url: string, init: RequestInit = {}) => {
      seen.push({ url, method: init.method ?? 'GET', headers: init.headers as Record<string, string>, body: init.body as string | undefined });
      if (url.endsWith('/mounts') && init.method === 'POST') return new Response(JSON.stringify({ ok: false, error: { code: 'Cas:MountSource', message: 'Cas:MountSource: the package sky130A is not a source you can read' } }), { status: 400 });
      return new Response(JSON.stringify({ ok: true, head: null, counter: 0 }), { status: 200 });
    }) as unknown as typeof fetch;
    const api = new PlatformApi({ baseUrl: 'https://portal.example/', token: 'tok', grants: { [A]: 'grant-a' }, fetch: fetchStub });
    const fs = new CasFs({ api, cache: new MemoryChunkCache() });
    const h = await fs.open(A);
    const e = await h.ensurePdkMount({ pkg: 'sky130A', commit: true }).catch((x) => x as CasError);
    expect((e as CasError).code).toBe('Cas:MountSource');
    expect((e as CasError).message).toBe('Cas:MountSource: the package sky130A is not a source you can read');
    const post = seen.find((s) => s.method === 'POST')!;
    expect(post.url).toBe(`https://portal.example/api/casfs/projects/${A}/mounts`);
    expect(post.headers.authorization).toBe('Bearer tok');
    expect(post.headers['x-casfs-grant']).toBe('grant-a');
    expect(JSON.parse(post.body!)).toEqual({ pdk: { pkg: 'sky130A' }, commit: true });
  });

  it('the CLI parses targets and sources and refuses what it does not know', async () => {
    expect(parseTarget(`${A}:pdk/kit`)).toEqual({ project: A, path: 'pdk/kit' });
    expect(parseSource('pdk:sky130A@2026.10')).toEqual({ kind: 'pdk', pkg: 'sky130A', version: '2026.10' });
    expect(parseSource(`project:${B}`)).toEqual({ kind: 'project', id: B });
    expect(() => parseSource('http://x')).toThrow(/Cas:MountSource/);
    expect(() => parseTarget('nope')).toThrow(/Cas:PathInvalid/);
    const fs = new CasFs({ api: null });
    expect(await code(runCas(fs, 'frobnicate', [], {}, { out: () => {}, write: () => {} }))).toBe('Cas:PathInvalid');
  });
});
