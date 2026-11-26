import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';

import { PlatformApi, PlatformBackend, RESUMABLE_AFTER } from '../../../src/casfs/index.js';

const sha = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');
const BASE = 'https://portal.example';
const PID = '11111111-2222-4333-8444-555555555555';
const LOC = '/api/content/uploads/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee';

// The resumable route in memory: checks each chunk against the declared list and the whole file against its hash.
function portal(o: { cutAt?: number; refuseAt?: number } = {}) {
  let got = Buffer.alloc(0);
  let decl: { sha256: string; size: number; chunkList: { chunkSize: number; hashes: string[] } } | null = null;
  let status = 'open';
  let cut = o.cutAt;
  const log: string[] = [];
  const auth: string[] = [];
  const f = (async (url: string, init: RequestInit = {}) => {
    const m = init.method ?? 'GET';
    const h = (init.headers ?? {}) as Record<string, string>;
    const path = url.replace(BASE, '');
    log.push(`${m} ${path}`);
    auth.push(h.authorization ?? '');
    if (m === 'POST' && path === `/api/casfs/projects/${PID}/uploads`) {
      decl = JSON.parse(String(init.body));
      return Response.json({ ok: true, mode: 'local', location: LOC }, { status: 201 });
    }
    if (m === 'PATCH' && path === LOC) {
      const off = Number(h['Upload-Offset']);
      if (off !== got.length) return new Response(null, { status: 409 });
      const b = Buffer.from(await new Response(init.body as BodyInit).arrayBuffer());
      if (cut === off) {
        cut = undefined;
        throw new TypeError('fetch failed');
      }
      if (o.refuseAt === off || sha(b) !== decl!.chunkList.hashes[off / decl!.chunkList.chunkSize]) return Response.json({ error: 'chunk' }, { status: 460 });
      got = Buffer.concat([got, b]);
      if (got.length === decl!.size) status = sha(got) === decl!.sha256 ? 'committed' : 'failed';
      return new Response(null, { status: 204 });
    }
    if (m === 'HEAD' && path === LOC) return new Response(null, { status: 200, headers: { 'upload-offset': String(got.length) } });
    if (m === 'GET' && path === LOC) return Response.json({ status });
    if (m === 'PUT' && path.startsWith(`/api/casfs/projects/${PID}/blobs/`)) return Response.json({ ok: true });
    return new Response(null, { status: 404 });
  }) as unknown as typeof fetch;
  return { f, log, auth, got: () => got };
}

describe('large blobs go through the resumable route', () => {
  const big = new Uint8Array(RESUMABLE_AFTER + 3 * 1024 * 1024 + 17);
  for (let i = 0; i < big.length; i += 4096) big[i] = i & 0xff;

  it('chunked, verified, resumed after a cut connection, with the caller credentials', async () => {
    const p = portal({ cutAt: 8 * 1024 * 1024 });
    const be = new PlatformBackend(new PlatformApi({ baseUrl: BASE, token: 'tok', fetch: p.f }), PID);
    await be.put(big, sha(big));
    expect(sha(p.got())).toBe(sha(big));
    expect(p.log).toContain(`HEAD ${LOC}`);
    expect(p.auth.every((a) => a === 'Bearer tok')).toBe(true);
    expect(p.log.some((l) => l.includes('/blobs/'))).toBe(false);
  }, 60_000);

  it('a chunk refused by its hash stops by name', async () => {
    const p = portal({ refuseAt: 0 });
    const be = new PlatformBackend(new PlatformApi({ baseUrl: BASE, fetch: p.f }), PID);
    await expect(be.put(big, sha(big))).rejects.toMatchObject({ code: 'Cas:ChunkMismatch' });
  }, 60_000);

  it('small blobs keep the single PUT', async () => {
    const p = portal();
    const small = new Uint8Array([1, 2, 3]);
    await new PlatformBackend(new PlatformApi({ baseUrl: BASE, fetch: p.f }), PID).put(small, sha(small));
    expect(p.log).toEqual([`PUT /api/casfs/projects/${PID}/blobs/${sha(small)}`]);
  });
});
