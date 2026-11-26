/**
 * Where CASFS blobs come from, and the platform API the commands use.
 *
 * A backend stores blobs by hash and answers ranged reads; it never
 * interprets a manifest, and nothing it returns is trusted: the project
 * (`./project.ts`) verifies every manifest, chunk and small file it reads.
 *
 *   PlatformBackend  `/api/casfs/projects/<id>/blobs/<sha>` on a Robutler
 *                    portal, under a bearer token and, optionally, a CASFS
 *                    grant for that project (`X-Casfs-Grant`).
 *   MemoryBackend    tests and scratch.
 *
 * The local directory backend lives in `./local.ts` (Node only) and the
 * browser backend in `./browser.ts`.
 */

import { uploadResumable, type ResumableSource } from './resumable.js';
import { CasError, type CasErrorCode, type MountSource } from './core.js';

export interface CasBackend {
  readonly name: string;
  has(sha: string): Promise<boolean>;
  /** Bytes `[start, end)` of a blob. */
  readRange(sha: string, start: number, end: number): Promise<Uint8Array>;
  /** Store a blob that hashes to `sha` (the caller has hashed it). */
  put(bytes: Uint8Array, sha: string): Promise<void>;
}

/** Content-addressed bytes kept on this machine for every project and mount: chunks, small files, manifests. */
export interface ChunkCache {
  get(sha: string): Promise<Uint8Array | null>;
  put(sha: string, bytes: Uint8Array): Promise<void>;
}

export class MemoryChunkCache implements ChunkCache {
  private map = new Map<string, Uint8Array>();
  private used = 0;
  constructor(private readonly limitBytes = 256 * 1024 * 1024) {}
  async get(sha: string): Promise<Uint8Array | null> {
    return this.map.get(sha) ?? null;
  }
  async put(sha: string, bytes: Uint8Array): Promise<void> {
    if (this.map.has(sha) || bytes.length > this.limitBytes) return;
    this.map.set(sha, bytes);
    this.used += bytes.length;
    for (const [k, v] of this.map) {
      if (this.used <= this.limitBytes) break;
      this.map.delete(k);
      this.used -= v.length;
    }
  }
}

export class MemoryBackend implements CasBackend {
  readonly blobs = new Map<string, Uint8Array>();
  readonly stats = { reads: 0, bytesRead: 0, puts: 0 };
  constructor(readonly name = 'memory') {}
  async has(sha: string): Promise<boolean> {
    return this.blobs.has(sha);
  }
  async readRange(sha: string, start: number, end: number): Promise<Uint8Array> {
    const b = this.blobs.get(sha);
    if (!b) throw new CasError('Cas:NotFound', `blob ${sha} is not in ${this.name}`, { sha });
    const out = b.slice(start, Math.min(end, b.length));
    this.stats.reads++;
    this.stats.bytesRead += out.length;
    return out;
  }
  async put(bytes: Uint8Array, sha: string): Promise<void> {
    if (!this.blobs.has(sha)) this.blobs.set(sha, Uint8Array.from(bytes));
    this.stats.puts++;
  }
}

export interface PlatformApiOptions {
  /** The portal, for example `https://robutler.ai`. */
  baseUrl: string;
  /** A bearer token: an agent's API key or a person's CLI token. */
  token?: string | null;
  /** CASFS grants by project id, presented as `X-Casfs-Grant` on that project's requests. */
  grants?: Record<string, string>;
  fetch?: typeof fetch;
}

/** The CASFS HTTP API of a portal: blobs, heads, commits, history and mounts. */
export class PlatformApi {
  readonly baseUrl: string;
  private readonly f: typeof fetch;
  constructor(private readonly opts: PlatformApiOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, '');
    this.f = opts.fetch ?? fetch.bind(globalThis);
  }

  headers(projectId: string, extra: Record<string, string> = {}): Record<string, string> {
    const h: Record<string, string> = { ...extra };
    if (this.opts.token) h.authorization = `Bearer ${this.opts.token}`;
    const g = this.opts.grants?.[projectId];
    if (g) h['x-casfs-grant'] = g;
    return h;
  }

  url(projectId: string, rest = ''): string {
    return `${this.baseUrl}/api/casfs/projects/${encodeURIComponent(projectId)}${rest}`;
  }

  raw(projectId: string, rest: string, init: RequestInit = {}): Promise<Response> {
    return this.f(this.url(projectId, rest), { ...init, headers: this.headers(projectId, (init.headers as Record<string, string>) ?? {}) });
  }

  async json<T>(projectId: string, rest: string, init: RequestInit = {}): Promise<T> {
    const res = await this.raw(projectId, rest, init);
    const body = (await res.json().catch(() => null)) as { ok?: boolean; error?: { code?: string; message?: string; args?: Record<string, unknown> } } | null;
    if (!res.ok || !body || body.ok === false) {
      const code = (body?.error?.code ?? (res.status === 404 ? 'Cas:NotFound' : res.status === 401 ? 'Cas:Forbidden' : 'Cas:SourceOffline')) as CasErrorCode;
      throw new CasError(code, (body?.error?.message ?? `HTTP ${res.status}`).replace(/^Cas:[A-Za-z]+: /, ''), body?.error?.args ?? {});
    }
    return body as T;
  }

  head(projectId: string): Promise<{ head: string | null; counter: number; platformBytes?: number }> {
    return this.json(projectId, '');
  }
  commit(projectId: string, base: string | null, root: string, message?: string | null): Promise<{ head: string; counter: number }> {
    return this.json(projectId, '/commit', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ base, root, message: message ?? null }) });
  }
  history(projectId: string, limit = 50): Promise<{ versions: unknown[] }> {
    return this.json(projectId, `/versions?limit=${limit}`);
  }
  mounts(projectId: string, root?: string | null): Promise<{ root: string | null; mounts: Array<{ path: string; source: MountSource; root: string }> }> {
    return this.json(projectId, `/mounts${root ? `?root=${root}` : ''}`);
  }
  mount(projectId: string, body: { path: string; source: unknown; root?: string | null }): Promise<MountResult> {
    return this.json(projectId, '/mounts', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  }
  ensurePdkMount(projectId: string, pdk: { pkg: string; version?: string; id?: string }, commit: boolean): Promise<MountResult & { added: boolean }> {
    return this.json(projectId, '/mounts', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ pdk, commit }) });
  }
  updateMount(projectId: string, path: string, root?: string | null): Promise<MountResult & { from: string }> {
    return this.json(projectId, '/mounts', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path, ...(root ? { root } : {}) }) });
  }
  /** Upload one large blob in verified, resumable chunks (see resumable.ts). */
  uploadResumable(projectId: string, sha: string, src: ResumableSource, o?: Parameters<typeof uploadResumable>[4]): Promise<void> {
    return uploadResumable(
      {
        platform: (path, init = {}) => this.f(`${this.baseUrl}${path}`, { ...init, headers: this.headers(projectId, (init.headers as Record<string, string>) ?? {}) }),
        external: (url, init = {}) => {
          if (!/^https:\/\//.test(url)) throw new CasError('Cas:SourceOffline', 'the storage session is not https');
          return this.f(url, init);
        },
      },
      projectId,
      sha,
      src,
      o,
    );
  }
  unmount(projectId: string, path: string): Promise<{ head: string | null; path: string }> {
    return this.json(projectId, `/mounts?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
  }
}

export interface MountResult {
  head: string | null;
  path: string;
  entry: { type: 'mount'; readOnly: true; root: string; source: MountSource };
}

/** Above this size a blob is uploaded through the resumable route rather than one PUT. */
export const RESUMABLE_AFTER = 64 * 1024 * 1024;

/** One project's blobs on the platform. */
export class PlatformBackend implements CasBackend {
  readonly name: string;
  constructor(private readonly api: PlatformApi, readonly projectId: string) {
    this.name = `platform:${projectId.slice(0, 8)}`;
  }
  async has(sha: string): Promise<boolean> {
    const r = await this.api.raw(this.projectId, `/blobs/${sha}`, { method: 'HEAD' });
    return r.status === 200;
  }
  async readRange(sha: string, start: number, end: number): Promise<Uint8Array> {
    if (end <= start) return new Uint8Array(0);
    const r = await this.api.raw(this.projectId, `/blobs/${sha}`, { headers: { range: `bytes=${start}-${end - 1}` } });
    if (r.status === 404) {
      const body = (await r.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
      throw new CasError((body?.error?.code as CasErrorCode) ?? 'Cas:NotFound', body?.error?.message ?? `blob ${sha} is not readable here`, { sha });
    }
    if (r.status === 403) {
      const body = (await r.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
      throw new CasError((body?.error?.code as CasErrorCode) ?? 'Cas:Forbidden', body?.error?.message ?? 'refused', { sha });
    }
    if (r.status !== 206 && r.status !== 200) throw new CasError('Cas:SourceOffline', `HTTP ${r.status} reading ${sha}`, { sha });
    const data = new Uint8Array(await r.arrayBuffer());
    return r.status === 200 ? data.slice(start, end) : data;
  }
  async put(bytes: Uint8Array, sha: string): Promise<void> {
    if (bytes.length > RESUMABLE_AFTER) {
      // Large blobs go up in verified chunks that survive a cut connection, and past the single-request limit.
      await this.api.uploadResumable(this.projectId, sha, { size: bytes.length, slice: async (a, b) => bytes.subarray(a, b) });
      return;
    }
    await this.api.json(this.projectId, `/blobs/${sha}`, { method: 'PUT', headers: { 'content-type': 'application/octet-stream' }, body: bytes as unknown as BodyInit });
  }
}
