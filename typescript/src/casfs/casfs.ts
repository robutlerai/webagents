/**
 * `CasFs`: one interface to CASFS projects on the platform or in a local
 * directory, with the same operations, names and `Cas:*` refusals as an app's
 * `host.cas` and the portal's HTTP API.
 *
 *   const fs = CasFs.platform({ baseUrl, token });
 *   const p = await fs.open(projectId);
 *   await p.list('pdk');
 *   await p.ensurePdkMount({ pkg: 'sky130A', commit: true });
 *   const lib = await p.read('pdk/sky130A@2026.10/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib');
 *
 * Mount commands on the platform run on the server, which checks that the
 * caller may read the source, refuses cycles and over-deep chains, and
 * commits; the handle then reloads its head. In a local directory, `mount`
 * places the pointer and commits locally; reading through it uses a local
 * package store or the platform.
 */

import { CasError, type MountEntry, type MountSource } from './core.js';
import { MemoryChunkCache, PlatformApi, PlatformBackend, type CasBackend, type ChunkCache, type MountResult, type PlatformApiOptions } from './backends.js';
import { CasProject, type Heads, type ListEntry, type MountOpener } from './project.js';

export type MountSourceInput = { kind: 'project'; id: string } | { kind: 'pdk'; pkg: string; version?: string; id?: string };

export interface CasFsOptions {
  api?: PlatformApi | null;
  cache?: ChunkCache;
  /** A directory of local package stores by source id (`<dir>/<sourceId>/cas/...`), read before the platform. */
  packagesDir?: string;
  /** Node only: the local directory backend, loaded on demand so browser bundles never import `node:fs`. */
  localPackage?: (packagesDir: string | undefined, sourceId: string) => Promise<CasBackend | null>;
}

export class CasFs {
  constructor(readonly opts: CasFsOptions) {}

  static platform(o: PlatformApiOptions & { cache?: ChunkCache; packagesDir?: string }): CasFs {
    return new CasFs({ api: new PlatformApi(o), cache: o.cache ?? new MemoryChunkCache(), packagesDir: o.packagesDir });
  }

  /** The mount opener every handle uses: a local package store first, then the platform under this caller's rights. */
  opener(): MountOpener {
    return {
      open: async ({ source, path }) => {
        const local = this.opts.localPackage ? await this.opts.localPackage(this.opts.packagesDir, source.id) : null;
        if (local) return { backend: local };
        if (!this.opts.api) throw new CasError('Cas:SourceOffline', `${path}: no local copy of its source and no platform to read it from`, { path });
        return { backend: new PlatformBackend(this.opts.api, source.id) };
      },
    };
  }

  /** Open a platform project at its head. */
  async open(projectId: string): Promise<CasHandle> {
    const api = this.opts.api;
    if (!api) throw new CasError('Cas:SourceOffline', 'no platform configured');
    const backend = new PlatformBackend(api, projectId);
    const heads: Heads = {
      swap: async (base, next) => {
        try {
          await api.commit(projectId, base, next);
          return { ok: true };
        } catch (e) {
          if (e instanceof CasError && e.code === 'Cas:HeadMoved') return { ok: false, current: (e.args.head as string | null) ?? null };
          throw e;
        }
      },
    };
    const h = new CasHandle(this, projectId, backend, heads, api);
    await h.reload();
    return h;
  }

  /** Open a project kept in a local directory (`backend` is a LocalDirBackend, `heads` its LocalHeads). */
  async openLocal(projectId: string, backend: CasBackend, heads: Heads & { get(): Promise<string | null> }): Promise<CasHandle> {
    const h = new CasHandle(this, projectId, backend, heads, null, heads);
    await h.reload();
    return h;
  }
}

export class CasHandle {
  project!: CasProject;
  head: string | null = null;

  constructor(
    private readonly fs: CasFs,
    readonly projectId: string,
    private readonly backend: CasBackend,
    private readonly heads: Heads,
    private readonly api: PlatformApi | null,
    private readonly localHeads: { get(): Promise<string | null> } | null = null,
  ) {}

  async reload(): Promise<void> {
    this.head = this.api ? (await this.api.head(this.projectId)).head : await this.localHeads!.get();
    this.project = await CasProject.open({ backend: this.backend, root: this.head, heads: this.heads, mounts: this.fs.opener(), cache: this.fs.opts.cache });
  }

  list(path = ''): Promise<ListEntry[]> {
    return this.project.list(path);
  }
  stat(path: string) {
    return this.project.stat(path);
  }
  read(path: string, range?: { offset?: number; length?: number }): Promise<Uint8Array> {
    return this.project.read(path, range);
  }
  readStream(path: string, range?: { offset?: number; length?: number }) {
    return this.project.readStream(path, range);
  }
  write(path: string, data: Uint8Array | string, opts?: { mediaType?: string; policy?: 'cas' | 'both' | 'local-served' | 'local-only' }) {
    return this.project.write(path, data, opts);
  }
  mkdir(path: string) {
    return this.project.mkdir(path);
  }
  delete(path: string) {
    return this.project.delete(path);
  }
  async commit(): Promise<string> {
    const root = await this.project.commit();
    this.head = root;
    return root;
  }
  history(limit = 50) {
    if (!this.api) throw new CasError('Cas:SourceOffline', 'history is kept on the platform');
    return this.api.history(this.projectId, limit);
  }

  // ── mounts ──

  async mounts(root?: string | null): Promise<Array<{ path: string; source: MountSource; root: string }>> {
    if (this.api) return (await this.api.mounts(this.projectId, root)).mounts;
    const out: Array<{ path: string; source: MountSource; root: string }> = [];
    const walk = async (path: string) => {
      for (const e of await this.project.list(path)) {
        const p = path ? `${path}/${e.name}` : e.name;
        if (e.type === 'mount') out.push({ path: p, source: e.source!, root: e.root! });
        else if (e.type === 'dir') await walk(p);
      }
    };
    await walk('');
    return out;
  }

  /** Mount another project or a published PDK at `path`, pinned by hash. One commit. */
  async mount(path: string, source: MountSourceInput, root?: string | null): Promise<MountResult> {
    if (this.api) {
      const out = await this.api.mount(this.projectId, { path, source, ...(root ? { root } : {}) });
      await this.reload();
      return out;
    }
    if (!root || !source.id || (source.kind === 'pdk' && !source.version)) {
      throw new CasError('Cas:MountSource', 'a local mount names the source id, the version of a PDK, and the root to pin');
    }
    const entry: MountEntry = { type: 'mount', readOnly: true, root, source: source.kind === 'pdk' ? { kind: 'pdk', id: source.id, pkg: source.pkg, version: source.version! } : { kind: 'project', id: source.id } };
    if ((await this.project.stat(path).catch(() => null)) !== null) throw new CasError('Cas:Exists', `${path} exists`, { path });
    await this.project.placeMount(path, entry);
    return { head: await this.commit(), path, entry };
  }

  async unmount(path: string): Promise<{ head: string | null; path: string }> {
    if (this.api) {
      const out = await this.api.unmount(this.projectId, path);
      await this.reload();
      return out;
    }
    await this.project.unmount(path);
    return { head: await this.commit(), path };
  }

  /** Move a project mount's pin (default: the source's current head). A PDK version never moves. */
  async updateMount(path: string, root?: string | null): Promise<MountResult & { from: string }> {
    if (!this.api) throw new CasError('Cas:SourceOffline', 'moving a pin needs the platform to read the source head');
    const out = await this.api.updateMount(this.projectId, path, root);
    await this.reload();
    return out;
  }

  /**
   * The PDK mount at `pdk/<pkg>@<version>`, shared by every cell that uses
   * the kit: the existing one when present, else the named or newest
   * published version. Without `commit`, answers the entry to `placeMount`
   * so it lands in one commit with the change that needs it.
   */
  async ensurePdkMount(a: { pkg: string; version?: string; id?: string; commit?: boolean }): Promise<MountResult & { added: boolean }> {
    if (!this.api) throw new CasError('Cas:SourceOffline', 'resolving a PDK by name needs the platform');
    const out = await this.api.ensurePdkMount(this.projectId, { pkg: a.pkg, ...(a.version ? { version: a.version } : {}), ...(a.id ? { id: a.id } : {}) }, !!a.commit);
    if (a.commit) await this.reload();
    return out;
  }

  placeMount(path: string, entry: MountEntry) {
    return this.project.placeMount(path, entry);
  }
}
