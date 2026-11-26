/**
 * A CASFS project: read by path and verified by hash, written by path and
 * committed by compare-and-swap.
 *
 *   const p = await CasProject.open({ backend, root, heads, mounts, cache });
 *   await p.read('cells/a/layout.json', { offset, length });   // only the chunks covering the range move
 *   await p.write('runs/r1/report.txt', bytes, { mediaType });
 *   await p.commit();                                           // merges by path when the head moved
 *
 * Reads verify everything: a manifest against its name and canonical form, a
 * chunk against its hash (`Cas:ChunkMismatch`), a small file against its own
 * hash (`Cas:HashMismatch`). Verified bytes go to the chunk cache, which is
 * content addressed, so one pinned PDK is cached once for every project that
 * mounts it.
 *
 * Mounts. A `{ type: 'mount', readOnly: true, root, source }` entry shows
 * another project, or a published PDK, as a read-only folder pinned to one
 * root of the source. Reads resolve through it to a sub-project over the
 * source's own backend (`mounts.open`), under the caller's rights on the
 * source; anything that would change the inside of a mount is
 * `Cas:MountReadOnly`. Chains deeper than MAX_MOUNT_DEPTH are `Cas:MountDepth`
 * and a chain that returns to a source already in it is `Cas:MountCycle`.
 * Every file read is listed in `opened` by its full path, and
 * `mountsOpened()` names every mount a reader went through, for run records.
 */

import {
  CHUNK_DEFAULT,
  CasError,
  DIR_KIND,
  INLINE_CHUNK_HASHES,
  MAX_MOUNT_DEPTH,
  canon,
  canonBytes,
  checkChunkSize,
  checkMountSource,
  checkPolicy,
  chunkCount,
  concat,
  isSha,
  parseDirManifest,
  sameMount,
  sha256,
  splitPath,
  type DirManifest,
  type Entry,
  type FileEntry,
  type MountEntry,
  type MountSource,
  type Policy,
} from './core.js';
import type { CasBackend, ChunkCache } from './backends.js';
import { fromHex, toHex } from './sha256.js';

export interface Heads {
  swap(base: string | null, next: string): Promise<{ ok: true } | { ok: false; current: string | null }>;
}

export interface MountOpener {
  /** Backends for a mount's source, and an opener scoped to that source for the mounts nested in it. */
  open(a: { path: string; source: MountSource; root: string }): Promise<{ backend: CasBackend; mounts?: MountOpener }>;
}

export interface OpenedFile {
  path: string;
  sha256?: string;
  size: number;
  policy: Policy;
}

export interface ProjectOptions {
  backend: CasBackend;
  root?: string | null;
  heads?: Heads;
  mounts?: MountOpener;
  cache?: ChunkCache;
  chunkSize?: number;
  rootPolicy?: Policy;
  /** Internal to mounts: depth, chain of source ids, path prefix and the shared list of opened files. */
  mountDepth?: number;
  mountChain?: string[];
  sourceId?: string | null;
  pathPrefix?: string;
  opened?: OpenedFile[];
}

type Node = { entries: Map<string, Entry & { node?: Node }>; dirty: boolean };
type Live = Entry & { node?: Node };

export interface ListEntry {
  name: string;
  type: Entry['type'];
  size?: number;
  sha256?: string;
  id?: string;
  policy?: Policy;
  mediaType?: string;
  root?: string;
  source?: MountSource;
}

export class CasProject {
  readonly backend: CasBackend;
  baseRoot: string | null;
  readonly opened: OpenedFile[];
  private rootNode: Node = { entries: new Map(), dirty: true };
  private manifests = new Map<string, DirManifest>();
  private subs = new Map<string, { project: CasProject; entry: MountEntry }>();
  private readonly chunkSize: number;
  private readonly rootPolicy: Policy;

  private constructor(private readonly o: ProjectOptions) {
    this.backend = o.backend;
    this.baseRoot = o.root ?? null;
    this.opened = o.opened ?? [];
    this.chunkSize = checkChunkSize(o.chunkSize ?? CHUNK_DEFAULT);
    this.rootPolicy = o.rootPolicy ?? 'cas';
  }

  static async open(o: ProjectOptions): Promise<CasProject> {
    const p = new CasProject(o);
    if (p.baseRoot) p.rootNode = await p.nodeFor(p.baseRoot);
    return p;
  }

  // ── blobs, verified and cached ──

  private async fetchVerified(sha: string, start: number, end: number, want: string): Promise<Uint8Array> {
    const cached = await this.o.cache?.get(want);
    if (cached && sha256(cached) === want) return cached;
    const b = await this.backend.readRange(sha, start, end);
    if (sha256(b) !== want) {
      throw new CasError(start === 0 && want === sha ? 'Cas:HashMismatch' : 'Cas:ChunkMismatch', `bytes of ${sha.slice(0, 12)} [${start}, ${end}) do not hash to ${want}`, { sha, start, end, expected: want });
    }
    await this.o.cache?.put(want, b);
    return b;
  }

  private async manifest(sha: string): Promise<DirManifest> {
    const have = this.manifests.get(sha);
    if (have) return have;
    const cached = await this.o.cache?.get(sha);
    const bytes = cached ?? (await this.backend.readRange(sha, 0, 32 * 1024 * 1024));
    const m = parseDirManifest(sha, bytes);
    if (!cached) await this.o.cache?.put(sha, bytes);
    this.manifests.set(sha, m);
    return m;
  }

  private async nodeFor(sha: string): Promise<Node> {
    const m = await this.manifest(sha);
    const entries = new Map<string, Live>();
    for (const [name, e] of Object.entries(m.entries)) entries.set(name, { ...e } as Live);
    return { entries, dirty: false };
  }

  // ── mounts ──

  private full(path: string): string {
    const pre = this.o.pathPrefix ?? '';
    return pre ? (path ? `${pre}/${path}` : pre) : path;
  }

  private async mountProject(path: string, entry: MountEntry): Promise<CasProject> {
    const have = this.subs.get(path);
    if (have && sameMount(have.entry, entry)) return have.project;
    checkMountSource(entry.source);
    const full = this.full(path);
    const depth = (this.o.mountDepth ?? 0) + 1;
    if (depth > MAX_MOUNT_DEPTH) throw new CasError('Cas:MountDepth', `mounts nest more than ${MAX_MOUNT_DEPTH} deep at ${full}`, { path: full, limit: MAX_MOUNT_DEPTH });
    const chain = [...(this.o.mountChain ?? []), ...(this.o.sourceId ? [this.o.sourceId] : [])];
    if (chain.includes(entry.source.id)) throw new CasError('Cas:MountCycle', `${full} leads back to a project already in the chain`, { path: full });
    if (!this.o.mounts) throw new CasError('Cas:SourceOffline', `${full} is a mount and this reader has no way to open mounts`, { path: full });
    const b = await this.o.mounts.open({ path: full, source: entry.source, root: entry.root });
    const project = await CasProject.open({
      backend: b.backend,
      root: entry.root,
      mounts: b.mounts ?? this.o.mounts,
      cache: this.o.cache,
      mountDepth: depth,
      mountChain: chain,
      sourceId: entry.source.id,
      pathPrefix: full,
      opened: this.opened,
    });
    this.subs.set(path, { project, entry });
    return project;
  }

  private async resolve(segs: string[], into: boolean): Promise<{ project: CasProject; segs: string[]; mount?: MountEntry }> {
    let node = this.rootNode;
    for (let i = 0; i < segs.length; i++) {
      const e = node.entries.get(segs[i]);
      if (!e) break;
      if (e.type === 'mount') {
        if (i === segs.length - 1 && !into) return { project: this, segs, mount: e };
        const sub = await this.mountProject(segs.slice(0, i + 1).join('/'), e);
        return sub.resolve(segs.slice(i + 1), into);
      }
      if (e.type !== 'dir' || i === segs.length - 1) break;
      if (!e.node) e.node = await this.nodeFor(e.sha256);
      node = e.node;
    }
    return { project: this, segs };
  }

  /** Every mount a reader went through, at any depth: `{ at, root, source }`. */
  mountsOpened(): Array<{ at: string; root: string; source: MountSource }> {
    const out: Array<{ at: string; root: string; source: MountSource }> = [];
    for (const [path, { project, entry }] of this.subs) {
      out.push({ at: this.full(path), root: entry.root, source: { ...entry.source } });
      out.push(...project.mountsOpened());
    }
    return out;
  }

  // ── reading ──

  private async dirChain(segs: string[], create = false): Promise<Node[]> {
    const chain = [this.rootNode];
    for (let i = 0; i < segs.length; i++) {
      const node = chain[chain.length - 1];
      let e = node.entries.get(segs[i]);
      if (e?.type === 'mount') throw readOnly(segs.slice(0, i + 1).join('/'));
      if (!e) {
        if (!create) throw new CasError('Cas:NotFound', `no directory ${segs.slice(0, i + 1).join('/')}`, { path: segs.slice(0, i + 1).join('/') });
        e = { type: 'dir', sha256: '', node: { entries: new Map(), dirty: true } };
        node.entries.set(segs[i], e);
        node.dirty = true;
      }
      if (e.type !== 'dir') throw new CasError('Cas:NotADirectory', `${segs.slice(0, i + 1).join('/')} is not a folder`, { path: segs.slice(0, i + 1).join('/') });
      if (!e.node) e.node = await this.nodeFor(e.sha256);
      chain.push(e.node);
    }
    return chain;
  }

  private policyAt(segs: string[], chain: Node[], e?: Live): Policy {
    let p = this.rootPolicy;
    for (let i = 0; i < segs.length - 1; i++) {
      const d = chain[i].entries.get(segs[i]);
      if (d && d.type === 'dir' && d.policy) p = d.policy;
    }
    if (e && (e.type === 'file' || e.type === 'dir') && e.policy) p = e.policy;
    return p;
  }

  async stat(path: string): Promise<ListEntry & { path: string }> {
    const segs = splitPath(path);
    const r = await this.resolve(segs, false);
    if (r.project !== this) return r.project.stat(r.segs.join('/'));
    if (r.mount) return { name: segs[segs.length - 1], path: segs.join('/'), type: 'mount', root: r.mount.root, source: { ...r.mount.source }, policy: 'cas' };
    if (!segs.length) return { name: '', path: '', type: 'dir', policy: this.rootPolicy };
    const chain = await this.dirChain(segs.slice(0, -1));
    const e = chain[chain.length - 1].entries.get(segs[segs.length - 1]);
    if (!e) throw new CasError('Cas:NotFound', `no such file or directory: ${path}`, { path });
    return { ...view(segs[segs.length - 1], e, this.policyAt(segs, chain, e)), path: segs.join('/') };
  }

  async list(path = ''): Promise<ListEntry[]> {
    const r = await this.resolve(splitPath(path), true);
    if (r.project !== this) return r.project.list(r.segs.join('/'));
    const chain = await this.dirChain(r.segs);
    return [...chain[chain.length - 1].entries]
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([name, e]) => view(name, e, this.policyAt([...r.segs, name], [...chain], e)));
  }

  private async hashes(path: string, e: FileEntry): Promise<string[]> {
    if (e.chunks!.hashes) return e.chunks!.hashes;
    const k = chunkCount(e.size, e.chunks!.size);
    const raw = await this.fetchVerified(e.chunks!.index!, 0, k * 32, e.chunks!.index!);
    if (raw.length !== k * 32) throw new CasError('Cas:HashMismatch', `chunk index of ${path} has the wrong length`, { path });
    const out: string[] = [];
    for (let i = 0; i < k; i++) out.push(toHex(raw.subarray(i * 32, i * 32 + 32)));
    return out;
  }

  async read(path: string, { offset = 0, length }: { offset?: number; length?: number } = {}): Promise<Uint8Array> {
    const segs = splitPath(path);
    const r = await this.resolve(segs, false);
    if (r.project !== this) return r.project.read(r.segs.join('/'), { offset, length });
    if (r.mount) throw new CasError('Cas:IsADirectory', `${path} is a mount (a folder)`, { path });
    const st = await this.stat(path);
    if (st.type !== 'file') throw new CasError('Cas:IsADirectory', `${path} is not a file`, { path });
    const chain = await this.dirChain(segs.slice(0, -1));
    const e = chain[chain.length - 1].entries.get(segs[segs.length - 1]) as FileEntry;
    if (e.growing || !e.sha256) throw new CasError('Cas:NotFound', `${path} is still being written`, { path });
    const end = Math.min(e.size, length === undefined ? e.size : offset + length);
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > e.size || end < offset) throw new CasError('Cas:RangeInvalid', `range outside ${path}`, { path });
    const full = this.full(segs.join('/'));
    if (!this.opened.some((x) => x.path === full)) this.opened.push({ path: full, sha256: e.sha256, size: e.size, policy: st.policy ?? 'cas' });
    if (end === offset) return new Uint8Array(0);
    if (!e.chunks) return (await this.fetchVerified(e.sha256, 0, e.size, e.sha256)).slice(offset, end);
    const hs = await this.hashes(path, e);
    const cs = e.chunks.size;
    const parts: Uint8Array[] = [];
    for (let i = Math.floor(offset / cs); i * cs < end; i++) {
      const s = i * cs;
      const b = await this.fetchVerified(e.sha256, s, Math.min(e.size, s + cs), hs[i]);
      parts.push(b.subarray(Math.max(0, offset - s), Math.min(b.length, end - s)));
    }
    return concat(parts, end - offset);
  }

  async *readStream(path: string, { offset = 0, length }: { offset?: number; length?: number } = {}): AsyncGenerator<Uint8Array> {
    const st = await this.stat(path);
    const size = st.size ?? 0;
    const end = Math.min(size, length === undefined ? size : offset + length);
    const step = 4 * 1024 * 1024;
    for (let o = offset; o < end; o += step) yield await this.read(path, { offset: o, length: Math.min(step, end - o) });
  }

  // ── writing (staged until commit) ──

  async write(path: string, data: Uint8Array | string, opts: { mediaType?: string; policy?: Policy } = {}): Promise<FileEntry> {
    checkPolicy(opts.policy);
    const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    const segs = splitPath(path);
    if (!segs.length) throw new CasError('Cas:PathInvalid', 'cannot write the root');
    const chain = await this.dirChain(segs.slice(0, -1), true);
    const dir = chain[chain.length - 1];
    const old = dir.entries.get(segs[segs.length - 1]);
    if (old?.type === 'mount') throw readOnly(segs.join('/'));
    if (old && old.type !== 'file') throw new CasError('Cas:IsADirectory', `${path} is a folder`, { path });
    const sha = sha256(bytes);
    const e: FileEntry = { type: 'file', size: bytes.length, sha256: sha, ...(opts.mediaType ? { mediaType: opts.mediaType } : {}) };
    if (old?.id) e.id = old.id;
    const policy = opts.policy ?? (old as FileEntry | undefined)?.policy;
    if (policy) e.policy = policy;
    if (bytes.length > this.chunkSize) {
      const hashes: string[] = [];
      for (let o = 0; o < bytes.length; o += this.chunkSize) hashes.push(sha256(bytes.subarray(o, o + this.chunkSize)));
      if (hashes.length <= INLINE_CHUNK_HASHES) e.chunks = { size: this.chunkSize, hashes };
      else {
        const raw = concat(hashes.map(fromHex));
        const idx = sha256(raw);
        await this.backend.put(raw, idx);
        e.chunks = { size: this.chunkSize, index: idx };
      }
    }
    if (policy !== 'local-only' && policy !== 'local-served') await this.backend.put(bytes, sha);
    dir.entries.set(segs[segs.length - 1], e);
    for (const n of chain) n.dirty = true;
    return e;
  }

  async mkdir(path: string): Promise<void> {
    const chain = await this.dirChain(splitPath(path), true);
    for (const n of chain) n.dirty = true;
  }

  async delete(path: string): Promise<void> {
    const segs = splitPath(path);
    const chain = await this.dirChain(segs.slice(0, -1));
    if (!chain[chain.length - 1].entries.delete(segs[segs.length - 1])) throw new CasError('Cas:NotFound', `no such file or directory: ${path}`, { path });
    for (const n of chain) n.dirty = true;
    this.subs.delete(segs.join('/'));
  }

  /** Place a mount entry (from `ensurePdkMount` or the mount API) so it commits together with other changes. */
  async placeMount(path: string, entry: MountEntry): Promise<MountEntry> {
    if (entry?.type !== 'mount' || entry.readOnly !== true || !isSha(entry.root)) throw new CasError('Cas:ManifestInvalid', 'not a mount entry');
    const source = checkMountSource(entry.source);
    const segs = splitPath(path);
    if (!segs.length) throw new CasError('Cas:PathInvalid', 'a mount needs a path below the root');
    const chain = await this.dirChain(segs.slice(0, -1), true);
    const dir = chain[chain.length - 1];
    const old = dir.entries.get(segs[segs.length - 1]);
    const e: MountEntry = { type: 'mount', readOnly: true, root: entry.root, source };
    if (old) {
      if (old.type === 'mount' && sameMount(old, e)) return old;
      throw new CasError('Cas:Exists', `${segs.join('/')} exists`, { path: segs.join('/') });
    }
    dir.entries.set(segs[segs.length - 1], e);
    for (const n of chain) n.dirty = true;
    return e;
  }

  async unmount(path: string): Promise<void> {
    const segs = splitPath(path);
    const chain = await this.dirChain(segs.slice(0, -1));
    const e = chain[chain.length - 1].entries.get(segs[segs.length - 1]);
    if (!e) throw new CasError('Cas:NotFound', `no such path: ${path}`, { path });
    if (e.type !== 'mount') throw new CasError('Cas:NotAMount', `${path} is not a mount`, { path });
    await this.delete(path);
  }

  private async writeNode(n: Node): Promise<string> {
    const entries: Record<string, Entry> = {};
    for (const [name, e] of n.entries) {
      if (e.type === 'dir' && e.node && (e.node.dirty || !e.sha256)) e.sha256 = await this.writeNode(e.node);
      const { node: _node, ...plain } = e as Live;
      void _node;
      entries[name] = plain as Entry;
    }
    const m: DirManifest = { kind: DIR_KIND, entries };
    const bytes = canonBytes(m);
    const sha = sha256(bytes);
    await this.backend.put(bytes, sha);
    this.manifests.set(sha, m);
    n.dirty = false;
    return sha;
  }

  /** Write the staged tree's manifests and answer its root (no head moves). */
  writeTree(): Promise<string> {
    return this.writeNode(this.rootNode);
  }

  /** Flat view: path -> entry, folders as markers (for merges). */
  async flatten(): Promise<Map<string, unknown>> {
    const out = new Map<string, unknown>();
    const walk = async (n: Node, pre: string) => {
      for (const [name, e] of n.entries) {
        const p = pre ? `${pre}/${name}` : name;
        if (e.type === 'dir') {
          if (!e.node) e.node = await this.nodeFor(e.sha256);
          out.set(p, { type: 'dir', ...(e.policy ? { policy: e.policy } : {}) });
          await walk(e.node, p);
        } else {
          const { node: _node, ...plain } = e as Live;
          void _node;
          out.set(p, plain);
        }
      }
    };
    await walk(this.rootNode, '');
    return out;
  }

  private fromFlat(flat: Map<string, unknown>): Node {
    const root: Node = { entries: new Map(), dirty: true };
    for (const p of [...flat.keys()].sort()) {
      const segs = p.split('/');
      let node = root;
      for (const s of segs.slice(0, -1)) {
        let d = node.entries.get(s);
        if (!d) {
          d = { type: 'dir', sha256: '', node: { entries: new Map(), dirty: true } };
          node.entries.set(s, d);
        }
        node = d.node!;
      }
      const v = flat.get(p) as Live;
      const name = segs[segs.length - 1];
      if (v.type === 'dir') {
        if (!node.entries.has(name)) node.entries.set(name, { type: 'dir', sha256: '', ...(v.policy ? { policy: v.policy } : {}), node: { entries: new Map(), dirty: true } });
      } else node.entries.set(name, { ...v });
    }
    return root;
  }

  /**
   * Compare-and-swap commit. When the head moved, merges by path against the
   * base: a path only one side changed takes that side; a path both sides
   * changed differently is `Cas:PathConflict` (nothing is renamed or dropped).
   */
  async commit(): Promise<string> {
    if (!this.o.heads) throw new CasError('Cas:Forbidden', 'this project was opened without a way to move its head');
    let mine = await this.writeTree();
    for (let attempt = 0; attempt < 5; attempt++) {
      const r = await this.o.heads.swap(this.baseRoot, mine);
      if (r.ok) {
        this.baseRoot = mine;
        return mine;
      }
      const view = (root: string | null) => CasProject.open({ backend: this.backend, root, cache: this.o.cache });
      const b = this.baseRoot ? await (await view(this.baseRoot)).flatten() : new Map<string, unknown>();
      const t = await (await view(r.current)).flatten();
      const o = await this.flatten();
      const same = (x: unknown, y: unknown) => (x === undefined && y === undefined) || (x !== undefined && y !== undefined && canon(x) === canon(y));
      const merged = new Map<string, unknown>();
      for (const p of new Set([...b.keys(), ...o.keys(), ...t.keys()])) {
        const bv = b.get(p);
        const ov = o.get(p);
        const tv = t.get(p);
        let v: unknown;
        if (same(ov, bv)) v = tv;
        else if (same(tv, bv) || same(ov, tv)) v = ov;
        else throw new CasError('Cas:PathConflict', `${p} changed on both sides`, { path: p, head: r.current });
        if (v !== undefined) merged.set(p, v);
      }
      this.rootNode = this.fromFlat(merged);
      this.baseRoot = r.current;
      mine = await this.writeTree();
    }
    throw new CasError('Cas:HeadMoved', 'the head kept moving; try again');
  }

  /** A `casfs.run/1` record of what this reader opened, with every mount it went through. */
  runRecord(r: { runId: string; tool: unknown; argv: string[]; env?: string[]; mounts?: Array<{ at: string; root: string }>; outputs: unknown[]; exitCode: number; startedAt?: string; finishedAt?: string; host?: string }) {
    const given = r.mounts ?? [];
    return {
      kind: 'casfs.run/1',
      runId: r.runId,
      tool: r.tool,
      argv: r.argv,
      env: r.env ?? [],
      mounts: [...given, ...this.mountsOpened().filter((m) => !given.some((x) => x.at === m.at && x.root === m.root))],
      inputs: [...this.opened].sort((a, b) => (a.path < b.path ? -1 : 1)),
      outputs: r.outputs,
      exitCode: r.exitCode,
      startedAt: r.startedAt,
      finishedAt: r.finishedAt,
      host: r.host,
    };
  }
}

function readOnly(path: string): CasError {
  return new CasError('Cas:MountReadOnly', `${path} is a mount: nothing inside it can change`, { path });
}

function view(name: string, e: Live, policy: Policy): ListEntry {
  if (e.type === 'mount') return { name, type: 'mount', root: e.root, source: { ...e.source }, policy: 'cas' };
  if (e.type === 'sealed') return { name, type: 'sealed' };
  if (e.type === 'dir') return { name, type: 'dir', sha256: e.sha256 || undefined, policy };
  return { name, type: 'file', size: e.size, sha256: e.sha256, id: e.id, policy, mediaType: e.mediaType };
}
