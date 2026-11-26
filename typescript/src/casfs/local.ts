/**
 * The local directory backend and the machine's chunk cache (Node only).
 *
 * A local CASFS store is a directory in the platform store's own layout:
 *
 *   <dir>/cas/<ab>/<sha256>      blobs, by hash
 *   <dir>/heads/<projectId>      the head of each project kept here
 *
 * A mount in a local project is what its manifest says: a pointer to the
 * source and a pinned root. Nothing of the source is copied into the
 * project's store. A reader resolves the mount from a local package store
 * (`<packagesDir>/<sourceId>/`, the same layout, for sites with no network)
 * or through the platform under its own rights on the source; every chunk it
 * reads lands in the machine's chunk cache, shared by every project and mount
 * on the machine. Bytes are cached exactly as stored, so an encrypted or
 * sealed source is cached as ciphertext.
 */

import { mkdir, open, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { CasError, isSha, sha256 } from './core.js';
import type { CasBackend, ChunkCache } from './backends.js';
import type { Heads } from './project.js';

const blobPath = (dir: string, sha: string) => join(dir, 'cas', sha.slice(0, 2), sha);

async function atomicWrite(path: string, bytes: Uint8Array | string): Promise<void> {
  const tmp = `${path}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
  await writeFile(tmp, bytes);
  await rename(tmp, path);
}

export class LocalDirBackend implements CasBackend {
  readonly name: string;
  constructor(readonly dir: string) {
    this.name = `local:${dir}`;
  }
  async has(sha: string): Promise<boolean> {
    if (!isSha(sha)) return false;
    return stat(blobPath(this.dir, sha)).then(() => true, () => false);
  }
  async readRange(sha: string, start: number, end: number): Promise<Uint8Array> {
    if (!isSha(sha)) throw new CasError('Cas:NotFound', 'not a sha256');
    let fh;
    try {
      fh = await open(blobPath(this.dir, sha), 'r');
    } catch {
      throw new CasError('Cas:NotFound', `blob ${sha} is not in ${this.dir}`, { sha });
    }
    try {
      const len = Math.max(0, end - start);
      const buf = new Uint8Array(len);
      const { bytesRead } = await fh.read(buf, 0, len, start);
      return buf.subarray(0, bytesRead);
    } finally {
      await fh.close();
    }
  }
  async put(bytes: Uint8Array, sha: string): Promise<void> {
    if (sha256(bytes) !== sha) throw new CasError('Cas:HashMismatch', 'the bytes do not hash to the name', { expected: sha });
    if (await this.has(sha)) return;
    await mkdir(join(this.dir, 'cas', sha.slice(0, 2)), { recursive: true });
    await atomicWrite(blobPath(this.dir, sha), bytes);
  }
}

/** The head of a project kept in a local directory, moved by compare-and-swap. */
export class LocalHeads implements Heads {
  constructor(readonly dir: string, readonly projectId: string) {}
  private get path() {
    return join(this.dir, 'heads', this.projectId);
  }
  async get(): Promise<string | null> {
    const s = await readFile(this.path, 'utf8').catch(() => '');
    return isSha(s.trim()) ? s.trim() : null;
  }
  async swap(base: string | null, next: string): Promise<{ ok: true } | { ok: false; current: string | null }> {
    const cur = await this.get();
    if (cur !== base) return { ok: false, current: cur };
    await mkdir(join(this.dir, 'heads'), { recursive: true });
    await atomicWrite(this.path, next + '\n');
    return { ok: true };
  }
}

/** The machine's chunk cache directory: `$XDG_CACHE_HOME/robutler/casfs`, else `~/.cache/robutler/casfs`. */
export function defaultCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME || join(homedir(), '.cache');
  return join(base, 'robutler', 'casfs');
}

/** Content-addressed chunk cache on disk; every read is checked against its name. */
export class DirChunkCache implements ChunkCache {
  constructor(readonly dir: string = defaultCacheDir()) {}
  async get(sha: string): Promise<Uint8Array | null> {
    if (!isSha(sha)) return null;
    const b = await readFile(blobPath(this.dir, sha)).catch(() => null);
    if (!b) return null;
    const u = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    return sha256(u) === sha ? u : null;
  }
  async put(sha: string, bytes: Uint8Array): Promise<void> {
    if (!isSha(sha) || sha256(bytes) !== sha) return;
    const p = blobPath(this.dir, sha);
    if (await stat(p).then(() => true, () => false)) return;
    await mkdir(join(this.dir, 'cas', sha.slice(0, 2)), { recursive: true });
    await atomicWrite(p, bytes);
  }
}

/** A local package store for one source project, if this machine holds one. */
export async function localPackageBackend(packagesDir: string | undefined, sourceId: string): Promise<LocalDirBackend | null> {
  if (!packagesDir) return null;
  const dir = join(packagesDir, sourceId);
  return (await stat(join(dir, 'cas')).then(() => true, () => false)) ? new LocalDirBackend(dir) : null;
}
