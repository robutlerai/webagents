/**
 * The browser backend: blobs in the origin private file system (OPFS) of the
 * page that runs the SDK, in the store's own layout (`casfs/cas/<ab>/<sha>`).
 * Useful as a local copy and as a chunk cache in a browser agent; it holds
 * exactly the bytes it was given, verified on write and on read.
 */

import { CasError, isSha, sha256 } from './core.js';
import type { CasBackend, ChunkCache } from './backends.js';

type Dir = FileSystemDirectoryHandle;

export class OpfsBackend implements CasBackend {
  readonly name = 'opfs';
  constructor(private readonly storage: StorageManager | null = typeof navigator !== 'undefined' ? navigator.storage : null) {}

  private async shard(sha: string, create: boolean): Promise<Dir | null> {
    if (!this.storage?.getDirectory) throw new CasError('Cas:SourceOffline', 'this browser has no origin private file system');
    try {
      const root = await (await this.storage.getDirectory()).getDirectoryHandle('casfs', { create });
      return await (await root.getDirectoryHandle('cas', { create })).getDirectoryHandle(sha.slice(0, 2), { create });
    } catch {
      return null;
    }
  }
  private async file(sha: string): Promise<File | null> {
    if (!isSha(sha)) return null;
    const d = await this.shard(sha, false);
    if (!d) return null;
    try {
      return await (await d.getFileHandle(sha)).getFile();
    } catch {
      return null;
    }
  }
  async has(sha: string): Promise<boolean> {
    return !!(await this.file(sha));
  }
  async readRange(sha: string, start: number, end: number): Promise<Uint8Array> {
    const f = await this.file(sha);
    if (!f) throw new CasError('Cas:NotFound', `blob ${sha} is not in this browser`, { sha });
    return new Uint8Array(await f.slice(start, end).arrayBuffer());
  }
  async put(bytes: Uint8Array, sha: string): Promise<void> {
    if (sha256(bytes) !== sha) throw new CasError('Cas:HashMismatch', 'the bytes do not hash to the name', { expected: sha });
    if (await this.has(sha)) return;
    const d = await this.shard(sha, true);
    if (!d) throw new CasError('Cas:SourceOffline', 'this browser refused to store the file');
    const w = await (await d.getFileHandle(sha, { create: true }) as FileSystemFileHandle & { createWritable(): Promise<FileSystemWritableFileStream> }).createWritable();
    try {
      await w.write(bytes as unknown as BufferSource);
    } finally {
      await w.close();
    }
  }
  async get(sha: string): Promise<Uint8Array | null> {
    const f = await this.file(sha);
    if (!f) return null;
    const b = new Uint8Array(await f.arrayBuffer());
    return sha256(b) === sha ? b : null;
  }
  /** This store as the chunk cache of a browser agent. */
  asCache(): ChunkCache {
    return { get: (sha) => this.get(sha), put: (sha, bytes) => this.put(bytes, sha).catch(() => undefined) };
  }
}
