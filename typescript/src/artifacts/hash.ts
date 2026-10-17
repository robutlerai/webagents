/**
 * Streaming hashes for artifact files: the whole-file SHA-256 and, for a file
 * above the chunk size, the chunk list. Nothing is read whole into memory; a
 * file passes through once in `highWaterMark`-sized pieces.
 */

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';

import { CHUNK_SIZE_DEFAULT, isValidChunkSize, type ChunkList } from './manifest';

export const HASH_READ_HIGH_WATER_MARK = 1024 * 1024;

export interface HashedBytes {
  sha256: string;
  size: number;
  /** Present only when the bytes span more than one chunk. */
  chunks?: ChunkList;
}

/** Hash an async byte source once, chunked at `chunkSize`. */
export async function hashStream(source: AsyncIterable<Uint8Array>, chunkSize = CHUNK_SIZE_DEFAULT): Promise<HashedBytes> {
  if (!isValidChunkSize(chunkSize)) throw new Error(`invalid chunk size ${chunkSize}`);
  const whole = createHash('sha256');
  let part = createHash('sha256');
  let filled = 0;
  let size = 0;
  const hashes: string[] = [];
  for await (const raw of source) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength);
    whole.update(chunk);
    size += chunk.length;
    let off = 0;
    while (off < chunk.length) {
      const take = Math.min(chunkSize - filled, chunk.length - off);
      part.update(chunk.subarray(off, off + take));
      filled += take;
      off += take;
      if (filled === chunkSize) {
        hashes.push(part.digest('hex'));
        part = createHash('sha256');
        filled = 0;
      }
    }
  }
  if (filled > 0) hashes.push(part.digest('hex'));
  const out: HashedBytes = { sha256: whole.digest('hex'), size };
  if (size > chunkSize) out.chunks = { chunkSize, hashes };
  return out;
}

/** Hash a file on disk, streaming. */
export async function hashFile(path: string, chunkSize = CHUNK_SIZE_DEFAULT): Promise<HashedBytes> {
  const st = await stat(path);
  if (!st.isFile()) throw new Error(`${path} is not a regular file`);
  const hashed = await hashStream(createReadStream(path, { highWaterMark: HASH_READ_HIGH_WATER_MARK }), chunkSize);
  if (hashed.size !== st.size) throw new Error(`${path} changed while it was being hashed (${st.size} then ${hashed.size} bytes)`);
  return hashed;
}

/** sha256 hex of an in-memory buffer. */
export function sha256Hex(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}
