/**
 * The agent side of the artifact relay: serving a platform's `blob.read`
 * requests over the agent's own outbound connection, under a credit window.
 *
 * Frames (the protocol is written down in the SDK documentation under
 * protocols/artifacts):
 *
 *   platform -> agent   blob.read   { request_id, sha256, offset, length, credit? }
 *   platform -> agent   blob.credit { request_id, frames }
 *   platform -> agent   blob.cancel { request_id }
 *   agent -> platform   blob.data   BINARY: version, kind, request id, offset, payload
 *   agent -> platform   blob.end    { request_id, offset, length, bytes }
 *   agent -> platform   blob.error  { request_id, error: { code, message } }
 *   agent -> platform   artifact.manifest { task_id, sha256, size, locations, manifest?, progress? }
 *
 * A `blob.data` frame carries at most 256 KiB. The agent sends no more frames
 * than the credit it holds: `blob.read.credit` (default 4) to begin with, plus
 * every `blob.credit`. Frames of concurrent reads interleave by request id,
 * one frame per turn. `blob.cancel` stops a read where it is; nothing more is
 * sent for it. A connection that drops ends every read on it; the platform
 * asks again for what it still needs after the agent reconnects.
 */

import { createReadStream } from 'node:fs';
import type { Readable } from 'node:stream';

import type { ArtifactLocation } from './manifest';
import type { ArtifactRegistry } from './registry';

export const BLOB_FRAME_VERSION = 1;
export const BLOB_FRAME_KIND_DATA = 1;
export const BLOB_FRAME_MAX_BYTES = 256 * 1024;
export const BLOB_FRAME_HEADER_MAX = 2 + 2 + 64 + 8;
export const DEFAULT_CREDIT_FRAMES = 4;
export const MAX_CREDIT_FRAMES = 1024;
export const MAX_CONCURRENT_READS = 16;
export const MAX_READ_BYTES = 64 * 1024 * 1024;
export const MAX_REQUEST_ID_BYTES = 64;

export interface RelaySocket {
  send(data: string | Uint8Array): void;
}

export interface BlobReadRequest {
  request_id: string;
  sha256: string;
  offset: number;
  length: number;
  credit?: number;
}

export type RelayErrorCode = 'bad_request' | 'unknown_blob' | 'range_invalid' | 'too_many_reads' | 'read_failed' | 'duplicate_request';

const utf8 = new TextEncoder();
const utf8d = new TextDecoder('utf-8', { fatal: true });

/** Encode one `blob.data` frame. */
export function encodeBlobDataFrame(requestId: string, offset: number, payload: Uint8Array): Uint8Array {
  const id = utf8.encode(requestId);
  if (id.length === 0 || id.length > MAX_REQUEST_ID_BYTES) throw new Error('request id must be 1 to 64 bytes');
  if (payload.byteLength > BLOB_FRAME_MAX_BYTES) throw new Error(`a blob.data payload is at most ${BLOB_FRAME_MAX_BYTES} bytes`);
  const out = new Uint8Array(2 + 2 + id.length + 8 + payload.byteLength);
  const view = new DataView(out.buffer);
  out[0] = BLOB_FRAME_VERSION;
  out[1] = BLOB_FRAME_KIND_DATA;
  view.setUint16(2, id.length);
  out.set(id, 4);
  view.setBigUint64(4 + id.length, BigInt(offset));
  out.set(payload, 12 + id.length);
  return out;
}

/** Decode a `blob.data` frame, or null for anything that is not one. */
export function decodeBlobDataFrame(frame: Uint8Array): { requestId: string; offset: number; payload: Uint8Array } | null {
  if (frame.byteLength < 12 || frame[0] !== BLOB_FRAME_VERSION || frame[1] !== BLOB_FRAME_KIND_DATA) return null;
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  const idLength = view.getUint16(2);
  if (idLength === 0 || idLength > MAX_REQUEST_ID_BYTES || frame.byteLength < 12 + idLength) return null;
  let requestId: string;
  try {
    requestId = utf8d.decode(frame.subarray(4, 4 + idLength));
  } catch {
    return null;
  }
  const offset = Number(view.getBigUint64(4 + idLength));
  if (!Number.isSafeInteger(offset)) return null;
  const payload = frame.subarray(12 + idLength);
  if (payload.byteLength > BLOB_FRAME_MAX_BYTES) return null;
  return { requestId, offset, payload };
}

interface ActiveRead {
  requestId: string;
  socket: RelaySocket;
  credit: number;
  cancelled: boolean;
  wake: (() => void) | null;
  stream: Readable | null;
}

export interface ArtifactRelayOptions {
  maxConcurrentReads?: number;
  frameBytes?: number;
  openRange?: (path: string, start: number, end: number, highWaterMark: number) => Readable;
}

function eventId(): string {
  return `evt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export class ArtifactRelay {
  private readonly reads = new Map<string, ActiveRead>();
  private readonly maxReads: number;
  private readonly frameBytes: number;
  private readonly openRange: NonNullable<ArtifactRelayOptions['openRange']>;

  constructor(readonly registry: ArtifactRegistry, options: ArtifactRelayOptions = {}) {
    this.maxReads = options.maxConcurrentReads ?? MAX_CONCURRENT_READS;
    this.frameBytes = Math.min(BLOB_FRAME_MAX_BYTES, options.frameBytes ?? BLOB_FRAME_MAX_BYTES);
    this.openRange = options.openRange ?? ((path, start, end, highWaterMark) => createReadStream(path, { start, end, highWaterMark }));
  }

  get activeReads(): number {
    return this.reads.size;
  }

  /** The `artifact.manifest` frame announcing a set on a task. */
  manifestFrame(args: { taskId: string; sessionId?: string; sha256: string; size: number; locations: ArtifactLocation[]; manifest?: unknown; progress?: { done: number; total: number } }): Record<string, unknown> {
    return {
      type: 'artifact.manifest',
      event_id: eventId(),
      timestamp: Date.now(),
      ...(args.sessionId ? { session_id: args.sessionId } : {}),
      task_id: args.taskId,
      sha256: args.sha256,
      size: args.size,
      locations: args.locations,
      ...(args.manifest !== undefined ? { manifest: args.manifest } : {}),
      ...(args.progress ? { progress: args.progress } : {}),
    };
  }

  announce(socket: RelaySocket, args: Parameters<ArtifactRelay['manifestFrame']>[0]): void {
    socket.send(JSON.stringify(this.manifestFrame(args)));
  }

  /**
   * Handle one JSON frame from the platform. Returns true when it was one of
   * this relay's (`blob.read`, `blob.credit`, `blob.cancel`), false otherwise.
   */
  handleFrame(socket: RelaySocket, frame: Record<string, unknown>): boolean {
    switch (frame.type) {
      case 'blob.read':
        void this.startRead(socket, frame);
        return true;
      case 'blob.credit': {
        const read = this.reads.get(String(frame.request_id));
        const frames = frame.frames;
        if (read && read.socket === socket && typeof frames === 'number' && Number.isSafeInteger(frames) && frames > 0) {
          read.credit = Math.min(MAX_CREDIT_FRAMES, read.credit + frames);
          read.wake?.();
        }
        return true;
      }
      case 'blob.cancel': {
        const read = this.reads.get(String(frame.request_id));
        if (read && read.socket === socket) {
          read.cancelled = true;
          read.stream?.destroy();
          read.wake?.();
        }
        return true;
      }
      default:
        return false;
    }
  }

  /** A connection dropped: every read on it is over. */
  socketClosed(socket: RelaySocket): void {
    for (const read of this.reads.values()) {
      if (read.socket !== socket) continue;
      read.cancelled = true;
      read.stream?.destroy();
      read.wake?.();
    }
  }

  private error(socket: RelaySocket, requestId: string, code: RelayErrorCode, message: string): void {
    socket.send(JSON.stringify({ type: 'blob.error', event_id: eventId(), timestamp: Date.now(), request_id: requestId, error: { code, message } }));
  }

  private async startRead(socket: RelaySocket, frame: Record<string, unknown>): Promise<void> {
    const requestId = typeof frame.request_id === 'string' ? frame.request_id : '';
    if (!requestId || utf8.encode(requestId).length > MAX_REQUEST_ID_BYTES) {
      this.error(socket, requestId || '?', 'bad_request', 'request_id must be a string of 1 to 64 bytes');
      return;
    }
    if (this.reads.has(requestId)) {
      this.error(socket, requestId, 'duplicate_request', 'a read with this request_id is in progress');
      return;
    }
    const sha256 = typeof frame.sha256 === 'string' && /^[0-9a-f]{64}$/.test(frame.sha256) ? frame.sha256 : null;
    const offset = frame.offset;
    const length = frame.length;
    if (!sha256 || typeof offset !== 'number' || typeof length !== 'number' || !Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length <= 0) {
      this.error(socket, requestId, 'bad_request', 'blob.read needs sha256 (hex), offset (>= 0) and length (> 0)');
      return;
    }
    if (length > MAX_READ_BYTES) {
      this.error(socket, requestId, 'range_invalid', `one read is at most ${MAX_READ_BYTES} bytes`);
      return;
    }
    const blob = this.registry.getBlob(sha256);
    const manifest = blob ? null : this.registry.getManifest(sha256);
    if (!blob && !manifest) {
      this.error(socket, requestId, 'unknown_blob', 'this agent has not published that hash');
      return;
    }
    const size = blob ? blob.size : manifest!.canonical.byteLength;
    if (offset >= size || offset + length > size) {
      this.error(socket, requestId, 'range_invalid', `the blob is ${size} bytes; ${offset}+${length} is past its end`);
      return;
    }
    if (this.reads.size >= this.maxReads) {
      this.error(socket, requestId, 'too_many_reads', `this agent serves at most ${this.maxReads} reads at once`);
      return;
    }
    const credit = typeof frame.credit === 'number' && Number.isSafeInteger(frame.credit) && frame.credit > 0 ? Math.min(MAX_CREDIT_FRAMES, frame.credit) : DEFAULT_CREDIT_FRAMES;
    const read: ActiveRead = { requestId, socket, credit, cancelled: false, wake: null, stream: null };
    this.reads.set(requestId, read);

    let sent = 0;
    let position = offset;
    const waitForCredit = () => new Promise<void>((resolve) => {
      if (read.credit > 0 || read.cancelled) { resolve(); return; }
      read.wake = () => { read.wake = null; resolve(); };
    });
    const sendPayload = async (payload: Uint8Array): Promise<boolean> => {
      for (let off = 0; off < payload.byteLength; off += this.frameBytes) {
        await waitForCredit();
        if (read.cancelled) return false;
        const slice = payload.subarray(off, Math.min(payload.byteLength, off + this.frameBytes));
        socket.send(encodeBlobDataFrame(requestId, position, slice));
        position += slice.byteLength;
        sent += slice.byteLength;
        read.credit -= 1;
        // One frame per turn so concurrent reads interleave.
        await new Promise<void>((r) => setImmediate(r));
      }
      return true;
    };

    try {
      if (manifest) {
        const ok = await sendPayload(manifest.canonical.subarray(offset, offset + length));
        if (!ok) return;
      } else {
        const stream = this.openRange(blob!.path, offset, offset + length - 1, this.frameBytes);
        read.stream = stream;
        for await (const chunk of stream) {
          const ok = await sendPayload(chunk as Uint8Array);
          if (!ok) return;
        }
      }
      if (read.cancelled) return;
      socket.send(JSON.stringify({ type: 'blob.end', event_id: eventId(), timestamp: Date.now(), request_id: requestId, offset, length, bytes: sent }));
    } catch (err) {
      if (!read.cancelled) this.error(socket, requestId, 'read_failed', err instanceof Error ? err.message : String(err));
    } finally {
      read.stream?.destroy();
      this.reads.delete(requestId);
    }
  }
}
