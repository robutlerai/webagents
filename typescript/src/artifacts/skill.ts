/**
 * The producer endpoint: `GET <base>/artifacts/blobs/sha256:<hex>` serves a
 * published blob, or a manifest, by hash.
 *
 *   - RFC 9110 ranges: `Accept-Ranges: bytes`, `206` with `Content-Range` for
 *     one range, `416` past the end, `200` for the whole representation;
 *   - RFC 9530 `Repr-Digest` of the whole representation on `200` and `206`;
 *   - streamed from disk, one read buffer per transfer, never loaded whole;
 *   - authorised by who signed the request: the endpoint is scoped to a
 *     verified caller, and the registry decides from the caller's principals
 *     (`agent:<url>` as the access skill records a verified signature) and,
 *     for a set published for a task, from the `Artifact-Task` header. A URL
 *     alone grants nothing; the refusals are named;
 *   - bounded: at most `maxConcurrentTransfers` bodies in flight, refused by
 *     name (`503 too_many_transfers`) past that.
 *
 * `HEAD` answers the same headers with no body.
 */

import { createReadStream } from 'node:fs';
import { Readable } from 'node:stream';

import { Skill } from '../core/skill';
import { http } from '../core/decorators';
import type { Context } from '../core/types';
import { ARTIFACT_SET_MEDIA_TYPE, parseDigestName } from './manifest';
import { ArtifactRegistry } from './registry';
import { parseRangeHeader, reprDigest } from './http-range';

export const ARTIFACT_BLOB_SUBPATH = '/artifacts/blobs/{digest}';
export const ARTIFACT_TASK_HEADER = 'artifact-task';
export const DEFAULT_MAX_CONCURRENT_TRANSFERS = 16;
export const TRANSFER_HIGH_WATER_MARK = 1024 * 1024;

export interface ArtifactsSkillConfig {
  registry?: ArtifactRegistry;
  maxConcurrentTransfers?: number;
  /** For tests: a reader for a path other than the filesystem. */
  openRange?: (path: string, start: number, end: number) => Readable;
}

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers } });
}

export class ArtifactsSkill extends Skill {
  readonly registry: ArtifactRegistry;
  private readonly maxTransfers: number;
  private inFlight = 0;
  private readonly openRange: (path: string, start: number, end: number) => Readable;

  constructor(config: ArtifactsSkillConfig = {}) {
    super({ name: 'artifacts' });
    this.registry = config.registry ?? new ArtifactRegistry();
    this.maxTransfers = config.maxConcurrentTransfers ?? DEFAULT_MAX_CONCURRENT_TRANSFERS;
    this.openRange = config.openRange ?? ((path, start, end) => createReadStream(path, { start, end, highWaterMark: TRANSFER_HIGH_WATER_MARK }));
  }

  /** Transfers in flight (for status lines and tests). */
  get activeTransfers(): number {
    return this.inFlight;
  }

  @http({ path: ARTIFACT_BLOB_SUBPATH, method: 'GET', scopes: ['user'], description: 'A published artifact by hash, ranged' })
  async getBlob(request: Request, context: Context): Promise<Response> {
    return this.serve(request, context, false);
  }

  @http({ path: ARTIFACT_BLOB_SUBPATH, method: 'HEAD', scopes: ['user'], description: 'A published artifact by hash, headers only' })
  async headBlob(request: Request, context: Context): Promise<Response> {
    return this.serve(request, context, true);
  }

  private async serve(request: Request, context: Context, headOnly: boolean): Promise<Response> {
    const url = new URL(request.url);
    const digest = url.pathname.slice(url.pathname.lastIndexOf('/') + 1);
    const sha256 = parseDigestName(decodeURIComponent(digest));
    if (!sha256) return json(404, { error: { code: 'not_found', message: 'Blobs are named sha256:<64 hex>' } });

    const auth = context.auth as { authenticated?: boolean; principals?: unknown } | undefined;
    const principals = Array.isArray(auth?.principals) ? (auth!.principals as unknown[]).filter((p): p is string => typeof p === 'string') : [];
    if (!auth?.authenticated || principals.length === 0) {
      return json(401, { error: { code: 'unauthorized', message: 'This blob is served to a caller whose signature this agent verified' } });
    }
    const task = request.headers.get(ARTIFACT_TASK_HEADER);
    const access = this.registry.access(sha256, principals, task);
    if (!access.ok) {
      // A blob this caller may not read and one that does not exist answer
      // the same, so the endpoint is not a hash oracle. The task refusals
      // are for a caller who is in the audience and only mis-named the task.
      if (access.reason === 'task_required' || access.reason === 'task_mismatch') {
        return json(403, { error: { code: access.reason, message: access.reason === 'task_required' ? `This set was published for a task; send it in ${ARTIFACT_TASK_HEADER}` : 'This set was published for a different task' } });
      }
      return json(404, { error: { code: 'not_found', message: 'No such blob for this caller' } });
    }

    const manifest = this.registry.getManifest(sha256);
    const blob = manifest ? null : this.registry.getBlob(sha256)!;
    const size = manifest ? manifest.canonical.byteLength : blob!.size;
    const mediaType = manifest ? ARTIFACT_SET_MEDIA_TYPE : blob!.mediaType;

    const headers: Record<string, string> = {
      'Content-Type': mediaType,
      'Accept-Ranges': 'bytes',
      'Repr-Digest': reprDigest(sha256),
      ETag: `"sha256:${sha256}"`,
      'Cache-Control': 'private, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
    };
    const range = parseRangeHeader(request.headers.get('range'), size);
    if (range.kind === 'unsatisfiable') {
      headers['Content-Range'] = `bytes */${size}`;
      return new Response(null, { status: 416, headers });
    }
    let status = 200;
    let start = 0;
    let end = size - 1;
    if (range.kind === 'single') {
      status = 206;
      start = range.start;
      end = range.end;
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
    }
    const length = size === 0 ? 0 : end - start + 1;
    headers['Content-Length'] = String(length);
    if (headOnly || length === 0) return new Response(null, { status, headers });

    if (this.inFlight >= this.maxTransfers) {
      return json(503, { error: { code: 'too_many_transfers', message: `This agent is carrying ${this.inFlight} transfers, its limit is ${this.maxTransfers}; retry shortly`, limit: this.maxTransfers } }, { 'Retry-After': '2' });
    }
    this.inFlight += 1;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      this.inFlight -= 1;
    };

    if (manifest) {
      const body = manifest.canonical.subarray(start, end + 1);
      release();
      return new Response(body as unknown as BodyInit, { status, headers });
    }
    const stream = this.openRange(blob!.path, start, end);
    stream.once('close', release);
    stream.once('error', release);
    return new Response(Readable.toWeb(stream) as ReadableStream<Uint8Array>, { status, headers });
  }
}
