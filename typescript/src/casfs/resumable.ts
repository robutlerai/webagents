/**
 * Resumable upload of one large blob to a CASFS project on the platform.
 *
 * `POST /api/casfs/projects/<id>/uploads` opens the upload with the file's SHA-256, its size and a chunk list
 * (the SHA-256 of every `chunkSize` slice). Bytes then go up one chunk per request: tus `PATCH` when the
 * answer has no `sessionUri`, `PUT` with `Content-Range` to the storage session when it has one (followed by
 * `POST <location>/complete`). The server checks every chunk against the list and the whole file against the
 * hash, then keeps the bytes as a blob of the project. A failed request asks the server for its offset and
 * continues from there; a chunk refused by its hash stops with `Cas:ChunkMismatch`.
 */

import { CasError, type CasErrorCode } from './core.js';

export const RESUMABLE_CHUNK = 8 * 1024 * 1024;
const ATTEMPTS = 5;

export interface ResumableSource {
  size: number;
  slice(start: number, end: number): Promise<Uint8Array>;
}

export interface ResumableTransport {
  /** Fetch a path on the platform with the caller's credentials. */
  platform(path: string, init?: RequestInit): Promise<Response>;
  /** Fetch an absolute storage session URL (no platform credentials). */
  external(url: string, init?: RequestInit): Promise<Response>;
}

const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, '0')).join('');

async function body(res: Response): Promise<Record<string, unknown>> {
  const j = (await res.json().catch(() => null)) as { ok?: boolean; error?: unknown } | null;
  if (!res.ok || !j || j.ok === false) {
    const e = (j?.error && typeof j.error === 'object' ? j.error : {}) as { code?: string; message?: string; args?: Record<string, unknown> };
    const code = (e.code ?? (res.status === 413 ? 'Cas:TooLarge' : 'Cas:SourceOffline')) as CasErrorCode;
    throw new CasError(code, (e.message ?? (typeof j?.error === 'string' ? j.error : `HTTP ${res.status}`)).replace(/^Cas:[A-Za-z]+: /, ''), e.args ?? {});
  }
  return j as Record<string, unknown>;
}

export async function uploadResumable(
  t: ResumableTransport,
  projectId: string,
  sha: string,
  src: ResumableSource,
  o: { chunkSize?: number; pollMs?: number; maxPolls?: number; onProgress?: (sent: number, size: number) => void } = {},
): Promise<void> {
  const cs = o.chunkSize ?? RESUMABLE_CHUNK;
  const size = src.size;
  const hashes: string[] = [];
  for (let at = 0; at < size; at += cs) hashes.push(hex(await globalThis.crypto.subtle.digest('SHA-256', (await src.slice(at, Math.min(size, at + cs))) as BufferSource)));
  const opened = await body(
    await t.platform(`/api/casfs/projects/${encodeURIComponent(projectId)}/uploads`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ sha256: sha, size, chunkList: { chunkSize: cs, hashes } }),
    }),
  );
  const location = String(opened.location);
  if (!/^\/api\/content\/uploads\/[0-9a-f-]{36}$/.test(location)) throw new CasError('Cas:SourceOffline', 'the upload route answered no location');
  const session = typeof opened.sessionUri === 'string' ? opened.sessionUri : null;

  const serverOffset = async (): Promise<number> => {
    if (!session) {
      const r = await t.platform(location, { method: 'HEAD', headers: { 'Tus-Resumable': '1.0.0' } });
      const n = Number(r.headers.get('upload-offset'));
      if (!r.ok || !Number.isSafeInteger(n)) throw new CasError('Cas:SourceOffline', `the upload cannot be resumed (HTTP ${r.status})`);
      return n;
    }
    const r = await t.external(session, { method: 'PUT', headers: { 'content-range': `bytes */${size}` } });
    if (r.status === 200 || r.status === 201) return size;
    const range = r.headers.get('range');
    return range ? Number(range.split('-')[1]) + 1 : 0;
  };

  let offset = 0;
  let failures = 0;
  while (offset < size) {
    const end = Math.min(size, offset + cs);
    try {
      const chunk = (await src.slice(offset, end)) as unknown as BodyInit;
      const r = session
        ? await t.external(session, { method: 'PUT', headers: { 'content-range': `bytes ${offset}-${end - 1}/${size}` }, body: chunk })
        : await t.platform(location, { method: 'PATCH', headers: { 'Tus-Resumable': '1.0.0', 'Upload-Offset': String(offset), 'Content-Type': 'application/offset+octet-stream' }, body: chunk });
      const ok = session ? r.status === 308 || r.status === 200 || r.status === 201 : r.status === 204;
      if (!ok) {
        if (r.status === 460 || r.status === 422) throw new CasError('Cas:ChunkMismatch', `chunk at ${offset} was refused by its hash`, { offset });
        if (r.status === 409 || r.status === 308 || r.status >= 500) throw new Error(`HTTP ${r.status}`);
        await body(r);
      }
      offset = end;
      failures = 0;
      o.onProgress?.(offset, size);
    } catch (err) {
      if (err instanceof CasError) throw err;
      if (++failures >= ATTEMPTS) throw new CasError('Cas:SourceOffline', `the upload stopped at ${offset} of ${size} bytes after ${ATTEMPTS} tries`, { offset, size });
      offset = await serverOffset().catch(() => offset);
    }
  }
  if (session) await body(await t.platform(`${location}/complete`, { method: 'POST' }));
  const maxPolls = o.maxPolls ?? 600;
  for (let i = 0; i < maxPolls; i++) {
    const st = await body(await t.platform(location));
    if (st.status === 'committed') return;
    if (st.status === 'failed' || st.status === 'aborted') throw new CasError('Cas:HashMismatch', String(st.error ?? 'the server refused the file'), {});
    if (o.pollMs !== 0) await new Promise((res) => setTimeout(res, Math.min(5000, (o.pollMs ?? 500) * (1 + i / 10))));
  }
  throw new CasError('Cas:SourceOffline', 'the server did not finish checking the file in time; it may still commit', {});
}
