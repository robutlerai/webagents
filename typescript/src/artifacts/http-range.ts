/**
 * RFC 9110 single-range parsing and the RFC 9530 `Repr-Digest` field, shared
 * by the producer endpoint and the client.
 */

export type ParsedRange =
  | { kind: 'none' }
  | { kind: 'single'; start: number; end: number }
  | { kind: 'unsatisfiable' };

/**
 * One `bytes=` range against a representation of `size` bytes. Several
 * ranges or another unit read as `none` (the whole representation), which is
 * always a correct answer to a range request.
 */
export function parseRangeHeader(header: string | null | undefined, size: number): ParsedRange {
  if (!header) return { kind: 'none' };
  const m = /^\s*bytes\s*=\s*([^,]+?)\s*$/i.exec(header);
  if (!m) return { kind: 'none' };
  const spec = /^(\d*)\s*-\s*(\d*)$/.exec(m[1]);
  if (!spec || (spec[1] === '' && spec[2] === '')) return { kind: 'none' };
  if (size <= 0) return { kind: 'unsatisfiable' };
  if (spec[1] === '') {
    const n = Number(spec[2]);
    if (!Number.isSafeInteger(n) || n <= 0) return { kind: 'unsatisfiable' };
    return { kind: 'single', start: Math.max(0, size - n), end: size - 1 };
  }
  const start = Number(spec[1]);
  if (!Number.isSafeInteger(start)) return { kind: 'none' };
  if (start >= size) return { kind: 'unsatisfiable' };
  let end = spec[2] === '' ? size - 1 : Number(spec[2]);
  if (!Number.isSafeInteger(end)) return { kind: 'none' };
  if (end < start) return { kind: 'none' };
  if (end >= size) end = size - 1;
  return { kind: 'single', start, end };
}

/** `Repr-Digest: sha-256=:<base64>:` for a representation whose SHA-256 hex is given. */
export function reprDigest(sha256Hex: string): string {
  return `sha-256=:${Buffer.from(sha256Hex, 'hex').toString('base64')}:`;
}

/** The hex SHA-256 a `Repr-Digest` field names, or null when it names none this reader understands. */
export function parseReprDigest(header: string | null | undefined): string | null {
  if (!header) return null;
  const m = /sha-256=:([A-Za-z0-9+/=]+):/.exec(header);
  if (!m) return null;
  const bytes = Buffer.from(m[1], 'base64');
  return bytes.length === 32 ? bytes.toString('hex') : null;
}

/** `Content-Range: bytes a-b/size` parsed, or null. */
export function parseContentRange(header: string | null | undefined): { start: number; end: number; size: number } | null {
  if (!header) return null;
  const m = /^\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)\s*$/i.exec(header);
  if (!m) return null;
  const start = Number(m[1]);
  const end = Number(m[2]);
  const size = m[3] === '*' ? -1 : Number(m[3]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end < start) return null;
  return { start, end, size };
}
