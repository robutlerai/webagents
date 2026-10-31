/**
 * CASFS core: the rules every CASFS reader and writer shares.
 *
 * A CASFS project is a Merkle tree of `casfs.dir/1` directory manifests over
 * content-addressed blobs. A manifest is canonical JSON (RFC 8785 for the
 * value space used here: sorted keys, no whitespace, safe integers only) and
 * is named by the SHA-256 of its bytes. Entries are files, folders, sealed
 * folders and mounts. Every name and every manifest read from anywhere is
 * treated as hostile input and refused by name (`Cas:*` codes), never
 * repaired.
 */

import { sha256 } from './sha256.js';

export type CasErrorCode =
  | 'Cas:NotFound'
  | 'Cas:PathInvalid'
  | 'Cas:PolicyInvalid'
  | 'Cas:ManifestInvalid'
  | 'Cas:HeadMoved'
  | 'Cas:PathConflict'
  | 'Cas:ChunkMismatch'
  | 'Cas:HashMismatch'
  | 'Cas:LocalOnly'
  | 'Cas:SourceOffline'
  | 'Cas:QuotaExceeded'
  | 'Cas:GrantScope'
  | 'Cas:GrantInvalid'
  | 'Cas:MissingBlob'
  | 'Cas:TooLarge'
  | 'Cas:IsADirectory'
  | 'Cas:NotADirectory'
  | 'Cas:Exists'
  | 'Cas:RangeInvalid'
  | 'Cas:Forbidden'
  | 'Cas:MountReadOnly'
  | 'Cas:MountSource'
  | 'Cas:MountCycle'
  | 'Cas:MountDepth'
  | 'Cas:NotAMount';

export class CasError extends Error {
  readonly code: CasErrorCode;
  readonly args: Record<string, unknown>;
  constructor(code: CasErrorCode, message: string, args: Record<string, unknown> = {}) {
    super(`${code}: ${message}`);
    this.name = 'CasError';
    this.code = code;
    this.args = args;
  }
}

export const DIR_KIND = 'casfs.dir/1';
export const RUN_KIND = 'casfs.run/1';
export const POLICIES = ['cas', 'both', 'local-served', 'local-only'] as const;
export type Policy = (typeof POLICIES)[number];
export const CHUNK_UNIT = 256 * 1024;
export const CHUNK_DEFAULT = 4 * 1024 * 1024;
export const CHUNK_MAX = 64 * 1024 * 1024;
export const INLINE_CHUNK_HASHES = 1024;
export const LIMITS = { maxNameBytes: 255, maxDepth: 64, maxEntries: 100_000, maxManifestBytes: 32 * 1024 * 1024 } as const;
/** Mounts nest at most this deep (a mount inside a mounted source counts one more). */
export const MAX_MOUNT_DEPTH = 4;

const SHA_RE = /^[0-9a-f]{64}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const PKG_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/;
export const isSha = (s: unknown): s is string => typeof s === 'string' && SHA_RE.test(s);
export const isPkgName = (s: unknown): s is string => typeof s === 'string' && PKG_RE.test(s);
export { sha256 };

export function canon(v: unknown): string {
  if (v === null || typeof v === 'boolean') return JSON.stringify(v);
  if (typeof v === 'number') {
    if (!Number.isSafeInteger(v)) throw new CasError('Cas:ManifestInvalid', `number ${v} is not a safe integer`);
    return String(v);
  }
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>;
    const keys = Object.keys(o).filter((k) => o[k] !== undefined).sort();
    return '{' + keys.map((k) => JSON.stringify(k) + ':' + canon(o[k])).join(',') + '}';
  }
  throw new CasError('Cas:ManifestInvalid', `value of type ${typeof v} cannot be canonical`);
}
export const canonBytes = (v: unknown): Uint8Array => new TextEncoder().encode(canon(v));

const enc = new TextEncoder();

export function checkName(name: unknown): string {
  if (typeof name !== 'string' || name.length === 0) throw new CasError('Cas:PathInvalid', 'empty name', { name });
  if (name !== name.normalize('NFC')) throw new CasError('Cas:PathInvalid', `name is not NFC: ${JSON.stringify(name)}`, { name });
  if (name === '.' || name === '..') throw new CasError('Cas:PathInvalid', `name ${name} is reserved`, { name });
  // eslint-disable-next-line no-control-regex
  if (/[/\\\u0000-\u001f\u007f]/.test(name)) throw new CasError('Cas:PathInvalid', `name has a separator or control character: ${JSON.stringify(name)}`, { name });
  if (enc.encode(name).length > LIMITS.maxNameBytes) throw new CasError('Cas:PathInvalid', `name longer than ${LIMITS.maxNameBytes} bytes`, { name });
  return name;
}

export function splitPath(p: unknown): string[] {
  if (typeof p !== 'string') throw new CasError('Cas:PathInvalid', 'path is not a string');
  const s = p.replace(/^\/+/, '');
  if (s === '') return [];
  const parts = s.split('/');
  if (parts.length > LIMITS.maxDepth) throw new CasError('Cas:PathInvalid', `path deeper than ${LIMITS.maxDepth}`, { path: p });
  for (const part of parts) checkName(part);
  return parts;
}
export const normPath = (p: unknown): string => splitPath(p).join('/');

export function checkPolicy(p: unknown): Policy | undefined {
  if (p === undefined) return undefined;
  if (typeof p !== 'string' || !(POLICIES as readonly string[]).includes(p)) throw new CasError('Cas:PolicyInvalid', `unknown policy ${JSON.stringify(p)}`, { policy: p });
  return p as Policy;
}
export function checkChunkSize(n: unknown): number {
  if (typeof n !== 'number' || !Number.isSafeInteger(n) || n < CHUNK_UNIT || n > CHUNK_MAX || n % CHUNK_UNIT) {
    throw new CasError('Cas:ManifestInvalid', `chunk size ${String(n)} must be a multiple of ${CHUNK_UNIT} between ${CHUNK_UNIT} and ${CHUNK_MAX}`);
  }
  return n;
}
export const chunkCount = (size: number, cs: number): number => (size === 0 ? 0 : Math.ceil(size / cs));

export interface ChunkSpec {
  size: number;
  hashes?: string[];
  index?: string;
}
export interface FileEntry {
  type: 'file';
  id?: string;
  size: number;
  sha256?: string;
  mediaType?: string;
  policy?: Policy;
  chunks?: ChunkSpec;
  growing?: boolean;
}
export interface DirEntry {
  type: 'dir';
  sha256: string;
  policy?: Policy;
}
export interface SealedEntry {
  type: 'sealed';
  keyId: string;
  seal: string;
}
/** Another project, or a published PDK, shown as a read-only folder pinned to one root of the source. */
export type MountSource = { kind: 'project'; id: string } | { kind: 'pdk'; id: string; pkg: string; version: string };
export interface MountEntry {
  type: 'mount';
  readOnly: true;
  root: string;
  source: MountSource;
}
export type Entry = FileEntry | DirEntry | SealedEntry | MountEntry;
export interface DirManifest {
  kind: typeof DIR_KIND;
  entries: Record<string, Entry>;
}

export function checkMountSource(v: unknown): MountSource {
  const s = v as Record<string, unknown> | null;
  if (!s || typeof s !== 'object' || Array.isArray(s)) throw new CasError('Cas:ManifestInvalid', 'a mount source is an object');
  if (typeof s.id !== 'string' || !UUID_RE.test(s.id)) throw new CasError('Cas:ManifestInvalid', 'a mount source names its project id (lower-case uuid)');
  if (s.kind === 'project') {
    if (Object.keys(s).some((k) => k !== 'kind' && k !== 'id')) throw new CasError('Cas:ManifestInvalid', 'a project mount source is { kind, id }');
    return { kind: 'project', id: s.id };
  }
  if (s.kind === 'pdk') {
    if (Object.keys(s).some((k) => !['kind', 'id', 'pkg', 'version'].includes(k))) throw new CasError('Cas:ManifestInvalid', 'a pdk mount source is { kind, id, pkg, version }');
    if (!isPkgName(s.pkg) || !isPkgName(s.version)) throw new CasError('Cas:ManifestInvalid', 'pkg and version: 1 to 64 of A-Z a-z 0-9 . _ + -, a letter or digit first');
    return { kind: 'pdk', id: s.id, pkg: s.pkg, version: s.version };
  }
  throw new CasError('Cas:ManifestInvalid', "a mount source kind is 'project' or 'pdk'");
}
/** The canonical path of a PDK mount: one per package version, shared by every cell that uses it. */
export const pdkMountPath = (pkg: string, version: string): string => `pdk/${pkg}@${version}`;
export const sameMount = (a: MountEntry, b: MountEntry): boolean => a.root === b.root && canon(a.source) === canon(b.source);

const KEY_ID_RE = /^[0-9a-f]{16,64}$/;

export function checkDirManifest(m: unknown): DirManifest {
  const mm = m as { kind?: unknown; entries?: unknown } | null;
  if (!mm || typeof mm !== 'object' || mm.kind !== DIR_KIND || !mm.entries || typeof mm.entries !== 'object' || Array.isArray(mm.entries) || Object.keys(mm).some((k) => k !== 'kind' && k !== 'entries')) {
    throw new CasError('Cas:ManifestInvalid', 'not a casfs.dir/1 manifest');
  }
  const entries = mm.entries as Record<string, Record<string, unknown> | null>;
  const names = Object.keys(entries);
  if (names.length > LIMITS.maxEntries) throw new CasError('Cas:ManifestInvalid', `more than ${LIMITS.maxEntries} entries`);
  for (const n of names) {
    checkName(n);
    const e = entries[n];
    if (!e || typeof e !== 'object' || Array.isArray(e)) throw new CasError('Cas:ManifestInvalid', `entry ${n} is not an object`);
    if (e.type === 'sealed') {
      if (typeof e.keyId !== 'string' || !KEY_ID_RE.test(e.keyId) || !isSha(e.seal) || Object.keys(e).some((k) => !['type', 'keyId', 'seal'].includes(k))) {
        throw new CasError('Cas:ManifestInvalid', `sealed entry ${n} carries only keyId and seal`);
      }
      continue;
    }
    if (e.type === 'mount') {
      if (Object.keys(e).some((k) => !['type', 'readOnly', 'root', 'source'].includes(k))) throw new CasError('Cas:ManifestInvalid', `mount ${n} carries more than type, readOnly, root and source`);
      if (e.readOnly !== true) throw new CasError('Cas:ManifestInvalid', `mount ${n}: readOnly is true`);
      if (!isSha(e.root)) throw new CasError('Cas:ManifestInvalid', `mount ${n} has no root`);
      checkMountSource(e.source);
      continue;
    }
    if (e.type !== 'file' && e.type !== 'dir') throw new CasError('Cas:ManifestInvalid', `entry ${n} has no valid type`);
    checkPolicy(e.policy);
    if (e.type === 'dir') {
      if (!isSha(e.sha256)) throw new CasError('Cas:ManifestInvalid', `entry ${n} has no sha256`);
      continue;
    }
    if (!isSha(e.sha256) && !e.growing) throw new CasError('Cas:ManifestInvalid', `entry ${n} has no sha256`);
    if (typeof e.size !== 'number' || !Number.isSafeInteger(e.size) || e.size < 0) throw new CasError('Cas:ManifestInvalid', `entry ${n} has a bad size`);
    if (e.chunks !== undefined) {
      const c = e.chunks as Record<string, unknown>;
      const k = chunkCount(e.size, checkChunkSize(c?.size));
      if (c.hashes !== undefined) {
        if (!Array.isArray(c.hashes) || c.hashes.length !== k || c.hashes.length > INLINE_CHUNK_HASHES || !c.hashes.every(isSha)) throw new CasError('Cas:ManifestInvalid', `entry ${n}: ${k} chunk hashes expected`);
      } else if (!isSha(c.index)) throw new CasError('Cas:ManifestInvalid', `entry ${n}: chunks need hashes or an index`);
    }
  }
  return m as DirManifest;
}

/** Parse manifest bytes that must hash to `sha` and be canonical. */
export function parseDirManifest(sha: string, bytes: Uint8Array): DirManifest {
  if (bytes.length > LIMITS.maxManifestBytes) throw new CasError('Cas:ManifestInvalid', `manifest ${sha} is too large`);
  if (sha256(bytes) !== sha) throw new CasError('Cas:HashMismatch', `directory manifest does not hash to ${sha}`, { sha });
  let m: unknown;
  try {
    m = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    throw new CasError('Cas:ManifestInvalid', `manifest ${sha} is not JSON`);
  }
  const dm = checkDirManifest(m);
  if (canon(dm) !== new TextDecoder().decode(bytes)) throw new CasError('Cas:ManifestInvalid', `manifest ${sha} is not in canonical form`);
  return dm;
}

export function concat(parts: Uint8Array[], total?: number): Uint8Array {
  const out = new Uint8Array(total ?? parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}
