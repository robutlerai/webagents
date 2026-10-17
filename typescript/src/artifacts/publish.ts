/**
 * `files.publish(folder)`: walk a folder, hash every regular file once
 * (streaming), write the manifest, register the blobs for serving, and return
 * the manifest with its hash.
 *
 * Symbolic links are refused rather than followed, so a published folder can
 * never reach outside itself. Hidden files are included: the folder is the
 * producer's own output, and a reader that wants less filters by path.
 */

import { readdir, lstat } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';

import { hashFile } from './hash';
import {
  ARTIFACT_SET_KIND,
  ARTIFACT_SET_LIMITS,
  CHUNK_SIZE_DEFAULT,
  canonicalManifestBytes,
  digestName,
  isValidChunkSize,
  manifestHash,
  parseArtifactSet,
  validateEntryPath,
  type ArtifactEntry,
  type ArtifactLocation,
  type ArtifactProducer,
  type ArtifactSet,
} from './manifest';
import { ArtifactRegistry, type Audience } from './registry';

export interface PublishOptions {
  /** Who produced the set: the agent URL, and the task when there is one. */
  producer: ArtifactProducer;
  /** Where the blobs are registered for serving. */
  registry: ArtifactRegistry;
  /**
   * The agent's public base URL. With it every entry gets an `https`
   * location `<baseUrl>/artifacts/blobs/sha256:<hex>`, and the manifest is
   * reachable at the same path by its own hash.
   */
  baseUrl?: string;
  /** Further locations every entry carries after the `https` one (a `relay`, a store). */
  locations?: ArtifactLocation[];
  /** Who may read: principals (`agent:<url>` or bare URLs), or `'*'` for any verified caller (the default). */
  audience?: Audience | Iterable<string>;
  expiresAt?: Date | string | null;
  chunkSize?: number;
  /** The media type for a file, by its relative path; `application/octet-stream` when it answers nothing. */
  mediaType?: (relativePath: string) => string | undefined;
  /** Hand-set `createdAt` (tests); defaults to now. */
  createdAt?: Date | string;
  /** Accept a plaintext `http` base URL beyond loopback (a private network in development). */
  allowHttp?: boolean;
}

export interface PublishedSet {
  set: ArtifactSet;
  sha256: string;
  canonical: Uint8Array;
  /** The manifest's own `https` location when `baseUrl` was given. */
  manifestUrl?: string;
  /** Files that were skipped and why (never silent). */
  skipped: Array<{ path: string; reason: 'symlink' | 'not_a_file' }>;
}

const DEFAULT_MEDIA_TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.txt': 'text/plain',
  '.md': 'text/markdown',
  '.csv': 'text/csv',
  '.log': 'text/plain',
  '.html': 'text/html',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.gz': 'application/gzip',
  '.zip': 'application/zip',
  '.gds': 'application/octet-stream',
  '.def': 'text/plain',
};

export function defaultMediaType(relativePath: string): string {
  const dot = relativePath.lastIndexOf('.');
  const ext = dot >= 0 ? relativePath.slice(dot).toLowerCase() : '';
  return DEFAULT_MEDIA_TYPES[ext] ?? 'application/octet-stream';
}

/** `https` location for a hash under an agent base URL. */
export function blobUrl(baseUrl: string, sha256: string): string {
  return `${baseUrl.replace(/\/+$/, '')}/artifacts/blobs/${digestName(sha256)}`;
}

async function walk(root: string, dir: string, out: string[], skipped: PublishedSet['skipped']): Promise<void> {
  const names = (await readdir(dir)).sort();
  for (const name of names) {
    const full = join(dir, name);
    const st = await lstat(full);
    const rel = relative(root, full).split(sep).join('/');
    if (st.isSymbolicLink()) {
      skipped.push({ path: rel, reason: 'symlink' });
      continue;
    }
    if (st.isDirectory()) {
      await walk(root, full, out, skipped);
      continue;
    }
    if (!st.isFile()) {
      skipped.push({ path: rel, reason: 'not_a_file' });
      continue;
    }
    out.push(full);
    if (out.length > ARTIFACT_SET_LIMITS.maxEntries) {
      throw new Error(`the folder has more than ${ARTIFACT_SET_LIMITS.maxEntries} files; a set is limited to that many entries`);
    }
  }
}

function iso(value: Date | string | null | undefined, fallback: () => string | null): string | null {
  if (value === null) return null;
  if (value === undefined) return fallback();
  return value instanceof Date ? value.toISOString() : value;
}

/** Publish a folder. See the file comment. */
export async function publishFolder(folder: string, options: PublishOptions): Promise<PublishedSet> {
  const chunkSize = options.chunkSize ?? CHUNK_SIZE_DEFAULT;
  if (!isValidChunkSize(chunkSize)) throw new Error(`invalid chunk size ${chunkSize}`);
  const root = folder;
  const files: string[] = [];
  const skipped: PublishedSet['skipped'] = [];
  await walk(root, root, files, skipped);

  const entries: ArtifactEntry[] = [];
  const hashed: Array<{ full: string; rel: string; sha256: string; size: number; chunks?: ArtifactEntry['chunks']; mediaType: string }> = [];
  for (const full of files) {
    const rel = validateEntryPath(relative(root, full).split(sep).join('/'), entries.length);
    const h = await hashFile(full, chunkSize);
    const mediaType = options.mediaType?.(rel) ?? defaultMediaType(rel);
    const entry: ArtifactEntry = { path: rel, size: h.size, sha256: h.sha256, mediaType };
    if (h.chunks) entry.chunks = h.chunks;
    const locations: ArtifactLocation[] = [];
    if (options.baseUrl) locations.push({ kind: 'https', url: blobUrl(options.baseUrl, h.sha256) });
    if (options.locations) locations.push(...options.locations);
    if (locations.length) entry.locations = locations;
    entries.push(entry);
    hashed.push({ full, rel, sha256: h.sha256, size: h.size, chunks: h.chunks, mediaType });
  }

  const set: ArtifactSet = {
    kind: ARTIFACT_SET_KIND,
    createdAt: iso(options.createdAt, () => new Date().toISOString()) as string,
    expiresAt: iso(options.expiresAt, () => null),
    producer: options.producer,
    entries,
  };
  // The writer holds itself to the reader's rules.
  parseArtifactSet(set, { allowHttp: options.allowHttp });
  const canonical = canonicalManifestBytes(set);
  const sha256 = manifestHash(set);

  for (const h of hashed) {
    options.registry.addBlob({ sha256: h.sha256, path: h.full, size: h.size, mediaType: h.mediaType, ...(h.chunks ? { chunks: h.chunks } : {}) });
  }
  options.registry.addManifest({ sha256, canonical, set, audience: options.audience, task: options.producer.task });

  return {
    set,
    sha256,
    canonical,
    ...(options.baseUrl ? { manifestUrl: blobUrl(options.baseUrl, sha256) } : {}),
    skipped,
  };
}

/** The `files` namespace: `files.publish(folder, options)`. */
export const files = {
  publish: publishFolder,
};
