/**
 * CASFS: project data as a tree of content-addressed files, read by path and
 * verified by hash, with mounts of other projects and published PDKs.
 *
 * Node: `CasFs.platform({ baseUrl, token, cache: new DirChunkCache() })`, and
 * `LocalDirBackend` / `LocalHeads` for a project kept in a directory (from
 * `webagents/casfs/local`). Browser: `OpfsBackend` (from `webagents/casfs/browser`).
 */

export * from './core.js';
export * from './backends.js';
export * from './project.js';
export * from './casfs.js';
export * from './resumable.js';
export { Sha256, sha256 as sha256Hex, toHex, fromHex } from './sha256.js';
