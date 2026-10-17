/**
 * What this agent serves by hash: the blobs it published (a path on disk) and
 * the manifests that name them (their canonical bytes), with who may read them.
 *
 * Access is by caller, never by link: a manifest is published to an audience
 * of principals (`agent:<url>`, the form the access skill records for a
 * caller whose signature it verified), or to every verified caller (`'*'`).
 * A blob is readable by the union of the audiences of the manifests that
 * list it. A manifest published for a task may require the reader to name it.
 */

import type { ArtifactSet, ChunkList } from './manifest';

export type Audience = '*' | ReadonlySet<string>;

export interface RegisteredBlob {
  sha256: string;
  path: string;
  size: number;
  mediaType: string;
  chunks?: ChunkList;
  /** Hashes of the manifests that list this blob. */
  manifests: Set<string>;
}

export interface RegisteredManifest {
  sha256: string;
  canonical: Uint8Array;
  set: ArtifactSet;
  audience: Audience;
  task?: string;
}

/** `agent:<url>` for a principal given with or without the prefix. */
export function agentPrincipal(value: string): string {
  return value.startsWith('agent:') ? value : `agent:${value}`;
}

export class ArtifactRegistry {
  private readonly blobs = new Map<string, RegisteredBlob>();
  private readonly manifests = new Map<string, RegisteredManifest>();

  addBlob(blob: Omit<RegisteredBlob, 'manifests'>): RegisteredBlob {
    const existing = this.blobs.get(blob.sha256);
    if (existing) {
      // Same bytes, perhaps another path or type: keep the first path, the first type.
      return existing;
    }
    const entry: RegisteredBlob = { ...blob, manifests: new Set() };
    this.blobs.set(blob.sha256, entry);
    return entry;
  }

  addManifest(manifest: Omit<RegisteredManifest, 'audience'> & { audience?: Audience | Iterable<string> }): RegisteredManifest {
    const audience: Audience = manifest.audience === undefined || manifest.audience === '*'
      ? '*'
      : new Set(Array.from(manifest.audience as Iterable<string>, agentPrincipal));
    const entry: RegisteredManifest = { sha256: manifest.sha256, canonical: manifest.canonical, set: manifest.set, audience, ...(manifest.task ? { task: manifest.task } : {}) };
    this.manifests.set(manifest.sha256, entry);
    for (const e of manifest.set.entries) this.blobs.get(e.sha256)?.manifests.add(manifest.sha256);
    return entry;
  }

  getBlob(sha256: string): RegisteredBlob | undefined {
    return this.blobs.get(sha256);
  }

  getManifest(sha256: string): RegisteredManifest | undefined {
    return this.manifests.get(sha256);
  }

  /** Let one more principal read a published manifest and its blobs. */
  grant(manifestSha256: string, principal: string): void {
    const m = this.manifests.get(manifestSha256);
    if (!m) throw new Error(`no manifest ${manifestSha256}`);
    if (m.audience === '*') return;
    (m.audience as Set<string>).add(agentPrincipal(principal));
  }

  /** Forget a manifest; blobs no other manifest lists go with it. */
  remove(manifestSha256: string): void {
    const m = this.manifests.get(manifestSha256);
    if (!m) return;
    this.manifests.delete(manifestSha256);
    for (const e of m.set.entries) {
      const blob = this.blobs.get(e.sha256);
      if (!blob) continue;
      blob.manifests.delete(manifestSha256);
      if (blob.manifests.size === 0) this.blobs.delete(e.sha256);
    }
  }

  /**
   * Whether a caller holding `principals` may read `sha256` (a blob or a
   * manifest), and the task the reader must name if any manifest involved
   * asks for one. `null` means not readable by this caller.
   */
  access(sha256: string, principals: readonly string[], task: string | null): { ok: true } | { ok: false; reason: 'unknown' | 'forbidden' | 'task_required' | 'task_mismatch' } {
    const manifest = this.manifests.get(sha256);
    const candidates: RegisteredManifest[] = manifest ? [manifest] : [];
    if (!manifest) {
      const blob = this.blobs.get(sha256);
      if (!blob) return { ok: false, reason: 'unknown' };
      for (const h of blob.manifests) {
        const m = this.manifests.get(h);
        if (m) candidates.push(m);
      }
      if (candidates.length === 0) return { ok: false, reason: 'unknown' };
    }
    let reason: 'forbidden' | 'task_required' | 'task_mismatch' = 'forbidden';
    for (const m of candidates) {
      const inAudience = m.audience === '*' || principals.some((p) => (m.audience as ReadonlySet<string>).has(p));
      if (!inAudience) continue;
      if (m.task !== undefined) {
        if (task === null) { reason = 'task_required'; continue; }
        if (task !== m.task) { reason = 'task_mismatch'; continue; }
      }
      return { ok: true };
    }
    return { ok: false, reason };
  }

  listManifests(): RegisteredManifest[] {
    return Array.from(this.manifests.values());
  }

  listBlobs(): RegisteredBlob[] {
    return Array.from(this.blobs.values());
  }
}
