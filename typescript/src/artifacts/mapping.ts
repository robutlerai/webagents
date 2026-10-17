/**
 * How a published set rides in the protocols an agent already speaks.
 *
 *   A2A   a result carries the manifest as a data part whose `data.kind` is
 *         `webagents.artifact-set/v1`, plus one file part per entry with the
 *         entry's preferred `https` location as `url`, its `mediaType` and its
 *         path as `filename`. A plain A2A client follows the URLs; a
 *         webagents client reads the data part and verifies against it.
 *   MCP   resource links (`type: "resource_link"`) play the same role: one for
 *         the manifest, one per entry, each with `uri`, `name`, `mimeType`
 *         and `size`.
 */

import type { Part } from '../skills/transport/a2a/types';
import { ARTIFACT_SET_MEDIA_TYPE, type ArtifactEntry, type ArtifactSet } from './manifest';
import { blobUrl } from './publish';

export interface McpResourceLink {
  type: 'resource_link';
  uri: string;
  name: string;
  mimeType?: string;
  description?: string;
  size?: number;
}

/** The entry's first `https` location, else the set's, else one built from `baseUrl`. */
export function preferredUrl(set: ArtifactSet, entry: ArtifactEntry, baseUrl?: string): string | undefined {
  const own = entry.locations?.find((l) => l.kind === 'https' && l.url)?.url;
  if (own) return own;
  const shared = set.locations?.find((l) => l.kind === 'https' && l.url)?.url;
  if (shared) return shared;
  return baseUrl ? blobUrl(baseUrl, entry.sha256) : undefined;
}

/** A2A parts for a result that returns `set` (whose canonical hash is `sha256`). */
export function artifactSetToA2AParts(set: ArtifactSet, sha256: string, options: { baseUrl?: string; manifestUrl?: string } = {}): Part[] {
  const parts: Part[] = [
    {
      data: set as unknown as Record<string, unknown>,
      mediaType: ARTIFACT_SET_MEDIA_TYPE,
      metadata: { sha256, ...(options.manifestUrl ? { url: options.manifestUrl } : {}) },
    },
  ];
  for (const entry of set.entries) {
    const url = preferredUrl(set, entry, options.baseUrl);
    if (!url) continue;
    parts.push({ url, mediaType: entry.mediaType, filename: entry.path, metadata: { sha256: entry.sha256, size: entry.size } });
  }
  return parts;
}

/** MCP resource links for `set`: the manifest first, then every entry with a URL. */
export function artifactSetToMcpResourceLinks(set: ArtifactSet, sha256: string, options: { baseUrl?: string; manifestUrl?: string } = {}): McpResourceLink[] {
  const links: McpResourceLink[] = [];
  const manifestUri = options.manifestUrl ?? (options.baseUrl ? blobUrl(options.baseUrl, sha256) : undefined);
  if (manifestUri) {
    links.push({ type: 'resource_link', uri: manifestUri, name: `artifact-set ${sha256.slice(0, 12)}`, mimeType: ARTIFACT_SET_MEDIA_TYPE, description: `${set.entries.length} files from ${set.producer.agent}` });
  }
  for (const entry of set.entries) {
    const uri = preferredUrl(set, entry, options.baseUrl);
    if (!uri) continue;
    links.push({ type: 'resource_link', uri, name: entry.path, mimeType: entry.mediaType, size: entry.size });
  }
  return links;
}

/** The manifest a list of A2A parts carries, if one does. */
export function artifactSetFromA2AParts(parts: readonly Part[]): { data: unknown; sha256?: string } | null {
  for (const part of parts) {
    const data = part.data as { kind?: unknown } | undefined;
    if (data && typeof data === 'object' && data.kind === 'webagents.artifact-set/v1') {
      const sha256 = part.metadata?.sha256;
      return { data, ...(typeof sha256 === 'string' ? { sha256 } : {}) };
    }
  }
  return null;
}
