/**
 * A set in the protocols an agent speaks: A2A parts and MCP resource links.
 */

import { describe, it, expect } from 'vitest';

import {
  artifactSetFromA2AParts,
  artifactSetToA2AParts,
  artifactSetToMcpResourceLinks,
  parseArtifactSet,
  ARTIFACT_SET_MEDIA_TYPE,
} from '../../../src/artifacts';

const H = (n: number) => n.toString(16).padStart(2, '0').repeat(32);
const BASE = 'https://agent.example/agents/flow';
const set = parseArtifactSet({
  kind: 'webagents.artifact-set/v1',
  createdAt: '2026-10-17T10:00:00Z',
  expiresAt: null,
  producer: { agent: BASE, task: 't1' },
  entries: [
    { path: 'gds/top.gds', size: 10, sha256: H(1), mediaType: 'application/octet-stream', locations: [{ kind: 'https', url: `${BASE}/artifacts/blobs/sha256:${H(1)}` }] },
    { path: 'report.txt', size: 3, sha256: H(2), mediaType: 'text/plain' },
  ],
});
const SHA = H(9);

describe('A2A mapping', () => {
  it('carries the manifest as a data part and every entry as a file part with its preferred URL', () => {
    const parts = artifactSetToA2AParts(set, SHA, { baseUrl: BASE, manifestUrl: `${BASE}/artifacts/blobs/sha256:${SHA}` });
    expect(parts[0].data).toBe(set);
    expect((parts[0].data as { kind: string }).kind).toBe('webagents.artifact-set/v1');
    expect(parts[0].mediaType).toBe(ARTIFACT_SET_MEDIA_TYPE);
    expect(parts[0].metadata).toEqual({ sha256: SHA, url: `${BASE}/artifacts/blobs/sha256:${SHA}` });
    expect(parts[1]).toEqual({ url: `${BASE}/artifacts/blobs/sha256:${H(1)}`, mediaType: 'application/octet-stream', filename: 'gds/top.gds', metadata: { sha256: H(1), size: 10 } });
    // No own location: built from the base URL.
    expect(parts[2].url).toBe(`${BASE}/artifacts/blobs/sha256:${H(2)}`);
    const found = artifactSetFromA2AParts(parts);
    expect(found?.sha256).toBe(SHA);
    expect(artifactSetFromA2AParts([{ text: 'hi' }])).toBeNull();
  });
});

describe('MCP mapping', () => {
  it('lists the manifest and each entry as resource links', () => {
    const links = artifactSetToMcpResourceLinks(set, SHA, { baseUrl: BASE });
    expect(links[0]).toMatchObject({ type: 'resource_link', uri: `${BASE}/artifacts/blobs/sha256:${SHA}`, mimeType: ARTIFACT_SET_MEDIA_TYPE });
    expect(links[1]).toEqual({ type: 'resource_link', uri: `${BASE}/artifacts/blobs/sha256:${H(1)}`, name: 'gds/top.gds', mimeType: 'application/octet-stream', size: 10 });
    expect(links).toHaveLength(3);
    // Without any URL nothing is linked (no silent made-up addresses).
    expect(artifactSetToMcpResourceLinks({ ...set, entries: [set.entries[1]] }, SHA)).toEqual([]);
  });
});
