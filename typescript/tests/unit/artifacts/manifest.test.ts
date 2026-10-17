/**
 * `webagents.artifact-set/v1`: the shared vectors (`python/tests/fixtures/
 * artifact_set/vectors.json`) pin the canonical bytes and the identity of
 * every valid manifest, and the refusal code of every hostile one. The Python
 * SDK runs the same file.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ARTIFACT_SET_LIMITS,
  ArtifactSetError,
  canonicalManifest,
  manifestHash,
  parseArtifactSet,
  readArtifactSet,
  chunkCount,
  chunkRange,
  digestName,
  parseDigestName,
} from '../../../src/artifacts';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VECTORS = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/artifact_set/vectors.json'), 'utf8')) as {
  valid: Array<{ name: string; manifest: unknown; canonical: string; sha256: string }>;
  hostile: Array<{ name: string; code: string; manifest?: unknown; depth?: number; generate?: { kind: string; count?: number; text?: string } }>;
};

function generated(spec: { kind: string; count?: number; text?: string }): unknown {
  if (spec.kind === 'entries') {
    const H = (n: number) => n.toString(16).padStart(8, '0').repeat(8);
    return {
      kind: 'webagents.artifact-set/v1',
      createdAt: '2026-10-17T10:00:00Z',
      expiresAt: null,
      producer: { agent: 'https://agent.example/agents/flow' },
      entries: Array.from({ length: spec.count! }, (_, i) => ({ path: `f/${i}`, size: 1, sha256: H(i), mediaType: 'text/plain' })),
    };
  }
  if (spec.kind === 'bytes') return new Uint8Array(spec.count!).fill(0x20);
  if (spec.kind === 'text') return spec.text;
  throw new Error(`unknown generate kind ${spec.kind}`);
}

describe('artifact-set vectors', () => {
  for (const v of VECTORS.valid) {
    it(`canonicalises and hashes "${v.name}" as the vectors say`, () => {
      const set = parseArtifactSet(v.manifest);
      expect(canonicalManifest(set)).toBe(v.canonical);
      expect(manifestHash(set)).toBe(v.sha256);
      // The canonical text parses back to the same set and the same identity.
      const again = readArtifactSet(new TextEncoder().encode(v.canonical));
      expect(again.sha256).toBe(v.sha256);
      expect(canonicalManifest(again.set)).toBe(v.canonical);
    });
  }

  for (const h of VECTORS.hostile) {
    it(`refuses "${h.name}" as ${h.code}`, () => {
      const input = h.generate ? generated(h.generate) : h.manifest;
      let caught: unknown;
      try {
        parseArtifactSet(input, h.depth !== undefined ? { depth: h.depth } : {});
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(ArtifactSetError);
      expect((caught as ArtifactSetError).code).toBe(h.code);
    });
  }
});

describe('chunk arithmetic and digest names', () => {
  it('counts and bounds chunks', () => {
    expect(chunkCount(0, 262144)).toBe(0);
    expect(chunkCount(1, 262144)).toBe(1);
    expect(chunkCount(262144, 262144)).toBe(1);
    expect(chunkCount(262145, 262144)).toBe(2);
    expect(chunkRange(0, 600000, 262144)).toEqual({ start: 0, end: 262143 });
    expect(chunkRange(2, 600000, 262144)).toEqual({ start: 524288, end: 599999 });
  });
  it('spells digests sha256:<hex> and reads nothing else', () => {
    const sha = 'ab'.repeat(32);
    expect(digestName(sha)).toBe(`sha256:${sha}`);
    expect(parseDigestName(`sha256:${sha}`)).toBe(sha);
    expect(parseDigestName(`sha256:${sha.toUpperCase()}`)).toBeNull();
    expect(parseDigestName(`md5:${sha}`)).toBeNull();
    expect(parseDigestName(sha)).toBeNull();
  });
  it('states its limits', () => {
    expect(ARTIFACT_SET_LIMITS.maxDepth).toBe(8);
    expect(ARTIFACT_SET_LIMITS.maxEntries).toBe(10000);
  });
});
