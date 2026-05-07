/**
 * Function manifest validator unit tests.
 */

import { describe, it, expect } from 'vitest';
import {
  validateManifest,
  parseFrontmatter,
} from '../../../../src/skills/functions/validator.js';
import type { FunctionManifest } from '../../../../src/skills/functions/manifest.js';

describe('validateManifest', () => {
  it('accepts a minimal js-v1 manifest', () => {
    const m: FunctionManifest = { runtime: 'js-v1', entrypoint: 'handler' };
    const r = validateManifest(m, { cloud: true, functionName: 'fn' });
    expect(r.ok).toBe(true);
  });

  it('rejects unknown runtime', () => {
    const m = { runtime: 'foo-v9' } as unknown as FunctionManifest;
    const r = validateManifest(m, { cloud: true, functionName: 'fn' });
    expect(r.ok).toBe(false);
    expect(r.errors?.[0]?.code).toBe('RUNTIME_UNKNOWN');
  });

  it('rejects wasm-v1 in v1 with RUNTIME_DISABLED', () => {
    const m: FunctionManifest = { runtime: 'wasm-v1' };
    const r = validateManifest(m, { cloud: true, functionName: 'fn' });
    expect(r.ok).toBe(false);
    expect(r.errors?.[0]?.code).toBe('RUNTIME_DISABLED');
  });

  it('rejects WS-style endpoints in v1', () => {
    const m = { runtime: 'js-v1', type: 'websocket' } as unknown as FunctionManifest;
    const r = validateManifest(m, { cloud: true, functionName: 'fn' });
    expect(r.ok).toBe(false);
    expect(r.errors?.[0]?.code).toBe('WS_NOT_YET_SUPPORTED');
  });

  it('rejects file:// codeRefs in cloud mode', () => {
    const m: FunctionManifest = {
      runtime: 'js-v1',
      code: { kind: 'file', path: '/etc/passwd' },
    };
    const r = validateManifest(m, { cloud: true, functionName: 'fn' });
    expect(r.ok).toBe(false);
  });
});

describe('parseFrontmatter', () => {
  it('extracts @robutler-function frontmatter from JS source', () => {
    const src = `
/**
 * @robutler-function
 * @runtime js-v1
 * @entrypoint handler
 * @description Hello
 */
export default async function handler(ctx) { return 'hi'; }
`;
    const m = parseFrontmatter(src);
    expect(m?.runtime).toBe('js-v1');
    expect(m?.entrypoint).toBe('handler');
  });

  it('returns null when frontmatter is absent', () => {
    expect(parseFrontmatter('export default async function handler() {}')).toBeNull();
  });

  it('extracts YAML frontmatter from JS source', () => {
    const src = `
/**
 * @robutler-function
 * runtime: js-v1
 * entrypoint: handler
 * description: Hello
 * permissions:
 *   fetch:
 *     - https://api.openai.com/*
 *   secrets:
 *     - OPENAI_KEY
 * limits:
 *   wallMs: 5000
 */
export default async function handler(ctx) { return 'hi'; }
`;
    const m = parseFrontmatter(src);
    expect(m?.runtime).toBe('js-v1');
    expect(m?.entrypoint).toBe('handler');
    expect(m?.description).toBe('Hello');
    expect((m?.permissions as { fetch?: string[] })?.fetch).toEqual(['https://api.openai.com/*']);
    expect((m?.permissions as { secrets?: string[] })?.secrets).toEqual(['OPENAI_KEY']);
    expect((m?.limits as { wallMs?: number })?.wallMs).toBe(5000);
  });

  it('extracts JSON frontmatter from JS source', () => {
    const src = `
/**
 * @robutler-function
 * { "runtime": "js-v1", "entrypoint": "handler" }
 */
`;
    const m = parseFrontmatter(src);
    expect(m?.runtime).toBe('js-v1');
    expect(m?.entrypoint).toBe('handler');
  });
});
