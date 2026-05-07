/**
 * Source-transform unit tests for the js-v1 runtime.
 *
 * Pure tests — no isolated-vm dependency. Verifies that:
 *
 *   - Each export shape (ESM default, ESM named, CJS default, CJS named)
 *     resolves to the right `__handler` slot.
 *   - The manifest's `entrypoint` overrides the resolution order.
 *   - Missing handlers fail with `JS_NO_HANDLER`.
 *   - The transformed source no longer contains an `export` keyword
 *     (would crash the bare-V8 isolate).
 */

import { describe, it, expect } from 'vitest';
import {
  detectExports,
  transformSource,
  TransformError,
} from '../../../../src/executor/runtimes/js-v1-transform.js';

describe('detectExports', () => {
  it('detects ESM default function', () => {
    const r = detectExports('export default async function (ctx) {}');
    expect(r.hasEsmDefault).toBe(true);
  });

  it('detects ESM default arrow', () => {
    const r = detectExports('export default async (ctx) => 42;');
    expect(r.hasEsmDefault).toBe(true);
  });

  it('detects ESM named function', () => {
    const r = detectExports('export async function handler(ctx) {}');
    expect(r.esmNamed.has('handler')).toBe(true);
  });

  it('detects ESM named const', () => {
    const r = detectExports('export const handler = async (ctx) => {};');
    expect(r.esmNamed.has('handler')).toBe(true);
  });

  it('detects CJS default', () => {
    const r = detectExports('module.exports = async function (ctx) {};');
    expect(r.hasCjsDefault).toBe(true);
  });

  it('detects CJS named', () => {
    const r = detectExports('module.exports.handler = async (ctx) => {};');
    expect(r.cjsNamed.has('handler')).toBe(true);
  });
});

describe('transformSource', () => {
  it('rewrites ESM default into __handler', () => {
    const r = transformSource('export default async function (ctx) { return 1; }');
    expect(r.resolvedEntrypoint).toBe('default');
    expect(r.source).toContain('__handler');
    expect(r.source).not.toMatch(/^\s*export\s+default/m);
  });

  it('respects manifest entrypoint hint', () => {
    const src =
      'export async function handler(ctx) {}\n' +
      'export async function ping(ctx) { return "pong"; }';
    const r = transformSource(src, 'ping');
    expect(r.resolvedEntrypoint).toBe('ping');
    expect(r.source).toMatch(/__handler\s*=\s*\(typeof\s+ping/);
  });

  it('throws JS_NO_HANDLER when no exports', () => {
    expect(() => transformSource('const x = 1;')).toThrow(TransformError);
    try {
      transformSource('const x = 1;');
    } catch (e) {
      expect((e as TransformError).code).toBe('JS_NO_HANDLER');
    }
  });

  it('manifest entrypoint "handler" falls back to default export when no named handler', () => {
    const r = transformSource(
      'export default async function (ctx) { return "ok"; }',
      'handler',
    );
    expect(r.resolvedEntrypoint).toBe('default');
    expect(r.source).toContain('__esmDefault');
  });

  it('manifest entrypoint "handler" still prefers explicit named export', () => {
    const src =
      'export default async function () { return "def"; }\n' +
      'export async function handler(ctx) { return "named"; }';
    const r = transformSource(src, 'handler');
    expect(r.resolvedEntrypoint).toBe('handler');
  });

  it('throws JS_NO_HANDLER when entrypoint is not exported', () => {
    expect(() =>
      transformSource('export async function handler() {}', 'missing'),
    ).toThrow(TransformError);
  });

  it('rewrites CJS default', () => {
    const r = transformSource('module.exports = async function (ctx) { return 7; };');
    expect(r.resolvedEntrypoint).toBe('default');
    // Bootstrap stub adds `var module = { exports: {} }` so the rewrite
    // doesn't blow up at script-eval time.
    expect(r.source).toContain('var module');
  });

  it('rewrites named ESM exports without crashing', () => {
    const r = transformSource(
      'export async function helper(ctx) { return 1; }',
    );
    expect(r.resolvedEntrypoint).toBe('helper');
    expect(r.source).not.toMatch(/^\s*export\s+/m);
  });
});
