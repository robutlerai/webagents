/**
 * `encodeInlineCodeRef` unit tests.
 */

import { describe, it, expect } from 'vitest';
import {
  encodeInlineCodeRef,
  CodeTooLargeError,
  MAX_INLINE_BYTES,
  MAX_INLINE_B64_BYTES,
} from '../../../../src/skills/functions/manifest.js';

describe('encodeInlineCodeRef', () => {
  it('returns inline for small UTF-8 sources', async () => {
    const src = 'export default async () => 1;';
    const ref = await encodeInlineCodeRef(src);
    expect(ref).toEqual({ kind: 'inline', source: src });
  });

  it('returns inlineB64 for sources over the inline cap', async () => {
    const src = 'a'.repeat(MAX_INLINE_BYTES + 1);
    const ref = await encodeInlineCodeRef(src);
    if (ref.kind !== 'inlineB64') throw new Error(`expected inlineB64, got ${ref.kind}`);
    expect(typeof ref.sha256).toBe('string');
    expect(ref.sha256!.length).toBe(64);
    // sanity: base64 source decodes back.
    const decoded = Buffer.from(ref.source, 'base64').toString('utf-8');
    expect(decoded.length).toBe(src.length);
  });

  it('throws CodeTooLargeError above 64KB', async () => {
    const src = 'a'.repeat(MAX_INLINE_B64_BYTES + 1);
    await expect(encodeInlineCodeRef(src)).rejects.toBeInstanceOf(CodeTooLargeError);
  });
});
