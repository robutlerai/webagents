/**
 * Content Helpers Unit Tests
 *
 * Tests for the generic UAMP content utility functions.
 */

import { describe, it, expect } from 'vitest';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl } from '../../../src/adapters/content.js';

describe('extractContentRef', () => {
  it('extracts URL from a plain string', () => {
    expect(extractContentRef('/api/content/abc-123')).toBe('/api/content/abc-123');
  });

  it('extracts URL from a { url } object', () => {
    expect(extractContentRef({ url: '/api/content/abc-123' })).toBe('/api/content/abc-123');
  });

  it('returns null for null/undefined', () => {
    expect(extractContentRef(null)).toBeNull();
    expect(extractContentRef(undefined)).toBeNull();
  });

  it('returns null for non-string/non-object', () => {
    expect(extractContentRef(42)).toBeNull();
    expect(extractContentRef(true)).toBeNull();
  });

  it('returns null for object without url property', () => {
    expect(extractContentRef({ path: '/foo' })).toBeNull();
  });
});

describe('isUAMPContentArray', () => {
  it('detects image items', () => {
    expect(isUAMPContentArray([{ type: 'image', image: '/api/content/abc' }])).toBe(true);
  });

  it('detects audio items', () => {
    expect(isUAMPContentArray([{ type: 'audio', audio: '/api/content/abc' }])).toBe(true);
  });

  it('detects video items', () => {
    expect(isUAMPContentArray([{ type: 'video', video: '/api/content/abc' }])).toBe(true);
  });

  it('detects file items', () => {
    expect(isUAMPContentArray([{ type: 'file', file: '/api/content/abc' }])).toBe(true);
  });

  it('rejects text-only arrays', () => {
    expect(isUAMPContentArray([{ type: 'text', text: 'hello' }])).toBe(false);
  });

  it('rejects OpenAI-format arrays (image_url)', () => {
    expect(isUAMPContentArray([{ type: 'image_url', image_url: { url: 'http://...' } }])).toBe(false);
  });

  it('rejects empty arrays', () => {
    expect(isUAMPContentArray([])).toBe(false);
  });

  it('rejects non-arrays', () => {
    expect(isUAMPContentArray('not an array')).toBe(false);
    expect(isUAMPContentArray(null)).toBe(false);
    expect(isUAMPContentArray(undefined)).toBe(false);
  });

  it('detects mixed text + media arrays', () => {
    expect(isUAMPContentArray([
      { type: 'text', text: 'look at this' },
      { type: 'image', image: { url: '/api/content/abc' } },
    ])).toBe(true);
  });
});

describe('canonicalContentUrl', () => {
  it('extracts canonical URL from /api/content/ path', () => {
    const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
    expect(canonicalContentUrl(`/api/content/${uuid}`)).toBe(`/api/content/${uuid}`);
  });

  it('extracts canonical URL from full URL', () => {
    const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
    expect(canonicalContentUrl(`https://example.com/api/content/${uuid}?token=abc`)).toBe(`/api/content/${uuid}`);
  });

  it('returns null for non-content URLs', () => {
    expect(canonicalContentUrl('https://example.com/image.png')).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(canonicalContentUrl('')).toBeNull();
  });
});
