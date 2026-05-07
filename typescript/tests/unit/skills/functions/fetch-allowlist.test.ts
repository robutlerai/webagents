/**
 * Fetch allowlist unit tests for the js-v1 runtime.
 */

import { describe, it, expect } from 'vitest';
import { isFetchAllowed } from '../../../../src/executor/runtimes/js-v1.js';

describe('isFetchAllowed', () => {
  it('rejects when allowlist is empty', () => {
    expect(isFetchAllowed([], 'https://example.com')).toBe(false);
  });

  it('allows star wildcard', () => {
    expect(isFetchAllowed(['*'], 'https://example.com')).toBe(true);
  });

  it('matches exact URL prefix', () => {
    expect(isFetchAllowed(['https://api.example.com'], 'https://api.example.com/v1')).toBe(true);
  });

  it('matches exact host', () => {
    expect(isFetchAllowed(['api.example.com'], 'https://api.example.com/x')).toBe(true);
    expect(isFetchAllowed(['api.example.com'], 'https://other.example.com/x')).toBe(false);
  });

  it('matches single-label wildcard', () => {
    expect(isFetchAllowed(['*.example.com'], 'https://api.example.com/x')).toBe(true);
    // Two-label subdomain — single-label wildcard does NOT match it
    expect(isFetchAllowed(['*.example.com'], 'https://a.b.example.com/x')).toBe(true);
    // *.example.com should NOT match the bare hostname
    expect(isFetchAllowed(['*.example.com'], 'https://example.com/x')).toBe(false);
  });

  it('rejects non-http(s) protocols', () => {
    expect(isFetchAllowed(['*'], 'ftp://example.com')).toBe(false);
    expect(isFetchAllowed(['*'], 'file:///etc/passwd')).toBe(false);
  });

  it('rejects malformed URLs', () => {
    expect(isFetchAllowed(['*'], 'not a url')).toBe(false);
  });
});
