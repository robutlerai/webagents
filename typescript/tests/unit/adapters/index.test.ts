/**
 * Adapter Registry Tests
 *
 * Verifies getAdapter returns correct adapters and throws for unknown providers.
 */

import { describe, it, expect } from 'vitest';
import { getAdapter, googleAdapter, anthropicAdapter, openaiAdapter, xaiAdapter, fireworksAdapter } from '../../../src/adapters/index.js';

describe('getAdapter', () => {
  it.each([
    ['google', googleAdapter],
    ['anthropic', anthropicAdapter],
    ['openai', openaiAdapter],
    ['xai', xaiAdapter],
    ['fireworks', fireworksAdapter],
  ])('returns %s adapter', (provider, expected) => {
    expect(getAdapter(provider)).toBe(expected);
  });

  it('throws for unknown provider', () => {
    expect(() => getAdapter('unknown')).toThrow('Unknown LLM provider: unknown');
  });

  it('error message lists available providers', () => {
    expect(() => getAdapter('bad')).toThrow(/google.*anthropic.*openai.*xai.*fireworks/);
  });
});

describe('adapter contracts', () => {
  const adapters = [googleAdapter, anthropicAdapter, openaiAdapter, xaiAdapter, fireworksAdapter];

  it.each(adapters.map(a => [a.name, a]))('%s has required LLMAdapter fields', (_name, adapter) => {
    expect(adapter).toHaveProperty('name');
    expect(typeof adapter.name).toBe('string');
    expect(adapter).toHaveProperty('mediaSupport');
    expect(adapter.mediaSupport).toHaveProperty('image');
    expect(adapter.mediaSupport).toHaveProperty('audio');
    expect(adapter.mediaSupport).toHaveProperty('video');
    expect(adapter.mediaSupport).toHaveProperty('document');
    expect(typeof adapter.buildRequest).toBe('function');
    expect(typeof adapter.parseStream).toBe('function');
  });
});
