/**
 * Anthropic Claude Adapter Unit Tests
 *
 * Tests buildRequest message conversion, system extraction, and media support.
 */

import { describe, it, expect } from 'vitest';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import type { AdapterRequestParams } from '../../../src/adapters/types.js';

function makeParams(overrides: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'claude-3.5-sonnet',
    apiKey: 'test-key',
    ...overrides,
  };
}

describe('anthropicAdapter', () => {
  it('has name "anthropic"', () => {
    expect(anthropicAdapter.name).toBe('anthropic');
  });

  describe('mediaSupport', () => {
    it('supports image and document as base64', () => {
      expect(anthropicAdapter.mediaSupport.image).toBe('base64');
      expect(anthropicAdapter.mediaSupport.document).toBe('base64');
    });

    it('does not support audio or video', () => {
      expect(anthropicAdapter.mediaSupport.audio).toBe('none');
      expect(anthropicAdapter.mediaSupport.video).toBe('none');
    });
  });

  describe('buildRequest', () => {
    it('builds a valid streaming request', () => {
      const req = anthropicAdapter.buildRequest(makeParams());
      expect(req.url).toContain('api.anthropic.com');
      expect(req.url).toContain('/messages');
      expect(req.headers['x-api-key']).toBe('test-key');
      expect(req.headers['anthropic-version']).toBeDefined();

      const body = JSON.parse(req.body);
      expect(body.model).toBe('claude-3.5-sonnet');
      expect(body.stream).toBe(true);
      expect(body.messages).toBeInstanceOf(Array);
    });

    it('extracts system message', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'system', content: 'Be helpful.' },
          { role: 'user', content: 'Hi' },
        ],
      }));
      const body = JSON.parse(req.body);
      expect(body.system).toBe('Be helpful.');
      expect(body.messages.length).toBe(1);
    });

    it('sets max_tokens with default', () => {
      const req = anthropicAdapter.buildRequest(makeParams());
      const body = JSON.parse(req.body);
      expect(body.max_tokens).toBe(4096);
    });

    it('uses custom maxTokens when provided', () => {
      const req = anthropicAdapter.buildRequest(makeParams({ maxTokens: 2000 }));
      const body = JSON.parse(req.body);
      expect(body.max_tokens).toBe(2000);
    });

    it('converts tools to Anthropic format', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: { name: 'search', description: 'Search', parameters: { type: 'object', properties: {} } },
        }],
      }));
      const body = JSON.parse(req.body);
      expect(body.tools).toBeDefined();
      expect(body.tools[0].name).toBe('search');
      expect(body.tools[0].input_schema).toBeDefined();
    });

    it('strips provider prefix from model name', () => {
      const req = anthropicAdapter.buildRequest(makeParams({ model: 'anthropic/claude-3.5-sonnet' }));
      const body = JSON.parse(req.body);
      expect(body.model).toBe('claude-3.5-sonnet');
    });
  });
});
