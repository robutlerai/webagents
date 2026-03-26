/**
 * OpenAI-Compatible Adapter Unit Tests
 *
 * Tests the base openaiAdapter, factory function, and derivative adapters (xAI, Fireworks).
 */

import { describe, it, expect } from 'vitest';
import { openaiAdapter, xaiAdapter, fireworksAdapter, createOpenAICompatibleAdapter } from '../../../src/adapters/openai.js';
import type { AdapterRequestParams } from '../../../src/adapters/types.js';

function makeParams(overrides: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'gpt-4o',
    apiKey: 'test-key',
    ...overrides,
  };
}

describe('openaiAdapter', () => {
  it('has name "openai"', () => {
    expect(openaiAdapter.name).toBe('openai');
  });

  describe('mediaSupport', () => {
    it('supports image via URL', () => {
      expect(openaiAdapter.mediaSupport.image).toBe('url');
    });

    it('supports audio via base64', () => {
      expect(openaiAdapter.mediaSupport.audio).toBe('base64');
    });
  });

  describe('buildRequest', () => {
    it('builds a valid streaming request', () => {
      const req = openaiAdapter.buildRequest(makeParams());
      expect(req.url).toContain('api.openai.com');
      expect(req.url).toContain('/chat/completions');
      expect(req.headers['Authorization']).toBe('Bearer test-key');
      expect(req.headers['Content-Type']).toBe('application/json');

      const body = JSON.parse(req.body);
      expect(body.model).toBe('gpt-4o');
      expect(body.stream).toBe(true);
      expect(body.stream_options).toEqual({ include_usage: true });
      expect(body.messages).toEqual([{ role: 'user', content: 'Hello' }]);
    });

    it('includes tools when provided', () => {
      const req = openaiAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: { name: 'test', description: 'Test tool' },
        }],
      }));
      const body = JSON.parse(req.body);
      expect(body.tools).toHaveLength(1);
    });

    it('passes messages through without conversion when no content_items', () => {
      const msgs = [
        { role: 'system', content: 'Be helpful.' },
        { role: 'user', content: 'Hi' },
      ];
      const req = openaiAdapter.buildRequest(makeParams({ messages: msgs }));
      const body = JSON.parse(req.body);
      expect(body.messages[0].role).toBe('system');
      expect(body.messages[0].content).toBe('Be helpful.');
      expect(body.messages[1].role).toBe('user');
      expect(body.messages[1].content).toBe('Hi');
    });

    it('converts content_items on user message to OpenAI image_url parts via resolvedMedia', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { mimeType: 'image/png', base64: 'iVBOR...' }],
      ]);
      const req = openaiAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'describe this',
          content_items: [
            { type: 'text', text: 'describe this' },
            { type: 'image', image: { url: `/api/content/${uuid}` } },
          ],
        }],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      expect(userMsg.role).toBe('user');
      expect(Array.isArray(userMsg.content)).toBe(true);
      const imgPart = userMsg.content.find((p: Record<string, unknown>) => p.type === 'image_url');
      expect(imgPart).toBeDefined();
      expect(imgPart.image_url.url).toContain('data:image/png;base64,');
    });

    it('converts audio content_items to input_audio parts via resolvedMedia', () => {
      const uuid = 'aaaa1111-2222-3333-4444-555566667777';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { mimeType: 'audio/wav', base64: 'UklGR...' }],
      ]);
      const req = openaiAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'transcribe this',
          content_items: [
            { type: 'text', text: 'transcribe this' },
            { type: 'audio', audio: { url: `/api/content/${uuid}` } },
          ],
        }],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      const audioPart = userMsg.content.find((p: Record<string, unknown>) => p.type === 'input_audio');
      expect(audioPart).toBeDefined();
      expect(audioPart.input_audio.data).toBe('UklGR...');
      expect(audioPart.input_audio.format).toBe('wav');
    });

    it('strips content_items field from output', () => {
      const req = openaiAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'hello',
          content_items: [{ type: 'text', text: 'hello' }],
        }],
      }));
      const body = JSON.parse(req.body);
      const raw = JSON.stringify(body);
      expect(raw).not.toContain('content_items');
    });

    it('messages without content_items pass through unchanged (backward compat)', () => {
      const msgs = [
        { role: 'user', content: 'Hello world' },
        { role: 'assistant', content: 'Hi!' },
      ];
      const req = openaiAdapter.buildRequest(makeParams({ messages: msgs }));
      const body = JSON.parse(req.body);
      expect(body.messages[0].content).toBe('Hello world');
      expect(body.messages[1].content).toBe('Hi!');
    });
  });
});

describe('xaiAdapter', () => {
  it('has name "xai"', () => {
    expect(xaiAdapter.name).toBe('xai');
  });

  it('builds requests to xAI API', () => {
    const req = xaiAdapter.buildRequest(makeParams({ model: 'grok-3' }));
    expect(req.url).toContain('/chat/completions');
    const body = JSON.parse(req.body);
    expect(body.model).toBe('grok-3');
  });
});

describe('fireworksAdapter', () => {
  it('has name "fireworks"', () => {
    expect(fireworksAdapter.name).toBe('fireworks');
  });

  it('builds requests to Fireworks API', () => {
    const req = fireworksAdapter.buildRequest(makeParams({ model: 'deepseek-v3p2' }));
    expect(req.url).toContain('/chat/completions');
    const body = JSON.parse(req.body);
    expect(body.model).toBe('deepseek-v3p2');
  });
});

describe('createOpenAICompatibleAdapter', () => {
  it('creates a custom adapter with given base URL', () => {
    const custom = createOpenAICompatibleAdapter({
      name: 'custom',
      baseUrl: 'https://custom.api.com/v1',
    });
    expect(custom.name).toBe('custom');
    const req = custom.buildRequest(makeParams());
    expect(req.url).toContain('custom.api.com');
  });

  it('applies model aliases', () => {
    const custom = createOpenAICompatibleAdapter({
      name: 'custom',
      baseUrl: 'https://example.com/v1',
      modelAliases: { 'my-model': 'real-model-name' },
    });
    const req = custom.buildRequest(makeParams({ model: 'my-model' }));
    const body = JSON.parse(req.body);
    expect(body.model).toBe('real-model-name');
  });

  it('allows custom media support overrides', () => {
    const custom = createOpenAICompatibleAdapter({
      name: 'custom',
      baseUrl: 'https://example.com/v1',
      mediaSupport: { video: 'url' },
    });
    expect(custom.mediaSupport.video).toBe('url');
    expect(custom.mediaSupport.image).toBe('url'); // default
  });
});
