/**
 * Anthropic Claude Adapter Unit Tests
 *
 * Tests buildRequest message conversion, system extraction, and media support.
 */

import { describe, it, expect } from 'vitest';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import type { AdapterRequestParams, AdapterChunk } from '../../../src/adapters/types.js';

function mockSSEResponse(chunks: unknown[]): Response {
  const lines = chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('');
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

async function collectChunks(gen: AsyncGenerator<AdapterChunk>): Promise<AdapterChunk[]> {
  const result: AdapterChunk[] = [];
  for await (const chunk of gen) result.push(chunk);
  return result;
}

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

    it('converts content_items with image to Anthropic base64 image blocks via resolvedMedia', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { mimeType: 'image/png', base64: 'iVBOR...' }],
      ]);
      const req = anthropicAdapter.buildRequest(makeParams({
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
      const imgBlock = userMsg.content.find((b: Record<string, unknown>) => b.type === 'image');
      expect(imgBlock).toBeDefined();
      expect(imgBlock.source.type).toBe('base64');
      expect(imgBlock.source.media_type).toBe('image/png');
      expect(imgBlock.source.data).toBe('iVBOR...');
    });

    it('does not forward content_items as extra field', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
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

    it('converts assistant tool_calls to tool_use content blocks', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'search for cats' },
          {
            role: 'assistant',
            content: 'Let me search.',
            tool_calls: [{
              id: 'call_1',
              type: 'function',
              function: { name: 'search', arguments: '{"q":"cats"}' },
            }],
          },
        ],
      }));
      const body = JSON.parse(req.body);
      const assistantMsg = body.messages[1];
      expect(assistantMsg.role).toBe('assistant');
      expect(Array.isArray(assistantMsg.content)).toBe(true);
      const toolUse = assistantMsg.content.find((b: Record<string, unknown>) => b.type === 'tool_use');
      expect(toolUse).toBeDefined();
      expect(toolUse.id).toBe('call_1');
      expect(toolUse.name).toBe('search');
      expect(toolUse.input).toEqual({ q: 'cats' });
    });

    it('converts tool role messages to tool_result blocks as role "user"', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'search' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: 'call_1',
              type: 'function',
              function: { name: 'search', arguments: '{}' },
            }],
          },
          { role: 'tool', content: 'results here', tool_call_id: 'call_1' },
        ],
      }));
      const body = JSON.parse(req.body);
      const toolResult = body.messages[2];
      expect(toolResult.role).toBe('user');
      expect(toolResult.content[0].type).toBe('tool_result');
      expect(toolResult.content[0].tool_use_id).toBe('call_1');
      expect(toolResult.content[0].content).toBe('results here');
    });

    it('concatenates multiple system messages with double newline', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'system', content: 'Be concise.' },
          { role: 'user', content: 'Hi' },
        ],
      }));
      const body = JSON.parse(req.body);
      expect(body.system).toBe('You are helpful.\n\nBe concise.');
      expect(body.messages.length).toBe(1);
    });

    it('handles mixed conversation with system + user + assistant + tool + content_items', () => {
      const uuid = 'aaaa1111-2222-3333-4444-555566667777';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { mimeType: 'image/jpeg', base64: '/9j/4AAQ...' }],
      ]);
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'system', content: 'You are an image analyst.' },
          {
            role: 'user',
            content: 'analyze this',
            content_items: [
              { type: 'text', text: 'analyze this' },
              { type: 'image', image: { url: `/api/content/${uuid}` } },
            ],
          },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: 'call_2',
              type: 'function',
              function: { name: 'analyze', arguments: '{}' },
            }],
          },
          { role: 'tool', content: '{"result":"cat"}', tool_call_id: 'call_2' },
          { role: 'assistant', content: 'It is a cat.' },
        ],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      expect(body.system).toBe('You are an image analyst.');
      expect(body.messages.length).toBe(4);
      expect(body.messages[0].role).toBe('user');
      expect(body.messages[1].role).toBe('assistant');
      expect(body.messages[2].role).toBe('user');
      expect(body.messages[3].role).toBe('assistant');
      expect(body.messages[3].content).toBe('It is a cat.');
    });

    it('includes top-level cache_control for automatic caching', () => {
      const req = anthropicAdapter.buildRequest(makeParams());
      const body = JSON.parse(req.body);
      expect(body.cache_control).toEqual({ type: 'ephemeral' });
    });
  });

  describe('parseStream', () => {
    it('yields cache_read_input from message_start.usage.cache_read_input_tokens', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 50, cache_read_input_tokens: 200, cache_creation_input_tokens: 0 } } },
        { type: 'content_block_start', content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hello' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 10 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.input).toBe(50);
      expect(usage!.output).toBe(10);
      expect(usage!.cache_read_input).toBe(200);
      expect(usage!.cache_creation_input).toBeUndefined();
    });

    it('yields cache_creation_input from message_start.usage.cache_creation_input_tokens', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 100, cache_read_input_tokens: 0, cache_creation_input_tokens: 500 } } },
        { type: 'content_block_start', content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', delta: { type: 'text_delta', text: 'World' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 20 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.input).toBe(100);
      expect(usage!.cache_creation_input).toBe(500);
      expect(usage!.cache_read_input).toBeUndefined();
    });

    it('omits cache fields when no cached tokens are present', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 100 } } },
        { type: 'content_block_start', content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hi' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 5 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.cache_read_input).toBeUndefined();
      expect(usage!.cache_creation_input).toBeUndefined();
    });
  });
});
