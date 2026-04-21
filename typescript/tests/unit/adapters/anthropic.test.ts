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

    // Thinking API: legacy "enabled + budget_tokens" works for sonnet-4-x, but
    // claude-opus-4-7 (and any future 4.7+) require the new "adaptive" shape with
    // output_config.effort. Sending the legacy shape returns a 400.
    it('uses legacy thinking shape for claude-sonnet-4 family', () => {
      const req = anthropicAdapter.buildRequest(makeParams({ model: 'anthropic/claude-sonnet-4-6' }));
      const body = JSON.parse(req.body);
      expect(body.thinking).toEqual({ type: 'enabled', budget_tokens: 10_000 });
      expect(body.output_config).toBeUndefined();
    });

    it('uses adaptive thinking + output_config for claude-opus-4-7', () => {
      const req = anthropicAdapter.buildRequest(makeParams({ model: 'anthropic/claude-opus-4-7' }));
      const body = JSON.parse(req.body);
      expect(body.thinking).toEqual({ type: 'adaptive' });
      expect(body.output_config).toEqual({ effort: 'medium' });
    });

    it('does not emit thinking config when thinking is disabled', () => {
      const req = anthropicAdapter.buildRequest(makeParams({ model: 'anthropic/claude-opus-4-7', thinking: false }));
      const body = JSON.parse(req.body);
      expect(body.thinking).toBeUndefined();
      expect(body.output_config).toBeUndefined();
    });

    it('converts content_items with image to Anthropic base64 image blocks via resolvedMedia', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'image/png', base64: 'iVBOR...' }],
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

    it('drops tool_use blocks whose id does not match Anthropic\'s ^[a-zA-Z0-9_-]+$ regex', () => {
      // Anthropic 400s the whole request if any tool_use.id contains chars
      // outside the regex (empty string, Gemini's `|ts:` thought-signature
      // suffix, dots, colons, etc.). Drop the offending blocks at conversion
      // time rather than discovering it on the wire.
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'go' },
          {
            role: 'assistant',
            content: 'partial',
            tool_calls: [
              { id: 'call_ok', type: 'function', function: { name: 'search', arguments: '{}' } },
              { id: '', type: 'function', function: { name: 'broken_empty', arguments: '{}' } },
              { id: 'call_abc|ts:sig', type: 'function', function: { name: 'broken_pipe', arguments: '{}' } },
            ],
          },
        ],
      }));
      const body = JSON.parse(req.body);
      const blocks = body.messages[1].content;
      const toolUses = blocks.filter((b: any) => b.type === 'tool_use');
      expect(toolUses).toHaveLength(1);
      expect(toolUses[0].id).toBe('call_ok');
    });

    it('drops tool messages whose tool_call_id does not match Anthropic\'s regex', () => {
      // Companion to the assistant-side drop above — the matching tool_use
      // would also have been dropped (same regex), so the tool_result has
      // nothing to pair against. Skipping the row keeps the conversation
      // valid without preserving an orphan that would 400 Anthropic.
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'go for it' },
          { role: 'assistant', content: '' },
          { role: 'tool', content: 'orphan tool result from legacy chat', tool_call_id: '' },
          { role: 'assistant', content: 'final answer' },
        ],
      }));
      const body = JSON.parse(req.body);
      const stringified = JSON.stringify(body.messages);
      expect(stringified).not.toContain('"tool_result"');
      expect(stringified).not.toContain('"tool_use_id":""');
      // The legible turns survive intact.
      expect(body.messages[0]).toEqual({ role: 'user', content: 'go for it' });
      expect(body.messages.at(-1)).toEqual({ role: 'assistant', content: 'final answer' });
    });

    it('keeps tool_result adjacent to tool_use even when an _inline_for_llm user message intervenes', () => {
      // Reproduces the read_content callback case: between an assistant
      // tool_use and the matching role=tool result row, a user message with
      // `_inline_for_llm: true` is pushed (so the next request can inline
      // base64 media). Anthropic requires tool_use to be IMMEDIATELY
      // followed by tool_result; the adapter must merge/reorder.
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'edit unicorn.html' },
          {
            role: 'assistant',
            content: 'Reading the file…',
            tool_calls: [{
              id: 'toolu_abc',
              type: 'function',
              function: { name: 'read_content', arguments: '{"content_id":"a89bd174"}' },
            }],
          },
          // Intervening user message (read_content's _inline_for_llm push)
          { role: 'user', content: '[Loaded content for analysis: a89bd174 (file)]' },
          // Tool result row
          { role: 'tool', content: 'Content a89bd174 (file) loaded into your context.', tool_call_id: 'toolu_abc' },
        ],
      }));
      const body = JSON.parse(req.body);
      // Expected order after rewrite:
      //   [0] user "edit unicorn.html"
      //   [1] assistant text + tool_use(toolu_abc)
      //   [2] user [tool_result#toolu_abc, …]
      //   [3] user "[Loaded content for analysis…]" (trailing inline)
      expect(body.messages[1].role).toBe('assistant');
      const asstBlocks = body.messages[1].content;
      expect(Array.isArray(asstBlocks)).toBe(true);
      expect(asstBlocks.some((b: Record<string, unknown>) => b.type === 'tool_use' && b.id === 'toolu_abc')).toBe(true);

      // Critical: messages[2] must be a user message whose FIRST block is a
      // tool_result for toolu_abc. Otherwise Anthropic rejects with
      // "tool_use ids were found without tool_result blocks immediately after".
      expect(body.messages[2].role).toBe('user');
      expect(Array.isArray(body.messages[2].content)).toBe(true);
      expect(body.messages[2].content[0].type).toBe('tool_result');
      expect(body.messages[2].content[0].tool_use_id).toBe('toolu_abc');

      // The trailing inline content should still be present, after the tool_result message.
      const remaining = body.messages.slice(3);
      const inlineUser = remaining.find((m: { role: string; content: unknown }) => m.role === 'user');
      expect(inlineUser).toBeDefined();
      const inlineBlocks = Array.isArray(inlineUser.content) ? inlineUser.content : [{ type: 'text', text: inlineUser.content }];
      expect(inlineBlocks.some((b: Record<string, unknown>) => b.type === 'text' && typeof b.text === 'string' && (b.text as string).includes('[Loaded content for analysis: a89bd174'))).toBe(true);
    });

    it('merges multiple tool_results into a single user message after assistant tool_use', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'do two things' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [
              { id: 'tu_a', type: 'function', function: { name: 'foo', arguments: '{}' } },
              { id: 'tu_b', type: 'function', function: { name: 'bar', arguments: '{}' } },
            ],
          },
          { role: 'tool', content: 'A done', tool_call_id: 'tu_a' },
          { role: 'tool', content: 'B done', tool_call_id: 'tu_b' },
        ],
      }));
      const body = JSON.parse(req.body);
      expect(body.messages[2].role).toBe('user');
      const blocks = body.messages[2].content as Array<Record<string, unknown>>;
      // Both tool_results merged into a single user message
      const ids = blocks.filter(b => b.type === 'tool_result').map(b => b.tool_use_id);
      expect(ids).toEqual(['tu_a', 'tu_b']);
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
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'image/jpeg', base64: '/9j/4AAQ...' }],
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

    it('preserves user message text when content_items has only media (regression: delegate body dropped)', () => {
      // Regression for the looping bug observed in
      // data/logs/llm-payloads/2026-04-19T04-13-55-236Z_anthropic_*claude-sonnet-4-6_*
      // where a delegate call carrying both a prompt body AND an attached image
      // (e.g. "Create unicorn.html using the attached image") was forwarded to
      // Claude as image-only — the model then had no instructions and just
      // described the picture instead of running text_editor.
      const uuid = 'abcd1234-1111-2222-3333-444455556666';
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'Create unicorn.html using the attached image. Use text_editor.',
          content_items: [
            { type: 'image', image: { url: `/api/content/${uuid}` } },
          ],
        }],
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      expect(userMsg.role).toBe('user');
      expect(Array.isArray(userMsg.content)).toBe(true);
      const firstBlock = userMsg.content[0];
      expect(firstBlock.type).toBe('text');
      expect(firstBlock.text).toBe('Create unicorn.html using the attached image. Use text_editor.');
    });

    it('does not duplicate text when first content block already matches msg.content', () => {
      const uuid = 'efef1234-1111-2222-3333-444455556666';
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'analyze this',
          content_items: [
            { type: 'text', text: 'analyze this' },
            { type: 'image', image: { url: `/api/content/${uuid}` } },
          ],
        }],
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      const textBlocks = userMsg.content.filter((b: { type: string }) => b.type === 'text');
      // Only the synthetic describeContentItem text should be present (the
      // type:'text' UAMP item is ignored by uampToAnthropicBlocks); the
      // backstop should NOT add a duplicate "analyze this" block. We accept
      // either zero or one "analyze this" text — never two.
      const matches = textBlocks.filter((b: { text: string }) => b.text === 'analyze this');
      expect(matches.length).toBeLessThanOrEqual(1);
    });

    it('includes top-level cache_control for automatic caching', () => {
      const req = anthropicAdapter.buildRequest(makeParams());
      const body = JSON.parse(req.body);
      expect(body.cache_control).toEqual({ type: 'ephemeral' });
    });

    it('emits PDF files as document base64 blocks', () => {
      const uuid = 'cccc1111-2222-3333-4444-555566667777';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'application/pdf', base64: 'JVBERi0xLjQK...' }],
      ]);
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'read this',
          content_items: [
            { type: 'text', text: 'read this' },
            { type: 'file', file: { url: `/api/content/${uuid}` }, filename: 'spec.pdf' },
          ],
        }],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      const docBlock = userMsg.content.find((b: Record<string, unknown>) => b.type === 'document');
      expect(docBlock).toBeDefined();
      expect(docBlock.source.type).toBe('base64');
      expect(docBlock.source.media_type).toBe('application/pdf');
    });

    it('inlines text/html files as text blocks (Anthropic rejects non-PDF documents)', () => {
      const uuid = 'dddd1111-2222-3333-4444-555566667777';
      const html = '<html><body><h1>Hello</h1></body></html>';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { kind: 'text' as const, mimeType: 'text/html', text: html }],
      ]);
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: 'edit this',
          content_items: [
            { type: 'text', text: 'edit this' },
            { type: 'file', file: { url: `/api/content/${uuid}` }, filename: 'unicorn.html' },
          ],
        }],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      const docBlock = userMsg.content.find((b: Record<string, unknown>) => b.type === 'document');
      expect(docBlock).toBeUndefined();
      const textBlocks = userMsg.content.filter((b: Record<string, unknown>) => b.type === 'text');
      const inlined = textBlocks.find((b: { text: string }) => b.text.includes('<file name="unicorn.html"'));
      expect(inlined).toBeDefined();
      expect(inlined.text).toContain('mime="text/html"');
      expect(inlined.text).toContain('Hello');
    });

    it('falls back to _extracted_text when media is not provided for non-PDF files', () => {
      const uuid = 'eeee1111-2222-3333-4444-555566667777';
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: '',
          content_items: [
            {
              type: 'file',
              file: { url: `/api/content/${uuid}` },
              filename: 'notes.html',
              mime_type: 'text/html',
              _extracted_text: '<h1>title</h1><p>body</p>',
            },
          ],
        }],
      }));
      const body = JSON.parse(req.body);
      const userMsg = body.messages[0];
      expect(userMsg.content.find((b: Record<string, unknown>) => b.type === 'document')).toBeUndefined();
      const textBlock = userMsg.content.find((b: Record<string, unknown>) => b.type === 'text');
      expect(textBlock.text).toContain('<file name="notes.html"');
      expect(textBlock.text).toContain('mime="text/html"');
      expect(textBlock.text).toContain('<h1>title</h1>');
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

    it('emits tool_call_start then tool_call_progress then tool_call for large tool args', async () => {
      const largeArg = 'x'.repeat(5000);
      const partialChunks: unknown[] = [];
      const chunkSize = 500;
      for (let i = 0; i < largeArg.length; i += chunkSize) {
        partialChunks.push({
          type: 'content_block_delta',
          index: 1,
          delta: { type: 'input_json_delta', partial_json: largeArg.slice(i, i + chunkSize) },
        });
      }

      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 10 } } },
        { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'tc_big', name: 'text_editor' } },
        ...partialChunks,
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 5 } },
      ]);

      const chunks = await collectChunks(anthropicAdapter.parseStream(response));

      const starts = chunks.filter(c => c.type === 'tool_call_start');
      expect(starts).toHaveLength(1);
      expect(starts[0]).toEqual({ type: 'tool_call_start', id: 'tc_big', name: 'text_editor' });

      const progress = chunks.filter(c => c.type === 'tool_call_progress');
      expect(progress.length).toBeGreaterThanOrEqual(1);
      for (const p of progress) {
        expect(p.type).toBe('tool_call_progress');
        expect((p as { id: string }).id).toBe('tc_big');
        expect((p as { bytes: number }).bytes).toBeGreaterThan(0);
      }

      const calls = chunks.filter(c => c.type === 'tool_call');
      expect(calls).toHaveLength(1);
      expect((calls[0] as { arguments: string }).arguments).toBe(largeArg);
    });

    it('emits tool_call_start without tool_call_progress for small args', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 10 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tc_sm', name: 'fs' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"action":"list"}' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 3 } },
      ]);

      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      expect(chunks.filter(c => c.type === 'tool_call_start')).toHaveLength(1);
      expect(chunks.filter(c => c.type === 'tool_call_progress')).toHaveLength(0);
      expect(chunks.filter(c => c.type === 'tool_call')).toHaveLength(1);
    });

    // Anthropic returns text_editor tool_use under one of two per-version
    // names. The adapter must normalize both back to the canonical UAMP name
    // so the rest of the system never sees provider-specific spellings.
    it('normalizes inbound str_replace_editor to canonical text_editor', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 10 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tc_legacy', name: 'str_replace_editor' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"command":"view","path":"/x.md"}' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 1 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const start = chunks.find(c => c.type === 'tool_call_start');
      const call = chunks.find(c => c.type === 'tool_call');
      expect((start as { name: string }).name).toBe('text_editor');
      expect((call as { name: string }).name).toBe('text_editor');
    });

    it('normalizes inbound str_replace_based_edit_tool to canonical text_editor', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 10 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tc_modern', name: 'str_replace_based_edit_tool' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"command":"view","path":"/x.md"}' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 1 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const start = chunks.find(c => c.type === 'tool_call_start');
      const call = chunks.find(c => c.type === 'tool_call');
      expect((start as { name: string }).name).toBe('text_editor');
      expect((call as { name: string }).name).toBe('text_editor');
    });

    it('passes through unknown tool names unchanged', async () => {
      const response = mockSSEResponse([
        { type: 'message_start', message: { usage: { input_tokens: 10 } } },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tc_x', name: 'custom_tool' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{}' } },
        { type: 'content_block_stop' },
        { type: 'message_delta', usage: { output_tokens: 1 } },
      ]);
      const chunks = await collectChunks(anthropicAdapter.parseStream(response));
      const call = chunks.find(c => c.type === 'tool_call');
      expect((call as { name: string }).name).toBe('custom_tool');
    });
  });

  describe('tool message media handling', () => {
    it('appends text metadata for media content_items in tool messages', () => {
      const req = anthropicAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Generate audio' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 'tc_aud', type: 'function' as const, function: { name: 'gen_audio', arguments: '{}' } }],
          },
          {
            role: 'tool',
            content: 'Audio generated',
            tool_call_id: 'tc_aud',
            name: 'gen_audio',
            content_items: [
              { type: 'audio', content_id: 'aud-002', duration_ms: 60000 },
            ],
          } as any,
        ],
      }));

      const body = JSON.parse(req.body);
      const toolResult = body.messages.find((m: any) =>
        m.content?.some?.((b: any) => b.type === 'tool_result'),
      );
      expect(toolResult).toBeDefined();
      const block = toolResult.content.find((b: any) => b.type === 'tool_result');
      expect(block.content).toContain('[Available audio:');
      expect(block.content).toContain('content_id=aud-002');
      expect(block.content).toContain('duration=1:00');
    });
  });
});
