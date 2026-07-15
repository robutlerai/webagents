/**
 * Responses API Adapter Unit Tests
 *
 * Covers the OpenAI + xAI Responses adapter (`responses.ts`):
 *   - Request shape (instructions, input items, flat function tools, native
 *     tool pass-through, reasoning block, parallel_tool_calls=false,
 *     store=false + include encrypted reasoning, max_output_tokens,
 *     temperature drop on gpt-5/o-series).
 *   - Encrypted reasoning replay across turns.
 *   - SSE event parsing for the AdapterChunk vocabulary.
 *   - URL citation annotations → tool_result(web_search).
 *   - Native built-in tool calls (web_search_call / code_interpreter_call
 *     / image_generation_call) → tool_result chunks.
 *   - Image generation streaming.
 *   - response.completed usage normalisation (input_tokens / output_tokens
 *     instead of prompt_tokens / completion_tokens).
 *   - response.error throwing.
 */

import { describe, it, expect } from 'vitest';
import {
  openaiAdapter,
  xaiAdapter,
  createResponsesApiAdapter,
} from '../../../src/adapters/responses.js';
import type { AdapterRequestParams, AdapterChunk, Message } from '../../../src/adapters/types.js';

function mockSSEResponse(events: unknown[]): Response {
  // Responses API SSE includes `event:` and `data:` lines; our reader only
  // consumes `data: ...` so we emit just those.
  const lines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

async function collectChunks(gen: AsyncGenerator<AdapterChunk>): Promise<AdapterChunk[]> {
  const out: AdapterChunk[] = [];
  for await (const c of gen) out.push(c);
  return out;
}

function makeParams(overrides: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'gpt-5.5',
    apiKey: 'test-key',
    ...overrides,
  };
}

// ────────────────────────────────────────────────────────────────
// Request shape
// ────────────────────────────────────────────────────────────────

describe('openaiAdapter (Responses)', () => {
  it('has name "openai" and targets /v1/responses', () => {
    expect(openaiAdapter.name).toBe('openai');
    const req = openaiAdapter.buildRequest(makeParams());
    expect(req.url).toBe('https://api.openai.com/v1/responses');
    expect(req.headers['Authorization']).toBe('Bearer test-key');
  });

  it('extracts leading system messages into top-level `instructions`', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      messages: [
        { role: 'system', content: 'You are a parrot.' },
        { role: 'system', content: 'Reply in haiku.' },
        { role: 'user', content: 'hi' },
      ],
    }));
    const body = JSON.parse(req.body);
    expect(body.instructions).toBe('You are a parrot.\n\nReply in haiku.');
    // System turns must NOT also appear in `input`.
    expect(body.input.find((i: { role?: string }) => i.role === 'system')).toBeUndefined();
  });

  it('emits user messages as `{type:"message", role:"user", content:[input_text]}`', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      messages: [{ role: 'user', content: 'Hello' }],
    }));
    const body = JSON.parse(req.body);
    const userItem = body.input[0];
    expect(userItem.type).toBe('message');
    expect(userItem.role).toBe('user');
    expect(userItem.content).toEqual([{ type: 'input_text', text: 'Hello' }]);
  });

  it('emits assistant tool_calls as `function_call` items with `call_id`', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      messages: [
        { role: 'user', content: 'q' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: 'fc_abc',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"Tokyo"}' },
          }],
        },
        { role: 'tool', content: 'sunny', tool_call_id: 'fc_abc' },
      ],
    }));
    const body = JSON.parse(req.body);
    const fc = body.input.find((i: { type: string }) => i.type === 'function_call');
    expect(fc).toBeDefined();
    expect(fc.call_id).toBe('fc_abc');
    expect(fc.name).toBe('get_weather');
    expect(fc.arguments).toBe('{"city":"Tokyo"}');
    const fco = body.input.find((i: { type: string }) => i.type === 'function_call_output');
    expect(fco).toBeDefined();
    expect(fco.call_id).toBe('fc_abc');
    expect(fco.output).toBe('sunny');
  });

  it('flattens function tools (no nested `function` wrapper)', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather',
          parameters: { type: 'object', properties: { city: { type: 'string' } } },
        },
      }],
    }));
    const body = JSON.parse(req.body);
    expect(body.tools).toHaveLength(1);
    const t = body.tools[0];
    expect(t.type).toBe('function');
    expect(t.name).toBe('get_weather');
    expect(t.description).toBe('Get weather');
    expect(t.parameters).toEqual({ type: 'object', properties: { city: { type: 'string' } } });
    expect(t).not.toHaveProperty('function');
  });

  it('passes native tools through verbatim (web_search, code_interpreter, image_generation)', () => {
    const nativeTools = [
      { type: 'web_search' as const },
      { type: 'code_interpreter' as const, container: { type: 'auto' } },
      { type: 'image_generation' as const },
    ];
    const req = openaiAdapter.buildRequest(makeParams({
      tools: nativeTools,
    }));
    const body = JSON.parse(req.body);
    expect(body.tools).toEqual(nativeTools);
  });

  it('sets reasoning.effort + summary:auto for thinking levels', () => {
    const req = openaiAdapter.buildRequest(makeParams({ thinking: 'high' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning).toEqual({ effort: 'high', summary: 'auto' });
  });

  it('maps off → minimal in reasoning.effort', () => {
    const req = openaiAdapter.buildRequest(makeParams({ thinking: 'off' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning).toEqual({ effort: 'minimal', summary: 'auto' });
  });

  it('omits the reasoning block when thinking is undefined', () => {
    const req = openaiAdapter.buildRequest(makeParams());
    const body = JSON.parse(req.body);
    expect(body.reasoning).toBeUndefined();
  });

  it('always sets parallel_tool_calls:false, store:false, include encrypted_content', () => {
    const req = openaiAdapter.buildRequest(makeParams());
    const body = JSON.parse(req.body);
    expect(body.parallel_tool_calls).toBe(false);
    expect(body.store).toBe(false);
    expect(body.include).toEqual(['reasoning.encrypted_content']);
  });

  it('uses max_output_tokens (not max_tokens)', () => {
    const req = openaiAdapter.buildRequest(makeParams({ maxTokens: 1024 }));
    const body = JSON.parse(req.body);
    expect(body.max_output_tokens).toBe(1024);
    expect(body.max_tokens).toBeUndefined();
    expect(body.max_completion_tokens).toBeUndefined();
  });

  it.each(['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini', 'o4-mini', 'o3'])(
    '%s: drops custom temperature (only default=1 allowed)',
    (model) => {
      const req = openaiAdapter.buildRequest(makeParams({ model, temperature: 0.7 }));
      const body = JSON.parse(req.body);
      expect(body.temperature).toBeUndefined();
    },
  );

  it('keeps custom temperature for non-gpt-5/o-series models', () => {
    const customAdapter = createResponsesApiAdapter({
      name: 'custom',
      baseUrl: 'https://example.com/v1',
    });
    const req = customAdapter.buildRequest(makeParams({ model: 'something-else', temperature: 0.4 }));
    const body = JSON.parse(req.body);
    expect(body.temperature).toBe(0.4);
  });
});

describe('xaiAdapter (Responses)', () => {
  it('has name "xai" and targets x.ai /v1/responses', () => {
    expect(xaiAdapter.name).toBe('xai');
    const req = xaiAdapter.buildRequest(makeParams({ model: 'grok-4.3' }));
    expect(req.url).toBe('https://api.x.ai/v1/responses');
  });

  it('applies model aliases for grok-4.20', () => {
    const req = xaiAdapter.buildRequest(makeParams({ model: 'grok-4.20-reasoning' }));
    const body = JSON.parse(req.body);
    expect(body.model).toBe('grok-4.20-reasoning-latest');
  });

  it('collapses thinking levels to xAI vocabulary (low|high)', () => {
    const cases: Array<['off' | 'low' | 'medium' | 'high', 'low' | 'high']> = [
      ['off', 'low'],
      ['low', 'low'],
      ['medium', 'high'],
      ['high', 'high'],
    ];
    for (const [requested, expected] of cases) {
      const req = xaiAdapter.buildRequest(makeParams({ model: 'grok-4.3', thinking: requested }));
      const body = JSON.parse(req.body);
      expect(body.reasoning).toEqual({ effort: expected });
    }
  });
});

// ────────────────────────────────────────────────────────────────
// Encrypted reasoning replay
// ────────────────────────────────────────────────────────────────

describe('encrypted reasoning replay', () => {
  it('serialises _encryptedReasoning into reasoning items in input', () => {
    const messages: Message[] = [
      { role: 'user', content: 'q' },
      {
        role: 'assistant',
        content: 'partial',
        _encryptedReasoning: ['enc_blob_1', 'enc_blob_2'],
        tool_calls: [{
          id: 'fc_x',
          type: 'function',
          function: { name: 'search', arguments: '{}' },
        }],
      },
      { role: 'tool', content: '{}', tool_call_id: 'fc_x' },
    ];
    const req = openaiAdapter.buildRequest(makeParams({ messages }));
    const body = JSON.parse(req.body);

    const reasoningItems = body.input.filter((i: { type: string }) => i.type === 'reasoning');
    expect(reasoningItems).toHaveLength(2);
    expect(reasoningItems[0].encrypted_content).toBe('enc_blob_1');
    expect(reasoningItems[1].encrypted_content).toBe('enc_blob_2');

    // Ordering: reasoning items precede the visible message + function_call
    // for that turn (matches the API's output order — the model needs the
    // chain-of-thought before the tool call to preserve context).
    const reasoningIdx = body.input.findIndex((i: { type: string }) => i.type === 'reasoning');
    const fcIdx = body.input.findIndex((i: { type: string }) => i.type === 'function_call');
    expect(reasoningIdx).toBeLessThan(fcIdx);
  });
});

// ────────────────────────────────────────────────────────────────
// Stream parsing
// ────────────────────────────────────────────────────────────────

describe('parseStream — text + thinking', () => {
  it('maps response.output_text.delta → text chunks', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_text.delta', item_id: 'msg_1', delta: 'Hi' },
      { type: 'response.output_text.delta', item_id: 'msg_1', delta: ' there' },
      { type: 'response.completed', response: { usage: { input_tokens: 5, output_tokens: 3 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'text', text: 'Hi' });
    expect(chunks[1]).toEqual({ type: 'text', text: ' there' });
  });

  it('maps response.reasoning_summary_text.delta → thinking chunks', async () => {
    const response = mockSSEResponse([
      { type: 'response.reasoning_summary_text.delta', delta: 'Let me think...' },
      { type: 'response.output_text.delta', delta: '42' },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'thinking', text: 'Let me think...' });
    expect(chunks[1]).toEqual({ type: 'text', text: '42' });
  });
});

describe('parseStream — function call lifecycle', () => {
  it('maps output_item.added + function_call_arguments.delta + output_item.done → tool_call chunks', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_item.added', item: { id: 'fc_1', type: 'function_call', call_id: 'call_1', name: 'get_weather', arguments: '' } },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_1', delta: '{"city":' },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_1', delta: '"Tokyo"}' },
      { type: 'response.output_item.done', item: { id: 'fc_1', type: 'function_call', call_id: 'call_1', name: 'get_weather', arguments: '{"city":"Tokyo"}' } },
      { type: 'response.completed', response: { usage: { input_tokens: 5, output_tokens: 5 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const start = chunks.find((c) => c.type === 'tool_call_start');
    expect(start).toEqual({ type: 'tool_call_start', id: 'call_1', name: 'get_weather' });
    const final = chunks.find((c) => c.type === 'tool_call');
    expect(final).toBeDefined();
    expect(final).toMatchObject({ id: 'call_1', name: 'get_weather', arguments: '{"city":"Tokyo"}' });
  });

  it('emits tool_call_progress at the 2 KiB cadence', async () => {
    const big = 'x'.repeat(2200);
    const response = mockSSEResponse([
      { type: 'response.output_item.added', item: { id: 'fc_2', type: 'function_call', call_id: 'call_2', name: 'foo' } },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_2', delta: big },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_2', delta: big },
      { type: 'response.output_item.done', item: { id: 'fc_2', type: 'function_call', call_id: 'call_2', name: 'foo', arguments: big + big } },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const progress = chunks.filter((c) => c.type === 'tool_call_progress');
    expect(progress.length).toBeGreaterThanOrEqual(1);
  });
});

describe('parseStream — native built-in tool calls', () => {
  it('emits tool_result for web_search_call lifecycle (call_id="web_search")', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_item.added', item: { id: 'ws_1', type: 'web_search_call', status: 'in_progress' } },
      { type: 'response.output_item.done', item: {
        id: 'ws_1', type: 'web_search_call', status: 'completed',
        action: { type: 'search', queries: ['robutler ai'] },
      } },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const start = chunks.find((c) => c.type === 'tool_call_start');
    expect(start).toMatchObject({ id: 'web_search', name: 'web_search' });
    const result = chunks.find((c) => c.type === 'tool_result');
    expect(result).toBeDefined();
    expect((result as Extract<AdapterChunk, { type: 'tool_result' }>).call_id).toBe('web_search');
  });

  it('emits tool_result for code_interpreter_call (call_id="code_execution")', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_item.added', item: { id: 'ci_1', type: 'code_interpreter_call', status: 'in_progress' } },
      { type: 'response.output_item.done', item: { id: 'ci_1', type: 'code_interpreter_call', status: 'completed' } },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const result = chunks.find((c) => c.type === 'tool_result');
    expect(result).toBeDefined();
    expect((result as Extract<AdapterChunk, { type: 'tool_result' }>).call_id).toBe('code_execution');
  });

  it('emits image + tool_result for image_generation_call', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_item.added', item: { id: 'img_1', type: 'image_generation_call', status: 'in_progress' } },
      { type: 'response.image_generation_call.partial_image', partial_image_b64: 'PARTIAL_BYTES_B64' },
      { type: 'response.output_item.done', item: { id: 'img_1', type: 'image_generation_call', status: 'completed', result: 'FINAL_BYTES_B64' } },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const images = chunks.filter((c) => c.type === 'image');
    expect(images.length).toBeGreaterThanOrEqual(1);
    expect((images[0] as Extract<AdapterChunk, { type: 'image' }>).base64.length).toBeGreaterThan(0);
    const result = chunks.find((c) => c.type === 'tool_result');
    expect((result as Extract<AdapterChunk, { type: 'tool_result' }>).call_id).toBe('image_generation');
  });
});

describe('parseStream — annotations', () => {
  it('maps url_citation annotation → tool_result(call_id="web_search")', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_text.delta', delta: 'Per ' },
      { type: 'response.output_text.annotation.added', annotation: {
        type: 'url_citation', url: 'https://example.com/x', title: 'Example',
      } },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const result = chunks.find((c) => c.type === 'tool_result');
    expect(result).toBeDefined();
    const tr = result as Extract<AdapterChunk, { type: 'tool_result' }>;
    expect(tr.call_id).toBe('web_search');
    expect(tr.result).toContain('https://example.com/x');
  });
});

describe('parseStream — usage normalisation', () => {
  it('normalises Responses-API usage fields (input_tokens / output_tokens / cached_tokens)', async () => {
    const response = mockSSEResponse([
      { type: 'response.output_text.delta', delta: 'hi' },
      { type: 'response.completed', response: {
        usage: {
          input_tokens: 500,
          output_tokens: 30,
          input_tokens_details: { cached_tokens: 200 },
        },
      } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const usage = chunks.find((c) => c.type === 'usage');
    expect(usage).toBeDefined();
    const u = usage as Extract<AdapterChunk, { type: 'usage' }>;
    expect(u.input).toBe(500);
    expect(u.output).toBe(30);
    expect(u.cache_read_input).toBe(200);
  });

  it('omits cache_read_input when no cached_tokens reported', async () => {
    const response = mockSSEResponse([
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const usage = chunks.find((c) => c.type === 'usage');
    expect(usage).toBeDefined();
    expect((usage as Extract<AdapterChunk, { type: 'usage' }>).cache_read_input).toBeUndefined();
  });
});

describe('parseStream — errors', () => {
  it('throws on response.error', async () => {
    const response = mockSSEResponse([
      { type: 'response.error', error: { message: 'boom' } },
    ]);
    await expect(collectChunks(openaiAdapter.parseStream(response))).rejects.toThrow(/boom/);
  });

  it('throws on response.failed', async () => {
    const response = mockSSEResponse([
      { type: 'response.failed', error: { message: 'rate_limit' } },
    ]);
    await expect(collectChunks(openaiAdapter.parseStream(response))).rejects.toThrow(/rate_limit/);
  });
});

// ────────────────────────────────────────────────────────────────
// Custom factory event-name remap (xAI parity hook)
// ────────────────────────────────────────────────────────────────

describe('createResponsesApiAdapter — eventNameMap', () => {
  it('remaps inbound event names per config.eventNameMap', async () => {
    const adapter = createResponsesApiAdapter({
      name: 'remap-test',
      baseUrl: 'https://example.com/v1',
      eventNameMap: { 'foo.text.delta': 'response.output_text.delta' },
    });
    const response = mockSSEResponse([
      { type: 'foo.text.delta', delta: 'mapped' },
      { type: 'response.completed', response: { usage: { input_tokens: 1, output_tokens: 1 } } },
    ]);
    const chunks = await collectChunks(adapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'text', text: 'mapped' });
  });
});

describe('data-URL image items (ephemeral tool screenshots)', () => {
  it('inlines a data: image URL as input_image WITHOUT resolvedMedia', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      messages: [
        {
          role: 'user',
          content: 'what do you see?',
          content_items: [{ type: 'image', image: { url: 'data:image/png;base64,QUJD' } }],
        },
      ],
    }));
    const body = JSON.parse(req.body);
    const userMsg = (body.input as Array<Record<string, any>>).find((m) => m.role === 'user' && Array.isArray(m.content));
    const img = (userMsg.content as Array<Record<string, any>>).find((p) => p.type === 'input_image');
    expect(img).toBeDefined();
    expect(img.image_url).toBe('data:image/png;base64,QUJD');
  });
});
