/**
 * OpenAI-Compatible Adapter Unit Tests
 *
 * Tests the base openaiAdapter, factory function, derivative adapters (xAI, Fireworks),
 * and parseStream for reasoning_content handling.
 */

import { describe, it, expect } from 'vitest';
import { openaiAdapter, xaiAdapter, fireworksAdapter, createOpenAICompatibleAdapter } from '../../../src/adapters/openai.js';
import type { AdapterRequestParams, AdapterChunk } from '../../../src/adapters/types.js';

function mockSSEResponse(chunks: unknown[]): Response {
  const lines = chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n';
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
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'image/png', base64: 'iVBOR...' }],
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
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'audio/wav', base64: 'UklGR...' }],
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

    it('emits PDF files as native file_data parts', () => {
      const uuid = 'aaaa1111-2222-3333-4444-555566667777';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { kind: 'binary' as const, mimeType: 'application/pdf', base64: 'JVBERi0xLjQK...' }],
      ]);
      const req = openaiAdapter.buildRequest(makeParams({
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
      const parts = body.messages[0].content as Array<Record<string, unknown>>;
      const filePart = parts.find(p => p.type === 'file');
      expect(filePart).toBeDefined();
      const fileObj = filePart!.file as { filename: string; file_data: string };
      expect(fileObj.filename).toBe('spec.pdf');
      expect(fileObj.file_data.startsWith('data:application/pdf;base64,')).toBe(true);
    });

    it('inlines text/html files as text parts (OpenAI rejects non-PDF inline file_data)', () => {
      const uuid = 'bbbb1111-2222-3333-4444-555566667777';
      const html = '<html><body>hi</body></html>';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, { kind: 'text' as const, mimeType: 'text/html', text: html }],
      ]);
      const req = openaiAdapter.buildRequest(makeParams({
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
      const parts = body.messages[0].content as Array<Record<string, unknown>>;
      expect(parts.find(p => p.type === 'file')).toBeUndefined();
      const inlined = parts.find(p => p.type === 'text' && (p.text as string).includes('<file name="unicorn.html"'));
      expect(inlined).toBeDefined();
      expect((inlined!.text as string)).toContain('mime="text/html"');
      expect((inlined!.text as string)).toContain('hi');
    });

    it('falls back to _extracted_text when media is not provided for non-PDF files', () => {
      const uuid = 'cccc1111-2222-3333-4444-555566667777';
      const req = openaiAdapter.buildRequest(makeParams({
        messages: [{
          role: 'user',
          content: '',
          content_items: [
            {
              type: 'file',
              file: { url: `/api/content/${uuid}` },
              filename: 'notes.html',
              mime_type: 'text/html',
              _extracted_text: '<h1>title</h1>',
            },
          ],
        }],
      }));
      const body = JSON.parse(req.body);
      const parts = body.messages[0].content as Array<Record<string, unknown>>;
      expect(parts.find(p => p.type === 'file')).toBeUndefined();
      const textPart = parts.find(p => p.type === 'text');
      expect((textPart!.text as string)).toContain('<file name="notes.html"');
      expect((textPart!.text as string)).toContain('mime="text/html"');
      expect((textPart!.text as string)).toContain('<h1>title</h1>');
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
    expect(body.model).toBe('accounts/fireworks/models/deepseek-v3p2');
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

describe('openaiAdapter parseStream', () => {
  it('yields thinking chunk for delta.reasoning_content', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { reasoning_content: 'Let me think...' }, index: 0 }] },
      { choices: [{ delta: { content: 'The answer is 42.' }, index: 0, finish_reason: 'stop' }] },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'thinking', text: 'Let me think...' });
    expect(chunks[1]).toEqual({ type: 'text', text: 'The answer is 42.' });
  });

  it('yields both thinking and text when both fields are in same delta', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { reasoning_content: 'thinking', content: 'answer' }, index: 0, finish_reason: 'stop' }] },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const types = chunks.map(c => c.type);
    expect(types).toContain('thinking');
    expect(types).toContain('text');
  });

  it('yields regular text for delta.content without reasoning_content', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { content: 'Hello world' }, index: 0, finish_reason: 'stop' }] },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'text', text: 'Hello world' });
  });

  it('yields cache_read_input from usage.prompt_tokens_details.cached_tokens', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { content: 'Hello' }, index: 0, finish_reason: 'stop' }] },
      { usage: { prompt_tokens: 500, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 300 } } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const usage = chunks.find(c => c.type === 'usage');
    expect(usage).toBeDefined();
    expect(usage!.input).toBe(500);
    expect(usage!.output).toBe(20);
    expect(usage!.cache_read_input).toBe(300);
  });

  it('omits cache_read_input when prompt_tokens_details is absent (backward compat)', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { content: 'Hello' }, index: 0, finish_reason: 'stop' }] },
      { usage: { prompt_tokens: 100, completion_tokens: 10 } },
    ]);
    const chunks = await collectChunks(openaiAdapter.parseStream(response));
    const usage = chunks.find(c => c.type === 'usage');
    expect(usage).toBeDefined();
    expect(usage!.cache_read_input).toBeUndefined();
  });

  it('works for fireworksAdapter (shared parseStream)', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { reasoning_content: 'DeepSeek reasoning...' }, index: 0 }] },
      { choices: [{ delta: { content: 'Result' }, index: 0, finish_reason: 'stop' }] },
    ]);
    const chunks = await collectChunks(fireworksAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'thinking', text: 'DeepSeek reasoning...' });
    expect(chunks[1]).toEqual({ type: 'text', text: 'Result' });
  });
});

describe('fireworksAdapter session affinity', () => {
  it('includes x-session-affinity header when sessionId is set', () => {
    const req = fireworksAdapter.buildRequest(makeParams({
      model: 'deepseek-v3p2',
      sessionId: 'chat-abc-123',
    }));
    expect(req.headers['x-session-affinity']).toBe('chat-abc-123');
  });

  it('omits x-session-affinity header when sessionId is undefined', () => {
    const req = fireworksAdapter.buildRequest(makeParams({ model: 'deepseek-v3p2' }));
    expect(req.headers['x-session-affinity']).toBeUndefined();
  });
});

describe('openaiAdapter tool message media handling', () => {
  it('replaces media content_items with text metadata for tool messages', () => {
    const req = openaiAdapter.buildRequest(makeParams({
      messages: [
        { role: 'user', content: 'Generate an image' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'tc_img', type: 'function' as const, function: { name: 'gen_image', arguments: '{}' } }],
        },
        {
          role: 'tool',
          content: 'Image generated',
          tool_call_id: 'tc_img',
          name: 'gen_image',
          content_items: [
            { type: 'image', content_id: 'img-001', mime_type: 'image/png' },
          ],
        } as any,
      ],
    }));

    const body = JSON.parse(req.body);
    const toolMsg = body.messages.find((m: any) => m.role === 'tool');
    expect(toolMsg).toBeDefined();
    expect(toolMsg.tool_call_id).toBe('tc_img');
    expect(typeof toolMsg.content).toBe('string');
    expect(toolMsg.content).toContain('[Available image:');
    expect(toolMsg.content).toContain('content_id=img-001');
    // No image_url parts
    expect(toolMsg.content).not.toContain('image_url');
  });
});

describe('openaiAdapter thinking level matrix', () => {
  type LevelExpect = { level: 'off' | 'low' | 'medium' | 'high' | undefined; expected: unknown };

  // Reasoning models — accept canonical levels (off → minimal). undefined
  // omits the field so the model uses its API default.
  const reasoningCases: LevelExpect[] = [
    { level: undefined, expected: undefined },
    { level: 'off',     expected: 'minimal' },
    { level: 'low',     expected: 'low' },
    { level: 'medium',  expected: 'medium' },
    { level: 'high',    expected: 'high' },
  ];

  for (const model of ['o4-mini', 'o3', 'gpt-5.4-pro']) {
    for (const { level, expected } of reasoningCases) {
      const label = level === undefined ? 'undefined' : level;
      it(`${model}: thinking=${label} → reasoning_effort=${expected ?? '(omitted)'}`, () => {
        const req = openaiAdapter.buildRequest(makeParams({ model, thinking: level }));
        const body = JSON.parse(req.body);
        expect(body.reasoning_effort).toBe(expected);
      });
    }
  }

  // Note on contract: the adapter NO LONGER gates by model name. The proxy
  // is the source of truth (see `resolveEffectiveThinking` in
  // lib/llm/uamp-proxy.ts), which consults the catalog and drops the level
  // for non-reasoning models before the adapter is ever called. If callers
  // bypass the proxy and hand the adapter `thinking: 'high'` on a non-
  // reasoning model, GIGO — the adapter will dutifully emit it. This is
  // tested at the catalog-↔-adapter contract layer
  // (tests/unit/models/thinking-capability.test.ts).

  it('boolean compat: false ≡ off → reasoning_effort=minimal', () => {
    const req = openaiAdapter.buildRequest(makeParams({ model: 'o4-mini', thinking: false as unknown as 'off' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBe('minimal');
  });

  it('boolean compat: true ≡ undefined (no override)', () => {
    const req = openaiAdapter.buildRequest(makeParams({ model: 'o4-mini', thinking: true as unknown as 'high' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBeUndefined();
  });

  // Regression: OpenAI rejects `reasoning_effort` for gpt-5.x on
  // /v1/chat/completions when function tools are present (it requires
  // /v1/responses for that combo). We silently drop the override so the
  // request still succeeds with the model's default reasoning budget.
  it('gpt-5 + tools: drops reasoning_effort (chat-completions limitation)', () => {
    const req = openaiAdapter.buildRequest(
      makeParams({
        model: 'gpt-5.4',
        thinking: 'medium',
        tools: [{ type: 'function', function: { name: 'noop', parameters: {} } } as unknown as never],
      }),
    );
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.tools).toBeDefined();
  });

  // Regression: gpt-5.x and o-series only accept temperature=1 (the API
  // default). Sending any other value returns 400 "Unsupported value:
  // 'temperature' does not support 0.7 with this model." Drop the field.
  it.each(['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini', 'o4-mini', 'o3'])(
    '%s: drops custom temperature (only default=1 allowed)',
    (model) => {
      const req = openaiAdapter.buildRequest(makeParams({ model, temperature: 0.7 }));
      const body = JSON.parse(req.body);
      expect(body.temperature).toBeUndefined();
    },
  );

  it('chat models (gpt-4o) keep custom temperature', () => {
    const req = openaiAdapter.buildRequest(makeParams({ model: 'gpt-4o', temperature: 0.7 }));
    const body = JSON.parse(req.body);
    expect(body.temperature).toBe(0.7);
  });

  it('o-series + tools: still emits reasoning_effort (supported by API)', () => {
    const req = openaiAdapter.buildRequest(
      makeParams({
        model: 'o4-mini',
        thinking: 'high',
        tools: [{ type: 'function', function: { name: 'noop', parameters: {} } } as unknown as never],
      }),
    );
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBe('high');
  });
});

describe('xaiAdapter thinking level matrix', () => {
  type LevelExpect = { level: 'off' | 'low' | 'medium' | 'high' | undefined; expected: unknown };

  // Grok 4 family — only 'low' / 'high' natively. Map medium → high.
  // off has no native value; we map to 'low' (Grok 4 has no off).
  const grok4Cases: LevelExpect[] = [
    { level: undefined, expected: undefined },
    { level: 'off',     expected: 'low' },
    { level: 'low',     expected: 'low' },
    { level: 'medium',  expected: 'high' },
    { level: 'high',    expected: 'high' },
  ];

  for (const model of ['grok-4-0709', 'grok-4.20-reasoning', 'grok-4-fast-reasoning']) {
    for (const { level, expected } of grok4Cases) {
      const label = level === undefined ? 'undefined' : level;
      it(`${model}: thinking=${label} → reasoning_effort=${expected ?? '(omitted)'}`, () => {
        const req = xaiAdapter.buildRequest(makeParams({ model, thinking: level }));
        const body = JSON.parse(req.body);
        expect(body.reasoning_effort).toBe(expected);
      });
    }
  }

  // Adapter trusts the proxy gate (see openaiAdapter contract note). Non-
  // reasoning xAI models like `grok-3` would only receive a thinking level
  // if the catalog says they support it; otherwise the proxy drops it.

  it('grok-3-mini supports reasoning_effort (low/high)', () => {
    const req = xaiAdapter.buildRequest(makeParams({ model: 'grok-3-mini', thinking: 'low' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBe('low');
  });
});

describe('fireworksAdapter thinking level matrix', () => {
  // Adapter just translates ThinkingLevel → reasoning_effort.
  // Catalog/proxy decides which models receive a level (instruct models
  // would 400 with "non-reasoning model does not support reasoning_effort"
  // — that's covered by the catalog-gating contract test).
  const cases: Array<[string, 'low' | 'medium' | 'high', string]> = [
    ['kimi-k2-thinking', 'low', 'low'],
    ['kimi-k2-thinking', 'medium', 'medium'],
    ['kimi-k2-thinking', 'high', 'high'],
    ['glm-5p1', 'medium', 'medium'],
  ];
  for (const [model, level, expected] of cases) {
    it(`${model}: thinking=${level} → reasoning_effort=${expected}`, () => {
      const req = fireworksAdapter.buildRequest(makeParams({ model, thinking: level }));
      const body = JSON.parse(req.body);
      expect(body.reasoning_effort).toBe(expected);
    });
  }

  it('omits reasoning_effort on off (lets hybrid models stay non-thinking)', () => {
    // For thinking-only models like kimi-k2-thinking, the catalog excludes
    // `off` from `levels` so the proxy clamps it up before reaching the adapter.
    // The adapter itself simply drops the param when it sees `off`, which is
    // the right behavior for hybrid models (deepseek-v3p2, kimi-k2p6, etc.).
    const req = fireworksAdapter.buildRequest(makeParams({ model: 'deepseek-v3p2', thinking: 'off' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBeUndefined();
  });

  it('omits reasoning_effort when thinking is undefined', () => {
    const req = fireworksAdapter.buildRequest(makeParams({ model: 'kimi-k2-thinking' }));
    const body = JSON.parse(req.body);
    expect(body.reasoning_effort).toBeUndefined();
  });
});
