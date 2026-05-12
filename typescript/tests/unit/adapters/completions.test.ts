/**
 * Chat-Completions Adapter Unit Tests
 *
 * Covers the chat-completions factory (`createChatCompletionsAdapter`) and
 * the only production adapter still wired through it: Fireworks. The
 * OpenAI/xAI Responses adapter has its own test file at `responses.test.ts`.
 *
 * The factory is also the rollback target for OpenAI/xAI when
 * `OPENAI_USE_CHAT_COMPLETIONS=1` / `XAI_USE_CHAT_COMPLETIONS=1` is set —
 * those code paths are exercised at a smoke level by the registry test
 * (`adapters/index.ts` env-flag dispatch).
 */

import { describe, it, expect } from 'vitest';
import {
  fireworksAdapter,
  createChatCompletionsAdapter,
  createOpenAICompatibleAdapter,
} from '../../../src/adapters/completions.js';
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
    model: 'kimi-k2p6',
    apiKey: 'test-key',
    ...overrides,
  };
}

describe('createChatCompletionsAdapter', () => {
  it('creates a custom adapter with given base URL', () => {
    const custom = createChatCompletionsAdapter({
      name: 'custom',
      baseUrl: 'https://custom.api.com/v1',
    });
    expect(custom.name).toBe('custom');
    const req = custom.buildRequest(makeParams());
    expect(req.url).toContain('custom.api.com');
    expect(req.url).toContain('/chat/completions');
  });

  it('back-compat alias `createOpenAICompatibleAdapter` is identical', () => {
    expect(createOpenAICompatibleAdapter).toBe(createChatCompletionsAdapter);
  });

  it('applies model aliases', () => {
    const custom = createChatCompletionsAdapter({
      name: 'custom',
      baseUrl: 'https://example.com/v1',
      modelAliases: { 'my-model': 'real-model-name' },
    });
    const req = custom.buildRequest(makeParams({ model: 'my-model' }));
    const body = JSON.parse(req.body);
    expect(body.model).toBe('real-model-name');
  });

  it('allows custom media support overrides', () => {
    const custom = createChatCompletionsAdapter({
      name: 'custom',
      baseUrl: 'https://example.com/v1',
      mediaSupport: { video: 'url' },
    });
    expect(custom.mediaSupport.video).toBe('url');
    expect(custom.mediaSupport.image).toBe('url'); // default
  });
});

describe('fireworksAdapter', () => {
  it('has name "fireworks"', () => {
    expect(fireworksAdapter.name).toBe('fireworks');
  });

  it('builds requests to Fireworks API with the accounts/fireworks/models prefix', () => {
    const req = fireworksAdapter.buildRequest(makeParams({ model: 'deepseek-v3p2' }));
    expect(req.url).toContain('api.fireworks.ai');
    expect(req.url).toContain('/chat/completions');
    const body = JSON.parse(req.body);
    expect(body.model).toBe('accounts/fireworks/models/deepseek-v3p2');
  });

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

  it('passes function tools through unchanged (chat-completions wrapper shape)', () => {
    const req = fireworksAdapter.buildRequest(makeParams({
      model: 'deepseek-v3p2',
      tools: [{
        type: 'function',
        function: { name: 'test', description: 'Test tool' },
      }],
    }));
    const body = JSON.parse(req.body);
    expect(body.tools).toEqual([
      { type: 'function', function: { name: 'test', description: 'Test tool' } },
    ]);
  });

  it('strips content_items field from output (chat-completions doesn\'t accept it)', () => {
    const req = fireworksAdapter.buildRequest(makeParams({
      model: 'deepseek-v3p2',
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
});

describe('chat-completions parseStream (shared)', () => {
  it('yields thinking chunk for delta.reasoning_content (Fireworks)', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { reasoning_content: 'DeepSeek reasoning...' }, index: 0 }] },
      { choices: [{ delta: { content: 'Result' }, index: 0, finish_reason: 'stop' }] },
    ]);
    const chunks = await collectChunks(fireworksAdapter.parseStream(response));
    expect(chunks[0]).toEqual({ type: 'thinking', text: 'DeepSeek reasoning...' });
    expect(chunks[1]).toEqual({ type: 'text', text: 'Result' });
  });

  it('yields cache_read_input from usage.prompt_tokens_details.cached_tokens', async () => {
    const response = mockSSEResponse([
      { choices: [{ delta: { content: 'Hello' }, index: 0, finish_reason: 'stop' }] },
      { usage: { prompt_tokens: 500, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 300 } } },
    ]);
    const chunks = await collectChunks(fireworksAdapter.parseStream(response));
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
    const chunks = await collectChunks(fireworksAdapter.parseStream(response));
    const usage = chunks.find(c => c.type === 'usage');
    expect(usage).toBeDefined();
    expect(usage!.cache_read_input).toBeUndefined();
  });
});

describe('fireworksAdapter thinking level matrix', () => {
  // Adapter just translates ThinkingLevel → reasoning_effort. Catalog/proxy
  // decides which models receive a level (instruct models would 400 with
  // "non-reasoning model does not support reasoning_effort" — that's covered
  // by the catalog-gating contract test).
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

describe('chat-completions factory: temperature handling', () => {
  // Custom (third-party) chat-completions endpoints don't need the
  // gpt-5/o-series temperature drop (Fireworks accepts arbitrary
  // temperatures). The OpenAI rollback factory (createOpenAICompletionsAdapter)
  // does — covered separately.
  it('passes custom temperature through for non-OpenAI models', () => {
    const req = fireworksAdapter.buildRequest(makeParams({
      model: 'deepseek-v3p2',
      temperature: 0.7,
    }));
    const body = JSON.parse(req.body);
    expect(body.temperature).toBe(0.7);
  });
});
