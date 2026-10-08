/**
 * Cache legs and cache keys on the other adapters.
 *
 * OpenAI reports cache WRITES as their own count (`cache_write_tokens`, a
 * priced leg on newer models), on the Responses and the Chat Completions
 * shapes; both parsers carry it as `cache_creation_input`. A caller's
 * already-hashed `cacheKey` goes out as `prompt_cache_key` on the OpenAI
 * adapters only. Gemini reports thinking tokens beside the candidates and
 * bills them as output, so the parser sums them.
 */

import { describe, it, expect } from 'vitest';
import { openaiAdapter, xaiAdapter } from '../../../src/adapters/responses.js';
import { createOpenAICompletionsAdapter, fireworksAdapter } from '../../../src/adapters/completions.js';
import { googleAdapter } from '../../../src/adapters/google.js';
import type { AdapterRequestParams, AdapterChunk } from '../../../src/adapters/types.js';

function mockSSEResponse(events: unknown[], done = false): Response {
  const lines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + (done ? 'data: [DONE]\n\n' : '');
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

async function usageOf(gen: AsyncGenerator<AdapterChunk>): Promise<Extract<AdapterChunk, { type: 'usage' }> | undefined> {
  for await (const c of gen) if (c.type === 'usage') return c;
  return undefined;
}

function makeParams(overrides: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'openai/gpt-6-luna',
    apiKey: 'test-key',
    ...overrides,
  };
}

describe('prompt_cache_key', () => {
  it('the Responses adapter sends the caller\'s cache key, and nothing without one', () => {
    const withKey = JSON.parse(openaiAdapter.buildRequest(makeParams({ cacheKey: '9f2a7c1e4b8d' })).body);
    expect(withKey.prompt_cache_key).toBe('9f2a7c1e4b8d');
    const without = JSON.parse(openaiAdapter.buildRequest(makeParams()).body);
    expect(without.prompt_cache_key).toBeUndefined();
  });

  it('the Chat Completions rollback adapter sends it too', () => {
    const adapter = createOpenAICompletionsAdapter();
    const body = JSON.parse(adapter.buildRequest(makeParams({ cacheKey: '9f2a7c1e4b8d' })).body);
    expect(body.prompt_cache_key).toBe('9f2a7c1e4b8d');
    expect(JSON.parse(adapter.buildRequest(makeParams()).body).prompt_cache_key).toBeUndefined();
  });

  it('the xAI and Fireworks adapters never send it', () => {
    expect(JSON.parse(xaiAdapter.buildRequest(makeParams({ model: 'xai/grok-4.3', cacheKey: 'abc' })).body).prompt_cache_key).toBeUndefined();
    expect(JSON.parse(fireworksAdapter.buildRequest(makeParams({ model: 'fireworks/kimi-k3', cacheKey: 'abc' })).body).prompt_cache_key).toBeUndefined();
  });
});

describe('OpenAI cache writes', () => {
  it('Responses: input_tokens_details.cache_write_tokens becomes cache_creation_input beside cached_tokens', async () => {
    const usage = await usageOf(openaiAdapter.parseStream(mockSSEResponse([
      { type: 'response.output_text.delta', delta: 'hi' },
      { type: 'response.completed', response: { usage: { input_tokens: 12_600, output_tokens: 30, input_tokens_details: { cached_tokens: 0, cache_write_tokens: 12_288 } } } },
    ])));
    expect(usage).toEqual({ type: 'usage', input: 12_600, output: 30, cache_creation_input: 12_288 });
  });

  it('Responses: a usage without the field carries no write leg', async () => {
    const usage = await usageOf(openaiAdapter.parseStream(mockSSEResponse([
      { type: 'response.completed', response: { usage: { input_tokens: 10, output_tokens: 2, input_tokens_details: { cached_tokens: 4 } } } },
    ])));
    expect(usage).toEqual({ type: 'usage', input: 10, output: 2, cache_read_input: 4 });
  });

  it('Chat Completions: prompt_tokens_details.cache_write_tokens becomes cache_creation_input', async () => {
    const adapter = createOpenAICompletionsAdapter();
    const usage = await usageOf(adapter.parseStream(mockSSEResponse([
      { choices: [{ delta: { content: 'Hello' }, index: 0, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 500, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 300, cache_write_tokens: 150 } } },
    ], true)));
    expect(usage).toEqual({ type: 'usage', input: 500, output: 20, cache_read_input: 300, cache_creation_input: 150 });
  });
});

describe('Gemini thinking tokens', () => {
  it('bills thoughtsTokenCount as output beside the candidates', async () => {
    const usage = await usageOf(googleAdapter.parseStream(mockSSEResponse([
      { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
      { usageMetadata: { promptTokenCount: 900, candidatesTokenCount: 150, thoughtsTokenCount: 50, toolUsePromptTokenCount: 2400 } },
    ])));
    expect(usage).toMatchObject({ input: 900, output: 200 });
  });

  it('reads the snake_case spelling too, and leaves output alone when no thoughts are reported', async () => {
    const snake = await usageOf(googleAdapter.parseStream(mockSSEResponse([
      { usageMetadata: { prompt_token_count: 10, candidates_token_count: 5, thoughts_token_count: 7 } },
    ])));
    expect(snake).toMatchObject({ input: 10, output: 12 });
    const none = await usageOf(googleAdapter.parseStream(mockSSEResponse([
      { usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 } },
    ])));
    expect(none).toMatchObject({ input: 10, output: 5 });
  });
});
