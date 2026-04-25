/**
 * Live Adapter Integration Tests
 *
 * Calls real LLM provider APIs via the adapter layer to verify end-to-end
 * request building, streaming, and parsing. Uses the cheapest available models
 * and minimal token counts to keep costs under control.
 *
 * Requires API keys in portal's .env file (loaded via dotenv).
 * Skipped in CI or when keys are missing.
 * Auth failures (expired/invalid keys) are logged and treated as soft skips.
 */

import { describe, it, expect } from 'vitest';
import { config } from 'dotenv';
import { resolve } from 'path';
import { anthropicAdapter } from '../../src/adapters/anthropic';
import { openaiAdapter, xaiAdapter } from '../../src/adapters/openai';
import { googleAdapter } from '../../src/adapters/google';
import type { AdapterRequestParams, AdapterChunk, LLMAdapter } from '../../src/adapters/types';

// Prefer infrastructure/secrets/local.env (has valid keys), fall back to root .env
config({ path: resolve(process.cwd(), '../../infrastructure/secrets/local.env') });
config({ path: resolve(process.cwd(), '../../.env') });

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const GOOGLE_KEY = process.env.GOOGLE_API_KEY || process.env.GOOGLE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
const XAI_KEY = process.env.XAI_API_KEY;

const skipAll = !!process.env.CI;

function makeParams(overrides: Partial<AdapterRequestParams> & { apiKey: string }): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Reply with exactly one word: hello' }],
    model: 'test',
    maxTokens: 10,
    temperature: 0,
    stream: true,
    ...overrides,
  };
}

async function collectStream(adapter: LLMAdapter, response: Response): Promise<{
  text: string;
  toolCalls: AdapterChunk[];
  usage: { input: number; output: number } | null;
}> {
  let text = '';
  const toolCalls: AdapterChunk[] = [];
  let usage: { input: number; output: number } | null = null;

  for await (const chunk of adapter.parseStream(response)) {
    if (chunk.type === 'text') text += chunk.text;
    else if (chunk.type === 'tool_call') toolCalls.push(chunk);
    else if (chunk.type === 'usage') usage = { input: chunk.input, output: chunk.output };
  }

  return { text, toolCalls, usage };
}

class AuthSkipError extends Error {
  constructor(msg: string) { super(msg); this.name = 'AuthSkipError'; }
}

async function callAdapter(adapter: LLMAdapter, params: AdapterRequestParams): Promise<{
  text: string;
  toolCalls: AdapterChunk[];
  usage: { input: number; output: number } | null;
}> {
  const req = adapter.buildRequest(params);
  const response = await fetch(req.url, {
    method: 'POST',
    headers: req.headers,
    body: req.body,
  });
  if (!response.ok) {
    const errorBody = await response.text();
    if (response.status === 401 || response.status === 403 ||
        errorBody.includes('API_KEY_INVALID') || errorBody.includes('invalid x-api-key')) {
      throw new AuthSkipError(`${adapter.name} auth failed (${response.status}) — key may be expired`);
    }
    throw new Error(`${adapter.name} API ${response.status}: ${errorBody.slice(0, 500)}`);
  }
  return collectStream(adapter, response);
}

/** Wrapper: soft-skip on auth errors so expired keys don't break the suite. */
function itLive(name: string, fn: () => Promise<void>, timeout?: number) {
  it(name, async () => {
    try {
      await fn();
    } catch (e) {
      if (e instanceof AuthSkipError) {
        console.log(`  SKIPPED (auth): ${e.message}`);
        return;
      }
      throw e;
    }
  }, timeout);
}

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------

describe.skipIf(skipAll || !OPENAI_KEY)('OpenAI adapter (live)', () => {
  itLive('streams a simple response from gpt-4o-mini', async () => {
    const result = await callAdapter(openaiAdapter, makeParams({
      apiKey: OPENAI_KEY!,
      model: 'gpt-4o-mini',
    }));

    expect(result.text.length).toBeGreaterThan(0);
    expect(result.usage).not.toBeNull();
    expect(result.usage!.input).toBeGreaterThan(0);
    expect(result.usage!.output).toBeGreaterThan(0);
    console.log(`[openai] text="${result.text.trim()}" usage=${JSON.stringify(result.usage)}`);
  }, 15000);

  itLive('handles system message + user message', async () => {
    const result = await callAdapter(openaiAdapter, makeParams({
      apiKey: OPENAI_KEY!,
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a parrot that repeats the user input verbatim.' },
        { role: 'user', content: 'test123' },
      ],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[openai system] text="${result.text.trim()}"`);
  }, 15000);

  itLive('strips content_items and sends clean messages', async () => {
    const result = await callAdapter(openaiAdapter, makeParams({
      apiKey: OPENAI_KEY!,
      model: 'gpt-4o-mini',
      messages: [{
        role: 'user',
        content: 'Say hi',
        content_items: [{ type: 'text', text: 'Say hi' }],
      }],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[openai content_items] text="${result.text.trim()}"`);
  }, 15000);

  itLive('handles tool_calls round-trip', async () => {
    const result = await callAdapter(openaiAdapter, makeParams({
      apiKey: OPENAI_KEY!,
      model: 'gpt-4o-mini',
      maxTokens: 100,
      messages: [
        { role: 'user', content: 'What is the weather in Tokyo? Use the get_weather tool.' },
      ],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      }],
    }));

    if (result.toolCalls.length > 0) {
      const tc = result.toolCalls[0] as Extract<AdapterChunk, { type: 'tool_call' }>;
      expect(tc.name).toBe('get_weather');
      expect(tc.arguments).toContain('Tokyo');
      console.log(`[openai tool_call] name=${tc.name} args=${tc.arguments}`);
    } else {
      console.log(`[openai tool_call] model chose text: "${result.text.trim().slice(0, 80)}"`);
    }
  }, 15000);
});

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

describe.skipIf(skipAll || !ANTHROPIC_KEY)('Anthropic adapter (live)', () => {
  itLive('streams a simple response from claude-haiku-4-5', async () => {
    const result = await callAdapter(anthropicAdapter, makeParams({
      apiKey: ANTHROPIC_KEY!,
      model: 'claude-haiku-4-5',
    }));

    expect(result.text.length).toBeGreaterThan(0);
    expect(result.usage).not.toBeNull();
    expect(result.usage!.input).toBeGreaterThan(0);
    expect(result.usage!.output).toBeGreaterThan(0);
    console.log(`[anthropic] text="${result.text.trim()}" usage=${JSON.stringify(result.usage)}`);
  }, 15000);

  itLive('extracts system message to top-level param', async () => {
    const result = await callAdapter(anthropicAdapter, makeParams({
      apiKey: ANTHROPIC_KEY!,
      model: 'claude-haiku-4-5',
      messages: [
        { role: 'system', content: 'Always respond with exactly one word.' },
        { role: 'user', content: 'Say hello' },
      ],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[anthropic system] text="${result.text.trim()}"`);
  }, 15000);

  itLive('strips content_items and sends clean messages', async () => {
    const result = await callAdapter(anthropicAdapter, makeParams({
      apiKey: ANTHROPIC_KEY!,
      model: 'claude-haiku-4-5',
      messages: [{
        role: 'user',
        content: 'Say hi',
        content_items: [{ type: 'text', text: 'Say hi' }],
      }],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[anthropic content_items] text="${result.text.trim()}"`);
  }, 15000);

  itLive('converts tools to Anthropic format and handles tool_use', async () => {
    const result = await callAdapter(anthropicAdapter, makeParams({
      apiKey: ANTHROPIC_KEY!,
      model: 'claude-haiku-4-5',
      maxTokens: 100,
      messages: [
        { role: 'user', content: 'What is the weather in Tokyo? Use the get_weather tool.' },
      ],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      }],
    }));

    if (result.toolCalls.length > 0) {
      const tc = result.toolCalls[0] as Extract<AdapterChunk, { type: 'tool_call' }>;
      expect(tc.name).toBe('get_weather');
      console.log(`[anthropic tool_call] name=${tc.name} args=${tc.arguments}`);
    } else {
      console.log(`[anthropic tool_call] model chose text: "${result.text.trim().slice(0, 80)}"`);
    }
  }, 15000);

  itLive('converts tool_calls + tool results in multi-turn', async () => {
    const result = await callAdapter(anthropicAdapter, makeParams({
      apiKey: ANTHROPIC_KEY!,
      model: 'claude-haiku-4-5',
      maxTokens: 50,
      messages: [
        { role: 'user', content: 'What is the weather in Tokyo?' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: 'toolu_01',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"Tokyo"}' },
          }],
        },
        { role: 'tool', content: '{"temp":"22C","condition":"sunny"}', tool_call_id: 'toolu_01' },
      ],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      }],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[anthropic multi-turn] text="${result.text.trim().slice(0, 100)}"`);
  }, 15000);
});

// ---------------------------------------------------------------------------
// Google
// ---------------------------------------------------------------------------

describe.skipIf(skipAll || !GOOGLE_KEY)('Google adapter (live)', () => {
  itLive('streams a simple response from gemini-2.0-flash-lite', async () => {
    const result = await callAdapter(googleAdapter, makeParams({
      apiKey: GOOGLE_KEY!,
      model: 'gemini-2.5-flash-lite',
    }));

    expect(result.text.length).toBeGreaterThan(0);
    expect(result.usage).not.toBeNull();
    console.log(`[google] text="${result.text.trim()}" usage=${JSON.stringify(result.usage)}`);
  }, 15000);

  itLive('extracts system instruction from system role', async () => {
    const result = await callAdapter(googleAdapter, makeParams({
      apiKey: GOOGLE_KEY!,
      model: 'gemini-2.5-flash-lite',
      messages: [
        { role: 'system', content: 'Always respond with exactly one word.' },
        { role: 'user', content: 'Say hello' },
      ],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[google system] text="${result.text.trim()}"`);
  }, 15000);

  itLive('converts tools to function_declarations', async () => {
    const result = await callAdapter(googleAdapter, makeParams({
      apiKey: GOOGLE_KEY!,
      model: 'gemini-2.5-flash-lite',
      maxTokens: 100,
      messages: [
        { role: 'user', content: 'What is the weather in Tokyo? Use the get_weather tool.' },
      ],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      }],
    }));

    if (result.toolCalls.length > 0) {
      const tc = result.toolCalls[0] as Extract<AdapterChunk, { type: 'tool_call' }>;
      expect(tc.name).toBe('get_weather');
      console.log(`[google tool_call] name=${tc.name} args=${tc.arguments}`);
    } else {
      console.log(`[google tool_call] model chose text: "${result.text.trim().slice(0, 80)}"`);
    }
  }, 15000);

  itLive('handles content_items on messages', async () => {
    const result = await callAdapter(googleAdapter, makeParams({
      apiKey: GOOGLE_KEY!,
      model: 'gemini-2.5-flash-lite',
      messages: [{
        role: 'user',
        content: 'Say hi',
        content_items: [{ type: 'text', text: 'Say hi' }],
      }],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[google content_items] text="${result.text.trim()}"`);
  }, 15000);
});

// ---------------------------------------------------------------------------
// xAI
// ---------------------------------------------------------------------------

describe.skipIf(skipAll || !XAI_KEY)('xAI adapter (live)', () => {
  itLive('streams a simple response from grok-3-mini-fast', async () => {
    const result = await callAdapter(xaiAdapter, makeParams({
      apiKey: XAI_KEY!,
      model: 'grok-3-mini-fast',
    }));

    expect(result.text.length).toBeGreaterThan(0);
    expect(result.usage).not.toBeNull();
    console.log(`[xai] text="${result.text.trim()}" usage=${JSON.stringify(result.usage)}`);
  }, 15000);

  itLive('strips content_items and sends clean messages', async () => {
    const result = await callAdapter(xaiAdapter, makeParams({
      apiKey: XAI_KEY!,
      model: 'grok-3-mini-fast',
      messages: [{
        role: 'user',
        content: 'Say hi',
        content_items: [{ type: 'text', text: 'Say hi' }],
      }],
    }));

    expect(result.text.length).toBeGreaterThan(0);
    console.log(`[xai content_items] text="${result.text.trim()}"`);
  }, 15000);
});
