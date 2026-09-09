/**
 * Live Adapter Integration Tests
 *
 * Calls real LLM provider APIs via the adapter layer to verify end-to-end
 * request building, streaming, and parsing. Uses the cheapest available models
 * and minimal token counts to keep costs under control.
 *
 * Requires API keys in portal's .env file (loaded via dotenv).
 * Skipped in CI or when keys are missing.
 *
 * WHAT COUNTS AS A FAILURE HERE. A suite run on a developer machine is gated
 * on a FUNDED third-party account, and the state of that account is not a
 * property of this repository. So a provider refusal is sorted by the
 * provider's own error code, never by prose:
 *
 *   - no key at all: the describe is skipped up front (`describe.skipIf`).
 *   - auth refused (401/403, expired or revoked key): SKIPPED, with the reason.
 *   - billing or quota refused (`insufficient_quota`,
 *     `credit_balance_exhausted`, ...): SKIPPED, quoting the provider's message.
 *     Observed 2026-09-07: OpenAI answered HTTP 200 and then put
 *     "You have no credits remaining" in the SSE stream, which used to be
 *     reported as a defect in the adapter.
 *   - anything else, in particular a 400 about a malformed request: FAILS.
 *     That is the contract this file exists to check.
 *
 * These are real `ctx.skip()` skips that show in the summary, not a silent
 * pass: a green test that never reached the provider is the dishonest shape.
 */

import { describe, it, expect } from 'vitest';
import { config } from 'dotenv';
import { resolve } from 'path';
import { anthropicAdapter } from '../../src/adapters/anthropic';
import { openaiAdapter, xaiAdapter } from '../../src/adapters/responses';
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

/**
 * The smallest output budget every provider here accepts. The OpenAI Responses
 * API rejects `max_output_tokens` below 16 with
 * `integer_below_min_value` (a 400, observed 2026-09-07); Anthropic and Google
 * have no floor that high. The one-word replies these tests ask for fit
 * either way, so the budget is the provider minimum rather than the smallest
 * number that used to work.
 */
const MIN_MAX_TOKENS = 16;

function makeParams(overrides: Partial<AdapterRequestParams> & { apiKey: string }): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Reply with exactly one word: hello' }],
    model: 'test',
    maxTokens: MIN_MAX_TOKENS,
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

/**
 * A provider refusal that is about the ACCOUNT, not about this repository.
 * Thrown by `callAdapter`, turned into a real skip by `itLive`.
 */
class ProviderSkip extends Error {
  constructor(readonly kind: 'auth' | 'billing', msg: string) {
    super(msg);
    this.name = 'ProviderSkip';
  }
}

/**
 * Provider error codes that mean "the account cannot pay for this call".
 * Matched against the `code`, `type` and `status` fields of the provider's
 * error object, whether it arrived as a non-2xx JSON body or inside the SSE
 * stream (`ResponsesStreamError.code` / `.errorType`).
 *
 * A `rate_limit_exceeded` is deliberately NOT here: it can be this suite
 * hammering the provider, which is ours to fix.
 */
const BILLING_ERROR_CODES = new Set([
  // OpenAI. `insufficient_quota` is the classic 429 code and the `type` on the
  // Responses-API stream error; `credit_balance_exhausted` is the Responses-API
  // `code` that came with "You have no credits remaining".
  'insufficient_quota',
  'credit_balance_exhausted',
  'billing_hard_limit_reached',
  'billing_not_active',
  // Anthropic.
  'billing_error',
  // Google: the `status` of a quota 429.
  'RESOURCE_EXHAUSTED',
]);

/** Every code-like field of a provider error body, whatever the provider's envelope. */
function providerErrorCodes(errorBody: string): string[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(errorBody);
  } catch {
    return [];
  }
  // OpenAI/Anthropic: `{ error: { code, type } }`; Google: `{ error: { code, status } }`
  // or an array of those.
  const envelopes = Array.isArray(parsed) ? parsed : [parsed];
  const codes: string[] = [];
  for (const envelope of envelopes) {
    const err = (envelope as { error?: Record<string, unknown> })?.error;
    if (!err) continue;
    for (const key of ['code', 'type', 'status']) {
      const value = err[key];
      if (typeof value === 'string') codes.push(value);
    }
  }
  return codes;
}

/** The provider's own message out of an error body, for the skip reason. */
function providerErrorMessage(errorBody: string): string {
  try {
    const parsed = JSON.parse(errorBody) as { error?: { message?: unknown } };
    if (typeof parsed?.error?.message === 'string') return parsed.error.message;
  } catch {
    // fall through to the raw body
  }
  return errorBody.slice(0, 200);
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
      throw new ProviderSkip('auth', `${adapter.name} auth failed (${response.status}), key may be expired`);
    }
    const billing = providerErrorCodes(errorBody).find((code) => BILLING_ERROR_CODES.has(code));
    if (billing) {
      throw new ProviderSkip(
        'billing',
        `${adapter.name} refused for billing/quota (${response.status} ${billing}): ${providerErrorMessage(errorBody)}`,
      );
    }
    // Everything else, a 400 about the request shape above all, is ours.
    throw new Error(`${adapter.name} API ${response.status}: ${errorBody.slice(0, 500)}`);
  }
  try {
    return await collectStream(adapter, response);
  } catch (e) {
    // The Responses API can accept the request (200) and only then report the
    // account is empty, inside the stream. The adapter keeps the provider's
    // code and type on the error so this is a code match, not a regex on prose.
    const { code, errorType } = e as { code?: string; errorType?: string };
    const billing = [code, errorType].find((c) => c !== undefined && BILLING_ERROR_CODES.has(c));
    if (billing) {
      throw new ProviderSkip(
        'billing',
        `${adapter.name} refused for billing/quota in-stream (${billing}): ${(e as Error).message}`,
      );
    }
    throw e;
  }
}

/**
 * Wrapper: a `ProviderSkip` becomes a real vitest skip with the provider's
 * message in the log, so an expired key or an unfunded account reads as
 * "not run here", never as a defect and never as a pass. Anything else fails.
 */
function itLive(name: string, fn: () => Promise<void>, timeout?: number) {
  it(name, async (ctx) => {
    try {
      await fn();
    } catch (e) {
      if (e instanceof ProviderSkip) {
        console.log(`  SKIPPED (${e.kind}): ${e.message}`);
        ctx.skip();
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
