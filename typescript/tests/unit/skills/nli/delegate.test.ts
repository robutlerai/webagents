/**
 * NLI delegate tool Unit Tests
 *
 * Tests the consolidated delegate tool from NLISkill. Uses transport: 'http'
 * to avoid WebSocket mocking. Covers URL resolution, HTTP request format,
 * response handling, and error cases.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context } from '../../../../src/core/types.js';

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockFetch = vi.fn();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeContext(data: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(data));
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined,
    auth: { authenticated: false },
    payment: { token: undefined },
    metadata: {},
  } as unknown as Context;
}

function createSSEStream(content: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const lines =
    content === ''
      ? ['data: [DONE]\n\n']
      : [
          `data: {"choices":[{"delta":{"content":"${content}"}}]}\n\n`,
          'data: [DONE]\n\n',
        ];
  return new ReadableStream({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(line));
      }
      controller.close();
    },
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('NLI delegate tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // ========================================================================
  // 1. Resolves username to URL correctly (@agent -> baseUrl/agents/agent)
  // ========================================================================

  it('resolves username to URL correctly (@agent -> baseUrl/agents/agent)', async () => {
    const baseUrl = 'https://portal.example.com';
    const skill = new NLISkill({
      baseUrl,
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream('hi'),
    });

    const context = makeContext();
    const result = await skill.delegate(
      { agent: '@fundraiser', message: 'hello' },
      context,
    );

    expect(result).toBe('hi');
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      `${baseUrl}/agents/fundraiser/chat/completions`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'hello' }],
          stream: true,
        }),
      }),
    );
  });

  it('resolves bare username (agent) to baseUrl/agents/agent', async () => {
    const baseUrl = 'https://portal.example.com';
    const skill = new NLISkill({
      baseUrl,
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream('ok'),
    });

    const result = await skill.delegate(
      { agent: 'fundraiser', message: 'test' },
      makeContext(),
    );

    expect(result).toBe('ok');
    expect(mockFetch).toHaveBeenCalledWith(
      `${baseUrl}/agents/fundraiser/chat/completions`,
      expect.any(Object),
    );
  });

  // ========================================================================
  // 2. Sends correct HTTP request (POST /chat/completions with messages)
  // ========================================================================

  it('sends correct HTTP request (POST /chat/completions with messages)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      apiKey: 'sk-test-key',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream('response'),
    });

    await skill.delegate(
      { agent: '@myagent', message: 'What is 2+2?' },
      makeContext(),
    );

    expect(mockFetch).toHaveBeenCalledWith(
      'https://portal.example.com/agents/myagent/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          Authorization: 'Bearer sk-test-key',
        }),
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'What is 2+2?' }],
          stream: true,
        }),
      }),
    );
  });

  // ========================================================================
  // 3. Returns agent response text
  // ========================================================================

  it('returns agent response text', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    const expected = 'Here is the analysis you requested.';
    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream(expected),
    });

    const result = await skill.delegate(
      { agent: '@analyst', message: 'Analyze this' },
      makeContext(),
    );

    expect(result).toBe(expected);
  });

  it('concatenates multiple SSE chunks', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
          ),
        );
        controller.enqueue(
          encoder.encode(
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
          ),
        );
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });

    mockFetch.mockResolvedValue({ ok: true, body: stream });

    const result = await skill.delegate(
      { agent: '@agent', message: 'hi' },
      makeContext(),
    );

    expect(result).toBe('Hello world');
  });

  // ========================================================================
  // 4. Returns "(no response)" for empty response
  // ========================================================================

  it('returns "(no response)" for empty response', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream(''),
    });

    const result = await skill.delegate(
      { agent: '@agent', message: 'hi' },
      makeContext(),
    );

    expect(result).toBe('(no response)');
  });

  // ========================================================================
  // 5. Handles non-ok status (throws error)
  // ========================================================================

  it('handles non-ok status (throws error)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    });

    await expect(
      skill.delegate({ agent: '@agent', message: 'hi' }, makeContext()),
    ).rejects.toThrow('NLI request failed: 500 Internal Server Error');
  });

  it('handles 404 status', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    });

    await expect(
      skill.delegate({ agent: '@nonexistent', message: 'hi' }, makeContext()),
    ).rejects.toThrow('NLI request failed: 404 Not Found');
  });

  // ========================================================================
  // 6. Agent URL with full URL is used directly
  // ========================================================================

  it('uses full URL directly when agent contains /', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    const customUrl = 'https://custom.example.com/agents/special';
    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream('from custom'),
    });

    const result = await skill.delegate(
      { agent: customUrl, message: 'hello' },
      makeContext(),
    );

    expect(result).toBe('from custom');
    expect(mockFetch).toHaveBeenCalledWith(
      `${customUrl}/chat/completions`,
      expect.any(Object),
    );
  });

  it('uses full URL directly when agent starts with https://', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    const fullUrl = 'https://robutler.ai/agents/fundraiser';
    mockFetch.mockResolvedValue({
      ok: true,
      body: createSSEStream('from robutler'),
    });

    await skill.delegate(
      { agent: fullUrl, message: 'hi' },
      makeContext(),
    );

    expect(mockFetch).toHaveBeenCalledWith(
      `${fullUrl}/chat/completions`,
      expect.any(Object),
    );
  });

  // ========================================================================
  // 7. Emits tool_progress via context callback when available
  // ========================================================================

  it('emits progress for each chunk when _toolProgressFn is set', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode('data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'),
        );
        controller.enqueue(
          encoder.encode('data: {"choices":[{"delta":{"content":"world"}}]}\n\n'),
        );
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    mockFetch.mockResolvedValue({ ok: true, body: stream });

    const progressFn = vi.fn();
    const context = makeContext({
      _toolProgressFn: progressFn,
      tool_call: { id: 'tc_123' },
    });

    const result = await skill.delegate(
      { agent: '@agent', message: 'hi' },
      context,
    );

    expect(result).toBe('Hello world');
    expect(progressFn).toHaveBeenCalledTimes(2);
    expect(progressFn).toHaveBeenCalledWith('tc_123', 'Hello ');
    expect(progressFn).toHaveBeenCalledWith('tc_123', 'world');
  });

  it('does not call progress when _toolProgressFn is absent', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('hi') });

    const result = await skill.delegate(
      { agent: '@agent', message: 'hi' },
      makeContext(),
    );

    expect(result).toBe('hi');
  });
});

// ===========================================================================
// Sleep tool
// ===========================================================================

describe('NLI sleep tool', () => {
  it('waits for the specified duration', async () => {
    const skill = new NLISkill({ baseUrl: 'https://example.com' });
    const start = Date.now();
    const result = await skill.sleep({ seconds: 0.05 }, makeContext());
    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(40);
    expect(result).toContain('Waited');
  });

  it('caps duration at 30 seconds', async () => {
    const skill = new NLISkill({ baseUrl: 'https://example.com' });
    const start = Date.now();
    const resultPromise = skill.sleep({ seconds: 100 }, makeContext());
    // We can't actually wait 30s in a test, so just verify the capping
    // by checking the return string if it resolves quickly via setTimeout mock
    vi.useFakeTimers();
    const promise = skill.sleep({ seconds: 100 }, makeContext());
    vi.advanceTimersByTime(30_000);
    const result = await promise;
    expect(result).toBe('Waited 30 seconds');
    vi.useRealTimers();
  });

  it('includes reason in the result', async () => {
    vi.useFakeTimers();
    const skill = new NLISkill({ baseUrl: 'https://example.com' });
    const promise = skill.sleep(
      { seconds: 5, reason: 'waiting for agent to recover' },
      makeContext(),
    );
    vi.advanceTimersByTime(5_000);
    const result = await promise;
    expect(result).toBe('Waited 5 seconds: waiting for agent to recover');
    vi.useRealTimers();
  });

  it('clamps negative seconds to 0', async () => {
    const skill = new NLISkill({ baseUrl: 'https://example.com' });
    const result = await skill.sleep({ seconds: -5 }, makeContext());
    expect(result).toBe('Waited 0 seconds');
  });
});
