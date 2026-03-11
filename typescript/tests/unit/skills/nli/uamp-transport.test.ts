/**
 * NLI UAMP Transport Unit Tests
 *
 * Tests the UAMP WebSocket transport path within NLISkill, including
 * fallback to HTTP when UAMP fails in 'auto' mode.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type Mock } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context } from '../../../../src/core/types.js';

// ---------------------------------------------------------------------------
// Mock UAMPClient - factory-based so each test gets a fresh mock
// ---------------------------------------------------------------------------

let mockClientInstance: any = null;
const capturedConfigs: any[] = [];

function defaultMockFactory(config: any) {
  capturedConfigs.push(config);
  mockClientInstance = {
    config,
    connect: vi.fn().mockResolvedValue(undefined),
    sendInput: vi.fn().mockResolvedValue(undefined),
    sendPayment: vi.fn().mockResolvedValue(undefined),
    cancel: vi.fn().mockResolvedValue(undefined),
    close: vi.fn(),
    on: vi.fn(),
    _handlers: new Map<string, Function[]>(),
  };
  mockClientInstance.on.mockImplementation((event: string, handler: Function) => {
    if (!mockClientInstance._handlers.has(event)) {
      mockClientInstance._handlers.set(event, []);
    }
    mockClientInstance._handlers.get(event)!.push(handler);
    return mockClientInstance;
  });
  return mockClientInstance;
}

vi.mock('../../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation((config: any) => defaultMockFactory(config)),
}));

import { UAMPClient } from '../../../../src/uamp/client.js';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function triggerEvent(name: string, ...args: unknown[]): void {
  const handlers = mockClientInstance?._handlers?.get(name) || [];
  for (const h of handlers) h(...args);
}

function makeContext(overrides: Record<string, unknown> = {}): Context {
  return {
    get: vi.fn(() => undefined),
    set: vi.fn(),
    delete: vi.fn(),
    signal: undefined,
    auth: { authenticated: false },
    payment: { token: undefined },
    metadata: {},
    ...overrides,
  } as unknown as Context;
}

async function collectStream(gen: AsyncGenerator<string, void, unknown>): Promise<string[]> {
  const chunks: string[] = [];
  for await (const chunk of gen) {
    chunks.push(chunk);
  }
  return chunks;
}

/**
 * Configure the mock UAMPClient so that `sendInput` triggers the given
 * sequence of events via microtask scheduling.
 */
function configureMockEvents(events: Array<{ name: string; args?: unknown[] }>): void {
  (UAMPClient as unknown as Mock).mockImplementation((config: any) => {
    const inst = defaultMockFactory(config);
    inst.sendInput = vi.fn().mockImplementation(async () => {
      let schedule = () => {
        const ev = events.shift();
        if (!ev) return;
        triggerEvent(ev.name, ...(ev.args || []));
        if (events.length > 0) queueMicrotask(schedule);
      };
      queueMicrotask(schedule);
    });
    return inst;
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('NLI UAMP Transport', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    capturedConfigs.length = 0;
    mockClientInstance = null;
    // Restore the default factory after each test
    (UAMPClient as unknown as Mock).mockImplementation((config: any) => defaultMockFactory(config));
  });

  // ========================================================================
  // 1. Uses UAMP transport when configured
  // ========================================================================

  it('uses UAMP transport when configured', async () => {
    const skill = new NLISkill({
      agentUrl: 'https://example.com/agents/foo',
      transport: 'uamp',
    });

    configureMockEvents([
      { name: 'delta', args: ['hello'] },
      { name: 'done' },
    ]);

    const chunks = await collectStream(
      skill.streamMessageUAMP(
        'https://example.com/agents/foo',
        [{ role: 'user', content: 'hi' }],
      ),
    );

    expect(UAMPClient).toHaveBeenCalled();
    expect(chunks).toContain('hello');
  });

  // ========================================================================
  // 2. Falls back to HTTP when UAMP fails with auto transport
  // ========================================================================

  it('falls back to HTTP when UAMP fails with auto transport', async () => {
    const skill = new NLISkill({
      agentUrl: 'https://example.com/agents/foo',
      transport: 'auto',
      timeout: 5000,
    });

    // Make UAMP fail on connect
    (UAMPClient as unknown as Mock).mockImplementation((config: any) => {
      const inst = defaultMockFactory(config);
      inst.connect = vi.fn().mockRejectedValue(new Error('WS connection failed'));
      return inst;
    });

    // Set up HTTP fallback response (SSE stream)
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"fallback"}}]}\n\n'));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });

    mockFetch.mockResolvedValue({
      ok: true,
      body: stream,
    });

    const chunks = await collectStream(
      skill.streamMessage(
        'https://example.com/agents/foo',
        [{ role: 'user', content: 'hi' }],
      ),
    );

    expect(UAMPClient).toHaveBeenCalled();
    expect(mockFetch).toHaveBeenCalled();
    expect(chunks).toContain('fallback');
  });

  // ========================================================================
  // 3. Forwards payment token in UAMP session
  // ========================================================================

  it('forwards payment token in UAMP session', async () => {
    const skill = new NLISkill({
      agentUrl: 'https://example.com/agents/foo',
      transport: 'uamp',
    });

    configureMockEvents([{ name: 'done' }]);

    const context = makeContext({
      metadata: { paymentToken: 'pay_tok_test' },
    });

    await collectStream(
      skill.streamMessageUAMP(
        'https://example.com/agents/foo',
        [{ role: 'user', content: 'test' }],
        context,
      ),
    );

    expect(capturedConfigs.length).toBeGreaterThan(0);
    expect(capturedConfigs[0].paymentToken).toBe('pay_tok_test');
  });

  // ========================================================================
  // 4. Throws error on payment required without token
  // ========================================================================

  it('throws error on payment required without token', async () => {
    const skill = new NLISkill({
      agentUrl: 'https://example.com/agents/foo',
      transport: 'uamp',
    });

    configureMockEvents([
      {
        name: 'paymentRequired',
        args: [{ amount: '1.00', currency: 'USD', schemes: [{ scheme: 'token' }] }],
      },
    ]);

    const context = makeContext();

    await expect(
      collectStream(
        skill.streamMessageUAMP(
          'https://example.com/agents/foo',
          [{ role: 'user', content: 'test' }],
          context,
        ),
      ),
    ).rejects.toThrow('Payment required');
  });

  // ========================================================================
  // 5. Streams response deltas from target agent
  // ========================================================================

  it('streams response deltas from target agent', async () => {
    const skill = new NLISkill({
      agentUrl: 'https://example.com/agents/foo',
      transport: 'uamp',
    });

    configureMockEvents([
      { name: 'delta', args: ['chunk1'] },
      { name: 'delta', args: ['chunk2'] },
      { name: 'delta', args: ['chunk3'] },
      { name: 'done' },
    ]);

    const chunks = await collectStream(
      skill.streamMessageUAMP(
        'https://example.com/agents/foo',
        [{ role: 'user', content: 'hello' }],
      ),
    );

    expect(chunks).toEqual(['chunk1', 'chunk2', 'chunk3']);
  });

  // ========================================================================
  // 6. Resolves UAMP URL from agent HTTP URL
  // ========================================================================

  it('resolves UAMP URL from agent HTTP URL', async () => {
    const skill = new NLISkill({ transport: 'uamp' });

    configureMockEvents([{ name: 'done' }]);

    await collectStream(
      skill.streamMessageUAMP(
        'https://example.com/agents/myagent',
        [{ role: 'user', content: 'test' }],
      ),
    );

    expect(capturedConfigs.length).toBe(1);
    const url = capturedConfigs[0].url;
    expect(url).toMatch(/^wss:\/\//);
    expect(url).toContain('/agents/myagent/uamp');
  });

  it('resolves ws:// from http://', async () => {
    const skill = new NLISkill({ transport: 'uamp' });

    configureMockEvents([{ name: 'done' }]);

    await collectStream(
      skill.streamMessageUAMP(
        'http://localhost:3000/agents/test',
        [{ role: 'user', content: 'test' }],
      ),
    );

    expect(capturedConfigs[0].url).toMatch(/^ws:\/\//);
    expect(capturedConfigs[0].url).toContain('/agents/test/uamp');
  });

  // ========================================================================
  // Additional: API key appended to UAMP URL
  // ========================================================================

  it('appends apiKey to UAMP URL as query param', async () => {
    const skill = new NLISkill({
      transport: 'uamp',
      apiKey: 'sk-test-key',
    });

    configureMockEvents([{ name: 'done' }]);

    await collectStream(
      skill.streamMessageUAMP(
        'https://example.com/agents/foo',
        [{ role: 'user', content: 'test' }],
      ),
    );

    expect(capturedConfigs[0].url).toContain('token=sk-test-key');
  });

  // ========================================================================
  // Error from UAMP client propagates
  // ========================================================================

  it('propagates error from UAMP client', async () => {
    const skill = new NLISkill({ transport: 'uamp' });

    configureMockEvents([
      { name: 'error', args: [new Error('Proxy exploded')] },
    ]);

    await expect(
      collectStream(
        skill.streamMessageUAMP(
          'https://example.com/agents/foo',
          [{ role: 'user', content: 'test' }],
        ),
      ),
    ).rejects.toThrow('Proxy exploded');
  });
});
