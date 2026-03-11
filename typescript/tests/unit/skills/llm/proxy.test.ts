/**
 * LLMProxySkill Unit Tests
 */

import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { LLMProxySkill } from '../../../../src/skills/llm/proxy/skill.js';
import type { Context } from '../../../../src/core/types.js';

// ---------------------------------------------------------------------------
// Mock UAMPClient
// ---------------------------------------------------------------------------

const mockConnect = vi.fn().mockResolvedValue(undefined);
const mockSendInput = vi.fn().mockResolvedValue(undefined);
const mockSendResponse = vi.fn().mockResolvedValue(undefined);
const mockSendPayment = vi.fn().mockResolvedValue(undefined);
const mockClose = vi.fn();
const mockOn = vi.fn();

vi.mock('../../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation((config: unknown) => {
    const instance = {
      config,
      connect: mockConnect,
      sendInput: mockSendInput,
      sendResponse: mockSendResponse,
      sendPayment: mockSendPayment,
      close: mockClose,
      on: mockOn,
    };
    return instance;
  }),
}));

// Re-import after mock so we can inspect constructor calls
import { UAMPClient } from '../../../../src/uamp/client.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeContext(overrides: Partial<Context> = {}): Context {
  return {
    get: vi.fn(() => undefined),
    set: vi.fn(),
    delete: vi.fn(),
    signal: undefined as unknown as AbortSignal,
    auth: { authenticated: false },
    payment: { token: undefined },
    metadata: {},
    ...overrides,
  } as unknown as Context;
}

function triggerClientEvent(eventName: string, ...args: unknown[]): void {
  for (const call of mockOn.mock.calls) {
    if (call[0] === eventName) {
      call[1](...args);
    }
  }
}

async function collectEvents(
  gen: AsyncGenerator<unknown, void, unknown>,
): Promise<unknown[]> {
  const events: unknown[] = [];
  for await (const event of gen) {
    events.push(event);
  }
  return events;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('LLMProxySkill', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockConnect.mockResolvedValue(undefined);

    // Default: simulate a successful done response after sendResponse
    mockSendResponse.mockImplementation(async () => {
      queueMicrotask(() => {
        triggerClientEvent('done', {
          output: [{ type: 'text', text: 'Hello from proxy' }],
          usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          id: 'resp_123',
          status: 'completed',
        });
      });
    });
  });

  // ========================================================================
  // 1. Constructor defaults
  // ========================================================================

  it('creates LLMProxySkill with default config', () => {
    const skill = new LLMProxySkill();
    expect(skill.name).toBe('llm-proxy');
    expect((skill as any).proxyUrl).toBe('wss://robutler.ai/llm');
    expect((skill as any).modelConfig.model).toBeUndefined();
  });

  it('accepts custom config', () => {
    const skill = new LLMProxySkill({
      proxyUrl: 'ws://custom:8080/llm',
      model: 'gpt-4',
      temperature: 0.5,
      max_tokens: 2048,
      name: 'my-proxy',
    });
    expect(skill.name).toBe('my-proxy');
    expect((skill as any).proxyUrl).toBe('ws://custom:8080/llm');
    expect((skill as any).modelConfig.model).toBe('gpt-4');
  });

  // ========================================================================
  // 2. Connects to UAMP proxy and sends response.create
  // ========================================================================

  it('connects to UAMP proxy and sends session.create', async () => {
    const skill = new LLMProxySkill({ proxyUrl: 'ws://test:3000/llm' });
    const context = makeContext();

    const events = [
      {
        type: 'session.create' as const,
        event_id: 'e1',
        session: { modalities: ['text'], instructions: 'Be helpful' },
      },
      { type: 'input.text' as const, event_id: 'e2', text: 'Hello', role: 'user' },
    ];

    await collectEvents(skill['processUAMP'](events, context));

    expect(UAMPClient).toHaveBeenCalledTimes(1);
    const constructorArg = (UAMPClient as unknown as Mock).mock.calls[0][0];
    expect(constructorArg.url).toBe('ws://test:3000/llm');
    expect(constructorArg.session.modalities).toEqual(['text']);
    expect(mockConnect).toHaveBeenCalledOnce();
    expect(mockSendResponse).toHaveBeenCalledOnce();

    const responseArg = mockSendResponse.mock.calls[0][0];
    expect(responseArg.messages).toEqual([
      { role: 'system', content: 'Be helpful' },
      { role: 'user', content: 'Hello' },
    ]);
    expect(mockClose).toHaveBeenCalledOnce();
  });

  // ========================================================================
  // 3. Streams response deltas from proxy
  // ========================================================================

  it('streams response deltas from proxy', async () => {
    mockSendResponse.mockImplementation(async () => {
      queueMicrotask(() => {
        // Skill creates response.delta from 'delta' and 'toolCall' events
        triggerClientEvent('delta', 'streamed text');
        triggerClientEvent('toolCall', { id: 'tc1', name: 'search', arguments: '{"q":"x"}' });
        triggerClientEvent('done', {
          output: [{ type: 'text', text: 'streamed text' }],
          usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
          id: 'resp_456',
          status: 'completed',
        });
      });
    });

    const skill = new LLMProxySkill();
    const context = makeContext();
    const events = [
      { type: 'input.text' as const, event_id: 'e1', text: 'test', role: 'user' },
    ];

    const output = await collectEvents(skill['processUAMP'](events, context));

    const types = output.map((e: any) => e.type);
    expect(types).toContain('response.created');
    expect(types).toContain('response.done');

    const textDelta = output.find(
      (e: any) => e.type === 'response.delta' && e.delta?.type === 'text',
    ) as any;
    expect(textDelta).toBeDefined();
    expect(textDelta.delta.text).toBe('streamed text');

    const toolDelta = output.find(
      (e: any) => e.type === 'response.delta' && e.delta?.type === 'tool_call',
    ) as any;
    expect(toolDelta).toBeDefined();
    expect(toolDelta.delta.tool_call.name).toBe('search');
  });

  // ========================================================================
  // 4. Forwards AbortSignal to UAMPClient
  // ========================================================================

  it('forwards AbortSignal to UAMPClient', async () => {
    const controller = new AbortController();
    const skill = new LLMProxySkill();
    const context = makeContext({ signal: controller.signal } as any);

    const events = [
      { type: 'input.text' as const, event_id: 'e1', text: 'hello', role: 'user' },
    ];

    await collectEvents(skill['processUAMP'](events, context));

    const constructorArg = (UAMPClient as unknown as Mock).mock.calls[0][0];
    expect(constructorArg.signal).toBe(controller.signal);
  });

  // ========================================================================
  // 5. Handles payment.required by resubmitting token
  // ========================================================================

  it('handles payment.required by resubmitting token', async () => {
    const paymentToken = 'pay_tok_abc';

    mockSendResponse.mockImplementation(async () => {
      queueMicrotask(() => {
        triggerClientEvent('paymentRequired', {
          amount: '0.50',
          currency: 'USD',
          schemes: [{ scheme: 'token' }],
        });
        // After payment, proxy responds with done
        queueMicrotask(() => {
          triggerClientEvent('done', {
            output: [{ type: 'text', text: 'paid response' }],
            id: 'resp_paid',
            status: 'completed',
          });
        });
      });
    });

    const skill = new LLMProxySkill();
    const context = makeContext({
      payment: { token: paymentToken },
    } as any);

    const events = [
      { type: 'input.text' as const, event_id: 'e1', text: 'hello', role: 'user' },
    ];

    await collectEvents(skill['processUAMP'](events, context));

    expect(mockSendPayment).toHaveBeenCalledWith({
      scheme: 'token',
      amount: '0.50',
      token: paymentToken,
    });
  });

  // ========================================================================
  // 6. Handles proxy connection error gracefully
  // ========================================================================

  it('handles proxy connection error gracefully', async () => {
    mockConnect.mockRejectedValue(new Error('Connection refused'));

    const skill = new LLMProxySkill();
    const context = makeContext();
    const events = [
      { type: 'input.text' as const, event_id: 'e1', text: 'hello', role: 'user' },
    ];

    const output = await collectEvents(skill['processUAMP'](events, context));

    const errorEvent = output.find((e: any) => e.type === 'response.error') as any;
    expect(errorEvent).toBeDefined();
    expect(errorEvent.error.message).toContain('Connection refused');
    expect(mockClose).toHaveBeenCalled();
  });

  // ========================================================================
  // 7. Uses payment token from context
  // ========================================================================

  it('uses payment token from context', async () => {
    const skill = new LLMProxySkill();
    const context = makeContext({
      payment: { token: 'ctx_payment_token' },
    } as any);

    const events = [
      { type: 'input.text' as const, event_id: 'e1', text: 'hello', role: 'user' },
    ];

    await collectEvents(skill['processUAMP'](events, context));

    const constructorArg = (UAMPClient as unknown as Mock).mock.calls[0][0];
    expect(constructorArg.paymentToken).toBe('ctx_payment_token');
  });

  // ========================================================================
  // 8. session.create extracts instructions and tools into sendResponse
  // ========================================================================

  it('extracts instructions from session.create into sendResponse messages', async () => {
    const skill = new LLMProxySkill();
    const context = makeContext();

    const events = [
      {
        type: 'session.create' as const,
        event_id: 'e1',
        session: {
          modalities: ['text'],
          instructions: 'System prompt here',
          tools: [{ name: 'my_tool', description: 'A tool', parameters: {} }],
        },
      },
      { type: 'input.text' as const, event_id: 'e2', text: 'User message', role: 'user' },
    ];

    await collectEvents(skill['processUAMP'](events, context));

    expect(mockSendResponse).toHaveBeenCalledOnce();
    const responseArg = mockSendResponse.mock.calls[0][0];
    expect(responseArg.messages[0]).toEqual({ role: 'system', content: 'System prompt here' });
    expect(responseArg.messages[1]).toEqual({ role: 'user', content: 'User message' });
    expect(responseArg.tools).toHaveLength(1);
    expect(responseArg.tools[0].name).toBe('my_tool');
  });
});
