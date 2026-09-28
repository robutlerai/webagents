/**
 * The LLM proxy skill's empty-completion handling (2026-09-27, the chat-fixes
 * lane; the Python twin is `tests/test_chat_fixes_proxy_tools.py`):
 *
 *  - an empty completion whose finish reason is MALFORMED_FUNCTION_CALL or
 *    UNEXPECTED_TOOL_CALL is sent once more, on a fresh client, and the
 *    `response.done` this skill yields says so (`finish_retried`);
 *  - the provider's reason rides on `response.done` (`finish_reason`,
 *    `finish_blocked`) so the chat can say it;
 *  - an empty STOP, a blocked prompt and a completion that said something
 *    are not retried.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LLMProxySkill } from '../../../../src/skills/llm/proxy/skill';
import type { Context } from '../../../../src/core/types';

const mockConnect = vi.fn().mockResolvedValue(undefined);
const mockSendResponse = vi.fn().mockResolvedValue(undefined);
const mockSendPayment = vi.fn().mockResolvedValue(undefined);
const mockClose = vi.fn();
const mockOn = vi.fn();
const mockCancel = vi.fn().mockResolvedValue(undefined);

vi.mock('../../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation((config: unknown) => ({
    config,
    connect: mockConnect,
    sendResponse: mockSendResponse,
    sendPayment: mockSendPayment,
    close: mockClose,
    on: mockOn,
    cancel: mockCancel,
  })),
}));

import { UAMPClient } from '../../../../src/uamp/client.js';

function makeContext(): Context {
  return {
    get: vi.fn(() => undefined),
    set: vi.fn(),
    delete: vi.fn(),
    signal: undefined as unknown as AbortSignal,
    auth: { authenticated: false },
    payment: { token: undefined },
    metadata: {},
  } as unknown as Context;
}

/** Fire `eventName` on the listeners registered since `since` (the latest client's). */
function fireLatest(since: number, eventName: string, ...args: unknown[]): void {
  for (const call of mockOn.mock.calls.slice(since)) {
    if (call[0] === eventName) call[1](...args);
  }
}

async function collect(gen: AsyncGenerator<unknown, void, unknown>): Promise<Array<Record<string, any>>> {
  const events: Array<Record<string, any>> = [];
  for await (const event of gen) events.push(event as Record<string, any>);
  return events;
}

/** Scripts one `done` (and optional deltas) per sendResponse call, in order. */
function script(attempts: Array<{ deltas?: string[]; done: Record<string, unknown> }>): void {
  let n = 0;
  // Listeners registered before this mark belong to an earlier client.
  let since = 0;
  mockSendResponse.mockImplementation(async () => {
    const attempt = attempts[n];
    n += 1;
    const from = since;
    queueMicrotask(() => {
      for (const text of attempt.deltas ?? []) fireLatest(from, 'delta', text);
      fireLatest(from, 'done', { id: 'r', status: 'completed', output: [], ...attempt.done });
      since = mockOn.mock.calls.length;
    });
  });
}

const USER = [{ type: 'input.text' as const, event_id: 'e1', text: 'list the folder', role: 'user' }];

describe('an empty completion is retried once', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockConnect.mockResolvedValue(undefined);
  });

  it('a malformed tool call is sent once more, on a fresh client, and the done says so', async () => {
    script([
      { done: { finish_reason: 'MALFORMED_FUNCTION_CALL', usage: { input_tokens: 100, output_tokens: 0, total_tokens: 100 } } },
      { deltas: ['One file.'], done: { output: [{ type: 'text', text: 'One file.' }], finish_reason: 'STOP', usage: { input_tokens: 100, output_tokens: 3, total_tokens: 103 } } },
    ]);
    const context = makeContext();
    const events = await collect(new LLMProxySkill()['processUAMP'](USER, context));
    expect(UAMPClient).toHaveBeenCalledTimes(2);
    expect(mockSendResponse).toHaveBeenCalledTimes(2);
    expect(mockClose).toHaveBeenCalledTimes(2);
    const done = events.find((e) => e.type === 'response.done')!;
    expect(done.response.output).toEqual([{ type: 'text', text: 'One file.' }]);
    expect(done.response.finish_reason).toBe('STOP');
    expect(done.response.finish_retried).toBe(true);
    // The person paid for both attempts.
    expect(done.response.usage).toMatchObject({ input_tokens: 200, output_tokens: 3, total_tokens: 203 });
    expect(context.set).toHaveBeenCalledWith('_llm_usage', expect.objectContaining({ input_tokens: 200, output_tokens: 3 }));
  });

  it('a second empty answer is said, not retried again', async () => {
    script([{ done: { finish_reason: 'UNEXPECTED_TOOL_CALL' } }, { done: { finish_reason: 'UNEXPECTED_TOOL_CALL' } }]);
    const events = await collect(new LLMProxySkill()['processUAMP'](USER, makeContext()));
    expect(mockSendResponse).toHaveBeenCalledTimes(2);
    const done = events.find((e) => e.type === 'response.done')!;
    expect(done.response.output).toEqual([]);
    expect(done.response).toMatchObject({ finish_reason: 'UNEXPECTED_TOOL_CALL', finish_retried: true });
  });

  it('an empty STOP is not retried, and the reason still travels', async () => {
    script([{ done: { finish_reason: 'STOP' } }]);
    const events = await collect(new LLMProxySkill()['processUAMP'](USER, makeContext()));
    expect(mockSendResponse).toHaveBeenCalledTimes(1);
    const done = events.find((e) => e.type === 'response.done')!;
    expect(done.response.finish_reason).toBe('STOP');
    expect(done.response.finish_retried).toBeUndefined();
  });

  it('a blocked prompt is not retried', async () => {
    script([{ done: { finish_reason: 'SAFETY', finish_blocked: true } }]);
    const events = await collect(new LLMProxySkill()['processUAMP'](USER, makeContext()));
    expect(mockSendResponse).toHaveBeenCalledTimes(1);
    expect(events.find((e) => e.type === 'response.done')!.response).toMatchObject({ finish_reason: 'SAFETY', finish_blocked: true });
  });

  it('a completion that said something is not retried, whatever the reason', async () => {
    script([{ deltas: ['Let me look.'], done: { finish_reason: 'MALFORMED_FUNCTION_CALL' } }]);
    const events = await collect(new LLMProxySkill()['processUAMP'](USER, makeContext()));
    expect(mockSendResponse).toHaveBeenCalledTimes(1);
    expect(events.filter((e) => e.type === 'response.delta').map((e) => e.delta.text)).toEqual(['Let me look.']);
  });
});

describe('a platform that goes away before any output is asked once more (2026-09-28)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ toFake: ['setTimeout'] });
    mockConnect.mockResolvedValue(undefined);
  });

  /** Per sendResponse call: fire a drain error, or deltas and a done. */
  function scriptDrains(attempts: Array<{ drain?: number; deltas?: string[]; done?: Record<string, unknown> }>): void {
    let n = 0;
    let since = 0;
    mockSendResponse.mockImplementation(async () => {
      const attempt = attempts[n];
      n += 1;
      const from = since;
      queueMicrotask(() => {
        for (const text of attempt.deltas ?? []) fireLatest(from, 'delta', text);
        if (attempt.drain) fireLatest(from, 'error', Object.assign(new Error(`WebSocket closed unexpectedly (code=${attempt.drain})`), { code: attempt.drain }));
        else fireLatest(from, 'done', { id: 'r', status: 'completed', output: [], ...(attempt.done ?? {}) });
        since = mockOn.mock.calls.length;
      });
    });
  }

  async function run(): Promise<Array<Record<string, any>>> {
    const pending = collect(new LLMProxySkill()['processUAMP'](USER, makeContext()));
    for (let i = 0; i < 5; i++) await vi.advanceTimersByTimeAsync(1000);
    return pending;
  }

  it('a drain (1001) before any output is sent once more, on a fresh client', async () => {
    scriptDrains([{ drain: 1001 }, { deltas: ['YO.'], done: { output: [{ type: 'text', text: 'YO.' }], finish_reason: 'STOP' } }]);
    const events = await run();
    vi.useRealTimers();
    expect(mockSendResponse).toHaveBeenCalledTimes(2);
    expect(events.filter((e) => e.type === 'response.delta').map((e) => e.delta.text)).toEqual(['YO.']);
    expect(events.some((e) => e.type === 'response.error')).toBe(false);
  });

  it('a drain after output is not retried', async () => {
    scriptDrains([{ deltas: ['Y'], drain: 1001 }]);
    await run();
    vi.useRealTimers();
    expect(mockSendResponse).toHaveBeenCalledTimes(1);
  });

  it('another close is not retried', async () => {
    scriptDrains([{ drain: 1011 }]);
    await run();
    vi.useRealTimers();
    expect(mockSendResponse).toHaveBeenCalledTimes(1);
  });
});
