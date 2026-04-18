/**
 * NLI delegate loop guard
 *
 * Plan §3.2: before issuing the next request to a delegate, the NLI skill
 * must consult the `_countConsecutiveDelegateEmpties` context hook (wired
 * by the portal runtime in `lib/agents/runtime.ts`). If the hook reports
 * 2 or more consecutive `(no response)` markers from this delegate in the
 * sub-chat, the call is short-circuited with a clear error WITHOUT
 * dispatching to the delegate.
 *
 * This is the safety net for the unicorn-loop bug (parent re-delegates
 * forever because every reload showed messageCount=1) — see plan §3.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context, AgenticMessage } from '../../../../src/core/types.js';

const mockFetch = vi.fn();

function makeContext(data: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(data));
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined,
    auth: { authenticated: true, user_id: 'user-A1' },
    payment: { valid: false },
    metadata: {},
    session: { id: 'test', created_at: 0, last_activity: 0, data: {} },
    hasScope: () => false,
    hasScopes: () => false,
  } as unknown as Context;
}

function createSSEStream(content: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const lines = [
    `data: {"choices":[{"delta":{"content":"${content}"}}]}\n\n`,
    'data: [DONE]\n\n',
  ];
  return new ReadableStream({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line));
      controller.close();
    },
  });
}

const SUB_CHAT_ID = 'sub-chat-loop-guard';

describe('NLI delegate loop guard (consecutive empty results)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('short-circuits when 2 consecutive empty markers are present (without dispatching)', async () => {
    const countFn = vi.fn(async () => 2);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [] as AgenticMessage[],
      _countConsecutiveDelegateEmpties: countFn,
    });

    const result = await skill.delegate(
      { agent: '@flaky', message: 'try again' },
      context,
    );

    expect(countFn).toHaveBeenCalledTimes(1);
    expect(countFn).toHaveBeenCalledWith(SUB_CHAT_ID);
    if (typeof result === 'string') throw new Error('expected StructuredToolResult on short-circuit');
    expect(result.text.startsWith('Error:')).toBe(true);
    expect(result.text).toMatch(/empty responses 2 times/);
    expect(result.text).toMatch(/@flaky/);
    expect(result.data).toMatchObject({ subChatId: SUB_CHAT_ID, loopGuard: true });
    expect(mockFetch).not.toHaveBeenCalled(); // critical: no dispatch
  });

  it('short-circuits when 3+ consecutive empty markers are present', async () => {
    const countFn = vi.fn(async () => 5);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: countFn,
    });

    const result = await skill.delegate(
      { agent: '@flaky', message: 'last try' },
      context,
    );
    if (typeof result === 'string') throw new Error('expected StructuredToolResult on short-circuit');
    expect(result.text.startsWith('Error:')).toBe(true);
    expect(result.text).toMatch(/5 times/);
    expect(result.data).toMatchObject({ subChatId: SUB_CHAT_ID, loopGuard: true, consecutiveEmpties: 5 });
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('proceeds when only 1 empty marker is present (under threshold)', async () => {
    const countFn = vi.fn(async () => 1);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: countFn,
    });

    const result = await skill.delegate(
      { agent: '@flaky', message: 'try once more' },
      context,
    );
    expect(countFn).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalled(); // dispatched
    expect(typeof result === 'string' && result.startsWith('Error:')).toBe(false);
  });

  it('proceeds when there are 0 empty markers', async () => {
    const countFn = vi.fn(async () => 0);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: countFn,
    });

    await skill.delegate({ agent: '@target', message: 'ping' }, context);
    expect(mockFetch).toHaveBeenCalled();
  });

  it('proceeds when no _countConsecutiveDelegateEmpties hook is wired (legacy callers)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({ _agentic_messages: [] });

    await skill.delegate({ agent: '@target', message: 'ping' }, context);
    expect(mockFetch).toHaveBeenCalled();
  });

  it('proceeds when the hook throws (non-fatal — must not break delegation)', async () => {
    const countFn = vi.fn(async () => { throw new Error('db transient'); });
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: countFn,
    });

    const result = await skill.delegate(
      { agent: '@target', message: 'ping' },
      context,
    );
    expect(countFn).toHaveBeenCalled();
    expect(mockFetch).toHaveBeenCalled();
    expect(typeof result === 'string' && result.startsWith('Error:')).toBe(false);
  });

  it('skips the loop guard when no sub-chat is resolved (manual chat_id path / null sub-chat)', async () => {
    const countFn = vi.fn(async () => 5);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => null,
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: countFn,
    });

    await skill.delegate({ agent: '@target', message: 'ping' }, context);
    expect(countFn).not.toHaveBeenCalled();
    expect(mockFetch).toHaveBeenCalled();
  });
});
