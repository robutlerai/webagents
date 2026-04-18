/**
 * NLI delegate subChatId threading
 *
 * Plan §4 step 1: when a sub-chat is resolved (via `resolveDelegateSubChat`),
 * the StructuredToolResult returned by the NLI `delegate(...)` tool must
 * carry `data.subChatId` so:
 *   1. The `core/agent.ts` tool-call loop forwards it onto the live
 *      `response.delta { tool_result }` envelope (verified by mocking
 *      `_executeInternalToolCall` and checking the envelope shape).
 *   2. The persisted `role='tool'` row metadata carries `tool_data.subChatId`
 *      so the `<DelegateSubChatPreview />` survives page reloads (covered
 *      separately by `tests/unit/messaging/record-tool-turn.test.ts`).
 *
 * This test focuses on (1): the NLI skill's return value shape across the
 * three delegate paths — outputItems present, plain-text only, loop-guard
 * short-circuit — and the legacy null-sub-chat fallback (no data field).
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

const SUB_CHAT_ID = 'sub-chat-thread-id-XYZ';

describe('NLI delegate threads subChatId into StructuredToolResult.data', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('hello from delegate') });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('attaches { subChatId } on the plain-text return path when sub-chat is resolved', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({ _agentic_messages: [] as AgenticMessage[] });

    const result = await skill.delegate({ agent: '@target', message: 'ping' }, context);

    expect(typeof result).toBe('object');
    if (typeof result === 'string') throw new Error('expected StructuredToolResult');
    expect(result.text).toBe('hello from delegate');
    expect(result.data).toEqual({ subChatId: SUB_CHAT_ID });
  });

  it('attaches { subChatId } on the outputItems return path (UAMP path with content_items)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const items = [{ type: 'image' as const, image: { url: 'https://example.com/x.png' }, content_id: 'img-1' }];
    const context = makeContext({
      _agentic_messages: [],
      _nli_output_items: items,
      _nli_text_only: 'here is your image',
    });

    const result = await skill.delegate({ agent: '@target', message: 'gen image' }, context);
    if (typeof result === 'string') throw new Error('expected StructuredToolResult');
    expect(result.content_items).toEqual(items);
    expect(result.text).toContain('here is your image');
    expect(result.data).toEqual({ subChatId: SUB_CHAT_ID });
  });

  it('attaches loop-guard metadata + subChatId on the short-circuit return path', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: false }),
    });
    const context = makeContext({
      _agentic_messages: [],
      _countConsecutiveDelegateEmpties: async () => 2,
    });

    const result = await skill.delegate({ agent: '@flaky', message: 'try again' }, context);
    if (typeof result === 'string') throw new Error('expected StructuredToolResult');
    expect(result.text).toMatch(/^Error:/);
    expect(result.data).toEqual({ subChatId: SUB_CHAT_ID, loopGuard: true, consecutiveEmpties: 2 });
  });

  it('returns a plain string (no data field) when no sub-chat is resolved (legacy)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => null,
    });
    const context = makeContext({ _agentic_messages: [] });

    const result = await skill.delegate({ agent: '@target', message: 'ping' }, context);
    expect(typeof result).toBe('string');
    expect(result).toBe('hello from delegate');
  });
});
