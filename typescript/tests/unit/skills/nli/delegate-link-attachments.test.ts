/**
 * NLI delegate-attach link invocation
 *
 * After resolveDelegateSubChat returns, NLISkill.delegate must invoke the
 * `_linkContentToSubChat` context hook with the resolved attachment
 * content_ids, the delegate sub-chat id, and the caller's user id.
 *
 * Without this, the delegated agent can READ the attached file (canAccessViaChatTree
 * via ancestor), but findByName/resolvePath inside the sub-chat won't see it
 * by basename — the sub-agent then races text_editor create on the same name.
 * See plans/delegate-attach-link-and-refresh_34e9142b.plan.md (Section A).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context, AgenticMessage } from '../../../../src/core/types.js';
import type { FileContent } from '../../../../src/uamp/types.js';

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
  const lines = content === ''
    ? ['data: [DONE]\n\n']
    : [
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

const SUB_CHAT_ID = 'sub-chat-XYZ';

describe('NLI delegate _linkContentToSubChat invocation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('invokes link hook with resolved attachment content_ids, sub-chat id, and caller user id', async () => {
    const linkFn = vi.fn(async () => undefined);
    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/from-db?sig=abc' },
      filename: 'unicorn.html',
      mime_type: 'text/html',
      content_id: 'att-1',
    } as FileContent;

    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: true }),
    });

    const resolveById = vi.fn(async (id: string) => (id === 'att-1' ? fromDb : null));
    const context = makeContext({
      _agentic_messages: [] as AgenticMessage[],
      _resolveContentById: resolveById,
      _linkContentToSubChat: linkFn,
    });

    await skill.delegate(
      { agent: '@target', message: 'edit it', attachments: ['att-1'] },
      context,
    );

    expect(linkFn).toHaveBeenCalledTimes(1);
    const [ids, chatIdArg, userIdArg] = linkFn.mock.calls[0];
    expect(ids).toEqual(['att-1']);
    expect(chatIdArg).toBe(SUB_CHAT_ID);
    expect(userIdArg).toBe('user-A1');
  });

  it('does NOT invoke link hook when no attachments are passed', async () => {
    const linkFn = vi.fn(async () => undefined);
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: true }),
    });

    const context = makeContext({
      _agentic_messages: [],
      _linkContentToSubChat: linkFn,
    });

    await skill.delegate({ agent: '@target', message: 'no files' }, context);

    expect(linkFn).not.toHaveBeenCalled();
  });

  it('does NOT invoke link hook when no sub-chat is resolved (resolveDelegateSubChat returns null)', async () => {
    const linkFn = vi.fn(async () => undefined);
    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/x' },
      filename: 'x.html',
      mime_type: 'text/html',
      content_id: 'att-A',
    } as FileContent;

    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => null,
    });

    const resolveById = vi.fn(async () => fromDb);
    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
      _linkContentToSubChat: linkFn,
    });

    await skill.delegate(
      { agent: '@target', message: 'edit', attachments: ['att-A'] },
      context,
    );

    expect(linkFn).not.toHaveBeenCalled();
  });

  it('swallows link-hook errors (delegate continues even when linking fails)', async () => {
    const linkFn = vi.fn(async () => { throw new Error('db down'); });
    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/y' },
      filename: 'y.html',
      mime_type: 'text/html',
      content_id: 'att-B',
    } as FileContent;

    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
      resolveDelegateSubChat: async () => ({ chatId: SUB_CHAT_ID, type: 'agent', created: true }),
    });

    const resolveById = vi.fn(async () => fromDb);
    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
      _linkContentToSubChat: linkFn,
    });

    const result = await skill.delegate(
      { agent: '@target', message: 'edit', attachments: ['att-B'] },
      context,
    );

    expect(linkFn).toHaveBeenCalledTimes(1);
    if (typeof result === 'string') throw new Error('expected StructuredToolResult when sub-chat is resolved');
    expect(result.text).toBe('ok');
    expect(result.data).toMatchObject({ subChatId: SUB_CHAT_ID });
  });
});
