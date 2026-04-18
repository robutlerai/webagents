/**
 * NLI Delegate Attachment DB-Fallback Tests
 *
 * Covers the resolver promotion ladder added in the Subchat access repair plan:
 *   1. convContentMap hit  -> use it
 *   2. convContentMap miss -> fall back to _resolveContentById hook
 *   3. both miss           -> return a hard error (no silent drop)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context, AgenticMessage } from '../../../../src/core/types.js';
import type { ContentItem, ImageContent, FileContent } from '../../../../src/uamp/types.js';

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

describe('NLI delegate attachment DB-fallback', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uses convContentMap hit and does NOT call _resolveContentById', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });

    const inMap: ImageContent = { type: 'image', image: { url: 'https://example.com/inmap.png' }, content_id: 'in-map' };
    const messages: AgenticMessage[] = [
      { role: 'tool', name: 'gen', content: 'gen', content_items: [inMap] },
    ];

    const resolveById = vi.fn(async (_id: string) => null);
    const context = makeContext({
      _agentic_messages: messages,
      _resolveContentById: resolveById,
    });

    const result = await skill.delegate(
      { agent: '@x', message: 'm', attachments: ['in-map'] },
      context,
    );

    expect(result).toBe('ok');
    expect(resolveById).not.toHaveBeenCalled();
    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ content_id: 'in-map' })]),
    );
  });

  it('falls back to _resolveContentById when ref is not in convMap', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });

    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/from-db?sig=abc' },
      filename: 'unicorn.html',
      mime_type: 'text/html',
      content_id: 'from-db',
    } as FileContent;

    const resolveById = vi.fn(async (id: string, callerUserId?: string) => {
      expect(callerUserId).toBe('user-A1');
      return id === 'from-db' ? (fromDb as ContentItem) : null;
    });

    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
    });

    const result = await skill.delegate(
      { agent: '@x', message: 'm', attachments: ['from-db'] },
      context,
    );

    expect(result).toBe('ok');
    expect(resolveById).toHaveBeenCalledTimes(1);
    expect(resolveById).toHaveBeenCalledWith('from-db', 'user-A1');

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          content_id: 'from-db',
          // Pre-signed URL from the hook must NOT be re-signed; signUrl is
          // not configured here so this also asserts no crash on missing
          // signFn for DB-resolved items.
          file: expect.objectContaining({ url: 'https://signed.example/from-db?sig=abc' }),
        }),
      ]),
    );
  });

  it('returns hard-error when neither convMap nor _resolveContentById resolves', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('should-not-call') });

    const resolveById = vi.fn(async () => null);
    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
    });

    const result = await skill.delegate(
      { agent: '@x', message: 'm', attachments: ['ghost-id', 'phantom-id'] },
      context,
    );

    expect(typeof result).toBe('string');
    expect(result as string).toMatch(/Error:.*2 attachment id\(s\) could not be resolved/);
    expect(result as string).toContain('ghost-id');
    expect(result as string).toContain('phantom-id');
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('treats _resolveContentById exceptions as a miss (continues to hard-error)', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('should-not-call') });

    const resolveById = vi.fn(async () => { throw new Error('db down'); });
    const context = makeContext({
      _agentic_messages: [],
      _resolveContentById: resolveById,
    });

    const result = await skill.delegate(
      { agent: '@x', message: 'm', attachments: ['unstable-id'] },
      context,
    );

    expect(typeof result).toBe('string');
    expect(result as string).toMatch(/Error:.*could not be resolved/);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('mixes convMap hit + DB hit + miss into one hard-error listing only the misses', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('should-not-call') });

    const inMap: ImageContent = { type: 'image', image: { url: 'u' }, content_id: 'in-map' };
    const fromDb: FileContent = {
      type: 'file',
      file: { url: 'https://signed.example/from-db' },
      filename: 'a.txt',
      mime_type: 'text/plain',
      content_id: 'from-db',
    } as FileContent;

    const messages: AgenticMessage[] = [
      { role: 'tool', name: 'gen', content: 'gen', content_items: [inMap] },
    ];

    const resolveById = vi.fn(async (id: string) =>
      id === 'from-db' ? (fromDb as ContentItem) : null,
    );

    const context = makeContext({
      _agentic_messages: messages,
      _resolveContentById: resolveById,
    });

    const result = await skill.delegate(
      { agent: '@x', message: 'm', attachments: ['in-map', 'from-db', 'missing-1', 'missing-2'] },
      context,
    );

    expect(typeof result).toBe('string');
    expect(result as string).toMatch(/Error:.*2 attachment id\(s\) could not be resolved/);
    expect(result as string).toContain('missing-1');
    expect(result as string).toContain('missing-2');
    expect(result as string).not.toContain('in-map');
    expect(result as string).not.toContain('from-db');
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
