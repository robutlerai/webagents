/**
 * NLI Delegate Content Item Tests
 *
 * Tests the delegate tool's content_id-based attachment resolution,
 * _content_registry lookup, URL-scan fallback, and user media forwarding.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';
import type { Context, AgenticMessage } from '../../../../src/core/types.js';
import type { ContentItem, Message, ImageContent, AudioContent } from '../../../../src/uamp/types.js';

const mockFetch = vi.fn();

function makeContext(data: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(data));
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined,
    auth: { authenticated: false },
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

describe('NLI delegate with content_id attachments', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('resolves attachments from _content_registry', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('got the image') });

    const imageItem: ImageContent = { type: 'image', image: { url: 'https://example.com/img.png' }, content_id: 'uuid-1' };
    const registry = new Map<string, ContentItem>([['uuid-1', imageItem]]);

    const context = makeContext({ _content_registry: registry });
    const result = await skill.delegate(
      { agent: '@analyst', message: 'Analyze this', attachments: ['uuid-1'] },
      context,
    );

    expect(result).toBe('got the image');
    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });

  it('resolves multiple attachments independently', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('processed') });

    const img: ImageContent = { type: 'image', image: 'base64img', content_id: 'img-uuid' };
    const aud: AudioContent = { type: 'audio', audio: 'base64aud', content_id: 'aud-uuid' };
    const registry = new Map<string, ContentItem>([['img-uuid', img], ['aud-uuid', aud]]);

    const context = makeContext({ _content_registry: registry });
    await skill.delegate(
      { agent: '@multimodal', message: 'Check these', attachments: ['img-uuid', 'aud-uuid'] },
      context,
    );

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toHaveLength(2);
    expect(fetchBody.messages[0].content_items[0].type).toBe('image');
    expect(fetchBody.messages[0].content_items[1].type).toBe('audio');
  });

  it('silently skips unknown attachment UUIDs', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });

    const registry = new Map<string, ContentItem>();
    const context = makeContext({ _content_registry: registry });
    await skill.delegate(
      { agent: '@test', message: 'Hello', attachments: ['nonexistent-uuid'] },
      context,
    );

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toBeUndefined();
  });

  it('falls back to URL-scan when attachments is empty', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('found') });

    const img: ImageContent = { type: 'image', image: { url: '/api/content/a1b2c3d4-e5f6-7890-abcd-ef1234567890' }, content_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890' };
    const registry = new Map<string, ContentItem>([['a1b2c3d4-e5f6-7890-abcd-ef1234567890', img]]);

    const context = makeContext({ _content_registry: registry });
    await skill.delegate(
      { agent: '@agent', message: 'Analyze /api/content/a1b2c3d4-e5f6-7890-abcd-ef1234567890' },
      context,
    );

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });

  it('includes user original media when no attachments specified', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('got it') });

    const imageItem: ImageContent = { type: 'image', image: { url: 'https://example.com/img.png' }, content_id: 'user-img-uuid' };
    const agenticMessages: AgenticMessage[] = [{
      role: 'user',
      content: 'Check this',
      content_items: [{ type: 'text', text: 'Check this' }, imageItem],
    }];

    const context = makeContext({ _agentic_messages: agenticMessages });
    await skill.delegate({ agent: '@test', message: 'Check this' }, context);

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image' })]),
    );
  });

  it('does not include text/tool_call/tool_result from user messages', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('ok') });

    const agenticMessages: AgenticMessage[] = [{
      role: 'user',
      content: 'Hello',
      content_items: [
        { type: 'text', text: 'Hello' },
        { type: 'tool_call', tool_call: { id: 'tc1', name: 'fn', arguments: '{}' } },
        { type: 'tool_result', tool_result: { call_id: 'tc1', result: 'ok' } },
      ],
    }];

    const context = makeContext({ _agentic_messages: agenticMessages });
    await skill.delegate({ agent: '@test', message: 'Hello' }, context);

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toBeUndefined();
  });

  it('works when _content_registry and _agentic_messages are absent', async () => {
    const skill = new NLISkill({
      baseUrl: 'https://portal.example.com',
      transport: 'http',
      timeout: 5000,
    });

    mockFetch.mockResolvedValue({ ok: true, body: createSSEStream('hi') });

    const context = makeContext();
    const result = await skill.delegate({ agent: '@agent', message: 'hi' }, context);

    expect(result).toBe('hi');
    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.messages[0].content_items).toBeUndefined();
  });
});
