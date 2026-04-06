/**
 * Tests for NLI delegate tool: conversation-based attachment resolution,
 * URL-scan fallback, and response label composition.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// We test the delegate's attachment resolution and response processing logic
// by exercising the NLISkill.delegate method with mocked context and streaming.

// Mock the UAMPClient so we don't make real connections
vi.mock('../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    connect: vi.fn(),
    sendEvents: vi.fn(),
    close: vi.fn(),
  })),
}));

import { NLISkill } from '../../../src/skills/nli/skill.js';
import type { Context, AgenticMessage, StructuredToolResult } from '../../../src/core/types.js';
import type { ContentItem, ImageContent } from '../../../src/uamp/types.js';

const mockSignUrl = vi.fn(async (id: string) => `/api/content/${id}?sig=test&exp=9999999999`);

function makeContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(overrides));
  return {
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    signal: new AbortController().signal,
  } as Context;
}

describe('NLI delegate attachment resolution', () => {
  let skill: NLISkill;

  beforeEach(() => {
    mockSignUrl.mockClear();
    skill = new NLISkill({
      baseUrl: 'https://example.com',
      transport: 'http',
      signUrl: mockSignUrl,
    });
  });

  it('resolves explicit attachment UUIDs from conversation content_items', async () => {
    const uuid = '12345678-1234-1234-1234-123456789abc';
    const messages: AgenticMessage[] = [
      {
        role: 'assistant',
        content: 'Generated image',
        content_items: [
          { type: 'image', image: { url: `/api/content/${uuid}` }, content_id: uuid } as ImageContent,
        ],
      },
    ];

    const ctx = makeContext({
      _agentic_messages: messages,
    });

    // Mock streamMessage to return immediately
    const streamMock = vi.fn().mockImplementation(async function* () {
      yield 'Done';
    });
    (skill as any).streamMessage = streamMock;

    const result = await skill.delegate(
      { agent: 'test-agent', message: 'edit it', attachments: [uuid] },
      ctx,
    );

    // Verify streamMessage was called with the media item
    const callArgs = streamMock.mock.calls[0];
    const sentMessages = callArgs[1];
    expect(sentMessages[0].content_items).toHaveLength(1);
    expect(sentMessages[0].content_items[0].content_id).toBe(uuid);
  });

  it.skip('falls back to URL-scan when UUID not in conversation content_items', () => {
    /* Removed: URL regex scanning replaced by content_id field access */
  });

  it('returns structured content_items without URL leakage in text', async () => {
    const uuid = 'deadbeef-1234-5678-9abc-def012345678';
    const ctx = makeContext({
      _agentic_messages: [],
      _nli_output_items: [
        { type: 'image', image: { url: `/api/content/${uuid}` }, content_id: uuid } as ImageContent,
      ],
    });

    const streamMock = vi.fn().mockImplementation(async function* () {
      yield 'Here is the result';
    });
    (skill as any).streamMessage = streamMock;

    const result = await skill.delegate(
      { agent: 'test-agent', message: 'generate something' },
      ctx,
    );

    expect(typeof result).toBe('object');
    const structured = result as StructuredToolResult;
    expect(structured.content_items).toHaveLength(1);
    expect(structured.text).toBe('Here is the result');
    expect(structured.text).not.toContain('/api/content/');
  });

  it('returns descriptive text instead of "(no response)" when media-only', async () => {
    const uuid = 'aabb1122-3344-5566-7788-99aabbccddee';
    const ctx = makeContext({
      _agentic_messages: [],
      _nli_output_items: [
        { type: 'image', image: { url: `/api/content/${uuid}` }, content_id: uuid } as ImageContent,
      ],
    });

    const streamMock = vi.fn().mockImplementation(async function* () {
      // Empty - no text from agent
    });
    (skill as any).streamMessage = streamMock;

    const result = await skill.delegate(
      { agent: 'test-agent', message: 'generate image' },
      ctx,
    );

    expect(typeof result).toBe('object');
    const structured = result as StructuredToolResult;
    expect(structured.text).toContain('[1 media item returned');
    expect(structured.text).toContain(uuid);
    expect(structured.text).not.toContain('(no response)');
    expect(structured.text).not.toContain('/api/content/');
  });

  it.skip('resolves /api/content/UUID URL attachments by extracting UUID', () => {
    /* Removed: URL regex scanning replaced by content_id field access */
  });

  it.skip('extracts URLs from response text when no output items from done', () => {
    /* Removed: URL regex scanning replaced by content_id field access */
  });

  it('forwards base64 content_items from conversation (MCP case)', async () => {
    const uuid = '99887766-5544-3322-1100-aabbccddeeff';
    const messages: AgenticMessage[] = [
      {
        role: 'assistant',
        content: 'MCP result',
        content_items: [
          { type: 'image', image: 'data:image/png;base64,iVBOR...', content_id: uuid } as unknown as ImageContent,
        ],
      },
    ];

    const ctx = makeContext({
      _agentic_messages: messages,
    });

    const streamMock = vi.fn().mockImplementation(async function* () {
      yield 'Done';
    });
    (skill as any).streamMessage = streamMock;

    await skill.delegate(
      { agent: 'test-agent', message: 'process this', attachments: [uuid] },
      ctx,
    );

    const callArgs = streamMock.mock.calls[0];
    const sentMessages = callArgs[1];
    expect(sentMessages[0].content_items).toHaveLength(1);
    expect(sentMessages[0].content_items[0].content_id).toBe(uuid);
    // signUrl replaces base64 with a signed URL object
    expect(sentMessages[0].content_items[0].image).toEqual({ url: expect.stringContaining(`/api/content/${uuid}`) });
  });
});
