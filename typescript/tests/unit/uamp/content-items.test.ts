/**
 * UAMP Content Item Discriminated Union Tests
 *
 * Validates the new ContentItem type hierarchy: text, audio, image, video,
 * file, tool_call, tool_result — with URL-or-base64 payloads and
 * ToolResult.content_items preservation through serialization.
 */

import { describe, it, expect } from 'vitest';
import type {
  ContentItem,
  TextContent,
  AudioContent,
  ImageContent,
  VideoContent,
  FileContent,
  ToolCallContent,
  ToolResultContent,
  ToolResult,
} from '../../../src/uamp/types.js';
import { getContentItemUrl, isMediaContent, ensureContentId } from '../../../src/uamp/content.js';

describe('ContentItem discriminated union', () => {
  it('creates TextContent', () => {
    const item: ContentItem = { type: 'text', text: 'hello' };
    expect(item.type).toBe('text');
    expect((item as TextContent).text).toBe('hello');
  });

  it('creates AudioContent with base64 string', () => {
    const item: ContentItem = { type: 'audio', audio: 'base64data==', format: 'mp3' };
    expect(item.type).toBe('audio');
    expect((item as AudioContent).audio).toBe('base64data==');
    expect((item as AudioContent).format).toBe('mp3');
  });

  it('creates AudioContent with URL object', () => {
    const item: ContentItem = {
      type: 'audio',
      audio: { url: 'https://cdn.example.com/clip.mp3' },
      format: 'mp3',
      duration_ms: 5000,
    };
    expect((item as AudioContent).audio).toEqual({ url: 'https://cdn.example.com/clip.mp3' });
    expect((item as AudioContent).duration_ms).toBe(5000);
  });

  it('creates ImageContent with base64 string', () => {
    const item: ContentItem = { type: 'image', image: 'iVBORw0KGgo=' };
    expect(item.type).toBe('image');
    expect(typeof (item as ImageContent).image).toBe('string');
  });

  it('creates ImageContent with URL object', () => {
    const item: ContentItem = {
      type: 'image',
      image: { url: '/api/content/abc-123' },
      detail: 'high',
      alt_text: 'A cat',
    };
    expect((item as ImageContent).image).toEqual({ url: '/api/content/abc-123' });
    expect((item as ImageContent).detail).toBe('high');
    expect((item as ImageContent).alt_text).toBe('A cat');
  });

  it('creates VideoContent with URL object', () => {
    const item: ContentItem = {
      type: 'video',
      video: { url: 'https://cdn.example.com/vid.mp4' },
      format: 'mp4',
      duration_ms: 30000,
    };
    expect(item.type).toBe('video');
    expect((item as VideoContent).video).toEqual({ url: 'https://cdn.example.com/vid.mp4' });
  });

  it('creates VideoContent with base64 string', () => {
    const item: ContentItem = { type: 'video', video: 'AAAA' };
    expect(typeof (item as VideoContent).video).toBe('string');
  });

  it('creates FileContent with URL object', () => {
    const item: ContentItem = {
      type: 'file',
      file: { url: 'https://storage.example.com/doc.pdf' },
      filename: 'doc.pdf',
      mime_type: 'application/pdf',
      size_bytes: 102400,
    };
    expect(item.type).toBe('file');
    expect((item as FileContent).filename).toBe('doc.pdf');
    expect((item as FileContent).size_bytes).toBe(102400);
  });

  it('creates FileContent with base64 string', () => {
    const item: ContentItem = {
      type: 'file',
      file: 'base64...',
      filename: 'report.csv',
      mime_type: 'text/csv',
    };
    expect(typeof (item as FileContent).file).toBe('string');
  });

  it('creates ToolCallContent', () => {
    const item: ContentItem = {
      type: 'tool_call',
      tool_call: { id: 'tc_1', name: 'generate_image', arguments: '{"prompt":"cat"}' },
    };
    expect(item.type).toBe('tool_call');
    expect((item as ToolCallContent).tool_call.name).toBe('generate_image');
  });

  it('creates ToolResultContent', () => {
    const item: ContentItem = {
      type: 'tool_result',
      tool_result: { call_id: 'tc_1', result: '{"url":"/api/content/xyz"}' },
    };
    expect(item.type).toBe('tool_result');
    expect((item as ToolResultContent).tool_result.call_id).toBe('tc_1');
  });

  it('ToolResult.content_items survives JSON round-trip', () => {
    const innerItems: ContentItem[] = [
      { type: 'image', image: { url: '/api/content/img-1' } },
      { type: 'text', text: 'caption' },
    ];
    const tr: ToolResult = {
      call_id: 'tc_2',
      result: 'ok',
      content_items: innerItems,
    };
    const serialized = JSON.stringify(tr);
    const deserialized: ToolResult = JSON.parse(serialized);

    expect(deserialized.content_items).toHaveLength(2);
    expect(deserialized.content_items![0].type).toBe('image');
    expect((deserialized.content_items![0] as ImageContent).image).toEqual({ url: '/api/content/img-1' });
    expect(deserialized.content_items![1].type).toBe('text');
  });

  it('ToolResult with is_error flag', () => {
    const tr: ToolResult = { call_id: 'tc_err', result: 'timeout', is_error: true };
    expect(tr.is_error).toBe(true);
  });

  it('all 7 content types are distinguishable by discriminant', () => {
    const items: ContentItem[] = [
      { type: 'text', text: 'hi' },
      { type: 'audio', audio: 'x' },
      { type: 'image', image: 'x' },
      { type: 'video', video: 'x' },
      { type: 'file', file: 'x', filename: 'f', mime_type: 'm' },
      { type: 'tool_call', tool_call: { id: '1', name: 'n', arguments: '{}' } },
      { type: 'tool_result', tool_result: { call_id: '1', result: 'r' } },
    ];
    const types = items.map(i => i.type);
    expect(new Set(types).size).toBe(7);
  });

  it('content_id is preserved through JSON serialization', () => {
    const item: ImageContent = {
      type: 'image',
      image: { url: '/api/content/abc-123' },
      content_id: 'abc-123',
    };
    const serialized = JSON.stringify(item);
    const deserialized: ImageContent = JSON.parse(serialized);
    expect(deserialized.content_id).toBe('abc-123');
  });
});

describe('getContentItemUrl', () => {
  it('extracts URL from ImageContent string', () => {
    expect(getContentItemUrl({ type: 'image', image: 'https://example.com/img.png' })).toBe('https://example.com/img.png');
  });

  it('extracts URL from ImageContent object', () => {
    expect(getContentItemUrl({ type: 'image', image: { url: '/api/content/abc-123' } })).toBe('/api/content/abc-123');
  });

  it('extracts URL from AudioContent', () => {
    expect(getContentItemUrl({ type: 'audio', audio: { url: '/api/content/aud-1' } })).toBe('/api/content/aud-1');
  });

  it('extracts URL from VideoContent', () => {
    expect(getContentItemUrl({ type: 'video', video: { url: '/api/content/vid-1' } })).toBe('/api/content/vid-1');
  });

  it('extracts URL from FileContent', () => {
    expect(getContentItemUrl({ type: 'file', file: { url: '/api/content/f-1' }, filename: 'f', mime_type: 'm' })).toBe('/api/content/f-1');
  });

  it('returns null for TextContent', () => {
    expect(getContentItemUrl({ type: 'text', text: 'hello' })).toBeNull();
  });

  it('returns null for ToolCallContent', () => {
    expect(getContentItemUrl({ type: 'tool_call', tool_call: { id: '1', name: 'n', arguments: '{}' } })).toBeNull();
  });
});

describe('isMediaContent', () => {
  it('returns true for image, audio, video, file', () => {
    expect(isMediaContent({ type: 'image', image: 'x' })).toBe(true);
    expect(isMediaContent({ type: 'audio', audio: 'x' })).toBe(true);
    expect(isMediaContent({ type: 'video', video: 'x' })).toBe(true);
    expect(isMediaContent({ type: 'file', file: 'x', filename: 'f', mime_type: 'm' })).toBe(true);
  });

  it('returns false for text, tool_call, tool_result', () => {
    expect(isMediaContent({ type: 'text', text: 'hello' })).toBe(false);
    expect(isMediaContent({ type: 'tool_call', tool_call: { id: '1', name: 'n', arguments: '{}' } })).toBe(false);
    expect(isMediaContent({ type: 'tool_result', tool_result: { call_id: '1', result: 'r' } })).toBe(false);
  });
});

describe('ensureContentId', () => {
  it('extracts UUID from /api/content/ URL', () => {
    const item: ImageContent = { type: 'image', image: { url: '/api/content/a1b2c3d4-e5f6-7890-abcd-ef1234567890' } };
    const result = ensureContentId(item) as ImageContent;
    expect(result.content_id).toBe('a1b2c3d4-e5f6-7890-abcd-ef1234567890');
  });

  it('generates a new UUID for base64 content', () => {
    const item: ImageContent = { type: 'image', image: 'iVBORw0KGgo=' };
    const result = ensureContentId(item) as ImageContent;
    expect(result.content_id).toBeDefined();
    expect(result.content_id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('is idempotent -- returns existing content_id if present', () => {
    const item: ImageContent = { type: 'image', image: 'x', content_id: 'existing-uuid' };
    const result = ensureContentId(item) as ImageContent;
    expect(result.content_id).toBe('existing-uuid');
    expect(result).toBe(item); // same reference
  });

  it('returns text items unchanged (no content_id added)', () => {
    const item: TextContent = { type: 'text', text: 'hello' };
    const result = ensureContentId(item);
    expect(result).toBe(item);
    expect('content_id' in result).toBe(false);
  });

  it('returns tool_call items unchanged', () => {
    const item: ToolCallContent = { type: 'tool_call', tool_call: { id: '1', name: 'n', arguments: '{}' } };
    const result = ensureContentId(item);
    expect(result).toBe(item);
  });
});
