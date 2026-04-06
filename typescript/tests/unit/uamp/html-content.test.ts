import { describe, it, expect } from 'vitest';
import type {
  ContentItem,
  HtmlContent,
  DisplayHint,
} from '../../../src/uamp/types.js';
import { getContentItemUrl, isMediaContent, inferDisplayHint } from '../../../src/uamp/content.js';

describe('HtmlContent', () => {
  it('creates HtmlContent with inline html string', () => {
    const item: HtmlContent = {
      type: 'html',
      html: '<div>Hello</div>',
      title: 'Test Widget',
    };
    expect(item.type).toBe('html');
    expect(item.html).toBe('<div>Hello</div>');
    expect(item.title).toBe('Test Widget');
  });

  it('creates HtmlContent with URL object', () => {
    const item: HtmlContent = {
      type: 'html',
      html: { url: '/api/content/abc-123' },
      title: 'Dashboard',
    };
    expect(item.html).toEqual({ url: '/api/content/abc-123' });
  });

  it('supports content_id, description, display_hint fields', () => {
    const item: HtmlContent = {
      type: 'html',
      html: '<p>test</p>',
      content_id: 'html-1',
      description: 'Interactive chart',
      display_hint: 'sandbox',
    };
    expect(item.content_id).toBe('html-1');
    expect(item.description).toBe('Interactive chart');
    expect(item.display_hint).toBe('sandbox');
  });

  it('supports dimensions field', () => {
    const item: HtmlContent = {
      type: 'html',
      html: '<canvas></canvas>',
      dimensions: { width: 800, height: 600 },
    };
    expect(item.dimensions).toEqual({ width: 800, height: 600 });
  });

  it('sandbox defaults to true conceptually', () => {
    const item: HtmlContent = { type: 'html', html: '<div>safe</div>' };
    // sandbox is undefined when not set, but protocol says default is true
    expect(item.sandbox).toBeUndefined();

    const explicit: HtmlContent = { type: 'html', html: '<div>trusted</div>', sandbox: false };
    expect(explicit.sandbox).toBe(false);
  });

  it('is part of ContentItem union', () => {
    const item: ContentItem = {
      type: 'html',
      html: '<div>test</div>',
      content_id: 'h-1',
    };
    expect(item.type).toBe('html');
  });

  it('survives JSON round-trip', () => {
    const item: HtmlContent = {
      type: 'html',
      html: { url: '/api/content/h-1' },
      title: 'Report',
      content_id: 'h-1',
      description: 'Q4 Revenue Report',
      display_hint: 'sandbox',
      dimensions: { width: 1024, height: 768 },
      sandbox: true,
    };
    const roundTripped = JSON.parse(JSON.stringify(item)) as HtmlContent;
    expect(roundTripped.type).toBe('html');
    expect(roundTripped.html).toEqual({ url: '/api/content/h-1' });
    expect(roundTripped.title).toBe('Report');
    expect(roundTripped.content_id).toBe('h-1');
    expect(roundTripped.description).toBe('Q4 Revenue Report');
    expect(roundTripped.display_hint).toBe('sandbox');
    expect(roundTripped.dimensions).toEqual({ width: 1024, height: 768 });
  });

  describe('content helpers with HtmlContent', () => {
    it('getContentItemUrl returns null for inline html string', () => {
      const item: ContentItem = { type: 'html', html: '<div>test</div>' };
      expect(getContentItemUrl(item)).toBeNull();
    });

    it('getContentItemUrl returns URL for html URL object', () => {
      const item: ContentItem = { type: 'html', html: { url: '/api/content/h-1' } };
      expect(getContentItemUrl(item)).toBe('/api/content/h-1');
    });

    it('isMediaContent returns true for html', () => {
      const item: ContentItem = { type: 'html', html: '<div>test</div>' };
      expect(isMediaContent(item)).toBe(true);
    });

    it('inferDisplayHint returns sandbox for html', () => {
      expect(inferDisplayHint('html')).toBe('sandbox');
    });
  });

  describe('all content types are distinguishable', () => {
    it('8 content types including html are unique', () => {
      const items: ContentItem[] = [
        { type: 'text', text: 'hi' },
        { type: 'audio', audio: 'x' },
        { type: 'image', image: 'x' },
        { type: 'video', video: 'x' },
        { type: 'file', file: 'x', filename: 'f', mime_type: 'm' },
        { type: 'html', html: '<div>x</div>' },
        { type: 'tool_call', tool_call: { id: '1', name: 'n', arguments: '{}' } },
        { type: 'tool_result', tool_result: { call_id: '1', result: 'r' } },
      ];
      const types = items.map(i => i.type);
      expect(new Set(types).size).toBe(8);
    });
  });

  describe('description and display_hint on all media types', () => {
    it('ImageContent has description and display_hint', () => {
      const item: ContentItem = {
        type: 'image',
        image: { url: '/test' },
        description: 'A sunset photo',
        display_hint: 'inline' as DisplayHint,
      };
      expect((item as any).description).toBe('A sunset photo');
      expect((item as any).display_hint).toBe('inline');
    });

    it('AudioContent has description and display_hint', () => {
      const item: ContentItem = {
        type: 'audio',
        audio: { url: '/test' },
        description: 'Podcast episode',
        display_hint: 'inline' as DisplayHint,
      };
      expect((item as any).description).toBe('Podcast episode');
    });

    it('VideoContent has description and display_hint', () => {
      const item: ContentItem = {
        type: 'video',
        video: { url: '/test' },
        description: 'Tutorial video',
        display_hint: 'inline' as DisplayHint,
      };
      expect((item as any).description).toBe('Tutorial video');
    });

    it('FileContent has description and display_hint', () => {
      const item: ContentItem = {
        type: 'file',
        file: { url: '/test' },
        filename: 'report.pdf',
        mime_type: 'application/pdf',
        description: 'Q4 report',
        display_hint: 'attachment' as DisplayHint,
      };
      expect((item as any).description).toBe('Q4 report');
      expect((item as any).display_hint).toBe('attachment');
    });
  });
});
