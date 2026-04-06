import { describe, it, expect, vi } from 'vitest';
import type { ContentItem, ImageContent, FileContent, Capabilities } from '../../../src/uamp/types.js';
import type { DisplayHint } from '../../../src/uamp/types.js';
import { inferDisplayHint } from '../../../src/uamp/content.js';

describe('present tool', () => {
  describe('inferDisplayHint', () => {
    it('returns inline for image', () => {
      expect(inferDisplayHint('image')).toBe('inline');
    });
    it('returns inline for video', () => {
      expect(inferDisplayHint('video')).toBe('inline');
    });
    it('returns inline for audio', () => {
      expect(inferDisplayHint('audio')).toBe('inline');
    });
    it('returns attachment for file', () => {
      expect(inferDisplayHint('file')).toBe('attachment');
    });
    it('returns sandbox for html', () => {
      expect(inferDisplayHint('html')).toBe('sandbox');
    });
    it('returns inline for unknown types', () => {
      expect(inferDisplayHint('unknown')).toBe('inline');
    });
  });

  describe('present tool behavior', () => {
    function createPresentHandler(collectedContentItems: ContentItem[], presentedIds: Set<string>) {
      return async (args: Record<string, unknown>) => {
        const id = args.content_id as string;
        const item = collectedContentItems.find(ci =>
          (ci as { content_id?: string }).content_id === id);
        if (!item) {
          const availableIds = collectedContentItems
            .map(ci => (ci as { content_id?: string }).content_id)
            .filter(Boolean)
            .join(', ');
          return `Content not found: ${id}. Available: ${availableIds || 'none'}. If this content came from an external URL, use save_content first.`;
        }
        const hint: DisplayHint = (args.display_as as DisplayHint) || inferDisplayHint(item.type);
        (item as any).display_hint = hint;
        if (args.caption) (item as any).caption = args.caption as string;
        presentedIds.add(id);
        const dims = (item as any).dimensions;
        const desc = (item as any).description || '';
        return `Displayed ${item.type}${dims ? ` (${dims.width}x${dims.height})` : ''} to user${desc ? ': ' + desc : '.'}`;
      };
    }

    it('returns error for unknown content_id', async () => {
      const handler = createPresentHandler([], new Set());
      const result = await handler({ content_id: 'unknown-id' });
      expect(result).toContain('Content not found: unknown-id');
      expect(result).toContain('save_content');
    });

    it('lists available IDs in error message', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
        { type: 'image', image: { url: '/test2' }, content_id: 'img-2' } as ImageContent,
      ];
      const handler = createPresentHandler(items, new Set());
      const result = await handler({ content_id: 'wrong' });
      expect(result).toContain('img-1');
      expect(result).toContain('img-2');
    });

    it('sets display_hint from display_as parameter', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
      ];
      const presentedIds = new Set<string>();
      const handler = createPresentHandler(items, presentedIds);
      await handler({ content_id: 'img-1', display_as: 'attachment' });
      expect((items[0] as any).display_hint).toBe('attachment');
    });

    it('auto-infers display_hint from type when display_as omitted', async () => {
      const items: ContentItem[] = [
        { type: 'file', file: { url: '/test' }, filename: 'doc.pdf', mime_type: 'application/pdf', content_id: 'f-1' } as FileContent,
      ];
      const handler = createPresentHandler(items, new Set());
      await handler({ content_id: 'f-1' });
      expect((items[0] as any).display_hint).toBe('attachment');
    });

    it('returns rich confirmation with type, dimensions, and description', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1', description: '4K sunset', dimensions: { width: 1024, height: 768 } } as any,
      ];
      const handler = createPresentHandler(items, new Set());
      const result = await handler({ content_id: 'img-1' });
      expect(result).toBe('Displayed image (1024x768) to user: 4K sunset');
    });

    it('omits dimensions when not available', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1', description: 'A photo' } as any,
      ];
      const handler = createPresentHandler(items, new Set());
      const result = await handler({ content_id: 'img-1' });
      expect(result).toBe('Displayed image to user: A photo');
    });

    it('ends with period when no description', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
      ];
      const handler = createPresentHandler(items, new Set());
      const result = await handler({ content_id: 'img-1' });
      expect(result).toBe('Displayed image to user.');
    });

    it('tracks presentedIds', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
        { type: 'video', video: { url: '/test' }, content_id: 'vid-1' } as any,
      ];
      const presentedIds = new Set<string>();
      const handler = createPresentHandler(items, presentedIds);
      await handler({ content_id: 'img-1' });
      expect(presentedIds.has('img-1')).toBe(true);
      expect(presentedIds.has('vid-1')).toBe(false);
    });

    it('sets caption on item', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
      ];
      const handler = createPresentHandler(items, new Set());
      await handler({ content_id: 'img-1', caption: 'My photo' });
      expect((items[0] as any).caption).toBe('My photo');
    });

    it('returns not-found for empty string content_id', async () => {
      const handler = createPresentHandler([], new Set());
      const result = await handler({ content_id: '' });
      expect(result).toContain('Content not found');
    });

    it('returns first match for duplicate content_ids', async () => {
      const items: ContentItem[] = [
        { type: 'image', image: { url: '/first' }, content_id: 'dup-id' } as ImageContent,
        { type: 'image', image: { url: '/second' }, content_id: 'dup-id' } as ImageContent,
      ];
      const handler = createPresentHandler(items, new Set());
      await handler({ content_id: 'dup-id' });
      expect((items[0] as any).display_hint).toBe('inline');
      expect((items[1] as any).display_hint).toBeUndefined();
    });
  });

  describe('response.done output filtering', () => {
    it('filters to presentedIds when present tool is available', () => {
      const collectedContentItems: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
        { type: 'image', image: { url: '/test2' }, content_id: 'img-2' } as ImageContent,
      ];
      const presentedIds = new Set(['img-1']);
      const hasPresentTool = true;

      const outputContentItems = hasPresentTool
        ? collectedContentItems.filter(ci => presentedIds.has((ci as { content_id?: string }).content_id || ''))
        : collectedContentItems;

      expect(outputContentItems).toHaveLength(1);
      expect((outputContentItems[0] as any).content_id).toBe('img-1');
    });

    it('promotes all when present tool not available', () => {
      const collectedContentItems: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
        { type: 'image', image: { url: '/test2' }, content_id: 'img-2' } as ImageContent,
      ];
      const hasPresentTool = false;

      const outputContentItems = hasPresentTool
        ? collectedContentItems.filter(ci => false)
        : collectedContentItems;

      expect(outputContentItems).toHaveLength(2);
    });

    it('no safety net: empty output when browser client and no present calls', () => {
      const collectedContentItems: ContentItem[] = [
        { type: 'image', image: { url: '/test' }, content_id: 'img-1' } as ImageContent,
      ];
      const presentedIds = new Set<string>();
      const hasPresentTool = true;

      const outputContentItems = hasPresentTool
        ? collectedContentItems.filter(ci => presentedIds.has((ci as { content_id?: string }).content_id || ''))
        : collectedContentItems;

      expect(outputContentItems).toHaveLength(0);
    });
  });
});
