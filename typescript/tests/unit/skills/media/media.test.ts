/**
 * MediaSkill Unit Tests
 *
 * Tests before_llm_call content resolution and after_llm_call image saving.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MediaSkill } from '../../../../src/skills/media/skill.js';
import type { MediaResolver, MediaSaver } from '../../../../src/skills/media/skill.js';
import type { Context, HookData } from '../../../../src/core/types.js';

function makeContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(overrides));
  return {
    get: vi.fn((key: string) => store.get(key)),
    set: vi.fn((key: string, value: unknown) => store.set(key, value)),
    delete: vi.fn((key: string) => store.delete(key)),
    signal: undefined as unknown as AbortSignal,
    auth: { authenticated: true, user_id: 'user-1' },
    payment: { token: undefined },
    metadata: { chatId: 'chat-1', agentId: 'agent-1' },
  } as unknown as Context;
}

function makeHookData(messages: unknown[] = []): HookData {
  return { messages } as HookData;
}

const mockResolver: MediaResolver = {
  resolve: vi.fn(),
};

const mockSaver: MediaSaver = {
  save: vi.fn(),
};

describe('MediaSkill', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('has name "media"', () => {
    const skill = new MediaSkill();
    expect(skill.name).toBe('media');
  });

  it('registers before_llm_call, after_tool, and after_llm_call hooks', () => {
    const skill = new MediaSkill();
    const hooks = skill.hooks;
    expect(hooks).toHaveLength(3);
    expect(hooks[0].lifecycle).toBe('before_llm_call');
    expect(hooks[1].lifecycle).toBe('after_tool');
    expect(hooks[2].lifecycle).toBe('after_llm_call');
  });

  describe('before_llm_call', () => {
    it('does nothing without a resolver', async () => {
      const skill = new MediaSkill();
      const ctx = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'Check /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx);
      expect(ctx.set).not.toHaveBeenCalled();
    });

    it('resolves content URLs in messages', async () => {
      (mockResolver.resolve as ReturnType<typeof vi.fn>).mockResolvedValue({
        mimeType: 'image/png',
        base64: 'iVBOR...',
      });
      const skill = new MediaSkill({ resolver: mockResolver });
      const ctx = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'Look at /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx);

      expect(mockResolver.resolve).toHaveBeenCalledWith(
        '/api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        'base64',
        'user-1',
      );
      expect(ctx.set).toHaveBeenCalledWith('_resolved_images', expect.any(Map));
    });

    it('skips messages without content URLs', async () => {
      const skill = new MediaSkill({ resolver: mockResolver });
      const ctx = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'No URLs here' },
      ]), ctx);
      expect(mockResolver.resolve).not.toHaveBeenCalled();
    });

    it('handles multimodal content arrays', async () => {
      (mockResolver.resolve as ReturnType<typeof vi.fn>).mockResolvedValue({
        mimeType: 'image/jpeg',
        base64: '/9j/4AA...',
      });
      const skill = new MediaSkill({ resolver: mockResolver });
      const ctx = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: [
          { type: 'text', text: 'See /api/content/11111111-2222-3333-4444-555555555555' },
        ] },
      ]), ctx);
      expect(mockResolver.resolve).toHaveBeenCalledTimes(1);
    });

    it('caches resolved content across calls', async () => {
      (mockResolver.resolve as ReturnType<typeof vi.fn>).mockResolvedValue({
        mimeType: 'image/png',
        base64: 'cached-data',
      });
      const skill = new MediaSkill({ resolver: mockResolver });
      const ctx1 = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'See /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx1);
      const ctx2 = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'See /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee again' },
      ]), ctx2);

      expect(mockResolver.resolve).toHaveBeenCalledTimes(1);
    });

    it('handles resolver errors gracefully', async () => {
      (mockResolver.resolve as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Disk error'));
      const skill = new MediaSkill({ resolver: mockResolver });
      const ctx = makeContext();
      await expect(skill.hooks[0].handler(makeHookData([
        { content: 'See /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx)).resolves.not.toThrow();
    });
  });

  describe('after_llm_call', () => {
    it('does nothing without a saver', async () => {
      const skill = new MediaSkill();
      const ctx = makeContext({
        '_inline_images': [{ base64: 'data', mimeType: 'image/png' }],
      });
      await skill.hooks[2].handler(makeHookData(), ctx);
      expect(ctx.set).not.toHaveBeenCalledWith('_saved_media_urls', expect.anything());
    });

    it('saves inline images and sets _saved_media_urls', async () => {
      (mockSaver.save as ReturnType<typeof vi.fn>).mockResolvedValue({ url: '/api/content/mock-uuid', content_id: 'mock-uuid' });
      const skill = new MediaSkill({ saver: mockSaver });
      const ctx = makeContext({
        '_inline_images': [{ base64: 'imgdata', mimeType: 'image/png' }],
      });
      await skill.hooks[2].handler(makeHookData(), ctx);

      expect(mockSaver.save).toHaveBeenCalledWith('imgdata', 'image/png', {
        chatId: 'chat-1',
        agentId: 'agent-1',
        userId: 'user-1',
      });
      expect(ctx.set).toHaveBeenCalledWith('_saved_media_urls', ['/api/content/mock-uuid']);
    });

    it('cleans up _inline_images after saving', async () => {
      (mockSaver.save as ReturnType<typeof vi.fn>).mockResolvedValue({ url: '/api/content/mock-uuid', content_id: 'mock-uuid' });
      const skill = new MediaSkill({ saver: mockSaver });
      const ctx = makeContext({
        '_inline_images': [{ base64: 'data', mimeType: 'image/png' }],
      });
      await skill.hooks[2].handler(makeHookData(), ctx);
      expect(ctx.delete).toHaveBeenCalledWith('_inline_images');
    });

    it('does nothing without inline images', async () => {
      const skill = new MediaSkill({ saver: mockSaver });
      const ctx = makeContext();
      await skill.hooks[2].handler(makeHookData(), ctx);
      expect(mockSaver.save).not.toHaveBeenCalled();
    });

    it('handles saver errors gracefully', async () => {
      (mockSaver.save as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Storage full'));
      const skill = new MediaSkill({ saver: mockSaver });
      const ctx = makeContext({
        '_inline_images': [{ base64: 'data', mimeType: 'image/png' }],
      });
      await expect(skill.hooks[2].handler(makeHookData(), ctx)).resolves.not.toThrow();
    });
  });

  describe('clearCache', () => {
    it('clears the resolver cache', async () => {
      (mockResolver.resolve as ReturnType<typeof vi.fn>).mockResolvedValue({
        mimeType: 'image/png',
        base64: 'cached',
      });
      const skill = new MediaSkill({ resolver: mockResolver });

      const ctx = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'See /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx);
      skill.clearCache();

      const ctx2 = makeContext();
      await skill.hooks[0].handler(makeHookData([
        { content: 'See /api/content/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' },
      ]), ctx2);

      expect(mockResolver.resolve).toHaveBeenCalledTimes(2);
    });
  });
});
