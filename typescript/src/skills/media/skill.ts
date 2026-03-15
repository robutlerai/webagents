/**
 * MediaSkill
 *
 * UAMP hook-based skill for multi-modal content resolution and saving.
 * Runs identically for ALL LLM skills via before_llm_call / after_llm_call hooks.
 *
 * - before_llm_call: Scans messages for content URLs, resolves to base64 or signed URL
 *   based on the target adapter's mediaSupport declaration
 * - after_llm_call: Scans response for generated images (base64), saves via saver,
 *   replaces base64 with content URLs
 *
 * SDK defines abstract interfaces (MediaResolver, MediaSaver).
 * Portal provides concrete implementations (PortalMediaResolver, PortalMediaSaver).
 * Standalone agents can provide their own or omit MediaSkill entirely.
 */

import { Skill } from '../../core/skill.js';
import type { HookData, HookResult, Context } from '../../core/types.js';
import type { MediaSupport } from '../../adapters/types.js';

const CONTENT_URL_RE = /\/api\/content\/([0-9a-f-]{36})/g;

/**
 * Resolves content URLs to base64 or signed URLs.
 * Portal provides PortalMediaResolver; standalone agents can provide their own.
 */
export interface MediaResolver {
  resolve(
    url: string,
    mode: 'base64' | 'url',
    userId?: string,
  ): Promise<ResolvedMedia | null>;
}

/**
 * Saves generated media (e.g., images from LLM output) to storage.
 */
export interface MediaSaver {
  save(
    base64: string,
    mimeType: string,
    meta?: { chatId?: string; agentId?: string; userId?: string },
  ): Promise<string>;
}

export interface ResolvedMedia {
  mimeType: string;
  base64?: string;
  signedUrl?: string;
  expiresAt?: number;
}

export interface CachedMedia extends ResolvedMedia {
  resolvedAt: number;
}

export interface MediaSkillConfig {
  resolver?: MediaResolver;
  saver?: MediaSaver;
  /** Default media support if adapter doesn't declare it */
  defaultMediaSupport?: MediaSupport;
}

export class MediaSkill extends Skill {
  private resolver?: MediaResolver;
  private saver?: MediaSaver;
  private defaultMediaSupport: MediaSupport;
  private cache = new Map<string, CachedMedia>();

  constructor(config: MediaSkillConfig = {}) {
    super({ name: 'media' });
    this.resolver = config.resolver;
    this.saver = config.saver;
    this.defaultMediaSupport = config.defaultMediaSupport ?? {
      image: 'base64',
      audio: 'none',
      video: 'none',
      document: 'none',
    };
  }

  get hooks() {
    return [
      {
        lifecycle: 'before_llm_call' as const,
        priority: 5,
        enabled: true,
        handler: this.beforeLLMCall.bind(this),
      },
      {
        lifecycle: 'after_llm_call' as const,
        priority: 5,
        enabled: true,
        handler: this.afterLLMCall.bind(this),
      },
    ];
  }

  /**
   * Scan messages for content URLs, resolve to base64 via resolver.
   * Sets _resolved_images on context for the LLM skill to use.
   */
  private async beforeLLMCall(
    data: HookData,
    context: Context,
  ): Promise<HookResult | void> {
    if (!this.resolver) return;

    const messages = data.messages || context.get<unknown[]>('_agentic_messages') || [];
    const resolvedMedia = new Map<string, { mimeType: string; base64: string }>();
    const contentIds = new Set<string>();

    for (const m of messages as Array<{ content?: unknown }>) {
      if (!m.content) continue;
      let text: string;
      if (typeof m.content === 'string') {
        text = m.content;
      } else if (Array.isArray(m.content)) {
        text = (m.content as Array<{ type?: string; text?: string }>)
          .filter(p => p.type === 'text' && p.text)
          .map(p => p.text!)
          .join('\n');
      } else {
        continue;
      }
      CONTENT_URL_RE.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = CONTENT_URL_RE.exec(text)) !== null) {
        contentIds.add(match[1]);
      }
    }

    if (contentIds.size === 0) return;

    const userId = context.auth?.user_id;

    await Promise.all([...contentIds].map(async (contentId) => {
      const url = `/api/content/${contentId}`;
      const cacheKey = `${url}:base64`;

      // Check cache
      const cached = this.cache.get(cacheKey);
      if (cached && cached.base64 && (!cached.expiresAt || cached.expiresAt > Date.now() / 1000)) {
        resolvedMedia.set(url, { mimeType: cached.mimeType, base64: cached.base64 });
        return;
      }

      try {
        const resolved = await this.resolver!.resolve(url, 'base64', userId);
        if (resolved && resolved.base64) {
          resolvedMedia.set(url, { mimeType: resolved.mimeType, base64: resolved.base64 });
          this.cache.set(cacheKey, { ...resolved, resolvedAt: Date.now() });
        }
      } catch (err) {
        console.warn(`[MediaSkill] Failed to resolve ${url}:`, (err as Error).message);
      }
    }));

    if (resolvedMedia.size > 0) {
      context.set('_resolved_images', resolvedMedia);
    }
  }

  /**
   * Scan response for generated images (base64), save via saver.
   * Currently a pass-through -- actual image saving happens in the proxy
   * and will be migrated here in later phases.
   */
  private async afterLLMCall(
    data: HookData,
    context: Context,
  ): Promise<HookResult | void> {
    if (!this.saver) return;

    // Check for inline images collected during streaming
    const inlineImages = context.get<Array<{ base64: string; mimeType: string }>>('_inline_images');
    if (!inlineImages || inlineImages.length === 0) return;

    const chatId = context.metadata?.chatId as string | undefined;
    const agentId = context.metadata?.agentId as string | undefined;
    const userId = context.auth?.user_id;
    const savedUrls: string[] = [];

    for (const img of inlineImages) {
      try {
        const url = await this.saver.save(img.base64, img.mimeType, { chatId, agentId, userId });
        savedUrls.push(url);
      } catch (err) {
        console.warn(`[MediaSkill] Failed to save generated image:`, (err as Error).message);
      }
    }

    if (savedUrls.length > 0) {
      context.set('_saved_media_urls', savedUrls);
    }

    context.delete('_inline_images');
  }

  clearCache(): void {
    this.cache.clear();
  }
}
