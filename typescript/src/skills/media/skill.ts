/**
 * StoreMediaSkill
 *
 * UAMP hook-based skill for multi-modal content resolution and saving.
 * Acts as the portal content boundary: tools produce raw UAMP content_items
 * (base64 or temp CDN URLs), and this skill intercepts via after_tool to
 * upload them to /content, returning structured MediaSaverResult {url, content_id}.
 *
 * Hooks:
 * - before_llm_call: Scans messages for content URLs, resolves to base64 or signed URL
 *   based on the target adapter's mediaSupport declaration. Exposes _media_saver on
 *   context to enable the save_content built-in tool.
 * - after_tool: Detects non-/content media in tool results, uploads via saver,
 *   replaces content_items with /api/content/UUID URLs and structured content_id.
 * - after_llm_call: Scans response for generated images (base64), saves via saver,
 *   replaces base64 with content URLs
 *
 * SDK defines abstract interfaces (MediaResolver, MediaSaver).
 * Portal provides concrete implementations (PortalMediaResolver, PortalMediaSaver).
 * Standalone agents can provide their own or omit StoreMediaSkill entirely.
 */

import { Skill } from '../../core/skill';
import type { HookData, HookResult, Context, StructuredToolResult } from '../../core/types';
import type { ContentItem } from '../../uamp/types';
import type { MediaSupport } from '../../adapters/types';
import { getContentItemUrl, isMediaContent } from '../../uamp/content';

const CONTENT_URL_RE = /\/api\/content\/([0-9a-f-]{36})/g;
const BASE64_RE = /^data:([^;]+);base64,/;

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

/** Structured result from MediaSaver including the content_id directly. */
export interface MediaSaverResult {
  url: string;
  content_id: string;
}

/**
 * Saves generated media (e.g., images from LLM output) to storage.
 * Returns a URL string or a MediaSaverResult with both url and content_id.
 */
export interface MediaSaver {
  save(
    base64: string,
    mimeType: string,
    meta?: { chatId?: string; agentId?: string; userId?: string },
  ): Promise<string | MediaSaverResult>;
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

export interface StoreMediaSkillConfig {
  resolver?: MediaResolver;
  saver?: MediaSaver;
  /** Default media support if adapter doesn't declare it */
  defaultMediaSupport?: MediaSupport;
}

/** @deprecated Use StoreMediaSkillConfig */
export type MediaSkillConfig = StoreMediaSkillConfig;

export class StoreMediaSkill extends Skill {
  private resolver?: MediaResolver;
  private saver?: MediaSaver;
  private cache = new Map<string, CachedMedia>();

  constructor(config: StoreMediaSkillConfig = {}) {
    super({ name: 'media' });
    this.resolver = config.resolver;
    this.saver = config.saver;
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
        lifecycle: 'after_tool' as const,
        priority: 10,
        enabled: true,
        handler: this.afterToolCall.bind(this),
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
    if (this.saver && !context.get('_media_saver')) {
      context.set('_media_saver', this.saver);
    }

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
        console.warn(`[StoreMediaSkill] Failed to resolve ${url}:`, (err as Error).message);
      }
    }));

    if (resolvedMedia.size > 0) {
      context.set('_resolved_images', resolvedMedia);
    }
  }

  /**
   * After a tool call: detect non-/content media in content_items,
   * upload via saver, replace with /api/content/UUID, append URLs to text.
   */
  private async afterToolCall(
    data: HookData,
    context: Context,
  ): Promise<HookResult | void> {
    if (!this.saver) return;

    const toolResult = data.tool_result;
    if (!toolResult || typeof toolResult !== 'object') return;

    const structured = toolResult as StructuredToolResult;
    if (!structured.content_items || structured.content_items.length === 0) return;

    const chatId = context.metadata?.chatId as string | undefined;
    const agentId = context.metadata?.agentId as string | undefined;
    const userId = context.auth?.user_id;
    const meta = { chatId, agentId, userId };

    const updatedItems: ContentItem[] = [];
    const savedUrls: string[] = [];
    let changed = false;

    for (const ci of structured.content_items) {
      if (!isMediaContent(ci)) {
        updatedItems.push(ci);
        continue;
      }

      const url = getContentItemUrl(ci);
      if (!url) {
        updatedItems.push(ci);
        continue;
      }

      // Already stored in /content -- nothing to do
      if (url.includes('/api/content/')) {
        updatedItems.push(ci);
        savedUrls.push(url);
        continue;
      }

      // base64 data URI
      const b64Match = url.match(BASE64_RE);
      if (b64Match) {
        const mimeType = b64Match[1];
        const base64Data = url.replace(BASE64_RE, '');
        try {
          const saveResult = await this.saver.save(base64Data, mimeType, meta);
          const savedUrl = typeof saveResult === 'string' ? saveResult : saveResult.url;
          const contentId = typeof saveResult === 'string'
            ? saveResult.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i)?.[1]
            : saveResult.content_id;
          const field = ci.type as 'image' | 'video' | 'audio' | 'file';
          const updated = { ...ci, [field]: { url: savedUrl }, content_id: contentId || ci.content_id };
          updatedItems.push(updated as ContentItem);
          savedUrls.push(savedUrl);
          changed = true;
          console.log(`[StoreMediaSkill] Saved base64 ${ci.type} → ${savedUrl}`);
        } catch (err) {
          console.warn(`[StoreMediaSkill] Failed to save base64 ${ci.type}:`, (err as Error).message);
          updatedItems.push(ci);
        }
        continue;
      }

      // External temp CDN URL -- download and save
      try {
        const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();
        const mimeType = response.headers.get('content-type') || 'application/octet-stream';
        const base64Data = Buffer.from(buffer).toString('base64');
        const saveResult = await this.saver.save(base64Data, mimeType, meta);
        const savedUrl = typeof saveResult === 'string' ? saveResult : saveResult.url;
        const contentId = typeof saveResult === 'string'
          ? saveResult.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i)?.[1]
          : saveResult.content_id;
        const field = ci.type as 'image' | 'video' | 'audio' | 'file';
        const updated = { ...ci, [field]: { url: savedUrl }, content_id: contentId || ci.content_id };
        updatedItems.push(updated as ContentItem);
        savedUrls.push(savedUrl);
        changed = true;
        console.log(`[StoreMediaSkill] Downloaded and saved ${ci.type} from temp URL → ${savedUrl}`);
      } catch (err) {
        console.warn(`[StoreMediaSkill] Failed to download/save ${ci.type} from ${url}:`, (err as Error).message);
        updatedItems.push(ci);
      }
    }

    if (!changed) return;

    const resultText = (structured.text || '').replace(/\n{3,}/g, '\n\n').trim();

    const modifiedResult: StructuredToolResult = {
      text: resultText,
      content_items: updatedItems,
    };

    // Preserve any extra fields (e.g. _billing metadata)
    for (const key of Object.keys(structured)) {
      if (key !== 'text' && key !== 'content_items') {
        (modifiedResult as unknown as Record<string, unknown>)[key] = (structured as unknown as Record<string, unknown>)[key];
      }
    }

    console.log(`[StoreMediaSkill] after_tool: saved ${savedUrls.length} media items, urls=${savedUrls.join(', ')}`);
    return { tool_result: modifiedResult };
  }

  /**
   * Scan response for generated images (base64), save via saver.
   */
  private async afterLLMCall(
    _data: HookData,
    context: Context,
  ): Promise<HookResult | void> {
    if (!this.saver) return;

    const inlineImages = context.get<Array<{ base64: string; mimeType: string }>>('_inline_images');
    if (!inlineImages || inlineImages.length === 0) return;

    const chatId = context.metadata?.chatId as string | undefined;
    const agentId = context.metadata?.agentId as string | undefined;
    const userId = context.auth?.user_id;
    const savedUrls: string[] = [];

    for (const img of inlineImages) {
      try {
        const saveResult = await this.saver.save(img.base64, img.mimeType, { chatId, agentId, userId });
        const url = typeof saveResult === 'string' ? saveResult : saveResult.url;
        savedUrls.push(url);
      } catch (err) {
        console.warn(`[StoreMediaSkill] Failed to save generated image:`, (err as Error).message);
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

/** @deprecated Use StoreMediaSkill */
export const MediaSkill = StoreMediaSkill;
