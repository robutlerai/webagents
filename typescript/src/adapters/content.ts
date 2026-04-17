/**
 * Generic UAMP Content Helpers
 *
 * Provider-agnostic utilities for detecting and extracting UAMP content items.
 * Provider-specific conversion functions live in their respective adapter files.
 */

/**
 * Extract a URL string from a UAMP content ref (string or { url: string }).
 */
export function extractContentRef(ref: unknown): string | null {
  if (typeof ref === 'string') return ref;
  if (ref && typeof ref === 'object' && 'url' in ref) return (ref as { url: string }).url;
  return null;
}

/**
 * Detect whether a content value is a UAMP content_items array
 * (items with .image/.audio/.video/.file fields) rather than
 * an OpenAI-format array (.image_url/.input_audio).
 */
export function isUAMPContentArray(content: unknown): boolean {
  if (!Array.isArray(content) || content.length === 0) return false;
  return content.some((item: Record<string, unknown>) =>
    (item.type === 'image' && 'image' in item) ||
    (item.type === 'audio' && 'audio' in item) ||
    (item.type === 'video' && 'video' in item) ||
    (item.type === 'file' && 'file' in item),
  );
}

/**
 * Given a content URL (or explicit content_id), return the canonical form
 * `/api/content/<uuid>` or null.
 *
 * When `content_id` is provided it is used directly, avoiding regex parsing.
 * The URL-based fallback is kept for backward compatibility with messages that
 * embed content references as URLs in text.
 */
export function canonicalContentUrl(url: string, content_id?: string): string | null {
  if (content_id) return `/api/content/${content_id}`;
  const m = /\/api\/content\/([0-9a-f-]{36})/.exec(url);
  return m ? `/api/content/${m[1]}` : null;
}

export interface DescribeContentOptions {
  /** Modalities supported by the current LLM provider (e.g. 'image', 'audio', 'video'). */
  supportedModalities?: Set<string>;
  /** Document MIME types supported by the current LLM provider for inline analysis. */
  supportedDocMimes?: Set<string>;
}

/**
 * Generate a rich text description for a UAMP content item.
 * Used by adapters to provide metadata to the LLM instead of raw media bytes.
 *
 * When `options` are provided, the closing hint is modality-aware: items the
 * current provider cannot process omit the `read_content` suggestion.
 */
export function describeContentItem(item: Record<string, unknown>, options?: DescribeContentOptions): string {
  const type = (item.type as string) || 'unknown';
  const cid = (item.content_id as string) || 'unknown';
  const parts = [`content_id=${cid}`];

  if (item.duration_ms) {
    const sec = Math.round(item.duration_ms as number / 1000);
    parts.push(`duration=${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`);
  }
  if (item.filename) parts.push(`filename=${item.filename}`);
  if (item.mime_type) parts.push(item.mime_type as string);
  if (item.format) parts.push(`format=${item.format}`);
  if (item.size_bytes) parts.push(`${Math.round(item.size_bytes as number / 1024)}KB`);
  if (item.alt_text) parts.push(`alt="${item.alt_text}"`);
  if (item.title) parts.push(`title="${item.title}"`);
  const dims = item.dimensions as { width?: number; height?: number } | undefined;
  if (dims && dims.width && dims.height) parts.push(`${dims.width}x${dims.height}`);
  const thumbnail = item.thumbnail;
  if (thumbnail) {
    const thumbUrl = extractContentRef(thumbnail);
    if (thumbUrl) parts.push(`thumbnail=${thumbUrl}`);
  }

  const desc = item.description as string | undefined;
  const descSuffix = desc ? `. "${desc}"` : '';

  const analysable = isModalitySupported(type, item.mime_type as string | undefined, options);
  const hint = analysable
    ? `To analyse this content you MUST call read_content('${cid}') first. Use present(content_id) to display to user.`
    : 'NOT analysable by current model \u2014 describe from metadata only. Use present(content_id) to display to user.';

  return `[Available ${type}: ${parts.join(', ')}${descSuffix}. ${hint}]`;
}

function isModalitySupported(type: string, mimeType: string | undefined, options?: DescribeContentOptions): boolean {
  if (!options) return true;
  if (type === 'file' && mimeType && options.supportedDocMimes) {
    return options.supportedDocMimes.has(mimeType);
  }
  if (options.supportedModalities) {
    return options.supportedModalities.has(type);
  }
  return true;
}

export type ResolvedMediaMap = Map<string, { mimeType: string; base64: string; thoughtSignature?: string }>;
