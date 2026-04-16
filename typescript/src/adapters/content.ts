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

/**
 * Generate a rich text description for a UAMP content item.
 * Used by adapters to provide metadata to the LLM instead of raw media bytes.
 */
export function describeContentItem(item: Record<string, unknown>): string {
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

  const desc = item.description as string | undefined;
  const descSuffix = desc ? `. "${desc}"` : '';

  return `[Available ${type}: ${parts.join(', ')}${descSuffix}. Use present(content_id) to display or read_content(content_id) to analyze.]`;
}

export type ResolvedMediaMap = Map<string, { mimeType: string; base64: string; thoughtSignature?: string }>;
