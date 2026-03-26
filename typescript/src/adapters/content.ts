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

/** Regex to extract a UUID from /api/content/ URLs. */
const CONTENT_UUID_RE = /\/api\/content\/([0-9a-f-]{36})/;

/**
 * Given a content URL, return the canonical form `/api/content/<uuid>` or null.
 */
export function canonicalContentUrl(url: string): string | null {
  const m = CONTENT_UUID_RE.exec(url);
  return m ? `/api/content/${m[1]}` : null;
}

export type ResolvedMediaMap = Map<string, { mimeType: string; base64: string; thoughtSignature?: string }>;
