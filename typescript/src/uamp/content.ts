import type { ContentItem, ImageContent, VideoContent, AudioContent, FileContent } from './types.js';

/**
 * Extract the URL (string) from any media ContentItem.
 * Returns null for text, tool_call, and tool_result items.
 */
export function getContentItemUrl(ci: ContentItem): string | null {
  if (ci.type === 'image' && 'image' in ci) return typeof ci.image === 'string' ? ci.image : ci.image.url;
  if (ci.type === 'video' && 'video' in ci) return typeof ci.video === 'string' ? ci.video : ci.video.url;
  if (ci.type === 'audio' && 'audio' in ci) return typeof ci.audio === 'string' ? ci.audio : ci.audio.url;
  if (ci.type === 'file' && 'file' in ci) return typeof ci.file === 'string' ? ci.file : ci.file.url;
  return null;
}

/**
 * Type guard: returns true for image, audio, video, and file content items.
 */
export function isMediaContent(ci: ContentItem): ci is ImageContent | AudioContent | VideoContent | FileContent {
  return ci.type === 'image' || ci.type === 'audio' || ci.type === 'video' || ci.type === 'file';
}

/**
 * Ensure a media ContentItem has a content_id.
 * - If already present, returns unchanged (idempotent).
 * - For /api/content/UUID URLs, extracts the UUID from the path.
 * - For base64 or external URLs, generates a new random UUID.
 * - Non-media items (text, tool_call, tool_result) are returned unchanged.
 */
export function ensureContentId<T extends ContentItem>(ci: T): T {
  if (!isMediaContent(ci)) return ci;
  if (ci.content_id) return ci;
  const url = getContentItemUrl(ci);
  const match = url?.match(/\/api\/content\/([0-9a-f-]{36})/i);
  return { ...ci, content_id: match?.[1] ?? crypto.randomUUID() };
}
