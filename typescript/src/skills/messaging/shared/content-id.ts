/**
 * Content-ID helpers shared by every messaging skill that supports the
 * Robutler `content_id` outbound-media reference.
 *
 * Centralised here (rather than copied per skill) so:
 *   - the canonical UUID / `/api/content/<uuid>` shape lives in one place,
 *   - per-provider tools share identical normalisation behaviour,
 *   - Portal-side and SDK-side test fixtures can target the same regex.
 */

/** Bare lowercase-or-mixed-case Robutler UUIDv4. */
export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Matches a Robutler UUID embedded in an `/api/content/<uuid>` URL. */
export const CONTENT_PATH_RE =
  /\/api\/content\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;

export interface ExtractContentIdInput {
  contentId?: string | null;
  url?: string | null;
}

/**
 * Resolve a `(contentId | url)` pair to a bare Robutler UUID, or `null`
 * when the input has no content reference (e.g. an absolute external URL
 * with no embedded UUID).
 *
 * Rules:
 *   - explicit `contentId` matching {@link UUID_RE} wins,
 *   - otherwise we look for an `/api/content/<uuid>` substring in `url`.
 *
 * Absolute external URLs (`https://example.com/foo.jpg`) deliberately
 * return `null` — callers send those straight to the platform.
 */
export function extractContentId(input: ExtractContentIdInput): string | null {
  const explicit = input.contentId?.trim();
  if (explicit && UUID_RE.test(explicit)) return explicit;
  const url = input.url?.trim();
  if (url) {
    const m = url.match(CONTENT_PATH_RE);
    if (m) return m[1];
  }
  return null;
}
