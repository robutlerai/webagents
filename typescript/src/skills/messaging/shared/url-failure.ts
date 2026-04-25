/**
 * URL-fetch-failure detection shared across messaging skills that use
 * the URL-first → bytes-fallback strategy (Telegram, WhatsApp).
 *
 * Each platform wraps URL-fetch failures in its own taxonomy:
 *   - Telegram: "failed to get HTTP URL content", "wrong type of the
 *     web page content", "wrong remote file identifier", …
 *   - WhatsApp: "Failed to download", "unable to fetch media".
 *
 * We share the transport-level patterns (timeouts, ECONN*, "fetch
 * failed") and let each provider extend the set with platform-specific
 * messages via {@link isUrlFetchFailure}'s second argument.
 */

/** Platform-agnostic patterns that indicate a URL-fetch problem. */
export const URL_FETCH_FAILURE_PATTERNS: RegExp[] = [
  /failed to get HTTP URL content/i,
  /wrong type of the web page content/i,
  /wrong remote file identifier/i,
  /URL host is empty/i,
  /webpage curl failed/i,
  /fetch failed/i,
  /timed? out/i,
  /ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN/i,
];

/**
 * Returns true when the provider error message looks like "I tried to
 * fetch the URL you gave me and couldn't" — those (and only those) are
 * the cases where retrying via multipart bytes is worth attempting.
 *
 * Pass {@link extraPatterns} to extend the set without rewriting all the
 * shared transport patterns (e.g. WhatsApp passes
 * `[/^Failed to download/i, /unable to fetch media/i]`).
 */
export function isUrlFetchFailure(
  message: string | undefined,
  extraPatterns?: RegExp[],
): boolean {
  if (!message) return false;
  if (URL_FETCH_FAILURE_PATTERNS.some((re) => re.test(message))) return true;
  if (extraPatterns && extraPatterns.some((re) => re.test(message))) return true;
  return false;
}
