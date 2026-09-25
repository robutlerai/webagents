/**
 * Does this token actually work?
 *
 * WHY (2026-09-23). `webagents login` made NO network call. It wrote whatever
 * was typed to disk and printed "Authenticated successfully", so a mistyped or
 * expired key was reported as a successful login and then failed later,
 * somewhere unrelated, with an error that pointed at the wrong thing.
 *
 * The endpoint is the portal's own CLI token exchange, which is what the Python
 * SDK's `--api-key` path already uses (`python/webagents/cli/platform/auth.py`):
 * `POST /api/auth/cli/token` with the key as a bearer, answering with the CLI
 * JWT and the user it belongs to. Verified to exist against the portal.
 */

export interface ValidationResult {
  ok: boolean;
  /** Why it failed, phrased for a person who just pasted a key. */
  reason?: string;
  username?: string;
  /** The exchanged CLI token, when the portal returned one. */
  accessToken?: string;
}

/** How long to wait before giving up. A login prompt must not hang. */
const TIMEOUT_MS = 15_000;

export async function validateToken(
  portalUrl: string,
  token: string,
  fetchImpl: typeof fetch = fetch,
): Promise<ValidationResult> {
  const url = `${portalUrl.replace(/\/+$/, '')}/api/auth/cli/token`;

  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: '{}',
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (e) {
    const err = e as Error;
    // Distinguish "we could not ask" from "the answer was no". Storing the key
    // anyway on a network blip would be defensible; claiming it is VALID is not.
    return {
      ok: false,
      reason:
        err.name === 'TimeoutError'
          ? `no response from ${portalUrl} within ${TIMEOUT_MS / 1000}s`
          : `could not reach ${portalUrl} (${err.message})`,
    };
  }

  if (response.status === 401 || response.status === 403) {
    return { ok: false, reason: 'the portal rejected that key' };
  }
  if (!response.ok) {
    return { ok: false, reason: `the portal answered ${response.status}` };
  }

  try {
    const body = (await response.json()) as {
      access_token?: string;
      username?: string;
    };
    return { ok: true, username: body.username, accessToken: body.access_token };
  } catch {
    // A 2xx we cannot parse still means the key was accepted; the caller only
    // needs to know it works.
    return { ok: true };
  }
}
