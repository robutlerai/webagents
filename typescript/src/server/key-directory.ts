/**
 * The well-known HTTP Message Signatures directory: the key set a
 * `legacy-string` signer's `Signature-Agent` resolves to (2026-09-19).
 *
 * WHY IT EXISTS. `SignRequestOptions.form: 'legacy-string'` sends a bare
 * ORIGIN as `Signature-Agent`, the form an edge that reads nothing else
 * accepts. A verifier resolves a bare origin to
 * `{origin}/.well-known/http-message-signatures-directory` and REQUIRES the
 * answer's media type to be
 * `application/http-message-signatures-directory+json`
 * (draft-ietf-webbotauth-httpsig-protocol-00 sections 5.5.1 and 8.1; the
 * platform's lib/auth/web-bot-auth/discovery.ts refuses any other type as
 * `key_set_invalid`). Neither SDK server served that path, so selecting the
 * form against an agent hosted by one failed discovery on every request and
 * counted against the platform's failed-signature window: the option was a
 * trap. The signer keeps the form; the servers now serve what it names.
 *
 * WHAT IT CARRIES. The Ed25519 public keys of EVERY agent the server hosts,
 * because a bare origin cannot say which agent signed. Each entry is what the
 * agent's own key set publishes, `{ kty, crv, x, kid, use }`, where `kid` IS
 * the RFC 7638 thumbprint, which is what the directory rule demands (a `kid`
 * at the well-known directory that is not the thumbprint is a refusal).
 * No other key type is listed: the platform counts entries against a cap of
 * 16 (`KEY_SET_MAX_KEYS`) before it filters them, so a server whose agents
 * hold more than `DIRECTORY_MAX_KEYS` keys between them publishes a directory
 * the platform refuses. It is served as it is, with a warning, rather than
 * truncated: dropping a key would fail one agent silently.
 *
 * WHAT THE FORM MEANS, which serving this does not change: the principal of a
 * `legacy-string` signature is the ORIGIN, so every agent that selects the
 * form on one origin is ONE principal to the platform, and registering it
 * also needs a card at `{origin}/.well-known/agent.json`, which only an agent
 * mounted at the origin root serves. Agents mounted under a path should keep
 * the default `dictionary-typed` form, which names their own key set.
 *
 * Identical in the Python server (`webagents/server/core/app.py`).
 */

import type { PublishedJwk } from '../crypto/identity';

/** P section 8.1: the path appended to a bare origin. */
export const DIRECTORY_WELL_KNOWN_PATH = '/.well-known/http-message-signatures-directory';
/** P section 5.5.1: the media type the directory MUST be served with. */
export const DIRECTORY_MEDIA_TYPE = 'application/http-message-signatures-directory+json';
/** The platform's `KEY_SET_MAX_KEYS`: a published set with more entries is refused whole. */
export const DIRECTORY_MAX_KEYS = 16;

/** What the directory needs from an agent's identity; `AgentIdentity` satisfies it. */
export interface DirectoryKeySource {
  getJwks(): { keys: PublishedJwk[] };
}

/** The union of the hosted agents' published Ed25519 keys, one entry per thumbprint, in hosting order. */
export function directoryKeys(identities: Iterable<DirectoryKeySource | undefined>): PublishedJwk[] {
  const seen = new Set<string>();
  const keys: PublishedJwk[] = [];
  for (const identity of identities) {
    if (!identity) continue;
    for (const key of identity.getJwks().keys) {
      if (key.kty !== 'OKP' || key.crv !== 'Ed25519' || seen.has(key.kid)) continue;
      seen.add(key.kid);
      keys.push({ kty: key.kty, crv: key.crv, x: key.x, kid: key.kid, use: 'sig' });
    }
  }
  return keys;
}

/** True for `GET {origin}/.well-known/http-message-signatures-directory`: the ORIGIN path only, whatever prefix the agents are mounted under. */
export function isKeyDirectoryRequest(method: string, pathname: string): boolean {
  return method === 'GET' && pathname === DIRECTORY_WELL_KNOWN_PATH;
}

/**
 * The directory answer: 200 with the directory media type and the same
 * `Cache-Control: max-age` the key set carries (the platform's refresh clamp
 * reads it), or 404 when no hosted agent has a signing identity.
 */
export function keyDirectoryResponse(
  identities: Iterable<DirectoryKeySource | undefined>,
  extraHeaders: Record<string, string> = {},
): Response {
  const keys = directoryKeys(identities);
  if (keys.length === 0) {
    return new Response(JSON.stringify({ error: 'No signing identity is hosted here' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json', ...extraHeaders },
    });
  }
  if (keys.length > DIRECTORY_MAX_KEYS) {
    console.warn(
      `[webagents] the signatures directory lists ${keys.length} keys and the platform refuses a key set above ${DIRECTORY_MAX_KEYS}: ` +
        'a legacy-string signature from this origin will not verify. Sign with the default dictionary-typed form instead.',
    );
  }
  return new Response(JSON.stringify({ keys }), {
    status: 200,
    headers: { 'Content-Type': DIRECTORY_MEDIA_TYPE, 'Cache-Control': 'public, max-age=3600', ...extraHeaders },
  });
}
