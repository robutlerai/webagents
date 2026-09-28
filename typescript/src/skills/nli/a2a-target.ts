/**
 * Where a `delegate` target goes (2026-09-27, the a2a-delegate lane of the
 * gap-closure build): the platform path the skill always had, or a peer that
 * speaks A2A v1.0 (`../transport/a2a/a2a-client.ts`). The Python twin is
 * `robutler/nli/a2a_target.py`; the shared fixture
 * `python/tests/fixtures/a2a/delegate_routing.json` pins every case and
 * every sentence in both SDKs.
 *
 * THREE ROUTES, DECIDED IN THIS ORDER:
 *   * `peer`: the target is a URL the agent file's `a2a.peers` configures, or
 *     lies under one (the longest configured URL that is the target or a path
 *     prefix of it, the rule `peerTokenFor` applies). The owner wrote that
 *     URL, so its scheme is not judged here: a peer on loopback is a peer.
 *     The hop carries the peer's configured bearer, when there is one.
 *   * `probe`: an https URL that is not the platform's own origin. Its card is
 *     fetched once and, when it names an A2A 1.0 JSON-RPC interface, the hop
 *     goes over A2A with no bearer; a URL that serves no such card stays on
 *     the platform path, exactly as before. Plain http off loopback is never
 *     probed, and loopback is a peer only when configured: a model must not
 *     be able to point the agent at a local port by naming it.
 *   * `platform`: everything else. `@name` and a bare name are agents on the
 *     platform; so is a URL at the platform's origin, unless the owner pinned
 *     that very URL as a peer, which is an explicit ask for A2A.
 *
 * WHAT A PEER HOP NEVER CARRIES: the run's payment token, the caller's auth
 * token (`X-Forwarded-Auth`), an owner assertion, the agent's platform key
 * (`never_sent_headers` in the fixture). A peer outside the platform is paid
 * by nobody, and the result says so with `A2A_UNPAID_NOTE`: no budget is
 * derived for the hop and there is no receipt.
 */

/** The `a2a` skill's `peers` table: URL (or origin) to what it gets. */
export type PeerTable = Record<string, { token?: string } | undefined>;

export type DelegateRoute =
  | { route: 'platform' }
  | { route: 'peer'; url: string; token: string | null }
  | { route: 'probe'; url: string };

/** How long a probe waits for a card (the fixture's `probe.timeout_seconds`). */
export const A2A_PROBE_TIMEOUT_MS = 10_000;
/** Appended to a peer hop's reply (the fixture's `unpaid_note`). */
export const A2A_UNPAID_NOTE =
  '[delegate over A2A: {url} is a peer outside the platform, paid by nobody: no budget applies to this hop and there is no receipt]';
/** A peer hop that could not complete (the fixture's `a2a_failed`). */
export const A2A_FAILED = 'delegation to {url} over A2A failed: {reason}';
/** Attachments name platform content a peer cannot read (the fixture's `attachments_refused`). */
export const A2A_ATTACHMENTS_REFUSED =
  'delegation to {url} refused: attachments cannot be forwarded to a peer outside the platform; put the content in the message or send it another way';
/** What an empty peer reply reads as (the fixture's `result.empty_reply`). */
export const A2A_EMPTY_REPLY = '(no response)';

function parseHttpUrl(value: string): URL | null {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  if (!url.hostname) return null;
  return url;
}

/** The peer entry `url` falls under: the longest configured URL that is `url` or a path prefix of it. */
export function peerEntryFor(url: string, peers: PeerTable): { peer: string; token: string | null } | null {
  const target = url.replace(/\/+$/, '');
  let best: { peer: string; token: string | null; length: number } | null = null;
  for (const [peer, entry] of Object.entries(peers ?? {})) {
    if (typeof peer !== 'string') continue;
    const prefix = peer.trim().replace(/\/+$/, '');
    if (!prefix) continue;
    if (target !== prefix && !target.startsWith(`${prefix}/`)) continue;
    if (best && prefix.length <= best.length) continue;
    const token = entry && typeof entry === 'object' && typeof entry.token === 'string' && entry.token ? entry.token : null;
    best = { peer: prefix, token, length: prefix.length };
  }
  return best ? { peer: best.peer, token: best.token } : null;
}

/** Whether a URL names this machine: `localhost`, `*.localhost`, 127/8 or `::1`. */
export function isLoopbackUrl(value: string): boolean {
  const url = parseHttpUrl(value);
  if (!url) return false;
  const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  return host === 'localhost' || host.endsWith('.localhost') || host.startsWith('127.') || host === '::1';
}

/**
 * The route for `target` (file comment). `platformBase` is where `@name`
 * lives (`NLISkill.platformBase()`); its origin keeps the platform path.
 */
export function classifyDelegateTarget(target: string, options: { peers: PeerTable; platformBase: string }): DelegateRoute {
  const trimmed = (target ?? '').trim();
  if (!trimmed || trimmed.startsWith('@') || !trimmed.includes('://')) return { route: 'platform' };
  const parsed = parseHttpUrl(trimmed);
  if (!parsed) return { route: 'platform' };
  const url = trimmed.replace(/\/+$/, '');
  const peer = peerEntryFor(url, options.peers);
  if (peer) return { route: 'peer', url, token: peer.token };
  if (parsed.protocol !== 'https:') return { route: 'platform' };
  const platform = parseHttpUrl(options.platformBase);
  if (platform && platform.origin === parsed.origin) return { route: 'platform' };
  return { route: 'probe', url };
}
