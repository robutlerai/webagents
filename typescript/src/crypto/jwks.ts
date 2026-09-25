/**
 * JWKS Manager for JWT verification (port from webagents-python).
 * Verifies RS256 JWTs against a remote key set; caches key-set clients via jose.
 *
 * S-135 (2026-09-17): WHERE a key set may be fetched from is decided by
 * configuration, never by the token. Until then `verifyJwt` decoded the
 * unverified token, took its `iss`, built `${iss}/.well-known/jwks.json` and
 * handed jose a remote key set for that URL. jose resolves the key, which is
 * the fetch, before it checks a single claim, so the `issuer` option never
 * prevented the dial; nothing filtered the destination, and a `#` or `?`
 * inside `iss` swallowed the fixed suffix so the path was the caller's
 * choice too; and the per-URL cache kept one remote-key-set client per
 * distinct `iss` forever. Anyone who could reach an SDK agent carrying
 * `AuthSkill` (its `before_run` hook runs `verifyJwt` on any bearer in the
 * request metadata, whatever `requireAuth` says) could make the agent's host
 * GET link-local, private and cloud-metadata addresses, blind, and grow its
 * memory per request. Three rules now hold, pinned by
 * tests/unit/crypto/jwks-verify-jwt.test.ts:
 *   1. the expected issuer is the caller's `issuer` option, else the
 *      configured platform issuer. The unverified `iss` is compared to it
 *      before any URL exists; a mismatch, or no expectation at all, is null
 *      with no fetch.
 *   2. the key-set URL is derived from that configured value: the platform's
 *      own key set (`platformApiUrl`) when the expected issuer is the
 *      platform, otherwise `keySetUrlFromIssuer`, which filters the
 *      destination.
 *   3. the remote-key-set cache is a small LRU (`MAX_KEY_SET_CLIENTS`).
 * jose's node runtime treats any non-200 answer as an error and never follows
 * a redirect, so a permitted origin cannot bounce the fetch elsewhere.
 */

import {
  createRemoteJWKSet,
  jwtVerify,
  decodeJwt,
  decodeProtectedHeader,
  type JWTVerifyResult,
} from 'jose';

/**
 * The audience the platform stamps when the call site has no target to name.
 *
 * `SERVICE_TOKEN_FALLBACK_AUD` in lib/agents/router.ts — used by
 * `getServiceToken()` with no `target` (the voice relay's outbound leg is the
 * live example). It is deliberately NOT the platform issuer, so a token
 * carrying it still cannot be replayed at the portal. Accepting it here is
 * what keeps those legs authenticating; refusing every non-self audience
 * would break them the moment this SDK ships.
 */
export const PLATFORM_FALLBACK_AUDIENCE = 'urn:robutler:agent-endpoint';

/**
 * Upper bound on remote-key-set clients kept per manager. After S-135 every
 * cache key is an operator-configured URL (the platform key set plus any
 * explicitly expected issuer), so a handful is the whole population; the
 * bound exists so that no future caller can turn the cache back into a
 * per-request allocation. Least recently used is evicted.
 */
export const MAX_KEY_SET_CLIENTS = 8;

const KEY_SET_PATH = '/.well-known/jwks.json';

export interface JWKSManagerConfig {
  /** Cache TTL in seconds for JWKS fetches */
  jwksCacheTtl?: number;
  /**
   * The platform (portal) base URL. Platform-signed service tokens are
   * verified against `${platformApiUrl}/.well-known/jwks.json` and ONLY
   * against it — never against a JWKS derived from the token's own
   * unverified `iss` (which let anyone self-sign `service:*` admin).
   * Since S-135 (2026-09-17) the same pin holds for every token `verifyJwt`
   * accepts as platform-issued.
   * Falls back to ROBUTLER_INTERNAL_API_URL / ROBUTLER_API_URL.
   */
  platformApiUrl?: string;
  /** Required `iss` on platform service tokens (defaults to ROBUTLER_PLATFORM_ISSUER, then ROBUTLER_API_URL, then platformApiUrl). */
  platformIssuer?: string;
  /**
   * This agent's own public URL — the expected `aud` of inbound platform
   * service tokens. Falls back to WEBAGENTS_PUBLIC_URL / WEBAGENTS_AGENT_URL.
   */
  agentPublicUrl?: string;
  /**
   * Audience transition flag (one release): when false (default), a service
   * token WITHOUT an `aud` claim is still accepted — the platform only
   * recently began stamping per-target audiences, and requiring `aud`
   * before the platform emits it fails closed for every on-prem agent.
   * A token WITH a mismatched `aud` is ALWAYS refused. Set true (or
   * WEBAGENTS_REQUIRE_SERVICE_AUD=1) to refuse no-aud tokens; that becomes
   * the default next release.
   */
  requireServiceAudience?: boolean;
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

/** A string issuer with surrounding whitespace and trailing slashes removed, or undefined. */
function normaliseIssuer(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim().replace(/\/+$/, '');
  return trimmed || undefined;
}

/**
 * `${base}/.well-known/jwks.json` as a URL every resolver may use, or null:
 * http or https, a host, no userinfo, no query, no fragment, and the
 * well-known path still in the PATH (a base containing `?` or `#` moves the
 * suffix into the query or fragment, which is how the S-135 request path
 * became the caller's choice).
 */
function wellFormedKeySetUrl(base: string): URL | null {
  let url: URL;
  try {
    url = new URL(`${base}${KEY_SET_PATH}`);
  } catch {
    return null;
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return null;
  if (!url.hostname || url.username || url.password || url.search || url.hash) return null;
  if (!url.pathname.endsWith(KEY_SET_PATH)) return null;
  return url;
}

function ipv4Octets(host: string): number[] | null {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return null;
  const octets = m.slice(1).map(Number);
  return octets.every((n) => n <= 255) ? octets : null;
}

/** IANA special-purpose IPv4 ranges: nothing here is a public internet host. */
function isSpecialIpv4([a, b, c]: number[]): boolean {
  return (
    a === 0 || // 0.0.0.0/8 "this network"; 0.0.0.0 itself reaches localhost
    a === 10 || // 10/8
    a === 127 || // loopback
    (a === 100 && b >= 64 && b <= 127) || // 100.64/10 CGNAT, what tailnets hand out
    (a === 169 && b === 254) || // link-local, including 169.254.169.254 metadata
    (a === 172 && b >= 16 && b <= 31) || // 172.16/12
    (a === 192 && b === 0 && c === 0) || // 192.0.0/24 IETF protocol assignments
    (a === 192 && b === 168) || // 192.168/16
    (a === 198 && (b === 18 || b === 19)) || // 198.18/15 benchmarking
    a >= 224 // multicast, reserved, broadcast
  );
}

/**
 * The 16 bytes of an IPv6 literal, or null if it does not parse. The WHATWG
 * URL parser hands us the compressed hex form, but the dotted-quad tail is
 * accepted too, so the check is spelling-independent.
 */
function ipv6Bytes(literal: string): number[] | null {
  let text = literal.split('%')[0];
  const dotted = /(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/.exec(text);
  if (dotted) {
    const q = ipv4Octets(dotted[1]);
    if (!q) return null;
    const hi = ((q[0] << 8) | q[1]).toString(16);
    const lo = ((q[2] << 8) | q[3]).toString(16);
    text = `${text.slice(0, dotted.index)}${hi}:${lo}`;
  }
  const halves = text.split('::');
  if (halves.length > 2) return null;
  const groupsOf = (s: string): number[] | null => {
    if (s === '') return [];
    const out: number[] = [];
    for (const g of s.split(':')) {
      if (!/^[0-9a-fA-F]{1,4}$/.test(g)) return null;
      out.push(parseInt(g, 16));
    }
    return out;
  };
  const head = groupsOf(halves[0]);
  const tail = halves.length === 2 ? groupsOf(halves[1]) : [];
  if (head === null || tail === null || head.length + tail.length > 8) return null;
  const groups =
    halves.length === 2
      ? [...head, ...new Array<number>(8 - head.length - tail.length).fill(0), ...tail]
      : head;
  if (groups.length !== 8) return null;
  return groups.flatMap((g) => [g >> 8, g & 0xff]);
}

/**
 * True when `hostname` (spelled as `URL.hostname` spells it, IPv6 in
 * brackets) is an IP literal in a loopback, private, link-local, CGNAT,
 * metadata, NAT64, multicast or otherwise special range. Names are never
 * resolved here: DNS-level filtering (a public name answering with a private
 * address) is out of the SDK's remit, and after S-135 every URL reaching this
 * check is operator configuration rather than caller input, which bounds
 * what a name can do.
 */
export function isBlockedIpLiteral(hostname: string): boolean {
  const v4 = ipv4Octets(hostname);
  if (v4) return isSpecialIpv4(v4);
  if (!hostname.startsWith('[') || !hostname.endsWith(']')) return false;
  const b = ipv6Bytes(hostname.slice(1, -1));
  if (!b) return true; // an IPv6 literal this parser cannot read: fail closed
  // :: (unspecified, connects to localhost on most stacks) and ::1, checked
  // before the v4-carrying forms, which would read them as 0.0.0.0 / 0.0.0.1.
  if (b.slice(0, 15).every((x) => x === 0) && (b[15] === 0 || b[15] === 1)) return true;
  // 64:ff9b::/32, the NAT64 translation prefixes: every address is a route
  // into another network, refused whole rather than half-parsed.
  if (b[0] === 0x00 && b[1] === 0x64 && b[2] === 0xff && b[3] === 0x9b) return true;
  // ::ffff:a.b.c.d (mapped, what `new URL()` produces) and ::a.b.c.d
  // (deprecated compatible form) are routed as the embedded v4 address.
  const v4Mapped = b.slice(0, 10).every((x) => x === 0) && b[10] === 0xff && b[11] === 0xff;
  const v4Compat = b.slice(0, 12).every((x) => x === 0);
  if (v4Mapped || v4Compat) return isSpecialIpv4(b.slice(12));
  if ((b[0] & 0xfe) === 0xfc) return true; // fc00::/7 unique-local (AWS IMDS fd00:ec2::254 lives here)
  if (b[0] === 0xfe && (b[1] & 0xc0) === 0x80) return true; // fe80::/10 link-local
  if (b[0] === 0xff) return true; // ff00::/8 multicast
  return false;
}

function isLocalhostName(hostname: string): boolean {
  return hostname === 'localhost' || hostname.endsWith('.localhost');
}

/**
 * The key-set URL for a configured expected issuer that is NOT the platform,
 * or null when the destination is refused. Strict, because this is the one
 * place a key-set URL is derived from something other than the platform URL:
 * https only, with http tolerated for `localhost` alone (the SDK's
 * local-development shape: AuthSkill's default platform URL is
 * `http://localhost:3000`, and the local overlay's ROBUTLER_API_URL is
 * `http://localhost`); no userinfo, query or fragment; and no IP literal in
 * a special range (isBlockedIpLiteral). Named hosts are not resolved.
 */
export function keySetUrlFromIssuer(issuer: string): string | null {
  const base = normaliseIssuer(issuer);
  if (!base) return null;
  const url = wellFormedKeySetUrl(base);
  if (!url) return null;
  if (url.protocol === 'http:' && !isLocalhostName(url.hostname)) return null;
  if (isBlockedIpLiteral(url.hostname)) return null;
  return url.href;
}

/**
 * JWKS manager: verify JWTs against the key set of a CONFIGURED issuer.
 * Uses jose's createRemoteJWKSet for fetching and per-client caching.
 */
export class JWKSManager {
  private jwksCache = new Map<string, ReturnType<typeof createRemoteJWKSet>>();
  private platformApiUrl?: string;
  private platformIssuer?: string;
  private agentPublicUrl?: string;
  private requireServiceAudience: boolean;

  constructor(config: JWKSManagerConfig = {}) {
    // jose createRemoteJWKSet handles its own caching
    this.platformApiUrl = (
      config.platformApiUrl ||
      envVar('ROBUTLER_INTERNAL_API_URL') ||
      envVar('ROBUTLER_API_URL') ||
      undefined
    )?.replace(/\/+$/, '');
    this.platformIssuer = (
      config.platformIssuer ||
      envVar('ROBUTLER_PLATFORM_ISSUER') ||
      envVar('ROBUTLER_API_URL') ||
      this.platformApiUrl ||
      undefined
    )?.replace(/\/+$/, '');
    // Trailing slash normalised on OUR side because the platform normalises
    // on ITS side: `serviceTokenAudienceFor` (lib/agents/router.ts) strips a
    // trailing slash and any `/chat/completions` suffix before stamping the
    // `aud`. A `WEBAGENTS_PUBLIC_URL` ending in `/` therefore refused every
    // token, silently, until this rstrip.
    this.agentPublicUrl = (
      config.agentPublicUrl ||
      envVar('WEBAGENTS_PUBLIC_URL') ||
      envVar('WEBAGENTS_AGENT_URL') ||
      undefined
    )?.replace(/\/+$/, '');
    this.requireServiceAudience =
      config.requireServiceAudience ?? envVar('WEBAGENTS_REQUIRE_SERVICE_AUD') === '1';
  }

  /**
   * Remote-key-set client for a URL one of the resolvers produced. Small
   * LRU: a hit moves to the tail, an insert past MAX_KEY_SET_CLIENTS drops
   * the head. Tests pre-seed this map with `createLocalJWKSet` under the
   * same keys, so the key is the exact URL string.
   */
  private getJwks(jwksUri: string): ReturnType<typeof createRemoteJWKSet> {
    const hit = this.jwksCache.get(jwksUri);
    if (hit) {
      this.jwksCache.delete(jwksUri);
      this.jwksCache.set(jwksUri, hit);
      return hit;
    }
    const created = createRemoteJWKSet(new URL(jwksUri));
    this.jwksCache.set(jwksUri, created);
    while (this.jwksCache.size > MAX_KEY_SET_CLIENTS) {
      const oldest = this.jwksCache.keys().next().value;
      if (oldest === undefined) break;
      this.jwksCache.delete(oldest);
    }
    return created;
  }

  /**
   * The platform's own key set, `${platformApiUrl}/.well-known/jwks.json`,
   * or null when no platform URL is configured or it is malformed. The
   * platform URL is the operator's own service and is routinely plain http
   * to an in-cluster name (`ROBUTLER_INTERNAL_API_URL=http://portal.<ns>.svc.cluster.local`
   * on every deployed overlay) or a loopback address in local development,
   * so it gets the well-formedness check only, not the address filter that
   * keySetUrlFromIssuer applies to a non-platform issuer.
   */
  private platformKeySetUrl(): string | null {
    if (!this.platformApiUrl) return null;
    return wellFormedKeySetUrl(this.platformApiUrl)?.href ?? null;
  }

  /**
   * Verify an RS256 JWT against the key set of the issuer the CALLER expects.
   *
   * `options.issuer` is the expected issuer; without it the configured
   * platform issuer is expected. The token's unverified `iss` must equal the
   * expectation before anything is fetched (S-135, see the file comment): an
   * unexpected issuer is null with no request, and with no expectation at
   * all every token is null, which is what lets PaymentX402Skill fall back
   * to the facilitator's verify API on an unconfigured agent. The key set is
   * the platform's own when the expected issuer is the platform (its
   * internal URL, the one an in-cluster agent can reach), otherwise the one
   * keySetUrlFromIssuer derives from the configured issuer.
   *
   * Callers in this SDK and what they pass:
   * - AuthSkill.verifyAuth: `{ issuer: config.issuer, audience: config.audience }`,
   *   both undefined in every construction in the tree, so the platform rule
   *   applies to the bearers it sees;
   * - AuthSkill's two owner-assertion paths: `{ audience: 'webagents-agent:<agentId>' }`;
   * - verifyPaymentToken below, for PaymentX402Skill: `{ audience: expectedAudience }`.
   */
  async verifyJwt(
    token: string,
    options?: { issuer?: string; audience?: string | string[] }
  ): Promise<JWTVerifyResult | null> {
    // A caller-supplied bearer that is not a JWT at all makes `decodeJwt`
    // THROW. This is a verification routine whose contract is "null when it
    // does not verify": letting that throw turned a garbage Authorization
    // header into a 500 from every caller (the completions handler included)
    // instead of an ordinary authentication failure.
    let unverified: ReturnType<typeof decodeJwt>;
    try {
      unverified = decodeJwt(token);
    } catch {
      return null;
    }
    const iss = normaliseIssuer(unverified.iss);
    if (!iss) return null;

    const expectedIssuer = normaliseIssuer(options?.issuer) ?? this.platformIssuer;
    if (!expectedIssuer || iss !== expectedIssuer) return null;

    // Platform issuer: the platform's own key set when a platform URL is
    // configured (a malformed one fails closed, as in verifyServiceToken);
    // only with NO platform URL at all is the key set derived from the
    // issuer, which is operator configuration and goes through the filter.
    const jwksUri =
      expectedIssuer === this.platformIssuer
        ? this.platformApiUrl
          ? this.platformKeySetUrl()
          : keySetUrlFromIssuer(expectedIssuer)
        : keySetUrlFromIssuer(expectedIssuer);
    if (!jwksUri) return null;

    const jwks = this.getJwks(jwksUri);
    try {
      return await jwtVerify(token, jwks, {
        algorithms: ['RS256'],
        issuer: expectedIssuer,
        audience: options?.audience,
      });
    } catch {
      return null;
    }
  }

  /**
   * Verify a payment JWT and return payload with payment claim.
   * When expectedAudience is provided, JWT aud claim must be present and match.
   */
  async verifyPaymentToken(
    token: string,
    options?: { expectedAudience?: string | string[] }
  ): Promise<{ balance: number; payload: Record<string, unknown> } | null> {
    const result = await this.verifyJwt(token, {
      audience: options?.expectedAudience,
    });
    if (!result) return null;
    const payment = (result.payload as Record<string, unknown>).payment as { balance?: number } | undefined;
    if (payment == null || typeof payment.balance !== 'number') return null;
    return {
      balance: payment.balance,
      payload: result.payload as Record<string, unknown>,
    };
  }

  /**
   * Verify an RS256 PLATFORM service JWT. Fails CLOSED.
   *
   * Service tokens have sub starting with "service:" (e.g.
   * service:robutler-router). Requirements:
   * - the JWKS comes from the CONFIGURED platform URL, never from the
   *   token's own unverified `iss` (the old behaviour let anyone self-sign
   *   `{sub:'service:x', iss:'https://attacker.example'}`, host a matching
   *   JWKS, and be verified);
   * - `iss` must equal the configured platform issuer;
   * - `aud`, when present, must equal this agent's own public URL OR the
   *   platform's targetless fallback audience (PLATFORM_FALLBACK_AUDIENCE =
   *   `SERVICE_TOKEN_FALLBACK_AUD` in lib/agents/router.ts). A token
   *   without an `aud` is accepted for exactly one release (see
   *   JWKSManagerConfig.requireServiceAudience) — the platform's per-target
   *   audience rollout lands in the same window.
   *
   * Returns verified payload or null. Never returns on a token whose JWKS,
   * issuer, or audience cannot be positively verified.
   */
  async verifyServiceToken(token: string): Promise<Record<string, unknown> | null> {
    try {
      const header = decodeProtectedHeader(token);
      if (header.alg !== 'RS256') return null;

      const unverified = decodeJwt(token);
      const sub = (unverified.sub ?? '') as string;
      if (typeof sub !== 'string' || !sub.startsWith('service:')) return null;

      // Pinned JWKS: no configured platform URL means no verification path;
      // fail closed rather than trusting the token's own issuer.
      if (!this.platformIssuer) return null;
      const jwksUri = this.platformKeySetUrl();
      if (!jwksUri) return null;
      const jwks = this.getJwks(jwksUri);

      const tokenAud = unverified.aud;
      const verifyOptions: Parameters<typeof jwtVerify>[2] = {
        algorithms: ['RS256'],
        issuer: this.platformIssuer,
      };

      if (tokenAud !== undefined && tokenAud !== null) {
        // aud present: it must match this agent's public URL, or be the
        // platform's targetless fallback audience (see
        // PLATFORM_FALLBACK_AUDIENCE).
        if (!this.agentPublicUrl) {
          if (this.requireServiceAudience) return null;
          console.warn(
            '[webagents] service token carries an aud claim but no agent public URL is ' +
            'configured to check it against. Set WEBAGENTS_PUBLIC_URL — the next release refuses this.',
          );
        } else {
          verifyOptions.audience = [this.agentPublicUrl, PLATFORM_FALLBACK_AUDIENCE];
        }
      } else if (this.requireServiceAudience) {
        // aud absent and the transition window is over.
        return null;
      }

      const result = await jwtVerify(token, jwks, verifyOptions);
      return result.payload as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /**
   * Whether a verified token's `aud` names this agent's own public URL: not
   * the platform's targetless fallback, and never true with no public URL
   * configured (the audience then went unchecked). Only such a service token
   * may make its sender the owner, or name them to the access block (S-240).
   */
  isOwnAudience(aud: unknown): boolean {
    if (!this.agentPublicUrl) return false;
    const values = typeof aud === 'string' ? [aud] : Array.isArray(aud) ? aud : [];
    return values.some((v) => typeof v === 'string' && v.replace(/\/+$/, '') === this.agentPublicUrl);
  }

  invalidateCache(jwksUri?: string): void {
    if (jwksUri) this.jwksCache.delete(jwksUri);
    else this.jwksCache.clear();
  }
}
