/**
 * JWKS Manager for JWT verification (port from webagents-python).
 * Fetches JWKS from issuer and verifies RS256 JWTs; caches keys via jose.
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

export interface JWKSManagerConfig {
  /** Cache TTL in seconds for JWKS fetches */
  jwksCacheTtl?: number;
  /**
   * The platform (portal) base URL. Platform-signed service tokens are
   * verified against `${platformApiUrl}/.well-known/jwks.json` and ONLY
   * against it — never against a JWKS derived from the token's own
   * unverified `iss` (which let anyone self-sign `service:*` admin).
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

/**
 * JWKS manager: verify JWTs using the issuer's /.well-known/jwks.json.
 * Uses jose's createRemoteJWKSet for fetching and caching.
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

  private getJwks(jwksUri: string): ReturnType<typeof createRemoteJWKSet> {
    let jwks = this.jwksCache.get(jwksUri);
    if (!jwks) {
      jwks = createRemoteJWKSet(new URL(jwksUri));
      this.jwksCache.set(jwksUri, jwks);
    }
    return jwks;
  }

  /**
   * Verify a JWT using the issuer's JWKS (from iss claim).
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
    const iss = (unverified.iss ?? '').toString().trim().replace(/\/$/, '');
    if (!iss) return null;
    const jwksUri = `${iss}/.well-known/jwks.json`;
    const jwks = this.getJwks(jwksUri);
    try {
      return await jwtVerify(token, jwks, {
        algorithms: ['RS256'],
        issuer: options?.issuer ?? iss,
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

      // Pinned JWKS: no configured platform URL means no verification path —
      // fail closed rather than trusting the token's own issuer.
      if (!this.platformApiUrl || !this.platformIssuer) return null;

      const jwksUri = `${this.platformApiUrl}/.well-known/jwks.json`;
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

  invalidateCache(jwksUri?: string): void {
    if (jwksUri) this.jwksCache.delete(jwksUri);
    else this.jwksCache.clear();
  }
}
