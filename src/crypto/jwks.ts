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

export interface JWKSManagerConfig {
  /** Cache TTL in seconds for JWKS fetches */
  jwksCacheTtl?: number;
}

/**
 * JWKS manager: verify JWTs using the issuer's /.well-known/jwks.json.
 * Uses jose's createRemoteJWKSet for fetching and caching.
 */
export class JWKSManager {
  private jwksCache = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

  constructor(_config: JWKSManagerConfig = {}) {
    // jose createRemoteJWKSet handles its own caching
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
    const unverified = decodeJwt(token);
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
   * Verify an HS256 service JWT signed with AUTH_SECRET.
   * Service tokens have sub starting with "service:" (e.g. service:roborum-router, service:webagentsd).
   * Returns verified payload or null.
   */
  async verifyServiceToken(token: string): Promise<Record<string, unknown> | null> {
    try {
      const header = decodeProtectedHeader(token);
      if (header.alg !== 'HS256') return null;

      const unverified = decodeJwt(token);
      const sub = (unverified.sub ?? '') as string;
      if (typeof sub !== 'string' || !sub.startsWith('service:')) return null;

      const secret = process.env.AUTH_SECRET;
      if (!secret) return null;

      const result = await jwtVerify(token, new TextEncoder().encode(secret), {
        algorithms: ['HS256'],
      });
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
