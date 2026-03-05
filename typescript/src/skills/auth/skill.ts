/**
 * Auth Skill - JWT verification using JWKS
 *
 * Verifies Bearer JWTs (e.g. from Robutler) and populates context.auth.
 * Expects the token in context.metadata.authorization (e.g. "Bearer <jwt>").
 */

import { Skill } from '../../core/skill.js';
import { hook } from '../../core/decorators.js';
import type { HookData, HookResult } from '../../core/types.js';
import type { Context } from '../../core/types.js';
import { JWKSManager } from '../../crypto/jwks.js';

export interface AuthSkillConfig {
  /** JWKS manager (default: new JWKSManager()) */
  jwksManager?: JWKSManager;
  /** Expected issuer for verification */
  issuer?: string;
  /** Expected audience(s) */
  audience?: string | string[];
}

/**
 * Auth skill: before_run hook verifies JWT from metadata.authorization
 * and sets context.auth (user_id, scopes, etc.).
 */
export class AuthSkill extends Skill {
  private jwks: JWKSManager;
  private issuer?: string;
  private audience?: string | string[];

  constructor(config: AuthSkillConfig = {}) {
    super();
    this.jwks = config.jwksManager ?? new JWKSManager();
    this.issuer = config.issuer;
    this.audience = config.audience;
  }

  @hook({ lifecycle: 'before_run', priority: 5 })
  async verifyAuth(data: HookData, context: Context): Promise<HookResult | void> {
    const raw = (context.metadata?.authorization as string) ?? (context.metadata?.Authorization as string);
    if (!raw || typeof raw !== 'string') return;

    const token = raw.trim().replace(/^Bearer\s+/i, '').trim();
    if (!token) return;

    // 1) Try JWKS RS256 verification first
    let result = await this.jwks.verifyJwt(token, {
      issuer: this.issuer,
      audience: this.audience,
    });

    // 2) If no result, try HS256 service token (platform-to-agent)
    if (!result) {
      const servicePayload = await this.jwks.verifyServiceToken(token);
      if (servicePayload && typeof servicePayload.sub === 'string' && servicePayload.sub.startsWith('service:')) {
        if ('setAuth' in context && typeof (context as { setAuth: (a: unknown) => void }).setAuth === 'function') {
          (context as { setAuth: (a: unknown) => void }).setAuth({
            authenticated: true,
            user_id: servicePayload.sub,
            scopes: ['admin', ...((servicePayload.scopes as string[]) ?? ['*'])],
            email: servicePayload.email as string | undefined,
          });
        }
        return;
      }
      return;
    }

    const payload = result.payload as Record<string, unknown>;
    const sub = payload.sub as string | undefined;
    const aud = payload.aud;
    const scopes = Array.isArray(aud) ? aud : aud ? [String(aud)] : undefined;

    if ('setAuth' in context && typeof (context as { setAuth: (a: unknown) => void }).setAuth === 'function') {
      (context as { setAuth: (a: unknown) => void }).setAuth({
        authenticated: true,
        user_id: sub,
        scopes,
        email: payload.email as string | undefined,
      });
    }
  }
}
