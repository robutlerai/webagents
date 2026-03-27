import { Skill } from '../../core/skill';
import { hook } from '../../core/decorators';
import type { HookData, HookResult, Context, AuthInfo } from '../../core/types';
import { AuthScope } from '../../core/types';
import { JWKSManager } from '../../crypto/jwks';

export class AuthenticationError extends Error {
  constructor(message = 'Authentication failed') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends Error {
  constructor(message = 'Authorization failed') {
    super(message);
    this.name = 'AuthorizationError';
  }
}

export interface AuthSkillConfig {
  jwksManager?: JWKSManager;
  issuer?: string;
  audience?: string | string[];
  requireAuth?: boolean;
  platformApiUrl?: string;
  apiKey?: string;
  agentId?: string;
  ownerUserId?: string;
  cacheTtl?: number;
}

export class AuthSkill extends Skill {
  private jwks: JWKSManager;
  private issuer?: string;
  private audience?: string | string[];
  private requireAuth: boolean;
  private platformApiUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private ownerUserId?: string;

  constructor(config: AuthSkillConfig = {}) {
    super();
    this.jwks = config.jwksManager ?? new JWKSManager();
    this.issuer = config.issuer;
    this.audience = config.audience;
    this.requireAuth = config.requireAuth ?? true;
    this.platformApiUrl =
      config.platformApiUrl ||
      (typeof process !== 'undefined' && process.env?.ROBUTLER_INTERNAL_API_URL) ||
      (typeof process !== 'undefined' && process.env?.ROBUTLER_API_URL) ||
      'http://localhost:3000';
    this.apiKey = config.apiKey;
    this.agentId = config.agentId;
    this.ownerUserId = config.ownerUserId;
  }

  // ---------------------------------------------------------------------------
  // on_connection: primary multi-mode auth (runs at connection time)
  // ---------------------------------------------------------------------------

  @hook({ lifecycle: 'on_connection', priority: 0 })
  async authenticateConnection(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.requireAuth) return;

    const token = this._extractToken(context);

    // Mode 1: API key validation via platform API
    let authInfo: AuthInfo | null = null;
    if (token) {
      authInfo = await this._authenticateApiKey(token, context);
    }

    // Mode 2: Owner assertion JWT (standalone, without API key)
    if (!authInfo?.authenticated) {
      const assertionAuth = await this._authenticateOwnerAssertion(context);
      if (assertionAuth?.authenticated) {
        this._setAuth(context, assertionAuth);
        return;
      }
    }

    // If API key auth succeeded, apply it
    if (authInfo?.authenticated) {
      this._setAuth(context, authInfo);
      return;
    }

    // Mode 3: Service token (RS256 JWT with sub starting "service:")
    if (token) {
      const serviceAuth = await this._authenticateServiceToken(token);
      if (serviceAuth?.authenticated) {
        this._setAuth(context, serviceAuth);
        return;
      }
    }

    throw new AuthenticationError(
      'Authentication failed (API key, owner assertion, or service token required)',
    );
  }

  // ---------------------------------------------------------------------------
  // before_run: backward-compatible fallback that populates auth from metadata
  // ---------------------------------------------------------------------------

  @hook({ lifecycle: 'before_run', priority: 5 })
  async verifyAuth(_data: HookData, context: Context): Promise<HookResult | void> {
    if (context.auth?.authenticated) return;

    const raw =
      (context.metadata?.authorization as string) ??
      (context.metadata?.Authorization as string);
    if (!raw || typeof raw !== 'string') return;

    const token = raw.trim().replace(/^Bearer\s+/i, '').trim();
    if (!token) return;

    const result = await this.jwks.verifyJwt(token, {
      issuer: this.issuer,
      audience: this.audience,
    });

    if (!result) {
      const servicePayload = await this.jwks.verifyServiceToken(token);
      if (
        servicePayload &&
        typeof servicePayload.sub === 'string' &&
        servicePayload.sub.startsWith('service:')
      ) {
        this._setAuth(context, {
          authenticated: true,
          user_id: servicePayload.sub,
          scopes: ['admin', ...((servicePayload.scopes as string[]) ?? ['*'])],
          scope: AuthScope.ADMIN,
          email: servicePayload.email as string | undefined,
          provider: 'service_token',
          claims: servicePayload,
        });
        return;
      }

      return;
    }

    const payload = result.payload as Record<string, unknown>;
    const sub = payload.sub as string | undefined;
    const aud = payload.aud;
    const scopes = Array.isArray(aud) ? aud : aud ? [String(aud)] : undefined;

    this._setAuth(context, {
      authenticated: true,
      user_id: sub,
      scopes,
      email: payload.email as string | undefined,
      provider: 'jwt',
      claims: payload,
    });
  }

  // ---------------------------------------------------------------------------
  // Token extraction helpers
  // ---------------------------------------------------------------------------

  private _extractToken(context: Context): string | undefined {
    const authHeader =
      (context.metadata?.authorization as string) ??
      (context.metadata?.Authorization as string);
    if (authHeader && typeof authHeader === 'string') {
      const stripped = authHeader.trim().replace(/^Bearer\s+/i, '').trim();
      if (stripped) return stripped;
    }

    const xApiKey =
      (context.metadata?.['x-api-key'] as string) ??
      (context.metadata?.['X-API-Key'] as string);
    if (xApiKey && typeof xApiKey === 'string') {
      return xApiKey.trim();
    }

    const queryKey = context.metadata?.api_key as string;
    if (queryKey && typeof queryKey === 'string') {
      return queryKey.trim();
    }

    return undefined;
  }

  private _extractOwnerAssertion(context: Context): string | undefined {
    const value =
      (context.metadata?.['x-owner-assertion'] as string) ??
      (context.metadata?.['X-Owner-Assertion'] as string);
    if (value && typeof value === 'string') return value.trim();
    return undefined;
  }

  // ---------------------------------------------------------------------------
  // Mode 1: API key validation via platform REST API
  // ---------------------------------------------------------------------------

  private async _authenticateApiKey(
    apiKey: string,
    context: Context,
  ): Promise<AuthInfo | null> {
    try {
      const url = `${this.platformApiUrl.replace(/\/+$/, '')}/api/auth/validate-key`;
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {}),
        },
        body: JSON.stringify({ apiKey }),
      });

      if (!res.ok) return null;

      const body = (await res.json()) as {
        success?: boolean;
        user?: {
          id?: string;
          email?: string;
          is_admin?: boolean;
        };
        message?: string;
      };

      if (!body.success || !body.user) return null;

      const userId = body.user.id;
      let scope: AuthScope;
      if (body.user.is_admin) {
        scope = AuthScope.ADMIN;
      } else if (userId && this._isAgentOwner(userId)) {
        scope = AuthScope.OWNER;
      } else {
        scope = AuthScope.USER;
      }

      const authInfo: AuthInfo = {
        authenticated: true,
        user_id: userId,
        email: body.user.email,
        scope,
        provider: 'api_key',
      };

      // Merge optional owner assertion claims on top of API key auth
      const assertionToken = this._extractOwnerAssertion(context);
      if (assertionToken) {
        try {
          const assertionResult = await this.jwks.verifyJwt(assertionToken, {
            audience: this.agentId ? `webagents-agent:${this.agentId}` : undefined,
          });
          if (assertionResult) {
            const claims = assertionResult.payload as Record<string, unknown>;
            if (
              claims.agent_id &&
              this.agentId &&
              claims.agent_id !== this.agentId
            ) {
              // agent_id mismatch — skip assertion but keep API key auth
            } else {
              authInfo.user_id = (claims.sub as string) || authInfo.user_id;
              authInfo.agent_id = (claims.agent_id as string) || authInfo.agent_id;
              authInfo.assertion = claims;
            }
          }
        } catch {
          // assertion verification failed — continue with API key auth only
        }
      }

      return authInfo;
    } catch {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Mode 2: Owner assertion JWT (standalone, without API key)
  // ---------------------------------------------------------------------------

  private async _authenticateOwnerAssertion(
    context: Context,
  ): Promise<AuthInfo | null> {
    const assertionToken = this._extractOwnerAssertion(context);
    if (!assertionToken) return null;

    try {
      const result = await this.jwks.verifyJwt(assertionToken, {
        audience: this.agentId ? `webagents-agent:${this.agentId}` : undefined,
      });
      if (!result) return null;

      const claims = result.payload as Record<string, unknown>;

      if (claims.agent_id && this.agentId && claims.agent_id !== this.agentId) {
        return null;
      }

      const actingUserId = claims.sub as string | undefined;
      const ownerUserIdClaim = claims.owner_user_id as string | undefined;

      const scope =
        ownerUserIdClaim && this.ownerUserId && ownerUserIdClaim === this.ownerUserId
          ? AuthScope.OWNER
          : AuthScope.USER;

      return {
        authenticated: true,
        user_id: actingUserId,
        agent_id: claims.agent_id as string | undefined,
        scope,
        assertion: claims,
        provider: 'owner_assertion',
        claims,
      };
    } catch {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Mode 3: Service token (RS256 JWT with sub = "service:*", verified via JWKS)
  // ---------------------------------------------------------------------------

  private async _authenticateServiceToken(
    token: string,
  ): Promise<AuthInfo | null> {
    try {
      const payload = await this.jwks.verifyServiceToken(token);
      if (!payload) return null;

      const sub = payload.sub as string | undefined;
      if (!sub || !sub.startsWith('service:')) return null;

      return {
        authenticated: true,
        user_id: sub,
        scope: AuthScope.ADMIN,
        scopes: ['admin', ...((payload.scopes as string[]) ?? ['*'])],
        provider: 'service_token',
        claims: payload,
      };
    } catch {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Ownership check
  // ---------------------------------------------------------------------------

  private _isAgentOwner(userId: string): boolean {
    return !!this.ownerUserId && userId === this.ownerUserId;
  }

  // ---------------------------------------------------------------------------
  // Context auth setter (direct property + setAuth fallback)
  // ---------------------------------------------------------------------------

  private _setAuth(context: Context, authInfo: AuthInfo): void {
    try {
      context.auth = authInfo;
    } catch {
      // property might be readonly in some implementations
    }
    if (
      'setAuth' in context &&
      typeof (context as unknown as { setAuth: (a: AuthInfo) => void }).setAuth === 'function'
    ) {
      (context as unknown as { setAuth: (a: AuthInfo) => void }).setAuth(authInfo);
    }
  }
}
