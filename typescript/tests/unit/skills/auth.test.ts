/**
 * Comprehensive unit tests for AuthSkill (multi-mode connection auth).
 *
 * Covers:
 *  - Error classes (AuthenticationError, AuthorizationError)
 *  - requireAuth=false bypass
 *  - Mode 1: API key validation via platform REST API
 *  - Mode 2: Owner assertion JWT (standalone)
 *  - Mode 3: Service token
 *  - Fallback ordering across modes
 *  - before_run JWT verification fallback
 *  - Token extraction from various header/query sources
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  AuthSkill,
  AuthenticationError,
  AuthorizationError,
  type AuthSkillConfig,
} from '../../../src/skills/auth/skill.js';
import type { JWKSManager } from '../../../src/crypto/jwks.js';
import type { Context, HookData } from '../../../src/core/types.js';
import { AuthScope } from '../../../src/core/types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockContext(overrides: Partial<Record<string, unknown>> = {}): Context {
  const store = new Map<string, unknown>();
  return {
    session: { id: 'test', created_at: Date.now(), last_activity: Date.now(), data: {} },
    auth: { authenticated: false },
    payment: { valid: false },
    metadata: (overrides.metadata as Record<string, unknown>) ?? {},
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: <T>(key: string, value: T) => store.set(key, value),
    delete: (key: string) => store.delete(key),
    hasScope: (_scope: string) => false,
    hasScopes: (_scopes: string[]) => false,
    ...overrides,
  } as Context;
}

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function createMockJwks() {
  return {
    verifyJwt: vi.fn().mockResolvedValue(null),
    verifyServiceToken: vi.fn().mockResolvedValue(null),
    verifyPaymentToken: vi.fn().mockResolvedValue(null),
    invalidateCache: vi.fn(),
  } as unknown as JWKSManager & {
    verifyJwt: ReturnType<typeof vi.fn>;
    verifyServiceToken: ReturnType<typeof vi.fn>;
  };
}

const hookData: HookData = {};
const AGENT_ID = 'agent-123';
const OWNER_USER_ID = 'owner-456';
const PLATFORM_URL = 'https://platform.test';

// ---------------------------------------------------------------------------
// Global fetch mock
// ---------------------------------------------------------------------------

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue(mockResponse(404, {}));
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ============================= Error Classes ===============================

describe('AuthenticationError', () => {
  it('uses default message', () => {
    const err = new AuthenticationError();
    expect(err.message).toBe('Authentication failed');
    expect(err.name).toBe('AuthenticationError');
  });

  it('accepts custom message', () => {
    const err = new AuthenticationError('bad token');
    expect(err.message).toBe('bad token');
    expect(err.name).toBe('AuthenticationError');
  });

  it('is an instance of Error', () => {
    expect(new AuthenticationError()).toBeInstanceOf(Error);
  });
});

describe('AuthorizationError', () => {
  it('uses default message', () => {
    const err = new AuthorizationError();
    expect(err.message).toBe('Authorization failed');
    expect(err.name).toBe('AuthorizationError');
  });

  it('accepts custom message', () => {
    const err = new AuthorizationError('insufficient scope');
    expect(err.message).toBe('insufficient scope');
    expect(err.name).toBe('AuthorizationError');
  });

  it('is an instance of Error', () => {
    expect(new AuthorizationError()).toBeInstanceOf(Error);
  });
});

// ========================= requireAuth=false ===============================

describe('AuthSkill (requireAuth=false)', () => {
  it('authenticateConnection returns immediately without touching auth', async () => {
    const jwks = createMockJwks();
    const skill = new AuthSkill({ jwksManager: jwks, requireAuth: false });
    const ctx = createMockContext({ metadata: { authorization: 'Bearer some-token' } });

    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(false);
    expect(jwks.verifyJwt).not.toHaveBeenCalled();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

// ========================= Token Extraction ================================

describe('Token extraction', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('extracts token from Bearer authorization header', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      requireAuth: true,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u1', email: 'u@t.co', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer my-api-key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${PLATFORM_URL}/api/auth/validate-key`,
      expect.objectContaining({
        body: JSON.stringify({ apiKey: 'my-api-key' }),
      }),
    );
  });

  it('extracts token from x-api-key header', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      requireAuth: true,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u1', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { 'x-api-key': 'key-from-header' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${PLATFORM_URL}/api/auth/validate-key`,
      expect.objectContaining({
        body: JSON.stringify({ apiKey: 'key-from-header' }),
      }),
    );
  });

  it('extracts token from api_key query param', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      requireAuth: true,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u1', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { api_key: 'key-from-query' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${PLATFORM_URL}/api/auth/validate-key`,
      expect.objectContaining({
        body: JSON.stringify({ apiKey: 'key-from-query' }),
      }),
    );
  });

  it('strips Bearer prefix case-insensitively', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      requireAuth: true,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u1', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'bearer  MY-KEY' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${PLATFORM_URL}/api/auth/validate-key`,
      expect.objectContaining({
        body: JSON.stringify({ apiKey: 'MY-KEY' }),
      }),
    );
  });
});

// ======================== Mode 1: API Key Auth =============================

describe('Mode 1: API key validation', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('ADMIN scope when user is_admin', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, {
        success: true,
        user: { id: 'admin-user', email: 'admin@t.co', is_admin: true },
      }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer test-key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.ADMIN);
    expect(ctx.auth.user_id).toBe('admin-user');
    expect(ctx.auth.email).toBe('admin@t.co');
    expect(ctx.auth.provider).toBe('api_key');
  });

  it('OWNER scope when user id matches ownerUserId', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, {
        success: true,
        user: { id: OWNER_USER_ID, email: 'owner@t.co', is_admin: false },
      }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer test-key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.OWNER);
    expect(ctx.auth.user_id).toBe(OWNER_USER_ID);
  });

  it('USER scope when user is neither admin nor owner', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, {
        success: true,
        user: { id: 'rando-user', email: 'rando@t.co', is_admin: false },
      }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer test-key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.USER);
    expect(ctx.auth.user_id).toBe('rando-user');
  });

  it('merges owner assertion JWT on top of API key auth', async () => {
    const assertionPayload = {
      sub: 'assertion-sub',
      agent_id: AGENT_ID,
      owner_user_id: OWNER_USER_ID,
      extra: 'claim',
    };

    jwks.verifyJwt.mockResolvedValueOnce({ payload: assertionPayload });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, {
        success: true,
        user: { id: 'apikey-user', email: 'ak@t.co', is_admin: false },
      }),
    );

    const ctx = createMockContext({
      metadata: {
        authorization: 'Bearer test-key',
        'x-owner-assertion': 'assertion-jwt-token',
      },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.provider).toBe('api_key');
    expect(ctx.auth.user_id).toBe('assertion-sub');
    expect(ctx.auth.agent_id).toBe(AGENT_ID);
    expect(ctx.auth.assertion).toEqual(assertionPayload);
  });

  it('keeps API key auth when assertion has agent_id mismatch', async () => {
    jwks.verifyJwt.mockResolvedValueOnce({
      payload: { sub: 'assertion-sub', agent_id: 'wrong-agent' },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, {
        success: true,
        user: { id: 'apikey-user', email: 'ak@t.co', is_admin: false },
      }),
    );

    const ctx = createMockContext({
      metadata: {
        authorization: 'Bearer test-key',
        'x-owner-assertion': 'bad-assertion-jwt',
      },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.user_id).toBe('apikey-user');
    expect(ctx.auth.assertion).toBeUndefined();
  });

  it('returns null when HTTP response is not ok', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(500, {}));

    const ctx = createMockContext({ metadata: { authorization: 'Bearer bad-key' } });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('returns null when body.success is false', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: false }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer bad-key' } });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('returns null when body has no user object', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer no-user-key' } });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('sends X-API-Key header when skill has its own apiKey configured', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      apiKey: 'internal-key',
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u1', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer user-key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'internal-key' }),
      }),
    );
  });
});

// ==================== Mode 2: Owner Assertion JWT ==========================

describe('Mode 2: Owner assertion JWT (standalone)', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('grants OWNER scope when owner_user_id matches', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'actor-user',
        agent_id: AGENT_ID,
        owner_user_id: OWNER_USER_ID,
      },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'valid-assertion-jwt' },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.OWNER);
    expect(ctx.auth.user_id).toBe('actor-user');
    expect(ctx.auth.provider).toBe('owner_assertion');
  });

  it('grants USER scope when owner_user_id does not match', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'actor-user',
        agent_id: AGENT_ID,
        owner_user_id: 'different-owner',
      },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'valid-assertion-jwt' },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.USER);
    expect(ctx.auth.provider).toBe('owner_assertion');
  });

  it('returns null when agent_id in JWT does not match configured agentId', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'actor-user',
        agent_id: 'wrong-agent-id',
        owner_user_id: OWNER_USER_ID,
      },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'mismatched-assertion-jwt' },
    });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
    expect(ctx.auth.authenticated).toBe(false);
  });

  it('returns null when JWT verification fails', async () => {
    jwks.verifyJwt.mockResolvedValue(null);

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'invalid-jwt' },
    });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('verifies with correct audience containing agentId', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: { sub: 'u', agent_id: AGENT_ID, owner_user_id: OWNER_USER_ID },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
      platformApiUrl: PLATFORM_URL,
    });

    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'jwt' },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(jwks.verifyJwt).toHaveBeenCalledWith('jwt', {
      audience: `webagents-agent:${AGENT_ID}`,
    });
  });
});

// ==================== Mode 3: Service Token ================================

describe('Mode 3: Service token', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('authenticates with ADMIN scope for valid service token', async () => {
    jwks.verifyServiceToken.mockResolvedValue({
      sub: 'service:robutler-router',
      scopes: ['agents:*'],
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));

    const ctx = createMockContext({ metadata: { authorization: 'Bearer svc-token' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.scope).toBe(AuthScope.ADMIN);
    expect(ctx.auth.user_id).toBe('service:robutler-router');
    expect(ctx.auth.provider).toBe('service_token');
    expect(ctx.auth.scopes).toContain('admin');
    expect(ctx.auth.scopes).toContain('agents:*');
  });

  it('rejects service token when sub does not start with service:', async () => {
    jwks.verifyServiceToken.mockResolvedValue({
      sub: 'user:someone',
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));

    const ctx = createMockContext({ metadata: { authorization: 'Bearer bad-svc' } });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('falls back to default scopes [*] when payload.scopes is absent', async () => {
    jwks.verifyServiceToken.mockResolvedValue({
      sub: 'service:default-scopes',
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));

    const ctx = createMockContext({ metadata: { authorization: 'Bearer svc' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.scopes).toEqual(['admin', '*']);
  });
});

// ======================== Fallback Order ===================================

describe('Fallback order', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('throws AuthenticationError when all modes fail', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));
    jwks.verifyJwt.mockResolvedValue(null);
    jwks.verifyServiceToken.mockResolvedValue(null);

    const ctx = createMockContext({ metadata: { authorization: 'Bearer failing-token' } });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      'API key, owner assertion, or service token required',
    );
  });

  it('throws AuthenticationError when no token is provided at all', async () => {
    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    const ctx = createMockContext({ metadata: {} });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });

  it('falls through API key → owner assertion when API key fails', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));

    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'owner-via-assertion',
        agent_id: AGENT_ID,
        owner_user_id: OWNER_USER_ID,
      },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: {
        authorization: 'Bearer failing-key',
        'x-owner-assertion': 'valid-assertion',
      },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.provider).toBe('owner_assertion');
    expect(ctx.auth.scope).toBe(AuthScope.OWNER);
  });

  it('falls through API key → assertion → service token', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResponse(401, {}));
    jwks.verifyJwt.mockResolvedValue(null);
    jwks.verifyServiceToken.mockResolvedValue({
      sub: 'service:fallback',
      scopes: ['*'],
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
    });

    const ctx = createMockContext({
      metadata: { authorization: 'Bearer svc-fallback' },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.provider).toBe('service_token');
    expect(ctx.auth.scope).toBe(AuthScope.ADMIN);
  });

  it('prefers owner assertion over API key when API key auth returns null', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: false }),
    );

    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'assertion-user',
        agent_id: AGENT_ID,
        owner_user_id: OWNER_USER_ID,
      },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const ctx = createMockContext({
      metadata: {
        authorization: 'Bearer bad-key',
        'x-owner-assertion': 'good-assertion',
      },
    });
    await skill.authenticateConnection(hookData, ctx);

    expect(ctx.auth.provider).toBe('owner_assertion');
  });
});

// ====================== before_run fallback ================================

describe('verifyAuth (before_run hook)', () => {
  let jwks: ReturnType<typeof createMockJwks>;

  beforeEach(() => {
    jwks = createMockJwks();
  });

  it('skips when context.auth is already authenticated', async () => {
    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      auth: { authenticated: true, user_id: 'already-authed' },
      metadata: { authorization: 'Bearer some-jwt' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(jwks.verifyJwt).not.toHaveBeenCalled();
    expect(jwks.verifyServiceToken).not.toHaveBeenCalled();
  });

  it('does nothing when no authorization header is present', async () => {
    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({ metadata: {} });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(false);
  });

  it('sets auth from verified JWT', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: {
        sub: 'jwt-user',
        aud: ['scope-a', 'scope-b'],
        email: 'jwt@test.co',
      },
    });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer valid-jwt' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.user_id).toBe('jwt-user');
    expect(ctx.auth.scopes).toEqual(['scope-a', 'scope-b']);
    expect(ctx.auth.email).toBe('jwt@test.co');
    expect(ctx.auth.provider).toBe('jwt');
  });

  it('handles single string audience', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: { sub: 'u', aud: 'single-scope' },
    });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer jwt' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.scopes).toEqual(['single-scope']);
  });

  it('handles missing audience', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: { sub: 'u' },
    });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer jwt' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.scopes).toBeUndefined();
  });

  it('falls back to service token when JWT verification returns null', async () => {
    jwks.verifyJwt.mockResolvedValue(null);
    jwks.verifyServiceToken.mockResolvedValue({
      sub: 'service:webagentsd',
      scopes: ['agents:*'],
      email: 'svc@internal',
    });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer svc-token' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(ctx.auth.user_id).toBe('service:webagentsd');
    expect(ctx.auth.scope).toBe(AuthScope.ADMIN);
    expect(ctx.auth.scopes).toContain('admin');
    expect(ctx.auth.provider).toBe('service_token');
  });

  it('does nothing when both JWT and service token verification fail', async () => {
    jwks.verifyJwt.mockResolvedValue(null);
    jwks.verifyServiceToken.mockResolvedValue(null);

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer invalid' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(false);
  });

  it('does nothing when service token sub does not start with service:', async () => {
    jwks.verifyJwt.mockResolvedValue(null);
    jwks.verifyServiceToken.mockResolvedValue({ sub: 'not-a-service' });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { authorization: 'Bearer bad-svc' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(false);
  });

  it('reads Authorization header with capital A', async () => {
    jwks.verifyJwt.mockResolvedValue({
      payload: { sub: 'user-cap' },
    });

    const skill = new AuthSkill({ jwksManager: jwks });
    const ctx = createMockContext({
      metadata: { Authorization: 'Bearer cap-jwt' },
    });

    await skill.verifyAuth(hookData, ctx);

    expect(ctx.auth.authenticated).toBe(true);
    expect(jwks.verifyJwt).toHaveBeenCalledWith('cap-jwt', expect.any(Object));
  });
});

// ========================= Configuration ===================================

describe('Configuration', () => {
  it('defaults platformApiUrl to localhost:3000', () => {
    const originalInternal = process.env.ROBUTLER_INTERNAL_API_URL;
    const originalApi = process.env.ROBUTLER_API_URL;
    delete process.env.ROBUTLER_INTERNAL_API_URL;
    delete process.env.ROBUTLER_API_URL;

    const skill = new AuthSkill({ jwksManager: createMockJwks() });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer key' } });
    skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:3000/api/auth/validate-key',
      expect.any(Object),
    );

    process.env.ROBUTLER_INTERNAL_API_URL = originalInternal!;
    process.env.ROBUTLER_API_URL = originalApi!;
  });

  it('uses explicit platformApiUrl over env vars', async () => {
    const skill = new AuthSkill({
      jwksManager: createMockJwks(),
      platformApiUrl: 'https://custom.api',
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'https://custom.api/api/auth/validate-key',
      expect.any(Object),
    );
  });

  it('strips trailing slashes from platformApiUrl', async () => {
    const skill = new AuthSkill({
      jwksManager: createMockJwks(),
      platformApiUrl: 'https://api.test///',
    });

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockResponse(200, { success: true, user: { id: 'u', is_admin: true } }),
    );

    const ctx = createMockContext({ metadata: { authorization: 'Bearer key' } });
    await skill.authenticateConnection(hookData, ctx);

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'https://api.test/api/auth/validate-key',
      expect.any(Object),
    );
  });

  it('requireAuth defaults to true', async () => {
    const jwks = createMockJwks();
    const skill = new AuthSkill({ jwksManager: jwks, platformApiUrl: PLATFORM_URL });
    const ctx = createMockContext({ metadata: {} });

    await expect(skill.authenticateConnection(hookData, ctx)).rejects.toThrow(
      AuthenticationError,
    );
  });
});

// ========================= _setAuth behavior ===============================

describe('_setAuth', () => {
  it('calls context.setAuth when available', async () => {
    const jwks = createMockJwks();
    jwks.verifyJwt.mockResolvedValue({
      payload: { sub: 'u', agent_id: AGENT_ID, owner_user_id: OWNER_USER_ID },
    });

    const skill = new AuthSkill({
      jwksManager: jwks,
      platformApiUrl: PLATFORM_URL,
      agentId: AGENT_ID,
      ownerUserId: OWNER_USER_ID,
    });

    const setAuthSpy = vi.fn();
    const ctx = createMockContext({
      metadata: { 'x-owner-assertion': 'jwt' },
      setAuth: setAuthSpy,
    });

    await skill.authenticateConnection(hookData, ctx);

    expect(setAuthSpy).toHaveBeenCalledWith(
      expect.objectContaining({ authenticated: true }),
    );
  });
});
