/**
 * AuthSkill HS256 service token integration tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SignJWT } from 'jose';
import { AuthSkill } from '../../../../src/skills/auth/skill.js';
import { JWKSManager } from '../../../../src/crypto/jwks.js';
import { createContext } from '../../../../src/core/context.js';
import type { Context } from '../../../../src/core/types.js';

const AUTH_SECRET = 'test-auth-secret';

async function signServiceToken(sub: string): Promise<string> {
  return new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuer('https://robutler.ai')
    .setSubject(sub)
    .setIssuedAt()
    .setExpirationTime('1h')
    .sign(new TextEncoder().encode(AUTH_SECRET));
}

describe('AuthSkill service token', () => {
  let authSkill: AuthSkill;
  const originalEnv = process.env.AUTH_SECRET;

  beforeEach(() => {
    authSkill = new AuthSkill({ jwksManager: new JWKSManager() });
    process.env.AUTH_SECRET = AUTH_SECRET;
  });

  afterEach(() => {
    process.env.AUTH_SECRET = originalEnv;
  });

  it('sets auth with admin scope when valid service token is provided', async () => {
    const token = await signServiceToken('service:robutler-router');
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context & { setAuth: (a: unknown) => void };

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe('service:robutler-router');
    expect(context.auth.scopes).toContain('admin');
  });

  it('AUTH_SECRET missing causes service token to be rejected', async () => {
    const token = await signServiceToken('service:webagentsd');
    delete process.env.AUTH_SECRET;
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context & { setAuth: (a: unknown) => void };

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(false);
    process.env.AUTH_SECRET = AUTH_SECRET;
  });

  it('wrong secret causes service token to be rejected', async () => {
    const token = await new SignJWT({ scopes: ['*'] })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject('service:robutler-router')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(new TextEncoder().encode('wrong-secret'));
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context & { setAuth: (a: unknown) => void };

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(false);
  });

  it('service token sets scopes including admin', async () => {
    const token = await signServiceToken('service:webagentsd');
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context & { setAuth: (a: unknown) => void };

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.scopes).toContain('admin');
    expect(context.hasScope('admin')).toBe(true);
  });
});
