/**
 * AuthSkill service token integration tests — RS256 via JWKS
 *
 * Portal auth is entirely RS256 JWT. Service tokens are RS256 JWTs
 * with sub starting "service:", verified via the issuer's JWKS endpoint.
 *
 * Service tokens intentionally omit the `aud` claim so they don't match
 * the standard verifyJwt audience check, falling through to the dedicated
 * verifyServiceToken path which grants admin scope.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SignJWT, exportJWK, generateKeyPair, createLocalJWKSet } from 'jose';
import { AuthSkill } from '../../../../src/skills/auth/skill.js';
import { JWKSManager } from '../../../../src/crypto/jwks.js';
import { createContext } from '../../../../src/core/context.js';
import type { Context } from '../../../../src/core/types.js';

let privateKey: CryptoKey;

async function signServiceToken(sub: string): Promise<string> {
  // Service tokens have NO aud claim — they're internal service identities.
  // This ensures they bypass verifyJwt (which checks audience) and reach
  // verifyServiceToken which grants admin scope.
  return new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer('https://robutler.ai')
    .setSubject(sub)
    .setIssuedAt()
    .setExpirationTime('1h')
    .sign(privateKey);
}

describe('AuthSkill service token', () => {
  let authSkill: AuthSkill;

  beforeEach(async () => {
    const kp = await generateKeyPair('RS256');
    privateKey = kp.privateKey;
    const pub = await exportJWK(kp.publicKey);
    const publicJwk = { ...pub, kid: 'test-sig-key', use: 'sig', alg: 'RS256' };

    const jwksManager = new JWKSManager();
    const localJwks = createLocalJWKSet({ keys: [publicJwk] as any });
    (jwksManager as any).jwksCache.set(
      'https://robutler.ai/.well-known/jwks.json',
      localJwks,
    );

    authSkill = new AuthSkill({
      jwksManager,
      issuer: 'https://robutler.ai',
      audience: 'https://robutler.ai',
      requireAuth: false,
    });
  });

  it('sets auth with admin scope when valid RS256 service token is provided', async () => {
    const token = await signServiceToken('service:robutler-router');
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe('service:robutler-router');
    expect(context.auth.scopes).toContain('admin');
    expect(context.auth.provider).toBe('service_token');
  });

  it('service token sets scopes including admin and token scopes', async () => {
    const token = await signServiceToken('service:webagentsd');
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.scopes).toContain('admin');
    expect(context.auth.scopes).toContain('agents:*');
    expect(context.hasScope('admin')).toBe(true);
  });

  it('rejects HS256 token (only RS256 accepted)', async () => {
    const token = await new SignJWT({ scopes: ['*'] })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject('service:robutler-router')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(new TextEncoder().encode('any-secret'));
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(false);
  });

  it('rejects token with wrong signing key', async () => {
    const wrongKp = await generateKeyPair('RS256');
    const token = await new SignJWT({ scopes: ['agents:*'] })
      .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
      .setIssuer('https://robutler.ai')
      .setSubject('service:robutler-router')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(wrongKp.privateKey);

    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(false);
  });

  it('regular JWT with matching audience goes through standard path, not service path', async () => {
    const token = await new SignJWT({ name: 'Alice' })
      .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
      .setIssuer('https://robutler.ai')
      .setSubject('user-abc')
      .setAudience('https://robutler.ai')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(privateKey);

    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe('user-abc');
    expect(context.auth.provider).toBe('jwt');
    expect(context.auth.scopes).not.toContain('admin');
  });
});
