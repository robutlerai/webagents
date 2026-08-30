/**
 * AuthSkill service token integration tests — RS256 via JWKS
 *
 * A verified platform service token is NOT an admin credential: the platform
 * is relaying a chat turn on behalf of a sender (`metadata.sender.id` on the
 * completions request), so the auth context is attributed to that sender,
 * with owner elevation only when the sender is this agent's owner. The old
 * suite pinned the blanket `AuthScope.ADMIN` + `['admin','*']` grant.
 *
 * Tokens here omit `aud` (the pre-rollout platform shape); the verifier's
 * one-release transition window accepts that — see jwks-service.test.ts for
 * the audience matrix.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SignJWT, exportJWK, generateKeyPair, createLocalJWKSet } from 'jose';
import { AuthSkill } from '../../../../src/skills/auth/skill.js';
import { JWKSManager } from '../../../../src/crypto/jwks.js';
import { AuthScope } from '../../../../src/core/types.js';
import { createContext } from '../../../../src/core/context.js';
import type { Context } from '../../../../src/core/types.js';

const PLATFORM = 'https://robutler.ai';
const OWNER_ID = 'owner-user-1';

let privateKey: CryptoKey;

async function signServiceToken(sub: string): Promise<string> {
  // No aud claim: the legacy platform shape, accepted during the
  // audience-transition window.
  return new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer(PLATFORM)
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

    const jwksManager = new JWKSManager({
      platformApiUrl: PLATFORM,
      platformIssuer: PLATFORM,
    });
    const localJwks = createLocalJWKSet({ keys: [publicJwk] as any });
    (jwksManager as any).jwksCache.set(
      `${PLATFORM}/.well-known/jwks.json`,
      localJwks,
    );

    authSkill = new AuthSkill({
      jwksManager,
      issuer: PLATFORM,
      audience: PLATFORM,
      requireAuth: false,
      ownerUserId: OWNER_ID,
    });
  });

  it('attributes a valid service token to the relayed sender, never as admin', async () => {
    const token = await signServiceToken('service:robutler-router');
    const context = createContext({
      metadata: {
        authorization: `Bearer ${token}`,
        sender: { id: 'user-42', username: 'alice', account_type: 'user' },
      },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe('user-42');
    expect(context.auth.scope).toBe(AuthScope.USER);
    expect(context.auth.scopes).toContain('platform');
    expect(context.auth.scopes).not.toContain('admin');
    expect(context.auth.scopes).not.toContain('*');
    expect(context.auth.provider).toBe('service_token');
  });

  it('elevates to OWNER only when the relayed sender is the agent owner', async () => {
    const token = await signServiceToken('service:robutler-router');
    const context = createContext({
      metadata: {
        authorization: `Bearer ${token}`,
        sender: { id: OWNER_ID, username: 'owner', account_type: 'user' },
      },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe(OWNER_ID);
    expect(context.auth.scope).toBe(AuthScope.OWNER);
    expect(context.auth.scopes).not.toContain('admin');
  });

  it('falls back to the service sub when no sender metadata is present, still not admin', async () => {
    const token = await signServiceToken('service:webagentsd');
    const context = createContext({
      metadata: { authorization: `Bearer ${token}` },
    }) as Context;

    await authSkill.verifyAuth({} as any, context);

    expect(context.auth.authenticated).toBe(true);
    expect(context.auth.user_id).toBe('service:webagentsd');
    expect(context.auth.scope).toBe(AuthScope.USER);
    expect(context.auth.scopes).not.toContain('admin');
    expect(context.hasScope('admin')).toBe(false);
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
      .setIssuer(PLATFORM)
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
      .setIssuer(PLATFORM)
      .setSubject('user-abc')
      .setAudience(PLATFORM)
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
