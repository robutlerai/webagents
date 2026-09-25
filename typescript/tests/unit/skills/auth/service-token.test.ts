/**
 * AuthSkill service token integration tests — RS256 via JWKS
 *
 * A verified platform service token is NOT an admin credential: the platform
 * is relaying a chat turn on behalf of a sender (`metadata.sender.id` on the
 * completions request), so the auth context is attributed to that sender,
 * with owner elevation only when the sender is this agent's owner. The old
 * suite pinned the blanket `AuthScope.ADMIN` + `['admin','*']` grant.
 *
 * Most tokens here omit `aud` (the pre-rollout platform shape); the
 * verifier's one-release transition window accepts that — see
 * jwks-service.test.ts for the audience matrix. Owner elevation needs a token
 * addressed to this agent's own URL, and reads the platform's signed `sender`
 * claim before the body (S-240, 2026-09-25): the last block below.
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

const AGENT_URL = 'https://agent.example.com/agents/mini';

async function signServiceToken(
  sub: string,
  options: { aud?: string; sender?: Record<string, unknown> } = {},
): Promise<string> {
  // No aud claim unless asked: the legacy platform shape, accepted during
  // the audience-transition window.
  const jwt = new SignJWT({ scopes: ['agents:*'], ...(options.sender ? { sender: options.sender } : {}) })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer(PLATFORM)
    .setSubject(sub)
    .setIssuedAt()
    .setExpirationTime('1h');
  if (options.aud) jwt.setAudience(options.aud);
  return jwt.sign(privateKey);
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
      agentPublicUrl: AGENT_URL,
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
    const token = await signServiceToken('service:robutler-router', { aud: AGENT_URL });
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

describe('AuthSkill service token: who the turn is for (S-240)', () => {
  let jwksFor: (publicUrl?: string) => JWKSManager;

  beforeEach(async () => {
    const kp = await generateKeyPair('RS256');
    privateKey = kp.privateKey;
    const publicJwk = { ...(await exportJWK(kp.publicKey)), kid: 'test-sig-key', use: 'sig', alg: 'RS256' };
    jwksFor = (publicUrl?: string) => {
      const manager = new JWKSManager({ platformApiUrl: PLATFORM, platformIssuer: PLATFORM, agentPublicUrl: publicUrl });
      (manager as any).jwksCache.set(`${PLATFORM}/.well-known/jwks.json`, createLocalJWKSet({ keys: [publicJwk] as any }));
      return manager;
    };
  });

  async function authOf(token: string, publicUrl: string | undefined, bodySender?: string): Promise<Context['auth']> {
    const skill = new AuthSkill({
      jwksManager: jwksFor(publicUrl),
      issuer: PLATFORM,
      audience: PLATFORM,
      requireAuth: false,
      ownerUserId: OWNER_ID,
    });
    const context = createContext({
      metadata: { authorization: `Bearer ${token}`, ...(bodySender ? { sender: { id: bodySender } } : {}) },
    }) as Context;
    await skill.verifyAuth({} as any, context);
    return context.auth;
  }

  it('reads the signed sender before the body', async () => {
    const token = await signServiceToken('service:robutler-router', { aud: AGENT_URL, sender: { id: 'stranger-2' } });
    const auth = await authOf(token, AGENT_URL, OWNER_ID);
    expect(auth.user_id).toBe('stranger-2');
    expect(auth.scope).toBe(AuthScope.USER);

    const owners = await signServiceToken('service:robutler-router', { aud: AGENT_URL, sender: { id: OWNER_ID } });
    const ownerAuth = await authOf(owners, AGENT_URL, 'stranger-2');
    expect(ownerAuth.scope).toBe(AuthScope.OWNER);
    expect(ownerAuth.audienceVerified).toBe(true);
  });

  it('never makes the owner from a token not addressed to this agent', async () => {
    for (const aud of [undefined, 'urn:robutler:agent-endpoint']) {
      const token = await signServiceToken('service:robutler-router', { aud, sender: { id: OWNER_ID } });
      const auth = await authOf(token, AGENT_URL, OWNER_ID);
      expect(auth.authenticated).toBe(true);
      expect(auth.scope).toBe(AuthScope.USER);
      expect(auth.audienceVerified).toBe(false);
    }
  });

  it('refuses owner to the replay S-240 described', async () => {
    // No public URL configured, so `aud` goes unchecked, and the token was
    // minted for another agent whose owner happened to chat there.
    const replayed = await signServiceToken('service:robutler-router', {
      aud: 'https://attacker.example/agents/x',
      sender: { id: OWNER_ID },
    });
    const auth = await authOf(replayed, undefined, OWNER_ID);
    expect(auth.scope).toBe(AuthScope.USER);
  });
});
