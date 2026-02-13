/**
 * JWKSManager.verifyServiceToken Unit Tests - HS256 service token validation
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SignJWT } from 'jose';
import { JWKSManager } from '../../../src/crypto/jwks.js';

const AUTH_SECRET = 'test-secret-key-for-service-tokens';

async function signServiceToken(sub: string, secret: string = AUTH_SECRET): Promise<string> {
  return new SignJWT({ scopes: ['*'] })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuer('https://test.robutler.ai')
    .setSubject(sub)
    .setIssuedAt()
    .setExpirationTime('1h')
    .sign(new TextEncoder().encode(secret));
}

describe('JWKSManager.verifyServiceToken', () => {
  let jwks: JWKSManager;
  const originalEnv = process.env.AUTH_SECRET;

  beforeEach(() => {
    jwks = new JWKSManager();
    process.env.AUTH_SECRET = AUTH_SECRET;
  });

  afterEach(() => {
    process.env.AUTH_SECRET = originalEnv;
  });

  it('returns payload for valid HS256 service token', async () => {
    const token = await signServiceToken('service:roborum-router');
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).not.toBeNull();
    expect(payload!.sub).toBe('service:roborum-router');
    expect(payload!.scopes).toEqual(['*']);
  });

  it('returns null for wrong secret', async () => {
    const token = await signServiceToken('service:webagentsd', 'wrong-secret');
    process.env.AUTH_SECRET = AUTH_SECRET;
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).toBeNull();
  });

  it('returns null when AUTH_SECRET is missing', async () => {
    const token = await signServiceToken('service:webagentsd');
    delete process.env.AUTH_SECRET;
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).toBeNull();
    process.env.AUTH_SECRET = AUTH_SECRET;
  });

  it('returns null for non-service sub (user token)', async () => {
    const token = await new SignJWT({})
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject('user-123')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(new TextEncoder().encode(AUTH_SECRET));
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).toBeNull();
  });

  it('returns null for invalid or malformed token', async () => {
    expect(await jwks.verifyServiceToken('not-a-jwt')).toBeNull();
    expect(await jwks.verifyServiceToken('eyJhbGciOiJIUzI1NiJ9.bad.payload')).toBeNull();
  });

  it('returns null for RS256 token (only HS256 accepted)', async () => {
    const header = Buffer.from(JSON.stringify({ alg: 'RS256', typ: 'JWT' }))
      .toString('base64url');
    const payload = Buffer.from(JSON.stringify({ sub: 'service:test', iat: 1, exp: 9999999999 }))
      .toString('base64url');
    const fakeSig = Buffer.from('fake').toString('base64url');
    const token = `${header}.${payload}.${fakeSig}`;
    const result = await jwks.verifyServiceToken(token);
    expect(result).toBeNull();
  });
});
