/**
 * JWKSManager.verifyServiceToken Unit Tests - RS256 service token validation
 *
 * Portal auth is entirely RS256 JWT via the unified JWKS keyring.
 * Service tokens are RS256 JWTs with sub starting "service:" and
 * verified via the issuer's /.well-known/jwks.json endpoint.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SignJWT, exportJWK, generateKeyPair, createLocalJWKSet } from 'jose';
import { JWKSManager } from '../../../src/crypto/jwks.js';

let privateKey: CryptoKey;
let jwks: JWKSManager;

async function signServiceToken(sub: string): Promise<string> {
  return new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer('https://test.robutler.ai')
    .setSubject(sub)
    .setAudience('https://robutler.ai')
    .setIssuedAt()
    .setExpirationTime('1h')
    .sign(privateKey);
}

describe('JWKSManager.verifyServiceToken', () => {
  beforeEach(async () => {
    const kp = await generateKeyPair('RS256');
    privateKey = kp.privateKey;
    const pub = await exportJWK(kp.publicKey);
    const publicJwk = { ...pub, kid: 'test-sig-key', use: 'sig', alg: 'RS256' };

    jwks = new JWKSManager();

    // Inject a local JWKS into the manager's cache so no HTTP fetch is needed
    const localJwks = createLocalJWKSet({ keys: [publicJwk] as any });
    (jwks as any).jwksCache.set(
      'https://test.robutler.ai/.well-known/jwks.json',
      localJwks,
    );
  });

  it('returns payload for valid RS256 service token', async () => {
    const token = await signServiceToken('service:robutler-router');
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).not.toBeNull();
    expect(payload!.sub).toBe('service:robutler-router');
    expect(payload!.scopes).toEqual(['agents:*']);
  });

  it('returns null for non-service sub (user token)', async () => {
    const token = await new SignJWT({})
      .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
      .setSubject('user-123')
      .setIssuer('https://test.robutler.ai')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(privateKey);
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).toBeNull();
  });

  it('returns null for invalid or malformed token', async () => {
    expect(await jwks.verifyServiceToken('not-a-jwt')).toBeNull();
  });

  it('returns null for HS256 token (only RS256 accepted)', async () => {
    const token = await new SignJWT({ scopes: ['*'] })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject('service:test')
      .setIssuer('https://test.robutler.ai')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(new TextEncoder().encode('some-secret'));
    const result = await jwks.verifyServiceToken(token);
    expect(result).toBeNull();
  });

  it('returns null when issuer is missing', async () => {
    const token = await new SignJWT({ scopes: ['*'] })
      .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
      .setSubject('service:webagentsd')
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(privateKey);
    const result = await jwks.verifyServiceToken(token);
    expect(result).toBeNull();
  });
});
