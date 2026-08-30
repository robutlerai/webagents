/**
 * JWKSManager.verifyServiceToken Unit Tests - RS256 service token validation
 *
 * The verifier is PINNED: the JWKS comes from the CONFIGURED platform URL
 * (never from the token's own unverified `iss`), the issuer must equal the
 * configured platform issuer, and an `aud` claim, when present, must equal
 * the agent's own public URL. The previous test suite pinned the
 * vulnerability instead: it constructed the manager with no platform URL and
 * asserted that a token from an arbitrary issuer verified against that
 * issuer's own JWKS — which is exactly the self-signed `service:*` admin
 * takeover this verifier now refuses.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SignJWT, exportJWK, generateKeyPair, createLocalJWKSet } from 'jose';
import { JWKSManager } from '../../../src/crypto/jwks.js';

const PLATFORM = 'https://test.robutler.ai';
const AGENT_URL = 'https://agent.example.com/agents/mini';

let privateKey: CryptoKey;
let publicJwk: Record<string, unknown>;

function makeManager(overrides: Record<string, unknown> = {}): JWKSManager {
  const jwks = new JWKSManager({
    platformApiUrl: PLATFORM,
    platformIssuer: PLATFORM,
    agentPublicUrl: AGENT_URL,
    ...overrides,
  });
  const localJwks = createLocalJWKSet({ keys: [publicJwk] as any });
  // Inject a local JWKS into the manager's cache — keyed on the PLATFORM
  // URL, which is the only JWKS the verifier may consult.
  (jwks as any).jwksCache.set(`${PLATFORM}/.well-known/jwks.json`, localJwks);
  return jwks;
}

async function signServiceToken(
  sub: string,
  opts: { iss?: string; aud?: string | null; key?: CryptoKey } = {},
): Promise<string> {
  const builder = new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer(opts.iss ?? PLATFORM)
    .setSubject(sub)
    .setIssuedAt()
    .setExpirationTime('1h');
  if (opts.aud !== null) builder.setAudience(opts.aud ?? AGENT_URL);
  return builder.sign(opts.key ?? privateKey);
}

describe('JWKSManager.verifyServiceToken', () => {
  beforeEach(async () => {
    vi.stubEnv('ROBUTLER_API_URL', '');
    vi.stubEnv('ROBUTLER_INTERNAL_API_URL', '');
    vi.stubEnv('ROBUTLER_PLATFORM_ISSUER', '');
    vi.stubEnv('WEBAGENTS_PUBLIC_URL', '');
    vi.stubEnv('WEBAGENTS_REQUIRE_SERVICE_AUD', '');
    const kp = await generateKeyPair('RS256');
    privateKey = kp.privateKey;
    const pub = await exportJWK(kp.publicKey);
    publicJwk = { ...pub, kid: 'test-sig-key', use: 'sig', alg: 'RS256' };
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('returns payload for a valid service token bound to this agent (aud match)', async () => {
    const jwks = makeManager();
    const token = await signServiceToken('service:robutler-router');
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).not.toBeNull();
    expect(payload!.sub).toBe('service:robutler-router');
    expect(payload!.scopes).toEqual(['agents:*']);
  });

  it('TRANSITION: accepts a legacy service token with NO aud claim (default mode)', async () => {
    const jwks = makeManager();
    const token = await signServiceToken('service:robutler-router', { aud: null });
    const payload = await jwks.verifyServiceToken(token);
    expect(payload).not.toBeNull();
    expect(payload!.aud).toBeUndefined();
  });

  it('refuses a no-aud token once requireServiceAudience is on (next release default)', async () => {
    const jwks = makeManager({ requireServiceAudience: true });
    const token = await signServiceToken('service:robutler-router', { aud: null });
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('accepts the platform fallback audience (targetless getServiceToken call sites)', async () => {
    // lib/agents/router.ts SERVICE_TOKEN_FALLBACK_AUD — minted whenever
    // getServiceToken() is called with no target. Refusing it stops the voice
    // relay's outbound leg from authenticating at all.
    const jwks = makeManager();
    const token = await signServiceToken('service:robutler-router', {
      aud: 'urn:robutler:agent-endpoint',
    });
    const payload = await jwks.verifyServiceToken(token);
    expect(payload?.sub).toBe('service:robutler-router');
  });

  it('matches the agent public URL with a trailing slash (both sides normalised)', async () => {
    // The platform rstrips the audience it mints; a WEBAGENTS_PUBLIC_URL with
    // a trailing slash used to refuse every token.
    const jwks = makeManager({ agentPublicUrl: `${AGENT_URL}/` });
    const token = await signServiceToken('service:robutler-router', { aud: AGENT_URL });
    const payload = await jwks.verifyServiceToken(token);
    expect(payload?.sub).toBe('service:robutler-router');
  });

  it('ALWAYS refuses an aud that is not this agent (token for another target)', async () => {
    const jwks = makeManager();
    const token = await signServiceToken('service:robutler-router', {
      aud: 'https://other-agent.example.net',
    });
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('refuses an iss that is not the configured platform, even with a valid signature', async () => {
    const jwks = makeManager();
    const token = await signServiceToken('service:robutler-router', {
      iss: 'https://attacker.example',
    });
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('never consults the JWKS of the token-supplied issuer (self-signed admin attack)', async () => {
    // The attacker self-signs and hosts a matching JWKS at their own origin.
    const attackerKp = await generateKeyPair('RS256');
    const attackerJwkPub = {
      ...(await exportJWK(attackerKp.publicKey)),
      kid: 'attacker-key',
      use: 'sig',
      alg: 'RS256',
    };
    const jwks = makeManager();
    // Even with the attacker's JWKS pre-cached under the attacker's origin,
    // verification must use only the pinned platform JWKS.
    (jwks as any).jwksCache.set(
      'https://attacker.example/.well-known/jwks.json',
      createLocalJWKSet({ keys: [attackerJwkPub] as any }),
    );
    const token = await new SignJWT({ scopes: ['agents:*'] })
      .setProtectedHeader({ alg: 'RS256', kid: 'attacker-key' })
      .setIssuer('https://attacker.example')
      .setSubject('service:x')
      .setAudience(AGENT_URL)
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(attackerKp.privateKey);
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('fails closed when no platform URL is configured at all', async () => {
    const jwks = new JWKSManager({});
    const token = await signServiceToken('service:robutler-router');
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('returns null for non-service sub (user token)', async () => {
    const jwks = makeManager();
    const token = await new SignJWT({})
      .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
      .setSubject('user-123')
      .setIssuer(PLATFORM)
      .setAudience(AGENT_URL)
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(privateKey);
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });

  it('returns null for invalid or malformed token', async () => {
    const jwks = makeManager();
    expect(await jwks.verifyServiceToken('not-a-jwt')).toBeNull();
  });

  it('returns null for HS256 token (only RS256 accepted)', async () => {
    const jwks = makeManager();
    const token = await new SignJWT({ scopes: ['*'] })
      .setProtectedHeader({ alg: 'HS256' })
      .setSubject('service:test')
      .setIssuer(PLATFORM)
      .setIssuedAt()
      .setExpirationTime('1h')
      .sign(new TextEncoder().encode('some-secret'));
    expect(await jwks.verifyServiceToken(token)).toBeNull();
  });
});
