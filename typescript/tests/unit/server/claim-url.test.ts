/**
 * `claimUrl`: the link a person opens to take ownership of an agent.
 *
 * It had no test of its own until 2026-09-19, the day the Python SDK gained
 * its twin (`claim_url`, `python/tests/server/test_claim_url.py`, which pins
 * the same things). What matters about the link is one property: a claim
 * token is a bearer until it is spent, so it rides in the URL FRAGMENT, which
 * a browser never sends, and never in the query, which reaches the platform's
 * access logs and any `Referer`. Pinned from the outside: what a server would
 * be sent for this URL carries no token. The shape is
 * `{platform}/claim/{agentUserId}#{token}` and the lifetime ten minutes
 * unless the caller says otherwise.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createLocalJWKSet, jwtVerify } from 'jose';
import { AgentIdentity } from '../../../src/crypto/identity';
import { claimUrl } from '../../../src/server/registration';

const ISSUER = 'https://agent.example.com/agents/mini';
const PLATFORM = 'https://robutler.ai';
const AGENT_USER_ID = '3f0c2a9e-5b7d-4c1e-9a66-0d2f8e1b7c44';

describe('claimUrl', () => {
  let identity: AgentIdentity;

  beforeEach(async () => {
    vi.stubEnv('ROBUTLER_API_URL', '');
    vi.stubEnv('ROBUTLER_INTERNAL_API_URL', '');
    identity = new AgentIdentity({ agentId: 'mini', issuer: ISSUER });
    await identity.initialize();
  });
  afterEach(() => vi.unstubAllEnvs());

  const claimsOf = async (token: string) =>
    (await jwtVerify(token, createLocalJWKSet(identity.getJwks()), { audience: `${PLATFORM}/claim`, algorithms: ['EdDSA'] })).payload;

  it('is {platform}/claim/{id} with the token in the fragment', async () => {
    const url = (await claimUrl(identity, AGENT_USER_ID, { platformUrl: PLATFORM })) as string;
    const pieces = url.split('#'); // exactly one `#`
    expect(pieces).toHaveLength(2);
    expect(pieces[0]).toBe(`${PLATFORM}/claim/${AGENT_USER_ID}`);

    const parsed = new URL(url);
    expect([parsed.origin, parsed.pathname, parsed.search]).toEqual([PLATFORM, `/claim/${AGENT_USER_ID}`, '']);
    expect(parsed.hash).toBe(`#${pieces[1]}`);

    const claims = await claimsOf(pieces[1]);
    expect(claims).toMatchObject({ sub: 'mini', iss: ISSUER, aud: `${PLATFORM}/claim`, scope: 'agent:claim' });
    expect(typeof claims.jti).toBe('string');
  });

  it('nothing a server is sent for the link carries the token', async () => {
    const url = new URL((await claimUrl(identity, AGENT_USER_ID, { platformUrl: PLATFORM })) as string);
    const token = url.hash.slice(1);
    // A browser sends the request target and the host, never the fragment:
    // this is everything an access log or a `Referer` can ever hold.
    url.hash = '';
    expect(url.toString()).toBe(`${PLATFORM}/claim/${AGENT_USER_ID}`);
    for (const part of [token, ...token.split('.')]) expect(url.toString()).not.toContain(part);
  });

  it('lives ten minutes unless the caller says otherwise', async () => {
    const lifetime = async (ttlSeconds?: number) => {
      const url = (await claimUrl(identity, AGENT_USER_ID, { platformUrl: PLATFORM, ttlSeconds })) as string;
      const claims = await claimsOf(url.split('#')[1]);
      return claims.exp! - claims.iat!;
    };
    expect(await lifetime()).toBe(600);
    expect(await lifetime(60)).toBe(60);
  });

  it('trailing slashes on the platform URL change neither the link nor the audience', async () => {
    const url = (await claimUrl(identity, AGENT_USER_ID, { platformUrl: `${PLATFORM}///` })) as string;
    expect(url.startsWith(`${PLATFORM}/claim/${AGENT_USER_ID}#`)).toBe(true);
    expect((await claimsOf(url.split('#')[1])).aud).toBe(`${PLATFORM}/claim`);
  });

  it('every link is its own single-use token', async () => {
    const jti = async () => (await claimsOf(((await claimUrl(identity, AGENT_USER_ID, { platformUrl: PLATFORM })) as string).split('#')[1])).jti;
    expect(await jti()).not.toBe(await jti());
  });

  it('takes the platform from the argument, then the environment, as registration resolves it', async () => {
    const minted: Array<[string, number | undefined]> = [];
    const recording = {
      mintClaimToken: async (platformUrl: string, ttlSeconds?: number) => {
        minted.push([platformUrl, ttlSeconds]);
        return 'header.payload.signature';
      },
    };
    vi.stubEnv('ROBUTLER_INTERNAL_API_URL', 'https://internal.example/');
    expect(await claimUrl(recording, 'u-1')).toBe('https://internal.example/claim/u-1#header.payload.signature');
    vi.stubEnv('ROBUTLER_API_URL', 'https://env.example');
    expect(await claimUrl(recording, 'u-1')).toBe('https://env.example/claim/u-1#header.payload.signature');
    expect(await claimUrl(recording, 'u-1', { platformUrl: PLATFORM })).toBe(`${PLATFORM}/claim/u-1#header.payload.signature`);
    // The audience is minted from the same base the link is built on, slash stripped, for ten minutes.
    expect(minted).toEqual([
      ['https://internal.example', 600],
      ['https://env.example', 600],
      [PLATFORM, 600],
    ]);
  });

  it('no platform URL is no link, and nothing is minted', async () => {
    const mintClaimToken = vi.fn(async () => 'never');
    expect(await claimUrl({ mintClaimToken }, AGENT_USER_ID)).toBeNull();
    expect(mintClaimToken).not.toHaveBeenCalled();
  });
});
