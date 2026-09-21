/**
 * JWKSManager.verifyJwt unit tests: the key set comes from CONFIGURATION,
 * never from the token (S-135, 2026-09-17).
 *
 * Until 2026-09-17 `verifyJwt` decoded the unverified token, took its `iss`,
 * built `${iss}/.well-known/jwks.json` and handed jose a remote key set for
 * that URL. jose resolves the key (the fetch) before it checks a single
 * claim, so a caller-supplied `issuer` never prevented the dial, nothing
 * filtered the destination (a `#` or `?` inside `iss` even swallowed the
 * fixed suffix), and the cache kept one remote-key-set client per distinct
 * `iss` forever. The first describe block is the reproduction: on the
 * pre-fix code its cases fail with the metadata-service URL in the recorded
 * dial list and a cache entry per forged issuer.
 *
 * Harness: jose's node runtime fetches through `node:http` / `node:https`
 * `get`, read off the module namespace at call time. Those two properties
 * are replaced on the CommonJS export object and re-synced into the ESM
 * bindings with `syncBuiltinESMExports`, so every dial the manager attempts
 * is recorded in `dialled` and none leaves the process. `served` is the key
 * set the fake answers with; null answers with a timeout, which jose turns
 * into a JWKSTimeout and the manager into null.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { EventEmitter } from 'node:events';
import { Readable } from 'node:stream';
import { SignJWT, exportJWK, generateKeyPair } from 'jose';
import {
  JWKSManager,
  MAX_KEY_SET_CLIENTS,
  isBlockedIpLiteral,
  keySetUrlFromIssuer,
} from '../../../src/crypto/jwks.js';
import { AuthSkill } from '../../../src/skills/auth/skill.js';
import { createContext } from '../../../src/core/context.js';
import type { Context } from '../../../src/core/types.js';

const require = createRequire(import.meta.url);
type GetModule = { get: (...args: unknown[]) => unknown };
const http = require('node:http') as GetModule;
const https = require('node:https') as GetModule;

const PLATFORM = 'https://robutler.ai';
const INTERNAL = 'http://portal.production.svc.cluster.local';
const METADATA_ISS = 'http://169.254.169.254/latest#';
const KID = 'platform-sig-key';
const KEY_SET_PATH = '/.well-known/jwks.json';

const dialled: string[] = [];
let served: { keys: Record<string, unknown>[] } | null = null;

function fakeGet(url: unknown): EventEmitter {
  dialled.push(String(url));
  const req = Object.assign(new EventEmitter(), { destroy() {} });
  const body = served;
  process.nextTick(() => {
    if (!body) {
      req.emit('timeout');
      return;
    }
    const res = Object.assign(Readable.from([Buffer.from(JSON.stringify(body))]), {
      statusCode: 200,
    });
    req.emit('response', res);
  });
  return req;
}

const realGet = { http: http.get, https: https.get };

let privateKey: CryptoKey;
let publicJwk: Record<string, unknown>;

const b64url = (s: string): string => Buffer.from(s).toString('base64url');

/** A token nobody signed: RS256 header, attacker-chosen claims, junk signature. */
function forgedToken(claims: Record<string, unknown>): string {
  const header = { alg: 'RS256', kid: 'forged', typ: 'JWT' };
  const payload = { sub: 'anyone', exp: Math.floor(Date.now() / 1000) + 3600, ...claims };
  return `${b64url(JSON.stringify(header))}.${b64url(JSON.stringify(payload))}.${b64url('not-a-signature')}`;
}

async function signedToken(
  claims: Record<string, unknown>,
  opts: { iss: string; sub?: string; aud?: string | string[] },
): Promise<string> {
  const builder = new SignJWT(claims)
    .setProtectedHeader({ alg: 'RS256', kid: KID })
    .setIssuer(opts.iss)
    .setSubject(opts.sub ?? 'user-1')
    .setIssuedAt()
    .setExpirationTime('1h');
  if (opts.aud) builder.setAudience(opts.aud);
  return builder.sign(privateKey);
}

const cacheOf = (m: JWKSManager): Map<string, unknown> =>
  (m as unknown as { jwksCache: Map<string, unknown> }).jwksCache;

beforeAll(async () => {
  http.get = fakeGet;
  https.get = fakeGet;
  syncBuiltinESMExports();
  const kp = await generateKeyPair('RS256');
  privateKey = kp.privateKey;
  publicJwk = { ...(await exportJWK(kp.publicKey)), kid: KID, use: 'sig', alg: 'RS256' };
});

afterAll(() => {
  http.get = realGet.http;
  https.get = realGet.https;
  syncBuiltinESMExports();
});

beforeEach(() => {
  vi.stubEnv('ROBUTLER_API_URL', '');
  vi.stubEnv('ROBUTLER_INTERNAL_API_URL', '');
  vi.stubEnv('ROBUTLER_PLATFORM_ISSUER', '');
  vi.stubEnv('WEBAGENTS_PUBLIC_URL', '');
  vi.stubEnv('WEBAGENTS_REQUIRE_SERVICE_AUD', '');
  dialled.length = 0;
  served = null;
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("S-135 reproduction: nothing is ever fetched from the token's own iss", () => {
  it('(a) an unsigned token naming the metadata service as iss triggers no request', async () => {
    const manager = new JWKSManager({ platformApiUrl: PLATFORM, platformIssuer: PLATFORM });
    const result = await manager.verifyJwt(forgedToken({ iss: METADATA_ISS }));
    expect(result).toBeNull();
    expect(dialled).toEqual([]);
  });

  it('(b) an expected issuer the token does not match is checked BEFORE any fetch', async () => {
    const manager = new JWKSManager({});
    const result = await manager.verifyJwt(forgedToken({ iss: METADATA_ISS }), {
      issuer: 'https://idp.example',
    });
    expect(result).toBeNull();
    expect(dialled).toEqual([]);
  });

  it('(c) distinct forged iss values leave no per-iss cache entries behind', async () => {
    const manager = new JWKSManager({ platformApiUrl: PLATFORM, platformIssuer: PLATFORM });
    for (let i = 1; i <= 5; i++) {
      await manager.verifyJwt(forgedToken({ iss: `http://10.0.0.${i}` }));
    }
    expect(cacheOf(manager).size).toBe(0);
    expect(dialled).toEqual([]);
  });

  it('the reachable path: AuthSkill.verifyAuth on a hostile bearer makes no request and grants nothing', async () => {
    // before_run hook, runs on any bearer in the request metadata whatever
    // `requireAuth` says (src/skills/auth/skill.ts verifyAuth).
    const jwksManager = new JWKSManager({ platformApiUrl: PLATFORM, platformIssuer: PLATFORM });
    const skill = new AuthSkill({ jwksManager, requireAuth: false });
    const context = createContext({
      metadata: { authorization: `Bearer ${forgedToken({ iss: METADATA_ISS })}` },
    }) as Context;
    await skill.verifyAuth({} as never, context);
    expect(context.auth?.authenticated).not.toBe(true);
    expect(dialled).toEqual([]);
  });
});

describe('legitimate callers keep working', () => {
  it('AuthSkill.verifyAuth shape (no issuer option): a platform-minted token verifies against the platform key set at the INTERNAL platform URL', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    const result = await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }));
    expect(result?.payload.sub).toBe('user-1');
    expect(dialled).toEqual([`${INTERNAL}${KEY_SET_PATH}`]);
  });

  it('a second verification reuses the fetched key set: one dial, one cache entry', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    expect(await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }))).not.toBeNull();
    expect(await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }))).not.toBeNull();
    expect(dialled).toHaveLength(1);
    expect(cacheOf(manager).size).toBe(1);
  });

  it('an explicit issuer equal to the platform still uses the platform key set', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    const result = await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }), {
      issuer: PLATFORM,
    });
    expect(result?.payload.sub).toBe('user-1');
    expect(dialled).toEqual([`${INTERNAL}${KEY_SET_PATH}`]);
  });

  it('an explicit third-party issuer (AuthSkill `issuer` config) fetches the key set at that CONFIGURED issuer', async () => {
    const manager = new JWKSManager({});
    served = { keys: [publicJwk] };
    const token = await signedToken({}, { iss: 'https://idp.example', aud: 'agent-1' });
    const result = await manager.verifyJwt(token, { issuer: 'https://idp.example', audience: 'agent-1' });
    expect(result?.payload.sub).toBe('user-1');
    expect(dialled).toEqual([`https://idp.example${KEY_SET_PATH}`]);
  });

  it('a trailing slash on the configured issuer is tolerated', async () => {
    const manager = new JWKSManager({});
    served = { keys: [publicJwk] };
    const token = await signedToken({}, { iss: 'https://idp.example' });
    expect(await manager.verifyJwt(token, { issuer: 'https://idp.example/' })).not.toBeNull();
  });

  it('owner-assertion shape (audience only): platform-minted, audience enforced', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    const token = await signedToken({ owner_user_id: 'owner-1' }, { iss: PLATFORM, aud: 'webagents-agent:agent-1' });
    const ok = await manager.verifyJwt(token, { audience: 'webagents-agent:agent-1' });
    expect(ok?.payload.owner_user_id).toBe('owner-1');
    expect(await manager.verifyJwt(token, { audience: 'webagents-agent:other' })).toBeNull();
  });

  it('verifyPaymentToken (PaymentX402Skill): a platform-minted payment token yields its balance', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    const token = await signedToken({ payment: { balance: 4.5 } }, { iss: PLATFORM, aud: 'agent-1' });
    const result = await manager.verifyPaymentToken(token, { expectedAudience: 'agent-1' });
    expect(result?.balance).toBe(4.5);
    expect(dialled).toEqual([`${INTERNAL}${KEY_SET_PATH}`]);
  });

  it('with no expected issuer anywhere, even a validly signed token is refused without a fetch (PaymentX402Skill then falls back to the facilitator)', async () => {
    const manager = new JWKSManager({});
    served = { keys: [publicJwk] };
    const token = await signedToken({ payment: { balance: 1 } }, { iss: 'https://idp.example' });
    expect(await manager.verifyJwt(token)).toBeNull();
    expect(await manager.verifyPaymentToken(token)).toBeNull();
    expect(dialled).toEqual([]);
  });

  it('a platform issuer configured without a platform URL derives the key set from that issuer, filtered', async () => {
    const manager = new JWKSManager({ platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    expect(await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }))).not.toBeNull();
    expect(dialled).toEqual([`${PLATFORM}${KEY_SET_PATH}`]);
  });

  it('a bearer that is not a JWT, or whose issuer is not a URL: null, no throw, no request', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    expect(await manager.verifyJwt('not-a-jwt')).toBeNull();
    expect(await manager.verifyJwt('')).toBeNull();
    expect(await manager.verifyJwt(forgedToken({ iss: 'robutler' }), { issuer: 'robutler' })).toBeNull();
    expect(await manager.verifyJwt(forgedToken({ iss: 42 }))).toBeNull();
    expect(dialled).toEqual([]);
  });

  it('verifyServiceToken keeps its pin through the same resolver (in-cluster http platform URL)', async () => {
    const manager = new JWKSManager({ platformApiUrl: INTERNAL, platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    const token = await signedToken({ scopes: ['agents:*'] }, { iss: PLATFORM, sub: 'service:robutler-router' });
    const payload = await manager.verifyServiceToken(token);
    expect(payload?.sub).toBe('service:robutler-router');
    expect(dialled).toEqual([`${INTERNAL}${KEY_SET_PATH}`]);
  });
});

describe('destination filter on the configured-issuer path', () => {
  const refused: ReadonlyArray<readonly [string, string]> = [
    ['plain http to a non-localhost name', 'http://idp.example'],
    ['userinfo', 'https://user:secret@idp.example'],
    ['query', 'https://idp.example/?jwks=1'],
    ['fragment', 'https://idp.example/#frag'],
    ['metadata address', 'https://169.254.169.254'],
    ['metadata address over http', 'http://169.254.169.254/latest'],
    ['loopback literal', 'https://127.0.0.1'],
    ['loopback literal, high octet', 'https://127.255.255.254'],
    ['private 10/8', 'https://10.0.0.5'],
    ['private 172.16/12', 'https://172.31.255.1'],
    ['private 192.168/16', 'https://192.168.1.1'],
    ['CGNAT 100.64/10', 'https://100.64.0.1'],
    ['this-network 0/8', 'https://0.0.0.0'],
    ['IETF protocol assignments 192.0.0/24', 'https://192.0.0.192'],
    ['benchmarking 198.18/15', 'https://198.19.0.1'],
    ['multicast', 'https://224.0.0.1'],
    ['IPv6 loopback', 'https://[::1]'],
    ['IPv6 unspecified', 'https://[::]'],
    ['IPv6 v4-mapped metadata', 'https://[::ffff:169.254.169.254]'],
    ['IPv6 v4-mapped private, hex spelling', 'https://[::ffff:a00:1]'],
    ['IPv6 unique-local (AWS IMDS v6)', 'https://[fd00:ec2::254]'],
    ['IPv6 link-local', 'https://[fe80::1]'],
    ['NAT64 prefix', 'https://[64:ff9b::a9fe:a9fe]'],
    ['IPv6 multicast', 'https://[ff02::1]'],
    ['non-http scheme', 'ftp://idp.example'],
    ['not a URL', 'robutler'],
  ];

  for (const [what, issuer] of refused) {
    it(`refuses ${what}: ${issuer}`, async () => {
      expect(keySetUrlFromIssuer(issuer)).toBeNull();
      const manager = new JWKSManager({});
      expect(await manager.verifyJwt(forgedToken({ iss: issuer }), { issuer })).toBeNull();
      expect(dialled).toEqual([]);
    });
  }

  it('allows https to a public name and keeps the well-known suffix in the path', () => {
    expect(keySetUrlFromIssuer('https://idp.example/tenant/')).toBe(`https://idp.example/tenant${KEY_SET_PATH}`);
    expect(keySetUrlFromIssuer('https://93.184.216.34')).toBe(`https://93.184.216.34${KEY_SET_PATH}`);
    expect(keySetUrlFromIssuer('https://[2606:4700::1111]')).toBe(`https://[2606:4700::1111]${KEY_SET_PATH}`);
  });

  it('allows http to localhost, the SDK local-development shape, and dials it', async () => {
    expect(keySetUrlFromIssuer('http://localhost:3000')).toBe(`http://localhost:3000${KEY_SET_PATH}`);
    const manager = new JWKSManager({});
    await manager.verifyJwt(forgedToken({ iss: 'http://localhost:3000' }), { issuer: 'http://localhost:3000' });
    expect(dialled).toEqual([`http://localhost:3000${KEY_SET_PATH}`]);
  });

  it('the platform URL is operator configuration: a loopback literal over http is allowed there', async () => {
    const manager = new JWKSManager({ platformApiUrl: 'http://127.0.0.1:3000', platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    expect(await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }))).not.toBeNull();
    expect(dialled).toEqual([`http://127.0.0.1:3000${KEY_SET_PATH}`]);
  });

  it('but a hostile platform URL shape is refused without a request, for user and service tokens alike', async () => {
    const manager = new JWKSManager({ platformApiUrl: 'https://u:p@portal.example', platformIssuer: PLATFORM });
    served = { keys: [publicJwk] };
    expect(await manager.verifyJwt(await signedToken({}, { iss: PLATFORM }))).toBeNull();
    const service = await signedToken({}, { iss: PLATFORM, sub: 'service:robutler-router' });
    expect(await manager.verifyServiceToken(service)).toBeNull();
    expect(dialled).toEqual([]);
  });

  it('isBlockedIpLiteral: names and public literals pass, special ranges do not', () => {
    expect(isBlockedIpLiteral('idp.example')).toBe(false);
    expect(isBlockedIpLiteral('metadata.google.internal')).toBe(false); // names are not resolved here
    expect(isBlockedIpLiteral('93.184.216.34')).toBe(false);
    expect(isBlockedIpLiteral('[2606:4700::1111]')).toBe(false);
    expect(isBlockedIpLiteral('[::ffff:5db8:d822]')).toBe(false); // mapped 93.184.216.34
    expect(isBlockedIpLiteral('169.254.169.254')).toBe(true);
    expect(isBlockedIpLiteral('[::ffff:a9fe:a9fe]')).toBe(true);
    expect(isBlockedIpLiteral('[fd00:ec2::254]')).toBe(true);
    expect(isBlockedIpLiteral('[64:ff9b::1]')).toBe(true);
  });
});

describe('bounded key-set client cache', () => {
  it(`keeps at most ${MAX_KEY_SET_CLIENTS} clients and evicts the least recently used`, () => {
    const manager = new JWKSManager({});
    const get = (u: string): unknown =>
      (manager as unknown as { getJwks: (u: string) => unknown }).getJwks(u);
    const urls = Array.from(
      { length: MAX_KEY_SET_CLIENTS + 3 },
      (_, i) => `https://idp-${i}.example${KEY_SET_PATH}`,
    );
    const hot = urls[0];
    get(hot);
    for (let i = 1; i < urls.length; i++) {
      get(hot); // keep the first client recently used
      get(urls[i]);
    }
    const cache = cacheOf(manager);
    expect(cache.size).toBe(MAX_KEY_SET_CLIENTS);
    expect(cache.has(hot)).toBe(true);
    expect(cache.has(urls[1])).toBe(false);
    expect(cache.has(urls[urls.length - 1])).toBe(true);
    // A hit hands back the same client rather than a fresh one.
    expect(get(hot)).toBe(cache.get(hot));
  });
});
