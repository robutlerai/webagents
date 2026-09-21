/**
 * S-180 (found 2026-09-19): the buyer followed redirects on requests that
 * carry a payment credential.
 *
 * `purchase()` set `redirect: 'follow'`, `send()` reused the seed's mode and
 * `payingFetch` inherited the caller's, which defaults to follow. Fetch strips
 * only `Authorization` and `Cookie` on a cross-origin redirect, so
 * `Payment-Authorization` and `Robutler-Terms-Accepted` travelled to whatever
 * the `Location` named, and that host's 2xx was then read as a settled
 * purchase. The realm allowlist (S-147) and the seller pin (S-154) only ever
 * looked at the FIRST URL.
 *
 * The first half of this file is not a stub: two real HTTP servers, the real
 * `fetch`, a real 307. The assertion that matters is that the second server
 * never hears a thing. The second half pins the refusal's shape: typed,
 * spends nothing, releases the reservation, and never discards a credential
 * whose outcome was already in doubt.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { generateKeyPairSync, sign as cryptoSign, createHash } from 'node:crypto';
import type { SigningIdentity } from '../../../../src/crypto/http-signature';
import {
  MppBuyer,
  MppRedirectError,
  MPP_CREDENTIAL_HEADER,
  TERMS_ACCEPTED_HEADER,
  TERMS_VERSION_HEADER,
  base64urlEncode,
  jcsCanonicalize,
  type MppBuyerConfig,
  type MppBuyerRefusal,
  type MppPurchaseRecord,
} from '../../../../src/skills/payments/mpp-buyer';

const AGENT_URL = 'https://agent.example/agents/mini';
const TERMS = '2026-07-31';

function identity(): SigningIdentity {
  const { privateKey } = generateKeyPairSync('ed25519');
  return {
    issuer: AGENT_URL,
    getHeldKeys: () => [{ kid: 'k'.repeat(43), sign: async (data: Uint8Array) => new Uint8Array(cryptoSign(null, Buffer.from(data), privateKey)) }],
  };
}

let counter = 0;
function challengeHeader(realm: string, cents = 500): string {
  counter += 1;
  const fields: Array<[string, string]> = [
    ['id', createHash('sha256').update(`redirect challenge ${counter}`).digest('base64url')],
    ['realm', realm],
    ['method', 'stripe'],
    ['intent', 'charge'],
    ['expires', new Date(Date.now() + 300_000).toISOString()],
    ['request', base64urlEncode(jcsCanonicalize({ amount: String(cents), currency: 'usd', methodDetails: { networkId: 'profile_test', paymentMethodTypes: ['card'] } }))],
    ['opaque', base64urlEncode(jcsCanonicalize({ packId: 'mpp_5', kind: 'pack', terms: TERMS }))],
    ['header', MPP_CREDENTIAL_HEADER],
  ];
  return `Payment ${fields.map(([k, v]) => `${k}="${v}"`).join(', ')}`;
}

interface Heard {
  method: string;
  url: string;
  headers: http.IncomingHttpHeaders;
}

async function listen(handler: (req: http.IncomingMessage, res: http.ServerResponse, heard: Heard[]) => void) {
  const heard: Heard[] = [];
  const server = http.createServer((req, res) => {
    req.resume();
    req.on('end', () => {
      heard.push({ method: req.method ?? '', url: req.url ?? '', headers: req.headers });
      handler(req, res, heard);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return { server, heard, host: `127.0.0.1:${port}`, origin: `http://127.0.0.1:${port}` };
}

function makeBuyer(realm: string, overrides: Partial<MppBuyerConfig> = {}) {
  const purchases: MppPurchaseRecord[] = [];
  const refusals: MppBuyerRefusal[] = [];
  const getSpt = vi.fn(async () => 'spt_test_0001');
  const buyer = new MppBuyer({
    identity: identity(),
    sources: { card: { getSpt } },
    onPurchase: (r) => purchases.push(r),
    onRefusal: (r) => refusals.push(r),
    sleep: async () => {},
    ...overrides,
    policy: { maxPerPurchaseCents: 2000, dailyCapCents: 5000, acceptTerms: TERMS, realms: [realm], stripeProfileId: 'profile_test', ...overrides.policy },
  });
  return { buyer, purchases, refusals, getSpt };
}

describe('S-180: a redirect never carries the credential anywhere (real servers, real fetch)', () => {
  let platform: Awaited<ReturnType<typeof listen>>;
  let elsewhere: Awaited<ReturnType<typeof listen>>;
  /** What the allowlisted host does with a request: set per test. */
  let platformBehaviour: (req: http.IncomingMessage, res: http.ServerResponse) => void;

  beforeEach(async () => {
    elsewhere = await listen((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end('{"ok":true}');
    });
    platform = await listen((req, res) => platformBehaviour(req, res));
  });

  afterEach(async () => {
    await Promise.all([platform, elsewhere].map((s) => new Promise((resolve) => s.server.close(resolve))));
  });

  const redirectTo = (res: http.ServerResponse, status: number) => {
    res.writeHead(status, { Location: `${elsewhere.origin}/collect` });
    res.end();
  };
  const challenge402 = (res: http.ServerResponse) => {
    res.writeHead(402, {
      'WWW-Authenticate': challengeHeader(platform.host),
      'Content-Type': 'application/problem+json',
      [TERMS_VERSION_HEADER]: TERMS,
    });
    res.end(JSON.stringify({ status: 402 }));
  };

  it('a paid retry answered with a redirect: nothing reaches the other host, nothing is spent, the refusal is typed', async () => {
    platformBehaviour = (req, res) => (req.headers[MPP_CREDENTIAL_HEADER.toLowerCase()] ? redirectTo(res, 307) : challenge402(res));
    const b = makeBuyer(platform.host);

    // `redirect: 'follow'` from the caller changes nothing: the buyer decides.
    const attempt = b.buyer.payingFetch(`${platform.origin}/agents/acme/chat/completions`, { method: 'POST', body: '{}', redirect: 'follow' });
    await expect(attempt).rejects.toBeInstanceOf(MppRedirectError);
    await expect(attempt).rejects.toMatchObject({ name: 'MppRedirectError', code: 'redirect_refused', followed: false });

    // The credential left once, to the allowlisted host, and went nowhere else.
    expect(platform.heard.map((h) => Boolean(h.headers['payment-authorization']))).toEqual([false, true]);
    expect(platform.heard[1].headers[TERMS_ACCEPTED_HEADER.toLowerCase()]).toBe(TERMS);
    expect(elsewhere.heard).toEqual([]);

    // Spends nothing, releases the reservation, holds nothing, and is not a purchase.
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    expect(b.purchases).toEqual([]);
    expect(b.refusals.map((r) => r.reason)).toEqual(['redirect_refused']);
    expect(b.refusals[0]).toMatchObject({ amountCents: 500, method: 'stripe', pendingCredential: null });
  });

  it('the first request answered with a redirect: not followed either, so not even the signed hints leave the host', async () => {
    platformBehaviour = (_req, res) => redirectTo(res, 302);
    const b = makeBuyer(platform.host);
    await expect(b.buyer.payingFetch(`${platform.origin}/agents/acme/chat/completions`, { method: 'POST', body: '{}' })).rejects.toBeInstanceOf(MppRedirectError);
    expect(platform.heard).toHaveLength(1);
    expect(elsewhere.heard).toEqual([]);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['redirect_refused']);
  });

  it('an in-band purchase answered with a redirect is an outcome, not a followed request', async () => {
    platformBehaviour = (_req, res) => redirectTo(res, 308);
    const b = makeBuyer(platform.host);
    const outcome = await b.buyer.purchase({ url: `${platform.origin}/api/mpp/credits`, challenge: challengeHeader(platform.host), terms: { version: TERMS } });
    expect(outcome).toMatchObject({ ok: false, reason: 'redirect_refused', status: null, response: null, pendingCredential: null });
    expect(platform.heard).toHaveLength(1);
    expect(platform.heard[0].headers['payment-authorization']).toBeTruthy();
    expect(elsewhere.heard).toEqual([]);
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.refusals.map((r) => r.reason)).toEqual(['redirect_refused']);
  });
});

describe('S-180: every request the buyer hands to fetch refuses redirects', () => {
  const RESOURCE = 'https://robutler.ai/agents/acme/v1/chat/completions';

  function problem402(): Response {
    return new Response('{}', { status: 402, headers: { 'WWW-Authenticate': challengeHeader('robutler.ai'), [TERMS_VERSION_HEADER]: TERMS } });
  }

  it('the seed, the paid retry and the retry without a credential are all redirect: "error", whatever the caller passed', async () => {
    const modes: string[] = [];
    const answers = [
      problem402(),
      // Granted, and the door asks for the call again WITHOUT the credential.
      new Response(JSON.stringify({ error: 'funding_failed', retry: { sameCredential: false } }), { status: 503, headers: { 'Retry-After': '0', 'Content-Type': 'application/problem+json' } }),
      new Response('{"ok":true}', { status: 200 }),
    ];
    const fetch = (async (input: Request) => {
      modes.push(input.redirect);
      return answers.shift()!;
    }) as unknown as typeof globalThis.fetch;
    const b = makeBuyer('robutler.ai', { fetch });
    for (const redirect of ['follow', 'manual'] as const) {
      answers.splice(0, answers.length, problem402(), new Response(JSON.stringify({ error: 'funding_failed', retry: { sameCredential: false } }), { status: 503, headers: { 'Retry-After': '0', 'Content-Type': 'application/problem+json' } }), new Response('{"ok":true}', { status: 200 }));
      modes.length = 0;
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}', redirect })).status).toBe(200);
      expect(modes).toEqual(['error', 'error', 'error']);
    }
  });

  it('a fetch that hands back the 3xx itself (no redirect modes) is the same typed refusal', async () => {
    const answers = [problem402(), new Response(null, { status: 307, headers: { Location: 'https://elsewhere.example/collect' } })];
    const fetch = (async () => answers.shift()!) as unknown as typeof globalThis.fetch;
    const b = makeBuyer('robutler.ai', { fetch });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toMatchObject({
      name: 'MppRedirectError',
      status: 307,
      location: 'https://elsewhere.example/collect',
      followed: false,
    });
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    expect(b.refusals.map((r) => r.reason)).toEqual(['redirect_refused']);
  });

  it('a fetch that FOLLOWED anyway is never read as a settled purchase: the credential is held, its outcome unknown', async () => {
    const followed = new Response('{"ok":true}', { status: 200 });
    Object.defineProperty(followed, 'redirected', { value: true });
    const answers = [problem402(), followed];
    const fetch = (async () => answers.shift()!) as unknown as typeof globalThis.fetch;
    const b = makeBuyer('robutler.ai', { fetch });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toMatchObject({ name: 'MppRedirectError', followed: true });
    expect(b.purchases).toEqual([]);
    // It left this process for a host nobody named, so it stays counted and held (S-150).
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_outcome_unknown']);
  });

  it('a redirect answered to a credential ALREADY in doubt keeps it held: a redirect says nothing about whether it settled', async () => {
    const pending = () =>
      new Response(JSON.stringify({ error: 'charge_outcome_unknown', retry: { sameCredential: true } }), { status: 503, headers: { 'Retry-After': '1', 'Content-Type': 'application/problem+json' } });
    const answers = [problem402(), pending(), new Response(null, { status: 302, headers: { Location: 'https://elsewhere.example/' } })];
    const fetch = (async () => answers.shift()!) as unknown as typeof globalThis.fetch;
    const b = makeBuyer('robutler.ai', { fetch });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toBeInstanceOf(MppRedirectError);
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['redirect_refused']);
    expect(b.refusals[0].pendingCredential).not.toBeNull();
  });
});
