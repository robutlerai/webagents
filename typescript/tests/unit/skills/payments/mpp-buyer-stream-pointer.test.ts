/**
 * Two 2026-09-19 changes to the MPP buyer, each pinned from the outside.
 *
 * A STREAMED ANSWER IS NEVER READ (finding sdk-1). After a paid retry the
 * buyer did `response.clone().json()` on every answer, so a streamed
 * completion was teed and read to its END before `payingFetch` resolved: the
 * caller saw its first chunk only after the last one was sent, and the whole
 * stream sat in memory. The first half of this file is not a stub: a real
 * HTTP server writes one chunk, then WAITS for the test to say so before it
 * writes the last, and the assertion is that the caller already holds the
 * first chunk while the server is still waiting.
 *
 * THE PURCHASE POINTER. Both LLM rails now answer a token holder whose token
 * ran dry with an `mpp` requirement that names `purchase_url` and carries NO
 * challenge (the portal's lib/payments/purchase-pointer.ts: nobody on those
 * rails was verified, so nothing could be minted for them). Both SDKs ignored
 * it. The second half pins what following it means: the buyer's own signed
 * request to the purchase URL, the challenge it is answered with paid under
 * the unchanged policy (realm allowlist S-147, seller pin S-154, the caps),
 * and the original call sent once more without its payment token.
 *
 * THE CEILING (the same day). A pointer carries no secret, so a peer behind an
 * allowlisted host can write one, call after call. Only `dailyCapCents` bounds
 * the total, so a buyer with none follows no pointer (`pointer_needs_daily_cap`)
 * and still pays a challenge on a URL its caller chose. Every test that expects
 * a pointer to be followed builds its buyer with `pointerBuyer`, which says so.
 *
 * The purchase 200 fixture is the body the portal sends from 2026-09-19: no
 * `credits` and no `creditsNano` (the granted amount is no longer stated).
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { generateKeyPairSync, sign as cryptoSign, createHash } from 'node:crypto';
import type { SigningIdentity } from '../../../../src/crypto/http-signature';
import {
  BUYER_BODY_MAX_BYTES,
  MppBuyer,
  MPP_CREDENTIAL_HEADER,
  MPP_RECEIPT_HEADER,
  TERMS_ACCEPTED_HEADER,
  TERMS_VERSION_HEADER,
  base64urlEncode,
  buyerReadsBody,
  isJsonMediaType,
  jcsCanonicalize,
  mppRequirementOf,
  type MppBuyerConfig,
  type MppBuyerRefusal,
  type MppPurchaseRecord,
} from '../../../../src/skills/payments/mpp-buyer';

const AGENT_URL = 'https://agent.example/agents/mini';
const TERMS = '2026-07-31';
const TERMS_URL = 'https://robutler.ai/doc/terms-of-service';

function identity(): SigningIdentity {
  const { privateKey } = generateKeyPairSync('ed25519');
  return {
    issuer: AGENT_URL,
    getHeldKeys: () => [{ kid: 'k'.repeat(43), sign: async (data: Uint8Array) => new Uint8Array(cryptoSign(null, Buffer.from(data), privateKey)) }],
  };
}

let counter = 0;
function challengeHeader(realm: string, opts: { cents?: number; networkId?: string } = {}): string {
  counter += 1;
  const fields: Array<[string, string]> = [
    ['id', createHash('sha256').update(`stream pointer challenge ${counter}`).digest('base64url')],
    ['realm', realm],
    ['method', 'stripe'],
    ['intent', 'charge'],
    ['expires', new Date(Date.now() + 300_000).toISOString()],
    ['request', base64urlEncode(jcsCanonicalize({ amount: String(opts.cents ?? 500), currency: 'usd', methodDetails: { networkId: opts.networkId ?? 'profile_test', paymentMethodTypes: ['card'] } }))],
    ['opaque', base64urlEncode(jcsCanonicalize({ packId: 'mpp_5', kind: 'pack', terms: TERMS }))],
    ['header', MPP_CREDENTIAL_HEADER],
  ];
  return `Payment ${fields.map(([k, v]) => `${k}="${v}"`).join(', ')}`;
}

function receiptHeader(reference = 'pi_test_1'): string {
  return base64urlEncode(jcsCanonicalize({ method: 'stripe', reference, status: 'success', timestamp: '2026-09-19T12:00:00.000Z' }));
}

function makeBuyer(realms: string[], overrides: Partial<MppBuyerConfig> & { policy?: Partial<MppBuyerConfig['policy']> } = {}) {
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
    policy: { maxPerPurchaseCents: 2000, acceptTerms: TERMS, realms, stripeProfileId: 'profile_test', ...overrides.policy },
  });
  return { buyer, purchases, refusals, getSpt };
}

// ---------------------------------------------------------------------------
// sdk-1: a streamed answer reaches the caller as it arrives
// ---------------------------------------------------------------------------

/** A server whose paid answer writes `first`, then waits for `release()` before it writes `last` and ends. */
async function slowServer(answerHeaders: http.OutgoingHttpHeaders, first: string, last: string) {
  let release!: () => void;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  const state = { lastSent: false, paidRequests: 0 };
  let host = '';
  const server = http.createServer((req, res) => {
    req.resume();
    req.on('end', () => {
      if (!req.headers[MPP_CREDENTIAL_HEADER.toLowerCase()]) {
        res.writeHead(402, { 'WWW-Authenticate': challengeHeader(host), 'Content-Type': 'application/problem+json', [TERMS_VERSION_HEADER]: TERMS });
        res.end(JSON.stringify({ status: 402, terms: { url: TERMS_URL, version: TERMS } }));
        return;
      }
      state.paidRequests += 1;
      // No Content-Length: Node frames this as `Transfer-Encoding: chunked`.
      res.writeHead(200, answerHeaders);
      res.write(first);
      void released.then(() => {
        state.lastSent = true;
        res.end(last);
      });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  host = `127.0.0.1:${(server.address() as AddressInfo).port}`;
  return { server, host, origin: `http://${host}`, state, release };
}

/** Resolves to `value`, or rejects after `ms`: how "it had already happened" is asserted without waiting on the thing under test. */
function within<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
  return Promise.race([promise, new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`${what} did not happen within ${ms} ms`)), ms))]);
}

describe('sdk-1: after a paid retry a streamed 2xx is handed to the caller untouched (real server, real fetch)', () => {
  let open: http.Server | null = null;
  afterEach(async () => {
    if (open) {
      open.closeAllConnections();
      await new Promise((resolve) => open!.close(resolve));
    }
    open = null;
  });

  for (const [name, headers, first, last] of [
    ['an event stream', { 'Content-Type': 'text/event-stream', [MPP_RECEIPT_HEADER]: receiptHeader('pi_stream') }, 'data: {"n":1}\n\n', 'data: [DONE]\n\n'],
    // A JSON content type is not a reason to read a 2xx either: with no declared length it is a stream somebody is waiting on.
    ['a chunked JSON answer', { 'Content-Type': 'application/json', [MPP_RECEIPT_HEADER]: receiptHeader('pi_stream') }, '{"choices":[', ']}'],
  ] as const) {
    it(`${name}: the first chunk reaches the caller before the last is sent`, async () => {
      const s = await slowServer(headers, first, last);
      open = s.server;
      const b = makeBuyer([s.host]);

      // The defect: this promise did not resolve until the server had sent its LAST chunk.
      const response = await within(b.buyer.payingFetch(`${s.origin}/agents/acme/v1/chat/completions`, { method: 'POST', body: '{"stream":true}' }), 2000, 'payingFetch resolving');
      expect(response.status).toBe(200);
      expect(s.state.paidRequests).toBe(1);

      const reader = response.body!.getReader();
      const chunk = await within(reader.read(), 2000, 'the first chunk');
      expect(new TextDecoder().decode(chunk.value)).toBe(first);
      // The whole point: the caller holds the first chunk and the server has not sent the last.
      expect(s.state.lastSent).toBe(false);

      // "Was this paid" was decided from the head: the status, and the receipt header for the record.
      expect(b.purchases).toHaveLength(1);
      expect(b.purchases[0]).toMatchObject({ status: 200, amountCents: 500, receipt: { reference: 'pi_stream', method: 'stripe' } });

      s.release();
      let rest = '';
      for (;;) {
        const { done, value } = await within(reader.read(), 2000, 'the rest of the stream');
        if (done) break;
        rest += new TextDecoder().decode(value);
      }
      expect(rest).toBe(last);
      expect(s.state.lastSent).toBe(true);
    });
  }
});

describe('sdk-1: which bodies the buyer reads for itself', () => {
  const head = (status: number, headers: Record<string, string>, body: string | null = '{}') => new Response(body, { status, headers });

  it('decides from the head: a JSON type, a bounded declared length, and on a 2xx a declared one', () => {
    expect(isJsonMediaType('application/json; charset=utf-8')).toBe(true);
    expect(isJsonMediaType('application/problem+json')).toBe(true);
    expect(isJsonMediaType('text/event-stream')).toBe(false);
    expect(isJsonMediaType('application/jsonl')).toBe(false);
    expect(isJsonMediaType(null)).toBe(false);

    // The platform's own error answers are chunked: no declared length is needed off a 2xx.
    expect(buyerReadsBody(head(402, { 'Content-Type': 'application/problem+json' }))).toBe(true);
    expect(buyerReadsBody(head(503, { 'Content-Type': 'application/json' }))).toBe(true);
    expect(buyerReadsBody(head(503, { 'Content-Type': 'text/html' }))).toBe(false);
    expect(buyerReadsBody(head(402, { 'Content-Type': 'application/json', 'Content-Length': String(BUYER_BODY_MAX_BYTES + 1) }))).toBe(false);

    expect(buyerReadsBody(head(200, { 'Content-Type': 'application/json' }))).toBe(false);
    expect(buyerReadsBody(head(200, { 'Content-Type': 'application/json', 'Content-Length': '11' }))).toBe(true);
    expect(buyerReadsBody(head(200, { 'Content-Type': 'text/event-stream', 'Content-Length': '11' }))).toBe(false);
    // The answer of a purchase URL the buyer itself called is a purchase document whatever its framing.
    expect(buyerReadsBody(head(200, { 'Content-Type': 'application/json' }), { purchaseDocument: true })).toBe(true);
    expect(buyerReadsBody(head(200, { 'Content-Type': 'text/plain' }), { purchaseDocument: true })).toBe(false);
    expect(buyerReadsBody(head(204, { 'Content-Type': 'application/json' }, null))).toBe(false);
  });

  it('an error body past the cap is unreadable to the buyer and whole for the caller: nothing past the cap is buffered', async () => {
    const huge = JSON.stringify({ retry: { sameCredential: true }, pad: 'x'.repeat(BUYER_BODY_MAX_BYTES * 2) });
    const answers = [
      new Response('{}', { status: 402, headers: { 'WWW-Authenticate': challengeHeader('robutler.ai'), 'Content-Type': 'application/problem+json', [TERMS_VERSION_HEADER]: TERMS } }),
      new Response(huge, { status: 503, headers: { 'Content-Type': 'application/problem+json' } }),
    ];
    const b = makeBuyer(['robutler.ai'], { fetch: (async () => answers.shift()!) as unknown as typeof fetch });
    const response = await b.buyer.payingFetch('https://robutler.ai/agents/acme/v1/chat/completions', { method: 'POST', body: '{}' });
    expect(response.status).toBe(503);
    // `sameCredential: true` was past the cap, so this is a 5xx that says nothing: held, never re-presented blind.
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_outcome_unknown']);
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(await response.text()).toBe(huge);
  });
});

// ---------------------------------------------------------------------------
// The purchase pointer
// ---------------------------------------------------------------------------

const RESOURCE = 'https://robutler.ai/api/llm/chat/completions';
const PURCHASE_URL = 'https://robutler.ai/api/mpp/credits';

/** The HTTP twin's 402 to a token holder: the token scheme first, then the pointer; no `WWW-Authenticate: Payment`. */
function pointer402(purchaseUrl = PURCHASE_URL): Response {
  return new Response(
    JSON.stringify({
      error: 'Budget exhausted',
      details: 'Insufficient token balance',
      requirements: { amount: '0.142857', currency: 'USD', schemes: [{ scheme: 'token' }, { scheme: 'mpp', purchase_url: purchaseUrl }], reason: 'Insufficient token balance' },
    }),
    { status: 402, headers: { 'Content-Type': 'application/json' } },
  );
}

/** What the purchase URL answers a signed request that carries no credential (the portal's `handlePurchase`). */
function challenge402(opts: { networkId?: string; cents?: number; realm?: string } = {}): Response {
  return new Response(JSON.stringify({ type: 'https://paymentauth.org/problems/payment-required', status: 402, terms: { url: TERMS_URL, version: TERMS } }), {
    status: 402,
    headers: {
      'WWW-Authenticate': challengeHeader(opts.realm ?? 'robutler.ai', opts),
      'Content-Type': 'application/problem+json',
      [TERMS_VERSION_HEADER]: TERMS,
      Link: `<${TERMS_URL}>; rel="terms-of-service"`,
    },
  });
}

/** The purchase 200 as the portal sends it from 2026-09-19: `credits` and `creditsNano` are gone. */
const PURCHASE_200_BODY = {
  ok: true,
  packId: 'mpp_5',
  granted: true,
  balance: { nano: '5000000000' },
  paymentIntentId: 'pi_pointer',
  receipt: { status: 'success', method: 'stripe', timestamp: '2026-09-19T12:00:00.000Z', reference: 'pi_pointer' },
};
function purchased200(): Response {
  return new Response(JSON.stringify(PURCHASE_200_BODY), {
    status: 200,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', [MPP_RECEIPT_HEADER]: receiptHeader('pi_pointer') },
  });
}

interface Seen {
  method: string;
  url: string;
  headers: Headers;
  body: string;
  covered: string[];
}

function fetchQueue(...responders: Array<Response | ((seen: Seen) => Response)>) {
  const seen: Seen[] = [];
  const fetch = vi.fn(async (input: Request | string | URL) => {
    const request = input instanceof Request ? input : new Request(input);
    const body = request.body ? await request.text() : '';
    const signatureInput = request.headers.get('signature-input') ?? '';
    const inner = signatureInput.includes('(') ? signatureInput.slice(signatureInput.indexOf('(') + 1, signatureInput.indexOf(')')) : '';
    const record: Seen = { method: request.method, url: request.url, headers: request.headers, body, covered: inner ? inner.split(' ').map((c) => c.replace(/"/g, '')) : [] };
    seen.push(record);
    const next = responders.shift();
    if (!next) throw new Error(`fetch stub exhausted after ${seen.length} requests`);
    return typeof next === 'function' ? next(record) : next;
  });
  return { fetch: fetch as unknown as typeof globalThis.fetch, seen };
}

const served200 = () => new Response('data: {"ok":true}\n\n', { status: 200, headers: { 'Content-Type': 'text/event-stream' } });

/**
 * A buyer that FOLLOWS pointers: what makes it one is a configured daily cap,
 * and nothing else. `makeBuyer` sets none, which is the buyer the ceiling
 * tests below are about.
 */
const DAILY_CAP_CENTS = 5000;
function pointerBuyer(realms: string[], overrides: Parameters<typeof makeBuyer>[1] = {}) {
  return makeBuyer(realms, { ...overrides, policy: { dailyCapCents: DAILY_CAP_CENTS, ...overrides.policy } });
}
const tokenCall = { method: 'POST', body: '{"messages":[]}', headers: { 'X-Payment-Token': 'jwt.dry.token', 'X-Chat-Id': 'chat-1', 'Content-Type': 'application/json' } };

describe('the purchase pointer through payingFetch', () => {
  it('asks the purchase URL as itself, pays the challenge it is answered with, and sends the call again without its payment token', async () => {
    const q = fetchQueue(pointer402(), challenge402(), purchased200(), served200());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });

    const response = await b.buyer.payingFetch(RESOURCE, tokenCall);
    expect(response.status).toBe(200);
    expect(await response.text()).toBe('data: {"ok":true}\n\n');

    expect(q.seen.map((s) => `${s.method} ${s.url}`)).toEqual([`POST ${RESOURCE}`, `POST ${PURCHASE_URL}`, `POST ${PURCHASE_URL}`, `POST ${RESOURCE}`]);
    // Every one of them is the buyer's own signed request.
    for (const s of q.seen) expect(s.headers.get('signature-input')).toContain('tag="web-bot-auth"');
    // The ask carries no credential and no body; the payment carries the credential and the assent, both covered.
    expect(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)).toBeNull();
    expect(q.seen[1].body).toBe('');
    expect(q.seen[2].headers.get(MPP_CREDENTIAL_HEADER)).toMatch(/^Payment /);
    expect(q.seen[2].headers.get(TERMS_ACCEPTED_HEADER)).toBe(TERMS);
    expect(q.seen[2].covered).toEqual(expect.arrayContaining([MPP_CREDENTIAL_HEADER.toLowerCase(), TERMS_ACCEPTED_HEADER.toLowerCase()]));
    // The caller's token never goes to the purchase URL, and the credential never goes to the resource.
    expect(q.seen[1].headers.get('x-payment-token')).toBeNull();
    expect(q.seen[2].headers.get('x-payment-token')).toBeNull();
    // The original call as it was, then the same call minus the token that ran dry: the
    // door serves the bought balance only to a signed request that names no token.
    expect(q.seen[0].headers.get('x-payment-token')).toBe('jwt.dry.token');
    expect(q.seen[3].headers.get('x-payment-token')).toBeNull();
    expect(q.seen[3].headers.get('x-chat-id')).toBe('chat-1');
    expect(q.seen[3].headers.get(MPP_CREDENTIAL_HEADER)).toBeNull();
    expect(q.seen[3].body).toBe('{"messages":[]}');

    // One purchase, at the purchase URL, counted like any other.
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.purchases).toHaveLength(1);
    expect(b.purchases[0]).toMatchObject({ url: PURCHASE_URL, amountCents: 500, status: 200, termsVersion: TERMS, receipt: { reference: 'pi_pointer' } });
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.refusals).toEqual([]);
  });

  it('a token in ?payment_token= or X-PAYMENT is dropped from the re-send too', async () => {
    const q = fetchQueue(pointer402(), challenge402(), purchased200(), served200());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    await b.buyer.payingFetch(`${RESOURCE}?payment_token=jwt&model=auto`, { method: 'POST', body: '{}', headers: { 'X-PAYMENT': 'x402-token' } });
    expect(q.seen[0].url).toBe(`${RESOURCE}?payment_token=jwt&model=auto`);
    expect(q.seen[3].url).toBe(`${RESOURCE}?model=auto`);
    expect(q.seen[3].headers.get('x-payment')).toBeNull();
  });

  it('S-147: a pointer named by a host off the allowlist is not followed, and nothing is sent anywhere', async () => {
    const q = fetchQueue(pointer402());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    const response = await b.buyer.payingFetch('https://delegate.example/chat/completions', tokenCall);
    expect(response.status).toBe(402);
    expect(await response.json()).toMatchObject({ error: 'Budget exhausted' });
    expect(q.seen).toHaveLength(1);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
    expect(b.buyer.spentTodayCents()).toBe(0);
  });

  it('S-147: a pointer TO a host off the allowlist is not followed: the signed request never leaves', async () => {
    const q = fetchQueue(pointer402('https://collector.example/api/mpp/credits'));
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
    expect(q.seen).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
  });

  it('S-147: the challenge the purchase URL answers with must name that host as its realm', async () => {
    const q = fetchQueue(pointer402(), challenge402({ realm: 'other.example' }));
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
    expect(q.seen).toHaveLength(2);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
  });

  it('S-154: a challenge at the purchase URL that names another seller is never paid', async () => {
    const q = fetchQueue(pointer402(), challenge402({ networkId: 'profile_attacker' }));
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, tokenCall);
    // The caller gets the 402 it was answered, not the purchase URL's.
    expect(await response.json()).toMatchObject({ error: 'Budget exhausted' });
    expect(q.seen).toHaveLength(2);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(b.buyer.spentTodayCents()).toBe(0);
  });

  it('the caps hold: over the per-purchase maximum or the daily cap nothing is paid', async () => {
    const over = pointerBuyer(['robutler.ai'], { fetch: fetchQueue(pointer402(), challenge402({ cents: 2500 })).fetch });
    expect((await over.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
    expect(over.refusals.map((r) => r.reason)).toEqual(['over_max_per_purchase']);

    const q = fetchQueue(pointer402(), challenge402(), purchased200(), served200(), pointer402(), challenge402());
    const capped = makeBuyer(['robutler.ai'], { fetch: q.fetch, policy: { dailyCapCents: 800 } });
    expect((await capped.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(200);
    expect((await capped.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
    expect(capped.refusals.map((r) => r.reason)).toEqual(['over_daily_cap']);
    expect(capped.getSpt).toHaveBeenCalledTimes(1);
    expect(capped.buyer.spentTodayCents()).toBe(500);
  });

  it('maxPurchasesPerCall: a second pointer on the same call buys nothing more, and is not even asked about', async () => {
    const q = fetchQueue(pointer402(), challenge402(), purchased200(), pointer402());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, tokenCall);
    expect(response.status).toBe(402);
    expect(q.seen).toHaveLength(4);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.purchases).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['purchase_limit_per_call']);
  });

  it('a 402 that names no pointer is returned as it came, as before', async () => {
    const plain = [
      new Response('{"error":"Budget exhausted"}', { status: 402, headers: { 'Content-Type': 'application/json' } }),
      new Response('Payment Required', { status: 402, headers: { 'Content-Type': 'text/plain' } }),
      new Response(JSON.stringify({ requirements: { schemes: [{ scheme: 'token' }, { scheme: 'mpp' }] } }), { status: 402, headers: { 'Content-Type': 'application/json' } }),
      new Response(JSON.stringify({ requirements: { schemes: [{ scheme: 'mpp', purchase_url: PURCHASE_URL, challenge: '  ' }] } }), { status: 402, headers: { 'Content-Type': 'application/json' } }),
    ];
    for (const answer of plain) {
      const q = fetchQueue(answer);
      const b = makeBuyer(['robutler.ai'], { fetch: q.fetch });
      expect((await b.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
      expect(q.seen).toHaveLength(1);
      expect(b.refusals).toEqual([]);
    }
  });
});

describe('purchaseAt: the pointer of a UAMP payment.required', () => {
  const FROM = 'wss://robutler.ai/llm';

  it('buys at the purchase URL and reports the purchase; nothing in the record depends on a granted amount', async () => {
    const q = fetchQueue(challenge402(), purchased200());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    const outcome = await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM });
    expect(outcome.ok).toBe(true);
    if (!outcome.ok) throw new Error('unreachable');
    expect(outcome.record).toEqual({
      url: PURCHASE_URL,
      challengeId: expect.any(String),
      method: 'stripe',
      amountCents: 500,
      currency: 'usd',
      termsVersion: TERMS,
      receipt: { status: 'success', method: 'stripe', timestamp: '2026-09-19T12:00:00.000Z', reference: 'pi_pointer' },
      status: 200,
    });
    // The body is the caller's to read, and it is the portal's: no `credits`, no `creditsNano`.
    const body = (await outcome.response.json()) as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(['balance', 'granted', 'ok', 'packId', 'paymentIntentId', 'receipt']);
    expect(q.seen.map((s) => `${s.method} ${s.url}`)).toEqual([`POST ${PURCHASE_URL}`, `POST ${PURCHASE_URL}`]);
  });

  it('S-147: nothing is sent unless the purchase URL AND the socket that named it are on the allowlist', async () => {
    for (const request of [
      { url: PURCHASE_URL, from: 'wss://delegate.example/agents/x/uamp' },
      { url: 'https://collector.example/api/mpp/credits', from: FROM },
      { url: PURCHASE_URL, from: 'not a url' },
    ]) {
      const q = fetchQueue();
      const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
      expect(await b.buyer.purchaseAt(request)).toMatchObject({ ok: false, reason: 'realm_not_allowed', status: null, response: null });
      expect(q.seen).toEqual([]);
      expect(b.getSpt).not.toHaveBeenCalled();
      expect(b.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
    }
  });

  it('maxPurchasesPerCall holds across the separate purchases of one call, and a new call starts at zero', async () => {
    const q = fetchQueue(challenge402(), purchased200(), challenge402(), purchased200());
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    const call = {};
    expect((await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call })).ok).toBe(true);
    // The same call again: refused before anything is sent.
    expect(await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call })).toMatchObject({ ok: false, reason: 'purchase_limit_per_call' });
    expect(q.seen).toHaveLength(2);
    // An in-band challenge for the same call is a second purchase too.
    expect(await b.buyer.purchase({ url: PURCHASE_URL, challenge: challengeHeader('robutler.ai'), terms: { version: TERMS }, call })).toMatchObject({ ok: false, reason: 'purchase_limit_per_call' });
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect((await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call: {} })).ok).toBe(true);
    expect(b.getSpt).toHaveBeenCalledTimes(2);
  });

  it('a purchase URL that answers without a challenge bought nothing', async () => {
    const q = fetchQueue(new Response('{"ok":true}', { status: 200, headers: { 'Content-Type': 'application/json' } }));
    const b = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    expect(await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM })).toMatchObject({ ok: false, reason: 'served_without_purchase' });
    expect(b.getSpt).not.toHaveBeenCalled();
  });
});

describe('the ceiling: a pointer is followed only under a daily cap', () => {
  const FROM = 'wss://robutler.ai/llm';

  /** A 402 BODY that names a purchase URL and carries a challenge with it: no platform rail sends this over HTTP, and any peer can. */
  function challengeEntry402(): Response {
    return new Response(
      JSON.stringify({
        error: 'Budget exhausted',
        requirements: { schemes: [{ scheme: 'mpp', challenge: challengeHeader('robutler.ai'), purchase_url: PURCHASE_URL, terms: { url: TERMS_URL, version: TERMS } }] },
      }),
      { status: 402, headers: { 'Content-Type': 'application/json' } },
    );
  }

  it('payingFetch: with no daily cap the pointer is refused before anything is sent, and the caller gets its 402 as it came', async () => {
    const q = fetchQueue(pointer402());
    const b = makeBuyer(['robutler.ai'], { fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, tokenCall);
    expect(response.status).toBe(402);
    expect(await response.json()).toMatchObject({ error: 'Budget exhausted' });
    // The call itself, and nothing else: the purchase URL was never asked.
    expect(q.seen.map((s) => s.url)).toEqual([RESOURCE]);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.refusals).toHaveLength(1);
    expect(b.refusals[0]).toMatchObject({ reason: 'pointer_needs_daily_cap', url: RESOURCE, challengeId: null, pendingCredential: null });
    // The operator is told which setting, and why.
    expect(b.refusals[0].detail).toContain('policy.dailyCapCents');
    expect(b.refusals[0].detail).toContain('Nothing was sent');
  });

  it('purchaseAt: with no daily cap the pointer is refused with the typed reason and nothing is sent', async () => {
    const q = fetchQueue();
    const b = makeBuyer(['robutler.ai'], { fetch: q.fetch });
    expect(await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM })).toMatchObject({ ok: false, reason: 'pointer_needs_daily_cap', status: null, response: null });
    expect(q.seen).toEqual([]);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['pointer_needs_daily_cap']);
  });

  it('an off-list pointer is refused for its host whether or not a cap is set: the allowlist is asked first', async () => {
    for (const build of [makeBuyer, pointerBuyer]) {
      const b = build(['robutler.ai'], { fetch: fetchQueue().fetch });
      expect(await b.buyer.purchaseAt({ url: 'https://collector.example/api/mpp/credits', from: FROM })).toMatchObject({ ok: false, reason: 'realm_not_allowed' });
    }
  });

  it('a purchase URL named in a 402 BODY is held to the same rule when the entry carries a challenge', async () => {
    const none = fetchQueue(challengeEntry402());
    const uncapped = makeBuyer(['robutler.ai'], { fetch: none.fetch });
    expect((await uncapped.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(402);
    expect(none.seen.map((s) => s.url)).toEqual([RESOURCE]);
    expect(uncapped.getSpt).not.toHaveBeenCalled();
    expect(uncapped.refusals.map((r) => r.reason)).toEqual(['pointer_needs_daily_cap']);

    // Under a cap it is paid as `purchase` pays one: the challenge is in hand, so the purchase URL is asked once, with the credential.
    const q = fetchQueue(challengeEntry402(), purchased200(), served200());
    const capped = pointerBuyer(['robutler.ai'], { fetch: q.fetch });
    expect((await capped.buyer.payingFetch(RESOURCE, tokenCall)).status).toBe(200);
    expect(q.seen.map((s) => s.url)).toEqual([RESOURCE, PURCHASE_URL, RESOURCE]);
    expect(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)).toMatch(/^Payment /);
    expect(capped.buyer.spentTodayCents()).toBe(500);
  });

  it('with no daily cap a challenge on the URL the caller chose, and an in-band purchase, are paid exactly as before', async () => {
    const q = fetchQueue(challenge402(), served200());
    const b = makeBuyer(['robutler.ai'], { fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"messages":[]}' });
    expect(response.status).toBe(200);
    expect(q.seen.map((s) => s.url)).toEqual([RESOURCE, RESOURCE]);
    expect(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)).toMatch(/^Payment /);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.refusals).toEqual([]);

    const inBand = makeBuyer(['robutler.ai'], { fetch: fetchQueue(purchased200()).fetch });
    expect((await inBand.buyer.purchase({ url: PURCHASE_URL, challenge: challengeHeader('robutler.ai'), terms: { version: TERMS } })).ok).toBe(true);
    expect(inBand.refusals).toEqual([]);
  });

  it('the cap that admits a pointer is what bounds it: call after call, the total stops at the cap', async () => {
    // A peer starts a new call each time, so `maxPurchasesPerCall` never bites. 500 cents a pack under a 1200 cent cap: two, never three.
    const q = fetchQueue(challenge402(), purchased200(), challenge402(), purchased200(), challenge402());
    const b = makeBuyer(['robutler.ai'], { fetch: q.fetch, policy: { dailyCapCents: 1200 } });
    expect((await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call: {} })).ok).toBe(true);
    expect((await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call: {} })).ok).toBe(true);
    expect(await b.buyer.purchaseAt({ url: PURCHASE_URL, from: FROM, call: {} })).toMatchObject({ ok: false, reason: 'over_daily_cap' });
    expect(b.getSpt).toHaveBeenCalledTimes(2);
    expect(b.buyer.spentTodayCents()).toBe(1000);
  });
});

describe('mppRequirementOf', () => {
  it('reads the pointer, the entry with a challenge, and nothing else', () => {
    expect(mppRequirementOf({ schemes: [{ scheme: 'token' }, { scheme: 'mpp', purchase_url: PURCHASE_URL }] })).toEqual({ purchaseUrl: PURCHASE_URL, challenge: null, terms: null });
    expect(mppRequirementOf({ schemes: [{ scheme: 'mpp', challenge: 'Payment id="c1"', purchase_url: PURCHASE_URL, terms: { url: TERMS_URL, version: TERMS } }] })).toEqual({
      purchaseUrl: PURCHASE_URL,
      challenge: 'Payment id="c1"',
      terms: { url: TERMS_URL, version: TERMS },
    });
    for (const nothing of [null, {}, { schemes: 'mpp' }, { schemes: [{ scheme: 'mpp' }] }, { schemes: [{ scheme: 'mpp', purchase_url: ' ' }] }, { schemes: [{ scheme: 'mpp', purchase_url: PURCHASE_URL, challenge: 7 }] }]) {
      expect(mppRequirementOf(nothing)).toBeNull();
    }
  });
});
