/**
 * The MPP buyer (machine-purchase design sections 2.1, 6.3 and 6.4; pass
 * P9a, 2026-09-18). Every network exchange here is a stubbed `fetch` that
 * records the signed `Request` it was handed, so the suite reads what the
 * platform would read: which headers rode the retry, which of them the
 * signature covered, and whether the body was the same bytes.
 *
 * The signing identity is a fresh Ed25519 key behind the `SigningIdentity`
 * interface (no jose, no AgentIdentity): the signer's own suite pins the
 * bytes, this one pins the loop.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { generateKeyPairSync, sign as cryptoSign, createHash } from 'node:crypto';
import type { SigningIdentity } from '../../../../src/crypto/http-signature';
import {
  MppBuyer,
  MPP_CREDENTIAL_HEADER,
  MPP_RECEIPT_HEADER,
  PAYMENT_METHODS_HINT_HEADER,
  PURCHASE_HINT_HEADER,
  TERMS_ACCEPTED_HEADER,
  TERMS_VERSION_HEADER,
  base64urlDecode,
  base64urlEncode,
  challengeAmountCents,
  decodeChallengeRequest,
  encodeMppCredential,
  jcsCanonicalize,
  parsePaymentReceipt,
  parseWwwAuthenticatePayment,
  readMppChallenge,
  readTermsNotice,
  retryAfterSeconds,
  tempoCredentialPayload,
  tempoPayerDid,
  afterCredential,
  sellerPinsFromDiscovery,
  credentialHeaderRefusal,
  CREDENTIAL_HEADERS_REFUSED,
  type MppBuyerConfig,
  type MppBuyerRefusal,
  type MppBuyerState,
  type MppPurchaseOutcome,
  type MppChallengeFields,
  type MppPurchaseRecord,
} from '../../../../src/skills/payments/mpp-buyer';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const AGENT_URL = 'https://agent.example/agents/mini';
const RESOURCE = 'https://robutler.ai/agents/acme/v1/chat/completions';
const PURCHASE_URL = 'https://robutler.ai/api/mpp/credits';
const TERMS = '2026-07-31';
const TERMS_URL = 'https://robutler.ai/doc/terms-of-service';
const NOW = new Date('2026-09-18T12:00:00.000Z');

function identity(): SigningIdentity {
  const { privateKey } = generateKeyPairSync('ed25519');
  return {
    issuer: AGENT_URL,
    getHeldKeys: () => [
      {
        kid: 'k'.repeat(43),
        sign: async (data: Uint8Array) => new Uint8Array(cryptoSign(null, Buffer.from(data), privateKey)),
      },
    ],
  };
}

const decode = (b64: string) => JSON.parse(new TextDecoder().decode(base64urlDecode(b64)!)) as Record<string, unknown>;

function formatChallenge(f: MppChallengeFields): string {
  const params: Array<[string, string]> = [
    ['id', f.id],
    ['realm', f.realm],
    ['method', f.method],
    ['intent', f.intent],
  ];
  if (f.expires) params.push(['expires', f.expires]);
  params.push(['request', f.request]);
  if (f.digest) params.push(['digest', f.digest]);
  if (f.opaque) params.push(['opaque', f.opaque]);
  if (f.header) params.push(['header', f.header]);
  return `Payment ${params.map(([k, v]) => `${k}="${v.replace(/(["\\])/g, '\\$1')}"`).join(', ')}`;
}

let counter = 0;
function challenge(opts: {
  amountCents?: number;
  method?: 'stripe' | 'tempo';
  realm?: string;
  intent?: string;
  expires?: string | null;
  header?: string | null;
  kind?: string;
  networkId?: string;
  recipient?: string;
} = {}): { fields: MppChallengeFields; header: string } {
  const method = opts.method ?? 'stripe';
  const cents = opts.amountCents ?? 500;
  const request =
    method === 'stripe'
      ? { amount: String(cents), currency: 'usd', methodDetails: { networkId: opts.networkId ?? 'profile_test', paymentMethodTypes: ['card'] } }
      : {
          amount: String(cents * 10_000),
          currency: '0xTOKEN',
          recipient: opts.recipient ?? '0xDEPOSIT',
          externalId: 'mpp_5',
          methodDetails: { chainId: 4217, memo: '0x' + 'ab'.repeat(32), supportedModes: ['pull'] },
        };
  counter += 1;
  const fields: MppChallengeFields = {
    id: createHash('sha256').update(`challenge ${counter}`).digest('base64url'),
    realm: opts.realm ?? 'robutler.ai',
    method,
    intent: opts.intent ?? 'charge',
    request: base64urlEncode(jcsCanonicalize(request)),
    opaque: base64urlEncode(jcsCanonicalize({ packId: 'mpp_5', userId: 'u1', kind: opts.kind ?? 'pack', terms: TERMS })),
  };
  if (opts.expires !== null) fields.expires = opts.expires ?? '2026-09-18T12:05:00.000Z';
  if (opts.header !== null) fields.header = opts.header ?? MPP_CREDENTIAL_HEADER;
  return { fields, header: formatChallenge(fields) };
}

function problem402(ch: { header: string }, opts: { termsVersion?: string | null; problem?: string; body?: Record<string, unknown> } = {}): Response {
  const version = opts.termsVersion === undefined ? TERMS : opts.termsVersion;
  const headers = new Headers({ 'WWW-Authenticate': ch.header, 'Content-Type': 'application/problem+json', 'Cache-Control': 'no-store' });
  if (version) {
    headers.set(TERMS_VERSION_HEADER, version);
    headers.set('Link', `<${TERMS_URL}>; rel="terms-of-service"`);
  }
  const body = {
    type: `https://paymentauth.org/problems/${opts.problem ?? 'payment-required'}`,
    title: 'Payment Required',
    status: 402,
    accepts: [{ scheme: 'token', network: 'robutler' }],
    ...(version ? { terms: { url: TERMS_URL, version, header: TERMS_ACCEPTED_HEADER } } : {}),
    ...opts.body,
  };
  return new Response(JSON.stringify(body), { status: 402, headers });
}

function receiptHeader(reference = 'pi_test_1'): string {
  return base64urlEncode(jcsCanonicalize({ method: 'stripe', reference, status: 'success', timestamp: NOW.toISOString() }));
}

function ok200(reference = 'pi_test_1'): Response {
  return new Response('{"ok":true}', { status: 200, headers: { 'Content-Type': 'application/json', [MPP_RECEIPT_HEADER]: receiptHeader(reference) } });
}

function pending503(extra: Record<string, unknown> = {}): Response {
  return new Response(JSON.stringify({ error: 'charge_outcome_unknown', ...extra, retry: { credentialHeader: MPP_CREDENTIAL_HEADER, sameCredential: true } }), {
    status: 503,
    headers: { 'Retry-After': '5', 'Content-Type': 'application/problem+json' },
  });
}

/** The door after a grant it could not fund from (S-151): a fresh 402 whose body carries the purchase block. */
function grantedButUnfunded402(ch: { header: string }): Response {
  return problem402(ch, { body: { purchase: { granted: true, packId: 'mpp_5', paymentIntentId: 'pi_granted' } } });
}

const PAYER = '0x' + '1f'.repeat(20);
const SIGNED_TX = '0x76' + 'ab'.repeat(24);

interface Seen {
  method: string;
  url: string;
  headers: Headers;
  body: string;
  covered: string[];
}

/** A fetch stub that answers from a queue and records every signed request as the platform would see it. */
function fetchQueue(...responders: Array<Response | ((seen: Seen) => Response)>) {
  const seen: Seen[] = [];
  const fetch = vi.fn(async (input: Request | string | URL) => {
    const request = input instanceof Request ? input : new Request(input);
    const body = request.body ? await request.text() : '';
    const input1 = request.headers.get('signature-input') ?? '';
    const inner = input1.includes('(') ? input1.slice(input1.indexOf('(') + 1, input1.indexOf(')')) : '';
    const record: Seen = { method: request.method, url: request.url, headers: request.headers, body, covered: inner ? inner.split(' ') : [] };
    seen.push(record);
    const next = responders.shift();
    if (!next) throw new Error(`fetch stub exhausted after ${seen.length} requests`);
    return typeof next === 'function' ? next(record) : next;
  });
  return { fetch: fetch as unknown as typeof globalThis.fetch, seen };
}

function buyer(overrides: Partial<MppBuyerConfig> & { policy?: Partial<MppBuyerConfig['policy']> } = {}) {
  const purchases: MppPurchaseRecord[] = [];
  const refusals: MppBuyerRefusal[] = [];
  const sleeps: number[] = [];
  const getSpt = vi.fn(async () => 'spt_test_0001');
  const config: MppBuyerConfig = {
    identity: identity(),
    sources: { card: { getSpt } },
    now: () => NOW,
    sleep: async (ms) => {
      sleeps.push(ms);
    },
    onPurchase: (r) => purchases.push(r),
    onRefusal: (r) => refusals.push(r),
    ...overrides,
    policy: {
      maxPerPurchaseCents: 2000,
      acceptTerms: TERMS,
      realms: ['robutler.ai'],
      // S-154: the sellers the fixtures' challenges name.
      stripeProfileId: 'profile_test',
      tempoDepositAddress: '0xDEPOSIT',
      ...overrides.policy,
    },
  };
  return { buyer: new MppBuyer(config), getSpt, purchases, refusals, sleeps };
}

// ---------------------------------------------------------------------------
// The pure pieces
// ---------------------------------------------------------------------------

describe('parseWwwAuthenticatePayment', () => {
  it('parses quoted and bare values, unescapes once, and keeps the optional fields only when present', () => {
    const parsed = parseWwwAuthenticatePayment('Payment id="a\\"b", realm=robutler.ai, method="stripe", intent="charge", request="cmVx", header="Payment-Authorization"');
    expect(parsed).toEqual({ id: 'a"b', realm: 'robutler.ai', method: 'stripe', intent: 'charge', request: 'cmVx', header: 'Payment-Authorization' });
    expect(Object.keys(parsed!)).not.toContain('expires');
  });

  it('finds the Payment challenge among other schemes and stops where the next scheme begins', () => {
    const value = 'Bearer realm="api", Payment id="x", realm="robutler.ai", method="stripe", intent="charge", request="cmVx", Basic realm="other"';
    expect(parseWwwAuthenticatePayment(value)).toMatchObject({ id: 'x', realm: 'robutler.ai' });
    const trailing = 'Payment id="x", realm="robutler.ai", method="stripe", intent="charge", request="cmVx", Bearer realm="other"';
    expect(parseWwwAuthenticatePayment(trailing)?.realm).toBe('robutler.ai');
  });

  it('is null without a Payment scheme or a required parameter', () => {
    expect(parseWwwAuthenticatePayment('Bearer realm="api"')).toBeNull();
    expect(parseWwwAuthenticatePayment('Payment id="x", realm="r", method="stripe", intent="charge"')).toBeNull();
    expect(parseWwwAuthenticatePayment(null)).toBeNull();
    expect(parseWwwAuthenticatePayment('')).toBeNull();
  });

  it('round-trips the platform format', () => {
    const { fields, header } = challenge();
    expect(parseWwwAuthenticatePayment(header)).toEqual(fields);
  });
});

describe('JCS and base64url', () => {
  it('sorts keys by code unit, drops undefined members, nulls undefined elements, keeps number forms', () => {
    expect(jcsCanonicalize({ b: 1, a: [undefined, 'x'], c: undefined, Z: true, d: { y: null, x: 1e30 } })).toBe('{"Z":true,"a":[null,"x"],"b":1,"d":{"x":1e+30,"y":null}}');
    expect(() => jcsCanonicalize({ a: Number.NaN })).toThrow(/non-finite/);
  });

  it('encodes without padding and decodes strictly', () => {
    const bytes = new Uint8Array([0, 255, 62, 63, 250]);
    const encoded = base64urlEncode(bytes);
    expect(encoded).not.toMatch(/[+/=]/);
    expect(Array.from(base64urlDecode(encoded)!)).toEqual(Array.from(bytes));
    expect(base64urlDecode('abc=')).toBeNull();
    expect(base64urlDecode('a+b')).toBeNull();
    expect(new TextDecoder().decode(base64urlDecode(base64urlEncode('{"a":1}'))!)).toBe('{"a":1}');
  });
});

describe('the decoded challenge', () => {
  it('reads the Stripe request and the Tempo request, amounts in whole cents per method unit', () => {
    const stripe = readMppChallenge(challenge({ amountCents: 500 }).header)!;
    expect(stripe.request).toMatchObject({ amount: '500', currency: 'usd', methodDetails: { networkId: 'profile_test' } });
    expect(stripe.amountCents).toBe(500);
    expect(stripe.expiresAt?.toISOString()).toBe('2026-09-18T12:05:00.000Z');
    expect(stripe.credentialHeader).toBe(MPP_CREDENTIAL_HEADER);

    const tempo = readMppChallenge(challenge({ method: 'tempo', amountCents: 7 }).header)!;
    expect(tempo.request.amount).toBe('70000');
    expect(tempo.request.recipient).toBe('0xDEPOSIT');
    expect(tempo.amountCents).toBe(7);
    expect(challengeAmountCents('tempo', { amount: '70001' })).toBeNull();
    expect(challengeAmountCents('stripe', { amount: '0' })).toBeNull();
  });

  it('defaults the credential header to Authorization when the challenge names none, and tolerates a missing expiry', () => {
    const ch = readMppChallenge(challenge({ header: null, expires: null }).header)!;
    expect(ch.credentialHeader).toBe('Authorization');
    expect(ch.expiresAt).toBeNull();
  });

  it('is null when the request does not decode to a charge request', () => {
    const { fields } = challenge();
    expect(decodeChallengeRequest({ request: base64urlEncode('{"amount":"5.00"}') })).toBeNull();
    expect(decodeChallengeRequest({ request: 'not-json' })).toBeNull();
    expect(readMppChallenge(formatChallenge({ ...fields, request: base64urlEncode('[]') }))).toBeNull();
  });
});

describe('credential and receipt', () => {
  it('encodes Payment <base64url JCS {challenge, payload}> with the challenge echoed field for field', () => {
    const { fields } = challenge();
    const value = encodeMppCredential(fields, { spt: 'spt_1' });
    expect(value.startsWith('Payment ')).toBe(true);
    const body = decode(value.slice('Payment '.length));
    expect(body).toEqual({ challenge: fields, payload: { spt: 'spt_1' } });
    expect(Object.keys(body.challenge as object)).not.toContain('digest');
    const sourced = decode(encodeMppCredential(fields, { spt: 'spt_1' }, '0xPAYER').slice(8));
    expect(sourced.source).toBe('0xPAYER');
  });

  it('is byte-stable: the same fields and payload encode the same string whatever the key order given', () => {
    const { fields } = challenge();
    const shuffled = { header: fields.header, opaque: fields.opaque, request: fields.request, intent: fields.intent, method: fields.method, realm: fields.realm, id: fields.id, expires: fields.expires } as MppChallengeFields;
    expect(encodeMppCredential(shuffled, { b: 2, a: 1 })).toBe(encodeMppCredential(fields, { a: 1, b: 2 }));
  });

  it('wraps a Tempo transaction as the transaction payload', () => {
    expect(tempoCredentialPayload('0xsigned')).toEqual({ type: 'transaction', signature: '0xsigned' });
  });

  it('parses a receipt and refuses anything else', () => {
    expect(parsePaymentReceipt(receiptHeader('pi_9'))).toEqual({ method: 'stripe', reference: 'pi_9', status: 'success', timestamp: NOW.toISOString() });
    expect(parsePaymentReceipt(base64urlEncode('{"status":"success"}'))).toBeNull();
    expect(parsePaymentReceipt(null)).toBeNull();
  });
});

describe('response readers', () => {
  it('retryAfterSeconds reads a delta, an HTTP date, and falls back', () => {
    expect(retryAfterSeconds('5', NOW)).toBe(5);
    expect(retryAfterSeconds(new Date(NOW.getTime() + 12_000).toUTCString(), NOW)).toBe(12);
    expect(retryAfterSeconds('soon', NOW)).toBe(5);
    expect(retryAfterSeconds(null, NOW, 9)).toBe(9);
  });

  it('readTermsNotice prefers the header, falls back to the body, and takes the url from the body or the Link', () => {
    const fromHeader = problem402(challenge(), { termsVersion: '2026-08-01' });
    expect(readTermsNotice(fromHeader, null)).toEqual({ version: '2026-08-01', url: TERMS_URL });
    const bodyOnly = new Response('{}', { status: 402 });
    expect(readTermsNotice(bodyOnly, { terms: { version: '2026-08-02', url: 'https://x/terms' } })).toEqual({ version: '2026-08-02', url: 'https://x/terms' });
    expect(readTermsNotice(bodyOnly, null)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// The buyer
// ---------------------------------------------------------------------------

describe('MppBuyer construction', () => {
  it('refuses a buyer with no source, no ceiling, or no acceptance policy', () => {
    const base = { identity: identity(), sources: { card: { getSpt: async () => 'spt_1' } } };
    expect(() => new MppBuyer({ ...base, sources: {}, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS } })).toThrow(/at least one payment source/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 0, acceptTerms: TERMS } })).toThrow(/maxPerPurchaseCents/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: '' } })).toThrow(/empty version/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1 } as never })).toThrow(/never accepts the Terms on its own/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, methods: ['paypal' as never], realms: ['robutler.ai'] } })).toThrow(/unknown method/);
  });

  it('S-147: refuses a buyer that names no host it pays, and defaults the list to the platform host', () => {
    const base = { identity: identity(), sources: { card: { getSpt: async () => 'spt_1' } } };
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS } })).toThrow(/policy.realms/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, realms: [] } })).toThrow(/policy.realms/);
    expect(() => new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, realms: ['robutler.ai/x'] } })).toThrow(/not a host/);
    const byPlatform = new MppBuyer({ ...base, platformUrl: 'https://robutler.ai', policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS } });
    expect(byPlatform.allowsUrl('https://robutler.ai/agents/acme/v1/chat/completions')).toBe(true);
    expect(byPlatform.allowsUrl('https://evil.example/agents/acme')).toBe(false);
    expect(byPlatform.allowsUrl('https://robutler.ai.evil.example/')).toBe(false);
    const withPort = new MppBuyer({ ...base, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, realms: ['localhost:3000'] } });
    expect(withPort.allowsUrl('http://localhost:3000/api/mpp/credits')).toBe(true);
    expect(withPort.allowsUrl('http://localhost:4000/api/mpp/credits')).toBe(false);
  });

  it('pays in the policy order, only with a source; default is stablecoin first', () => {
    const card = { getSpt: async () => 'spt_1' };
    const stablecoin = { signTempoTransfer: async () => '0x1' };
    const both = new MppBuyer({ identity: identity(), sources: { card, stablecoin }, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, realms: ['robutler.ai'] } });
    expect(both.methodOrder()).toEqual(['tempo', 'stripe']);
    const ordered = new MppBuyer({ identity: identity(), sources: { card, stablecoin }, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, methods: ['stripe', 'tempo'], realms: ['robutler.ai'] } });
    expect(ordered.methodOrder()).toEqual(['stripe', 'tempo']);
    const cardOnly = new MppBuyer({ identity: identity(), sources: { card }, policy: { maxPerPurchaseCents: 1, acceptTerms: TERMS, realms: ['robutler.ai'] } });
    expect(cardOnly.methodOrder()).toEqual(['stripe']);
  });
});

describe('payingFetch', () => {
  it('signs the request and returns a 200 untouched, with no source call; a card-only buyer names its one method', async () => {
    const q = fetchQueue(ok200());
    const b = buyer({ fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"messages":[]}', headers: { 'content-type': 'application/json' } });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: true });
    expect(q.seen).toHaveLength(1);
    const [first] = q.seen;
    expect(first.headers.get('signature-input')).toMatch(/tag="web-bot-auth"/);
    expect(first.headers.get('content-digest')).toMatch(/^sha-256=:/);
    expect(first.headers.get(PURCHASE_HINT_HEADER)).toBeNull();
    // 2026-09-18: without this a card-only buyer met the door's one Tempo
    // challenge whenever stablecoin was on, and refused it.
    expect(first.headers.get(PAYMENT_METHODS_HINT_HEADER)).toBe('stripe');
    expect(first.covered).toContain('"robutler-payment-methods"');
    expect(first.headers.get(TERMS_ACCEPTED_HEADER)).toBeNull();
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.purchases).toEqual([]);
  });

  it('buys the 402 from Robutler and re-sends the same request once, credential and assent covered by the signature', async () => {
    const ch = challenge({ amountCents: 500 });
    const q = fetchQueue(problem402(ch), ok200('pi_abc'));
    const b = buyer({ fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"messages":[1]}', headers: { 'content-type': 'application/json' } });
    expect(response.status).toBe(200);
    expect(q.seen).toHaveLength(2);
    const [first, retry] = q.seen;

    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.getSpt.mock.calls[0][0]).toEqual({
      networkId: 'profile_test',
      amountCents: 500,
      currency: 'usd',
      expiresAt: new Date('2026-09-18T12:05:00.000Z'),
      challengeId: ch.fields.id,
      paymentMethodTypes: ['card'],
    });

    // The same request: method, URL, body bytes and digest.
    expect(retry.method).toBe('POST');
    expect(retry.url).toBe(first.url);
    expect(retry.body).toBe('{"messages":[1]}');
    expect(retry.headers.get('content-digest')).toBe(first.headers.get('content-digest'));
    // A fresh signature: a new nonce.
    expect(retry.headers.get('signature-input')).not.toBe(first.headers.get('signature-input'));

    // The credential echoes the challenge and carries the SPT; the assent names the challenge's version.
    const credential = retry.headers.get(MPP_CREDENTIAL_HEADER)!;
    expect(decode(credential.slice('Payment '.length))).toEqual({ challenge: ch.fields, payload: { spt: 'spt_test_0001' } });
    expect(retry.headers.get(TERMS_ACCEPTED_HEADER)).toBe(TERMS);
    expect(retry.headers.get('authorization')).toBeNull();
    // Both among the covered components, after content-digest and before signature-agent.
    expect(retry.covered).toEqual([
      '"@method"',
      '"@authority"',
      '"@path"',
      '"@query"',
      '"content-digest"',
      // The card-only buyer's method hint (2026-09-18), then the credential and the assent.
      '"robutler-payment-methods"',
      '"payment-authorization"',
      '"robutler-terms-accepted"',
      '"signature-agent";key="sig1"',
    ]);
    expect(first.covered).not.toContain('"payment-authorization"');

    expect(b.purchases).toHaveLength(1);
    expect(b.purchases[0]).toMatchObject({ url: RESOURCE, challengeId: ch.fields.id, method: 'stripe', amountCents: 500, currency: 'usd', termsVersion: TERMS, status: 200 });
    expect(b.purchases[0].receipt?.reference).toBe('pi_abc');
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.refusals).toEqual([]);
  });

  it('sends the policy hints, covered, on every request when the policy sets them', async () => {
    const q = fetchQueue(problem402(challenge()), ok200());
    const b = buyer({ fetch: q.fetch, policy: { preferPurchase: 'exact', methods: ['stripe'] } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    for (const seen of q.seen) {
      expect(seen.headers.get(PURCHASE_HINT_HEADER)).toBe('exact');
      expect(seen.headers.get(PAYMENT_METHODS_HINT_HEADER)).toBe('stripe');
      expect(seen.covered).toContain('"robutler-purchase"');
      expect(seen.covered).toContain('"robutler-payment-methods"');
    }
  });

  it('never pays a challenge whose realm is not the host asked', async () => {
    const q = fetchQueue(problem402(challenge({ realm: 'agents.robutler.ai' })));
    const b = buyer({ fetch: q.fetch, policy: { realms: ['robutler.ai', 'agents.robutler.ai'] } });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(response.status).toBe(402);
    expect(await response.json()).toMatchObject({ status: 402 });
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals).toHaveLength(1);
    expect(b.refusals[0]).toMatchObject({ reason: 'realm_mismatch', amountCents: 500, method: 'stripe', termsVersion: TERMS });
  });

  it('S-147: a third-party host that answers 402 with its own challenge is refused before any source call', async () => {
    // The delegate case: the model chose a URL, and the host there names
    // itself as the realm, so "realm equals the host asked" holds.
    const q = fetchQueue(problem402(challenge({ realm: 'evil.example' })));
    const b = buyer({ fetch: q.fetch });
    const response = await b.buyer.payingFetch('https://evil.example/agents/x/chat/completions', { method: 'POST', body: '{}' });
    expect(response.status).toBe(402);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(q.seen).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
    // A realm on the list fetched from an off-list host is refused too.
    const q2 = fetchQueue(problem402(challenge({ realm: 'robutler.ai' })));
    const b2 = buyer({ fetch: q2.fetch });
    await b2.buyer.payingFetch('https://evil.example/agents/x/chat/completions', { method: 'POST', body: '{}' });
    expect(b2.getSpt).not.toHaveBeenCalled();
    expect(b2.refusals.map((r) => r.reason)).toEqual(['realm_not_allowed']);
  });

  it('refuses above maxPerPurchaseCents, above the daily cap, an unsupported intent, an expired challenge and a method with no source', async () => {
    const cases: Array<[Response, string, Partial<MppBuyerConfig['policy']>]> = [
      [problem402(challenge({ amountCents: 2001 })), 'over_max_per_purchase', {}],
      [problem402(challenge({ amountCents: 600 })), 'over_daily_cap', { dailyCapCents: 500 }],
      [problem402(challenge({ intent: 'session' })), 'unsupported_intent', {}],
      [problem402(challenge({ expires: '2026-09-18T11:59:59.000Z' })), 'challenge_expired', {}],
      [problem402(challenge({ method: 'tempo' })), 'method_unavailable', {}],
    ];
    for (const [response, reason, policy] of cases) {
      const q = fetchQueue(response);
      const b = buyer({ fetch: q.fetch, policy });
      const out = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
      expect(out.status, reason).toBe(402);
      expect(b.getSpt, reason).not.toHaveBeenCalled();
      expect(b.refusals.map((r) => r.reason), reason).toEqual([reason]);
    }
  });

  it('the daily cap counts what was presented in the last 24 hours across calls', async () => {
    const q = fetchQueue(problem402(challenge({ amountCents: 400 })), ok200(), problem402(challenge({ amountCents: 400 })));
    const b = buyer({ fetch: q.fetch, policy: { dailyCapCents: 700 } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(b.buyer.spentTodayCents()).toBe(400);
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(b.refusals.map((r) => r.reason)).toEqual(['over_daily_cap']);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
  });

  it('a pinned Terms version pays only that version; a callback is asked with the version and the url and its answer decides', async () => {
    const pinnedStale = buyer({ fetch: fetchQueue(problem402(challenge(), { termsVersion: '2026-08-15' })).fetch });
    expect((await pinnedStale.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(pinnedStale.getSpt).not.toHaveBeenCalled();
    expect(pinnedStale.refusals[0]).toMatchObject({ reason: 'terms_refused', termsVersion: '2026-08-15' });

    const asked: unknown[] = [];
    const refusing = buyer({ fetch: fetchQueue(problem402(challenge())).fetch, policy: { acceptTerms: (t) => (asked.push(t), false) } });
    expect((await refusing.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(asked).toEqual([{ version: TERMS, url: TERMS_URL }]);
    expect(refusing.getSpt).not.toHaveBeenCalled();

    const q = fetchQueue(problem402(challenge()), ok200());
    const accepting = buyer({ fetch: q.fetch, policy: { acceptTerms: async () => true } });
    expect((await accepting.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(q.seen[1].headers.get(TERMS_ACCEPTED_HEADER)).toBe(TERMS);
  });

  it('a 402 with no Terms notice is paid with no assent header and no assent component', async () => {
    const q = fetchQueue(problem402(challenge(), { termsVersion: null }), ok200());
    const b = buyer({ fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(q.seen[1].headers.get(TERMS_ACCEPTED_HEADER)).toBeNull();
    expect(q.seen[1].covered).not.toContain('"robutler-terms-accepted"');
    expect(b.purchases[0].termsVersion).toBeNull();
  });

  it('re-presents the SAME credential after Retry-After on a 503 that asks for it, and never asks the source again', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), pending503(), ok200('pi_settled'));
    const b = buyer({ fetch: q.fetch });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(response.status).toBe(200);
    expect(q.seen).toHaveLength(4);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    const credentials = q.seen.slice(1).map((s) => s.headers.get(MPP_CREDENTIAL_HEADER));
    expect(new Set(credentials).size).toBe(1);
    // Backoff: Retry-After 5 s, doubling, capped at maxRetryAfterSeconds.
    expect(b.sleeps).toEqual([5000, 10000]);
    expect(b.purchases[0].receipt?.reference).toBe('pi_settled');
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('bounds the settlement retries and keeps the presentation counted, since the charge may have settled', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), pending503());
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 1 } });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(response.status).toBe(503);
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_retries_exhausted']);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('caps the Retry-After it honours', async () => {
    const slow = new Response(JSON.stringify({ retry: { sameCredential: true } }), { status: 503, headers: { 'Retry-After': '600', 'Content-Type': 'application/problem+json' } });
    const q = fetchQueue(problem402(challenge()), slow, ok200());
    const b = buyer({ fetch: q.fetch, policy: { maxRetryAfterSeconds: 7 } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(b.sleeps).toEqual([7000]);
  });

  it('a fresh challenge after paying (a stale Terms version) is paid again when the policy accepts the new version, bounded by maxChallenges', async () => {
    const first = challenge();
    const second = challenge();
    const versions: string[] = [];
    const q = fetchQueue(problem402(first), problem402(second, { termsVersion: '2026-10-01', problem: 'terms-version-stale' }), ok200());
    const b = buyer({ fetch: q.fetch, policy: { acceptTerms: (t) => (versions.push(t.version), true) } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(b.getSpt).toHaveBeenCalledTimes(2);
    expect(versions).toEqual([TERMS, '2026-10-01']);
    expect(q.seen[2].headers.get(TERMS_ACCEPTED_HEADER)).toBe('2026-10-01');
    expect(decode(q.seen[2].headers.get(MPP_CREDENTIAL_HEADER)!.slice(8)).challenge).toEqual(second.fields);
    // The first presentation was refused before any charge, so only the second counts.
    expect(b.buyer.spentTodayCents()).toBe(500);

    const bounded = buyer({ fetch: fetchQueue(problem402(challenge()), problem402(challenge(), { problem: 'payment-expired' })).fetch, policy: { maxChallenges: 1 } });
    const out = await bounded.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(out.status).toBe(402);
    expect(bounded.getSpt).toHaveBeenCalledTimes(1);
    expect(bounded.refusals.map((r) => r.reason)).toEqual(['challenges_exhausted']);
  });

  it('returns a 402 with no readable challenge, a 401, and a 403 as they came', async () => {
    const noChallenge = new Response('{"status":402}', { status: 402, headers: { 'WWW-Authenticate': 'Bearer realm="x"' } });
    for (const response of [noChallenge, new Response('{}', { status: 401 }), new Response('{}', { status: 403 })]) {
      const q = fetchQueue(response);
      const b = buyer({ fetch: q.fetch });
      const out = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
      expect(out).toBe(response);
      expect(q.seen).toHaveLength(1);
      expect(b.getSpt).not.toHaveBeenCalled();
    }
  });

  it('an error after the credential releases the presentation unless the body carries a purchase block', async () => {
    const q1 = fetchQueue(problem402(challenge()), new Response('{"error":"buyer_inactive"}', { status: 403 }));
    const released = buyer({ fetch: q1.fetch });
    expect((await released.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(403);
    expect(released.buyer.spentTodayCents()).toBe(0);

    const q2 = fetchQueue(problem402(challenge()), new Response('{"error":"run_failed","purchase":{"granted":true}}', { status: 500, headers: { 'Content-Type': 'application/json' } }));
    const kept = buyer({ fetch: q2.fetch });
    expect((await kept.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(500);
    expect(kept.buyer.spentTodayCents()).toBe(500);
    // The purchase block means the pack was granted: it is a purchase.
    expect(kept.purchases.map((p) => p.status)).toEqual([500]);
    expect(kept.buyer.pendingCredentials()).toEqual([]);
  });

  it('pays a Tempo challenge with a signed, unbroadcast transaction and names the payer as the credential source', async () => {
    const ch = challenge({ method: 'tempo', amountCents: 7 });
    const q = fetchQueue(problem402(ch), ok200('0xhash'));
    const signTempoTransfer = vi.fn(async () => SIGNED_TX);
    const b = buyer({ fetch: q.fetch, sources: { stablecoin: { signTempoTransfer, address: PAYER } } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(signTempoTransfer).toHaveBeenCalledWith({
      chainId: 4217,
      currency: '0xTOKEN',
      recipient: '0xDEPOSIT',
      amount: '70000',
      memo: '0x' + 'ab'.repeat(32),
      validBefore: new Date('2026-09-18T12:05:00.000Z'),
      challengeId: ch.fields.id,
    });
    const body = decode(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)!.slice(8));
    // The source is the DID the platform's parseTempoPayerDid requires, on the challenge's chain.
    expect(body).toEqual({ challenge: ch.fields, payload: { type: 'transaction', signature: SIGNED_TX }, source: `did:pkh:eip155:4217:${PAYER}` });
    expect(b.purchases[0]).toMatchObject({ method: 'tempo', amountCents: 7, currency: '0xTOKEN' });
    // A stablecoin-only buyer names its one method.
    expect(q.seen[0].headers.get(PAYMENT_METHODS_HINT_HEADER)).toBe('tempo');
  });

  it('refuses a payer address the platform would refuse, and a transaction that is not 0x76, before anything is presented', async () => {
    for (const address of ['0xPAYER', `did:pkh:eip155:1:${PAYER}`]) {
      const q = fetchQueue(problem402(challenge({ method: 'tempo', amountCents: 7 })));
      const signTempoTransfer = vi.fn(async () => SIGNED_TX);
      const b = buyer({ fetch: q.fetch, sources: { stablecoin: { signTempoTransfer, address } } });
      await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow(/stablecoin source/);
      // Nothing was signed for a credential the platform would refuse.
      expect(signTempoTransfer).not.toHaveBeenCalled();
      expect(q.seen).toHaveLength(1);
      expect(b.buyer.spentTodayCents()).toBe(0);
    }
    const q = fetchQueue(problem402(challenge({ method: 'tempo', amountCents: 7 })));
    const b = buyer({ fetch: q.fetch, sources: { stablecoin: { signTempoTransfer: async () => '0xsignedtx', address: PAYER } } });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow(/0x76/);
    expect(q.seen).toHaveLength(1);
  });

  it('a source that fails propagates: nothing is retried against it', async () => {
    const q = fetchQueue(problem402(challenge()));
    const b = buyer({ fetch: q.fetch, sources: { card: { getSpt: async () => { throw new Error('wallet declined'); } } } });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow('wallet declined');
    expect(q.seen).toHaveLength(1);
    expect(b.buyer.spentTodayCents()).toBe(0);
  });

  it('a source that returns something other than an spt_ id is refused before anything is sent', async () => {
    const q = fetchQueue(problem402(challenge()));
    const b = buyer({ fetch: q.fetch, sources: { card: { getSpt: async () => 'tok_visa' } } });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow(/spt_/);
    expect(q.seen).toHaveLength(1);
  });

  it('uses the challenge header parameter for the credential, Authorization when it names none', async () => {
    const q = fetchQueue(problem402(challenge({ header: null })), ok200());
    const b = buyer({ fetch: q.fetch });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(q.seen[1].headers.get('authorization')).toMatch(/^Payment /);
    expect(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)).toBeNull();
    expect(q.seen[1].covered).toContain('"authorization"');
  });
});

describe('purchase (in-band)', () => {
  it('pays a challenge received on the socket at the purchase URL with no prior 402, POST with no body', async () => {
    const ch = challenge();
    const q = fetchQueue(ok200('pi_inband'));
    const b = buyer({ fetch: q.fetch });
    const outcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: ch.header, terms: { url: TERMS_URL, version: TERMS } });
    expect(outcome.ok).toBe(true);
    if (!outcome.ok) return;
    expect(outcome.record).toMatchObject({ url: PURCHASE_URL, challengeId: ch.fields.id, amountCents: 500, termsVersion: TERMS });
    expect(outcome.record.receipt?.reference).toBe('pi_inband');
    expect(q.seen).toHaveLength(1);
    const [sent] = q.seen;
    expect(sent.method).toBe('POST');
    expect(sent.url).toBe(PURCHASE_URL);
    expect(sent.body).toBe('');
    expect(sent.headers.get('content-digest')).toBeNull();
    expect(sent.headers.get(TERMS_ACCEPTED_HEADER)).toBe(TERMS);
    expect(sent.covered).toEqual(['"@method"', '"@authority"', '"@path"', '"@query"', '"robutler-payment-methods"', '"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent";key="sig1"']);
    // An in-band purchase is a purchase: the operator's onPurchase sees it too.
    expect(b.purchases).toHaveLength(1);
    expect(b.purchases[0]).toEqual(outcome.record);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('reports a refusal with its reason, a garbage challenge, and a purchase URL that answers an error', async () => {
    const b1 = buyer({ fetch: fetchQueue().fetch, policy: { maxPerPurchaseCents: 100 } });
    const refused = await b1.buyer.purchase({ url: PURCHASE_URL, challenge: challenge({ amountCents: 500 }).header, terms: { version: TERMS } });
    expect(refused).toMatchObject({ ok: false, reason: 'over_max_per_purchase' });
    expect(b1.getSpt).not.toHaveBeenCalled();

    const b2 = buyer({ fetch: fetchQueue().fetch });
    expect(await b2.buyer.purchase({ url: PURCHASE_URL, challenge: 'Bearer realm="x"' })).toMatchObject({ ok: false, reason: 'no_challenge' });

    const b3 = buyer({ fetch: fetchQueue(new Response('{"error":"buyer_inactive"}', { status: 403 })).fetch });
    const failed = await b3.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
    expect(failed).toMatchObject({ ok: false, reason: 'unexpected_status', status: 403 });
  });

  it('a purchase URL realm must be the purchase host, and a 503 there is re-presented like anywhere else', async () => {
    const q = fetchQueue(pending503(), ok200());
    const b = buyer({ fetch: q.fetch });
    const outcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
    expect(outcome.ok).toBe(true);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(q.seen.map((s) => s.headers.get(MPP_CREDENTIAL_HEADER))).toEqual([q.seen[0].headers.get(MPP_CREDENTIAL_HEADER), q.seen[0].headers.get(MPP_CREDENTIAL_HEADER)]);

    const elsewhere = buyer({ fetch: fetchQueue().fetch });
    expect(await elsewhere.buyer.purchase({ url: 'https://other.example/api/mpp/credits', challenge: challenge().header })).toMatchObject({ ok: false, reason: 'realm_not_allowed' });
  });

  it('S-147: a UAMP peer naming its own purchase URL and realm is refused with no source call and no request', async () => {
    const q = fetchQueue();
    const b = buyer({ fetch: q.fetch });
    const outcome = await b.buyer.purchase({ url: 'https://peer.example/buy', challenge: challenge({ realm: 'peer.example' }).header, terms: { version: TERMS } });
    expect(outcome).toMatchObject({ ok: false, reason: 'realm_not_allowed' });
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(q.seen).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 2026-09-18 fix pass: S-150, S-151, the 503 contract, the ledger, the upgrade
// ---------------------------------------------------------------------------

describe('S-151: a 402 after a settled grant is a paid pack', () => {
  it('counts it, reports it, and does not pay the fresh challenge by default', async () => {
    const q = fetchQueue(problem402(challenge()), grantedButUnfunded402(challenge()));
    const b = buyer({ fetch: q.fetch, policy: { dailyCapCents: 1000 } });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(response.status).toBe(402);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.purchases.map((p) => [p.status, p.amountCents])).toEqual([[402, 500]]);
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.refusals.map((r) => r.reason)).toEqual(['purchase_limit_per_call']);
  });

  it('a second pack is paid only when the policy allows it, and both are counted against the cap', async () => {
    const q = fetchQueue(problem402(challenge()), grantedButUnfunded402(challenge()), ok200('pi_second'));
    const b = buyer({ fetch: q.fetch, policy: { maxPurchasesPerCall: 2 } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(b.getSpt).toHaveBeenCalledTimes(2);
    expect(b.purchases.map((p) => p.status)).toEqual([402, 200]);
    expect(b.buyer.spentTodayCents()).toBe(1000);

    // The reviewer's repro: a cap of one pack now binds, even with a second purchase allowed.
    const capped = buyer({ fetch: fetchQueue(problem402(challenge()), grantedButUnfunded402(challenge())).fetch, policy: { maxPurchasesPerCall: 2, dailyCapCents: 500 } });
    expect((await capped.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(capped.getSpt).toHaveBeenCalledTimes(1);
    expect(capped.refusals.map((r) => r.reason)).toEqual(['over_daily_cap']);
    expect(capped.buyer.spentTodayCents()).toBe(500);
  });
});

describe('the 503 contract (retry.sameCredential)', () => {
  it('reads each answer after a credential by the contract, never by Retry-After alone', () => {
    expect(afterCredential(200, null)).toBe('settled');
    expect(afterCredential(402, { purchase: {} })).toBe('settled');
    // By code, not the flag alone: only funding_failed (or a purchase block) is paid.
    expect(afterCredential(503, { error: 'funding_failed', retry: { sameCredential: false } })).toBe('settled');
    expect(afterCredential(503, { error: 'mpp_not_configured', retry: { sameCredential: false } })).toBe('refused');
    expect(afterCredential(503, { error: 'metered_not_ready', retry: { sameCredential: false } })).toBe('refused');
    expect(afterCredential(503, { error: 'sale_serve_pending', retry: { sameCredential: true } })).toBe('retry_same');
    expect(afterCredential(409, { error: 'purchase_already_granted', purchase: {}, retry: { sameCredential: false } })).toBe('settled');
    expect(afterCredential(409, { error: 'call_already_served', retry: { sameCredential: false } })).toBe('served_elsewhere');
    expect(afterCredential(409, { error: 'call_in_progress', retry: { sameCredential: false } })).toBe('served_elsewhere');
    expect(afterCredential(409, { error: 'call_refunded', retry: { sameCredential: false } })).toBe('refunded');
    expect(afterCredential(429, { error: 'metered_principal_cap' })).toBe('refused');
    expect(afterCredential(503, { retry: { sameCredential: true } })).toBe('retry_same');
    expect(afterCredential(503, null)).toBe('unknown');
    expect(afterCredential(502, null)).toBe('unknown');
    expect(afterCredential(402, null)).toBe('refused');
    expect(afterCredential(403, { error: 'buyer_inactive' })).toBe('refused');
  });

  it('sameCredential false: the purchase is done, and the call is sent once more WITHOUT the credential', async () => {
    const fundingFailed = new Response(JSON.stringify({ error: 'funding_failed', purchase: { granted: true }, retry: { sameCredential: false } }), {
      status: 503,
      headers: { 'Retry-After': '5', 'Content-Type': 'application/problem+json' },
    });
    const q = fetchQueue(problem402(challenge()), fundingFailed, ok200());
    const b = buyer({ fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"n":1}' })).status).toBe(200);
    expect(q.seen).toHaveLength(3);
    expect(q.seen[2].headers.get(MPP_CREDENTIAL_HEADER)).toBeNull();
    expect(q.seen[2].headers.get(TERMS_ACCEPTED_HEADER)).toBeNull();
    expect(q.seen[2].body).toBe('{"n":1}');
    expect(b.sleeps).toEqual([5000]);
    expect(b.purchases.map((p) => p.status)).toEqual([503]);
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.buyer.pendingCredentials()).toEqual([]);
  });

  it('a 503 with Retry-After but no retry member is not re-presented; the credential is held, since money may have moved', async () => {
    const bare = new Response('{}', { status: 503, headers: { 'Retry-After': '5' } });
    const q = fetchQueue(problem402(challenge()), bare);
    const b = buyer({ fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(503);
    expect(q.seen).toHaveLength(2);
    expect(b.sleeps).toEqual([]);
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_outcome_unknown']);
    expect(b.refusals[0].pendingCredential?.value).toBe(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER));
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });
});

describe('S-150: a credential whose outcome is in doubt is never discarded', () => {
  it('re-presents with backoff until the challenge expiry plus the grace, then holds and exposes the credential', async () => {
    let clock = NOW.getTime();
    const tx = '0x' + 'cd'.repeat(32);
    const responders: Response[] = [problem402(challenge({ method: 'tempo', amountCents: 7 }))];
    for (let i = 0; i < 40; i += 1) responders.push(pending503({ error: 'tempo_settlement_pending', transactionHash: tx }));
    const q = fetchQueue(...responders);
    const b = buyer({
      fetch: q.fetch,
      sources: { stablecoin: { signTempoTransfer: async () => SIGNED_TX, address: PAYER } },
      now: () => new Date(clock),
      sleep: async (ms) => {
        clock += ms;
      },
      policy: { settlementGraceSeconds: 60 },
    });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(response.status).toBe(503);
    // Expiry 12:05 plus 60 s: every re-presentation lands before 12:06.
    expect(clock).toBeLessThanOrEqual(Date.parse('2026-09-18T12:06:00.000Z'));
    expect(clock).toBeGreaterThanOrEqual(Date.parse('2026-09-18T12:05:30.000Z'));
    expect(q.seen.length).toBeGreaterThan(5);
    const credentials = new Set(q.seen.slice(1).map((s) => s.headers.get(MPP_CREDENTIAL_HEADER)));
    expect(credentials.size).toBe(1);
    const [refusal] = b.refusals;
    expect(refusal.reason).toBe('settlement_retries_exhausted');
    expect(refusal.pendingCredential).toMatchObject({ transactionHash: tx, paymentMethod: 'tempo', amountCents: 7, httpMethod: 'POST', url: RESOURCE, headerName: MPP_CREDENTIAL_HEADER });
    expect(b.buyer.pendingCredentials()).toEqual([refusal.pendingCredential]);
    expect(b.buyer.spentTodayCents()).toBe(7);
  });

  it('the next call to the same resource re-presents the held credential BEFORE paying again, and it completes', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), problem402(challenge()), ok200('pi_completed'));
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"call":1}' })).status).toBe(503);
    const held = b.buyer.pendingCredentials();
    expect(held).toHaveLength(1);

    const second = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"call":2}' });
    expect(second.status).toBe(200);
    // One SPT for both calls: the second call carried the held credential.
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(q.seen[3].headers.get(MPP_CREDENTIAL_HEADER)).toBe(held[0].value);
    expect(q.seen[3].body).toBe('{"call":2}');
    expect(b.purchases.map((p) => [p.challengeId, p.receipt?.reference])).toEqual([[held[0].id, 'pi_completed']]);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('a held credential the platform now refuses is cleared and reported, and nothing new is paid in that call', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), problem402(challenge()), problem402(challenge(), { problem: 'payment-expired' }));
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const out = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(out.status).toBe(402);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_retries_exhausted', 'pending_credential_refused']);
    expect(b.refusals[1].pendingCredential?.id).toBe(b.refusals[0].pendingCredential?.id);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    // Still counted: money may have moved before the doubt began.
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('an abort while the credential is out holds it, and the error still reaches the caller', async () => {
    const q = fetchQueue(problem402(challenge()), () => {
      throw new Error('socket hang up');
    });
    const b = buyer({ fetch: q.fetch });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow('socket hang up');
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_outcome_unknown']);
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('a held pack credential from a door is re-presented at the in-band purchase URL before a new challenge is paid', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), ok200('pi_inband_completed'));
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const [held] = b.buyer.pendingCredentials();
    const outcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
    expect(outcome.ok).toBe(true);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(q.seen[2].url).toBe(PURCHASE_URL);
    expect(q.seen[2].headers.get(MPP_CREDENTIAL_HEADER)).toBe(held.value);
    if (outcome.ok) expect(outcome.record.challengeId).toBe(held.id);
  });

  it('a held credential is never re-presented to a host off the allowlist', async () => {
    const q = fetchQueue(problem402(challenge()), pending503());
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const outcome = await b.buyer.purchase({ url: 'https://peer.example/buy', challenge: challenge({ realm: 'peer.example' }).header });
    expect(outcome).toMatchObject({ ok: false, reason: 'realm_not_allowed' });
    expect(q.seen).toHaveLength(2);
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
  });

  it('the purchase outcome carries the held credential', async () => {
    const q = fetchQueue(pending503());
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    const outcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
    expect(outcome.ok).toBe(false);
    if (outcome.ok) return;
    expect(outcome.reason).toBe('settlement_retries_exhausted');
    expect(outcome.pendingCredential?.value).toBe(q.seen[0].headers.get(MPP_CREDENTIAL_HEADER));
  });

  it('policy.persist carries the held credential and the ledger to a new process, which re-presents instead of paying', async () => {
    const saved: MppBuyerState[] = [];
    const persist = { load: () => null, save: (state: MppBuyerState) => void saved.push(JSON.parse(JSON.stringify(state))) };
    const first = buyer({ fetch: fetchQueue(problem402(challenge()), pending503()).fetch, policy: { maxSettlementRetries: 0, persist } });
    await first.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const last = saved[saved.length - 1];
    expect(last.pending).toHaveLength(1);
    expect(last.ledger.map((e) => e.cents)).toEqual([500]);

    const q = fetchQueue(problem402(challenge()), ok200('pi_after_restart'));
    const restarted = buyer({ fetch: q.fetch, policy: { dailyCapCents: 600, persist: { load: async () => last, save: () => undefined } } });
    expect((await restarted.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(restarted.getSpt).not.toHaveBeenCalled();
    expect(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER)).toBe(last.pending[0].value);
    expect(restarted.buyer.spentTodayCents()).toBe(500);
    expect(restarted.buyer.pendingCredentials()).toEqual([]);
  });
});

describe('the daily cap under concurrency', () => {
  it('is reserved before any await, so concurrent calls cannot all pass the check', async () => {
    const responders: Array<Response | ((seen: Seen) => Response)> = [];
    for (let i = 0; i < 6; i += 1) responders.push(() => problem402(challenge({ amountCents: 500 })));
    for (let i = 0; i < 6; i += 1) responders.push(() => ok200());
    const q = fetchQueue(...responders);
    const b = buyer({ fetch: q.fetch, policy: { dailyCapCents: 1000 } });
    b.getSpt.mockImplementation(async () => {
      await new Promise((r) => setTimeout(r, 5));
      return 'spt_test_0001';
    });
    const answers = await Promise.all(Array.from({ length: 6 }, () => b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })));
    expect(b.getSpt).toHaveBeenCalledTimes(2);
    expect(answers.filter((r) => r.status === 200)).toHaveLength(2);
    expect(b.buyer.spentTodayCents()).toBe(1000);
    expect(b.refusals.filter((r) => r.reason === 'over_daily_cap')).toHaveLength(4);
  });
});

describe('upgradeHeaders', () => {
  it('signs a GET on the https form of the socket URL, the request the socket door rebuilds, with the hints covered', async () => {
    const b = buyer();
    const headers = await b.buyer.upgradeHeaders('wss://robutler.ai/agents/acme/uamp');
    expect(headers[PAYMENT_METHODS_HINT_HEADER]).toBe('stripe');
    expect(headers['signature-input']).toMatch(/^sig1=\("@method" "@authority" "@path" "@query" "robutler-payment-methods" "signature-agent";key="sig1"\)/);
    expect(headers['content-digest']).toBeUndefined();
    expect(headers['signature-agent']).toContain(AGENT_URL);
    expect(headers.signature).toMatch(/^sig1=:/);
  });

  it('signs nothing for a host the buyer does not buy from', async () => {
    expect(await buyer().buyer.upgradeHeaders('wss://peer.example/agents/acme/uamp')).toEqual({});
  });
});

describe('tempoPayerDid', () => {
  it('builds the DID the platform parses, passes a matching DID through, and refuses anything else', () => {
    expect(tempoPayerDid(4217, PAYER)).toBe(`did:pkh:eip155:4217:${PAYER}`);
    expect(tempoPayerDid(4217, `did:pkh:eip155:4217:${PAYER}`)).toBe(`did:pkh:eip155:4217:${PAYER}`);
    expect(() => tempoPayerDid(4217, `did:pkh:eip155:42431:${PAYER}`)).toThrow(/chain/);
    expect(() => tempoPayerDid(4217, '0xPAYER')).toThrow(/40 hex/);
  });
});

// ---------------------------------------------------------------------------
// 2026-09-18, second fix pass: the portal door's new answers to a credential
// ---------------------------------------------------------------------------

function problemResponse(status: number, body: Record<string, unknown>, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/problem+json', ...headers } });
}

const BODY_RECEIPT = { method: 'stripe', reference: 'pi_lost_200', status: 'success', timestamp: '2026-09-18T12:00:01.000Z' };

describe('the portal door answers of 2026-09-18', () => {
  it('409 purchase_already_granted: paid once, the held credential dropped, the call sent once more WITHOUT a credential', async () => {
    const alreadyGranted = problemResponse(409, {
      error: 'purchase_already_granted',
      receipt: BODY_RECEIPT,
      purchase: { granted: false, receipt: BODY_RECEIPT },
      retry: { sameCredential: false },
    });
    const q = fetchQueue(problem402(challenge()), () => {
      throw new Error('socket hang up');
    }, problem402(challenge()), alreadyGranted, ok200('served_from_balance'));
    const b = buyer({ fetch: q.fetch });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"call":1}' })).rejects.toThrow('socket hang up');
    const [held] = b.buyer.pendingCredentials();

    const second = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{"call":2}' });
    expect(second.status).toBe(200);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(q.seen[3].headers.get(MPP_CREDENTIAL_HEADER)).toBe(held.value);
    expect(q.seen[4].headers.get(MPP_CREDENTIAL_HEADER)).toBeNull();
    expect(q.seen[4].body).toBe('{"call":2}');
    expect(b.sleeps).toEqual([]);
    expect(b.purchases.map((p) => [p.challengeId, p.status, p.receipt?.reference])).toEqual([[held.id, 409, 'pi_lost_200']]);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    expect(b.buyer.spentTodayCents()).toBe(500);
  });

  it('409 call_already_served and call_in_progress are terminal: paid, never paid again, never re-presented, the receipt surfaced', async () => {
    for (const code of ['call_already_served', 'call_in_progress'] as const) {
      const q = fetchQueue(problemResponse(409, { error: code, challengeId: 'x', receipt: BODY_RECEIPT, retry: { sameCredential: false } }));
      const b = buyer({ fetch: q.fetch });
      const outcome: MppPurchaseOutcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
      expect(outcome.ok, code).toBe(false);
      if (outcome.ok) continue;
      expect(outcome.reason).toBe(code);
      expect(outcome.receipt?.reference).toBe('pi_lost_200');
      expect(q.seen, code).toHaveLength(1);
      expect(b.sleeps, code).toEqual([]);
      expect(b.getSpt).toHaveBeenCalledTimes(1);
      expect(b.purchases.map((p) => p.status), code).toEqual([409]);
      expect(b.refusals, code).toEqual([]);
      expect(b.buyer.spentTodayCents(), code).toBe(500);
      expect(b.buyer.pendingCredentials()).toEqual([]);
    }
  });

  it('409 call_refunded frees the reservation and drops the held credential; nothing is paid again', async () => {
    const q = fetchQueue(problem402(challenge()), pending503(), problem402(challenge()), problemResponse(409, { error: 'call_refunded', receipt: BODY_RECEIPT, retry: { sameCredential: false } }));
    const b = buyer({ fetch: q.fetch, policy: { maxSettlementRetries: 0 } });
    await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(b.buyer.spentTodayCents()).toBe(500);
    const out = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    expect(out.status).toBe(409);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.buyer.pendingCredentials()).toEqual([]);
    expect(b.purchases).toEqual([]);
  });

  it('the velocity 429s are refusals: nothing charged, the reservation freed, reported, never retried', async () => {
    for (const code of ['metered_principal_cap', 'metered_instrument_cap', 'metered_platform_ceiling', 'buyer_daily_cap']) {
      const q = fetchQueue(problem402(challenge()), problemResponse(429, { error: code }, { 'Retry-After': '3600' }));
      const b = buyer({ fetch: q.fetch });
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status, code).toBe(429);
      expect(q.seen, code).toHaveLength(2);
      expect(b.sleeps, code).toEqual([]);
      expect(b.buyer.spentTodayCents(), code).toBe(0);
      expect(b.purchases, code).toEqual([]);
      expect(b.refusals.map((r) => r.reason), code).toEqual(['platform_refused']);
      expect(b.refusals[0].detail).toContain(code);
      expect(b.refusals[0].challengeId).toBeTruthy();
    }
  });

  it('503 sale_serve_pending (sameCredential true) is re-presented with the same credential', async () => {
    const pendingServe = problemResponse(503, { error: 'sale_serve_pending', challengeId: 'x', retry: { credentialHeader: MPP_CREDENTIAL_HEADER, sameCredential: true } }, { 'Retry-After': '5' });
    const q = fetchQueue(problem402(challenge()), pendingServe, ok200());
    const b = buyer({ fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(q.seen[2].headers.get(MPP_CREDENTIAL_HEADER)).toBe(q.seen[1].headers.get(MPP_CREDENTIAL_HEADER));
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(b.purchases).toHaveLength(1);
  });

  it('503 mpp_not_configured and metered_not_ready say sameCredential false and are NOT a purchase: freed, reported, no retry', async () => {
    for (const code of ['mpp_not_configured', 'metered_not_ready']) {
      const q = fetchQueue(problem402(challenge()), problemResponse(503, { error: code, retry: { sameCredential: false } }, { 'Retry-After': '3600' }));
      const b = buyer({ fetch: q.fetch });
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status, code).toBe(503);
      expect(q.seen, code).toHaveLength(2);
      expect(b.purchases, code).toEqual([]);
      expect(b.buyer.spentTodayCents(), code).toBe(0);
      expect(b.buyer.pendingCredentials(), code).toEqual([]);
      expect(b.refusals.map((r) => r.reason), code).toEqual(['platform_refused']);
    }
  });
});

// ---------------------------------------------------------------------------
// 2026-09-18, S-154 residual: the seller is pinned
// ---------------------------------------------------------------------------

function discoveryDoc(offers: Array<Record<string, unknown>>): Response {
  const doc = { openapi: '3.1.0', paths: { '/api/mpp/credits': { post: { 'x-payment-info': { offers } } } } };
  return new Response(JSON.stringify(doc), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

describe('S-154: the seller is pinned', () => {
  it('a forged card challenge with the platform realm and a foreign Stripe profile is refused before the source is asked; the genuine one is paid', async () => {
    const q = fetchQueue(problem402(challenge({ networkId: 'profile_attacker' })), problem402(challenge()), ok200());
    const b = buyer({ fetch: q.fetch });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(b.refusals[0].detail).toContain('profile_attacker');
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
  });

  it('an in-band challenge naming a foreign profile is refused with no request at all', async () => {
    const q = fetchQueue();
    const b = buyer({ fetch: q.fetch });
    const outcome = await b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge({ networkId: 'profile_attacker' }).header, terms: { version: TERMS } });
    expect(outcome).toMatchObject({ ok: false, reason: 'seller_not_pinned' });
    expect(q.seen).toHaveLength(0);
    expect(b.getSpt).not.toHaveBeenCalled();
  });

  it('a forged Tempo challenge naming a foreign recipient is refused before the wallet signs; the genuine one is paid', async () => {
    const signTempoTransfer = vi.fn(async () => SIGNED_TX);
    const q = fetchQueue(problem402(challenge({ method: 'tempo', amountCents: 7, recipient: '0xATTACKER' })), problem402(challenge({ method: 'tempo', amountCents: 7, recipient: '0xdeposit' })), ok200());
    const b = buyer({ fetch: q.fetch, sources: { stablecoin: { signTempoTransfer, address: PAYER } } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(signTempoTransfer).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    // Addresses compare case-insensitively.
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(signTempoTransfer).toHaveBeenCalledTimes(1);
  });

  it('with no explicit pin, the pin is read once from https://<platform host>/openapi.json and cached', async () => {
    const q = fetchQueue(
      problem402(challenge()),
      discoveryDoc([{ intent: 'charge', method: 'stripe', amount: null, currency: 'usd', payTo: 'profile_test' }]),
      ok200(),
      problem402(challenge({ networkId: 'profile_attacker' })),
    );
    const b = buyer({ fetch: q.fetch, policy: { stripeProfileId: undefined } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(q.seen[1].method).toBe('GET');
    expect(q.seen[1].url).toBe('https://robutler.ai/openapi.json');
    // Unsigned: discovery is public, and nothing identifies the agent to it.
    expect(q.seen[1].headers.get('signature-input')).toBeNull();
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(q.seen.filter((s) => s.url.endsWith('/openapi.json'))).toHaveLength(1);
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
  });

  it('fails closed: a discovery document naming no seller, or two, pins nothing and nothing is paid', async () => {
    for (const offers of [
      [{ intent: 'charge', method: 'stripe', amount: null, currency: 'usd' }],
      [{ method: 'stripe', payTo: 'profile_test' }, { method: 'stripe', payTo: 'profile_other' }],
    ]) {
      const q = fetchQueue(problem402(challenge()), discoveryDoc(offers));
      const b = buyer({ fetch: q.fetch, policy: { stripeProfileId: undefined } });
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
      expect(b.getSpt).not.toHaveBeenCalled();
      expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    }
  });

  it('discovery goes over https even for a plaintext platform, a failed fetch pays nothing, and the tempo pin reads recipient or extra.recipient', async () => {
    const local = challenge({ realm: 'localhost' });
    const q = fetchQueue(problem402(local), () => {
      throw new Error('connect ECONNREFUSED');
    });
    const b = buyer({ fetch: q.fetch, policy: { stripeProfileId: undefined, realms: ['localhost:3000'] } });
    expect((await b.buyer.payingFetch('http://localhost:3000/agents/acme/v1/chat/completions', { method: 'POST', body: '{}' })).status).toBe(402);
    expect(q.seen[1].url).toBe('https://localhost:3000/openapi.json');
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(sellerPinsFromDiscovery({ paths: { '/x': { post: { 'x-payment-info': { offers: [{ method: 'tempo', extra: { recipient: '0xABC' } }] } } } } })).toEqual({
      stripeProfileId: null,
      tempoDepositAddress: '0xabc',
    });
    expect(sellerPinsFromDiscovery(null)).toEqual({ stripeProfileId: null, tempoDepositAddress: null });
  });
});

// ---------------------------------------------------------------------------
// 2026-09-19 review: every send is bounded, the discovery read is bounded
// while it streams and its pin expires, and a challenge that names a field
// the credential cannot be sent in is refused before any source call.
// (S-180, the redirects, has its own file: mpp-buyer-redirect.test.ts.)
// ---------------------------------------------------------------------------

/** A fetch that never answers: it settles only when the request's own signal aborts, with that signal's reason. */
function hangingFetch(seen: Request[]) {
  return (async (input: Request) => {
    seen.push(input);
    return new Promise<Response>((_, reject) => {
      input.signal.addEventListener('abort', () => reject(input.signal.reason), { once: true });
    });
  }) as unknown as typeof globalThis.fetch;
}

/** Only the timeout clock is faked; signing and body reads run on real event-loop turns, which `reached` waits out. */
const fakeTimeoutClock = () => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
async function reached(condition: () => boolean): Promise<void> {
  for (let i = 0; i < 500 && !condition(); i += 1) await new Promise((resolve) => setImmediate(resolve));
  expect(condition()).toBe(true);
}

describe('every send is bounded by policy.purchaseTimeoutSeconds', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('payingFetch with no caller signal times out instead of hanging with nothing reserved', async () => {
    fakeTimeoutClock();
    const seen: Request[] = [];
    const b = buyer({ fetch: hangingFetch(seen), policy: { purchaseTimeoutSeconds: 7, dailyCapCents: 5000 } });
    const attempt = b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const settled = expect(attempt).rejects.toMatchObject({ name: 'TimeoutError' });
    await reached(() => seen.length === 1);
    await vi.advanceTimersByTimeAsync(6_999);
    expect(seen[0].signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(2);
    await settled;
    expect(b.buyer.spentTodayCents()).toBe(0);
  });

  it('the default bound is 60 s, and it covers the paid retry: a retry that never answers ends, its credential held', async () => {
    fakeTimeoutClock();
    const seen: Request[] = [];
    const hang = hangingFetch(seen);
    let first = true;
    const fetch = (async (input: Request) => {
      if (first) {
        first = false;
        return problem402(challenge());
      }
      return hang(input);
    }) as unknown as typeof globalThis.fetch;
    const b = buyer({ fetch, policy: { dailyCapCents: 5000 } });
    const attempt = b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    const settled = expect(attempt).rejects.toMatchObject({ name: 'TimeoutError' });
    await reached(() => seen.length === 1);
    await vi.advanceTimersByTimeAsync(59_000);
    expect(seen[0].signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1_500);
    await settled;
    // The credential LEFT, so its outcome is unknown: held and still counted (S-150), not a hang.
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
    expect(b.buyer.spentTodayCents()).toBe(500);
    expect(b.refusals.map((r) => r.reason)).toEqual(['settlement_outcome_unknown']);
  });

  it("is combined with the caller's signal: whichever fires first ends the send", async () => {
    fakeTimeoutClock();
    const seen: Request[] = [];
    const b = buyer({ fetch: hangingFetch(seen), policy: { purchaseTimeoutSeconds: 60 } });
    const caller = new AbortController();
    const attempt = b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}', signal: caller.signal });
    const settled = expect(attempt).rejects.toThrow('the caller gave up');
    await reached(() => seen.length === 1);
    caller.abort(new Error('the caller gave up'));
    await settled;
    expect(seen[0].signal.aborted).toBe(true);
  });

  it('bounds the wait for the answer to BEGIN, not a body the caller streams afterwards', async () => {
    fakeTimeoutClock();
    const requests: Request[] = [];
    const fetch = (async (input: Request) => {
      requests.push(input);
      return new Response('{"ok":true}', { status: 200 });
    }) as unknown as typeof globalThis.fetch;
    const b = buyer({ fetch, policy: { purchaseTimeoutSeconds: 1 } });
    const response = await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' });
    await vi.advanceTimersByTimeAsync(5_000);
    expect(requests[0].signal.aborted).toBe(false);
    expect(await response.json()).toEqual({ ok: true });
  });

  it('the in-band purchase is bounded the same way', async () => {
    fakeTimeoutClock();
    const seen: Request[] = [];
    const b = buyer({ fetch: hangingFetch(seen), policy: { purchaseTimeoutSeconds: 3 } });
    const attempt = b.buyer.purchase({ url: PURCHASE_URL, challenge: challenge().header, terms: { version: TERMS } });
    const settled = expect(attempt).rejects.toMatchObject({ name: 'TimeoutError' });
    await reached(() => seen.length === 1);
    await vi.advanceTimersByTimeAsync(3_001);
    await settled;
    expect(b.buyer.pendingCredentials()).toHaveLength(1);
  });
});

describe('S-154 discovery: the read is bounded while it streams, and a pin is believed for an hour', () => {
  it('stops reading at 1 MB instead of buffering whatever the host sends', async () => {
    let pulled = 0;
    let cancelled = false;
    const endless = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled += 1;
        controller.enqueue(new Uint8Array(64 * 1024).fill(0x20));
      },
      cancel() {
        cancelled = true;
      },
    });
    // No Content-Length: only counting the stream can bound it.
    const q = fetchQueue(problem402(challenge()), new Response(endless, { status: 200, headers: { 'Content-Type': 'application/json' } }));
    const b = buyer({ fetch: q.fetch, policy: { stripeProfileId: undefined } });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(b.getSpt).not.toHaveBeenCalled();
    expect(cancelled).toBe(true);
    // 1 MB is 16 chunks of 64 KB; a few more may be queued by the stream itself, never an unbounded number.
    expect(pulled).toBeGreaterThanOrEqual(17);
    expect(pulled).toBeLessThan(40);
  });

  it('re-reads a positive pin after an hour, so a rotated deposit address is picked up without a restart', async () => {
    let now = NOW;
    const signTempoTransfer = vi.fn(async () => SIGNED_TX);
    const tempoOffer = (recipient: string) => discoveryDoc([{ intent: 'charge', method: 'tempo', recipient }]);
    const q = fetchQueue(
      problem402(challenge({ method: 'tempo', recipient: '0xDEPOSIT', expires: null })),
      tempoOffer('0xDEPOSIT'),
      ok200(),
      // 59 minutes later: still believed, so the rotated address is refused and nothing is fetched.
      problem402(challenge({ method: 'tempo', recipient: '0xROTATED', expires: null })),
      // 61 minutes later: read again.
      problem402(challenge({ method: 'tempo', recipient: '0xROTATED', expires: null })),
      tempoOffer('0xROTATED'),
      ok200(),
    );
    const b = buyer({
      fetch: q.fetch,
      now: () => now,
      sources: { stablecoin: { signTempoTransfer, address: PAYER } },
      policy: { tempoDepositAddress: undefined },
    });
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);

    now = new Date(NOW.getTime() + 59 * 60_000);
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
    expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    expect(q.seen.filter((s) => s.url.endsWith('/openapi.json'))).toHaveLength(1);

    now = new Date(NOW.getTime() + 61 * 60_000);
    expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
    expect(q.seen.filter((s) => s.url.endsWith('/openapi.json'))).toHaveLength(2);
    expect(signTempoTransfer).toHaveBeenCalledTimes(2);
    expect(signTempoTransfer.mock.calls[1][0]).toMatchObject({ recipient: '0xROTATED' });
  });

  it('an expired pin is never a fallback: a re-read that fails, or names two sellers, pays nothing', async () => {
    for (const reread of [
      () => {
        throw new Error('connect ETIMEDOUT');
      },
      () => discoveryDoc([{ method: 'stripe', payTo: 'profile_test' }, { method: 'stripe', payTo: 'profile_other' }]),
    ]) {
      let now = NOW;
      const q = fetchQueue(problem402(challenge()), discoveryDoc([{ method: 'stripe', payTo: 'profile_test' }]), ok200(), problem402(challenge()), reread);
      const b = buyer({ fetch: q.fetch, now: () => now, policy: { stripeProfileId: undefined } });
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(200);
      now = new Date(NOW.getTime() + 61 * 60_000);
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
      expect(b.getSpt).toHaveBeenCalledTimes(1);
      expect(b.refusals.map((r) => r.reason)).toEqual(['seller_not_pinned']);
    }
  });
});

describe('the credential header is checked when the challenge is read', () => {
  const REFUSED = [
    'Content-Digest',
    'signature',
    'Signature-Input',
    'Signature-Agent',
    TERMS_ACCEPTED_HEADER,
    PURCHASE_HINT_HEADER,
    PAYMENT_METHODS_HINT_HEADER,
    'Host',
    'Content-Length',
    'Transfer-Encoding',
    'Connection',
    'Cookie',
    'bad name',
    '@method',
    'x:y',
  ];

  it('a challenge naming a field the signer, the buyer or the HTTP client owns is not a readable challenge', () => {
    for (const header of REFUSED) {
      expect(readMppChallenge(challenge({ header }).header), header).toBeNull();
      expect(credentialHeaderRefusal(header), header).toEqual(expect.any(String));
    }
    // The platform's name, the core spec's default (named or absent), and an ordinary extension field are fine.
    expect(readMppChallenge(challenge().header)?.credentialHeader).toBe(MPP_CREDENTIAL_HEADER);
    expect(readMppChallenge(challenge({ header: 'authorization' }).header)?.credentialHeader).toBe('authorization');
    expect(readMppChallenge(challenge({ header: null }).header)?.credentialHeader).toBe('Authorization');
    expect(readMppChallenge(challenge({ header: 'X-Payment-Credential' }).header)?.credentialHeader).toBe('X-Payment-Credential');
    expect([...CREDENTIAL_HEADERS_REFUSED]).toEqual(expect.arrayContaining(['content-digest', 'signature', 'signature-input', 'signature-agent']));
  });

  it('is refused BEFORE the source is asked: no token issued, nothing reserved, nothing held', async () => {
    for (const header of ['Content-Digest', 'Signature', 'Content-Length']) {
      const q = fetchQueue(problem402(challenge({ header })));
      const b = buyer({ fetch: q.fetch, policy: { dailyCapCents: 5000 } });
      // It used to throw out of the signer (or out of fetch) AFTER getSpt, and hold the credential against the cap.
      expect((await b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).status).toBe(402);
      expect(b.getSpt).not.toHaveBeenCalled();
      expect(q.seen).toHaveLength(1);
      expect(b.buyer.spentTodayCents()).toBe(0);
      expect(b.buyer.pendingCredentials()).toEqual([]);

      const inBand = buyer({ fetch: fetchQueue().fetch });
      expect(await inBand.buyer.purchase({ url: PURCHASE_URL, challenge: challenge({ header }).header, terms: { version: TERMS } })).toMatchObject({
        ok: false,
        reason: 'no_challenge',
      });
      expect(inBand.getSpt).not.toHaveBeenCalled();
    }
  });

  it('a paid retry the signer refuses BEFORE it is sent releases the reservation instead of holding a credential that never left', async () => {
    // A Terms version the operator's callback accepts but the signer cannot cover (not printable ASCII).
    const q = fetchQueue(problem402(challenge(), { termsVersion: 'v1é' }));
    const b = buyer({ fetch: q.fetch, policy: { acceptTerms: () => true, dailyCapCents: 5000 } });
    await expect(b.buyer.payingFetch(RESOURCE, { method: 'POST', body: '{}' })).rejects.toThrow(/not printable ASCII/);
    expect(b.getSpt).toHaveBeenCalledTimes(1);
    expect(q.seen).toHaveLength(1);
    expect(b.buyer.spentTodayCents()).toBe(0);
    expect(b.buyer.pendingCredentials()).toEqual([]);
  });
});
