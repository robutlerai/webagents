/**
 * Standard x402 on a priced endpoint (webagents gap-closure plan 2.6,
 * 2026-09-26): the wire codec against the spec pack's section 1.10 vectors,
 * the well-formedness rule, the credits nonce, and the paywall's flow with a
 * fake platform (credits) and a local fake facilitator (chain): a 402 with
 * the v2 header and the v1 body, verify before the handler, settle once
 * after it, nothing settled on a handler error, `upto` through
 * `Settlement-Overrides`, and `settlement_pending` delivered but never
 * settled twice. The fixture `python/tests/fixtures/payments/
 * paywall_x402_mpp.json` is read by the Python suite too.
 */

import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  CREDITS_ASSET,
  CREDITS_DECIMALS,
  CREDITS_NETWORK,
  CREDITS_PAY_TO,
  CREDITS_SCHEME,
  X402_CORS_ALLOW_HEADERS,
  X402_CORS_EXPOSE_HEADERS,
  X402_HEADERS,
  V1_NETWORK_NAMES,
  creditsToNanocredits,
  decodeBase64Json,
  encodeBase64Json,
  isWellFormedRequirement,
  requirementMatches,
  type PaymentRequiredV1,
  type PaymentRequiredV2,
  type PaymentRequirement,
} from '../../../../src/skills/payments/x402-wire.js';
import { mintCreditsNonce, verifyCreditsNonce, CreditsScheme, type CreditsClient } from '../../../../src/skills/payments/x402-credits.js';
import { Paywall, type PricedEndpoint } from '../../../../src/skills/payments/paywall.js';
import type { FacilitatorClient } from '../../../../src/skills/payments/x402-facilitator.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/payments/paywall_x402_mpp.json'), 'utf8'));
const X = FIXTURE.x402;

// ── The names ────────────────────────────────────────────────────────────────

describe('the fixture pins the names', () => {
  it('headers, CORS lists, the credits scheme and the v1 network names', () => {
    expect(X402_HEADERS.required).toBe(X.headers.required);
    expect(X402_HEADERS.signature).toBe(X.headers.signature);
    expect(X402_HEADERS.response).toBe(X.headers.response);
    expect(X402_HEADERS.v1Payment).toBe(X.headers.v1_payment);
    expect(X402_HEADERS.v1Response).toBe(X.headers.v1_response);
    expect(X402_HEADERS.settlementOverrides).toBe(X.headers.settlement_overrides);
    expect([...X402_CORS_EXPOSE_HEADERS]).toEqual(X.cors.expose);
    expect([...X402_CORS_ALLOW_HEADERS]).toEqual(X.cors.allow);
    expect(CREDITS_SCHEME).toBe(X.credits_scheme.scheme);
    expect(CREDITS_NETWORK).toBe(X.credits_scheme.network);
    expect(CREDITS_ASSET).toBe(X.credits_scheme.asset);
    expect(CREDITS_PAY_TO).toBe(X.credits_scheme.pay_to);
    expect(CREDITS_DECIMALS).toBe(X.credits_scheme.decimals);
    for (const [caip2, name] of Object.entries(X.v1_network_names)) expect(V1_NETWORK_NAMES[caip2]).toBe(name);
  });
});

// ── Section 1.10 vectors ─────────────────────────────────────────────────────

describe('section 1.10 vectors', () => {
  it('PAYMENT-REQUIRED, PAYMENT-SIGNATURE and both settle responses decode and re-encode byte for byte', () => {
    for (const name of ['payment_required_v2', 'payment_signature_v2', 'settle_response_success', 'settle_response_failure']) {
      const v = X.x402_vectors[name];
      expect(decodeBase64Json(v.base64)).toEqual(v.json);
      expect(encodeBase64Json(v.json)).toBe(v.base64);
    }
    expect(X.x402_vectors.payment_signature_v2.base64).toHaveLength(1088);
  });

  it('refuses anything that is not standard base64 with padding', () => {
    for (const bad of X.x402_vectors.not_base64_standard) expect(decodeBase64Json(bad)).toBeNull();
  });

  it('the verify request names both documents under x402Version 2', () => {
    const req = X.x402_vectors.verify_request.json;
    expect(req.x402Version).toBe(2);
    expect(req.paymentPayload.accepted).toEqual(req.paymentRequirements);
    for (const r of X.x402_vectors.verify_request.responses) expect(typeof r.isValid).toBe('boolean');
  });
});

// ── Section 1.9: well-formed for strict parsers ──────────────────────────────

describe('well-formedness and matching', () => {
  it('accepts the well-formed entries and refuses the private scheme and every malformed one', () => {
    for (const ok of X.well_formed.accepted) expect(isWellFormedRequirement(ok)).toBe(true);
    for (const bad of X.well_formed.rejected) expect(isWellFormedRequirement(bad)).toBe(false);
  });

  it('matches every field but extra, types included; the offered extra must be contained', () => {
    const offered: PaymentRequirement = X.well_formed.accepted[0];
    expect(requirementMatches(offered, { ...offered })).toBe(true);
    expect(requirementMatches(offered, { ...offered, extra: { added: 1 } })).toBe(true);
    expect(requirementMatches(offered, { ...offered, amount: '9999' })).toBe(false);
    expect(requirementMatches(offered, { ...offered, maxTimeoutSeconds: '60' })).toBe(false);
    const withExtra = { ...offered, extra: { name: 'USDC', version: '2' } };
    expect(requirementMatches(withExtra, { ...withExtra, extra: { name: 'USDC', version: '2', more: true } })).toBe(true);
    expect(requirementMatches(withExtra, { ...withExtra, extra: { name: 'USDC' } })).toBe(false);
  });
});

// ── The credits nonce ────────────────────────────────────────────────────────

describe('the credits nonce', () => {
  const N = X.credits_nonce;

  it('mints the fixture vectors and converts credits to nanocredits exactly', async () => {
    for (const v of N.vectors) {
      const nonce = await mintCreditsNonce(N.secret, v.resource, v.amount, { id: v.id, now: new Date((v.expires - 300) * 1000) });
      expect(nonce).toBe(v.nonce);
    }
    for (const c of N.credits_to_nanocredits) expect(creditsToNanocredits(c.credits)).toBe(c.nanocredits);
  });

  it('verifies its own, refuses another amount, another resource, a tampered mac and an expired one', async () => {
    const [a, b] = N.vectors;
    const now = new Date((a.expires - 10) * 1000);
    expect(await verifyCreditsNonce(N.secret, a.nonce, a.resource, a.amount, now)).toMatchObject({ ok: true, id: a.id });
    expect(await verifyCreditsNonce(N.secret, a.nonce, a.resource, b.amount, now)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
    expect(await verifyCreditsNonce(N.secret, a.nonce, `${a.resource}/other`, a.amount, now)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
    expect(await verifyCreditsNonce(N.secret, `${a.nonce.slice(0, -1)}0`, a.resource, a.amount, now)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
    expect(await verifyCreditsNonce(N.secret, a.nonce, a.resource, a.amount, new Date((a.expires + 1) * 1000))).toMatchObject({ ok: false, reason: 'nonce_expired' });
    expect(await verifyCreditsNonce(N.secret, 'garbage', a.resource, a.amount, now)).toMatchObject({ ok: false, reason: 'nonce_malformed' });
  });
});

// ── The paywall ──────────────────────────────────────────────────────────────

const URL_ = 'https://agent.example.com/agents/mini/quote';

function fakeCredits(overrides: Partial<CreditsClient> = {}): CreditsClient & { settles: unknown[]; verifies: unknown[] } {
  const client = {
    settles: [] as unknown[],
    verifies: [] as unknown[],
    async verifyToken(token: string, options?: unknown) {
      client.verifies.push({ token, options });
      return token === 'tok_valid' ? { valid: true, balance: 5 } : { valid: false, invalidReason: 'token_invalid' };
    },
    async settleToken(token: string, amountCredits: number, options: { idempotencyKey: string }) {
      client.settles.push({ token, amountCredits, ...options });
      return { success: true, charged: String(Math.round(amountCredits * 1e9)), chargedCredits: amountCredits };
    },
    ...overrides,
  };
  return client;
}

function fakeFacilitator(behaviour: { verify?: unknown; settle?: unknown } = {}): FacilitatorClient & { calls: string[] } {
  const f = {
    calls: [] as string[],
    async verify() {
      f.calls.push('verify');
      return (behaviour.verify as never) ?? { isValid: true, payer: '0x857b06519E91e3A54538791bDbb0E22373e36b66' };
    },
    async settle(_payload: unknown, requirement: PaymentRequirement) {
      f.calls.push(`settle:${requirement.amount}`);
      return (behaviour.settle as never) ?? { success: true, transaction: '0xabc', network: requirement.network, payer: '0x857b' };
    },
  };
  return f;
}

const CHAIN = {
  payTo: '0x209693Bc6afc0C5328bA36FaF03C514EF312287C',
  network: 'eip155:84532',
  asset: '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
  decimals: 6,
  extra: { name: 'USDC', version: '2' },
};

function paywall(opts: { credits?: CreditsClient; facilitator?: FacilitatorClient; schemes?: Array<'exact' | 'upto'>; log?: (e: string, f: Record<string, unknown>) => void } = {}) {
  return new Paywall({
    ...(opts.credits ? { credits: { client: opts.credits, nonceSecret: 's3cret', platformUrl: 'https://robutler.ai' } } : {}),
    ...(opts.facilitator ? { chain: { ...CHAIN, schemes: opts.schemes, facilitator: opts.facilitator } } : {}),
    resource: { serviceName: 'Mini', description: 'A quote' },
    log: opts.log,
  });
}

const ENDPOINT: PricedEndpoint = { path: '/quote', method: 'GET', pricing: { creditsPerCall: 0.01 }, description: 'A quote' };

function get(headers: Record<string, string> = {}): Request {
  return new Request(`${URL_}?q=1`, { method: 'GET', headers });
}

async function challenge(p: Paywall, endpoint = ENDPOINT): Promise<{ res: Response; v2: PaymentRequiredV2; v1: PaymentRequiredV1 }> {
  const res = await p.handle(endpoint, get(), async () => new Response('never'));
  const v2 = decodeBase64Json(res.headers.get(X402_HEADERS.required)) as unknown as PaymentRequiredV2;
  const v1 = (await res.json()) as PaymentRequiredV1;
  return { res, v2, v1 };
}

function v2Payment(accepted: PaymentRequirement, payload: Record<string, unknown>): Record<string, string> {
  return { [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted, payload }) };
}

describe('the 402', () => {
  it('carries the v2 header, the v1 body, no-store, and only well-formed entries', async () => {
    const p = paywall({ credits: fakeCredits(), facilitator: fakeFacilitator() });
    const { res, v2, v1 } = await challenge(p);
    expect(res.status).toBe(402);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(v2.x402Version).toBe(2);
    expect(v2.resource.url).toBe(URL_);
    expect(v2.resource.serviceName).toBe('Mini');
    expect(v2.accepts).toHaveLength(2);
    for (const entry of v2.accepts) expect(isWellFormedRequirement(entry)).toBe(true);
    const credits = v2.accepts.find((r) => r.scheme === CREDITS_SCHEME)!;
    expect(credits).toMatchObject({ network: CREDITS_NETWORK, amount: '10000000', asset: CREDITS_ASSET, payTo: CREDITS_PAY_TO, maxTimeoutSeconds: 300 });
    expect(typeof credits.extra?.nonce).toBe('string');
    const chain = v2.accepts.find((r) => r.scheme === 'exact')!;
    expect(chain).toMatchObject({ network: 'eip155:84532', amount: '10000', payTo: CHAIN.payTo, asset: CHAIN.asset, extra: CHAIN.extra });
    // v1: the same offers with `maxAmountRequired`, the resource folded in, the v1 network name.
    expect(v1.x402Version).toBe(1);
    expect(v1.error).toBeTruthy();
    expect(v1.accepts[1]).toMatchObject({ scheme: 'exact', network: 'base-sepolia', maxAmountRequired: '10000', resource: URL_, description: 'A quote' });
    expect(v1.accepts[0]).toMatchObject({ scheme: CREDITS_SCHEME, network: CREDITS_NETWORK, maxAmountRequired: '10000000' });
  });

  it('offers only credits without a chain seller, and a metered endpoint offers upto on chain when allowed', async () => {
    const creditsOnly = await challenge(paywall({ credits: fakeCredits() }));
    expect(creditsOnly.v2.accepts.map((r) => r.scheme)).toEqual([CREDITS_SCHEME]);
    const metered: PricedEndpoint = { ...ENDPOINT, pricing: { creditsPerCall: 0.001, lock: 0.05 } };
    const upto = await challenge(paywall({ credits: fakeCredits(), facilitator: fakeFacilitator(), schemes: ['exact', 'upto'] }), metered);
    expect(upto.v2.accepts.map((r) => r.scheme)).toEqual([CREDITS_SCHEME, 'upto']);
    expect(upto.v2.accepts[0].amount).toBe('50000000');
    expect(upto.v2.accepts[1].amount).toBe('50000');
    const exactOnly = await challenge(paywall({ credits: fakeCredits(), facilitator: fakeFacilitator() }), metered);
    expect(exactOnly.v2.accepts[1].scheme).toBe('exact');
  });

  it('carries the Bazaar declaration in v2 extensions and its output as v1 outputSchema', async () => {
    const discovered: PricedEndpoint = {
      ...ENDPOINT,
      discovery: { input: { type: 'http', method: 'GET', queryParams: { q: 'string' } }, output: { type: 'json', example: { price: 1 } } },
    };
    const { v2, v1 } = await challenge(paywall({ credits: fakeCredits() }), discovered);
    expect(v2.extensions?.bazaar).toEqual({ info: { input: { type: 'http', method: 'GET', queryParams: { q: 'string' } }, output: { type: 'json', example: { price: 1 } } } });
    expect(v1.accepts[0].outputSchema).toEqual({ type: 'json', example: { price: 1 } });
  });
});

describe('a credits payment', () => {
  it('verifies before the handler, settles once after it with the nonce key, and answers PAYMENT-RESPONSE with private caching', async () => {
    const credits = fakeCredits();
    const events: string[] = [];
    const p = paywall({ credits, log: (e) => events.push(e) });
    const { v2 } = await challenge(p);
    const entry = v2.accepts[0];
    const order: string[] = [];
    const res = await p.handle(ENDPOINT, get(v2Payment(entry, { token: 'tok_valid' })), async () => {
      order.push('handler');
      expect(credits.verifies).toHaveLength(1);
      expect(credits.settles).toHaveLength(0);
      return new Response(JSON.stringify({ price: 42 }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    expect(order).toEqual(['handler']);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ price: 42 });
    expect(res.headers.get('Cache-Control')).toBe('private');
    expect(credits.settles).toHaveLength(1);
    const settle = credits.settles[0] as { amountCredits: number; idempotencyKey: string; token: string; resource: string };
    expect(settle.token).toBe('tok_valid');
    expect(settle.amountCredits).toBeCloseTo(0.01, 12);
    expect(settle.resource).toBe(URL_);
    const nonceId = String(entry.extra!.nonce).split('.')[0];
    expect(settle.idempotencyKey).toBe(`settle:x402:${nonceId}`);
    const settled = decodeBase64Json(res.headers.get(X402_HEADERS.response))!;
    expect(settled).toMatchObject({ success: true, transaction: '', network: CREDITS_NETWORK, amount: '10000000' });
    expect(events).toContain('x402.settled');
  });

  it('a replay of the same request (the same nonce) is refused and settles nothing more', async () => {
    const credits = fakeCredits();
    const p = paywall({ credits });
    const { v2 } = await challenge(p);
    const headers = v2Payment(v2.accepts[0], { token: 'tok_valid' });
    const first = await p.handle(ENDPOINT, get(headers), async () => new Response('ok'));
    expect(first.status).toBe(200);
    const again = await p.handle(ENDPOINT, get(headers), async () => new Response('ok'));
    expect(again.status).toBe(402);
    expect((decodeBase64Json(again.headers.get(X402_HEADERS.required)) as { error: string }).error).toBe('nonce_used');
    expect(credits.settles).toHaveLength(1);
  });

  it('an invalid token, a tampered amount, or a nonce for another endpoint are refused before the handler', async () => {
    const credits = fakeCredits();
    const p = paywall({ credits });
    const { v2 } = await challenge(p);
    const entry = v2.accepts[0];
    const ran = vi.fn(async () => new Response('ok'));
    const bad = await p.handle(ENDPOINT, get(v2Payment(entry, { token: 'tok_bad' })), ran);
    expect(bad.status).toBe(402);
    expect((decodeBase64Json(bad.headers.get(X402_HEADERS.required)) as { error: string }).error).toBe('token_invalid');
    const tampered = await p.handle(ENDPOINT, get(v2Payment({ ...entry, amount: '1' }, { token: 'tok_valid' })), ran);
    expect(tampered.status).toBe(402);
    const other = { ...ENDPOINT, path: '/other' };
    const elsewhere = await p.handle(other, new Request('https://agent.example.com/agents/mini/other', { headers: v2Payment(entry, { token: 'tok_valid' }) }), ran);
    expect(elsewhere.status).toBe(402);
    expect(ran).not.toHaveBeenCalled();
    expect(credits.settles).toHaveLength(0);
  });

  it('a handler error settles nothing and passes through without a payment response', async () => {
    const credits = fakeCredits();
    const p = paywall({ credits });
    const { v2 } = await challenge(p);
    const res = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0], { token: 'tok_valid' })), async () => new Response('boom', { status: 500 }));
    expect(res.status).toBe(500);
    expect(res.headers.get(X402_HEADERS.response)).toBeNull();
    expect(credits.settles).toHaveLength(0);
  });

  it('a failed settle answers 402 with the failed response in the header and {} as the body', async () => {
    const credits = fakeCredits({ async settleToken() { return { success: false, error: 'lock refused' }; } });
    const p = paywall({ credits });
    const { v2 } = await challenge(p);
    const res = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0], { token: 'tok_valid' })), async () => new Response('ok'));
    expect(res.status).toBe(402);
    expect(await res.text()).toBe('{}');
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.response))).toMatchObject({ success: false, errorReason: 'lock refused' });
  });

  it('a metered endpoint settles the handler\'s Settlement-Overrides, clamped to the maximum, and strips the header', async () => {
    const credits = fakeCredits();
    const p = paywall({ credits });
    const metered: PricedEndpoint = { ...ENDPOINT, pricing: { creditsPerCall: 0.001, lock: 0.05 } };
    const { v2 } = await challenge(p, metered);
    const res = await p.handle(metered, get(v2Payment(v2.accepts[0], { token: 'tok_valid' })), async () =>
      new Response('ok', { headers: { [X402_HEADERS.settlementOverrides]: JSON.stringify({ credits: '0.002' }) } }));
    expect(res.status).toBe(200);
    expect(res.headers.get(X402_HEADERS.settlementOverrides)).toBeNull();
    expect((credits.settles[0] as { amountCredits: number }).amountCredits).toBeCloseTo(0.002, 12);
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.response))).toMatchObject({ amount: '2000000' });

    const { v2: again } = await challenge(p, metered);
    await p.handle(metered, get(v2Payment(again.accepts[0], { token: 'tok_valid' })), async () =>
      new Response('ok', { headers: { [X402_HEADERS.settlementOverrides]: JSON.stringify({ credits: '9' }) } }));
    expect((credits.settles[1] as { amountCredits: number }).amountCredits).toBeCloseTo(0.05, 12);
  });

  it('v1: X-PAYMENT names the scheme and carries the nonce; the answer is X-PAYMENT-RESPONSE', async () => {
    const credits = fakeCredits();
    const p = paywall({ credits });
    const { v1 } = await challenge(p);
    const entry = v1.accepts[0];
    const headers = { [X402_HEADERS.v1Payment]: encodeBase64Json({ x402Version: 1, scheme: entry.scheme, network: entry.network, payload: { token: 'tok_valid', nonce: entry.extra!.nonce } }) };
    const res = await p.handle(ENDPOINT, get(headers), async () => new Response('ok'));
    expect(res.status).toBe(200);
    expect(res.headers.get(X402_HEADERS.response)).toBeNull();
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.v1Response))).toMatchObject({ success: true, network: CREDITS_NETWORK });
    expect(credits.settles).toHaveLength(1);
  });

  it('a malformed payment header is a 400, not a challenge', async () => {
    const p = paywall({ credits: fakeCredits() });
    const res = await p.handle(ENDPOINT, get({ [X402_HEADERS.signature]: 'not-base64!' }), async () => new Response('ok'));
    expect(res.status).toBe(400);
    expect((await res.json()).error.code).toBe('invalid_payment');
  });
});

describe('a chain payment through a local facilitator', () => {
  it('verifies before the handler, settles after it, and answers the facilitator\'s response', async () => {
    const facilitator = fakeFacilitator();
    const p = paywall({ credits: fakeCredits(), facilitator });
    const { v2 } = await challenge(p);
    const entry = v2.accepts.find((r) => r.scheme === 'exact')!;
    const payload = X.x402_vectors.payment_signature_v2.json.payload;
    const res = await p.handle(ENDPOINT, get(v2Payment(entry, payload)), async () => {
      expect(facilitator.calls).toEqual(['verify']);
      return new Response('ok');
    });
    expect(res.status).toBe(200);
    expect(facilitator.calls).toEqual(['verify', 'settle:10000']);
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.response))).toMatchObject({ success: true, transaction: '0xabc', network: 'eip155:84532' });
    expect(res.headers.get('Cache-Control')).toBe('private');
  });

  it('a handler error settles nothing; a verify failure answers the reason in a fresh 402', async () => {
    const facilitator = fakeFacilitator();
    const p = paywall({ facilitator });
    const { v2 } = await challenge(p);
    const entry = v2.accepts[0];
    const failed = await p.handle(ENDPOINT, get(v2Payment(entry, { signature: '0x' })), async () => new Response('no', { status: 503 }));
    expect(failed.status).toBe(503);
    expect(facilitator.calls).toEqual(['verify']);

    const refusing = fakeFacilitator({ verify: { isValid: false, invalidReason: 'insufficient_funds' } });
    const q = paywall({ facilitator: refusing });
    const { v2: offers } = await challenge(q);
    const ran = vi.fn(async () => new Response('ok'));
    const res = await q.handle(ENDPOINT, get(v2Payment(offers.accepts[0], { signature: '0x' })), ran);
    expect(res.status).toBe(402);
    expect((decodeBase64Json(res.headers.get(X402_HEADERS.required)) as { error: string }).error).toBe('insufficient_funds');
    expect(ran).not.toHaveBeenCalled();
    expect(refusing.calls).toEqual(['verify']);
  });

  it('settlement_pending is delivered with its transaction and never settled twice', async () => {
    const facilitator = fakeFacilitator({ settle: { success: false, errorReason: 'settlement_pending', transaction: '0xpending', network: 'eip155:84532' } });
    const events: string[] = [];
    const p = paywall({ facilitator, log: (e) => events.push(e) });
    const { v2 } = await challenge(p);
    const res = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0], { signature: '0x' })), async () => new Response('ok'));
    expect(res.status).toBe(200);
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.response))).toMatchObject({ success: false, errorReason: 'settlement_pending', transaction: '0xpending' });
    expect(facilitator.calls.filter((c) => c.startsWith('settle'))).toHaveLength(1);
    expect(events).toContain('x402.settlement_pending');
  });

  it('upto: the handler\'s override is what the facilitator settles, at most the signed maximum', async () => {
    const facilitator = fakeFacilitator();
    const p = paywall({ facilitator, schemes: ['exact', 'upto'] });
    const metered: PricedEndpoint = { ...ENDPOINT, pricing: { creditsPerCall: 0.001, lock: 0.05 } };
    const { v2 } = await challenge(p, metered);
    expect(v2.accepts[0].scheme).toBe('upto');
    await p.handle(metered, get(v2Payment(v2.accepts[0], { signature: '0x' })), async () =>
      new Response('ok', { headers: { [X402_HEADERS.settlementOverrides]: JSON.stringify({ amount: '500' }) } }));
    expect(facilitator.calls).toEqual(['verify', 'settle:500']);
    const { v2: again } = await challenge(p, metered);
    await p.handle(metered, get(v2Payment(again.accepts[0], { signature: '0x' })), async () =>
      new Response('ok', { headers: { [X402_HEADERS.settlementOverrides]: JSON.stringify({ amount: '999999' }) } }));
    expect(facilitator.calls[3]).toBe('settle:50000');
  });

  it('v1: a base-sepolia payload is matched to the eip155:84532 offer and verified with the CAIP-2 name', async () => {
    const seen: unknown[] = [];
    const facilitator: FacilitatorClient = {
      async verify(payload) { seen.push(payload); return { isValid: true }; },
      async settle(_p, r) { return { success: true, transaction: '0x1', network: r.network }; },
    };
    const p = paywall({ facilitator });
    const { v1 } = await challenge(p);
    expect(v1.accepts[0].network).toBe('base-sepolia');
    const res = await p.handle(ENDPOINT, get({ [X402_HEADERS.v1Payment]: encodeBase64Json({ x402Version: 1, scheme: 'exact', network: 'base-sepolia', payload: { signature: '0x' } }) }), async () => new Response('ok'));
    expect(res.status).toBe(200);
    expect((seen[0] as { network: string }).network).toBe('eip155:84532');
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.v1Response))).toMatchObject({ success: true });
  });
});

describe('CreditsScheme.settleKey', () => {
  it('is the fixture\'s fresh x402 shape', () => {
    expect(CreditsScheme.settleKey('3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c')).toBe('settle:x402:3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c');
  });
});
