/**
 * MPP alongside x402 on a priced endpoint (webagents gap-closure plan 2.6,
 * spec pack section 2; 2026-09-26): the stateless challenge ids against the
 * section 2.7 vectors (the CURRENT slot order, the IETF-text order refused),
 * the challenge on the same 402 as the x402 offers, the credential's
 * binding, expiry and single use, the `stripe` method against a Stripe
 * double (a receipt on success, a fresh challenge otherwise), the
 * `robutler` method through the credits scheme, and the account rule: the
 * credits method is only offered beside an account-less one.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MppSeller,
  MPP_METHOD_RE,
  MPP_METHOD_ROBUTLER,
  MPP_METHOD_STRIPE,
  challengeHmacInput,
  computeChallengeId,
  formatChallenge,
  readMppCredential,
  type StripeSptClient,
} from '../../../../src/skills/payments/mpp-seller.js';
import { MPP_INTENT_CHARGE, MPP_PROBLEM_BASE, MPP_RECEIPT_HEADER, base64urlDecode, base64urlEncode, parseWwwAuthenticatePayment } from '../../../../src/skills/payments/mpp-buyer.js';
import { Paywall, type PricedEndpoint } from '../../../../src/skills/payments/paywall.js';
import { X402_HEADERS, decodeBase64Json, encodeBase64Json } from '../../../../src/skills/payments/x402-wire.js';
import type { CreditsClient } from '../../../../src/skills/payments/x402-credits.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/payments/paywall_x402_mpp.json'), 'utf8'));
const M = FIXTURE.mpp;

describe('the fixture pins the names', () => {
  it('headers, methods, intent, problem base, and the method id rule', () => {
    expect(MPP_RECEIPT_HEADER).toBe(M.headers.receipt);
    expect(MPP_METHOD_STRIPE).toBe(M.methods.stripe);
    expect(MPP_METHOD_ROBUTLER).toBe(M.methods.credits);
    expect(MPP_INTENT_CHARGE).toBe(M.intent);
    expect(MPP_PROBLEM_BASE).toBe(M.problem_base);
    expect(MPP_METHOD_RE.source).toBe(new RegExp(M.method_id_pattern).source);
    for (const bad of M.invalid_method_ids) expect(MPP_METHOD_RE.test(bad)).toBe(false);
    expect(MPP_METHOD_RE.test(MPP_METHOD_ROBUTLER)).toBe(true);
  });
});

describe('section 2.7: stateless ids in the current slot order', () => {
  const V = M.id_vectors;
  const base = { realm: V.realm, method: V.method, intent: V.intent, request: V.request };

  it('encodes the request and the opaque as JCS base64url without padding', () => {
    expect(base64urlEncode(JSON.stringify(V.request_json))).toBe(V.request);
    expect(base64urlEncode(JSON.stringify(V.opaque_json))).toBe(V.opaque);
  });

  it('lays the slots out as the fixture prints them, the header before opaque', () => {
    expect(challengeHmacInput(base)).toBe(V.hmac_input_plain);
    expect(challengeHmacInput({ ...base, header: 'Payment-Authorization' })).toBe(V.hmac_input_with_header);
    expect(challengeHmacInput({ ...base, header: 'Payment-Authorization', opaque: V.opaque })).toBe(V.hmac_input_with_header_and_opaque);
  });

  it('computes all three ids, and never the IETF-text-order one', async () => {
    expect(await computeChallengeId(V.secret, base)).toBe(V.plain);
    expect(await computeChallengeId(V.secret, { ...base, header: 'Payment-Authorization' })).toBe(V.with_header);
    const both = await computeChallengeId(V.secret, { ...base, header: 'Payment-Authorization', opaque: V.opaque });
    expect(both).toBe(V.with_header_and_opaque);
    expect(both).not.toBe(V.ietf_text_order_with_header_and_opaque_NOT_ours);
  });

  it('a formatted challenge round-trips through the buyer\'s parser', () => {
    const fields = { id: V.plain, ...base, expires: '2026-09-26T12:00:00Z', opaque: V.opaque };
    const parsed = parseWwwAuthenticatePayment(formatChallenge(fields));
    expect(parsed).toEqual(fields);
  });
});

describe('the Stripe end-to-end shapes', () => {
  it('the receipt encodes to the fixture bytes', () => {
    const S = M.stripe_e2e;
    expect(base64urlEncode(JSON.stringify(S.receipt_json))).toBe(S.receipt_base64url);
    expect(JSON.parse(new TextDecoder().decode(base64urlDecode(S.receipt_base64url)!))).toEqual(S.receipt_json);
  });
});

// ── The paywall with MPP ─────────────────────────────────────────────────────

const URL_ = 'https://agent.example.com/agents/mini/quote';
/** Priced at 50 credits: 5000 cents on the Stripe side, the spec pack's own e2e amount. */
const ENDPOINT: PricedEndpoint = { path: '/quote', method: 'GET', pricing: { creditsPerCall: 50 }, description: 'AI generation' };

function stripeDouble(status = 'succeeded'): StripeSptClient & { calls: Array<{ params: unknown; options: unknown }> } {
  const d = {
    calls: [] as Array<{ params: unknown; options: unknown }>,
    async createPaymentIntent(params: unknown, options: unknown) {
      d.calls.push({ params, options });
      return { id: 'pi_1N4Zv32eZvKYlo2CPhVPkJlW', status };
    },
  };
  return d;
}

function creditsDouble(): CreditsClient & { settles: unknown[] } {
  const c = {
    settles: [] as unknown[],
    async verifyToken(token: string) { return token === 'tok_valid' ? { valid: true, balance: 90 } : { valid: false, invalidReason: 'token_invalid' }; },
    async settleToken(token: string, amountCredits: number, options: { idempotencyKey: string }) {
      c.settles.push({ token, amountCredits, ...options });
      return { success: true, charged: '1' };
    },
  };
  return c;
}

const NOW = new Date('2026-09-26T12:00:00Z');

function build(opts: { stripe?: StripeSptClient; credits?: CreditsClient; chain?: boolean; now?: () => Date; mppCredits?: boolean } = {}) {
  const now = opts.now ?? (() => NOW);
  const mpp = new MppSeller({
    realm: 'agent.example.com',
    secret: 'mpp-secret',
    stripe: opts.stripe ? { profileId: 'profile_test_123', client: opts.stripe } : undefined,
    credits: opts.mppCredits,
    now,
  });
  return new Paywall({
    ...(opts.credits ? { credits: { client: opts.credits, nonceSecret: 's', platformUrl: 'https://robutler.ai', now } } : {}),
    ...(opts.chain
      ? { chain: { payTo: '0x1', network: 'eip155:84532', asset: '0x2', decimals: 6, facilitator: { async verify() { return { isValid: true }; }, async settle(_p: unknown, r: { network: string }) { return { success: true, transaction: '0x', network: r.network }; } } } }
      : {}),
    mpp,
  });
}

function get(headers: Record<string, string> = {}): Request {
  return new Request(`${URL_}?x=1`, { headers });
}

async function challenges(p: Paywall): Promise<{ res: Response; byMethod: Record<string, ReturnType<typeof parseWwwAuthenticatePayment>> }> {
  const res = await p.handle(ENDPOINT, get(), async () => new Response('never'));
  const raw = res.headers.get('WWW-Authenticate') ?? '';
  // A Fetch Headers object joins repeated fields with ", "; split at each scheme.
  const values = raw.split(/,\s*(?=Payment\s)/).filter(Boolean);
  const byMethod: Record<string, ReturnType<typeof parseWwwAuthenticatePayment>> = {};
  for (const v of values) {
    const parsed = parseWwwAuthenticatePayment(v);
    if (parsed) byMethod[parsed.method] = parsed;
  }
  return { res, byMethod };
}

function credentialHeader(challenge: NonNullable<ReturnType<typeof parseWwwAuthenticatePayment>>, payload: Record<string, unknown>): Record<string, string> {
  return { Authorization: `Payment ${base64urlEncode(JSON.stringify({ challenge, payload }))}` };
}

function receiptOf(res: Response): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(base64urlDecode(res.headers.get(MPP_RECEIPT_HEADER)!)!));
}

describe('the 402 with MPP', () => {
  it('carries a Payment challenge per method beside the x402 offers, no-store', async () => {
    const p = build({ stripe: stripeDouble(), credits: creditsDouble() });
    const { res, byMethod } = await challenges(p);
    expect(res.status).toBe(402);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    expect(res.headers.get(X402_HEADERS.required)).toBeTruthy();
    expect(Object.keys(byMethod).sort()).toEqual(['robutler', 'stripe']);
    const stripe = byMethod.stripe!;
    expect(stripe.realm).toBe('agent.example.com');
    expect(stripe.intent).toBe('charge');
    expect(stripe.expires).toBe('2026-09-26T12:05:00Z');
    const request = JSON.parse(new TextDecoder().decode(base64urlDecode(stripe.request)!));
    expect(request).toEqual({ amount: '5000', currency: 'usd', description: 'AI generation', methodDetails: { networkId: 'profile_test_123', paymentMethodTypes: ['card', 'link'] } });
    expect(JSON.parse(new TextDecoder().decode(base64urlDecode(stripe.opaque!)!))).toEqual({ resource: URL_ });
    // The robutler challenge carries the same nonce the x402 credits entry does.
    const v2 = decodeBase64Json(res.headers.get(X402_HEADERS.required)) as { accepts: Array<{ extra?: { nonce?: string } }> };
    const robutlerRequest = JSON.parse(new TextDecoder().decode(base64urlDecode(byMethod.robutler!.request)!));
    expect(robutlerRequest.methodDetails.nonce).toBe(v2.accepts[0].extra!.nonce);
    expect(robutlerRequest.currency).toBe('credits');
  });

  it('never offers the account-backed method alone: no Stripe and no chain means no robutler challenge', async () => {
    const alone = build({ credits: creditsDouble() });
    expect(Object.keys((await challenges(alone)).byMethod)).toEqual([]);
    const besideChain = build({ credits: creditsDouble(), chain: true });
    expect(Object.keys((await challenges(besideChain)).byMethod)).toEqual(['robutler']);
  });
});

describe('the stripe method against a Stripe double', () => {
  it('runs the handler, creates the PaymentIntent under the challenge idempotency key, and answers a receipt', async () => {
    const stripe = stripeDouble();
    const p = build({ stripe, credits: creditsDouble() });
    const { byMethod } = await challenges(p);
    const challenge = byMethod.stripe!;
    const res = await p.handle(ENDPOINT, get(credentialHeader(challenge, { spt: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW' })), async () => {
      expect(stripe.calls).toHaveLength(0);
      return new Response('image');
    });
    expect(res.status).toBe(200);
    expect(res.headers.get('Cache-Control')).toBe('private');
    expect(stripe.calls).toHaveLength(1);
    expect(stripe.calls[0].params).toEqual({
      amount: 5000,
      currency: 'usd',
      shared_payment_granted_token: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW',
      confirm: true,
      automatic_payment_methods: { enabled: true, allow_redirects: 'never' },
      metadata: { challenge_id: challenge.id },
    });
    expect(stripe.calls[0].options).toEqual({ idempotencyKey: `${challenge.id}_spt_1N4Zv32eZvKYlo2CPhVPkJlW` });
    expect(receiptOf(res)).toEqual({ status: 'success', method: 'stripe', timestamp: '2026-09-26T12:00:00Z', reference: 'pi_1N4Zv32eZvKYlo2CPhVPkJlW' });
  });

  it('a credential is single use, a tampered id or realm is invalid-challenge, and an expired challenge is payment-expired', async () => {
    const stripe = stripeDouble();
    let now = NOW;
    const p = build({ stripe, credits: creditsDouble(), now: () => now });
    const { byMethod } = await challenges(p);
    const challenge = byMethod.stripe!;
    const headers = credentialHeader(challenge, { spt: 'spt_1' });
    expect((await p.handle(ENDPOINT, get(headers), async () => new Response('ok'))).status).toBe(200);
    const replay = await p.handle(ENDPOINT, get(headers), async () => new Response('ok'));
    expect(replay.status).toBe(402);
    expect(replay.headers.get('Content-Type')).toBe('application/problem+json');
    expect((await replay.json()).type).toBe(`${MPP_PROBLEM_BASE}invalid-challenge`);
    expect(replay.headers.get('WWW-Authenticate')).toContain('Payment ');
    expect(stripe.calls).toHaveLength(1);

    const tampered = await p.handle(ENDPOINT, get(credentialHeader({ ...challenge, id: 'AAAA' }, { spt: 'spt_2' })), async () => new Response('ok'));
    expect((await tampered.json()).type).toBe(`${MPP_PROBLEM_BASE}invalid-challenge`);
    const elsewhere = await p.handle(ENDPOINT, get(credentialHeader({ ...challenge, realm: 'other.example' }, { spt: 'spt_2' })), async () => new Response('ok'));
    expect((await elsewhere.json()).type).toBe(`${MPP_PROBLEM_BASE}invalid-challenge`);

    const { byMethod: fresh } = await challenges(p);
    now = new Date('2026-09-26T12:06:00Z');
    const expired = await p.handle(ENDPOINT, get(credentialHeader(fresh.stripe!, { spt: 'spt_3' })), async () => new Response('ok'));
    expect((await expired.json()).type).toBe(`${MPP_PROBLEM_BASE}payment-expired`);
    expect(stripe.calls).toHaveLength(1);
  });

  it('a PaymentIntent that did not succeed answers a fresh challenge with verification-failed and no receipt', async () => {
    const p = build({ stripe: stripeDouble('requires_action'), credits: creditsDouble() });
    const { byMethod } = await challenges(p);
    const res = await p.handle(ENDPOINT, get(credentialHeader(byMethod.stripe!, { spt: 'spt_9' })), async () => new Response('ok'));
    expect(res.status).toBe(402);
    expect((await res.json()).type).toBe(`${MPP_PROBLEM_BASE}verification-failed`);
    expect(res.headers.get(MPP_RECEIPT_HEADER)).toBeNull();
  });

  it('a malformed credential, an unknown method, and two payments on one request', async () => {
    const p = build({ stripe: stripeDouble(), credits: creditsDouble() });
    const malformed = await p.handle(ENDPOINT, get({ Authorization: 'Payment not-base64!' }), async () => new Response('ok'));
    expect(malformed.status).toBe(402);
    expect((await malformed.json()).type).toBe(`${MPP_PROBLEM_BASE}malformed-credential`);
    const { byMethod } = await challenges(p);
    const unknown = await p.handle(ENDPOINT, get(credentialHeader({ ...byMethod.stripe!, method: 'tempo' }, { type: 'hash', hash: '0x' })), async () => new Response('ok'));
    expect(unknown.status).toBe(400);
    expect((await unknown.json()).type).toBe(`${MPP_PROBLEM_BASE}method-unsupported`);
    const both = await p.handle(ENDPOINT, get({ ...credentialHeader(byMethod.stripe!, { spt: 'spt_1' }), [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted: {}, payload: {} }) }), async () => new Response('ok'));
    expect(both.status).toBe(400);
    // A bearer that is not a Payment credential is not read as one.
    const bearer = await p.handle(ENDPOINT, get({ Authorization: 'Bearer abc' }), async () => new Response('ok'));
    expect(bearer.status).toBe(402);
    expect(bearer.headers.get('Content-Type')).toBe('application/json');
  });
});

describe('the robutler method', () => {
  it('settles through the credits scheme with the challenge nonce, once, and answers a receipt naming the settle key', async () => {
    const credits = creditsDouble();
    const p = build({ stripe: stripeDouble(), credits });
    const { byMethod } = await challenges(p);
    const challenge = byMethod.robutler!;
    const res = await p.handle(ENDPOINT, get(credentialHeader(challenge, { token: 'tok_valid' })), async () => new Response('ok'));
    expect(res.status).toBe(200);
    expect(credits.settles).toHaveLength(1);
    const settle = credits.settles[0] as { amountCredits: number; idempotencyKey: string };
    expect(settle.amountCredits).toBeCloseTo(50, 12);
    const receipt = receiptOf(res);
    expect(receipt).toMatchObject({ status: 'success', method: 'robutler', reference: settle.idempotencyKey });
    const replay = await p.handle(ENDPOINT, get(credentialHeader(challenge, { token: 'tok_valid' })), async () => new Response('ok'));
    expect(replay.status).toBe(402);
    expect((await replay.json()).type).toBe(`${MPP_PROBLEM_BASE}invalid-challenge`);
    expect(credits.settles).toHaveLength(1);
  });

  it('an invalid token is verification-failed before the handler runs', async () => {
    const credits = creditsDouble();
    const p = build({ stripe: stripeDouble(), credits });
    const { byMethod } = await challenges(p);
    let ran = false;
    const res = await p.handle(ENDPOINT, get(credentialHeader(byMethod.robutler!, { token: 'tok_bad' })), async () => { ran = true; return new Response('ok'); });
    expect(res.status).toBe(402);
    expect((await res.json()).type).toBe(`${MPP_PROBLEM_BASE}verification-failed`);
    expect(ran).toBe(false);
  });
});

describe('readMppCredential', () => {
  it('reads Authorization: Payment and Payment-Authorization, and nothing else', () => {
    const cred = base64urlEncode(JSON.stringify({ challenge: { id: 'i', realm: 'r', method: 'stripe', intent: 'charge', request: 'e30' }, payload: { spt: 'spt_1' } }));
    expect(readMppCredential(new Headers({ Authorization: `Payment ${cred}` }))).toMatchObject({ ok: true });
    expect(readMppCredential(new Headers({ 'Payment-Authorization': `Payment ${cred}` }))).toMatchObject({ ok: true });
    expect(readMppCredential(new Headers({ Authorization: 'Bearer x' }))).toBeNull();
    expect(readMppCredential(new Headers())).toBeNull();
    expect(readMppCredential(new Headers({ Authorization: `Payment ${base64urlEncode('{"payload":{}}')}` }))).toMatchObject({ ok: false });
  });
});
