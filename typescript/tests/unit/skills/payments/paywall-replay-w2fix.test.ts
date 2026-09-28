/**
 * S-297 (2026-09-26): a replayed x402 or MPP payment never runs the handler
 * again for free, and a nonce is bound to the request the 402 answered.
 * The twin of the Python `test_paywall_replay_w2fix.py`; the fixture
 * `python/tests/fixtures/payments/paywall_x402_mpp.json` pins the binding
 * format and the bound nonce vectors for both SDKs.
 *
 *   - the request binding: `METHOD|path|query|sha256(body)`, and the bound
 *     nonce vectors mint and verify (an unbound verify of a bound nonce is
 *     `nonce_not_ours`, and the other way round);
 *   - a client with a shared store claims the nonce there before the handler
 *     runs: a second process (a second Paywall over the same store) refuses
 *     the replay, and an invalid token burns nothing;
 *   - a settle the platform answers `replayed` is withheld: 402 and `{}`;
 *   - a retry with another query or another body is `nonce_not_ours`;
 *   - the MPP `robutler` method is withheld the same way, and the Stripe
 *     method's `claimChallenge` hook refuses a second process.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CreditsScheme, mintCreditsNonce, nonceMessage, verifyCreditsNonce, type CreditsClient, type NonceClaim } from '../../../../src/skills/payments/x402-credits';
import { Paywall, requestBinding, type PricedEndpoint } from '../../../../src/skills/payments/paywall';
import { MppSeller, type StripeSptClient } from '../../../../src/skills/payments/mpp-seller';
import { parseWwwAuthenticatePayment, base64urlEncode } from '../../../../src/skills/payments/mpp-buyer';
import { X402_HEADERS, decodeBase64Json, encodeBase64Json, type PaymentRequiredV2, type PaymentRequirement } from '../../../../src/skills/payments/x402-wire';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/payments/paywall_x402_mpp.json'), 'utf8'));
const N = FIXTURE.x402.credits_nonce;
const URL_ = 'https://agent.example.com/agents/mini/quote';
const ENDPOINT: PricedEndpoint = { path: '/quote', method: 'GET', pricing: { creditsPerCall: 0.01 }, description: 'A quote' };
const POST_ENDPOINT: PricedEndpoint = { ...ENDPOINT, method: 'POST' };

describe('the request binding and the bound nonce (fixture vectors)', () => {
  it('binds the method, the path, the query and the body digest, exactly as the fixture prints it', async () => {
    const R = N.request_binding;
    const posted = new Request(`https://agent.example.com${R.path}?${R.query}`, { method: R.method, body: R.body, headers: { 'content-type': 'application/json' } });
    expect(await requestBinding(posted)).toBe(R.binding);
    // Read from a clone: the body is still the handler's to read.
    expect(await posted.text()).toBe(R.body);
    expect(await requestBinding(new Request(`https://agent.example.com${R.path}?${R.query}`))).toBe(R.get_binding);
    expect(await requestBinding(new Request(`https://agent.example.com${R.path}`))).toBe(`GET|${R.path}||${R.get_binding.split('|')[3]}`);
  });

  it('mints and verifies the bound vectors; bound and unbound do not verify each other', async () => {
    for (const v of N.bound_vectors) {
      expect(nonceMessage(v.resource, v.amount, v.id, v.expires, v.binding)).toBe(v.message);
      expect(await mintCreditsNonce(N.secret, v.resource, v.amount, { id: v.id, now: new Date((v.expires - 300) * 1000), binding: v.binding })).toBe(v.nonce);
      const now = new Date((v.expires - 10) * 1000);
      expect(await verifyCreditsNonce(N.secret, v.nonce, v.resource, v.amount, now, v.binding)).toMatchObject({ ok: true, id: v.id });
      expect(await verifyCreditsNonce(N.secret, v.nonce, v.resource, v.amount, now)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
      expect(await verifyCreditsNonce(N.secret, v.nonce, v.resource, v.amount, now, `${v.binding}x`)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
    }
    // The unbound vectors still hold, and an unbound nonce does not verify with a binding.
    const [a] = N.vectors;
    expect(await verifyCreditsNonce(N.secret, a.nonce, a.resource, a.amount, new Date((a.expires - 10) * 1000), N.bound_vectors[1].binding)).toMatchObject({ ok: false, reason: 'nonce_not_ours' });
  });
});

/** A shared store two "processes" share, and a credits client over it. */
function sharedStore() {
  const claimed = new Set<string>();
  const claims: NonceClaim[] = [];
  return { claimed, claims };
}

function fakeCredits(store?: ReturnType<typeof sharedStore>, settle?: () => Record<string, unknown>): CreditsClient & { settles: unknown[]; verifies: unknown[] } {
  const client = {
    settles: [] as unknown[],
    verifies: [] as unknown[],
    async verifyToken(token: string) {
      client.verifies.push(token);
      // Enough for the 50-credit MPP endpoint below as well as the 0.01-credit one.
      return token === 'tok_valid' ? { valid: true, balance: 90 } : { valid: false, invalidReason: 'token_invalid' };
    },
    async settleToken(token: string, amountCredits: number, options: { idempotencyKey: string }) {
      client.settles.push({ token, amountCredits, ...options });
      return settle ? settle() : { success: true, charged: String(Math.round(amountCredits * 1e9)), chargedCredits: amountCredits };
    },
    ...(store
      ? {
          async claimNonce(claim: NonceClaim) {
            store.claims.push(claim);
            if (store.claimed.has(claim.nonceId)) return false;
            store.claimed.add(claim.nonceId);
            return true;
          },
        }
      : {}),
  };
  return client;
}

function paywall(credits: CreditsClient, events: string[] = []) {
  return new Paywall({ credits: { client: credits, nonceSecret: 'fleet', platformUrl: 'https://robutler.ai' }, resource: { serviceName: 'Mini' }, log: (e) => events.push(e) });
}

function get(headers: Record<string, string> = {}, url = `${URL_}?q=1`): Request {
  return new Request(url, { headers });
}

function post(body: string, headers: Record<string, string> = {}): Request {
  return new Request(`${URL_}?q=1`, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body });
}

async function challenge(p: Paywall, request: Request, endpoint = ENDPOINT): Promise<PaymentRequiredV2> {
  const res = await p.handle(endpoint, request, async () => new Response('never'));
  expect(res.status).toBe(402);
  return decodeBase64Json(res.headers.get(X402_HEADERS.required)) as unknown as PaymentRequiredV2;
}

function v2Payment(accepted: PaymentRequirement, token = 'tok_valid'): Record<string, string> {
  return { [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted, payload: { token } }) };
}

function refusal(res: Response): string {
  return (decodeBase64Json(res.headers.get(X402_HEADERS.required)) as { error: string }).error;
}

describe('single use across processes', () => {
  it('a client with a shared store claims the nonce there before the handler runs, and a second process refuses the replay', async () => {
    const store = sharedStore();
    const podA = paywall(fakeCredits(store));
    const podB = paywall(fakeCredits(store));
    const v2 = await challenge(podA, get());
    let ran = 0;
    const first = await podA.handle(ENDPOINT, get(v2Payment(v2.accepts[0])), async () => {
      ran += 1;
      expect(store.claims).toHaveLength(1);
      return new Response('ok');
    });
    expect(first.status).toBe(200);
    const nonceId = String(v2.accepts[0].extra!.nonce).split('.')[0];
    expect(store.claims[0]).toMatchObject({ nonceId, resource: URL_ });
    expect(store.claims[0].binding).toMatch(/^GET\|\/agents\/mini\/quote\|q=1\|[0-9a-f]{64}$/);

    // The same nonce secret and the same store: the other replica.
    const replay = await podB.handle(ENDPOINT, get(v2Payment(v2.accepts[0])), async () => { ran += 1; return new Response('ok'); });
    expect(replay.status).toBe(402);
    expect(refusal(replay)).toBe('nonce_used');
    expect(ran).toBe(1);
  });

  it('an invalid token burns nothing: the nonce is claimed only after the token verified', async () => {
    const store = sharedStore();
    const p = paywall(fakeCredits(store));
    const v2 = await challenge(p, get());
    const bad = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0], 'tok_bad')), async () => new Response('ok'));
    expect(bad.status).toBe(402);
    expect(store.claims).toHaveLength(0);
    const good = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0])), async () => new Response('ok'));
    expect(good.status).toBe(200);
  });

  it('a settle the platform answers replayed is withheld: 402 and {}, logged as a replay', async () => {
    const events: string[] = [];
    const credits = fakeCredits(undefined, () => ({ success: true, replayed: true, charged: '10000000' }));
    const p = paywall(credits, events);
    const v2 = await challenge(p, get());
    const res = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0])), async () => new Response('the answer'));
    expect(res.status).toBe(402);
    expect(await res.text()).toBe('{}');
    expect(decodeBase64Json(res.headers.get(X402_HEADERS.response))).toMatchObject({ success: false, errorReason: 'replayed' });
    expect(events).toContain('x402.replayed');
    expect(events).not.toContain('x402.settled');
  });
});

describe('bound to the request', () => {
  it('a retry with another query is nonce_not_ours and never runs', async () => {
    const p = paywall(fakeCredits());
    const v2 = await challenge(p, get());
    let ran = 0;
    const res = await p.handle(ENDPOINT, get(v2Payment(v2.accepts[0]), `${URL_}?q=2`), async () => { ran += 1; return new Response('ok'); });
    expect(res.status).toBe(402);
    expect(refusal(res)).toBe('nonce_not_ours');
    expect(ran).toBe(0);
  });

  it('a retry with another body is refused; the same body runs, and the handler still reads it', async () => {
    const p = paywall(fakeCredits());
    const body = JSON.stringify({ q: 'how much' });
    const v2 = await challenge(p, post(body), POST_ENDPOINT);
    let ran = 0;
    const changed = await p.handle(POST_ENDPOINT, post(JSON.stringify({ q: 'everything' }), v2Payment(v2.accepts[0])), async () => { ran += 1; return new Response('ok'); });
    expect(changed.status).toBe(402);
    expect(refusal(changed)).toBe('nonce_not_ours');
    const request = post(body, v2Payment(v2.accepts[0]));
    const same = await p.handle(POST_ENDPOINT, request, async () => {
      ran += 1;
      expect(await request.text()).toBe(body);
      return new Response('ok');
    });
    expect(same.status).toBe(200);
    expect(ran).toBe(1);
  });

  it('a v1 payment is bound the same way', async () => {
    const p = paywall(fakeCredits());
    const v2 = await challenge(p, get());
    const entry = v2.accepts[0];
    const v1 = { [X402_HEADERS.v1Payment]: encodeBase64Json({ x402Version: 1, scheme: entry.scheme, network: entry.network, payload: { token: 'tok_valid', nonce: entry.extra?.nonce } }) };
    let ran = 0;
    expect(refusal(await p.handle(ENDPOINT, get(v1, `${URL_}?q=other`), async () => { ran += 1; return new Response('ok'); }))).toBe('nonce_not_ours');
    expect((await p.handle(ENDPOINT, get(v1), async () => { ran += 1; return new Response('ok'); })).status).toBe(200);
    expect(ran).toBe(1);
  });
});

describe('MPP', () => {
  const NOW = new Date('2026-09-26T12:00:00Z');
  const MPP_ENDPOINT: PricedEndpoint = { path: '/quote', method: 'GET', pricing: { creditsPerCall: 50 }, description: 'AI generation' };

  function stripeDouble(): StripeSptClient & { calls: unknown[] } {
    const s = { calls: [] as unknown[], async createPaymentIntent(params: unknown, options: unknown) { s.calls.push({ params, options }); return { id: 'pi_1', status: 'succeeded' }; } };
    return s;
  }

  function build(opts: { credits: CreditsClient; stripe?: StripeSptClient; claimChallenge?: (c: { challengeId: string }) => Promise<boolean> }) {
    const mpp = new MppSeller({ realm: 'agent.example.com', secret: 'mpp-secret', stripe: opts.stripe ? { profileId: 'profile_test_123', client: opts.stripe } : undefined, claimChallenge: opts.claimChallenge, now: () => NOW });
    return new Paywall({ credits: { client: opts.credits, nonceSecret: 's', platformUrl: 'https://robutler.ai', now: () => NOW }, mpp });
  }

  async function challenges(p: Paywall): Promise<Record<string, NonNullable<ReturnType<typeof parseWwwAuthenticatePayment>>>> {
    const res = await p.handle(MPP_ENDPOINT, get(), async () => new Response('never'));
    const byMethod: Record<string, NonNullable<ReturnType<typeof parseWwwAuthenticatePayment>>> = {};
    for (const v of (res.headers.get('WWW-Authenticate') ?? '').split(/,\s*(?=Payment\s)/).filter(Boolean)) {
      const parsed = parseWwwAuthenticatePayment(v);
      if (parsed) byMethod[parsed.method] = parsed;
    }
    return byMethod;
  }

  function credential(challenge: NonNullable<ReturnType<typeof parseWwwAuthenticatePayment>>, payload: Record<string, unknown>): Record<string, string> {
    return { Authorization: `Payment ${base64urlEncode(JSON.stringify({ challenge, payload }))}` };
  }

  it('the robutler method: a replayed settle is withheld with a fresh challenge, never the answer', async () => {
    const credits = fakeCredits(undefined, () => ({ success: true, replayed: true, charged: '1' }));
    const p = build({ credits, stripe: stripeDouble() });
    const byMethod = await challenges(p);
    const res = await p.handle(MPP_ENDPOINT, get(credential(byMethod.robutler!, { token: 'tok_valid' })), async () => new Response('the answer'));
    expect(res.status).toBe(402);
    const body = await res.text();
    expect(body).not.toBe('the answer');
    expect(res.headers.get('Content-Type')).toContain('application/problem+json');
    expect(body).toContain('invalid-challenge');
  });

  it('the robutler method shares the credits nonce claim, so a second process refuses the replay before the handler', async () => {
    const store = sharedStore();
    const podA = build({ credits: fakeCredits(store), stripe: stripeDouble() });
    const podB = build({ credits: fakeCredits(store), stripe: stripeDouble() });
    const byMethod = await challenges(podA);
    let ran = 0;
    expect((await podA.handle(MPP_ENDPOINT, get(credential(byMethod.robutler!, { token: 'tok_valid' })), async () => { ran += 1; return new Response('ok'); })).status).toBe(200);
    const replay = await podB.handle(MPP_ENDPOINT, get(credential(byMethod.robutler!, { token: 'tok_valid' })), async () => { ran += 1; return new Response('ok'); });
    expect(replay.status).toBe(402);
    expect(ran).toBe(1);
  });

  it('the stripe method: the claimChallenge hook lets a fleet refuse a second process, and a single process still refuses its own replay', async () => {
    const claimed = new Set<string>();
    const claimChallenge = async (c: { challengeId: string }) => (claimed.has(c.challengeId) ? false : (claimed.add(c.challengeId), true));
    const stripeA = stripeDouble();
    const stripeB = stripeDouble();
    const podA = build({ credits: fakeCredits(), stripe: stripeA, claimChallenge });
    const podB = build({ credits: fakeCredits(), stripe: stripeB, claimChallenge });
    const byMethod = await challenges(podA);
    let ran = 0;
    const first = await podA.handle(MPP_ENDPOINT, get(credential(byMethod.stripe!, { spt: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW' })), async () => { ran += 1; return new Response('image'); });
    expect(first.status).toBe(200);
    expect(stripeA.calls).toHaveLength(1);
    const replay = await podB.handle(MPP_ENDPOINT, get(credential(byMethod.stripe!, { spt: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW' })), async () => { ran += 1; return new Response('image'); });
    expect(replay.status).toBe(402);
    expect(stripeB.calls).toHaveLength(0);
    expect(ran).toBe(1);
    // Without the hook, a single process still refuses its own replay (the per-process set).
    const alone = build({ credits: fakeCredits(), stripe: stripeDouble() });
    const ch = await challenges(alone);
    expect((await alone.handle(MPP_ENDPOINT, get(credential(ch.stripe!, { spt: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW' })), async () => new Response('image'))).status).toBe(200);
    expect((await alone.handle(MPP_ENDPOINT, get(credential(ch.stripe!, { spt: 'spt_1N4Zv32eZvKYlo2CPhVPkJlW' })), async () => new Response('image'))).status).toBe(402);
  });
});
