/**
 * A priced `@http` endpoint served end to end (webagents gap-closure plan
 * 2.6, 2026-09-26): `@pricing` stacked on `@http` reaches the registry as
 * `pricing`, and every server the SDK ships (`createFetchHandler`, the app
 * `serve()` builds, `WebAgentsServer`) answers an unpaid request with a
 * standard x402 402 (v2 header, v1 body), verifies a credits payment against
 * the platform before the handler, settles once after it, exposes the
 * payment headers over CORS, and refuses a priced endpoint on an agent with
 * no payment skill instead of serving it free. A no-code `custom_http`
 * endpoint with a `price` is priced the same way.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { http as endpoint, pricing } from '../../../src/core/decorators';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import { PaymentSkill } from '../../../src/skills/payments/skill';
import { CustomHttpSkill, pricingOfPrice } from '../../../src/skills/custom-http/skill';
import { X402_HEADERS, CREDITS_SCHEME, decodeBase64Json, encodeBase64Json, type PaymentRequiredV2, type PaymentRequiredV1 } from '../../../src/skills/payments/x402-wire';

const BASE = '/agents/priced';
const PLATFORM = 'https://platform.test';

const served: string[] = [];
class Quotes extends Skill {
  constructor() {
    super({ name: 'quotes' });
  }

  @pricing({ creditsPerCall: 0.01, reason: 'A quote' })
  @endpoint({ path: '/quote', method: 'GET', description: 'A quote' })
  async quote(): Promise<Response> {
    served.push('quote');
    return Response.json({ price: 42 });
  }

  @endpoint({ path: '/free', method: 'GET' })
  async free(): Promise<Response> {
    served.push('free');
    return Response.json({ free: true });
  }
}

function agent(options: { payments?: boolean } = {}): BaseAgent {
  const skills: Skill[] = [new Quotes()];
  if (options.payments !== false) {
    skills.push(new PaymentSkill({ enableBilling: true, platformUrl: PLATFORM, apiKey: 'rok_agent', x402: { nonceSecret: 'fleet-secret' } }));
  }
  return new BaseAgent({ name: 'priced', instructions: 'Priced.', skills });
}

type Fetch = (request: Request) => Promise<Response>;

const SERVERS: Array<[string, (a: BaseAgent) => Promise<Fetch>]> = [
  ['createFetchHandler', async (a) => createFetchHandler(a, { basePath: BASE, corsOrigin: '*' })],
  ['serve()', async (a) => {
    const { app } = createAgentApp(a, { basePath: BASE, logging: false, cors: true });
    return (request) => Promise.resolve(app.fetch(request));
  }],
  ['WebAgentsServer', async (a) => {
    const server = new WebAgentsServer({ logging: false });
    await server.addAgent('priced', a);
    return (request) => Promise.resolve(server.getApp().fetch(request));
  }],
];

/** The platform, as the credits scheme calls it: verify answers valid for `tok_valid`, settle records. */
const platform = { verifies: 0, settles: [] as Array<{ body: Record<string, unknown>; headers: Record<string, string> }> };
let fetchMock: ReturnType<typeof vi.fn>;
const originalFetch = globalThis.fetch;

beforeEach(() => {
  served.length = 0;
  platform.verifies = 0;
  platform.settles.length = 0;
  fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
    if (url === `${PLATFORM}/api/payments/verify`) {
      platform.verifies += 1;
      return Response.json(body.token === 'tok_valid' ? { valid: true, balanceCredits: 3 } : { valid: false, error: 'invalid token' });
    }
    if (url === `${PLATFORM}/api/payments/settle`) {
      platform.settles.push({ body, headers: (init?.headers as Record<string, string>) ?? {} });
      return Response.json({ success: true, charged: '12500000', chargedCredits: 0.0125 });
    }
    throw new Error(`unexpected fetch ${url}`);
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function get(fetch: Fetch, sub: string, headers: Record<string, string> = {}): Promise<Response> {
  return fetch(new Request(`http://agent.test${BASE}${sub}`, { headers }));
}

describe.each(SERVERS)('%s', (_name, make) => {
  it('serves the free endpoint as before and answers a priced one 402 with the v2 header and the v1 body', async () => {
    const fetch = await make(agent());
    const free = await get(fetch, '/free');
    expect(free.status).toBe(200);
    expect(served).toEqual(['free']);

    const res = await get(fetch, '/quote');
    expect(res.status).toBe(402);
    expect(served).toEqual(['free']);
    expect(res.headers.get('Cache-Control')).toBe('no-store');
    const v2 = decodeBase64Json(res.headers.get(X402_HEADERS.required)) as unknown as PaymentRequiredV2;
    expect(v2.x402Version).toBe(2);
    expect(v2.resource.url).toBe(`http://agent.test${BASE}/quote`);
    expect(v2.resource.description).toBe('A quote');
    expect(v2.accepts.map((r) => r.scheme)).toEqual([CREDITS_SCHEME]);
    expect(v2.accepts[0].amount).toBe('10000000');
    const v1 = (await res.json()) as PaymentRequiredV1;
    expect(v1.x402Version).toBe(1);
    expect(v1.accepts[0].maxAmountRequired).toBe('10000000');
    expect(platform.verifies).toBe(0);
  });

  it('a credits payment is verified before the handler and settled once after it, under the nonce key', async () => {
    const fetch = await make(agent());
    const challenge = await get(fetch, '/quote');
    const v2 = decodeBase64Json(challenge.headers.get(X402_HEADERS.required)) as unknown as PaymentRequiredV2;
    const paid = await get(fetch, '/quote', {
      [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted: v2.accepts[0], payload: { token: 'tok_valid' } }),
    });
    expect(paid.status).toBe(200);
    expect(await paid.json()).toEqual({ price: 42 });
    expect(served).toEqual(['quote']);
    expect(platform.verifies).toBe(1);
    expect(platform.settles).toHaveLength(1);
    const nonceId = String(v2.accepts[0].extra!.nonce).split('.')[0];
    expect(platform.settles[0].body).toMatchObject({ token: 'tok_valid', amount: 0.01, idempotencyKey: `settle:x402:${nonceId}` });
    expect(platform.settles[0].headers['Idempotency-Key']).toBe(`settle:x402:${nonceId}`);
    expect(platform.settles[0].headers.Authorization).toBe('Bearer rok_agent');
    expect(decodeBase64Json(paid.headers.get(X402_HEADERS.response))).toMatchObject({ success: true, amount: '10000000' });
    expect(paid.headers.get('Cache-Control')).toBe('private');

    // The same payment again: refused, the handler does not run, nothing more is settled.
    const again = await get(fetch, '/quote', {
      [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted: v2.accepts[0], payload: { token: 'tok_valid' } }),
    });
    expect(again.status).toBe(402);
    expect(served).toEqual(['quote']);
    expect(platform.settles).toHaveLength(1);
  });

  it('an invalid token never reaches the handler and settles nothing', async () => {
    const fetch = await make(agent());
    const challenge = await get(fetch, '/quote');
    const v2 = decodeBase64Json(challenge.headers.get(X402_HEADERS.required)) as unknown as PaymentRequiredV2;
    const res = await get(fetch, '/quote', {
      [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted: v2.accepts[0], payload: { token: 'tok_bad' } }),
    });
    expect(res.status).toBe(402);
    expect(served).toEqual([]);
    expect(platform.settles).toHaveLength(0);
  });

  it('a priced endpoint on an agent with no payment skill is refused, never served free', async () => {
    const fetch = await make(agent({ payments: false }));
    const res = await get(fetch, '/quote');
    expect(res.status).toBe(503);
    expect((await res.json()).error.code).toBe('payment_not_configured');
    expect(served).toEqual([]);
  });
});

describe('CORS', () => {
  it('the fetch handler exposes and allows the payment headers on every preflight', async () => {
    const fetch = createFetchHandler(agent(), { basePath: BASE, corsOrigin: '*' });
    const res = await fetch(new Request(`http://agent.test${BASE}/quote`, { method: 'OPTIONS' }));
    expect(res.headers.get('Access-Control-Expose-Headers')).toContain('PAYMENT-REQUIRED');
    expect(res.headers.get('Access-Control-Expose-Headers')).toContain('PAYMENT-RESPONSE');
    expect(res.headers.get('Access-Control-Allow-Headers')).toContain('PAYMENT-SIGNATURE');
    expect(res.headers.get('Access-Control-Allow-Headers')).toContain('X-PAYMENT');
    expect(res.headers.get('Access-Control-Allow-Headers')).toContain('Access-Control-Expose-Headers');
  });

  it('serve() answers a preflight with the same lists', async () => {
    const { app } = createAgentApp(agent(), { basePath: BASE, logging: false, cors: true });
    const res = await app.fetch(new Request(`http://agent.test${BASE}/quote`, {
      method: 'OPTIONS',
      headers: { Origin: 'https://buyer.example', 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'PAYMENT-SIGNATURE' },
    }));
    expect(res.headers.get('Access-Control-Allow-Headers')).toContain('PAYMENT-SIGNATURE');
    expect(res.headers.get('Access-Control-Expose-Headers')).toContain('PAYMENT-REQUIRED');
  });
});

describe('a no-code custom_http endpoint with a price', () => {
  it('registers with the same pricing shape @pricing produces, and a free one with none', () => {
    expect(pricingOfPrice({ credits: 0.02 })).toEqual({ creditsPerCall: 0.02 });
    expect(pricingOfPrice({ credits: 0.001, maxCredits: 0.05, description: 'metered' })).toEqual({ creditsPerCall: 0.001, lock: 0.05, reason: 'metered' });
    expect(pricingOfPrice({ credits: 0 })).toBeUndefined();
    expect(pricingOfPrice(undefined)).toBeUndefined();
    const skill = new CustomHttpSkill({
      endpoints: [
        { id: 'a', path: '/paid', method: 'GET', auth: 'public', use: 'fn', price: { credits: 0.02 } },
        { id: 'b', path: '/open', method: 'GET', auth: 'public', use: 'fn' },
      ],
    });
    const [paid, open] = skill.httpEndpoints;
    expect(paid.pricing).toEqual({ creditsPerCall: 0.02 });
    expect(open.pricing).toBeUndefined();
  });
});
