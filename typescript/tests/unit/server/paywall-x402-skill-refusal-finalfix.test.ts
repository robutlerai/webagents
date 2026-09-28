/**
 * A priced `@http` endpoint on an agent whose only payment skill is
 * `PaymentX402Skill` (2026-09-27, the final e2e re-run): that skill verifies
 * payment tokens on chat turns and carries no paywall, so the endpoint
 * answered the generic 503, which named nothing. It now answers a 503 whose
 * sentence names `PaymentSkill`, the seller, from the shared fixture
 * `python/tests/fixtures/payments/paywall_x402_mpp.json` (`not_configured`).
 * With `PaymentSkill` on the agent the same request gets the 402 challenge,
 * and with no payment skill at all the generic sentence, on every server the
 * SDK ships.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { http as endpoint, pricing } from '../../../src/core/decorators';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import { PaymentSkill } from '../../../src/skills/payments/skill';
import { PaymentX402Skill, X402_SKILL_PAYWALL_REFUSAL } from '../../../src/skills/payments/x402';
import { PAYMENT_NOT_CONFIGURED_MESSAGE, resolvePaywall, resolvePaywallRefusal } from '../../../src/skills/payments/paywall';
import { X402_HEADERS } from '../../../src/skills/payments/x402-wire';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/payments/paywall_x402_mpp.json'), 'utf8')) as {
  x402: { not_configured: { status: number; code: string; message: string; typescript_x402_skill: { class: string; message: string } } };
};
const NOT_CONFIGURED = FIXTURE.x402.not_configured;

const BASE = '/agents/priced';

class Quotes extends Skill {
  static served: string[] = [];
  constructor() {
    super({ name: 'quotes' });
  }

  @pricing({ creditsPerCall: 0.01, reason: 'A quote' })
  @endpoint({ path: '/quote', method: 'GET', description: 'A quote' })
  async quote(): Promise<Response> {
    Quotes.served.push('quote');
    return Response.json({ price: 42 });
  }
}

function agent(payments: 'x402' | 'seller' | 'none'): BaseAgent {
  const skills: Skill[] = [new Quotes()];
  if (payments === 'x402') skills.push(new PaymentX402Skill({ facilitatorUrl: 'https://platform.test', apiKey: 'rok_agent' }));
  if (payments === 'seller') skills.push(new PaymentSkill({ enableBilling: true, platformUrl: 'https://platform.test', apiKey: 'rok_agent', x402: { nonceSecret: 'fleet-secret' } }));
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

describe('the fixture is the contract', () => {
  it('pins both sentences and the class', () => {
    expect(PAYMENT_NOT_CONFIGURED_MESSAGE).toBe(NOT_CONFIGURED.message);
    expect(X402_SKILL_PAYWALL_REFUSAL).toBe(NOT_CONFIGURED.typescript_x402_skill.message);
    expect(X402_SKILL_PAYWALL_REFUSAL).toContain('PaymentSkill');
    expect(PaymentX402Skill.name).toBe(NOT_CONFIGURED.typescript_x402_skill.class);
    expect(NOT_CONFIGURED.code).toBe('payment_not_configured');
    expect(NOT_CONFIGURED.status).toBe(503);
  });

  it('PaymentX402Skill carries the refusal and no paywall; PaymentSkill carries a paywall and no refusal', () => {
    expect(resolvePaywall(agent('x402'))).toBeUndefined();
    expect(resolvePaywallRefusal(agent('x402'))).toBe(X402_SKILL_PAYWALL_REFUSAL);
    expect(resolvePaywall(agent('seller'))).toBeDefined();
    expect(resolvePaywallRefusal(agent('seller'))).toBeUndefined();
    expect(resolvePaywallRefusal(agent('none'))).toBeUndefined();
  });
});

describe.each(SERVERS)('%s', (_name, make) => {
  it('refuses a priced endpoint on an agent with only PaymentX402Skill, naming PaymentSkill, and never runs the handler', async () => {
    Quotes.served.length = 0;
    const fetch = await make(agent('x402'));
    const res = await fetch(new Request(`http://localhost${BASE}/quote`));
    expect(res.status).toBe(NOT_CONFIGURED.status);
    expect(await res.json()).toEqual({ error: { code: NOT_CONFIGURED.code, message: NOT_CONFIGURED.typescript_x402_skill.message } });
    expect(Quotes.served).toEqual([]);
  });

  it('answers the generic sentence with no payment skill at all', async () => {
    const fetch = await make(agent('none'));
    const res = await fetch(new Request(`http://localhost${BASE}/quote`));
    expect(res.status).toBe(503);
    expect(((await res.json()) as { error: { message: string } }).error.message).toBe(NOT_CONFIGURED.message);
  });

  it('challenges with 402 once PaymentSkill is on the agent', async () => {
    Quotes.served.length = 0;
    const fetch = await make(agent('seller'));
    const res = await fetch(new Request(`http://localhost${BASE}/quote`));
    expect(res.status).toBe(402);
    expect(res.headers.get(X402_HEADERS.required)).toBeTruthy();
    expect(Quotes.served).toEqual([]);
  });
});
