/**
 * Where a priced endpoint says to pay, and what it answers when the platform
 * cannot be reached (B12, 2026-09-28), against the shared fixture
 * `payments/final_sdk_x402_platform.json` the Python suite reads too.
 *
 * With nothing naming a platform, the payment skill's last resort was
 * `http://localhost:3000`, which a 402 published as the platform to pay
 * through, and a paid retry against a platform that was not there answered
 * 500: the connection error escaped the paywall.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { BaseAgent } from '../../../../src/core/agent';
import { Skill } from '../../../../src/core/skill';
import { http as endpoint, pricing } from '../../../../src/core/decorators';
import { createFetchHandler } from '../../../../src/server/handler';
import { PaymentSkill } from '../../../../src/skills/payments/skill';
import { platformCreditsClient } from '../../../../src/skills/payments/x402-credits';
import { X402_HEADERS, decodeBase64Json, encodeBase64Json, type PaymentRequiredV2 } from '../../../../src/skills/payments/x402-wire';
import { tempDirs } from '../../../helpers/cli';

const FIXTURE = JSON.parse(
  readFileSync(path.resolve(__dirname, '../../../../../python/tests/fixtures/payments/final_sdk_x402_platform.json'), 'utf8'),
) as {
  default_platform: string;
  unreachable: { reason: string; status: number; retry_after: string; body: unknown };
  unreachable_platform: string;
};

const tempDir = tempDirs();
const ISOLATED = ['HOME', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_INTERNAL_API_URL', 'ROBUTLER_PLATFORM_URL'];
const saved: Record<string, string | undefined> = {};
const originalFetch = globalThis.fetch;

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-final-sdk-x402-');
});

afterEach(() => {
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

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
}

function priced(payment: PaymentSkill) {
  const agent = new BaseAgent({ name: 'priced', instructions: 'Priced.', skills: [new Quotes(), payment] });
  return { agent, fetch: createFetchHandler(agent, { basePath: '/agents/priced', corsOrigin: '*' }) };
}

describe('the platform a 402 names', () => {
  it('is never localhost when nothing names one', async () => {
    const { agent, fetch } = priced(new PaymentSkill({ x402: { nonceSecret: 'fleet-secret' } }));
    await agent.initialize();
    const res = await fetch(new Request('http://127.0.0.1/agents/priced/quote'));
    expect(res.status).toBe(402);
    const v2 = decodeBase64Json(res.headers.get(X402_HEADERS.required)!) as PaymentRequiredV2;
    expect((v2.accepts[0].extra as { platform?: string }).platform).toBe(FIXTURE.default_platform);
    expect(await res.text()).not.toContain('localhost');
  });
});

describe('a platform that cannot be reached', () => {
  const refuse = async () => {
    throw new TypeError('fetch failed: connect ECONNREFUSED 127.0.0.1:9');
  };

  it('is an answer from the client, not an exception', async () => {
    const client = platformCreditsClient({ platformUrl: FIXTURE.unreachable_platform, fetch: refuse as unknown as typeof fetch });
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(await client.verifyToken('tok')).toEqual({ valid: false, invalidReason: FIXTURE.unreachable.reason });
    const settled = await client.settleToken('tok', 0.01, { idempotencyKey: 'settle:x402:x' });
    expect(settled.success).toBe(false);
    expect(settled.error).toBe(FIXTURE.unreachable.reason);
  });

  it('answers a paid retry 503, and runs nothing', async () => {
    served.length = 0;
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    globalThis.fetch = refuse as unknown as typeof fetch;
    const { agent, fetch } = priced(new PaymentSkill({ platformUrl: FIXTURE.unreachable_platform, x402: { nonceSecret: 'fleet-secret' } }));
    await agent.initialize();
    const challenge = await fetch(new Request('http://127.0.0.1/agents/priced/quote'));
    const entry = (decodeBase64Json(challenge.headers.get(X402_HEADERS.required)!) as PaymentRequiredV2).accepts[0];
    const paid = await fetch(new Request('http://127.0.0.1/agents/priced/quote', {
      headers: { [X402_HEADERS.signature]: encodeBase64Json({ x402Version: 2, accepted: entry, payload: { token: 'tok_valid' } }) },
    }));
    expect(paid.status).toBe(FIXTURE.unreachable.status);
    expect(await paid.json()).toEqual(FIXTURE.unreachable.body);
    expect(paid.headers.get('retry-after')).toBe(FIXTURE.unreachable.retry_after);
    expect(served).toEqual([]);
  });
});
