/**
 * Per-call settle keys (wave-0 review finding 5, 2026-09-26): a settle made
 * for one tool call carries that call's id in its key, so two per-call
 * settles for one purpose on one lock never derive the same key. Pinned
 * against the shared fixture's `per_call` section
 * (`python/tests/fixtures/payments/settle_idempotency.json`), which the
 * Python suite and the portal read too.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PaymentSkill } from '../../../src/skills/payments/skill.js';
import { IDEMPOTENCY_KEY_BODY_FIELD, IDEMPOTENCY_KEY_HEADER, settleIdempotencyKey } from '../../../src/skills/payments/idempotency.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/payments/settle_idempotency.json'), 'utf8'),
) as {
  pattern: string;
  per_call: {
    format: string;
    call_id_pattern: string;
    vectors: Array<{ lock_id: string; purpose: string; call_id: string; key: string }>;
    invalid_call_ids: string[];
  };
};
const VALID = new RegExp(FIXTURE.pattern);

describe('the fixture pins the per-call derivation', () => {
  it('derives every per-call vector, and each passes the platform pattern', () => {
    expect(FIXTURE.per_call.vectors.length).toBeGreaterThan(0);
    for (const v of FIXTURE.per_call.vectors) {
      expect(settleIdempotencyKey(v.lock_id, v.purpose, v.call_id)).toBe(v.key);
      expect(v.key).toMatch(VALID);
    }
  });

  it('two calls on one lock for one purpose are two keys; without a call id the key is the lifecycle one', () => {
    const [a, b] = FIXTURE.per_call.vectors;
    expect(a.lock_id).toBe(b.lock_id);
    expect(a.purpose).toBe(b.purpose);
    expect(a.key).not.toBe(b.key);
    expect(settleIdempotencyKey(a.lock_id, a.purpose)).toBe(`settle:${a.lock_id}:${a.purpose}`);
  });

  it('refuses the fixture\'s invalid call ids', () => {
    for (const bad of FIXTURE.per_call.invalid_call_ids) {
      expect(() => settleIdempotencyKey('lock-1', 'agent_fee', bad)).toThrow(/call id/);
    }
  });
});

describe('PaymentSkill derives a per-call key when a settle names its tool call', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  const originalFetch = globalThis.fetch;
  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({ success: true }) });
    globalThis.fetch = fetchMock;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function keyOf(call: unknown[]): { header: string | undefined; body: string | undefined } {
    const init = call[1] as { headers: Record<string, string>; body: string };
    return { header: init.headers[IDEMPOTENCY_KEY_HEADER], body: (JSON.parse(init.body) as Record<string, string>)[IDEMPOTENCY_KEY_BODY_FIELD] };
  }

  it('amount + chargeType + callId derives settle:<lock>:<chargeType>:<callId>, in header and body', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: 'https://platform.test' });
    const v = FIXTURE.per_call.vectors[0];
    await (skill as any)._settlePayment(v.lock_id, { amount: 0.02, chargeType: v.purpose, callId: v.call_id });
    await (skill as any)._settlePayment(v.lock_id, { amount: 0.02, chargeType: v.purpose });
    const [perCall, lifecycle] = fetchMock.mock.calls.map(keyOf);
    expect(perCall.header).toBe(v.key);
    expect(perCall.body).toBe(v.key);
    expect(lifecycle.header).toBe(`settle:${v.lock_id}:${v.purpose}`);
  });

  it('a release never carries a call id: one release per lock', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: 'https://platform.test' });
    await (skill as any)._settlePayment('lock-r', { amount: 0, release: true, callId: 'call_1' });
    expect(keyOf(fetchMock.mock.calls[0]).header).toBe('settle:lock-r:release');
  });
});
