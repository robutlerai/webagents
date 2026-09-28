/**
 * Every settle carries an Idempotency-Key (2026-09-26), pinned against the
 * fixture both SDKs and the portal read
 * (`python/tests/fixtures/payments/settle_idempotency.json`): the header and
 * body names, the stable derivation `settle:<lockId>:<purpose>` for the
 * settles the skill's lifecycle names, a fresh `settle:<scope>:<uuid>` for the
 * ones nothing names, and the reader keeping `replayed`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PaymentSkill, PaymentContext } from '../../../src/skills/payments/skill.js';
import { PaymentX402Skill } from '../../../src/skills/payments/x402.js';
import { readSettleResult } from '../../../src/skills/payments/settle-result.js';
import {
  IDEMPOTENCY_KEY_BODY_FIELD,
  IDEMPOTENCY_KEY_HEADER,
  IDEMPOTENCY_KEY_MAX_LENGTH,
  IDEMPOTENT_REPLAYED_HEADER,
  freshSettleIdempotencyKey,
  settleIdempotencyKey,
} from '../../../src/skills/payments/idempotency.js';
import type { Context, HookData } from '../../../src/core/types.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/payments/settle_idempotency.json'), 'utf8'),
) as {
  header: string;
  body_field: string;
  replayed_header: string;
  replayed_body_field: string;
  max_length: number;
  pattern: string;
  stable: { format: string; purposes: string[]; vectors: Array<{ lock_id: string; purpose: string; key: string }> };
  fresh: { pattern: string; scopes: string[] };
  invalid_keys: string[];
};

function createMockContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>();
  if (overrides._store) {
    for (const [k, v] of Object.entries(overrides._store as Record<string, unknown>)) store.set(k, v);
  }
  return {
    session: { id: 'test', created_at: Date.now(), last_activity: Date.now(), data: {} },
    auth: { authenticated: true, user_id: 'user-1' },
    payment: { valid: false },
    metadata: {},
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: <T>(key: string, value: T) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    hasScope: () => false,
    hasScopes: () => false,
    ...overrides,
  } as Context;
}

function mockResponse(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) } as Response;
}

const PLATFORM = 'https://platform.test';
const HOOK_DATA: HookData = {};
const VALID = new RegExp(FIXTURE.pattern);
const FRESH = new RegExp(FIXTURE.fresh.pattern);

let fetchMock: ReturnType<typeof vi.fn>;
const originalFetch = globalThis.fetch;
beforeEach(() => {
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** The key a recorded settle request carried, from its header and its body. */
function keyOf(call: unknown[]): { header: string | undefined; body: string | undefined; parsed: Record<string, unknown> } {
  const init = call[1] as { headers: Record<string, string>; body: string };
  const parsed = JSON.parse(init.body) as Record<string, unknown>;
  return { header: init.headers[IDEMPOTENCY_KEY_HEADER], body: parsed[IDEMPOTENCY_KEY_BODY_FIELD] as string | undefined, parsed };
}

describe('the fixture pins the names', () => {
  it('header, body field, replayed header and length match the fixture', () => {
    expect(IDEMPOTENCY_KEY_HEADER).toBe(FIXTURE.header);
    expect(IDEMPOTENCY_KEY_BODY_FIELD).toBe(FIXTURE.body_field);
    expect(IDEMPOTENT_REPLAYED_HEADER).toBe(FIXTURE.replayed_header);
    expect(IDEMPOTENCY_KEY_MAX_LENGTH).toBe(FIXTURE.max_length);
  });

  it('derives every stable vector and nothing else', () => {
    for (const v of FIXTURE.stable.vectors) {
      expect(settleIdempotencyKey(v.lock_id, v.purpose)).toBe(v.key);
      expect(v.key).toMatch(VALID);
    }
    expect(() => settleIdempotencyKey('', 'usage')).toThrow(/lock id/);
    expect(() => settleIdempotencyKey('lock-1', 'has space')).toThrow(/purpose/);
  });

  it('mints a fresh key per call in the fixture shape, for every scope it names', () => {
    for (const scope of FIXTURE.fresh.scopes as Array<'x402' | 'redeem' | 'client'>) {
      const a = freshSettleIdempotencyKey(scope);
      const b = freshSettleIdempotencyKey(scope);
      expect(a).toMatch(FRESH);
      expect(a).toMatch(VALID);
      expect(a.startsWith(`settle:${scope}:`)).toBe(true);
      expect(a).not.toBe(b);
    }
  });

  it('the fixture\'s invalid keys fail the platform pattern', () => {
    for (const bad of FIXTURE.invalid_keys) expect(bad).not.toMatch(VALID);
  });
});

describe('PaymentSkill sends a stable key on every settle it issues', () => {
  it('finalize: agent fee, usage and release each carry settle:<lock>:<purpose>, in header and body', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM, agentFee: 0.01 });
    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-fin';
    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm', model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });
    fetchMock.mockResolvedValue(mockResponse(200, { success: true, charged: '1', chargedDollars: 0.000000001 }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [fee, usage, release] = fetchMock.mock.calls.map(keyOf);
    expect(fee.parsed.chargeType).toBe('agent_fee');
    expect(fee.header).toBe('settle:lock-fin:agent_fee');
    expect(fee.body).toBe(fee.header);
    expect(usage.parsed.usage).toEqual([{ type: 'llm', model: 'gpt-4', prompt_tokens: 10, completion_tokens: 5 }]);
    expect(usage.header).toBe('settle:lock-fin:usage');
    expect(usage.body).toBe(usage.header);
    expect(release.parsed.release).toBe(true);
    expect(release.parsed.amount).toBe(0);
    expect(release.header).toBe('settle:lock-fin:release');
    expect(release.body).toBe(release.header);
    for (const k of [fee, usage, release]) expect(k.header).toMatch(VALID);
  });

  it('a second finalize on the same context derives the same keys, so a repeat is a replay and never a second charge', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-twice';
    const usage = [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 1, completionTokens: 1 }];
    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, createMockContext({ _store: { _payment_context: payCtx, usage } }));
    const first = fetchMock.mock.calls.map(keyOf).map((k) => k.header);
    fetchMock.mockClear();
    fetchMock.mockResolvedValue(mockResponse(200, { success: true, replayed: true }));
    await (skill as any).finalizePayment(HOOK_DATA, createMockContext({ _store: { _payment_context: payCtx, usage } }));
    const second = fetchMock.mock.calls.map(keyOf).map((k) => k.header);

    expect(first).toEqual(['settle:lock-twice:usage', 'settle:lock-twice:release']);
    expect(second).toEqual(first);
  });

  it('two locks never share a key', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));
    for (const lockId of ['lock-a', 'lock-b']) {
      const payCtx = new PaymentContext();
      payCtx.lockId = lockId;
      await (skill as any).finalizePayment(HOOK_DATA, createMockContext({
        _store: { _payment_context: payCtx, usage: [{ type: 'llm', model: 'm', promptTokens: 1, completionTokens: 1 }] },
      }));
    }
    const keys = fetchMock.mock.calls.map(keyOf).map((k) => k.header);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('an explicit key overrides the derived one: a caller retrying its own settle keeps the key it used', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));
    await (skill as any)._settlePayment('lock-x', { amount: 0.02, chargeType: 'agent_fee', idempotencyKey: 'settle:lock-x:agent_fee' });
    await (skill as any)._settlePayment('lock-x', { amount: 0.02 });
    const [explicit, bare] = fetchMock.mock.calls.map(keyOf);
    expect(explicit.header).toBe('settle:lock-x:agent_fee');
    // An amount with nothing to name it gets a fresh key: two such calls are two settles.
    expect(bare.header).toMatch(FRESH);
    expect(bare.header!.startsWith('settle:client:')).toBe(true);
    expect(bare.body).toBe(bare.header);
  });

  it('the paywall\'s credits scheme settles by token under the nonce\'s key, fresh x402 shape, in header and body (2026-09-26)', async () => {
    // The private `verifyX402Payment` is retired; a priced endpoint settles
    // through the credits scheme, whose key is the challenge nonce's uuid.
    const { CreditsScheme, platformCreditsClient } = await import('../../../src/skills/payments/x402-credits.js');
    const client = platformCreditsClient({ platformUrl: PLATFORM, apiKey: 'k' });
    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));
    const keys = ['3f2a9c1e-5b7d-4e8a-9c0b-1d2e3f4a5b6c', '9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d'].map((id) => CreditsScheme.settleKey(id));
    for (const key of keys) await client.settleToken('a.b.c', 0.01, { idempotencyKey: key });
    const settles = fetchMock.mock.calls.map(keyOf);
    settles.forEach((s, i) => {
      expect(s.header).toBe(keys[i]);
      expect(s.header).toMatch(FRESH);
      expect(s.body).toBe(s.header);
    });
    expect(settles[0].header).not.toBe(settles[1].header);
  });
});

describe('PaymentX402Skill.settlePayment', () => {
  it('carries a fresh x402 key, or the one the caller retries with', async () => {
    const skill = new PaymentX402Skill({ facilitatorUrl: PLATFORM, apiKey: 'rok' });
    fetchMock.mockResolvedValue(mockResponse(200, { success: true, charged: '1' }));
    await skill.settlePayment('tok', 0.01);
    await skill.settlePayment('tok', 0.01, { idempotencyKey: 'settle:x402:00000000-0000-4000-8000-000000000000' });
    const [fresh, given] = fetchMock.mock.calls.map(keyOf);
    expect(fresh.header).toMatch(FRESH);
    expect(fresh.body).toBe(fresh.header);
    expect(given.header).toBe('settle:x402:00000000-0000-4000-8000-000000000000');
    expect(given.body).toBe(given.header);
  });
});

describe('readSettleResult', () => {
  it(`keeps \`${FIXTURE.replayed_body_field}\` on a success and never on a failure`, () => {
    expect(readSettleResult({ success: true, charged: '5', replayed: true }, 't').replayed).toBe(true);
    expect(readSettleResult({ success: true, charged: '5' }, 't').replayed).toBeUndefined();
    expect(readSettleResult({ success: false, replayed: true, error: 'x' }, 't').replayed).toBeUndefined();
  });
});
