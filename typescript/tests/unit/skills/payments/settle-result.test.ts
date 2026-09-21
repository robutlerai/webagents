/**
 * Partial settles (2026-09-18). Since the portal's S-148 fix a short lock is
 * charged what it holds, and `POST /api/payments/settle` answers `success:
 * true` with `partial`, and, when partial, `unbilled` and `requested`
 * (app/api/payments/settle/route.ts). The SDK read only `success`, so a
 * partial settle was reported as charged in full. These cases pin the
 * reader and the three callers: the payment skill's finalize, its x402
 * verify-and-settle, and `PaymentX402Skill.settlePayment`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readSettleResult } from '../../../../src/skills/payments/settle-result';
import { PaymentSkill, PaymentContext } from '../../../../src/skills/payments/skill';
import { PaymentX402Skill } from '../../../../src/skills/payments/x402';
import type { Context } from '../../../../src/core/types';

const FULL = { success: true, partial: false, charged: '5000000', chargedDollars: 0.005, remaining: '0', remainingDollars: 0 };
const PARTIAL = {
  success: true,
  partial: true,
  charged: '3000000',
  chargedDollars: 0.003,
  unbilled: '2000000',
  unbilledDollars: 0.002,
  requested: '5000000',
  requestedDollars: 0.005,
  remaining: '0',
  remainingDollars: 0,
};

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function context(store: Record<string, unknown>): Context {
  const m = new Map(Object.entries(store));
  return {
    session: { id: 't', created_at: 0, last_activity: 0, data: {} },
    auth: { authenticated: true, user_id: 'u1' },
    payment: { valid: false },
    metadata: {},
    get: <T>(k: string) => m.get(k) as T | undefined,
    set: <T>(k: string, v: T) => void m.set(k, v),
    delete: (k: string) => void m.delete(k),
    hasScope: () => false,
    hasScopes: () => false,
  } as unknown as Context;
}

let warn: ReturnType<typeof vi.spyOn>;
const fetchMock = vi.fn();
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
  warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
});
afterEach(() => {
  vi.unstubAllGlobals();
  warn.mockRestore();
});

describe('readSettleResult', () => {
  it('keeps partial, charged and unbilled, and warns on a partial only', () => {
    expect(readSettleResult(FULL, 't')).toMatchObject({ success: true, partial: false, charged: '5000000', chargedDollars: 0.005 });
    expect(warn).not.toHaveBeenCalled();
    expect(readSettleResult(PARTIAL, 't')).toMatchObject({ success: true, partial: true, unbilled: '2000000', unbilledDollars: 0.002, requested: '5000000' });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toMatch(/PARTIAL settle, charged 0.003 of 0.005 \(unbilled 0.002\)/);
  });

  it('an older platform with no partial field reads as not partial; a failure is never partial', () => {
    expect(readSettleResult({ success: true, charged: '1' }, 't').partial).toBe(false);
    expect(readSettleResult({ success: false, partial: true, error: 'x' }, 't')).toMatchObject({ success: false, partial: false, error: 'x' });
    expect(readSettleResult(null, 't')).toMatchObject({ success: false, partial: false });
  });
});

describe('the callers never report a partial settle as charged in full', () => {
  it('finalizePayment marks the run settled AND partial, with what went unbilled', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: 'https://platform.test' });
    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-1';
    const ctx = context({ _payment_context: payCtx, usage: [{ type: 'llm', model: 'gpt-4', promptTokens: 1, completionTokens: 1 }] });
    fetchMock.mockResolvedValueOnce(json(PARTIAL)).mockResolvedValueOnce(json(FULL));
    await (skill as unknown as { finalizePayment: (d: unknown, c: Context) => Promise<void> }).finalizePayment({}, ctx);
    expect(payCtx.settlePartial).toBe(true);
    expect(payCtx.unbilledDollars).toBeCloseTo(0.002);
    expect(ctx.payment).toMatchObject({ settled: true, partial: true, unbilledDollars: 0.002 });
    expect(warn).toHaveBeenCalled();

    const fullCtx = context({ _payment_context: Object.assign(new PaymentContext(), { lockId: 'lock-2' }), usage: [{ type: 'llm', model: 'gpt-4' }] });
    fetchMock.mockResolvedValueOnce(json(FULL)).mockResolvedValueOnce(json(FULL));
    await (skill as unknown as { finalizePayment: (d: unknown, c: Context) => Promise<void> }).finalizePayment({}, fullCtx);
    expect(fullCtx.payment).toMatchObject({ settled: true });
    expect((fullCtx.payment as Record<string, unknown>).partial).toBeUndefined();
  });

  it('verifyX402Payment says partial, with charged and unbilled, instead of a plain valid', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: 'https://platform.test', maxPayment: 10 });
    fetchMock.mockResolvedValueOnce(json({ valid: true, balance: 10 })).mockResolvedValueOnce(json(PARTIAL));
    expect(await skill.verifyX402Payment('tok', 0.005)).toEqual({ valid: true, partial: true, chargedDollars: 0.003, unbilledDollars: 0.002 });
    fetchMock.mockResolvedValueOnce(json({ valid: true, balance: 10 })).mockResolvedValueOnce(json(FULL));
    expect(await skill.verifyX402Payment('tok', 0.005)).toEqual({ valid: true });
  });

  it('PaymentX402Skill.settlePayment returns the partial fields', async () => {
    const skill = new PaymentX402Skill({ facilitatorUrl: 'https://platform.test' });
    fetchMock.mockResolvedValueOnce(json(PARTIAL));
    expect(await skill.settlePayment('tok', 0.005)).toMatchObject({ success: true, partial: true, unbilledDollars: 0.002, requestedDollars: 0.005 });
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
