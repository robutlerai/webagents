/**
 * PaymentSkill Unit Tests – verify→lock→settle lifecycle
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  PaymentSkill,
  PaymentContext,
  type PaymentSkillConfig,
  type UsageRecord,
} from '../../../src/skills/payments/skill.js';
import { PaymentRequiredError } from '../../../src/skills/payments/x402.js';
import type { Context, HookData } from '../../../src/core/types.js';

// ============================================================================
// Helpers
// ============================================================================

function createMockContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>();
  if (overrides._store) {
    for (const [k, v] of Object.entries(overrides._store as Record<string, unknown>)) {
      store.set(k, v);
    }
  }
  return {
    session: { id: 'test', created_at: Date.now(), last_activity: Date.now(), data: {} },
    auth: { authenticated: true, user_id: 'user-1' },
    payment: { valid: false },
    metadata: (overrides.metadata as Record<string, unknown>) ?? {},
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: <T>(key: string, value: T) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    hasScope: () => false,
    hasScopes: () => false,
    ...overrides,
  } as Context;
}

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

const PLATFORM = 'https://platform.test';
const HOOK_DATA: HookData = {};

// ============================================================================
// Fetch mock lifecycle
// ============================================================================

let fetchMock: ReturnType<typeof vi.fn>;
const originalFetch = globalThis.fetch;

beforeEach(() => {
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ============================================================================
// Tests
// ============================================================================

describe('PaymentContext class', () => {
  it('has sensible defaults', () => {
    const ctx = new PaymentContext();
    expect(ctx.lockedAmountDollars).toBe(0);
    expect(ctx.paymentSuccessful).toBe(false);
    expect(ctx.usageRecords).toEqual([]);
    expect(ctx.paymentToken).toBeUndefined();
    expect(ctx.lockId).toBeUndefined();
  });
});

// --------------------------------------------------------------------------
// setupPaymentContext
// --------------------------------------------------------------------------

describe('setupPaymentContext', () => {
  describe('billing disabled', () => {
    let skill: PaymentSkill;
    beforeEach(() => {
      skill = new PaymentSkill({ enableBilling: false, platformApiUrl: PLATFORM });
    });

    it('stores token for passthrough when present but does not verify/lock', async () => {
      const ctx = createMockContext({
        _store: { payment_token: 'pass-tok' },
      });

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      expect(fetchMock).not.toHaveBeenCalled();
      expect(ctx.payment.token).toBe('pass-tok');
      expect(ctx.payment.valid).toBe(false);
      const stored = ctx.get<PaymentContext>('_payment_context');
      expect(stored).toBeDefined();
      expect(stored!.paymentToken).toBe('pass-tok');
    });

    it('does nothing when no token is present', async () => {
      const ctx = createMockContext();

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      expect(fetchMock).not.toHaveBeenCalled();
      expect(ctx.get('_payment_context')).toBeUndefined();
    });
  });

  describe('token extraction', () => {
    let skill: PaymentSkill;
    beforeEach(() => {
      skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    });

    function mockVerifyAndLock() {
      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 5.0 }))
        .mockResolvedValueOnce(mockResponse(200, { lockId: 'lock-1', lockedAmountDollars: 0.005 }));
    }

    it('reads from context.get("payment_token") first', async () => {
      mockVerifyAndLock();
      const ctx = createMockContext({
        _store: { payment_token: 'ctx-tok' },
        metadata: { 'x-payment-token': 'header-tok' },
      });

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      const verifyBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
      expect(verifyBody.token).toBe('ctx-tok');
    });

    it('falls back to x-payment-token header', async () => {
      mockVerifyAndLock();
      const ctx = createMockContext({
        metadata: { 'x-payment-token': 'hdr-tok' },
      });

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      const verifyBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
      expect(verifyBody.token).toBe('hdr-tok');
    });

    it('falls back to x-payment header', async () => {
      mockVerifyAndLock();
      const ctx = createMockContext({
        metadata: { 'x-payment': 'pay-tok' },
      });

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      const verifyBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
      expect(verifyBody.token).toBe('pay-tok');
    });
  });

  describe('verify + lock happy path', () => {
    let skill: PaymentSkill;
    beforeEach(() => {
      skill = new PaymentSkill({
        enableBilling: true,
        platformApiUrl: PLATFORM,
        perMessageLock: 0.005,
      });
    });

    it('verifies token, locks budget, and sets context.payment', async () => {
      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 2.5 }))
        .mockResolvedValueOnce(mockResponse(200, { lockId: 'lock-abc', lockedAmountDollars: 0.005 }));

      const ctx = createMockContext({ _store: { payment_token: 'good-tok' } });
      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      // Verify call
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(fetchMock.mock.calls[0][0]).toBe(`${PLATFORM}/api/payments/verify`);
      // Lock call
      expect(fetchMock.mock.calls[1][0]).toBe(`${PLATFORM}/api/payments/lock`);
      const lockBody = JSON.parse(fetchMock.mock.calls[1][1].body as string);
      expect(lockBody.amount).toBe(0.005);

      expect(ctx.payment).toEqual(expect.objectContaining({
        valid: true,
        token: 'good-tok',
        balance: 2.5,
        lockId: 'lock-abc',
        lockedAmount: 0.005,
        currency: 'USD',
      }));

      const stored = ctx.get<PaymentContext>('_payment_context');
      expect(stored!.lockId).toBe('lock-abc');
    });

    it('lock amount is min(balance, perMessageLock)', async () => {
      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 0.002 }))
        .mockResolvedValueOnce(mockResponse(200, { lockId: 'lock-small', lockedAmountDollars: 0.002 }));

      const ctx = createMockContext({ _store: { payment_token: 'tok' } });
      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      const lockBody = JSON.parse(fetchMock.mock.calls[1][1].body as string);
      expect(lockBody.amount).toBe(0.002);
    });
  });

  describe('invalid token', () => {
    it('throws PaymentRequiredError when token is not valid', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
      fetchMock.mockResolvedValueOnce(
        mockResponse(200, { valid: false, invalidReason: 'expired' }),
      );

      const ctx = createMockContext({ _store: { payment_token: 'bad-tok' } });

      await expect(
        (skill as any).setupPaymentContext(HOOK_DATA, ctx),
      ).rejects.toThrow(PaymentRequiredError);
    });

    it('includes invalidReason in the error message', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
      fetchMock.mockResolvedValueOnce(
        mockResponse(200, { valid: false, invalidReason: 'expired' }),
      );

      const ctx = createMockContext({ _store: { payment_token: 'bad-tok' } });

      await expect(
        (skill as any).setupPaymentContext(HOOK_DATA, ctx),
      ).rejects.toThrow(/expired/);
    });
  });

  describe('insufficient balance', () => {
    it('throws PaymentRequiredError when balance < 0.001', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
      fetchMock.mockResolvedValue(
        mockResponse(200, { valid: true, balance: 0.0005 }),
      );

      const ctx = createMockContext({ _store: { payment_token: 'low-tok' } });

      await expect(
        (skill as any).setupPaymentContext(HOOK_DATA, ctx),
      ).rejects.toThrow(PaymentRequiredError);
    });

    it('includes insufficient balance text in error message', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
      fetchMock.mockResolvedValue(
        mockResponse(200, { valid: true, balance: 0.0001 }),
      );

      const ctx = createMockContext({ _store: { payment_token: 'low-tok' } });

      await expect(
        (skill as any).setupPaymentContext(HOOK_DATA, ctx),
      ).rejects.toThrow(/Insufficient token balance/);
    });
  });

  describe('lock failure with fallback', () => {
    it('falls back to zero-amount lock on HTTP 400', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 5.0 }))
        // First lock attempt → 400
        .mockImplementationOnce(async () => {
          const err = new Error('Lock failed: HTTP 400') as Error & { status: number };
          err.status = 400;
          throw err;
        })
        // Fallback zero-amount lock → success
        .mockResolvedValueOnce(mockResponse(200, { lockId: 'fallback-lock', lockedAmountDollars: 0 }));

      const ctx = createMockContext({ _store: { payment_token: 'tok' } });

      // _lockBudget throws with status=400, so we need to simulate the skill's _lockBudget
      // Actually, looking at the impl, _lockBudget calls fetch and checks !res.ok, throws with status.
      // Let's mock fetch to return a 400 response instead.
      fetchMock.mockReset();
      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 5.0 }))
        .mockResolvedValueOnce(mockResponse(400, { error: 'cannot lock' }))
        .mockResolvedValueOnce(mockResponse(200, { lockId: 'fallback-lock', lockedAmountDollars: 0 }));

      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);

      expect(fetchMock).toHaveBeenCalledTimes(3);
      const fallbackBody = JSON.parse(fetchMock.mock.calls[2][1].body as string);
      expect(fallbackBody.amount).toBe(0);

      const stored = ctx.get<PaymentContext>('_payment_context');
      expect(stored!.lockId).toBe('fallback-lock');
      expect(stored!.lockedAmountDollars).toBe(0);
    });

    it('re-throws non-400 lock errors', async () => {
      const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
      fetchMock
        .mockResolvedValueOnce(mockResponse(200, { valid: true, balance: 5.0 }))
        .mockResolvedValueOnce(mockResponse(500, { error: 'server error' }));

      const ctx = createMockContext({ _store: { payment_token: 'tok' } });

      await expect(
        (skill as any).setupPaymentContext(HOOK_DATA, ctx),
      ).rejects.toThrow(/Lock failed/);
    });
  });

  describe('no token + billing enabled', () => {
    it('throws PaymentRequiredError with accepts when minimumBalance > 0', async () => {
      const skill = new PaymentSkill({
        enableBilling: true,
        platformApiUrl: PLATFORM,
        minimumBalance: 0.01,
      });

      const ctx = createMockContext();

      try {
        await (skill as any).setupPaymentContext(HOOK_DATA, ctx);
        expect.unreachable('should have thrown');
      } catch (err) {
        expect(err).toBeInstanceOf(PaymentRequiredError);
        const pre = err as PaymentRequiredError;
        expect(pre.accepts).toBeDefined();
        expect(pre.accepts).toHaveLength(1);
        expect(pre.accepts![0]).toEqual(expect.objectContaining({
          scheme: 'token',
          network: 'robutler',
          asset: 'robutler:credits',
        }));
      }
    });

    it('does not throw when minimumBalance is 0', async () => {
      const skill = new PaymentSkill({
        enableBilling: true,
        platformApiUrl: PLATFORM,
        minimumBalance: 0,
      });

      const ctx = createMockContext();
      await (skill as any).setupPaymentContext(HOOK_DATA, ctx);
      // Should complete without throwing; PaymentContext still stored
      expect(ctx.get('_payment_context')).toBeDefined();
    });
  });

});

// --------------------------------------------------------------------------
// preauthToolLock
// --------------------------------------------------------------------------

describe('preauthToolLock', () => {
  it('returns immediately when billing is disabled', async () => {
    const skill = new PaymentSkill({ enableBilling: false, platformApiUrl: PLATFORM });
    const ctx = createMockContext();

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns immediately when no tool_name in context', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    const ctx = createMockContext();

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('extends existing lock successfully', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'existing-lock';
    payCtx.paymentToken = 'tok';
    payCtx.lockedAmountDollars = 0.005;

    const ctx = createMockContext({
      _store: {
        tool_name: 'web_search',
        _payment_context: payCtx,
      },
    });

    fetchMock.mockResolvedValueOnce(mockResponse(200, { success: true }));

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain('/api/payments/lock/existing-lock');
    expect(fetchMock.mock.calls[0][1].method).toBe('PATCH');
    expect(payCtx.lockedAmountDollars).toBe(0.005 + 0.20);
    expect(ctx.get('tool_skipped')).toBeUndefined();
  });

  it('blocks tool when lock extension fails', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'existing-lock';
    payCtx.paymentToken = 'tok';

    const ctx = createMockContext({
      _store: {
        tool_name: 'expensive_tool',
        _payment_context: payCtx,
      },
    });

    fetchMock.mockResolvedValueOnce(mockResponse(402, { error: 'insufficient' }));

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(ctx.get('tool_skipped')).toBe(true);
    const result = ctx.get<string>('tool_result');
    expect(result).toContain('blocked');
    expect(result).toContain('spending limit exceeded');
  });

  it('creates fresh lock when no lock exists but token is present', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();
    payCtx.paymentToken = 'fresh-tok';
    // No lockId

    const ctx = createMockContext({
      _store: {
        tool_name: 'some_tool',
        _payment_context: payCtx,
      },
    });

    fetchMock.mockResolvedValueOnce(
      mockResponse(200, { lockId: 'new-lock', lockedAmountDollars: 0.20 }),
    );

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PLATFORM}/api/payments/lock`);
    expect(payCtx.lockId).toBe('new-lock');
    expect(ctx.get('tool_skipped')).toBeUndefined();
  });

  it('blocks tool when fresh lock also fails', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();
    payCtx.paymentToken = 'broke-tok';

    const ctx = createMockContext({
      _store: {
        tool_name: 'some_tool',
        _payment_context: payCtx,
      },
    });

    fetchMock.mockResolvedValueOnce(mockResponse(400, { error: 'denied' }));

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(ctx.get('tool_skipped')).toBe(true);
    expect(ctx.get<string>('tool_result')).toContain('no payment lock available');
  });

  it('skips extension when lockAmount is 0', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'existing';
    payCtx.paymentToken = 'tok';

    const ctx = createMockContext({
      _store: {
        tool_name: 'free_tool',
        _payment_context: payCtx,
      },
    });

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// --------------------------------------------------------------------------
// finalizePayment
// --------------------------------------------------------------------------

describe('finalizePayment', () => {
  it('returns immediately when billing is disabled', async () => {
    const skill = new PaymentSkill({ enableBilling: false, platformApiUrl: PLATFORM });
    const ctx = createMockContext();

    await (skill as any).finalizePayment(HOOK_DATA, ctx);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns when no PaymentContext stored', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });
    const ctx = createMockContext();

    await (skill as any).finalizePayment(HOOK_DATA, ctx);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('releases lock with amount=0 when no usage records', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-noops';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    fetchMock.mockResolvedValueOnce(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body.lockId).toBe('lock-noops');
    expect(body.amount).toBe(0);
    expect(body.release).toBe(true);
  });

  it('settles all usage records in normal (non-BYOK) mode', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-settle';

    const llmRecord: UsageRecord = {
      type: 'llm',
      model: 'gpt-4',
      promptTokens: 100,
      completionTokens: 50,
    };
    const toolRecord: UsageRecord = {
      type: 'tool',
      pricing: { credits: 0.05, reason: 'web search' },
    };

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [llmRecord, toolRecord],
      },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    // settle call (combined) + release call = 2 fetches
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const settleBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(settleBody.lockId).toBe('lock-settle');
    expect(settleBody.usage).toHaveLength(2);
    expect(settleBody.description).toBe('LLM + tool usage');

    const releaseBody = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(releaseBody.amount).toBe(0);
    expect(releaseBody.release).toBe(true);
  });

  it('sets paymentSuccessful and settled on context after settlement', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-done';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    expect(payCtx.paymentSuccessful).toBe(true);
    expect(ctx.payment.settled).toBe(true);
  });

  it('does not throw when settlement API fails (best-effort)', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-fail';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });

    fetchMock.mockRejectedValue(new Error('network down'));

    // Should not throw
    await (skill as any).finalizePayment(HOOK_DATA, ctx);
  });

  it('skips settlement when lockId is missing despite having usage', async () => {
    const skill = new PaymentSkill({ enableBilling: true, platformApiUrl: PLATFORM });

    const payCtx = new PaymentContext();
    // No lockId

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// --------------------------------------------------------------------------
// Constructor defaults
// --------------------------------------------------------------------------

describe('PaymentSkill constructor', () => {
  it('strips trailing slash from platformApiUrl', () => {
    const skill = new PaymentSkill({ platformApiUrl: 'https://api.test/' });
    expect((skill as any).platformApiUrl).toBe('https://api.test');
  });

  it('defaults enableBilling to true', () => {
    const skill = new PaymentSkill({ platformApiUrl: PLATFORM });
    expect((skill as any).enableBilling).toBe(true);
  });

  it('uses agentName as agentId fallback', () => {
    const skill = new PaymentSkill({ agentName: 'my-agent', platformApiUrl: PLATFORM });
    expect((skill as any).agentId).toBe('my-agent');
  });

  it('prefers agentId over agentName', () => {
    const skill = new PaymentSkill({
      agentId: 'id-1',
      agentName: 'name-1',
      platformApiUrl: PLATFORM,
    });
    expect((skill as any).agentId).toBe('id-1');
  });
});
