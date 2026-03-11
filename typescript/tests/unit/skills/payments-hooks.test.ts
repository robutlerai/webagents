/**
 * PaymentSkill Hooks – on_message, after_toolcall, finalize_connection lifecycle
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  PaymentSkill,
  PaymentContext,
} from '../../../src/skills/payments/skill.js';
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
// on_message: lockFundsForMessage
// ============================================================================

describe('lockFundsForMessage (on_message hook)', () => {
  it('creates per-message lock when token exists but no lock yet', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      perMessageLock: 0.005,
    });

    const payCtx = new PaymentContext();
    payCtx.paymentToken = 'msg-tok';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    fetchMock.mockResolvedValueOnce(
      mockResponse(200, { lockId: 'msg-lock-1', lockedAmountDollars: 0.005 }),
    );

    await (skill as any).lockFundsForMessage(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PLATFORM}/api/payments/lock`);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body.token).toBe('msg-tok');
    expect(body.amount).toBe(0.005);
    expect(payCtx.lockId).toBe('msg-lock-1');
    expect(payCtx.lockedAmountDollars).toBe(0.005);
  });

  it('skips when lock already exists from on_connection', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      perMessageLock: 0.005,
    });

    const payCtx = new PaymentContext();
    payCtx.paymentToken = 'msg-tok';
    payCtx.lockId = 'existing-lock';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    await (skill as any).lockFundsForMessage(HOOK_DATA, ctx);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('skips when billing is disabled', async () => {
    const skill = new PaymentSkill({
      enableBilling: false,
      platformApiUrl: PLATFORM,
    });

    const ctx = createMockContext();
    await (skill as any).lockFundsForMessage(HOOK_DATA, ctx);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not throw when lock attempt fails', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      perMessageLock: 0.005,
    });

    const payCtx = new PaymentContext();
    payCtx.paymentToken = 'msg-tok';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    fetchMock.mockRejectedValueOnce(new Error('network error'));

    await (skill as any).lockFundsForMessage(HOOK_DATA, ctx);

    expect(payCtx.lockId).toBeUndefined();
  });
});

// ============================================================================
// after_toolcall: handleToolCompletion
// ============================================================================

describe('handleToolCompletion (after_toolcall hook)', () => {
  it('settles tool_fee with @pricing metadata', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-tool';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        tool_name: 'web_search',
        tool_result: 'some result',
      },
    });

    // Mock _findPricingForTool to return pricing config
    (skill as any)._findPricingForTool = vi.fn().mockReturnValue({
      creditsPerCall: 0.05,
      reason: 'Web search query',
    });

    fetchMock.mockResolvedValueOnce(mockResponse(200, { success: true }));

    await (skill as any).handleToolCompletion(HOOK_DATA, ctx);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PLATFORM}/api/payments/settle`);

    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body.lockId).toBe('lock-tool');
    expect(body.amount).toBe(0.05);
    expect(body.chargeType).toBe('tool_fee');
    expect(body.description).toContain('web_search');

    expect(payCtx.usageRecords).toHaveLength(1);
    expect(payCtx.usageRecords[0].type).toBe('tool');
    expect(payCtx.usageRecords[0].pricing!.credits).toBe(0.05);
  });

  it('skips settlement when tool result is an error', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-tool';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        tool_name: 'web_search',
        tool_result: 'Error: request timed out',
      },
    });

    await (skill as any).handleToolCompletion(HOOK_DATA, ctx);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(payCtx.usageRecords).toHaveLength(0);
  });

  it('skips when no lockId present', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        tool_name: 'web_search',
        tool_result: 'ok',
      },
    });

    await (skill as any).handleToolCompletion(HOOK_DATA, ctx);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not throw when settle API fails (best-effort)', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-tool';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        tool_name: 'web_search',
        tool_result: 'ok',
      },
    });

    (skill as any)._findPricingForTool = vi.fn().mockReturnValue({
      creditsPerCall: 0.05,
    });

    fetchMock.mockRejectedValueOnce(new Error('network down'));

    await (skill as any).handleToolCompletion(HOOK_DATA, ctx);

    expect(payCtx.usageRecords).toHaveLength(0);
  });
});

// ============================================================================
// finalize_connection: finalizePayment – agent_fee and release
// ============================================================================

describe('finalizePayment – agent_fee and release', () => {
  it('settles agent_fee during finalization', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      agentFee: 0.01,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-final';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    // agent_fee settle + usage settle + release = 3 calls
    expect(fetchMock).toHaveBeenCalledTimes(3);

    const agentFeeBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(agentFeeBody.lockId).toBe('lock-final');
    expect(agentFeeBody.amount).toBe(0.01);
    expect(agentFeeBody.chargeType).toBe('agent_fee');
  });

  it('releases unused lock balance after settlement', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-release';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 100, completionTokens: 50 }],
      },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    // settle + release = 2 calls
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const releaseBody = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(releaseBody.lockId).toBe('lock-release');
    expect(releaseBody.amount).toBe(0);
    expect(releaseBody.release).toBe(true);
  });

  it('skips agent_fee when agentFee is 0', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      agentFee: 0,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-no-fee';

    const ctx = createMockContext({
      _store: {
        _payment_context: payCtx,
        usage: [{ type: 'llm' as const, model: 'gpt-4', promptTokens: 10, completionTokens: 5 }],
      },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    // settle + release = 2 (no agent_fee)
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const firstBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(firstBody.chargeType).toBeUndefined();
  });
});

// ============================================================================
// before_toolcall: preauthToolLock – insufficient lock
// ============================================================================

describe('preauthToolLock – insufficient lock blocks tool', () => {
  it('blocks tool when lock extension fails (402)', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-low';
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

  it('blocks tool when no token and no lock are available', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      defaultToolLock: 0.20,
    });

    const payCtx = new PaymentContext();

    const ctx = createMockContext({
      _store: {
        tool_name: 'some_tool',
        _payment_context: payCtx,
      },
    });

    await (skill as any).preauthToolLock(HOOK_DATA, ctx);

    expect(ctx.get('tool_skipped')).toBe(true);
    expect(ctx.get<string>('tool_result')).toContain('no payment lock available');
  });
});

// ============================================================================
// Cancel path: finalize_connection fires on abort
// ============================================================================

describe('finalizePayment – cancel / abort path', () => {
  it('releases lock with amount=0 when no usage records (abort scenario)', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
      agentFee: 0.01,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-abort';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    fetchMock.mockResolvedValue(mockResponse(200, { success: true }));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);

    // agent_fee settle + release = 2 calls
    expect(fetchMock).toHaveBeenCalledTimes(2);

    const agentFeeBody = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(agentFeeBody.lockId).toBe('lock-abort');
    expect(agentFeeBody.chargeType).toBe('agent_fee');

    const releaseBody = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(releaseBody.lockId).toBe('lock-abort');
    expect(releaseBody.amount).toBe(0);
    expect(releaseBody.release).toBe(true);
  });

  it('does not throw when release fails on abort', async () => {
    const skill = new PaymentSkill({
      enableBilling: true,
      platformApiUrl: PLATFORM,
    });

    const payCtx = new PaymentContext();
    payCtx.lockId = 'lock-abort-fail';

    const ctx = createMockContext({
      _store: { _payment_context: payCtx },
    });

    fetchMock.mockRejectedValueOnce(new Error('network gone'));

    await (skill as any).finalizePayment(HOOK_DATA, ctx);
  });
});
