/**
 * Per-hop budgets in the open SDK (webagents gap-closure plan 2.3,
 * 2026-09-26): `delegate` takes a `budget` (the fixture's parameter), a
 * self-hosted agent mints a child token for exactly that budget through the
 * platform's delegate route instead of forwarding its own token, a refusal
 * fails the hop closed (the parent's balance is never handed over), the host
 * callback receives the budget as its fifth argument, and each hop returns
 * a receipt in the fixture's shape.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation(() => ({ on: vi.fn(), connect: vi.fn(), sendEvents: vi.fn(), close: vi.fn() })),
}));

import { NLISkill } from '../../../src/skills/nli/skill.js';
import {
  DELEGATE_BUDGET_PARAMETER,
  formatDelegateReceipt,
  mintChildToken,
  readChildReceipt,
  resolveDelegateBudget,
  tokenFingerprint,
  tokenIdOf,
} from '../../../src/skills/nli/budget.js';
import type { Context } from '../../../src/core/types.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/payments/delegate_budget.json'), 'utf8')) as {
  parameter: { name: string; description: string; default: number; max: number };
  delegate_route: { path: string; body: string[] };
  receipt: { vectors: Array<{ budget: number; remaining: number; spent: number; tokenId: string; token: string; text: string }> };
};

const b64url = (o: unknown) => Buffer.from(JSON.stringify(o)).toString('base64url');
const CHILD_ID = '9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d';
const CHILD_JWT = `${b64url({ alg: 'RS256' })}.${b64url({ jti: CHILD_ID, payment: { balance: 0.1 } })}.sig`;
const PARENT_JWT = `${b64url({ alg: 'RS256' })}.${b64url({ jti: 'parent-1', payment: { balance: 3 } })}.sig`;

function makeContext(opts: { userId?: string; token?: string; agentToken?: string } = {}): Context {
  const store = new Map<string, unknown>([['_agentic_messages', []]]);
  return {
    auth: { authenticated: !!opts.userId, user_id: opts.userId },
    payment: { valid: !!opts.token, token: opts.token, agentToken: opts.agentToken },
    metadata: {},
    session: { data: {} },
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    signal: new AbortController().signal,
  } as unknown as Context;
}

let fetchMock: ReturnType<typeof vi.fn>;
const originalFetch = globalThis.fetch;
beforeEach(() => {
  fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
    if (url.endsWith('/api/payments/delegate')) {
      if (body.amount === 4.5) return Response.json({ error: 'Delegation refused: max_depth exhausted' }, { status: 400 });
      return Response.json({ token: CHILD_JWT, tokenId: CHILD_ID, amountCredits: body.amount });
    }
    if (url.endsWith('/api/payments/verify')) return Response.json({ valid: true, balanceCredits: 0.0877 });
    throw new Error(`unexpected fetch ${url}`);
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('the fixture pins the parameter and the receipt', () => {
  it('name, description, default and maximum', () => {
    expect(DELEGATE_BUDGET_PARAMETER.name).toBe(FIXTURE.parameter.name);
    expect(DELEGATE_BUDGET_PARAMETER.description).toBe(FIXTURE.parameter.description);
    expect(DELEGATE_BUDGET_PARAMETER.default).toBe(FIXTURE.parameter.default);
    expect(DELEGATE_BUDGET_PARAMETER.max).toBe(FIXTURE.parameter.max);
    expect(resolveDelegateBudget(undefined)).toEqual({ ok: true, budget: FIXTURE.parameter.default });
    expect(resolveDelegateBudget(0.5)).toEqual({ ok: true, budget: 0.5 });
    expect(resolveDelegateBudget(FIXTURE.parameter.max + 1)).toMatchObject({ ok: false });
    expect(resolveDelegateBudget(0)).toMatchObject({ ok: false });
    expect(resolveDelegateBudget('nope')).toMatchObject({ ok: false });
  });

  it('every receipt vector renders to the fixture text', () => {
    for (const v of FIXTURE.receipt.vectors) {
      // The receipt names the child by a fingerprint of its id, never the id (S-303).
      expect(tokenFingerprint(v.tokenId)).toBe(v.token);
      expect(formatDelegateReceipt({ budget: v.budget, spent: v.spent, remaining: v.remaining, token: v.token })).toBe(v.text);
      expect(v.text).not.toContain(v.tokenId);
    }
  });

  it('the delegate tool declares the parameter', () => {
    const skill = new NLISkill({ baseUrl: 'https://platform.test', transport: 'http' });
    const tool = skill.tools.find((t) => t.name === 'delegate')!;
    const props = (tool.parameters as { properties: Record<string, { type: string; description: string }> }).properties;
    expect(props[FIXTURE.parameter.name]).toEqual({ type: 'number', description: FIXTURE.parameter.description });
  });
});

describe('a self-hosted agent (no host callback)', () => {
  function skill(): NLISkill & { streamMessage: ReturnType<typeof vi.fn> } {
    const s = new NLISkill({ baseUrl: 'https://platform.test', transport: 'http', apiKey: 'rok_agent' });
    (s as unknown as { streamMessage: unknown }).streamMessage = vi.fn().mockImplementation(async function* () { yield 'answer'; });
    return s as NLISkill & { streamMessage: ReturnType<typeof vi.fn> };
  }

  it('mints a child for exactly the budget through the delegate route, sends THAT token, and appends the receipt', async () => {
    const s = skill();
    const result = await s.delegate({ agent: '@callee', message: 'hi', budget: 0.1 }, makeContext({ userId: 'u', token: PARENT_JWT }));
    const delegateCall = fetchMock.mock.calls.find((c) => String(c[0]).endsWith(FIXTURE.delegate_route.path))!;
    const sent = JSON.parse(String((delegateCall[1] as RequestInit).body)) as Record<string, unknown>;
    expect(Object.keys(sent).sort()).toEqual([...FIXTURE.delegate_route.body].sort());
    expect(sent).toEqual({ parentToken: PARENT_JWT, delegateTo: 'callee', amount: 0.1 });
    expect((delegateCall[1] as RequestInit).headers).toMatchObject({ Authorization: 'Bearer rok_agent' });
    // The hop is sent the CHILD, never the parent.
    expect(s.streamMessage.mock.calls[0][3]).toBe(CHILD_JWT);
    const text = typeof result === 'string' ? result : result.text;
    expect(text).toContain('answer');
    expect(text).toContain(FIXTURE.receipt.vectors[0].text);
    expect(typeof result === 'string' ? undefined : result.data).toMatchObject({ receipt: { budget: 0.1, spent: 0.0123, remaining: 0.0877, token: tokenFingerprint(CHILD_ID) } });
    expect(JSON.stringify(result)).not.toContain(CHILD_ID);
  });

  it('a refusal by the platform fails the hop closed: nothing is streamed and the parent is never forwarded', async () => {
    const s = skill();
    const result = await s.delegate({ agent: '@callee', message: 'hi', budget: 4.5 }, makeContext({ userId: 'u', token: PARENT_JWT }));
    expect(String(result)).toMatch(/^Error: delegation to @callee refused: Delegation refused: max_depth exhausted/);
    expect(s.streamMessage).not.toHaveBeenCalled();
  });

  it('a budget above the maximum is refused before anything is resolved', async () => {
    const s = skill();
    const result = await s.delegate({ agent: '@callee', message: 'hi', budget: 99 }, makeContext({ userId: 'u', token: PARENT_JWT }));
    expect(String(result)).toMatch(/exceeds maximum allowed/);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(s.streamMessage).not.toHaveBeenCalled();
  });

  it('with a parent token but no platform key the hop is refused rather than paid with the parent', async () => {
    const s = new NLISkill({ baseUrl: 'https://platform.test', transport: 'http' });
    const streamed = vi.fn().mockImplementation(async function* () { yield 'answer'; });
    (s as unknown as { streamMessage: unknown }).streamMessage = streamed;
    const result = await s.delegate({ agent: '@callee', message: 'hi' }, makeContext({ userId: 'u', token: PARENT_JWT }));
    expect(String(result)).toMatch(/no platform key/);
    expect(streamed).not.toHaveBeenCalled();
  });

  it('without any parent token the hop runs unfunded, as before, with no receipt', async () => {
    const s = skill();
    const result = await s.delegate({ agent: '@callee', message: 'hi' }, makeContext({ userId: 'u' }));
    expect(result).toBe('answer');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('a hosted agent (the host derives the child)', () => {
  it('the callback receives the budget the model named fifth, and the receipt reads the child the host returned', async () => {
    const createDelegateToken = vi.fn(async () => CHILD_JWT);
    const s = new NLISkill({ baseUrl: 'https://platform.test', transport: 'http', createDelegateToken, apiKey: 'rok' });
    (s as unknown as { streamMessage: unknown }).streamMessage = vi.fn().mockImplementation(async function* () { yield 'ok'; });
    const result = await s.delegate({ agent: '@callee', message: 'hi', budget: 0.1 }, makeContext({ userId: 'human-1', agentToken: PARENT_JWT }));
    expect(createDelegateToken).toHaveBeenCalledWith('@callee', 'human-1', PARENT_JWT, undefined, 0.1);
    expect(typeof result === 'string' ? result : result.text).toContain(FIXTURE.receipt.vectors[0].text);
    expect(fetchMock.mock.calls.some((c) => String(c[0]).endsWith('/api/payments/delegate'))).toBe(false);
  });

  it('with no budget named the host gets nothing and keeps its own policy; the receipt uses the child\'s own balance claim', async () => {
    const createDelegateToken = vi.fn(async () => CHILD_JWT);
    const s = new NLISkill({ baseUrl: 'https://platform.test', transport: 'http', createDelegateToken, apiKey: 'rok' });
    (s as unknown as { streamMessage: unknown }).streamMessage = vi.fn().mockImplementation(async function* () { yield 'ok'; });
    const result = await s.delegate({ agent: '@callee', message: 'hi' }, makeContext({ userId: 'human-1', agentToken: PARENT_JWT }));
    expect(createDelegateToken).toHaveBeenCalledWith('@callee', 'human-1', PARENT_JWT, undefined, undefined);
    // CHILD_JWT claims a 0.1 balance: that is the receipt's budget.
    expect(typeof result === 'string' ? result : result.text).toContain(FIXTURE.receipt.vectors[0].text);
  });
});

describe('the helpers', () => {
  it('tokenIdOf reads the jti; mintChildToken and readChildReceipt speak the routes', async () => {
    expect(tokenIdOf(CHILD_JWT)).toBe(CHILD_ID);
    expect(tokenIdOf('not-a-jwt')).toBeUndefined();
    const minted = await mintChildToken({ platformUrl: 'https://platform.test', apiKey: 'k', parentToken: PARENT_JWT, delegateTo: '@x', budget: 0.25 });
    expect(minted).toEqual({ ok: true, token: CHILD_JWT, tokenId: CHILD_ID, amountCredits: 0.25 });
    const receipt = await readChildReceipt({ platformUrl: 'https://platform.test', childToken: CHILD_JWT, budget: 0.1 });
    expect(receipt).toEqual({ budget: 0.1, spent: 0.0123, remaining: 0.0877, token: tokenFingerprint(CHILD_ID) });
  });
});
