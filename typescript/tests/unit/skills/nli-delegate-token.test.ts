/**
 * The `createDelegateToken` callback is a money control on the host side, and
 * its ARITY is the contract (build plan 1A-01).
 *
 * The host (the Robutler portal) derives the next hop's payment token from
 * the parent the run carries and bills it to the run's payer. Both values
 * travel only as the third and fourth arguments of this callback. For a day
 * the host declared the widened signature while this skill still called it
 * with two arguments: every hop minted a fresh root token with a full depth
 * budget (S-009), each repository typechecked on its own, and nothing failed.
 * These tests drive the real `NLISkill.delegate` and pin what it passes.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    connect: vi.fn(),
    sendEvents: vi.fn(),
    close: vi.fn(),
  })),
}));

import { NLISkill } from '../../../src/skills/nli/skill.js';
import type { Context } from '../../../src/core/types.js';

const PARENT_JWT = 'parent.payload.signature';

function makeContext(opts: {
  userId?: string;
  agentToken?: string;
  payerId?: string;
} = {}): Context {
  const store = new Map<string, unknown>([['_agentic_messages', []]]);
  return {
    auth: { authenticated: !!opts.userId, user_id: opts.userId },
    payment: { valid: false, agentToken: opts.agentToken, payerId: opts.payerId },
    metadata: {},
    session: { data: {} },
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    signal: new AbortController().signal,
  } as unknown as Context;
}

describe('NLISkill.delegate → createDelegateToken', () => {
  let createDelegateToken: ReturnType<typeof vi.fn>;
  let streamed: ReturnType<typeof vi.fn>;
  let skill: NLISkill;

  beforeEach(() => {
    createDelegateToken = vi.fn(async () => 'child.jwt');
    skill = new NLISkill({ baseUrl: 'https://example.com', transport: 'http', createDelegateToken });
    streamed = vi.fn().mockImplementation(async function* () { yield 'ok'; });
    (skill as any).streamMessage = streamed;
  });

  it('passes the caller, the run\'s agent-scoped token as the parent, and the payer, in that order', async () => {
    await skill.delegate(
      { agent: '@callee', message: 'hi' },
      makeContext({ userId: 'human-1', agentToken: PARENT_JWT, payerId: 'owner-9' }),
    );

    expect(createDelegateToken).toHaveBeenCalledTimes(1);
    expect(createDelegateToken).toHaveBeenCalledWith('@callee', 'human-1', PARENT_JWT, 'owner-9');
    // The exact arity is the contract: a host that reads the parent from
    // argument three must find it there and nowhere else.
    expect(createDelegateToken.mock.calls[0]).toHaveLength(4);
  });

  it('never offers `payment.token` as the parent: that is the unrestricted LLM token on the host', async () => {
    const ctx = makeContext({ userId: 'human-1' });
    ctx.payment.token = 'llm.proxy.token';

    await skill.delegate({ agent: '@callee', message: 'hi' }, ctx);

    expect(createDelegateToken).toHaveBeenCalledWith('@callee', 'human-1', null, undefined);
  });

  it('puts the token the host answered with on the wire, not the run\'s own', async () => {
    await skill.delegate(
      { agent: '@callee', message: 'hi' },
      makeContext({ userId: 'human-1', agentToken: PARENT_JWT }),
    );

    // streamMessage(url, messages, context, delegatePaymentToken, chatId)
    expect(streamed.mock.calls[0][3]).toBe('child.jwt');
  });

  it('a rejected callback is a refusal: the model gets the reason and nothing is streamed', async () => {
    createDelegateToken.mockRejectedValue(new Error('Delegation refused: Maximum agent delegation depth reached (max_depth=0)'));

    const result = await skill.delegate(
      { agent: '@callee', message: 'hi' },
      makeContext({ userId: 'human-1', agentToken: PARENT_JWT }),
    );

    expect(result).toMatch(/^Error: delegation to @callee refused: Delegation refused: Maximum agent delegation depth/);
    expect(streamed).not.toHaveBeenCalled();
  });

  it('a null answer still means "no such agent" and short-circuits with the registry error', async () => {
    createDelegateToken.mockResolvedValue(null);

    const result = await skill.delegate(
      { agent: '@nobody', message: 'hi' },
      makeContext({ userId: 'human-1', agentToken: PARENT_JWT }),
    );

    expect(result).toMatch(/does not exist/);
    expect(streamed).not.toHaveBeenCalled();
  });

  it('without a caller identity the callback is not consulted at all', async () => {
    await skill.delegate({ agent: '@callee', message: 'hi' }, makeContext({ agentToken: PARENT_JWT }));

    expect(createDelegateToken).not.toHaveBeenCalled();
    expect(streamed).toHaveBeenCalledTimes(1);
  });
});
