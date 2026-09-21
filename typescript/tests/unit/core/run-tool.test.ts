/**
 * `BaseAgent.runTool`: one tool, inside a run context bound to the call,
 * never on the shared base context (2026-09-17, portal security log S-136).
 *
 * THE DEFECT THIS PINS AGAINST. The portal serves one cached agent instance
 * to every caller for 60 s. Its voice path used to write the caller's
 * identity and payment token onto `agent.context` and call `executeTool`
 * with no run bound, so the writes landed on the base context: a later run
 * carrying no identity of its own ran as the last voice caller, was billed
 * to their token, and two overlapping calls overwrote each other (last
 * writer wins, scope included). The portal fixed its side by reaching two
 * SDK privates; `runTool` is the public entry that replaces that reach.
 *
 * Three properties, each a case below: the base context is never touched;
 * two overlapping calls each see their own options, in the hooks and in the
 * handler; and when the binding cannot take, the tool is refused rather than
 * run on the shared context.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { tool, hook } from '../../../src/core/decorators';
import type { Context, HookData, HookResult } from '../../../src/core/types';

interface Seen {
  userId: unknown;
  token: unknown;
  chatId: unknown;
  /** What the `before_tool` hook saw through the agent's `context` getter. */
  hookUserId: unknown;
  /** What the `after_tool` hook saw, written after the handler resolved. */
  afterUserId?: unknown;
}

class ProbeSkill extends Skill {
  readonly gates = new Map<string, () => void>();
  readonly seen: Seen[] = [];
  private hookUserId: unknown;

  @hook({ lifecycle: 'before_tool' })
  async beforeTool(_data: HookData, ctx: Context): Promise<HookResult | void> {
    this.hookUserId = ctx.auth?.user_id;
  }

  @hook({ lifecycle: 'after_tool' })
  async afterTool(data: HookData, ctx: Context): Promise<HookResult | void> {
    const result = (data as { tool_result?: Seen }).tool_result;
    if (result) result.afterUserId = ctx.auth?.user_id;
  }

  @tool({
    description: 'Record the context this call runs in, parking at a gate when asked',
    parameters: { type: 'object', properties: { key: { type: 'string' } } },
  })
  async probe(params: { key?: string }, ctx: Context): Promise<Seen> {
    const hookUserId = this.hookUserId;
    if (params.key) await new Promise<void>((resolve) => this.gates.set(params.key!, resolve));
    const entry: Seen = {
      userId: ctx.auth?.user_id,
      token: ctx.payment?.token,
      chatId: ctx.metadata?.chatId,
      hookUserId,
    };
    this.seen.push(entry);
    return entry;
  }
}

async function probeAgent(): Promise<{ agent: BaseAgent; skill: ProbeSkill }> {
  const skill = new ProbeSkill();
  const agent = new BaseAgent({ name: 'run-tool-probe', skills: [skill] });
  await agent.initialize();
  return { agent, skill };
}

/** The base context, read the way a run with no binding would read it. */
function baseContext(agent: BaseAgent): { auth: Record<string, unknown>; payment: Record<string, unknown>; metadata: Record<string, unknown> } {
  return (agent as unknown as { _baseContext: { auth: Record<string, unknown>; payment: Record<string, unknown>; metadata: Record<string, unknown> } })._baseContext;
}

describe('BaseAgent.runTool', () => {
  it('runs the tool with the call options bound, and never touches the base context', async () => {
    const { agent, skill } = await probeAgent();

    const result = (await agent.runTool('probe', {}, {
      userId: 'payer-A',
      paymentToken: 'tok-A',
      chatId: 'chat-A',
      auth: { scope: 'owner' },
    })) as Seen;

    expect(result).toMatchObject({ userId: 'payer-A', token: 'tok-A', chatId: 'chat-A', hookUserId: 'payer-A', afterUserId: 'payer-A' });
    expect(skill.seen).toHaveLength(1);

    // Outside any run the getter answers the base context, and it carries
    // nothing of the call: the old defect was exactly this object holding
    // `payer-A` for the next caller.
    const base = baseContext(agent);
    expect(base.auth.user_id).toBeUndefined();
    expect(base.auth.scope).toBeUndefined();
    expect(base.payment.token).toBeUndefined();
    expect(base.metadata.chatId).toBeUndefined();
    const outside = (agent as unknown as { context: { auth: Record<string, unknown> } }).context;
    expect(outside).toBe(base);
    expect(outside.auth.user_id).toBeUndefined();

    // A later tokenless call sees nobody, not payer-A.
    const anonymous = (await agent.runTool('probe', {})) as Seen;
    expect(anonymous.userId).toBeUndefined();
    expect(anonymous.token).toBeUndefined();
    expect(anonymous.hookUserId).toBeUndefined();
  });

  it('gives two overlapping calls their own options, in the hooks and in the handler', async () => {
    const { agent, skill } = await probeAgent();

    const a = agent.runTool('probe', { key: 'a' }, { userId: 'payer-A', paymentToken: 'tok-A', chatId: 'chat-A' });
    await new Promise((r) => setTimeout(r, 5));
    const b = agent.runTool('probe', { key: 'b' }, { userId: 'payer-B', paymentToken: 'tok-B', chatId: 'chat-B' });
    await new Promise((r) => setTimeout(r, 5));
    expect(skill.gates.size).toBe(2);

    // Release B first, then A: last writer would win on a shared object.
    skill.gates.get('b')!();
    const resultB = (await b) as Seen;
    skill.gates.get('a')!();
    const resultA = (await a) as Seen;

    expect(resultA).toEqual({ userId: 'payer-A', token: 'tok-A', chatId: 'chat-A', hookUserId: 'payer-A', afterUserId: 'payer-A' });
    expect(resultB).toEqual({ userId: 'payer-B', token: 'tok-B', chatId: 'chat-B', hookUserId: 'payer-B', afterUserId: 'payer-B' });
    expect(baseContext(agent).auth.user_id).toBeUndefined();
  });

  it('applies the same gates as executeTool', async () => {
    const { agent } = await probeAgent();
    await expect(agent.runTool('nope', {}, { userId: 'payer-A' })).rejects.toThrow('Tool not found: nope');
  });

  it('refuses to run on the shared context when the binding cannot take', async () => {
    const { agent, skill } = await probeAgent();
    // A pass-through store is what the SDK degrades to without
    // AsyncLocalStorage (a browser build). `run()` tolerates it; this entry
    // must not, because a tool that ran here would run as whoever the base
    // context last described.
    (agent as unknown as { _runStore: unknown })._runStore = {
      getStore: () => undefined,
      run: <R>(_value: unknown, fn: () => R): R => fn(),
    };
    await expect(agent.runTool('probe', {}, { userId: 'payer-A' })).rejects.toThrow(/refusing to run tool "probe" on the shared context/);
    expect(skill.seen).toHaveLength(0);
    expect(baseContext(agent).auth.user_id).toBeUndefined();
  });

  it('is on the IAgent surface, so a host needs no cast and no private', () => {
    const agent = new BaseAgent({ name: 'surface' });
    expect(typeof agent.runTool).toBe('function');
    for (const name of ['_runStore', '_deriveRunContext']) {
      // Still present: the portal reaches them until it switches to runTool.
      expect(name in agent).toBe(true);
    }
  });
});
