/**
 * S-285 (2026-09-26): in the SDK's own server, a payment token one caller
 * sent on the UAMP socket landed on the agent's shared base context, and a
 * later A2A run by a different caller inherited it, so that caller's work was
 * billed to the first caller's token, and a caller who sent no token ran on
 * someone else's budget.
 *
 * The review's probe, turned into a test: after the base context carries
 * caller U's token (what the UAMP transport used to write, and what any turn
 * on the SDK's shared-context serve path still leaves there), an A2A run by
 * caller B sees B's token, or none when B sent none, never U's. Plus the
 * unit-level guarantees behind it: `run()` seeds identity and payment from
 * its options alone, the UAMP `session.create` no longer writes the token
 * onto the base context, and `processUAMP` reads the token from the session
 * extensions onto the context of the run it serves.
 *
 * The Python twin is `tests/agents/skills/test_payment_token_isolation_s285_w1fix.py`;
 * Python builds a fresh context per request (a contextvar, not a shared base),
 * so it pins the same isolation rather than a fix.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, hook } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { A2ATransportSkill } from '../../../src/skills/transport/a2a/skill';
import { UAMPTransportSkill } from '../../../src/skills/transport/uamp/skill';

const ORIGIN = 'https://agent.example.test';
const BASE_PATH = '/agents/probe';

interface Seen {
  token: unknown;
  paymentObjToken: unknown;
  userId: unknown;
}

/** Records, in an on_connection hook, what THIS run's context carries. */
class Recorder extends Skill {
  seen: Seen[] = [];

  @hook({ lifecycle: 'on_connection', priority: 50 })
  async record(_data: HookData, context: Context): Promise<void> {
    this.seen.push({
      token: context.get('payment_token'),
      paymentObjToken: (context.payment as { token?: unknown } | undefined)?.token,
      userId: (context.auth as { user_id?: unknown } | undefined)?.user_id,
    });
  }

  last(): Seen {
    return this.seen[this.seen.length - 1];
  }
}

class StubLLM extends Skill {
  @handoff({ name: 'stub-llm' })
  async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent> {
    yield createResponseDeltaEvent('r1', { type: 'text', text: 'ok' });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: 'ok' }]);
  }
}

function agentWith(...extra: Skill[]): { agent: BaseAgent; recorder: Recorder } {
  const recorder = new Recorder();
  const agent = new BaseAgent({ name: 'probe', instructions: 'x', skills: [new StubLLM(), recorder, ...extra] });
  return { agent, recorder };
}

/** What a caller's earlier turn leaves on the SDK's shared base context. */
function poisonBase(agent: BaseAgent): void {
  const ctx = (agent as unknown as { context: Context }).context;
  ctx.set('payment_token', 'tok-U');
  ctx.payment = { valid: true, token: 'tok-U' };
  ctx.auth = { authenticated: true, user_id: 'user:U' };
  ctx.set('_payment_context', { token: 'tok-U' });
}

describe('run() seeds identity and payment from its options, not the base', () => {
  it('a run with no options does not inherit the base token or user', async () => {
    const { agent, recorder } = agentWith();
    poisonBase(agent);
    await agent.run([{ role: 'user', content: 'hi' }], {});
    const seen = recorder.last();
    expect(seen.token).toBeUndefined();
    expect(seen.paymentObjToken).toBeUndefined();
    expect(seen.userId).toBeUndefined();
  });

  it("a run's own options win", async () => {
    const { agent, recorder } = agentWith();
    poisonBase(agent);
    await agent.run([{ role: 'user', content: 'hi' }], { paymentToken: 'tok-B', userId: 'user:B' });
    const seen = recorder.last();
    expect(seen.token).toBe('tok-B');
    expect(seen.userId).toBe('user:B');
  });

  it('the base context is left untouched by a run', async () => {
    const { agent } = agentWith();
    poisonBase(agent);
    await agent.run([{ role: 'user', content: 'hi' }], { paymentToken: 'tok-B' });
    const ctx = (agent as unknown as { context: Context }).context;
    expect(ctx.get('payment_token')).toBe('tok-U');
  });
});

describe('an A2A run by a later caller does not inherit the base token', () => {
  async function sendA2A(fetchHandler: (r: Request) => Promise<Response>, headers: Record<string, string>): Promise<void> {
    const res = await fetchHandler(
      new Request(`${ORIGIN}${BASE_PATH}/a2a`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'a2a-version': '1.0', authorization: 'Bearer cred-B', ...headers },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'SendMessage',
          params: { message: { messageId: `m-${Math.random()}`, role: 'ROLE_USER', parts: [{ text: 'hi' }] } },
        }),
      }),
    );
    expect(res.status).toBe(200);
    // The send blocks until the task settles by default, so the run has recorded.
    await res.json();
  }

  it('sees the A2A caller B token, not U', async () => {
    const { agent, recorder } = agentWith(new UAMPTransportSkill(), new A2ATransportSkill());
    poisonBase(agent);
    const fetchHandler = createFetchHandler(agent, { basePath: BASE_PATH, publicUrl: ORIGIN });
    await sendA2A(fetchHandler, { 'x-payment-token': 'tok-B' });
    expect(recorder.last().token).toBe('tok-B');
    expect(recorder.last().token).not.toBe('tok-U');
  });

  it('sees NO token when B sent none, never U', async () => {
    const { agent, recorder } = agentWith(new UAMPTransportSkill(), new A2ATransportSkill());
    poisonBase(agent);
    const fetchHandler = createFetchHandler(agent, { basePath: BASE_PATH, publicUrl: ORIGIN });
    await sendA2A(fetchHandler, {});
    expect(recorder.last().token).toBeUndefined();
    expect(recorder.last().paymentObjToken).toBeUndefined();
  });
});

describe('UAMP does not write the caller token onto the base context', () => {
  function mockWs(): { ws: WebSocket; sent: string[]; send: (data: string) => Promise<void> } {
    const sent: string[] = [];
    let onmessage: ((ev: MessageEvent) => void) | null = null;
    const ws = {
      send: (data: string) => void sent.push(data),
      close: () => undefined,
      get onmessage() {
        return onmessage;
      },
      set onmessage(fn: ((ev: MessageEvent) => void) | null) {
        onmessage = fn;
      },
      set onclose(_fn: (() => void) | null) {},
      onerror: null,
    } as unknown as WebSocket;
    return {
      ws,
      sent,
      send: async (data: string) => {
        if (onmessage) await onmessage({ data } as MessageEvent);
      },
    };
  }

  it('session.create with X-Payment-Token leaves the base context clean', async () => {
    const { agent } = agentWith();
    const uamp = new UAMPTransportSkill();
    uamp.setAgent(agent);
    const { ws, send } = mockWs();
    uamp.handleConnection(ws, { auth: { authenticated: false }, metadata: {} } as unknown as Context);
    await send(JSON.stringify({ type: 'session.create', session: { extensions: { 'X-Payment-Token': 'tok-U' } } }));
    const ctx = (agent as unknown as { context: Context }).context;
    expect(ctx.get('payment_token')).toBeUndefined();
    expect((ctx.payment as { token?: unknown }).token).toBeUndefined();
  });

  it('processUAMP reads the token from the session extensions onto its run context', async () => {
    const { agent, recorder } = agentWith();
    const events: ClientEvent[] = [
      { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { extensions: { 'X-Payment-Token': 'tok-X' } } } as unknown as ClientEvent,
      { type: 'input.text', event_id: 'e2', text: 'hi', role: 'user' } as unknown as ClientEvent,
      { type: 'response.create', event_id: 'e3' } as unknown as ClientEvent,
    ];
    for await (const _e of agent.processUAMP(events)) void _e;
    expect(recorder.last().token).toBe('tok-X');
  });
});
