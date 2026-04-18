/**
 * Portal Transport — Abort Propagation Tests
 *
 * Verifies that PortalTransportSkill aborts an in-flight `processUAMP` loop
 * when the parent sends `response.cancel` (mirroring per-agent-uamp.ts).
 * Without this, a parent abort does not propagate into a delegate sub-agent
 * and the delegate keeps running for tens of seconds — see
 * plans/surface_platform_tool_history_3596ddbe.
 */

import { describe, it, expect, vi } from 'vitest';
import { PortalTransportSkill } from '../../../src/skills/transport/portal/skill.js';
import type { IAgent, Context, Capabilities } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';

/**
 * Minimal in-memory Context implementation for the test agent.
 */
function makeContext(): Context & { _data: Map<string, unknown> } {
  const data = new Map<string, unknown>();
  return {
    _data: data,
    get<T = unknown>(key: string): T | undefined { return data.get(key) as T; },
    set(key: string, value: unknown) { data.set(key, value); },
    has(key: string) { return data.has(key); },
    delete(key: string) { data.delete(key); return true; },
  } as unknown as Context & { _data: Map<string, unknown> };
}

/**
 * Long-running mock agent: yields one event per 50ms tick; honors
 * `context.signal` (the path PortalTransportSkill aborts via).
 */
function makeMockAgent(context: Context) {
  const observed: { sawSignal: boolean; sawAborted: boolean } = {
    sawSignal: false,
    sawAborted: false,
  };
  const agent = {
    name: 'mock',
    getCapabilities(): Capabilities { return { protocols: ['uamp'] } as Capabilities; },
    context,
    observed,
    async *processUAMP(_events: ClientEvent[]): AsyncGenerator<ServerEvent, void, unknown> {
      const sig = (context as Context & { signal?: AbortSignal }).signal;
      if (sig) observed.sawSignal = true;
      for (let i = 0; i < 50; i++) {
        const cur = (context as Context & { signal?: AbortSignal }).signal;
        if (cur?.aborted) {
          observed.sawAborted = true;
          return;
        }
        await new Promise(r => setTimeout(r, 20));
        yield {
          type: 'response.delta',
          event_id: `evt_${i}`,
          timestamp: Date.now(),
          response_id: 'r1',
          delta: { type: 'text', text: `chunk-${i}` },
        } as ServerEvent;
      }
    },
    async run() { return { content: '' } as unknown as Awaited<ReturnType<IAgent['run']>>; },
    async *runStreaming() { /* unused */ },
  } as unknown as IAgent & { context: Context; observed: typeof observed };
  return agent;
}

/**
 * In-memory WebSocket double that captures sent messages and lets the
 * test push messages back into `onmessage` as if from the network.
 */
function makeWs() {
  const sent: string[] = [];
  let onmessage: ((ev: { data: string }) => void) | null = null;
  let onclose: (() => void) | null = null;
  return {
    sent,
    readyState: 1,
    send(data: string) { sent.push(data); },
    close() { onclose?.(); },
    set onmessage(fn: (ev: { data: string }) => void) { onmessage = fn; },
    get onmessage() { return onmessage!; },
    set onclose(fn: () => void) { onclose = fn; },
    get onclose() { return onclose!; },
    set onerror(_fn: (err: unknown) => void) { /* noop */ },
    inject(msg: object) { onmessage?.({ data: JSON.stringify(msg) }); },
  };
}

describe('PortalTransportSkill — abort propagation', () => {
  it('aborts in-flight processUAMP when client sends response.cancel', async () => {
    const ctx = makeContext();
    const agent = makeMockAgent(ctx);
    const skill = new PortalTransportSkill();
    skill.setAgent(agent);

    const ws = makeWs();
    // Cast ws to any since handleConnection expects browser WebSocket.
    skill.handleConnection(ws as unknown as WebSocket, makeContext());

    // Kick off a long uamp processing loop.
    ws.inject({
      type: 'uamp',
      events: [{ type: 'response.create', event_id: 'e1', timestamp: Date.now() }] as ClientEvent[],
    });

    // Let a couple of deltas stream out.
    await new Promise(r => setTimeout(r, 80));
    const sentBeforeCancel = ws.sent.length;
    expect(sentBeforeCancel).toBeGreaterThan(0);
    expect(sentBeforeCancel).toBeLessThan(50);

    // Simulate parent abort → ws message with response.cancel.
    ws.inject({ type: 'response.cancel', event_id: 'e2', timestamp: Date.now() });

    // Wait for the loop to wind down (it should stop within one tick).
    await new Promise(r => setTimeout(r, 100));
    const sentAfterCancel = ws.sent.length;

    // The loop should have stopped — at most a handful of additional
    // events (the in-flight chunk that was already mid-yield). Without
    // the fix this would equal 50.
    expect(sentAfterCancel - sentBeforeCancel).toBeLessThan(5);
    expect(sentAfterCancel).toBeLessThan(50);

    // The skill installed an AbortSignal on the agent's context — the
    // mock agent captures this on entry. Without setAgentSignal the
    // signal would be undefined and parent abort would never propagate.
    expect((agent as unknown as { observed: { sawSignal: boolean } }).observed.sawSignal).toBe(true);

    // The skill should have sent a response.cancelled ack.
    const acks = ws.sent.map(s => { try { return JSON.parse(s); } catch { return null; } })
      .filter((e: { type?: string } | null): e is { type: string } => !!e && typeof e.type === 'string');
    expect(acks.some(e => e.type === 'response.cancelled')).toBe(true);
  });

  it('aborts in-flight processUAMP when ws closes mid-stream', async () => {
    const ctx = makeContext();
    const agent = makeMockAgent(ctx);
    const skill = new PortalTransportSkill();
    skill.setAgent(agent);

    const ws = makeWs();
    skill.handleConnection(ws as unknown as WebSocket, makeContext());

    ws.inject({
      type: 'uamp',
      events: [{ type: 'response.create', event_id: 'e1', timestamp: Date.now() }] as ClientEvent[],
    });

    await new Promise(r => setTimeout(r, 80));
    const beforeClose = ws.sent.length;
    expect(beforeClose).toBeGreaterThan(0);
    expect(beforeClose).toBeLessThan(50);

    ws.close();

    await new Promise(r => setTimeout(r, 200));
    // The loop should have torn down — additional sends limited to the
    // one in-flight chunk. Without ws.onclose abort it would drift up to 50.
    expect(ws.sent.length).toBeLessThan(50);
    expect(ws.sent.length - beforeClose).toBeLessThan(5);
  });
});
