/**
 * `runPortalBridge()` — the reverse WS bridge, against a stub portal on
 * loopback. (Formerly exported as `connect()`; the documented entry point is
 * now `PortalConnectSkill`, which drives this loop.)
 *
 * Three behaviours that each fail SILENTLY when they regress:
 *  - the `workspace.terminal` envelope must reach a router (the portal sends
 *    it onto this very socket; an unhandled frame just vanishes);
 *  - a dropped socket must reconnect (resolving the bridge on the first
 *    close leaves a healthy-looking process that receives nothing);
 *  - a history that sanitises to nothing must fall back to the turn's text,
 *    never call the agent with an empty message list.
 */

import { describe, it, expect } from 'vitest';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';
import { runPortalBridge } from '../../../src/portal/connect';
import type { IAgent } from '../../../src/core/types';

function fakeAgentToken(): string {
  const seg = (obj: Record<string, unknown>) =>
    Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1' })}.x`;
}

interface StubPortal {
  port: number;
  frames: Array<Record<string, unknown>>;
  waitFor: (type: string, timeoutMs?: number) => Promise<Record<string, unknown>>;
  send: (frame: Record<string, unknown>) => void;
  dropSocket: () => void;
  close: () => Promise<void>;
}

/** A portal that answers session.create and records everything else. */
async function startStubPortal(
  onSessionCreate?: (portal: StubPortal) => void,
): Promise<StubPortal> {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise<void>((resolve) => wss.on('listening', () => resolve()));

  const frames: Array<Record<string, unknown>> = [];
  const waiters: Array<{ type: string; resolve: (f: Record<string, unknown>) => void }> = [];
  let socket: WsSocket | null = null;

  const portal: StubPortal = {
    port: (wss.address() as { port: number }).port,
    frames,
    waitFor(type, timeoutMs = 5000) {
      const existing = frames.find((f) => f.type === type);
      if (existing) return Promise.resolve(existing);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`timeout waiting for ${type}`)), timeoutMs);
        waiters.push({
          type,
          resolve: (f) => {
            clearTimeout(timer);
            resolve(f);
          },
        });
      });
    },
    send(frame) {
      socket?.send(JSON.stringify(frame));
    },
    dropSocket() {
      socket?.close();
    },
    async close() {
      await new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };

  wss.on('connection', (ws) => {
    socket = ws;
    ws.on('message', (raw) => {
      const frame = JSON.parse(String(raw)) as Record<string, unknown>;
      frames.push(frame);
      for (let i = waiters.length - 1; i >= 0; i--) {
        if (waiters[i].type === frame.type) {
          waiters.splice(i, 1)[0].resolve(frame);
        }
      }
      if (frame.type === 'session.create') {
        ws.send(
          JSON.stringify({
            type: 'session.created',
            session_id: 'sess_1',
            session: { agent: (frame.session as { agent: string }).agent },
          }),
        );
        onSessionCreate?.(portal);
      }
    });
  });

  return portal;
}

function stubAgent(
  onRun?: (messages: unknown[]) => void,
): IAgent {
  return {
    name: 'mini',
    getCapabilities: () => ({}) as never,
    processUAMP: (async function* () {})() as never,
    run: async () => ({ content: 'ok' }) as never,
    runStreaming: async function* (messages: unknown[]) {
      onRun?.(messages);
      yield { type: 'delta', delta: 'ok' } as never;
    },
  } as unknown as IAgent;
}

describe('runPortalBridge() workspace.terminal envelope', () => {
  it('routes the envelope to the terminal router and writes the reply back', async () => {
    const seen: unknown[] = [];
    const router = {
      async handlePayload(
        payload: unknown,
        send: (out: unknown) => void,
        opts?: { extension_version?: number },
      ) {
        seen.push({ payload, opts });
        send({ type: 'ready', session_id: 'term-1' });
      },
      async shutdown() {},
    };

    const portal = await startStubPortal((p) => {
      p.send({
        type: 'extension.message',
        namespace: 'workspace.terminal',
        extension_version: 1,
        payload: { type: 'open', session_id: 'term-1', peer_id: 'peer', cols: 80, rows: 24 },
      });
    });

    const abort = new AbortController();
    const done = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token: fakeAgentToken(),
      signal: abort.signal,
      terminal: router,
      autoReconnect: false,
    });

    const reply = await portal.waitFor('extension.message');
    expect(reply.namespace).toBe('workspace.terminal');
    expect((reply.payload as { type: string }).type).toBe('ready');
    expect(seen).toHaveLength(1);
    expect((seen[0] as { payload: { type: string } }).payload.type).toBe('open');

    abort.abort();
    await done;
    await portal.close();
  }, 15000);
});

describe('runPortalBridge() resilience', () => {
  it('reconnects after the socket drops', async () => {
    let sessionCreates = 0;
    const portal = await startStubPortal((p) => {
      sessionCreates += 1;
      if (sessionCreates === 1) p.dropSocket();
    });

    const abort = new AbortController();
    const done = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token: fakeAgentToken(),
      signal: abort.signal,
      reconnectDelayS: 0,
    });

    const deadline = Date.now() + 8000;
    while (sessionCreates < 2 && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 25));
    }
    expect(sessionCreates).toBeGreaterThanOrEqual(2);

    abort.abort();
    await done;
    await portal.close();
  }, 20000);

  it('never runs the agent with an empty message list', async () => {
    const runs: unknown[][] = [];
    const portal = await startStubPortal((p) => {
      p.send({
        type: 'input.text',
        session_id: 'sess_1',
        text: 'the actual turn',
        // Everything here sanitises away: tool rows and an empty assistant.
        messages: [
          { role: 'tool', content: 'tool output', tool_call_id: 't1' },
          { role: 'assistant', content: '' },
        ],
      });
    });

    const abort = new AbortController();
    const done = runPortalBridge(stubAgent((messages) => runs.push(messages as unknown[])), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token: fakeAgentToken(),
      signal: abort.signal,
      autoReconnect: false,
    });

    await portal.waitFor('response.done');
    expect(runs).toHaveLength(1);
    expect(runs[0]).toEqual([{ role: 'user', content: 'the actual turn' }]);

    abort.abort();
    await done;
    await portal.close();
  }, 15000);
});
