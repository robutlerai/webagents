/**
 * A server that closes the socket mid-response is reported (2026-09-28).
 *
 * After the socket opened, a close only marked the client disconnected: a drain
 * or restart during a reply left the caller waiting out the 120 s response
 * timeout. The client now emits an error carrying the close code while a
 * response is awaited, and stays quiet once the response ended or when it
 * closed the socket itself.
 *
 * The helper waits for the client to attach its listeners, not for a number of
 * ticks (2026-11-13). The client's connect call first awaits `import('ws')`, which under
 * vitest goes through the main process's module loader; on a loaded node that
 * took more than the five ticks this file used to allow, so 'open' and
 * 'session.created' were fired at a socket nobody listened to yet and the
 * client's 10 s connect timer closed it ("WebSocket closed unexpectedly
 * (code=1000)", a red shard 1 on a four-shard gate).
 */
import { describe, expect, it, vi } from 'vitest';

type Listener = (...args: unknown[]) => void;

/** The socket the next `new WebSocket()` answers with (the client imports `ws`). */
const sockets = vi.hoisted(() => ({ next: null as unknown }));
vi.mock('ws', () => ({ default: vi.fn(() => sockets.next) }));

import { UAMPClient } from '../../../src/uamp/client';

function fakeSocket() {
  const listeners = new Map<string, Listener[]>();
  const fire = (event: string, arg?: unknown) => (listeners.get(event) ?? []).forEach((fn) => fn(arg));
  const ws = {
    readyState: 1,
    addEventListener: (event: string, fn: Listener) => listeners.set(event, [...(listeners.get(event) ?? []), fn]),
    removeEventListener: () => undefined,
    send: () => undefined,
    close: () => {
      ws.readyState = 3;
      fire('close', { code: 1000 });
    },
  };
  /** True once the client has wired the socket: the moment a real server's 'open' could reach it. */
  const wired = () => ['open', 'message', 'close'].every((e) => listeners.has(e));
  return { ws, fire, wired };
}

/** Waits until the client has wired its socket; bounded far below the client's own 10 s connect timeout. */
async function untilWired(wired: () => boolean): Promise<void> {
  const deadline = Date.now() + 5_000;
  while (!wired()) {
    if (Date.now() > deadline) throw new Error('the client never attached its socket listeners');
    await new Promise((r) => setTimeout(r, 1));
  }
}

async function openClient() {
  const { ws, fire, wired } = fakeSocket();
  sockets.next = ws;
  const client = new UAMPClient({ url: 'ws://test/llm' } as never);
  const errors: Error[] = [];
  client.on('error', (e: Error) => errors.push(e));
  const connected = client.connect();
  await untilWired(wired);
  fire('open');
  fire('message', { data: JSON.stringify({ type: 'session.created', event_id: 'x', session: { id: 's' } }) });
  await connected;
  return { client, fire, errors };
}

describe('UAMPClient and a server that goes away', () => {
  it('a close while a response is awaited is an error with the close code', async () => {
    const { client, fire, errors } = await openClient();
    await client.sendResponse({ messages: [] } as never);
    fire('close', { code: 1001, reason: 'server draining' });
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toBe('WebSocket closed unexpectedly (code=1001)');
    expect((errors[0] as Error & { code?: number }).code).toBe(1001);
  });

  it('a close after the response ended, or its own close, says nothing', async () => {
    const a = await openClient();
    await a.client.sendResponse({ messages: [] } as never);
    a.fire('message', { data: JSON.stringify({ type: 'response.done', event_id: 'd', response: { id: 'r', status: 'completed', output: [] } }) });
    a.fire('close', { code: 1001 });
    expect(a.errors).toHaveLength(0);

    const b = await openClient();
    await b.client.sendResponse({ messages: [] } as never);
    b.client.close();
    expect(b.errors).toHaveLength(0);
  });
});
