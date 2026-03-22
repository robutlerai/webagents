import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { generateEventId } from '../../../src/uamp/events.js';

// ---------------------------------------------------------------------------
// Mock WebSocket
// ---------------------------------------------------------------------------

type WSListener = (...args: unknown[]) => void;

class MockWebSocket {
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = 0; // CONNECTING
  url: string;
  sent: string[] = [];

  private _listeners: Map<string, Set<WSListener>> = new Map();

  constructor(url: string) {
    this.url = url;
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this._emit('close');
  }

  addEventListener(event: string, handler: WSListener): void {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event)!.add(handler);
  }

  removeEventListener(event: string, handler: WSListener): void {
    this._listeners.get(event)?.delete(handler);
  }

  // --- helpers for tests ---

  _emit(event: string, ...args: unknown[]): void {
    if (event === 'open') {
      this.readyState = MockWebSocket.OPEN;
    }
    for (const handler of this._listeners.get(event) ?? []) {
      handler(...args);
    }
  }

  _serverSend(payload: object): void {
    const json = JSON.stringify(payload);
    this._emit('message', json);
  }
}

let lastCreatedWs: MockWebSocket | null = null;
let autoOpen = true;
let autoSessionCreated = true;

vi.mock('ws', () => ({
  default: class WS extends MockWebSocket {
    constructor(url: string) {
      super(url);
      lastCreatedWs = this;
      if (autoOpen) {
        queueMicrotask(() => this._emit('open'));
      }
    }
    override send(data: string): void {
      super.send(data);
      if (autoSessionCreated) {
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'session.create') {
            queueMicrotask(() =>
              this._serverSend({
                type: 'session.created',
                event_id: generateEventId(),
                session: parsed.session,
              }),
            );
          }
        } catch { /* non-json, ignore */ }
      }
    }
  },
}));

import { UAMPClient } from '../../../src/uamp/client.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getLastWs(): MockWebSocket {
  if (!lastCreatedWs) throw new Error('No WebSocket created yet');
  return lastCreatedWs;
}

function parseSent(ws: MockWebSocket): Array<{ type: string; [k: string]: unknown }> {
  return ws.sent.map(s => JSON.parse(s));
}

/**
 * Connect a client and flush microtasks so the WS open + session.created
 * handshake completes. Works with both real and fake timers.
 */
async function connectClient(client: UAMPClient): Promise<void> {
  const p = client.connect();
  // Flush microtasks: open → session.create sent → session.created received
  await new Promise(r => setTimeout(r, 0));
  await p;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('UAMPClient', () => {
  beforeEach(() => {
    lastCreatedWs = null;
    autoOpen = true;
    autoSessionCreated = true;
  });

  it('connects to server and sends session.create', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    expect(ws.url).toBe('ws://localhost:9000/ws');
    expect(client.isConnected).toBe(true);

    const events = parseSent(ws);
    expect(events.length).toBeGreaterThanOrEqual(1);
    expect(events[0].type).toBe('session.create');
    expect((events[0] as any).uamp_version).toBe('1.0');
  });

  it('sends input.text and response.create on sendInput()', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendInput('Hello agent');

    const events = parseSent(ws);
    expect(events.length).toBe(2);
    expect(events[0].type).toBe('input.text');
    expect((events[0] as any).text).toBe('Hello agent');
    expect((events[0] as any).role).toBe('user');
    expect(events[1].type).toBe('response.create');
  });

  it('sends response.cancel on cancel()', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.cancel();

    const events = parseSent(ws);
    expect(events.length).toBe(1);
    expect(events[0].type).toBe('response.cancel');
  });

  it('sends payment.submit on sendPayment()', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendPayment({ scheme: 'token', amount: '1.00', token: 'tok_abc' });

    const events = parseSent(ws);
    expect(events.length).toBe(1);
    expect(events[0].type).toBe('payment.submit');
    expect((events[0] as any).payment).toMatchObject({
      scheme: 'token',
      amount: '1.00',
      token: 'tok_abc',
    });
  });

  it('emits delta events from response.delta', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    const deltas: string[] = [];
    client.on('delta', (text) => deltas.push(text));

    ws._serverSend({
      type: 'response.delta',
      event_id: generateEventId(),
      response_id: 'resp-1',
      delta: { type: 'text', text: 'Hello ' },
    });

    ws._serverSend({
      type: 'response.delta',
      event_id: generateEventId(),
      response_id: 'resp-1',
      delta: { type: 'text', text: 'world' },
    });

    expect(deltas).toEqual(['Hello ', 'world']);
  });

  it('emits done events from response.done', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    const dones: Array<{ id: string; status: string }> = [];
    client.on('done', (resp) => dones.push(resp));

    ws._serverSend({
      type: 'response.done',
      event_id: generateEventId(),
      response_id: 'resp-1',
      response: {
        id: 'resp-1',
        status: 'completed',
        output: [{ type: 'text', text: 'Final answer' }],
        usage: { input_tokens: 10, output_tokens: 5 },
      },
    });

    expect(dones).toHaveLength(1);
    expect(dones[0].id).toBe('resp-1');
    expect(dones[0].status).toBe('completed');
  });

  it('emits error on response.error', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    const errors: Error[] = [];
    client.on('error', (err) => errors.push(err));

    ws._serverSend({
      type: 'response.error',
      event_id: generateEventId(),
      error: { code: 'rate_limit', message: 'Too many requests' },
    });

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toBe('Too many requests');
  });

  it('handles payment.required events', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    const payments: Array<{ amount: string; currency: string }> = [];
    client.on('paymentRequired', (req) => payments.push(req));

    ws._serverSend({
      type: 'payment.required',
      event_id: generateEventId(),
      requirements: {
        amount: '0.50',
        currency: 'USD',
        schemes: [{ scheme: 'token', network: 'robutler' }],
        reason: 'llm_usage',
      },
    });

    expect(payments).toHaveLength(1);
    expect(payments[0].amount).toBe('0.50');
    expect(payments[0].currency).toBe('USD');
  });

  it('auto-cancels and closes on AbortSignal abort', async () => {
    const ac = new AbortController();
    const client = new UAMPClient({
      url: 'ws://localhost:9000/ws',
      signal: ac.signal,
    });

    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    ac.abort();

    const events = parseSent(ws);
    const hasCancel = events.some(e => e.type === 'response.cancel');
    expect(hasCancel).toBe(true);
    expect(ws.readyState).toBe(MockWebSocket.CLOSED);
  });

  it('throws on connection timeout', async () => {
    autoOpen = false;
    autoSessionCreated = false;

    const client = new UAMPClient({
      url: 'ws://localhost:9000/ws',
      connectTimeout: 50,
    });

    // The timeout fires before session.created arrives. Either error message is valid
    // since the close() call in the timeout handler may race with the timeout rejection.
    await expect(client.connect()).rejects.toThrow(/Connection timeout|WebSocket closed/);
  });

  it('throws when sending on closed connection', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    client.close();

    await expect(client.sendInput('hi')).rejects.toThrow('WebSocket is not connected');
  });

  it('sendInput preserves content_id on input.image event', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendInput('see this', 'user', [
      { type: 'image', image: { url: '/api/content/abc-123' }, content_id: 'abc-123' } as any,
    ]);

    const events = parseSent(ws);
    const imgEvent = events.find(e => e.type === 'input.image');
    expect(imgEvent).toBeDefined();
    expect((imgEvent as any).content_id).toBe('abc-123');
  });

  it('sendInput preserves content_id on input.audio event', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendInput('', 'user', [
      { type: 'audio', audio: 'base64', format: 'mp3', content_id: 'aud-uuid' } as any,
    ]);

    const events = parseSent(ws);
    const audioEvent = events.find(e => e.type === 'input.audio');
    expect(audioEvent).toBeDefined();
    expect((audioEvent as any).content_id).toBe('aud-uuid');
  });

  it('sendInput preserves content_id on input.video event', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendInput('', 'user', [
      { type: 'video', video: { url: '/api/content/vid-1' }, content_id: 'vid-1' } as any,
    ]);

    const events = parseSent(ws);
    const vidEvent = events.find(e => e.type === 'input.video');
    expect(vidEvent).toBeDefined();
    expect((vidEvent as any).content_id).toBe('vid-1');
  });

  it('sendInput preserves content_id on input.file event', async () => {
    const client = new UAMPClient({ url: 'ws://localhost:9000/ws' });
    await connectClient(client);

    const ws = getLastWs();
    ws.sent = [];

    await client.sendInput('', 'user', [
      { type: 'file', file: { url: '/api/content/f-1' }, filename: 'doc.pdf', mime_type: 'application/pdf', content_id: 'f-1' } as any,
    ]);

    const events = parseSent(ws);
    const fileEvent = events.find(e => e.type === 'input.file');
    expect(fileEvent).toBeDefined();
    expect((fileEvent as any).content_id).toBe('f-1');
  });
});
