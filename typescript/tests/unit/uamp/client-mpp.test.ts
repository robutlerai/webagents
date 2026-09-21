/**
 * The UAMP client's in-band purchase (machine-purchase design section 6.1
 * and 6.4; pass P9a, 2026-09-18). A `payment.required` whose `schemes`
 * carry an `mpp` entry beside the token scheme at index 0 is paid over
 * HTTP by the configured buyer at the entry's purchase URL, after which the
 * client sends `payment.submit` with scheme `balance` and the server
 * resumes the run. Without a buyer, or when the buyer refuses, the event
 * reaches `paymentRequired` listeners exactly as before.
 *
 * The `ws` mock is the one `client.test.ts` uses, copied rather than
 * shared: that file owns its module-level state and this suite must not
 * reach into it.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { generateEventId } from '../../../src/uamp/events.js';

type WSListener = (...args: unknown[]) => void;

class MockWebSocket {
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  readyState = 0;
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
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event)!.add(handler);
  }
  removeEventListener(event: string, handler: WSListener): void {
    this._listeners.get(event)?.delete(handler);
  }
  _emit(event: string, ...args: unknown[]): void {
    if (event === 'open') this.readyState = MockWebSocket.OPEN;
    for (const handler of this._listeners.get(event) ?? []) handler(...args);
  }
  _serverSend(payload: object): void {
    this._emit('message', JSON.stringify(payload));
  }
}

let lastCreatedWs: MockWebSocket | null = null;
let lastHandshake: { headers?: Record<string, string> } | undefined;

vi.mock('ws', () => ({
  default: class WS extends MockWebSocket {
    constructor(url: string, opts?: { headers?: Record<string, string> }) {
      super(url);
      lastCreatedWs = this;
      lastHandshake = opts;
      queueMicrotask(() => this._emit('open'));
    }
    override send(data: string): void {
      super.send(data);
      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'session.create') {
          queueMicrotask(() => this._serverSend({ type: 'session.created', event_id: generateEventId(), session: parsed.session }));
        }
      } catch {
        /* ignore */
      }
    }
  },
}));

import { UAMPClient, type UAMPInBandBuyer } from '../../../src/uamp/client.js';

const CHALLENGE = 'Payment id="abc", realm="robutler.ai", method="stripe", intent="charge", expires="2026-09-18T12:05:00.000Z", request="cmVx", opaque="b3Bh", header="Payment-Authorization"';
const PURCHASE_URL = 'https://robutler.ai/api/mpp/credits';
const TERMS = { url: 'https://robutler.ai/doc/terms-of-service', version: '2026-07-31' };

function paymentRequired(schemes: unknown[]) {
  return {
    type: 'payment.required',
    event_id: generateEventId(),
    requirements: { amount: '5.00', currency: 'USD', schemes, reason: 'tool_call' },
  };
}

const IN_BAND = [{ scheme: 'token' }, { scheme: 'mpp', challenge: CHALLENGE, purchase_url: PURCHASE_URL, terms: TERMS }];

async function connected(buyer?: UAMPInBandBuyer) {
  const client = new UAMPClient({ url: 'ws://localhost:9000/ws', ...(buyer ? { buyer } : {}) });
  const p = client.connect();
  await new Promise((r) => setTimeout(r, 0));
  await p;
  const ws = lastCreatedWs!;
  ws.sent = [];
  return { client, ws };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('UAMPClient in-band purchase', () => {
  beforeEach(() => {
    lastCreatedWs = null;
    lastHandshake = undefined;
  });

  // 2026-09-18: nothing signed the upgrade, so the platform's socket door
  // (which acts only on `Signature-Input` and no payment token) never saw a
  // signed agent and never sent an in-band `mpp` entry to an SDK client.
  it('signs the upgrade through the buyer, merged over the configured handshake headers', async () => {
    const upgradeHeaders = vi.fn(async () => ({ 'signature-input': 'sig1=("@method")', signature: 'sig1=:AA==:', 'signature-agent': 'sig1="https://agent.example/agents/mini"' }));
    const client = new UAMPClient({
      url: 'wss://robutler.ai/agents/acme/uamp',
      headers: { 'X-Chat-Id': 'chat-1' },
      buyer: { purchase: vi.fn(async () => ({ ok: true })), upgradeHeaders },
    });
    const p = client.connect();
    await flush();
    await p;
    expect(upgradeHeaders).toHaveBeenCalledWith('wss://robutler.ai/agents/acme/uamp');
    expect(lastHandshake?.headers).toEqual({
      'X-Chat-Id': 'chat-1',
      'signature-input': 'sig1=("@method")',
      signature: 'sig1=:AA==:',
      'signature-agent': 'sig1="https://agent.example/agents/mini"',
    });
  });

  it('never signs an upgrade that carries a payment token: the token is the caller\'s choice of how to pay', async () => {
    for (const config of [
      { paymentToken: 'jwt' },
      { headers: { 'X-Payment-Token': 'jwt' } },
      { url: 'wss://robutler.ai/agents/acme/uamp?payment_token=jwt' },
    ]) {
      const upgradeHeaders = vi.fn(async () => ({ signature: 'x' }));
      const client = new UAMPClient({ url: 'wss://robutler.ai/agents/acme/uamp', buyer: { purchase: vi.fn(async () => ({ ok: true })), upgradeHeaders }, ...config });
      const p = client.connect();
      await flush();
      await p;
      expect(upgradeHeaders).not.toHaveBeenCalled();
      expect(lastHandshake?.headers?.signature).toBeUndefined();
    }
  });

  it('hands the mpp entry to the buyer and resumes with payment.submit scheme balance; paymentRequired is not emitted', async () => {
    const purchase = vi.fn(async () => ({ ok: true, status: 200 }));
    const { client, ws } = await connected({ purchase });
    const required: unknown[] = [];
    const purchasing: unknown[] = [];
    client.on('paymentRequired', (r) => required.push(r));
    client.on('paymentPurchasing', (p) => purchasing.push(p));

    ws._serverSend(paymentRequired(IN_BAND));
    await flush();

    // The entry verbatim, plus the per-response key a buyer counts this call's purchases on.
    expect(purchase).toHaveBeenCalledWith({ url: PURCHASE_URL, challenge: CHALLENGE, terms: TERMS, call: expect.any(Object) });
    expect(purchasing).toEqual([{ purchase_url: PURCHASE_URL, amount: '5.00', currency: 'USD', terms: TERMS }]);
    expect(required).toEqual([]);
    const sent = ws.sent.map((s) => JSON.parse(s));
    expect(sent).toHaveLength(1);
    expect(sent[0]).toMatchObject({ type: 'payment.submit', payment: { scheme: 'balance', amount: '5.00' } });
  });

  it('a refusing buyer falls through to paymentRequired with the full schemes, and nothing is submitted', async () => {
    const purchase = vi.fn(async () => ({ ok: false, reason: 'over_daily_cap', detail: 'cap' }));
    const { client, ws } = await connected({ purchase });
    const required: Array<{ schemes: unknown[] }> = [];
    client.on('paymentRequired', (r) => required.push(r));

    ws._serverSend(paymentRequired(IN_BAND));
    await flush();

    expect(required).toHaveLength(1);
    expect(required[0].schemes).toEqual(IN_BAND);
    expect(ws.sent).toEqual([]);
  });

  it('a buyer that throws surfaces an error event', async () => {
    const purchase = vi.fn(async () => {
      throw new Error('wallet unreachable');
    });
    const { client, ws } = await connected({ purchase });
    const errors: Error[] = [];
    client.on('error', (e) => errors.push(e));
    ws._serverSend(paymentRequired(IN_BAND));
    await flush();
    expect(errors.map((e) => e.message)).toEqual(['wallet unreachable']);
    expect(ws.sent).toEqual([]);
  });

  it('without a buyer, or without a usable mpp entry, paymentRequired is emitted as before and the buyer is never asked', async () => {
    const noBuyer = await connected();
    const required: unknown[] = [];
    noBuyer.client.on('paymentRequired', (r) => required.push(r));
    noBuyer.ws._serverSend(paymentRequired(IN_BAND));
    await flush();
    expect(required).toHaveLength(1);
    expect(noBuyer.ws.sent).toEqual([]);

    const purchase = vi.fn(async () => ({ ok: true }));
    const { client, ws } = await connected({ purchase });
    const again: unknown[] = [];
    client.on('paymentRequired', (r) => again.push(r));
    ws._serverSend(paymentRequired([{ scheme: 'token' }]));
    ws._serverSend(paymentRequired([{ scheme: 'token' }, { scheme: 'mpp', challenge: CHALLENGE }]));
    ws._serverSend(paymentRequired([{ scheme: 'token' }, { scheme: 'mpp', purchase_url: PURCHASE_URL, challenge: '  ' }]));
    await flush();
    expect(purchase).not.toHaveBeenCalled();
    expect(again).toHaveLength(3);
    expect(ws.sent).toEqual([]);
  });

  // 2026-09-19, THE PURCHASE POINTER: the `/llm` socket tells a token holder
  // whose token ran dry where to buy, with an `mpp` entry that carries NO
  // challenge (nobody on that rail was verified). The client ignored it.
  describe('the purchase pointer: an mpp entry with a purchase URL and no challenge', () => {
    const POINTER = [{ scheme: 'token' }, { scheme: 'mpp', purchase_url: PURCHASE_URL }];

    it('is handed to purchaseAt with this socket as its source, and a session with no token resumes with scheme balance', async () => {
      const purchase = vi.fn(async () => ({ ok: true }));
      const purchaseAt = vi.fn(async () => ({ ok: true, status: 200 }));
      const { client, ws } = await connected({ purchase, purchaseAt });
      const required: unknown[] = [];
      const purchasing: unknown[] = [];
      client.on('paymentRequired', (r) => required.push(r));
      client.on('paymentPurchasing', (p) => purchasing.push(p));

      ws._serverSend(paymentRequired(POINTER));
      await flush();

      expect(purchaseAt).toHaveBeenCalledWith({ url: PURCHASE_URL, from: 'ws://localhost:9000/ws', call: expect.any(Object) });
      expect(purchase).not.toHaveBeenCalled();
      expect(purchasing).toEqual([{ purchase_url: PURCHASE_URL, amount: '5.00', currency: 'USD' }]);
      expect(required).toEqual([]);
      const sent = ws.sent.map((s) => JSON.parse(s));
      expect(sent).toHaveLength(1);
      expect(sent[0]).toMatchObject({ type: 'payment.submit', payment: { scheme: 'balance', amount: '5.00' } });
    });

    it('a session that pays by token hands the event to its listeners once the purchase is made: they own the token', async () => {
      const purchaseAt = vi.fn(async () => ({ ok: true }));
      const client = new UAMPClient({ url: 'wss://robutler.ai/llm', paymentToken: 'jwt', buyer: { purchase: vi.fn(async () => ({ ok: true })), purchaseAt } });
      const p = client.connect();
      await flush();
      await p;
      const ws = lastCreatedWs!;
      ws.sent = [];
      const required: Array<{ purchased?: boolean; schemes: unknown[] }> = [];
      client.on('paymentRequired', (r) => required.push(r));

      ws._serverSend(paymentRequired(POINTER));
      await flush();

      expect(purchaseAt).toHaveBeenCalledWith({ url: PURCHASE_URL, from: 'wss://robutler.ai/llm', call: expect.any(Object) });
      expect(required).toHaveLength(1);
      expect(required[0].purchased).toBe(true);
      expect(required[0].schemes).toEqual(POINTER);
      // The client submits nothing on its own: `balance` is not a scheme that socket resumes on.
      expect(ws.sent).toEqual([]);
    });

    it('a refusing buyer falls through to the listeners, not marked purchased, and nothing is submitted', async () => {
      const purchaseAt = vi.fn(async () => ({ ok: false, reason: 'realm_not_allowed', detail: 'off the allowlist' }));
      const { client, ws } = await connected({ purchase: vi.fn(async () => ({ ok: true })), purchaseAt });
      const required: Array<{ purchased?: boolean }> = [];
      client.on('paymentRequired', (r) => required.push(r));
      ws._serverSend(paymentRequired(POINTER));
      await flush();
      expect(required).toHaveLength(1);
      expect(required[0].purchased).toBeUndefined();
      expect(ws.sent).toEqual([]);
    });

    it('a buyer that cannot follow a pointer is never handed one: today\'s behaviour exactly', async () => {
      const purchase = vi.fn(async () => ({ ok: true }));
      const { client, ws } = await connected({ purchase });
      const required: unknown[] = [];
      const purchasing: unknown[] = [];
      client.on('paymentRequired', (r) => required.push(r));
      client.on('paymentPurchasing', (p) => purchasing.push(p));
      ws._serverSend(paymentRequired(POINTER));
      await flush();
      expect(purchase).not.toHaveBeenCalled();
      expect(purchasing).toEqual([]);
      expect(required).toHaveLength(1);
      expect(ws.sent).toEqual([]);
    });

    it('a malformed challenge is not a pointer, and an entry with no purchase URL is nothing', async () => {
      const purchaseAt = vi.fn(async () => ({ ok: true }));
      const purchase = vi.fn(async () => ({ ok: true }));
      const { client, ws } = await connected({ purchase, purchaseAt });
      const required: unknown[] = [];
      client.on('paymentRequired', (r) => required.push(r));
      ws._serverSend(paymentRequired([{ scheme: 'token' }, { scheme: 'mpp', purchase_url: PURCHASE_URL, challenge: '  ' }]));
      ws._serverSend(paymentRequired([{ scheme: 'token' }, { scheme: 'mpp', purchase_url: PURCHASE_URL, challenge: 7 }]));
      ws._serverSend(paymentRequired([{ scheme: 'token' }, { scheme: 'mpp' }]));
      await flush();
      expect(purchaseAt).not.toHaveBeenCalled();
      expect(purchase).not.toHaveBeenCalled();
      expect(required).toHaveLength(3);
    });

    it('every purchase of one response carries the same call key, and the next response a new one', async () => {
      const calls: unknown[] = [];
      const purchaseAt = vi.fn(async (r: { call?: object }) => {
        calls.push(r.call);
        return { ok: true };
      });
      const { client, ws } = await connected({ purchase: vi.fn(async () => ({ ok: true })), purchaseAt });
      await client.sendInput('one');
      ws._serverSend(paymentRequired(POINTER));
      ws._serverSend(paymentRequired(POINTER));
      await flush();
      await client.sendInput('two');
      ws._serverSend(paymentRequired(POINTER));
      await flush();
      expect(calls).toHaveLength(3);
      expect(calls[0]).toBe(calls[1]);
      expect(calls[2]).not.toBe(calls[0]);
    });
  });

  it('keeps the token scheme at index 0 of what listeners see (the positional readers of both SDKs)', async () => {
    const purchase = vi.fn(async () => ({ ok: false, reason: 'terms_refused' }));
    const { client, ws } = await connected({ purchase });
    const required: Array<{ schemes: Array<{ scheme: string }> }> = [];
    client.on('paymentRequired', (r) => required.push(r));
    ws._serverSend(paymentRequired(IN_BAND));
    await flush();
    expect(required[0].schemes[0]).toEqual({ scheme: 'token' });
  });
});
