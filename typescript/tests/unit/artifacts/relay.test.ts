/**
 * The artifacts relay, driven by a stub platform over a local WebSocket:
 * the credit window is never exceeded, a cancel stops the read where it is,
 * frames of concurrent reads interleave, a dropped data connection comes
 * back and a re-asked range is served, and a read on the control connection
 * is answered on the control connection.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { randomBytes, createHash } from 'node:crypto';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';
import type { IncomingMessage } from 'node:http';

import { runPortalBridge } from '../../../src/portal/connect';
import type { IAgent } from '../../../src/core/types';
import {
  ArtifactRegistry,
  ArtifactRelay,
  decodeBlobDataFrame,
  encodeBlobDataFrame,
  files,
  BLOB_FRAME_MAX_BYTES,
} from '../../../src/artifacts';

function fakeAgentToken(): string {
  const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1' })}.x`;
}

function stubAgent(): IAgent {
  return {
    name: 'flow',
    getCapabilities: () => ({}) as never,
    processUAMP: (async function* () {})() as never,
    run: async () => ({ content: 'ok' }) as never,
    runStreaming: async function* () {
      yield { type: 'delta', delta: 'ok' } as never;
    },
  } as unknown as IAgent;
}

interface Conn {
  kind: 'control' | 'data';
  ws: WsSocket;
  headers: IncomingMessage['headers'];
  url: string;
  json: Array<Record<string, unknown>>;
  binary: Array<{ requestId: string; offset: number; payload: Uint8Array }>;
}

/** The platform's end: accepts control and data connections and records everything. */
async function startStubPortal() {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise<void>((resolve) => wss.on('listening', () => resolve()));
  const conns: Conn[] = [];
  const waiters: Array<{ test: () => boolean; resolve: () => void }> = [];
  const poke = () => {
    for (let i = waiters.length - 1; i >= 0; i -= 1) if (waiters[i]!.test()) waiters.splice(i, 1)[0]!.resolve();
  };
  wss.on('connection', (ws, req) => {
    const url = req.url ?? '';
    const kind: Conn['kind'] = req.headers['uamp-connection'] === 'data' || url.includes('conn=data') ? 'data' : 'control';
    const conn: Conn = { kind, ws, headers: req.headers, url, json: [], binary: [] };
    conns.push(conn);
    ws.on('message', (raw, isBinary) => {
      if (isBinary) {
        const decoded = decodeBlobDataFrame(new Uint8Array(raw as Buffer));
        if (decoded) conn.binary.push(decoded);
        poke();
        return;
      }
      const frame = JSON.parse(String(raw)) as Record<string, unknown>;
      conn.json.push(frame);
      if (frame.type === 'session.create') {
        ws.send(JSON.stringify({ type: 'session.created', session_id: 'sess_1', session: { agent: (frame.session as { agent: string }).agent } }));
      }
      if (frame.type === 'data.hello') ws.send(JSON.stringify({ type: 'data.ready' }));
      poke();
    });
    poke();
  });
  const until = (test: () => boolean, what: string, timeoutMs = 5000) =>
    new Promise<void>((resolve, reject) => {
      if (test()) { resolve(); return; }
      const timer = setTimeout(() => reject(new Error(`timeout waiting for ${what}`)), timeoutMs);
      waiters.push({ test, resolve: () => { clearTimeout(timer); resolve(); } });
    });
  return {
    port: (wss.address() as { port: number }).port,
    conns,
    until,
    latest(kind: Conn['kind']): Conn | undefined {
      return [...conns].reverse().find((c) => c.kind === kind && c.ws.readyState === c.ws.OPEN);
    },
    async close() {
      for (const c of conns) c.ws.terminate();
      await new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };
}

const closers: Array<() => Promise<void> | void> = [];
afterEach(async () => {
  for (const close of closers.splice(0).reverse()) await close();
});

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('blob.data framing', () => {
  it('round-trips request id, offset and payload, and refuses what is not a frame', () => {
    const payload = randomBytes(1000);
    const frame = encodeBlobDataFrame('req-1', 2 ** 40 + 7, payload);
    const back = decodeBlobDataFrame(frame)!;
    expect(back.requestId).toBe('req-1');
    expect(back.offset).toBe(2 ** 40 + 7);
    expect(Buffer.from(back.payload).equals(payload)).toBe(true);
    expect(decodeBlobDataFrame(new Uint8Array([2, 1, 0, 1, 65]))).toBeNull();
    expect(() => encodeBlobDataFrame('r', 0, new Uint8Array(BLOB_FRAME_MAX_BYTES + 1))).toThrow();
    expect(() => encodeBlobDataFrame('x'.repeat(65), 0, new Uint8Array(1))).toThrow();
  });
});

describe('the relay over a stub platform', () => {
  it('serves reads under credit, interleaves, cancels, survives a dropped data connection, and answers on the control connection too', async () => {
    const root = mkdtempSync(path.join(tmpdir(), 'artifacts-relay-'));
    closers.push(() => rmSync(root, { recursive: true, force: true }));
    const folder = path.join(root, 'out');
    rmSync(folder, { recursive: true, force: true });
    const { mkdirSync } = await import('node:fs');
    mkdirSync(folder, { recursive: true });
    const big = randomBytes(5 * BLOB_FRAME_MAX_BYTES + 777); // 6 frames when read whole
    const small = randomBytes(3 * 1024);
    writeFileSync(path.join(folder, 'big.bin'), big);
    writeFileSync(path.join(folder, 'small.bin'), small);
    const registry = new ArtifactRegistry();
    const published = await files.publish(folder, { producer: { agent: 'https://agent.example/agents/flow' }, registry, chunkSize: 262144 });
    const bigSha = published.set.entries.find((e) => e.path === 'big.bin')!.sha256;
    const smallSha = published.set.entries.find((e) => e.path === 'small.bin')!.sha256;
    const relay = new ArtifactRelay(registry);

    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const abort = new AbortController();
    const bridge = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}`,
      token: fakeAgentToken(),
      pingIntervalS: 3600,
      reconnectDelayS: 0.05,
      maxReconnectAttempts: 20,
      signal: abort.signal,
      artifacts: { relay },
    });
    closers.push(async () => { abort.abort(); await bridge; });

    // Control session, then the data connection the bridge opens by itself.
    await portal.until(() => portal.conns.some((c) => c.kind === 'control' && c.json.some((f) => f.type === 'session.create')), 'session.create');
    await portal.until(() => portal.conns.some((c) => c.kind === 'data' && c.json.some((f) => f.type === 'data.hello')), 'data.hello');
    const data = portal.latest('data')!;
    expect(data.headers['uamp-connection']).toBe('data');
    expect(data.url).toContain('conn=data');

    // 1. Credit is never exceeded: 2 frames granted, 6 needed.
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r1', sha256: bigSha, offset: 0, length: big.length, credit: 2 }));
    await portal.until(() => data.binary.filter((b) => b.requestId === 'r1').length >= 2, 'two frames');
    await sleep(100);
    expect(data.binary.filter((b) => b.requestId === 'r1')).toHaveLength(2);
    expect(data.json.some((f) => f.type === 'blob.end')).toBe(false);
    data.ws.send(JSON.stringify({ type: 'blob.credit', request_id: 'r1', frames: 2 }));
    await portal.until(() => data.binary.filter((b) => b.requestId === 'r1').length >= 4, 'four frames');
    await sleep(100);
    expect(data.binary.filter((b) => b.requestId === 'r1')).toHaveLength(4);
    data.ws.send(JSON.stringify({ type: 'blob.credit', request_id: 'r1', frames: 10 }));
    await portal.until(() => data.json.some((f) => f.type === 'blob.end' && f.request_id === 'r1'), 'blob.end r1');
    const r1 = data.binary.filter((b) => b.requestId === 'r1');
    expect(r1).toHaveLength(6);
    const assembled = Buffer.concat(r1.sort((a, b) => a.offset - b.offset).map((b) => Buffer.from(b.payload)));
    expect(assembled.equals(big)).toBe(true);
    const end = data.json.find((f) => f.type === 'blob.end' && f.request_id === 'r1')!;
    expect(end.bytes).toBe(big.length);

    // 2. Two concurrent reads interleave by request id; a small one is not starved.
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r2', sha256: bigSha, offset: 0, length: big.length, credit: 1000 }));
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r3', sha256: smallSha, offset: 0, length: small.length, credit: 1000 }));
    await portal.until(() => data.json.some((f) => f.type === 'blob.end' && f.request_id === 'r3') && data.json.some((f) => f.type === 'blob.end' && f.request_id === 'r2'), 'r2 and r3 end');
    const order = data.binary.filter((b) => b.requestId === 'r2' || b.requestId === 'r3').map((b) => b.requestId);
    const firstR3 = order.indexOf('r3');
    const lastR2 = order.lastIndexOf('r2');
    expect(firstR3).toBeGreaterThanOrEqual(0);
    expect(firstR3).toBeLessThan(lastR2); // the small read finished before the big one did
    expect(Buffer.from(data.binary.find((b) => b.requestId === 'r3')!.payload).equals(small)).toBe(true);

    // 3. Cancel mid-transfer: one frame of credit, then cancel; nothing more arrives.
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r4', sha256: bigSha, offset: 0, length: big.length, credit: 1 }));
    await portal.until(() => data.binary.some((b) => b.requestId === 'r4'), 'r4 first frame');
    data.ws.send(JSON.stringify({ type: 'blob.cancel', request_id: 'r4' }));
    data.ws.send(JSON.stringify({ type: 'blob.credit', request_id: 'r4', frames: 100 }));
    await sleep(150);
    expect(data.binary.filter((b) => b.requestId === 'r4')).toHaveLength(1);
    expect(data.json.some((f) => (f.type === 'blob.end' || f.type === 'blob.error') && f.request_id === 'r4')).toBe(false);
    expect(relay.activeReads).toBe(0);

    // 4. Refusals are named.
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r5', sha256: '00'.repeat(32), offset: 0, length: 10 }));
    await portal.until(() => data.json.some((f) => f.type === 'blob.error' && f.request_id === 'r5'), 'r5 error');
    expect((data.json.find((f) => f.type === 'blob.error' && f.request_id === 'r5')!.error as { code: string }).code).toBe('unknown_blob');
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r6', sha256: bigSha, offset: big.length - 5, length: 10 }));
    await portal.until(() => data.json.some((f) => f.type === 'blob.error' && f.request_id === 'r6'), 'r6 error');
    expect((data.json.find((f) => f.type === 'blob.error' && f.request_id === 'r6')!.error as { code: string }).code).toBe('range_invalid');

    // 5. The data connection drops mid-read; the agent reconnects it and serves the rest when asked again.
    data.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r7', sha256: bigSha, offset: 0, length: big.length, credit: 2 }));
    await portal.until(() => data.binary.filter((b) => b.requestId === 'r7').length >= 2, 'r7 two frames');
    const got = data.binary.filter((b) => b.requestId === 'r7').reduce((s, b) => s + b.payload.byteLength, 0);
    data.ws.terminate();
    await portal.until(() => portal.conns.filter((c) => c.kind === 'data').length >= 2 && portal.latest('data') !== undefined && portal.latest('data')!.json.some((f) => f.type === 'data.hello'), 'data reconnect');
    const data2 = portal.latest('data')!;
    expect(data2).not.toBe(data);
    data2.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'r8', sha256: bigSha, offset: got, length: big.length - got, credit: 100 }));
    await portal.until(() => data2.json.some((f) => f.type === 'blob.end' && f.request_id === 'r8'), 'r8 end');
    const rest = Buffer.concat(data2.binary.filter((b) => b.requestId === 'r8').sort((a, b) => a.offset - b.offset).map((b) => Buffer.from(b.payload)));
    expect(rest.equals(big.subarray(got))).toBe(true);
    expect(data2.binary.find((b) => b.requestId === 'r8')!.offset).toBe(got);

    // 6. A read on the control connection is answered on the control connection.
    const control = portal.latest('control')!;
    control.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'c1', sha256: smallSha, offset: 0, length: small.length }));
    await portal.until(() => control.json.some((f) => f.type === 'blob.end' && f.request_id === 'c1'), 'c1 end');
    expect(Buffer.from(control.binary.find((b) => b.requestId === 'c1')!.payload).equals(small)).toBe(true);
    expect(data2.binary.some((b) => b.requestId === 'c1')).toBe(false);

    // 7. The manifest itself is readable by its hash.
    control.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'm1', sha256: published.sha256, offset: 0, length: published.canonical.byteLength }));
    await portal.until(() => control.json.some((f) => f.type === 'blob.end' && f.request_id === 'm1'), 'm1 end');
    const manifestBytes = Buffer.concat(control.binary.filter((b) => b.requestId === 'm1').map((b) => Buffer.from(b.payload)));
    expect(createHash('sha256').update(manifestBytes).digest('hex')).toBe(published.sha256);

    // 8. Announcing a manifest on a task.
    const controlRelaySocket = { send: (d: string | Uint8Array) => { throw new Error(String(d).length ? 'unused' : 'unused'); } };
    void controlRelaySocket;
    const frame = relay.manifestFrame({ taskId: 'task-1', sha256: published.sha256, size: published.canonical.byteLength, locations: [{ kind: 'relay', agent: 'flow' }], progress: { done: 2, total: 2 } });
    expect(frame).toMatchObject({ type: 'artifact.manifest', task_id: 'task-1', sha256: published.sha256, progress: { done: 2, total: 2 } });
  }, 20000);
});
