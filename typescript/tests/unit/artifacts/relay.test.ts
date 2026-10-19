/**
 * The artifacts relay, driven by a stub platform over a local WebSocket:
 * the credit window is never exceeded, a cancel stops the read where it is,
 * frames of concurrent reads interleave, a dropped data connection comes
 * back and a re-asked range is served, and a read on the control connection
 * is answered on the control connection. Announcing through the bridge: a set
 * announced after the session carries its id, one announced before waits for
 * it, a refusal reaches the caller, and a closed connection is no session.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { randomBytes, createHash } from 'node:crypto';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';
import type { IncomingMessage } from 'node:http';

import { runPortalBridge, portalTurnOf, PORTAL_TURN_METADATA_KEY } from '../../../src/portal/connect';
import type { IAgent } from '../../../src/core/types';
import {
  ArtifactAnnounceError,
  ArtifactRegistry,
  ArtifactRelay,
  canonicalManifest,
  decodeBlobDataFrame,
  encodeBlobDataFrame,
  files,
  BLOB_FRAME_MAX_BYTES,
} from '../../../src/artifacts';

function fakeAgentToken(extra: Record<string, unknown> = {}): string {
  const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1', ...extra })}.x`;
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

interface StubOptions {
  /** Answer an `artifact.manifest` with this refusal (or accept it in silence when null). */
  refuse?: (frame: Record<string, unknown>) => { code: string; message: string } | null;
}

/** The platform's end: accepts control and data connections and records everything. */
async function startStubPortal(stubOptions: StubOptions = {}) {
  const wss = new WebSocketServer({ port: 0 });
  let sessions = 0;
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
        sessions += 1;
        ws.send(JSON.stringify({ type: 'session.created', session_id: `sess_${sessions}`, session: { agent: (frame.session as { agent: string }).agent } }));
      }
      if (frame.type === 'artifact.manifest') {
        const refusal = stubOptions.refuse?.(frame) ?? null;
        if (refusal) ws.send(JSON.stringify({ type: 'artifact.error', event_id: frame.event_id, sha256: frame.sha256, error: refusal }));
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

/** Poll for state the agent side changes (the stub's `until` only re-checks on the stub's own events). */
async function waitFor(test: () => boolean, what: string, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!test()) {
    if (Date.now() > deadline) throw new Error(`timeout waiting for ${what}`);
    await sleep(5);
  }
}

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

    // 8. The announcement frame for a set on a task.
    const frame = relay.manifestFrame({ taskId: 'task-1', sha256: published.sha256, size: published.canonical.byteLength, locations: [{ kind: 'relay', agent: 'flow' }], progress: { done: 2, total: 2 } });
    expect(frame).toMatchObject({ type: 'artifact.manifest', task_id: 'task-1', sha256: published.sha256, progress: { done: 2, total: 2 } });
  }, 20000);
});

async function publishedFixture() {
  const root = mkdtempSync(path.join(tmpdir(), 'artifacts-announce-'));
  closers.push(() => rmSync(root, { recursive: true, force: true }));
  const { mkdirSync } = await import('node:fs');
  const folder = path.join(root, 'out');
  mkdirSync(folder, { recursive: true });
  writeFileSync(path.join(folder, 'a.bin'), randomBytes(4096));
  const registry = new ArtifactRegistry();
  const published = await files.publish(folder, { producer: { agent: 'https://agent.example/agents/flow', task: 'task-1' }, registry });
  return { registry, published, relay: new ArtifactRelay(registry) };
}

function startBridge(port: number, relay: ArtifactRelay) {
  const abort = new AbortController();
  const bridge = runPortalBridge(stubAgent(), {
    portalUrl: `ws://127.0.0.1:${port}`,
    token: fakeAgentToken(),
    pingIntervalS: 3600,
    reconnectDelayS: 0.05,
    maxReconnectAttempts: 20,
    signal: abort.signal,
    artifacts: { relay, dataConnection: false },
  });
  closers.push(async () => { abort.abort(); await bridge; });
}

const announcements = (portal: Awaited<ReturnType<typeof startStubPortal>>) =>
  portal.conns.filter((c) => c.kind === 'control').flatMap((c) => c.json.filter((f) => f.type === 'artifact.manifest'));

describe('announcing through the bridge', () => {
  it('announces on the attached session, with the manifest inlined and the relay as the default location', async () => {
    const { published, relay } = await publishedFixture();
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    startBridge(portal.port, relay);
    await waitFor(() => relay.controlSession !== null, 'the control session attached');

    const result = await relay.announceSet({ taskId: 'task-1', published, progress: { done: 1, total: 1 } });
    expect(result).toEqual({ sha256: published.sha256, sessionId: 'sess_1' });
    await portal.until(() => announcements(portal).length === 1, 'artifact.manifest');
    const frame = announcements(portal)[0]!;
    expect(frame).toMatchObject({
      type: 'artifact.manifest',
      session_id: 'sess_1',
      task_id: 'task-1',
      sha256: published.sha256,
      size: published.canonical.byteLength,
      locations: [{ kind: 'relay', agent: 'flow' }],
      progress: { done: 1, total: 1 },
    });
    // The inlined manifest canonicalises to exactly the bytes the hash names.
    expect(createHash('sha256').update(canonicalManifest(frame.manifest as never)).digest('hex')).toBe(published.sha256);

    // Without the manifest, and with locations of the caller's choosing.
    await relay.announceSet({ taskId: 'task-1', published, inline: false, locations: [{ kind: 'https', url: 'https://agent.example/agents/flow/artifacts/blobs/sha256:00' }] });
    await portal.until(() => announcements(portal).length === 2, 'second artifact.manifest');
    const second = announcements(portal)[1]!;
    expect(second.manifest).toBeUndefined();
    expect(second.locations).toEqual([{ kind: 'https', url: 'https://agent.example/agents/flow/artifacts/blobs/sha256:00' }]);
  }, 10000);

  it('waits for the session when the connection is not up yet, and refuses as not_connected after the wait', async () => {
    const { published, relay } = await publishedFixture();
    await expect(relay.announceSet({ taskId: 'task-1', published, connectTimeoutMs: 50 })).rejects.toMatchObject({ code: 'not_connected' });

    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const pending = relay.announceSet({ taskId: 'task-1', published });
    startBridge(portal.port, relay);
    await expect(pending).resolves.toEqual({ sha256: published.sha256, sessionId: 'sess_1' });
    await portal.until(() => announcements(portal).length === 1, 'artifact.manifest');
    const control = portal.latest('control')!;
    const order = control.json.map((f) => f.type);
    expect(order.indexOf('session.create')).toBeLessThan(order.indexOf('artifact.manifest'));
  }, 10000);

  it('hands a refusal to the caller: onAnnounceError always, and the announceSet promise within its confirm window', async () => {
    const { published, relay } = await publishedFixture();
    let refuse = true;
    const portal = await startStubPortal({ refuse: () => (refuse ? { code: 'forbidden', message: 'The manifest names another producer' } : null) });
    closers.push(() => portal.close());
    const heard: ArtifactAnnounceError[] = [];
    const stop = relay.onAnnounceError((err) => heard.push(err));
    startBridge(portal.port, relay);
    await waitFor(() => relay.controlSession !== null, 'the control session attached');

    const err = await relay.announceSet({ taskId: 'task-1', published, confirmWithinMs: 2000 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ArtifactAnnounceError);
    expect(err).toMatchObject({ code: 'forbidden', sha256: published.sha256, message: 'The manifest names another producer' });
    expect(heard.map((e) => e.code)).toEqual(['forbidden']);

    // Without a window the promise resolves on send; the refusal still reaches the listener.
    await relay.announceSet({ taskId: 'task-1', published });
    await waitFor(() => heard.length === 2, 'the second refusal');

    // Silence for the whole window is acceptance.
    refuse = false;
    await expect(relay.announceSet({ taskId: 'task-1', published, confirmWithinMs: 150 })).resolves.toMatchObject({ sha256: published.sha256 });
    stop();
    expect(heard).toHaveLength(2);
  }, 10000);

  it('PortalConnectSkill carries the relay, so an agent run by serve() serves and announces too', async () => {
    const { PortalConnectSkill } = await import('../../../src/skills/transport/portal-connect/skill');
    const { published, relay } = await publishedFixture();
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const skill = new PortalConnectSkill({
      portalUrl: `ws://127.0.0.1:${portal.port}`,
      token: fakeAgentToken(),
      artifacts: { relay, dataConnection: false },
      autostart: false,
    });
    skill.setAgent(stubAgent());
    await skill.start();
    closers.push(() => skill.stop());
    await waitFor(() => relay.controlSession !== null, 'the control session attached');
    await relay.announceSet({ taskId: 'task-1', published });
    await portal.until(() => announcements(portal).length === 1, 'artifact.manifest');
    const control = portal.latest('control')!;
    control.ws.send(JSON.stringify({ type: 'blob.read', request_id: 'm1', sha256: published.sha256, offset: 0, length: published.canonical.byteLength }));
    await portal.until(() => control.json.some((f) => f.type === 'blob.end' && f.request_id === 'm1'), 'the manifest read through the skill');
  }, 10000);

  it('names the relayed turn on the run, and announces on that turn when asked', async () => {
    const { published, relay } = await publishedFixture();
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const runs: Array<Record<string, unknown>> = [];
    const agent = {
      ...stubAgent(),
      runStreaming: async function* (_messages: unknown, options: Record<string, unknown> = {}) {
        runs.push(options);
        yield { type: 'delta', delta: 'ok' } as never;
      },
    } as unknown as IAgent;
    const abort = new AbortController();
    const bridge = runPortalBridge(agent, {
      portalUrl: `ws://127.0.0.1:${portal.port}`,
      token: fakeAgentToken(),
      pingIntervalS: 3600,
      signal: abort.signal,
      artifacts: { relay, dataConnection: false },
    });
    closers.push(async () => { abort.abort(); await bridge; });
    await waitFor(() => relay.controlSession !== null, 'the control session attached');

    // A relayed turn reaches the run with its session id on the run's metadata.
    portal.latest('control')!.ws.send(JSON.stringify({ type: 'input.text', session_id: 'turn-7', text: 'go' }));
    await waitFor(() => runs.length === 1, 'the relayed turn ran');
    expect((runs[0]!.metadata as Record<string, unknown>)[PORTAL_TURN_METADATA_KEY]).toEqual({ sessionId: 'turn-7' });
    expect(portalTurnOf({ metadata: runs[0]!.metadata })).toEqual({ sessionId: 'turn-7' });
    expect(portalTurnOf({ metadata: {} })).toBeNull();
    expect(portalTurnOf(null)).toBeNull();

    // Announced on that turn: the frame names the turn, and still goes out on the control connection.
    const res = await relay.announceSet({ taskId: 'task-1', published, sessionId: 'turn-7' });
    expect(res.sessionId).toBe('turn-7');
    await portal.until(() => announcements(portal).length === 1, 'artifact.manifest');
    expect(announcements(portal)[0]!.session_id).toBe('turn-7');
    // Without a turn it falls back to the control session.
    await relay.announceSet({ taskId: 'task-1', published });
    await portal.until(() => announcements(portal).length === 2, 'second artifact.manifest');
    expect(announcements(portal)[1]!.session_id).toBe('sess_1');
  }, 10000);

  it('names the agent as its key does on the control and the data connection alike', async () => {
    const { relay } = await publishedFixture();
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const abort = new AbortController();
    const bridge = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}`,
      token: fakeAgentToken({ agent_name: 'owner-1.flow' }),
      pingIntervalS: 3600,
      signal: abort.signal,
      artifacts: { relay },
    });
    closers.push(async () => { abort.abort(); await bridge; });
    await portal.until(() => portal.conns.some((c) => c.kind === 'data' && c.json.some((f) => f.type === 'data.hello')), 'data.hello');
    const create = portal.latest('control')!.json.find((f) => f.type === 'session.create')!;
    const hello = portal.latest('data')!.json.find((f) => f.type === 'data.hello')!;
    expect((create.session as { agent: string }).agent).toBe('owner-1.flow');
    expect(hello.agent).toBe('owner-1.flow');
    await waitFor(() => relay.controlSession?.agent === 'owner-1.flow', 'the session attached under the key name');
  }, 10000);

  it('forgets the session when the connection closes and announces on the next one, without re-announcing by itself', async () => {
    const { published, relay } = await publishedFixture();
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    startBridge(portal.port, relay);
    await waitFor(() => relay.controlSession !== null, 'the control session attached');
    await relay.announceSet({ taskId: 'task-1', published });
    await portal.until(() => announcements(portal).length === 1, 'artifact.manifest');

    portal.latest('control')!.ws.terminate();
    await waitFor(() => relay.controlSession?.sessionId === 'sess_2', 'the second session attached');
    await sleep(100);
    expect(announcements(portal)).toHaveLength(1);
    const again = await relay.announceSet({ taskId: 'task-1', published });
    expect(again.sessionId).toBe('sess_2');
    await portal.until(() => announcements(portal).length === 2, 'the announcement on the new session');
    expect(announcements(portal)[1]!.session_id).toBe('sess_2');
  }, 10000);
});
