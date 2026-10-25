/**
 * A bridge that opens its sockets with a signing identity (no token): the
 * platform rebuilds each handshake's signature base from the request it
 * actually receives (method, Host, path, query, Signature-Agent) and the
 * signature must verify against the agent's public key. The data connection
 * dials `?conn=data`, and `@query` is covered, so a signature made over the
 * bare socket URL fails here exactly as it fails at the platform.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import type { IncomingMessage } from 'node:http';
import { WebSocketServer } from 'ws';
import { exportJWK, generateKeyPair, type KeyLike } from 'jose';
import { AgentIdentity } from '../../../src/crypto/identity';
import { runPortalBridge } from '../../../src/portal/connect';
import { ArtifactRegistry, ArtifactRelay } from '../../../src/artifacts';
import type { IAgent } from '../../../src/core/types';

const closers: Array<() => unknown> = [];
afterEach(async () => {
  while (closers.length) await closers.pop()!();
});

function stubAgent(): IAgent {
  return {
    name: 'mini',
    initialize: async () => {},
    getCapabilities: () => ({}) as never,
    processUAMP: (async function* () {})() as never,
    run: async () => ({ content: 'ok' }) as never,
    runStreaming: async function* () {
      yield { type: 'delta', delta: 'ok' } as never;
    },
  } as unknown as IAgent;
}

/** The platform's check, rebuilt from the received request alone. */
function verifies(req: { headers: IncomingMessage['headers']; url: string }, publicJwk: object): boolean {
  const input = String(req.headers['signature-input'] ?? '');
  const sigHeader = String(req.headers['signature'] ?? '');
  const agentHeader = String(req.headers['signature-agent'] ?? '');
  const m = /^sig1=(.*)$/.exec(input);
  const s = /^sig1=:([^:]+):$/.exec(sigHeader);
  const a = /^sig1=(.*)$/.exec(agentHeader);
  if (!m || !s || !a) return false;
  const u = new URL(req.url, `http://${req.headers.host}`);
  const base = [
    `"@method": GET`,
    `"@authority": ${req.headers.host}`,
    `"@path": ${u.pathname}`,
    `"@query": ${u.search === '' ? '?' : u.search}`,
    `"signature-agent";key="sig1": ${a[1]}`,
    `"@signature-params": ${m[1]}`,
  ].join('\n');
  const key = createPublicKey({ key: publicJwk as never, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(base), key, Buffer.from(s[1]!, 'base64'));
}

describe('signed portal handshakes', () => {
  it('signs the control and the data connection over the exact URL each dials', async () => {
    const { privateKey, publicKey } = await generateKeyPair('EdDSA', { crv: 'Ed25519', extractable: true });
    const publicJwk = await exportJWK(publicKey);
    const identity = new AgentIdentity({
      agentId: 'mini',
      issuer: 'https://agent.example/agents/mini',
      privateKey: privateKey as KeyLike,
      publicKey: publicKey as KeyLike,
    });
    await identity.initialize();

    const wss = new WebSocketServer({ port: 0 });
    await new Promise<void>((resolve) => wss.on('listening', () => resolve()));
    closers.push(() => new Promise<void>((resolve) => wss.close(() => resolve())));
    const seen: Array<{ kind: string; headers: IncomingMessage['headers']; url: string }> = [];
    wss.on('connection', (ws, req) => {
      const url = req.url ?? '';
      seen.push({ kind: url.includes('conn=data') ? 'data' : 'control', headers: req.headers, url });
      ws.on('message', (raw) => {
        const frame = JSON.parse(String(raw)) as Record<string, unknown>;
        if (frame.type === 'session.create') {
          ws.send(JSON.stringify({ type: 'session.created', session_id: 'sess_1', session: { agent: 'mini' } }));
        }
        if (frame.type === 'data.hello') ws.send(JSON.stringify({ type: 'data.ready' }));
      });
    });

    const abort = new AbortController();
    const bridge = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${(wss.address() as { port: number }).port}`,
      identity,
      pingIntervalS: 3600,
      reconnectDelayS: 0.05,
      maxReconnectAttempts: 3,
      signal: abort.signal,
      artifacts: { relay: new ArtifactRelay(new ArtifactRegistry()) },
    });
    closers.push(async () => {
      abort.abort();
      await bridge;
    });

    const deadline = Date.now() + 5000;
    while (!(seen.some((c) => c.kind === 'control') && seen.some((c) => c.kind === 'data')) && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 20));
    }
    const control = seen.find((c) => c.kind === 'control');
    const data = seen.find((c) => c.kind === 'data');
    expect(control, 'a control connection').toBeDefined();
    expect(data, 'a data connection').toBeDefined();
    expect(data!.url).toContain('conn=data');
    expect(verifies(control!, publicJwk), 'the control handshake verifies').toBe(true);
    expect(verifies(data!, publicJwk), 'the data handshake verifies over ?conn=data').toBe(true);
  });
});
