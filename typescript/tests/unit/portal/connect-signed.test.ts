/**
 * The bridge signs the `/ws` handshake when it has an identity and no token
 * (2026-09-23).
 *
 * Until this day `runPortalBridge()` refused at startup unless a per-agent
 * token was configured, although the very same agent proves its identity to
 * the platform's HTTP surface by signing (RFC 9421, the Web Bot Auth
 * profile, `src/crypto/http-signature.ts`). The platform's `/ws` upgrade now
 * accepts that signature too (portal `lib/ws/signed-upgrade.ts`), so:
 *
 *   - with an identity and no token, the handshake carries the three
 *     signature headers, no `?token=`, and `session.create` carries no token;
 *   - every connection attempt is signed afresh (a signature has a single-use
 *     nonce and a sixty-second window, so reusing one across a reconnect
 *     would be `signature_replayed` or `signature_expired`);
 *   - a token, when configured, is used exactly as before and wins over an
 *     identity beside it;
 *   - with NEITHER, the bridge refuses before any socket is opened, and the
 *     sentence names both ways in;
 *   - an identity the platform could not fetch a key set from (a loopback
 *     issuer) is refused at startup too, with the fix in the message, rather
 *     than as a bare 4001 from the platform;
 *   - `PortalConnectSkill` takes the identity from its config, or reads the
 *     one `serve()` handed the agent (`agent.identity`), so the stock
 *     `serve()` setup signs with no token configured at all.
 *
 * Against a stub portal on loopback that records each handshake's request
 * line and headers (the platform reads exactly these) and answers
 * `session.create` with `session.created`.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';
import type { IncomingMessage } from 'node:http';
import { runPortalBridge, PortalCredentialError, checkPortalCredential } from '../../../src/portal/connect';
import { PortalConnectSkill } from '../../../src/skills/transport/portal-connect/skill';
import { AgentIdentity } from '../../../src/crypto/identity';
import type { IAgent } from '../../../src/core/types';

const AGENT_URL = 'https://agent.example/agents/mini';
const KEY_SET_URL = `${AGENT_URL}/.well-known/jwks.json`;

function fakeAgentToken(): string {
  const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1' })}.x`;
}

interface Handshake {
  url: string;
  headers: IncomingMessage['headers'];
}

interface StubPortal {
  port: number;
  handshakes: Handshake[];
  frames: Array<Record<string, unknown>>;
  waitFor: (type: string, count?: number, timeoutMs?: number) => Promise<Record<string, unknown>>;
  dropSocket: () => void;
  close: () => Promise<void>;
}

/** A portal that records every handshake and every frame, and answers session.create. */
async function startStubPortal(): Promise<StubPortal> {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise<void>((resolve) => wss.on('listening', () => resolve()));

  const handshakes: Handshake[] = [];
  const frames: Array<Record<string, unknown>> = [];
  const waiters: Array<{ type: string; count: number; resolve: (f: Record<string, unknown>) => void }> = [];
  let socket: WsSocket | null = null;

  const portal: StubPortal = {
    port: (wss.address() as { port: number }).port,
    handshakes,
    frames,
    waitFor(type, count = 1, timeoutMs = 5000) {
      const matching = () => frames.filter((f) => f.type === type);
      if (matching().length >= count) return Promise.resolve(matching()[count - 1]!);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`timeout waiting for ${type} #${count}; saw ${JSON.stringify(frames.map((f) => f.type))}`)), timeoutMs);
        waiters.push({ type, count, resolve: (f) => { clearTimeout(timer); resolve(f); } });
      });
    },
    dropSocket() {
      socket?.close();
    },
    async close() {
      await new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };

  wss.on('connection', (ws, req) => {
    socket = ws;
    handshakes.push({ url: req.url ?? '', headers: req.headers });
    ws.on('message', (raw) => {
      const frame = JSON.parse(String(raw)) as Record<string, unknown>;
      frames.push(frame);
      const seen = frames.filter((f) => f.type === frame.type).length;
      for (let i = waiters.length - 1; i >= 0; i -= 1) {
        if (waiters[i]!.type === frame.type && seen >= waiters[i]!.count) waiters.splice(i, 1)[0]!.resolve(frame);
      }
      if (frame.type === 'session.create') {
        ws.send(JSON.stringify({ type: 'session.created', session_id: `sess_${seen}`, session: { agent: (frame.session as { agent: string }).agent } }));
      }
    });
  });

  return portal;
}

function stubAgent(): IAgent {
  return {
    name: 'mini',
    getCapabilities: () => ({}) as never,
    processUAMP: (async function* () {})() as never,
    run: async () => ({ content: 'ok' }) as never,
    runStreaming: async function* () {
      yield { type: 'delta', delta: 'ok' } as never;
    },
  } as unknown as IAgent;
}

async function identityFor(issuer = AGENT_URL): Promise<AgentIdentity> {
  const identity = new AgentIdentity({ agentId: 'mini', issuer });
  await identity.initialize();
  return identity;
}

const closers: Array<() => Promise<void> | void> = [];
afterEach(async () => {
  for (const close of closers.splice(0).reverse()) await close();
});

/** The three headers a signed handshake carries, plus the absence of `content-digest` (a GET has no body). */
function expectSigned(handshake: Handshake): void {
  expect(handshake.headers['signature-input']).toMatch(/^sig1=\("@method" "@authority" "@path" "@query" "signature-agent";key="sig1"\);created=\d+;expires=\d+;keyid="[^"]+";alg="ed25519";nonce="[^"]+";tag="web-bot-auth"$/);
  expect(handshake.headers.signature).toMatch(/^sig1=:[A-Za-z0-9+/]+=*:$/);
  expect(handshake.headers['signature-agent']).toBe(`sig1="${KEY_SET_URL}";type=jwks_uri`);
  expect(handshake.headers['content-digest']).toBeUndefined();
}

describe('runPortalBridge() with an identity and no token', () => {
  it('signs the handshake, sends no token anywhere, and signs every reconnect afresh', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const identity = await identityFor();

    const abort = new AbortController();
    const done = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      identity,
      signal: abort.signal,
      reconnectDelayS: 0,
    });
    closers.push(async () => { abort.abort(); await done; });

    const created = await portal.waitFor('session.create');
    expect(created.session).toEqual({ agent: 'mini' });
    expect(portal.handshakes).toHaveLength(1);
    expect(portal.handshakes[0]!.url).toBe('/ws');
    expectSigned(portal.handshakes[0]!);
    expect(portal.handshakes[0]!.headers.authorization).toBeUndefined();

    // The socket drops; the bridge reconnects with a NEW signature.
    portal.dropSocket();
    await portal.waitFor('session.create', 2);
    expect(portal.handshakes).toHaveLength(2);
    expectSigned(portal.handshakes[1]!);
    const nonce = (h: Handshake) => /nonce="([^"]+)"/.exec(String(h.headers['signature-input']))![1];
    expect(nonce(portal.handshakes[1]!)).not.toBe(nonce(portal.handshakes[0]!));
    expect(portal.handshakes[1]!.headers.signature).not.toBe(portal.handshakes[0]!.headers.signature);
  }, 15000);

  it('a configured token is used exactly as before and wins over the identity beside it', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const identity = await identityFor();
    const token = fakeAgentToken();

    const abort = new AbortController();
    const done = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token,
      identity,
      signal: abort.signal,
      autoReconnect: false,
    });
    closers.push(async () => { abort.abort(); await done; });

    const created = await portal.waitFor('session.create');
    expect(created.session).toEqual({ agent: 'mini', token });
    expect(portal.handshakes[0]!.url).toBe(`/ws?token=${encodeURIComponent(token)}`);
    expect(portal.handshakes[0]!.headers['signature-input']).toBeUndefined();
    expect(portal.handshakes[0]!.headers.signature).toBeUndefined();
  }, 15000);
});

describe('the credential guard: token, identity, or nothing', () => {
  it('neither a token nor an identity is refused before any socket is opened, naming both ways in', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const previous = process.env.WEBAGENTS_AGENT_TOKEN;
    delete process.env.WEBAGENTS_AGENT_TOKEN;
    try {
      await expect(runPortalBridge(stubAgent(), { portalUrl: `ws://127.0.0.1:${portal.port}/ws` })).rejects.toBeInstanceOf(PortalCredentialError);
      await expect(runPortalBridge(stubAgent(), { portalUrl: `ws://127.0.0.1:${portal.port}/ws` })).rejects.toThrow(/WEBAGENTS_AGENT_TOKEN/);
      await expect(runPortalBridge(stubAgent(), { portalUrl: `ws://127.0.0.1:${portal.port}/ws` })).rejects.toThrow(/sign/i);
      expect(portal.handshakes).toHaveLength(0);
    } finally {
      if (previous !== undefined) process.env.WEBAGENTS_AGENT_TOKEN = previous;
    }
  });

  it('an identity the platform cannot fetch a key set from (a loopback issuer) is refused at startup with the fix in the message', async () => {
    const identity = await identityFor('http://localhost:3000/agents/mini');
    expect(() => checkPortalCredential({ identity })).toThrow(PortalCredentialError);
    expect(() => checkPortalCredential({ identity })).toThrow(/WEBAGENTS_PUBLIC_URL/);
    expect(() => checkPortalCredential({ identity })).toThrow(/WEBAGENTS_AGENT_TOKEN/);
  });

  it('a public https identity passes, a bound token passes, and an unbound token is still refused as before', async () => {
    const identity = await identityFor();
    expect(() => checkPortalCredential({ identity: undefined, token: fakeAgentToken() })).not.toThrow();
    expect(() => checkPortalCredential({ identity })).not.toThrow();
    const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
    const ownerKey = `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1' })}.x`;
    expect(() => checkPortalCredential({ token: ownerKey })).toThrow(/api-key/);
  });
});

describe('PortalConnectSkill', () => {
  it('signs with the identity in its config when no token is configured', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const previous = process.env.WEBAGENTS_AGENT_TOKEN;
    delete process.env.WEBAGENTS_AGENT_TOKEN;
    const skill = new PortalConnectSkill({
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      identity: await identityFor(),
      autoReconnect: false,
    });
    skill.setAgent(stubAgent());
    closers.push(async () => {
      await skill.stop();
      if (previous !== undefined) process.env.WEBAGENTS_AGENT_TOKEN = previous;
    });

    await skill.start();
    const created = await portal.waitFor('session.create');
    expect(created.session).toEqual({ agent: 'mini' });
    expectSigned(portal.handshakes[0]!);
  }, 15000);

  it('reads the identity its host handed the agent (agent.identity), unless one was configured', async () => {
    const configured = await identityFor('https://configured.example/agents/mini');
    const served = await identityFor('https://served.example/agents/mini');

    const bare = new PortalConnectSkill({ portalUrl: 'ws://127.0.0.1:1/ws' });
    bare.setAgent(stubAgent());
    expect(bare.identity).toBeUndefined();
    bare.setAgent({ ...stubAgent(), identity: served } as IAgent);
    expect(bare.identity).toBe(served);

    const withConfig = new PortalConnectSkill({ portalUrl: 'ws://127.0.0.1:1/ws', identity: configured });
    withConfig.setAgent({ ...stubAgent(), identity: served } as IAgent);
    expect(withConfig.identity).toBe(configured);
  });

  it('under serve() with no token configured, the stock setup signs with the identity serve() persisted for the agent', async () => {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const previous = process.env.WEBAGENTS_AGENT_TOKEN;
    delete process.env.WEBAGENTS_AGENT_TOKEN;
    const keysDir = mkdtempSync(path.join(tmpdir(), 'webagents-signed-ws-'));
    closers.push(() => {
      rmSync(keysDir, { recursive: true, force: true });
      if (previous !== undefined) process.env.WEBAGENTS_AGENT_TOKEN = previous;
    });

    const { BaseAgent, serve } = await import('../../../src/index');
    const skill = new PortalConnectSkill({ portalUrl: `ws://127.0.0.1:${portal.port}/ws`, autoReconnect: false });
    const agent = new BaseAgent({ name: 'mini', instructions: 'You are helpful.', skills: [skill] });
    const server = await serve(agent, {
      port: 0,
      basePath: '/agents/mini',
      publicUrl: 'https://agent.example',
      keysDir,
      heartbeat: false,
    });
    closers.push(() => server.close());

    const created = await portal.waitFor('session.create');
    expect(created.session).toEqual({ agent: 'mini' });
    expectSigned(portal.handshakes[0]!);
    // The signature names the key set THIS server serves for the agent.
    expect(server.identity.keySetUrl).toBe(KEY_SET_URL);
    expect(/keyid="([^"]+)"/.exec(String(portal.handshakes[0]!.headers['signature-input']))![1]).toBe(server.identity.kid);
  }, 20000);
});

describe('the session is opened under the name the key was issued for (2026-09-24)', () => {
  // A per-agent key carries `agent_name` (`<owner>.<name>`) while the code calls
  // the agent `<name>`. Sending the local name got `session.error
  // agent_not_found` from a real portal and a socket that never received a
  // turn; found by walking the quickstart as a first-time developer.
  function tokenNaming(agentName?: string): string {
    const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
    const claims: Record<string, unknown> = { sub: 'owner-1', agent_id: 'agent-1' };
    if (agentName) claims.agent_name = agentName;
    return `${seg({ alg: 'none' })}.${seg(claims)}.x`;
  }

  async function sessionAgentFor(token: string): Promise<unknown> {
    const portal = await startStubPortal();
    closers.push(() => portal.close());
    const abort = new AbortController();
    const done = runPortalBridge(stubAgent(), {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token,
      signal: abort.signal,
      reconnectDelayS: 0,
    });
    closers.push(async () => { abort.abort(); await done; });
    const created = await portal.waitFor('session.create');
    return (created.session as { agent?: unknown }).agent;
  }

  it('uses the key\'s agent_name when it differs from the local name', async () => {
    // THE BUG: this was 'mini', which the platform does not know.
    expect(await sessionAgentFor(tokenNaming('owner.mini'))).toBe('owner.mini');
  }, 15000);

  it('keeps the local name for a key that carries no agent_name', async () => {
    expect(await sessionAgentFor(tokenNaming())).toBe('mini');
  }, 15000);
});
