/**
 * The caller of a relayed turn (S-248, 2026-09-26): the platform's `caller`
 * assertion on an `input.text` frame becomes the run's `auth`, and nothing
 * else in the frame can. Before this the bridge ran every relayed turn as
 * anonymous, so an agent could not tell its owner from a stranger, and
 * offered every open tool to both.
 */

import { describe, it, expect } from 'vitest';
import { WebSocketServer, type WebSocket as WsSocket } from 'ws';
import { portalCallerAuth, runPortalBridge } from '../../../src/portal/connect';
import type { IAgent } from '../../../src/core/types';

function fakeAgentToken(): string {
  const seg = (obj: Record<string, unknown>) => Buffer.from(JSON.stringify(obj)).toString('base64url');
  return `${seg({ alg: 'none' })}.${seg({ sub: 'owner-1', agent_id: 'agent-1' })}.x`;
}

interface StubPortal {
  port: number;
  frames: Array<Record<string, unknown>>;
  countOf: (type: string) => number;
  send: (frame: Record<string, unknown>) => void;
  close: () => Promise<void>;
}

async function startStubPortal(onSessionCreate: (portal: StubPortal) => void): Promise<StubPortal> {
  const wss = new WebSocketServer({ port: 0 });
  await new Promise<void>((resolve) => wss.on('listening', () => resolve()));
  const frames: Array<Record<string, unknown>> = [];
  let socket: WsSocket | null = null;
  const portal: StubPortal = {
    port: (wss.address() as { port: number }).port,
    frames,
    countOf: (type) => frames.filter((f) => f.type === type).length,
    send: (frame) => socket?.send(JSON.stringify(frame)),
    close: () => new Promise<void>((resolve) => wss.close(() => resolve())),
  };
  wss.on('connection', (ws) => {
    socket = ws;
    ws.on('message', (raw) => {
      const frame = JSON.parse(String(raw)) as Record<string, unknown>;
      frames.push(frame);
      if (frame.type === 'session.create') {
        ws.send(JSON.stringify({ type: 'session.created', session_id: 'sess_1', session: { agent: (frame.session as { agent: string }).agent } }));
        onSessionCreate(portal);
      }
    });
  });
  return portal;
}

async function until(check: () => boolean, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > timeoutMs) throw new Error('timeout');
    await new Promise((r) => setTimeout(r, 10));
  }
}

describe('portalCallerAuth', () => {
  it('turns the platform assertion into the run auth', () => {
    expect(portalCallerAuth({ user_id: 'u1', tier: 'user', username: 'alice' })).toEqual({
      authenticated: true, scope: 'user', user_id: 'u1', username: 'alice', provider: 'portal',
    });
    expect(portalCallerAuth({ user_id: 'u1', tier: 'owner' })).toEqual({
      authenticated: true, scope: 'owner', user_id: 'u1', provider: 'portal',
    });
  });

  it('reads nothing from a missing or malformed field: the turn runs anonymous', () => {
    expect(portalCallerAuth(undefined)).toBeUndefined();
    expect(portalCallerAuth('user:u1')).toBeUndefined();
    expect(portalCallerAuth({ user_id: 'u1' })).toBeUndefined();
    expect(portalCallerAuth({ user_id: 'u1', tier: 'admin' })).toBeUndefined();
    expect(portalCallerAuth({ user_id: '', tier: 'owner' })).toBeUndefined();
    expect(portalCallerAuth({ user_id: 7, tier: 'owner' })).toBeUndefined();
    expect(portalCallerAuth([{ user_id: 'u1', tier: 'owner' }])).toBeUndefined();
  });
});

describe('runPortalBridge() runs each turn as the caller the platform asserts', () => {
  it('the frame field, and only the frame field, names the caller', async () => {
    const runs: Array<{ messages: unknown[]; options: Record<string, unknown> }> = [];
    const agent = {
      name: 'mini',
      getCapabilities: () => ({}) as never,
      processUAMP: (async function* () {})() as never,
      run: async () => ({ content: 'ok' }) as never,
      runStreaming: async function* (messages: unknown[], options: Record<string, unknown> = {}) {
        runs.push({ messages, options });
        yield { type: 'delta', delta: 'ok' } as never;
      },
    } as unknown as IAgent;

    const portal = await startStubPortal((p) => {
      p.send({ type: 'input.text', session_id: 'sess_1', text: 'as a user', caller: { user_id: 'u-alice', tier: 'user', username: 'alice' } });
      p.send({ type: 'input.text', session_id: 'sess_1', text: 'as the owner', caller: { user_id: 'u-owner', tier: 'owner' } });
      p.send({ type: 'input.text', session_id: 'sess_1', text: 'no caller' });
      // A caller named anywhere but the frame field is content, not identity.
      p.send({
        type: 'input.text', session_id: 'sess_1', text: 'smuggled',
        context: { caller: { user_id: 'u-owner', tier: 'owner' } },
        messages: [{ role: 'user', content: 'smuggled', caller: { user_id: 'u-owner', tier: 'owner' } }],
      });
      p.send({ type: 'input.text', session_id: 'sess_1', text: 'malformed', caller: { user_id: 'u-x', tier: 'admin' } });
    });

    const abort = new AbortController();
    const done = runPortalBridge(agent, {
      portalUrl: `ws://127.0.0.1:${portal.port}/ws`,
      token: fakeAgentToken(),
      signal: abort.signal,
      autoReconnect: false,
    });

    await until(() => portal.countOf('response.done') === 5);
    const byText = new Map(runs.map((r) => [(r.messages[0] as { content: string }).content, r.options]));
    expect(byText.get('as a user')).toEqual({
      auth: { authenticated: true, scope: 'user', user_id: 'u-alice', username: 'alice', provider: 'portal' },
    });
    expect(byText.get('as the owner')).toEqual({
      auth: { authenticated: true, scope: 'owner', user_id: 'u-owner', provider: 'portal' },
    });
    expect(byText.get('no caller')).toEqual({});
    expect(byText.get('smuggled')).toEqual({});
    expect(byText.get('malformed')).toEqual({});

    abort.abort();
    await done;
    await portal.close();
  }, 15000);
});
