/**
 * Every 401 this SDK's servers answer carries the bearer challenge
 * (2026-09-29), against the shared fixture
 * `python/tests/fixtures/credential_floor/www_authenticate.json`, which the
 * Python suite reads too (`tests/server/test_every_401_challenges_noticed3.py`).
 *
 * `mcp serve --http` got its `WWW-Authenticate` in the morning; that afternoon
 * every other 401 was found bare: the floor's own refusal on `chat/completions`
 * and the `command` paths, the gate's `no_caller` on a scoped endpoint, an
 * auth hook's refusal on `chat/completions`, the scoped websocket refusal,
 * the raw `HTTP/1.1 401` on a billable websocket upgrade, the daemon's
 * registry routes. RFC 7235 makes the header a MUST on every 401, and an MCP
 * or OAuth-shaped client reads the scheme from it.
 *
 * Pinned here, door by door:
 *
 *   - the values are the fixture's, and the MCP fixture's `refusals` say the
 *     same (one rule, two fixtures, no drift);
 *   - the variant follows the ONE rule: `bad_credential` when the request
 *     carried a credential (`hasCredential`), whoever refused it, and
 *     `no_credential` when it carried none;
 *   - every door the fixture says this SDK serves is probed, and every probe
 *     is a door the fixture names, so a new 401 fails here until it is listed.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, hook, http as endpoint, websocket } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { NEEDS_CALLER, refusalResponse } from '../../../src/server/endpoint-gate';
import {
  BEARER_REALM,
  bearerChallenge,
  challengeFor,
  challengeHeaders,
  refusalHeaders,
  unauthorizedResponse,
  unauthorizedUpgradeReply,
} from '../../../src/server/credential-floor';
import { UAMPTransportSkill } from '../../../src/skills/transport/uamp/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const CHALLENGE = JSON.parse(readFileSync(path.join(FIXTURES, 'credential_floor/www_authenticate.json'), 'utf8')) as {
  header: string;
  realm: string;
  no_credential: string;
  bad_credential: string;
  doors: Array<{ name: string; challenge: 'no_credential' | 'bad_credential'; typescript: string | null; typescript_why?: string }>;
};
const SERVE = JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/serve.json'), 'utf8')) as {
  refusals: { no_credential: { status: number; www_authenticate: string }; bad_credential: { status: number; www_authenticate: string } };
};
const HEADER = CHALLENGE.header.toLowerCase();
const NONE = CHALLENGE.no_credential;
const BAD = CHALLENGE.bad_credential;
const DOORS = new Map(CHALLENGE.doors.map((door) => [door.name, door]));
const BASE = '/agents/guarded';
const BODY = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] });

/** An auth skill's refusal, matched by name as the gate matches it (`isAuthError`). */
function refused(message: string): Error {
  return Object.assign(new Error(message), { name: 'AuthenticationError', statusCode: 401, code: 'unauthorized' });
}

/**
 * Stands in for an auth skill: `Bearer owner-token` is the owner,
 * `Bearer bad-token` is refused with the auth skill's own error, anything
 * else verifies no one.
 */
class Identity extends Skill {
  static readonly identifiesCaller = true;

  constructor() {
    super({ name: 'identity' });
  }

  @hook({ lifecycle: 'on_connection', priority: 0 })
  async who(_data: HookData, context: Context): Promise<void> {
    // The servers put the credential headers on `metadata`; the daemon puts
    // the request in session data only (ADR-0045), where the access skill reads it.
    const inbound = context.get<{ headers?: Record<string, string> }>('_inboundRequest');
    const bearer = String(context.metadata?.authorization ?? inbound?.headers?.authorization ?? '');
    if (bearer === 'Bearer bad-token') throw refused('this bearer is refused');
    if (bearer === 'Bearer owner-token') {
      (context as unknown as { setAuth(auth: unknown): void }).setAuth({ authenticated: true, scope: 'owner', user_id: 'owner-1' });
    }
  }
}

/** A model that answers `ok`, so a turn that gets past the hooks ends. */
class Model extends Skill {
  constructor() {
    super({ name: 'model' });
  }

  @handoff({ name: 'model' })
  async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent> {
    const responseId = generateEventId();
    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
    yield createResponseDeltaEvent(responseId, { type: 'text', text: 'ok' });
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'ok' }]);
  }
}

class Doors extends Skill {
  constructor() {
    super({ name: 'doors' });
  }

  @endpoint({ path: '/mine', method: 'GET', scopes: ['owner'] })
  async mine(): Promise<Response> {
    return Response.json({ mine: true });
  }

  @websocket({ path: '/live', scopes: ['owner'] })
  live(ws: WebSocket): void {
    ws.send(JSON.stringify({ live: true }));
    ws.close();
  }
}

function agent(): BaseAgent {
  return new BaseAgent({ name: 'guarded', instructions: 'Guarded.', skills: [new Doors(), new Identity(), new Model()] });
}

type Fetch = (request: Request) => Promise<Response>;

const SERVERS: Array<[string, (a: BaseAgent) => Promise<Fetch>]> = [
  ['createFetchHandler', async (a) => createFetchHandler(a, { basePath: BASE })],
  ['serve()', async (a) => {
    const { app } = createAgentApp(a, { basePath: BASE, logging: false });
    return (request) => Promise.resolve(app.fetch(request));
  }],
  ['WebAgentsServer', async (a) => {
    const server = new WebAgentsServer({ logging: false });
    await server.addAgent('guarded', a);
    return (request) => Promise.resolve(server.getApp().fetch(request));
  }],
];

async function everyServer(probe: (fetch: Fetch, name: string) => Promise<void>): Promise<void> {
  for (const [name, build] of SERVERS) await probe(await build(agent()), name);
}

function request(sub: string, init: RequestInit = {}, headers: Record<string, string> = {}): Request {
  return new Request(`http://agent.test${BASE}${sub}`, { ...init, headers: { 'content-type': 'application/json', ...headers } });
}

async function expectChallenge(res: Response, challenge: string, where: string): Promise<void> {
  expect(res.status, where).toBe(401);
  expect(res.headers.get(HEADER), where).toBe(challenge);
}

function daemon(): WebAgentsDaemon {
  const d = new WebAgentsDaemon({ port: 0, watch: false, cron: false, healthChecks: false });
  d.registry.registerLocal(agent() as never);
  return d;
}

function daemonCall(d: WebAgentsDaemon, sub: string, init: RequestInit = {}, headers: Record<string, string> = {}): Promise<Response> {
  return d.app.fetch(new Request(`http://localhost${sub}`, { ...init, headers: { 'content-type': 'application/json', ...headers } }));
}

// -- the values and the rule -------------------------------------------------------------------

describe('the challenge values and the one rule', () => {
  it("the values are the fixture's, and the MCP fixture agrees", () => {
    expect(BEARER_REALM).toBe(CHALLENGE.realm);
    expect(bearerChallenge()).toBe(NONE);
    expect(bearerChallenge(true)).toBe(BAD);
    expect(SERVE.refusals.no_credential.www_authenticate).toBe(NONE);
    expect(SERVE.refusals.bad_credential.www_authenticate).toBe(BAD);
    expect(challengeHeaders()).toEqual({ 'WWW-Authenticate': NONE });
    expect(challengeHeaders(true)).toEqual({ 'WWW-Authenticate': BAD });
  });

  it('the rule reads the request', () => {
    const headers = (record: Record<string, string>) => ({ headers: { get: (name: string) => record[name.toLowerCase()] ?? null } });
    expect(challengeFor(headers({}))).toBe(NONE);
    expect(challengeFor(headers({ authorization: 'Bearer' }))).toBe(NONE); // a bare Bearer is not a credential
    expect(challengeFor(headers({ authorization: 'Bearer x' }))).toBe(BAD);
    expect(challengeFor(headers({ 'x-api-key': 'k' }))).toBe(BAD);
    expect(challengeFor(headers({ 'signature-input': 'sig1=()' }))).toBe(BAD);
    expect(refusalHeaders(401, headers({}))).toEqual({ 'WWW-Authenticate': NONE });
    expect(refusalHeaders(401, headers({ authorization: 'Bearer x' }))).toEqual({ 'WWW-Authenticate': BAD });
    expect(refusalHeaders(403, headers({ authorization: 'Bearer x' }))).toEqual({});
    // The floor's own response carries the plain challenge without being asked, and a caller may name the variant.
    expect(unauthorizedResponse().headers.get(HEADER)).toBe(NONE);
    expect(unauthorizedResponse({ 'WWW-Authenticate': BAD }).headers.get(HEADER)).toBe(BAD);
    expect(unauthorizedUpgradeReply()).toBe(`HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: ${NONE}\r\n\r\n`);
    // A gate refusal built for a request carries the rule's variant; a 403 carries none; without a request an auth error is a presented credential's.
    expect(refusalResponse(refused('x'), new Request('http://a.test/', { headers: { authorization: 'Bearer x' } }))?.headers).toEqual({ 'WWW-Authenticate': BAD });
    expect(refusalResponse(refused('x'), new Request('http://a.test/'))?.headers).toEqual({ 'WWW-Authenticate': NONE });
    expect(refusalResponse(refused('x'))?.headers).toEqual({ 'WWW-Authenticate': BAD });
    expect(refusalResponse(Object.assign(new Error('no'), { name: 'AuthorizationError', statusCode: 403, code: 'forbidden' }))?.headers).toEqual({});
    expect(refusalResponse(new Error('other'))).toBeNull();
  });
});

// -- the doors ---------------------------------------------------------------------------------

type Upgrade = (req: http.IncomingMessage, socket: import('stream').Duplex, head: Buffer) => void;

const SOCKET_SERVERS: Array<[string, (a: BaseAgent) => Promise<Upgrade>]> = [
  ['serve()', async (a) => createAgentApp(a, { basePath: BASE, logging: false }).handleUpgrade],
  ['WebAgentsServer', async (a) => {
    const server = new WebAgentsServer({ logging: false });
    await server.addAgent('guarded', a);
    const upgrade = (server as unknown as { handleWebSocketUpgrade: Upgrade }).handleWebSocketUpgrade;
    return upgrade.bind(server);
  }],
];

const openServers: http.Server[] = [];

afterEach(async () => {
  for (const server of openServers.splice(0)) await new Promise<void>((resolve) => server.close(() => resolve()));
});

/** Open a socket against `upgrade` and report the refusal's status and headers. */
async function refusedUpgrade(upgrade: Upgrade, sub: string, headers: Record<string, string> = {}): Promise<{ status?: number; challenge?: string }> {
  const server = http.createServer();
  openServers.push(server);
  server.on('upgrade', upgrade);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
  const { port } = server.address() as AddressInfo;
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}${BASE}${sub}`, { headers });
    ws.on('message', () => {
      resolve({});
      ws.close();
    });
    ws.on('unexpected-response', (_req, res) => {
      res.resume();
      res.on('end', () => resolve({ status: res.statusCode, challenge: String(res.headers[HEADER] ?? '') }));
    });
    ws.on('error', (error) => reject(error));
  });
}

/** The raw bytes `handleUpgrade` writes for an upgrade, on a hand-made socket (as `ws-upgrade-esm.test.ts` reads them). */
async function rawUpgradeReply(upgrade: Upgrade, url: string): Promise<string> {
  const written: string[] = [];
  const socket = {
    write: (chunk: string) => { written.push(String(chunk)); return true; },
    destroy: () => {},
    on: () => socket,
    removeListener: () => socket,
    readable: true,
    writable: true,
  };
  const req = { url, method: 'GET', headers: { host: 'agent.example.com', upgrade: 'websocket', connection: 'Upgrade' } };
  try {
    upgrade(req as never, socket as never, Buffer.alloc(0));
  } catch {
    // `ws` may reject the hand-made socket further in; that is past the point under test.
  }
  await new Promise((resolve) => setTimeout(resolve, 20));
  return written.join('');
}

const PROBES: Record<string, () => Promise<void>> = {
  async floor_billable() {
    await everyServer(async (fetch, name) => {
      await expectChallenge(await fetch(request('/chat/completions', { method: 'POST', body: BODY })), NONE, name);
    });
    await expectChallenge(await daemonCall(daemon(), '/agents/guarded/chat/completions', { method: 'POST', body: BODY }), NONE, 'daemon');
  },

  async floor_credentialed() {
    await everyServer(async (fetch, name) => {
      await expectChallenge(await fetch(request('/command/any/thing', { method: 'POST', body: '{}' })), NONE, name);
      await expectChallenge(await fetch(request('/command')), NONE, name);
    });
  },

  async completions_refused_credential() {
    // `WebAgentsServer` included since S-345 (2026-09-29): its fallback
    // `chat/completions` route called `agent.run(msgs)` with no request
    // metadata, so no identity skill saw the credential there and this probe
    // had to leave it out. Every server hands the run the same request now
    // (`handler.ts` `servedRunOptions`), and the daemon's metadata with it.
    for (const stream of [false, true]) {
      for (const [name, build] of SERVERS) {
        const fetch = await build(agent());
        const body = JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream });
        const res = await fetch(request('/chat/completions', { method: 'POST', body }, { authorization: 'Bearer bad-token' }));
        await expectChallenge(res, BAD, `${name} stream=${stream}`);
        expect(((await res.json()) as { error: { code: string } }).error.code).toBe('unauthorized');
      }
      const res = await daemonCall(daemon(), '/agents/guarded/chat/completions', { method: 'POST', body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream }) }, { authorization: 'Bearer bad-token' });
      await expectChallenge(res, BAD, `daemon stream=${stream}`);
    }
  },

  async uamp_refused_credential() {
    // The built-in `uamp` and `uamp/stream` routes of all three servers
    // (S-345, 2026-09-29): they ran `agent.processUAMP(body)` on a context
    // that carried no credential, so an identity skill could refuse nothing
    // there, and a refusal that did surface was a 500 `uamp_error`.
    const session = JSON.stringify([{ type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } }]);
    for (const sub of ['/uamp', '/uamp/stream']) {
      for (const [name, build] of SERVERS) {
        const fetch = await build(agent());
        const res = await fetch(request(sub, { method: 'POST', body: session }, { authorization: 'Bearer bad-token' }));
        await expectChallenge(res, BAD, `${name} ${sub}`);
        expect(((await res.json()) as { error: { code: string } }).error.code).toBe('unauthorized');
      }
    }
  },

  async scoped_endpoint_anonymous() {
    await everyServer(async (fetch, name) => {
      const res = await fetch(request('/mine'));
      await expectChallenge(res, NONE, name);
      expect(((await res.json()) as { error: { message: string } }).error.message).toBe(NEEDS_CALLER);
    });
  },

  async scoped_endpoint_made_up_credential() {
    await everyServer(async (fetch, name) => {
      const res = await fetch(request('/mine', {}, { authorization: 'Bearer made-up' }));
      await expectChallenge(res, BAD, name);
      expect(((await res.json()) as { error: { message: string } }).error.message).toBe(NEEDS_CALLER);
      // An api key nobody verifies is a presented credential too.
      await expectChallenge(await fetch(request('/mine', {}, { 'x-api-key': 'made-up' })), BAD, name);
    });
  },

  async scoped_endpoint_refused_credential() {
    await everyServer(async (fetch, name) => {
      const res = await fetch(request('/mine', {}, { authorization: 'Bearer bad-token' }));
      await expectChallenge(res, BAD, name);
      expect(((await res.json()) as { error: { message: string } }).error.message).toBe('this bearer is refused');
      // And the owner still gets in: the challenge changed nothing else.
      expect(await (await fetch(request('/mine', {}, { authorization: 'Bearer owner-token' }))).json()).toEqual({ mine: true });
    });
  },

  async scoped_websocket_anonymous() {
    for (const [name, build] of SOCKET_SERVERS) {
      expect(await refusedUpgrade(await build(agent()), '/live'), name).toEqual({ status: 401, challenge: NONE });
      // A `?token` a browser sends is a presented credential: the same rule.
      expect(await refusedUpgrade(await build(agent()), '/live?token=made-up'), name).toEqual({ status: 401, challenge: BAD });
    }
  },

  async billable_websocket_anonymous() {
    for (const [name, build] of SOCKET_SERVERS) {
      const skill = new UAMPTransportSkill();
      const a = new BaseAgent({ name: 'guarded', instructions: 'x', skills: [skill] });
      await skill.initialize?.(a as never);
      const reply = await rawUpgradeReply(await build(a), `${BASE}/uamp`);
      expect(reply, name).toContain('HTTP/1.1 401 Unauthorized\r\n');
      expect(reply, name).toContain(`WWW-Authenticate: ${NONE}\r\n`);
    }
  },

  async daemon_registry_anonymous() {
    const remote = JSON.stringify({ name: 'far', url: 'https://far.example', capabilities: {} });
    await expectChallenge(await daemonCall(daemon(), '/agents/register', { method: 'POST', body: remote }), NONE, 'register');
    await expectChallenge(await daemonCall(daemon(), '/agents/guarded', { method: 'DELETE' }), NONE, 'remove');
  },

  async mcp_anonymous() {
    // Pinned by tests/unit/server/mcp-server.test.ts against mcp_tool/serve.json; the values agree.
    expect(SERVE.refusals.no_credential).toEqual({ status: 401, www_authenticate: NONE });
  },

  async mcp_refused_credential() {
    expect(SERVE.refusals.bad_credential).toEqual({ status: 401, www_authenticate: BAD });
  },
};

describe('every door this SDK serves', () => {
  it('is probed, and every probe is a door the fixture names', () => {
    const served = CHALLENGE.doors.filter((door) => door.typescript !== null).map((door) => door.name).sort();
    expect(Object.keys(PROBES).sort()).toEqual(served);
    for (const door of CHALLENGE.doors) {
      if (door.typescript === null) expect(door.typescript_why, door.name).toBeTruthy();
      expect(['no_credential', 'bad_credential'], door.name).toContain(door.challenge);
    }
  });

  for (const door of CHALLENGE.doors) {
    if (door.typescript === null) {
      it.skip(`${door.name}: ${door.typescript_why}`, () => {});
      continue;
    }
    it(`${door.name} answers its ${door.challenge} challenge`, async () => {
      // Each probe asserts the fixture's variant for its door in its own
      // body; the door's `challenge` says which, and the two must not disagree.
      const expected = door.challenge === 'no_credential' ? 'NONE' : 'BAD';
      expect(String(PROBES[door.name]), door.name).toContain(expected);
      await PROBES[door.name]!();
    });
  }
});
