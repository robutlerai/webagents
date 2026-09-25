/**
 * S-242 (2026-09-25): a scoped `@http` or `@websocket` endpoint is checked
 * against the caller the agent verified, in every server that serves one, and
 * an endpoint with no scopes stays open to anyone, exactly as before.
 *
 * Until then the decorators stored `scopes` and no server looked at them:
 * `createFetchHandler`, the app `serve()` builds and `WebAgentsServer` all ran
 * the handler for anyone who reached it. The refusals are the ones the Python
 * servers answer (`python/tests/fixtures/endpoint_gate/refusals.json`); the
 * Python cases are `python/tests/server/test_endpoint_scopes.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';
import { SignJWT, createLocalJWKSet, exportJWK, generateKeyPair } from 'jose';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { hook, http as endpoint, websocket } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import { accessSkillFor } from '../../../src/access/install';
import { AuthSkill } from '../../../src/skills/auth/skill';
import { JWKSManager } from '../../../src/crypto/jwks';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REFUSALS = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/endpoint_gate/refusals.json'), 'utf8'),
) as Record<'no_caller' | 'not_open', { status: number; body: unknown }>;
const { no_caller: NO_CALLER, not_open: NOT_OPEN } = REFUSALS;

const BASE = '/agents/guarded';

/** Stands in for an auth skill: three bearer tokens name three tiers. */
const TIERS: Record<string, string> = {
  'Bearer owner-token': 'owner',
  'Bearer user-token': 'user',
  'Bearer admin-token': 'admin',
};

class BearerIdentity extends Skill {
  static readonly identifiesCaller = true;

  constructor() {
    super({ name: 'identity' });
  }

  @hook({ lifecycle: 'on_connection', priority: 0 })
  async who(_data: HookData, context: Context): Promise<void> {
    const tier = TIERS[String(context.metadata?.authorization ?? '')];
    if (tier) {
      (context as unknown as { setAuth(auth: unknown): void }).setAuth({ authenticated: true, scope: tier, user_id: `${tier}-1` });
    }
  }
}

/** A connection hook that is not identification: an endpoint call must not run it. */
const charged: string[] = [];
class Payments extends Skill {
  constructor() {
    super({ name: 'payments' });
  }

  @hook({ lifecycle: 'on_connection', priority: 0 })
  async charge(): Promise<void> {
    charged.push('charged');
  }
}

const saved: unknown[] = [];
class Endpoints extends Skill {
  constructor() {
    super({ name: 'endpoints' });
  }

  @endpoint({ path: '/open', method: 'GET' })
  async open(): Promise<Response> {
    return Response.json({ open: true });
  }

  @endpoint({ path: '/mine', method: 'GET', scopes: ['owner'] })
  async mine(_req: Request, context: Context): Promise<Response> {
    return Response.json({ mine: true, who: context.auth?.user_id ?? null });
  }

  @endpoint({ path: '/admins', method: 'GET', scopes: ['admin'] })
  async admins(): Promise<Response> {
    return Response.json({ admins: true });
  }

  @endpoint({ path: '/save', method: 'POST', scopes: ['owner'] })
  async save(req: Request): Promise<Response> {
    const body = await req.json();
    saved.push(body);
    return Response.json({ saved: body });
  }

  @websocket({ path: '/live', scopes: ['owner'] })
  live(ws: WebSocket): void {
    ws.send(JSON.stringify({ live: true }));
    ws.close();
  }

  @websocket({ path: '/public' })
  open_socket(ws: WebSocket): void {
    ws.send(JSON.stringify({ public: true }));
    ws.close();
  }
}

function agent(options: { identity?: boolean; access?: unknown } = {}): BaseAgent {
  const skills: Skill[] = [new Endpoints(), new Payments()];
  if (options.identity !== false) skills.push(new BearerIdentity());
  if (options.access !== undefined) skills.push(accessSkillFor(options.access, undefined).skill as unknown as Skill);
  return new BaseAgent({ name: 'guarded', instructions: 'Guarded.', skills });
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

function get(fetch: Fetch, sub: string, bearer?: string): Promise<Response> {
  return fetch(
    new Request(`http://agent.test${BASE}${sub}`, { headers: bearer ? { authorization: `Bearer ${bearer}` } : {} }),
  );
}

beforeEach(() => {
  charged.length = 0;
  saved.length = 0;
});

describe.each(SERVERS)('%s: scoped @http endpoints', (_name, build) => {
  it('leaves an endpoint with no scopes open', async () => {
    const res = await get(await build(agent()), '/open');
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ open: true });
  });

  it('asks an anonymous caller to identify', async () => {
    const res = await get(await build(agent()), '/mine');
    expect(res.status).toBe(NO_CALLER.status);
    expect(await res.json()).toEqual(NO_CALLER.body);
  });

  it('forbids a verified caller the scopes do not include', async () => {
    const res = await get(await build(agent()), '/mine', 'user-token');
    expect(res.status).toBe(NOT_OPEN.status);
    expect(await res.json()).toEqual(NOT_OPEN.body);
  });

  it('lets the owner in, with the verified caller on the context', async () => {
    const fetch = await build(agent());
    expect(await (await get(fetch, '/mine', 'owner-token')).json()).toEqual({ mine: true, who: 'owner-1' });
    // An admin passes an owner scope; an owner does not pass an admin one.
    expect((await get(fetch, '/mine', 'admin-token')).status).toBe(200);
    expect((await get(fetch, '/admins', 'owner-token')).status).toBe(403);
  });

  it('never runs a refused handler, and a let-in handler still reads its body', async () => {
    const fetch = await build(agent());
    const post = (bearer?: string) =>
      fetch(
        new Request(`http://agent.test${BASE}/save`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', ...(bearer ? { authorization: `Bearer ${bearer}` } : {}) },
          body: JSON.stringify({ value: bearer ?? 'anonymous' }),
        }),
      );
    expect((await post()).status).toBe(401);
    expect((await post('user-token')).status).toBe(403);
    expect(saved).toEqual([]);
    expect(await (await post('owner-token')).json()).toEqual({ saved: { value: 'owner-token' } });
  });

  it('does not count a made-up bearer as a verified caller', async () => {
    // The servers' own contexts set `authenticated` for ANY bearer.
    const res = await get(await build(agent({ identity: false })), '/mine', 'anything');
    expect(res.status).toBe(401);
  });

  it('takes no payment to identify the caller', async () => {
    await get(await build(agent()), '/mine', 'owner-token');
    expect(charged).toEqual([]);
  });

  it("answers with the access block's own refusal", async () => {
    const fetch = await build(agent({ access: { default: 'none' } }));
    const res = await get(fetch, '/mine');
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({
      error: { code: 'forbidden', message: 'This agent does not accept requests from this caller.' },
    });
    // An endpoint with no scopes is still open: the block decides turns and scoped endpoints.
    expect((await get(fetch, '/open')).status).toBe(200);
  });
});

// -- websockets ---------------------------------------------------------------------------

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

describe.each(SOCKET_SERVERS)('%s: scoped @websocket endpoints', (_name, build) => {
  let server: http.Server | null = null;

  afterEach(async () => {
    await new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve()));
    server = null;
  });

  async function openSocket(
    a: BaseAgent,
    sub: string,
    headers: Record<string, string> = {},
  ): Promise<{ message?: unknown; status?: number; body?: unknown }> {
    const upgrade = await build(a);
    server = http.createServer();
    server.on('upgrade', upgrade);
    await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', () => resolve()));
    const { port } = server.address() as AddressInfo;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`ws://127.0.0.1:${port}${BASE}${sub}`, { headers });
      ws.on('message', (data) => {
        resolve({ message: JSON.parse(String(data)) });
        ws.close();
      });
      ws.on('unexpected-response', (_req, res) => {
        let text = '';
        res.on('data', (chunk) => (text += chunk));
        res.on('end', () => resolve({ status: res.statusCode, body: text ? JSON.parse(text) : undefined }));
      });
      ws.on('error', (error) => reject(error));
    });
  }

  it('opens a socket with no scopes for anyone', async () => {
    expect(await openSocket(agent(), '/public')).toEqual({ message: { public: true } });
  });

  it("refuses an anonymous upgrade with the gate's answer", async () => {
    expect(await openSocket(agent(), '/live')).toEqual({ status: NO_CALLER.status, body: NO_CALLER.body });
  });

  it('refuses a caller the scopes do not include', async () => {
    const outcome = await openSocket(agent(), '/live', { authorization: 'Bearer user-token' });
    expect(outcome).toEqual({ status: NOT_OPEN.status, body: NOT_OPEN.body });
  });

  it('lets the owner in, with the credential in ?token as a browser sends it', async () => {
    expect(await openSocket(agent(), '/live?token=owner-token')).toEqual({ message: { live: true } });
  });
});

// -- the real AuthSkill and a real platform service token ----------------------------------

describe('with the platform AuthSkill', () => {
  // The gate against the actual credential path: the AuthSkill verifying an
  // RS256 service token, which names its sender (S-240). Python:
  // test_endpoint_scopes.py::TestWithThePlatformAuthSkill.
  const PLATFORM = 'https://robutler.test';
  const AGENT_URL = 'https://agent.example.com/agents/guarded';
  let privateKey: CryptoKey;
  let fetchAgent: Fetch;

  beforeEach(async () => {
    const pair = await generateKeyPair('RS256');
    privateKey = pair.privateKey;
    const publicJwk = { ...(await exportJWK(pair.publicKey)), kid: 'sig', use: 'sig', alg: 'RS256' };
    const jwks = new JWKSManager({ platformApiUrl: PLATFORM, platformIssuer: PLATFORM, agentPublicUrl: AGENT_URL });
    (jwks as unknown as { jwksCache: Map<string, unknown> }).jwksCache.set(
      `${PLATFORM}/.well-known/jwks.json`,
      createLocalJWKSet({ keys: [publicJwk] as never }),
    );
    // No api-key service in a unit test: validation fails, so the service token path decides.
    vi.stubGlobal('fetch', async () => new Response('nope', { status: 401 }));
    const a = new BaseAgent({
      name: 'guarded',
      instructions: 'Guarded.',
      skills: [new Endpoints(), new AuthSkill({ jwksManager: jwks, requireAuth: true, ownerUserId: 'owner-1' })],
    });
    fetchAgent = createFetchHandler(a, { basePath: BASE });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const token = (sender: string) =>
    new SignJWT({ sender: { id: sender } })
      .setProtectedHeader({ alg: 'RS256', kid: 'sig' })
      .setIssuer(PLATFORM)
      .setSubject('service:robutler-router')
      .setAudience(AGENT_URL)
      .setIssuedAt()
      .setExpirationTime('5m')
      .sign(privateKey);

  it('lets in the owner the platform relays for', async () => {
    const res = await get(fetchAgent, '/mine', await token('owner-1'));
    expect(await res.json()).toEqual({ mine: true, who: 'owner-1' });
  });

  it('forbids anyone else it relays for', async () => {
    const res = await get(fetchAgent, '/mine', await token('stranger-2'));
    expect(res.status).toBe(NOT_OPEN.status);
    expect(await res.json()).toEqual(NOT_OPEN.body);
  });

  it("answers a credential the skill cannot verify with the skill's refusal", async () => {
    const res = await get(fetchAgent, '/mine', 'made-up');
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({
      error: { code: 'unauthorized', message: 'Authentication failed (API key, owner assertion, or service token required)' },
    });
  });
});

// -- `auth` modes (S-247) -------------------------------------------------------------------

class Modes extends Skill {
  constructor() {
    super({ name: 'modes' });
  }

  @endpoint({ path: '/admin-page', method: 'GET', auth: 'session' })
  async adminPage(): Promise<Response> {
    return Response.json({ admin: true });
  }

  @endpoint({ path: '/from-platform', method: 'GET', auth: 'portal_token' })
  async fromPlatform(_req: Request, context: Context): Promise<Response> {
    return Response.json({ who: context.auth?.user_id ?? null });
  }

  @endpoint({ path: '/visitor', method: 'GET', auth: 'visitor_session' })
  async visitor(_req: Request, context: Context): Promise<Response> {
    return Response.json({ who: context.auth?.user_id ?? null, authenticated: Boolean(context.auth?.authenticated) });
  }

  @endpoint({ path: '/hook', method: 'POST', auth: 'signature' })
  async hook(): Promise<Response> {
    return Response.json({ hook: true });
  }
}

function modesAgent(): BaseAgent {
  return new BaseAgent({ name: 'guarded', instructions: 'Guarded.', skills: [new Modes(), new BearerIdentity()] });
}

describe.each(SERVERS)('%s: the `auth` mode of an @http endpoint', (_name, build) => {
  it("keeps an owner-only page ('session') to the owner", async () => {
    const fetch = await build(modesAgent());
    expect((await get(fetch, '/admin-page')).status).toBe(NO_CALLER.status);
    expect(await (await get(fetch, '/admin-page', 'user-token')).json()).toEqual(NOT_OPEN.body);
    expect(await (await get(fetch, '/admin-page', 'owner-token')).json()).toEqual({ admin: true });
  });

  it("needs a verified caller for 'portal_token'", async () => {
    const fetch = await build(modesAgent());
    expect((await get(fetch, '/from-platform')).status).toBe(401);
    expect((await get(fetch, '/from-platform', 'made-up')).status).toBe(401);
    expect(await (await get(fetch, '/from-platform', 'user-token')).json()).toEqual({ who: 'user-1' });
  });

  it("lets an anonymous visitor through as anonymous ('visitor_session'), never as the bearer it sent", async () => {
    const fetch = await build(modesAgent());
    expect(await (await get(fetch, '/visitor')).json()).toEqual({ who: null, authenticated: false });
    expect(await (await get(fetch, '/visitor', 'made-up')).json()).toEqual({ who: null, authenticated: false });
    expect(await (await get(fetch, '/visitor', 'user-token')).json()).toEqual({ who: 'user-1', authenticated: true });
  });

  it("asks nothing of a 'signature' endpoint, which checks its provider itself", async () => {
    const fetch = await build(modesAgent());
    const res = await fetch(new Request(`http://agent.test${BASE}/hook`, { method: 'POST', body: '{}' }));
    expect(await res.json()).toEqual({ hook: true });
  });
});

describe("the Telegram skill's owner-only webhook setup (S-247's in-tree case)", () => {
  it('refuses a caller who is not the owner before anything reaches Telegram', async () => {
    const { TelegramSkill } = await import('../../../src/skills/messaging/telegram/skill');
    const calls: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      calls.push(String(url));
      return new Response('{}');
    });
    try {
      const a = new BaseAgent({ name: 'guarded', instructions: 'x', skills: [new TelegramSkill() as unknown as Skill, new BearerIdentity()] });
      const handler = createFetchHandler(a, { basePath: BASE });
      const res = await handler(new Request(`http://agent.test${BASE}/messaging/telegram/set-webhook`, { method: 'POST' }));
      expect(res.status).toBe(401);
      const stranger = await handler(
        new Request(`http://agent.test${BASE}/messaging/telegram/set-webhook`, { method: 'POST', headers: { authorization: 'Bearer user-token' } }),
      );
      expect(stranger.status).toBe(403);
      expect(calls).toEqual([]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
