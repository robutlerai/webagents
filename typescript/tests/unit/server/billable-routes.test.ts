/**
 * THE ENUMERATING FLOOR TEST — TypeScript.
 *
 * Every previous round of this bug was missed the same way: the tests checked
 * the doors somebody remembered. `POST /chat/completions` was pinned, so the
 * next door found was `/uamp`; that got pinned, so the next was
 * `WebAgentsServer`, which had no floor on anything at all. Then this file was
 * written to DISCOVER routes instead of listing them — and the round after that
 * found `POST /a2a` answering an anonymous 200 with `processUAMP: 1`.
 *
 * WHAT WENT WRONG WITH THE PREVIOUS VERSION, because it is the whole design of
 * this one:
 *
 *   1. It discovered ROUTES, not HANDLERS. `discoverHonoRoutes` walked
 *      `app.routes` and `discoverFetchHandlerRoutes` scanned source for path
 *      literals. Neither could see `agent.wsRegistry`, so the WebSocket surface
 *      was never classified at all — the WS tests hardcoded `/uamp`. In Python
 *      the identical blind spot shipped two anonymous model-reaching sockets.
 *   2. Its fixture named two skills. `A2ATransportSkill` ships in the SDK,
 *      mounts `@http({ path: '/a2a', method: 'POST' })`, and reaches
 *      `agent.processUAMP` — and was invisible, because it was not in the
 *      fixture's `skills: [...]`. A hard-coded route list had been traded for a
 *      hard-coded skill list. (The reviewer proved this: adding the skill to
 *      the old fixture made the old tripwire fire immediately on the existing,
 *      shipped route.)
 *
 * SO THIS VERSION DISCOVERS FROM THE REGISTRIES, NOT FROM THE ROUTES.
 *
 *   - The fixture agent carries every transport skill the SDK EXPORTS,
 *     enumerated out of `src/skills/transport/index.ts` rather than named here.
 *     Adding a skill to the SDK is itself the thing that trips the wire.
 *   - `agent.listHttpEndpoints()` and `agent.listWebSocketEndpoints()` are the
 *     real registries every server class dispatches from — `getHttpHandler` at
 *     `multi.ts:325` and in `handler.ts`, `getWebSocketHandler` in the two
 *     upgrade handlers. Walking them finds a handler because it EXISTS, not
 *     because a path set already named it.
 *   - `app.routes` and the `createFetchHandler` source scan are still walked,
 *     for the framework surface the registries know nothing about — but they
 *     are now the SUPPLEMENT, not the source of truth.
 *
 * Three assertions run against what is discovered:
 *
 *   1. CLASSIFICATION, and it FAILS CLOSED. Every discovered handler — HTTP or
 *      WebSocket — is in `BILLABLE_PATHS` / `BILLABLE_WS_PATHS`, or in the
 *      shipped allow-list `PUBLIC_SUBPATHS` / `PUBLIC_WS_SUBPATHS`, or (for
 *      framework chrome) in `FRAMEWORK_PUBLIC_SUBPATHS` below. A handler in
 *      none of them fails the test. That is the property this file exists for;
 *      everything else is a consequence of it.
 *   2. REFUSAL. Every billable route answers exactly 401 to an anonymous
 *      caller, and every billable socket upgrade is refused. Exactly 401, not
 *      "not 200": a 500 from some unrelated bug upstream of the provider is not
 *      a security property, and treating it as one is how `uamp/completions`
 *      looked closed on the Python side while it was open.
 *   3. THE MODEL IS NOT REACHED. Counted on `run`, `runStreaming` and
 *      `processUAMP`, and asserted zero after the anonymous sweep — including
 *      the socket sweep.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { websocket } from '../../../src/core/decorators.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import { createFetchHandler } from '../../../src/server/handler.js';
import { createAgentApp } from '../../../src/server/node.js';
import { WebAgentsServer } from '../../../src/server/multi.js';
import {
  BILLABLE_PATHS,
  BILLABLE_WS_PATHS,
  PUBLIC_SUBPATHS,
  PUBLIC_WS_SUBPATHS,
  isBillablePath,
  isBillableWebSocketPath,
} from '../../../src/server/credential-floor.js';
import * as transportSkills from '../../../src/skills/transport/index.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(HERE, '../../../src');

const AGENT_NAME = 'og';
const BASE_PATH = `/agents/${AGENT_NAME}`;

/**
 * Framework chrome: not agent surface, not part of the cross-SDK contract, and
 * different in every server class — so it is allow-listed here rather than in
 * the shipped `PUBLIC_SUBPATHS`.
 *
 * `agents` is `WebAgentsServer`'s own listing endpoint. Nothing here dispatches
 * to a skill handler.
 */
const FRAMEWORK_PUBLIC_SUBPATHS = new Set(['agents']);

/**
 * Skills that must not be `initialize()`d in a test, declared with the reason.
 *
 * `PortalConnectSkill` is a CLIENT: its `initialize()` opens a live socket to
 * the portal and throws without `WEBAGENTS_AGENT_TOKEN`. It is still
 * constructed and still attached to the fixture agent, so anything it registers
 * is still discovered and still classified — only the side effect is skipped.
 * The list is asserted to be a subset of what the module exports, so renaming
 * the skill fails here instead of silently un-excluding something.
 */
const NOT_INITIALIZED_IN_TESTS = new Set(['PortalConnectSkill']);

/**
 * Every transport skill class the SDK EXPORTS, enumerated from the module.
 *
 * Not a list of class names. The previous fixture named two, and the skills it
 * did not name shipped `POST /a2a` — an anonymous 200 that ran the model — with
 * no test looking at it.
 */
function transportSkillClasses(): Array<[string, new () => Skill]> {
  return Object.entries(transportSkills).filter(
    ([name, value]) => typeof value === 'function' && /Skill$/.test(name),
  ) as Array<[string, new () => Skill]>;
}

/** How many times the model was actually reached. */
interface Counters {
  run: number;
  processUAMP: number;
}

function buildAgent(): { agent: BaseAgent; counters: Counters } {
  const counters: Counters = { run: 0, processUAMP: 0 };
  const skills = transportSkillClasses().map(([, Ctor]) => new Ctor());
  const agent = new BaseAgent({
    name: AGENT_NAME,
    instructions: 'test agent',
    // Every shipped transport, because their `@http` / `@websocket`
    // declarations ARE the surface being enumerated:
    // `@http({ path: '/v1/chat/completions' })` mounts its own Hono route that
    // SHADOWS the fetch-handler fallback, `@http({ path: '/a2a' })` reaches
    // `processUAMP`, and `@websocket({ path: '/uamp' })` is the same model call
    // over a socket.
    skills,
  });

  // Count reaching the model rather than inferring it from a status code.
  (agent as unknown as { run: unknown }).run = async () => {
    counters.run += 1;
    return { content: 'ok', usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
  };
  (agent as unknown as { runStreaming: unknown }).runStreaming = async function* () {
    counters.run += 1;
    yield { type: 'delta', delta: 'ok' };
  };
  (agent as unknown as { processUAMP: unknown }).processUAMP = async function* (
    _events: ClientEvent[],
  ): AsyncGenerator<ServerEvent> {
    counters.processUAMP += 1;
    return;
  };

  return { agent, counters };
}

/**
 * Initialize the skills that can be initialized here.
 *
 * `agent.initialize()` would run every skill's `initialize()`, and one of them
 * dials the portal. Skipping exactly that one — by declared name — keeps the
 * rest of the fixture real.
 */
async function initializeAgent(agent: BaseAgent): Promise<void> {
  for (const skill of (agent as unknown as { skills: Skill[] }).skills) {
    if (NOT_INITIALIZED_IN_TESTS.has(skill.constructor.name)) continue;
    await skill.initialize?.();
  }
}

/** Strip leading/trailing slashes, the way the floor normalizes. */
function normalize(p: string): string {
  return p.replace(/^\/+/, '').replace(/\/+$/, '');
}

/** Drop the agent mount prefix so a path can be classified by subpath. */
function subPathOf(p: string): string {
  const n = normalize(p);
  const prefix = normalize(BASE_PATH);
  if (n === prefix) return '';
  if (n.startsWith(`${prefix}/`)) return n.slice(prefix.length + 1);
  return n;
}

/**
 * `'billable' | 'public' | null` — and `null` is a failure.
 *
 * Method-agnostic on purpose: a subpath is billable if the floor would gate its
 * POST, whatever verb this particular handler uses.
 */
function classifyHttp(subPath: string): 'billable' | 'public' | null {
  if (isBillablePath(subPath)) return 'billable';
  if ((PUBLIC_SUBPATHS as readonly string[]).includes(subPath)) return 'public';
  if (FRAMEWORK_PUBLIC_SUBPATHS.has(subPath)) return 'public';
  return null;
}

/**
 * The socket half. Deliberately does NOT fall back to the HTTP allow-list: an
 * HTTP path being safe says nothing about a socket at the same name, and `live`
 * — a bare liveness probe over HTTP — is exactly the name the proof-of-
 * blindness handler used.
 */
function classifyWs(subPath: string): 'billable' | 'public' | null {
  if (isBillableWebSocketPath(subPath)) return 'billable';
  if ((PUBLIC_WS_SUBPATHS as readonly string[]).includes(subPath)) return 'public';
  return null;
}

const UNCLASSIFIED_HELP =
  'handler(s) discovered that are in neither the billable set nor a public ' +
  'allow-list. If they can reach the model, add them to BILLABLE_PATHS / ' +
  'BILLABLE_WS_PATHS in src/server/credential-floor.ts (and its Python twin, ' +
  'or the parity test fails). If they are safe to serve anonymously, add them ' +
  'to PUBLIC_SUBPATHS / PUBLIC_WS_SUBPATHS there — deliberately, not to make ' +
  'the test pass.';

interface AgentSurface {
  http: Array<{ method: string; subPath: string }>;
  ws: string[];
}

/**
 * Walk the agent's REAL handler registries.
 *
 * This is the source of truth every server class dispatches from, so a handler
 * is found here because it was registered — no path set is consulted, which is
 * what makes this discovery rather than confirmation. It also finds a handler
 * whether or not the server class in question mounts a listable route for it,
 * which is exactly the `WebAgentsServer` case: every agent route there is
 * served by one `/agents/:name/*` catch-all, so `POST /a2a` appears in no route
 * table anywhere.
 */
function discoverAgentSurface(agent: BaseAgent): AgentSurface {
  return {
    http: agent
      .listHttpEndpoints()
      .map((e) => ({ method: e.method.toUpperCase(), subPath: normalize(e.path) })),
    ws: agent.listWebSocketEndpoints().map((e) => normalize(e.path)),
  };
}

interface Discovered {
  /** Concrete routes, classifiable. */
  concrete: Array<{ method: string; path: string }>;
  /** Every path that must answer 401 anonymously. */
  billable: string[];
}

/**
 * Walk a Hono app's real route table.
 *
 * Wildcard and parameterised routes cannot be listed, so the billable path set
 * is expanded THROUGH them, at the agent mount prefixes the app actually serves
 * — that is the only way the multi-agent server's HTTP surface is reached at
 * all, since every one of its agent routes is served by a single
 * `/agents/:name/*` catch-all.
 */
function discoverHonoRoutes(
  app: { routes: Array<{ method: string; path: string }> },
  mountPrefixes: string[],
): Discovered {
  const concrete: Array<{ method: string; path: string }> = [];
  const billable = new Set<string>();

  for (const route of app.routes) {
    const p = route.path;
    const isWildcard = p.includes('*') || p.includes(':');
    if (!isWildcard) {
      concrete.push({ method: route.method, path: p });
      if (isBillablePath(p)) billable.add(p);
      continue;
    }
    for (const prefix of mountPrefixes) {
      const agentName = normalize(prefix).split('/').pop() ?? '';
      // A bare `/*` (a middleware, or `app.all('*')`) is anchored at each mount
      // prefix; a parameterised catch-all has the prefix in it already.
      const template =
        normalize(p) === '*' ? `${prefix}/*` : p.replace(/:[^/]+/g, agentName);
      for (const b of BILLABLE_PATHS) {
        billable.add(template.replace(/\*$/, b));
      }
    }
  }

  return { concrete, billable: [...billable] };
}

/**
 * `createFetchHandler` has no route table, so discover its branches from the
 * source: every path literal it compares `path` against. A new
 * `if (path === \`${basePath}/whatever\`)` is picked up here without anyone
 * updating this test.
 */
function discoverFetchHandlerRoutes(agent: BaseAgent): Discovered {
  const source = fs.readFileSync(path.join(SRC, 'server', 'handler.ts'), 'utf-8');
  const subPaths = new Set<string>();

  for (const m of source.matchAll(/path === `\$\{basePath\}\/([^`]*)`/g)) {
    subPaths.add(m[1]);
  }
  for (const m of source.matchAll(/path === '\/([^']*)'/g)) {
    subPaths.add(m[1]);
  }
  // The `httpRegistry` dispatch at the bottom serves whatever the agent's
  // skills declared, so those are routes of this handler too.
  for (const endpoint of discoverAgentSurface(agent).http) {
    subPaths.add(endpoint.subPath);
  }

  expect(subPaths.size).toBeGreaterThan(3); // the scan actually found something

  const concrete = [...subPaths].map((s) => ({ method: 'POST', path: `${BASE_PATH}/${s}` }));
  const billable = [...subPaths].filter((s) => isBillablePath(s)).map((s) => `${BASE_PATH}/${s}`);
  // Plus every billable path the shared set knows about, whether or not this
  // handler has a branch for it: the floor is upstream of dispatch, so an
  // unrouted billable path must still be refused rather than 404'd.
  for (const b of BILLABLE_PATHS) billable.push(`${BASE_PATH}/${b}`);

  return { concrete, billable: [...new Set(billable)] };
}

/**
 * Every URL that must answer 401 to an anonymous POST on a given server.
 *
 * Registry first — each billable handler at the server's own mount prefixes —
 * unioned with whatever the route walk found, so no single discovery step going
 * quiet can shrink the sweep.
 */
function billableHttpTargets(
  agent: BaseAgent,
  discovered: Discovered,
  mountPrefixes: string[],
): string[] {
  const targets = new Set(discovered.billable);
  for (const { subPath } of discoverAgentSurface(agent).http) {
    if (classifyHttp(subPath) !== 'billable') continue;
    for (const prefix of mountPrefixes) targets.add(`${prefix}/${subPath}`);
  }
  return [...targets];
}

/**
 * Every socket URL that must be refused at the upgrade, discovered from the
 * WebSocket registry.
 *
 * Registry-only, and that is a real constraint rather than an oversight: both
 * upgrade handlers look the path up in `wsRegistry` and answer 404 BEFORE the
 * floor runs, so a path in `BILLABLE_WS_PATHS` with no handler behind it cannot
 * be probed here at all. `realtime` and `acp/stream` are in that position in
 * this SDK — they are Python sockets, listed for parity. The Python suite
 * probes them for real.
 */
function billableWsTargets(agent: BaseAgent, mountPrefixes: string[]): string[] {
  const targets = new Set<string>();
  for (const subPath of discoverAgentSurface(agent).ws) {
    if (classifyWs(subPath) !== 'billable') continue;
    for (const prefix of mountPrefixes) targets.add(`${prefix}/${subPath}`);
  }
  return [...targets];
}

/**
 * Probe every discovered billable path and return the ones that did NOT answer
 * exactly 401. Collected rather than asserted one at a time so a failure names
 * EVERY open door at once — the point of an enumerating test is to hand you the
 * whole list, not the first entry.
 */
async function openDoors(
  fetchLike: (request: Request) => Promise<Response> | Response,
  billable: string[],
): Promise<string[]> {
  const open: string[] = [];
  for (const url of billable) {
    const res = await fetchLike(anonymousPost(url));
    if (res.status !== 401) open.push(`POST ${url} -> ${res.status}`);
  }
  return open;
}

function anonymousPost(url: string): Request {
  return new Request(`https://agent.example.com${url}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
  });
}

function credentialedPost(url: string): Request {
  return new Request(`https://agent.example.com${url}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer a-real-looking-token' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
  });
}

/** Every mounted route is classified: billable, or explicitly public. */
function expectEveryRouteClassified(discovered: Discovered, serverName: string): void {
  const unclassified = discovered.concrete
    .map((r) => subPathOf(r.path))
    .filter((s) => classifyHttp(s) === null);

  expect(unclassified, `${serverName}: ${UNCLASSIFIED_HELP}`).toEqual([]);
}

/** A fake `http.IncomingMessage` / socket pair for the upgrade handlers. */
function fakeUpgrade(url: string, headers: Record<string, string> = {}) {
  const written: string[] = [];
  const req = { url, headers: { host: 'agent.example.com', ...headers } };
  const socket = { write: (chunk: string) => written.push(chunk), destroy: () => {} };
  return { req, socket, written };
}

describe('every billable route on every server class refuses an anonymous caller', () => {
  let agent: BaseAgent;
  let counters: Counters;

  beforeEach(async () => {
    ({ agent, counters } = buildAgent());
    await initializeAgent(agent);
  });

  describe('the fixture really carries the whole shipped transport surface', () => {
    it('loads every transport skill the SDK exports', () => {
      const exported = transportSkillClasses().map(([name]) => name);
      expect(exported.length).toBeGreaterThan(0);
      const loaded = new Set(
        (agent as unknown as { skills: Skill[] }).skills.map((s) => s.constructor.name),
      );
      expect(exported.filter((name) => !loaded.has(name))).toEqual([]);
    });

    it('only skips initialization for skills it names, and they still exist', () => {
      const exported = new Set(transportSkillClasses().map(([name]) => name));
      expect([...NOT_INITIALIZED_IN_TESTS].filter((n) => !exported.has(n))).toEqual([]);
    });

    it('the registries really hold the shipped handlers', () => {
      const surface = discoverAgentSurface(agent);
      const httpSubPaths = new Set(surface.http.map((h) => h.subPath));
      // One from each shipped server-side transport, so a skill quietly falling
      // out of the fixture fails here rather than silently losing coverage.
      expect(httpSubPaths).toContain('v1/chat/completions');
      expect(httpSubPaths).toContain('a2a');
      expect(httpSubPaths).toContain('.well-known/agent.json');
      expect(surface.ws).toContain('uamp');
    });
  });

  describe('the tripwire: every discovered handler is classified', () => {
    it('classifies every registered HTTP handler', () => {
      const unclassified = discoverAgentSurface(agent)
        .http.filter((h) => classifyHttp(h.subPath) === null)
        .map((h) => `${h.method} ${h.subPath}`);
      expect(unclassified, UNCLASSIFIED_HELP).toEqual([]);
    });

    it('classifies every registered WebSocket handler', () => {
      // The assertion that did not exist. `agent.wsRegistry` was never walked
      // by anything, so a model-reaching `@websocket` handler could be added
      // with the whole suite green — which is exactly what happened in the
      // Python SDK, twice.
      const surface = discoverAgentSurface(agent);
      expect(surface.ws.length, 'discovery found no WebSocket handlers').toBeGreaterThan(0);
      const unclassified = surface.ws.filter((p) => classifyWs(p) === null);
      expect(unclassified, UNCLASSIFIED_HELP).toEqual([]);
    });

    it('classifies every branch createFetchHandler serves', () => {
      expectEveryRouteClassified(discoverFetchHandlerRoutes(agent), 'createFetchHandler');
    });

    it('classifies every route createAgentApp mounts', () => {
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const discovered = discoverHonoRoutes(app, [BASE_PATH]);
      // The skill mount is real: this is the route that SHADOWS the fetch
      // handler's guarded branch, which is why the floor cannot live there.
      expect(discovered.concrete.map((r) => r.path)).toContain(
        `${BASE_PATH}/v1/chat/completions`,
      );
      expectEveryRouteClassified(discovered, 'createAgentApp');
    });

    it('classifies every route WebAgentsServer mounts', async () => {
      const server = new WebAgentsServer({ logging: false });
      await server.addAgent('m1', agent);
      expectEveryRouteClassified(
        discoverHonoRoutes(server.getApp(), ['/agents/m1']),
        'WebAgentsServer',
      );
    });
  });

  describe('createFetchHandler (the universal handler)', () => {
    it('answers exactly 401 on every discovered billable path, model untouched', async () => {
      const handler = createFetchHandler(agent, { basePath: BASE_PATH });
      const billable = billableHttpTargets(agent, discoverFetchHandlerRoutes(agent), [BASE_PATH]);
      expect(billable).toContain(`${BASE_PATH}/a2a`);

      expect(await openDoors(handler, billable)).toEqual([]);
      expect(counters).toEqual({ run: 0, processUAMP: 0 });
    });

    it('the 401 is the floor talking, not a missing route', async () => {
      // Guard the guard: with a credential, the same path gets THROUGH the
      // floor. Otherwise a typo'd path would pass the test above forever.
      const handler = createFetchHandler(agent, { basePath: BASE_PATH });
      const res = await handler(credentialedPost(`${BASE_PATH}/uamp`));
      expect(res.status).not.toBe(401);
      expect(counters.processUAMP).toBe(1);
    });
  });

  describe('createAgentApp / serve (the single-agent Hono server)', () => {
    it('answers exactly 401 on every discovered billable route, model untouched', async () => {
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const billable = billableHttpTargets(agent, discoverHonoRoutes(app, [BASE_PATH]), [
        BASE_PATH,
      ]);
      // The route the review proved was an anonymous 200 with `processUAMP: 1`.
      expect(billable).toContain(`${BASE_PATH}/a2a`);

      expect(await openDoors((r) => app.fetch(r), billable)).toEqual([]);
      expect(counters).toEqual({ run: 0, processUAMP: 0 });
    });

    it('the 401 is the floor talking, not a missing route', async () => {
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const res = await app.fetch(credentialedPost(`${BASE_PATH}/uamp`));
      expect(res.status).not.toBe(401);
      expect(counters.processUAMP).toBe(1);
    });

    it('a credentialed POST /a2a really reaches the model, so the 401 above meant something', async () => {
      // Without this, `/a2a` could be "closed" because the route stopped
      // existing rather than because the floor refused it.
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const res = await app.fetch(
        new Request(`https://agent.example.com${BASE_PATH}/a2a`, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: 'Bearer a-real-looking-token',
          },
          body: JSON.stringify({
            jsonrpc: '2.0',
            id: 1,
            method: 'tasks/send',
            params: { message: { parts: [{ type: 'text', text: 'hi' }] } },
          }),
        }),
      );
      expect(res.status).not.toBe(401);
      expect(counters.processUAMP).toBe(1);
    });

    it('does not gate the routes that are meant to be public', async () => {
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      for (const p of ['health', 'info', 'v1/models', '.well-known/agent.json']) {
        const res = await app.fetch(new Request(`https://agent.example.com${BASE_PATH}/${p}`));
        expect(res.status, `anonymous GET ${p}`).not.toBe(401);
      }
    });
  });

  describe('WebAgentsServer (the documented multi-agent server)', () => {
    it('answers exactly 401 on every discovered billable route, model untouched', async () => {
      const server = new WebAgentsServer({ logging: false });
      await server.addAgent('m1', agent);
      const billable = billableHttpTargets(
        agent,
        discoverHonoRoutes(server.getApp(), ['/agents/m1']),
        ['/agents/m1'],
      );
      // The catch-all is the ONLY agent route here, so nothing below the mount
      // prefix appears in the route table: `/a2a` is reachable only because the
      // REGISTRY was walked. The review could not close this case empirically;
      // this is the assertion that closes it.
      expect(billable).toContain('/agents/m1/chat/completions');
      expect(billable).toContain('/agents/m1/uamp');
      expect(billable).toContain('/agents/m1/a2a');

      expect(await openDoors((r) => server.getApp().fetch(r), billable)).toEqual([]);
      expect(counters).toEqual({ run: 0, processUAMP: 0 });
    });

    it('the 401 is the floor talking, not a missing route', async () => {
      const server = new WebAgentsServer({ logging: false });
      await server.addAgent('m1', agent);
      const res = await server.getApp().fetch(credentialedPost('/agents/m1/uamp'));
      expect(res.status).not.toBe(401);
      expect(counters.processUAMP).toBe(1);
    });

    it('does not gate the routes that are meant to be public', async () => {
      const server = new WebAgentsServer({ logging: false });
      await server.addAgent('m1', agent);
      for (const p of ['/health', '/agents', '/agents/m1/info']) {
        const res = await server.getApp().fetch(new Request(`https://agent.example.com${p}`));
        expect(res.status, `anonymous GET ${p}`).not.toBe(401);
      }
    });
  });

  describe('the WebSocket door, discovered from the registry', () => {
    /**
     * `UAMPTransportSkill` and `PortalTransportSkill` both register
     * `@websocket({ path: '/uamp' })`, which runs the model on the owner's
     * credit exactly like `POST /uamp` does — and the upgrade handlers checked
     * nothing at all. Same money, different protocol, so the same predicate and
     * the same path set from `credential-floor.ts`.
     *
     * What is new is where the list of sockets comes from: the tests below
     * enumerate `agent.listWebSocketEndpoints()` and refuse to run on an empty
     * list, so a new `@websocket` handler is probed on the day it is written
     * rather than on the day someone adds it here.
     */

    it('probes every registered billable socket, not a hardcoded one', () => {
      const targets = billableWsTargets(agent, [BASE_PATH]);
      expect(targets.length, 'no billable WebSocket handler was discovered').toBeGreaterThan(0);
      expect(targets).toContain(`${BASE_PATH}/uamp`);
    });

    it('refuses an anonymous upgrade to every discovered billable socket (single-agent server)', () => {
      const { handleUpgrade } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const open: string[] = [];
      for (const url of billableWsTargets(agent, [BASE_PATH])) {
        const { req, socket, written } = fakeUpgrade(url);
        handleUpgrade(
          req as unknown as import('http').IncomingMessage,
          socket as unknown as import('stream').Duplex,
          Buffer.alloc(0),
        );
        if (!written.join('').includes('401 Unauthorized')) open.push(`WS ${url}`);
      }
      expect(open, 'anonymous socket upgrades were not refused').toEqual([]);
      expect(counters).toEqual({ run: 0, processUAMP: 0 });
    });

    it('refuses an anonymous upgrade to every discovered billable socket (multi-agent server)', async () => {
      const server = new WebAgentsServer({ logging: false });
      await server.addAgent('m1', agent);
      // `handleWebSocketUpgrade` is private, and it is also the ONLY entry
      // point: `start()` wires it to the http server's 'upgrade' event. Testing
      // the door means knocking on the door.
      const knock = (
        server as unknown as {
          handleWebSocketUpgrade: (r: unknown, s: unknown, h: Buffer) => void;
        }
      ).handleWebSocketUpgrade.bind(server);

      const open: string[] = [];
      for (const url of billableWsTargets(agent, ['/agents/m1'])) {
        const { req, socket, written } = fakeUpgrade(url);
        knock(req, socket, Buffer.alloc(0));
        if (!written.join('').includes('401 Unauthorized')) open.push(`WS ${url}`);
      }
      expect(open, 'anonymous socket upgrades were not refused').toEqual([]);
      expect(counters).toEqual({ run: 0, processUAMP: 0 });
    });

    it('lets a credentialed upgrade past the floor', () => {
      const { handleUpgrade } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      // Header form, and the `?token=` form a browser has to use because it
      // cannot set headers on a WebSocket handshake.
      for (const [url, headers] of [
        [`${BASE_PATH}/uamp`, { authorization: 'Bearer a-real-looking-token' }],
        [`${BASE_PATH}/uamp?token=a-real-looking-token`, {}],
      ] as Array<[string, Record<string, string>]>) {
        const { req, socket, written } = fakeUpgrade(url, headers);
        try {
          handleUpgrade(
            req as unknown as import('http').IncomingMessage,
            socket as unknown as import('stream').Duplex,
            Buffer.alloc(0),
          );
        } catch {
          // Past the floor the real `ws` server takes over and chokes on this
          // stub socket. That throw IS the evidence: it can only happen from
          // code the floor would have short-circuited.
        }
        expect(written.join(''), url).not.toContain('401 Unauthorized');
      }
    });

    it('leaves a socket that is declared non-billable alone', async () => {
      // The example MUST NOT be a real transport subpath. This test used to use
      // `/realtime`, which is vacuous in TypeScript (no TS skill registers it)
      // and which asserted as CORRECT the non-refusal of a path that was, in
      // the Python SDK, a live anonymous model endpoint — so whoever fixed that
      // hole would be told by this test that they were wrong.
      //
      // The path below is registered by a skill that exists only in this test,
      // on an agent that exists only in this test, so it can never collide with
      // a shipped transport subpath or with the classification assertions above.
      class ProbeSocketSkill extends Skill {
        constructor() {
          super({ name: 'probe-socket' });
        }
        @websocket({ path: '/not-a-model-path' })
        async probe(): Promise<void> {
          /* never reached: the floor decision is made before dispatch */
        }
      }
      const bare = new BaseAgent({ name: 'bare', skills: [new ProbeSocketSkill()] });
      expect(bare.listWebSocketEndpoints().map((e) => e.path)).toContain('/not-a-model-path');

      const { handleUpgrade } = createAgentApp(bare, { basePath: BASE_PATH, logging: false });
      const { req, socket, written } = fakeUpgrade(`${BASE_PATH}/not-a-model-path`);
      try {
        handleUpgrade(
          req as unknown as import('http').IncomingMessage,
          socket as unknown as import('stream').Duplex,
          Buffer.alloc(0),
        );
      } catch {
        // Past the floor the real `ws` server chokes on the stub socket — which
        // is itself proof the floor let this through.
      }
      expect(written.join('')).not.toContain('401 Unauthorized');
      expect(written.join('')).not.toContain('404');
    });
  });

  describe('the floor is scoped to what actually costs money', () => {
    it('leaves a GET on a billable-looking path alone', async () => {
      // An agent named `uamp` has an info page at `/uamp`. Only POST is gated,
      // which is what keeps the suffix match from swallowing it — and what lets
      // `GET /tasks/{task_id}` be a public status read next to a billable
      // `POST /tasks` in the Python SDK.
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const res = await app.fetch(new Request(`https://agent.example.com${BASE_PATH}/uamp`));
      expect(res.status).not.toBe(401);
    });

    it('leaves a CORS preflight alone', async () => {
      const { app } = createAgentApp(agent, { basePath: BASE_PATH, logging: false });
      const res = await app.fetch(
        new Request(`https://agent.example.com${BASE_PATH}/chat/completions`, {
          method: 'OPTIONS',
          headers: { origin: 'https://example.com', 'access-control-request-method': 'POST' },
        }),
      );
      expect(res.status).not.toBe(401);
    });
  });
});
