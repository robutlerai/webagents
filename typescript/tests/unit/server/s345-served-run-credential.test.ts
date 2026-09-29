/**
 * S-345 (2026-09-29): every served run gets the caller's credential.
 *
 * `AuthSkill` reads the bearer from `context.metadata.authorization`
 * (`skills/auth/skill.ts`, `_extractToken`) and the access skill reads the
 * signed request from `context.get('_inboundRequest')`. `createFetchHandler`'s
 * `chat/completions` seeded both (`buildRequestMetadata`, `inboundRequest`),
 * and nothing else did: `WebAgentsServer`'s built-in `chat/completions`,
 * `uamp` and `uamp/stream` called `agent.run(msgs)` and `agent.processUAMP(body)`
 * bare, the built-in `uamp` routes of `createFetchHandler` and `serve()` did
 * the same, and the daemon passed the request in session data and no
 * metadata. On those doors an auth skill saw no credential, so it could
 * neither verify nor refuse one: behind the floor's presence check, any
 * non-empty credential header ran the model on the owner's key.
 *
 * Pinned here, door by door, with a witness that reads EXACTLY where the auth
 * skill reads (`metadata.authorization`) and refuses what the auth skill
 * would refuse (a token it does not know, and no token at all):
 *
 *   - the run's `on_connection` hook sees the credential headers on
 *     `metadata` and the request in `_inboundRequest` (method, target, the
 *     headers, the body bytes as they arrived);
 *   - a credential the hook refuses is a 401 with the `invalid_token`
 *     challenge and the model is not reached, streaming or not;
 *   - the request body's `metadata` reaches the hook (the platform's sender
 *     attribution) and cannot overwrite the header a credential arrived in.
 *
 * `serve()`'s `chat/completions` is probed too: the reference every other
 * door is held to.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, hook } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { WebAgentsServer } from '../../../src/server/multi';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { bearerChallenge } from '../../../src/server/credential-floor';

const BASE = '/agents/guarded';
const OWNER = 'Bearer owner-token';
const BAD = 'Bearer bad-token';
const INVALID_TOKEN = bearerChallenge(true);

/** What the auth skill's request looks like to the hook. */
interface Inbound {
  method: string;
  target: string;
  headers: Record<string, string>;
  body: Uint8Array;
}

interface Seen {
  metadata: Record<string, unknown>;
  inbound: Inbound | undefined;
}

/** An auth skill's refusal, matched by name as the servers match it (`isAuthError`). */
function refused(message: string): Error {
  return Object.assign(new Error(message), { name: 'AuthenticationError', statusCode: 401, code: 'unauthorized' });
}

/**
 * Stands in for `AuthSkill`, reading where it reads and refusing what it
 * refuses: `owner-token` on `metadata.authorization` is the owner; any other
 * token, and no token at all, is refused with the auth skill's own error.
 * It records what it saw, so a test can say what reached the hook.
 */
class Witness extends Skill {
  static readonly identifiesCaller = true;
  readonly seen: Seen[] = [];

  constructor() {
    super({ name: 'witness' });
  }

  @hook({ lifecycle: 'on_connection', priority: 0 })
  async who(_data: HookData, context: Context): Promise<void> {
    this.seen.push({ metadata: { ...(context.metadata ?? {}) }, inbound: context.get<Inbound>('_inboundRequest') });
    const bearer = context.metadata?.authorization;
    if (bearer === OWNER) {
      (context as unknown as { setAuth(auth: unknown): void }).setAuth({ authenticated: true, scope: 'owner', user_id: 'owner-1' });
      return;
    }
    throw refused(bearer === BAD ? 'this bearer is refused' : 'no credential seen');
  }
}

/** A model that answers `ok` and counts how often it was reached. */
class Model extends Skill {
  reached = 0;

  constructor() {
    super({ name: 'model' });
  }

  @handoff({ name: 'model' })
  async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent> {
    this.reached += 1;
    const responseId = generateEventId();
    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
    yield createResponseDeltaEvent(responseId, { type: 'text', text: 'ok' });
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'ok' }]);
  }
}

function fixture(): { agent: BaseAgent; witness: Witness; model: Model } {
  const witness = new Witness();
  const model = new Model();
  const agent = new BaseAgent({ name: 'guarded', instructions: 'Guarded.', skills: [witness, model] });
  return { agent, witness, model };
}

type Fetch = (request: Request) => Promise<Response>;

/** Every server, answering `BASE + sub`. */
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

/** The daemon, answering `/agents/guarded/chat/completions` only (it has no uamp route). */
async function daemonFetch(a: BaseAgent): Promise<Fetch> {
  const d = new WebAgentsDaemon({ port: 0, watch: false, cron: false, healthChecks: false });
  d.registry.registerLocal(a as never);
  return (request) => Promise.resolve(d.app.fetch(request));
}

/** Every door: the three servers' three billable routes, and the daemon's one. */
const DOORS: Array<[string, string, (a: BaseAgent) => Promise<Fetch>]> = [
  ...SERVERS.flatMap(([name, build]): Array<[string, string, (a: BaseAgent) => Promise<Fetch>]> => [
    [name, '/chat/completions', build],
    [name, '/uamp', build],
    [name, '/uamp/stream', build],
  ]),
  ['daemon', '/chat/completions', daemonFetch],
];

const EXTRA_HEADERS = { 'x-api-key': 'key-1', 'x-owner-assertion': 'assert-1', 'user-agent': 's345-probe' };

/** The request a door takes: a completion, or a UAMP session that asks for a response. */
function bodyFor(sub: string, stream: boolean, metadata?: Record<string, unknown>): string {
  if (sub === '/chat/completions') {
    return JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream, ...(metadata ? { metadata } : {}) });
  }
  return JSON.stringify([
    { type: 'session.create', event_id: 'e1', uamp_version: '1.0', session: { modalities: ['text'] } },
    { type: 'input.text', event_id: 'e2', text: 'hi', role: 'user' },
    { type: 'response.create', event_id: 'e3' },
  ]);
}

function post(sub: string, body: string, headers: Record<string, string>): Request {
  return new Request(`http://agent.test${BASE}${sub}?probe=1`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body,
  });
}

/** The streams a door answers: `uamp/stream` always, `chat/completions` when asked. */
function streams(sub: string): boolean[] {
  return sub === '/chat/completions' ? [false, true] : [sub === '/uamp/stream'];
}

async function drained(res: Response): Promise<string> {
  return res.text();
}

describe('S-345: every served run carries the request', () => {
  for (const [server, sub, build] of DOORS) {
    for (const stream of streams(sub)) {
      const where = `${server} ${sub}${sub === '/chat/completions' ? ` stream=${stream}` : ''}`;

      it(`${where}: the hook sees the credential headers and the inbound request, and the owner runs`, async () => {
        const { agent, witness, model } = fixture();
        const fetch = await build(agent);
        const body = bodyFor(sub, stream);
        const res = await fetch(post(sub, body, { authorization: OWNER, ...EXTRA_HEADERS }));
        const text = await drained(res);
        expect(res.status, `${where}: ${text}`).toBe(200);
        expect(model.reached, where).toBe(1);
        expect(witness.seen, where).toHaveLength(1);
        const { metadata, inbound } = witness.seen[0]!;
        // The credential headers, where `AuthSkill._extractToken` reads them.
        expect(metadata.authorization, where).toBe(OWNER);
        expect(metadata['x-api-key'], where).toBe('key-1');
        expect(metadata['x-owner-assertion'], where).toBe('assert-1');
        expect(metadata.userAgent, where).toBe('s345-probe');
        expect(metadata.method, where).toBe('POST');
        // The request, where the access skill verifies a signature: the
        // path and query, every header, and the bytes as they arrived.
        expect(inbound, where).toBeDefined();
        expect(inbound!.method, where).toBe('POST');
        expect(inbound!.target, where).toBe(`${BASE}${sub}?probe=1`);
        expect(inbound!.headers.authorization, where).toBe(OWNER);
        expect(inbound!.headers['x-api-key'], where).toBe('key-1');
        expect(new TextDecoder().decode(inbound!.body), where).toBe(body);
        // And the answer is the model's.
        if (stream) expect(text, where).toContain('data: ');
        else expect(text, where).toContain('ok');
      });

      it(`${where}: a refused credential is 401 invalid_token and the model is not reached`, async () => {
        const { agent, model } = fixture();
        const fetch = await build(agent);
        const res = await fetch(post(sub, bodyFor(sub, stream), { authorization: BAD }));
        const text = await drained(res);
        expect(res.status, `${where}: ${text}`).toBe(401);
        expect(res.headers.get('www-authenticate'), where).toBe(INVALID_TOKEN);
        const parsed = JSON.parse(text) as { error: { code: string; message: string } };
        expect(parsed.error.code, where).toBe('unauthorized');
        expect(parsed.error.message, where).toBe('this bearer is refused');
        expect(model.reached, where).toBe(0);
      });
    }
  }
});

describe("S-345: the body's metadata reaches the hook and cannot carry a credential", () => {
  const COMPLETIONS = DOORS.filter(([, sub]) => sub === '/chat/completions');

  for (const [server, sub, build] of COMPLETIONS) {
    for (const stream of [false, true]) {
      const where = `${server} ${sub} stream=${stream}`;

      it(`${where}: the platform's sender attribution reaches the hook`, async () => {
        const { agent, witness } = fixture();
        const fetch = await build(agent);
        const body = bodyFor(sub, stream, { chat_id: 'chat-1', sender: { id: 'user-9', username: 'nine' } });
        const res = await fetch(post(sub, body, { authorization: OWNER }));
        expect(res.status, `${where}: ${await drained(res)}`).toBe(200);
        const { metadata } = witness.seen[0]!;
        expect(metadata.sender, where).toEqual({ id: 'user-9', username: 'nine' });
        expect(metadata.chat_id, where).toBe('chat-1');
        expect(metadata.chatId, where).toBe('chat-1');
      });

      it(`${where}: a credential written into the body does not replace the header`, async () => {
        const { agent, witness, model } = fixture();
        const fetch = await build(agent);
        const body = bodyFor(sub, stream, { authorization: OWNER, 'x-api-key': 'forged' });
        const res = await fetch(post(sub, body, { authorization: BAD }));
        expect(res.status, `${where}: ${await drained(res)}`).toBe(401);
        expect(res.headers.get('www-authenticate'), where).toBe(INVALID_TOKEN);
        expect(witness.seen[0]!.metadata.authorization, where).toBe(BAD);
        expect(witness.seen[0]!.metadata['x-api-key'], where).toBeUndefined();
        expect(model.reached, where).toBe(0);
      });
    }
  }
});
