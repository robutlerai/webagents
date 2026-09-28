/**
 * A2A v1.0 transport, replayed from the fixture both SDKs share
 * (`python/tests/fixtures/a2a/vectors.json`, `config_shapes.json`; plan item
 * 1.3, 2026-09-26). The Python twin is `python/tests/test_transport_a2a_v1.py`.
 *
 * The agent under test is the fixture's echo agent: a handoff that answers
 * `echo: ` plus the input texts, one tool anyone may use and one only the
 * owner may, and an identity skill that turns a bearer into a principal (so
 * tasks are owned by principal here; the credential-hash key is tested on
 * its own below). It is served through `createFetchHandler` under a prefix,
 * the way `serve()` and `WebAgentsServer` dispatch a skill's `@http` routes.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, hook, tool } from '../../../src/core/decorators';
import type { Context, HookData } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';
import { createAgentApp } from '../../../src/server/node';
import { AgentIdentity } from '../../../src/crypto/identity';
import { AuthenticationError } from '../../../src/skills/auth/skill';
import { A2ATransportSkill, peerTokenFor, resolveA2ASettings } from '../../../src/skills/transport/a2a/skill';
import {
  base64url,
  base64urlDecode,
  cardSigningBytes,
  cardSigningPayload,
  hmacCardSigner,
  identityCardSigner,
  signAgentCard,
  verifyAgentCard,
} from '../../../src/skills/transport/a2a/card';
import { canonicalize } from '../../../src/skills/transport/a2a/jcs';
import { readPart } from '../../../src/skills/transport/a2a/protocol';
import { callAgent, pickInterface, readSendResult, replyOfResult } from '../../../src/skills/transport/a2a/a2a-client';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import type { StreamResponse, Task } from '../../../src/skills/transport/a2a/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/a2a');
const vectors = JSON.parse(readFileSync(path.join(FIXTURES, 'vectors.json'), 'utf8'));
const configShapes = JSON.parse(readFileSync(path.join(FIXTURES, 'config_shapes.json'), 'utf8'));

const BASE_PATH = '/agents/a2a-echo';
const ORIGIN = 'https://agent.example.test';

// ---------------------------------------------------------------------------
// The fixture agent
// ---------------------------------------------------------------------------

class EchoLLM extends Skill {
  /** Set to make the run wait, so a task can be observed WORKING and canceled. */
  delayMs = 0;

  @handoff({ name: 'echo-llm' })
  async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text' && (e as { text: string }).text) texts.push((e as { text: string }).text);
    }
    if (this.delayMs) await new Promise((r) => setTimeout(r, this.delayMs));
    const reply = `${vectors.agent.reply_prefix}${texts.join(' ')}`;
    yield createResponseDeltaEvent('r1', { type: 'text', text: reply });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: reply }]);
  }
}

class ToolsSkill extends Skill {
  @tool({ name: 'public_lookup', description: 'Anyone may call this' })
  async publicLookup(_params: { q: string }, _ctx: Context) {
    return 'ok';
  }

  @tool({ name: 'owner_only_admin', description: 'Only the owner may call this', scopes: ['owner'] })
  async ownerOnly(_params: Record<string, never>, _ctx: Context) {
    return 'ok';
  }
}

/** Turns `Bearer <token>` into the principal `user:<token>`; refuses `Bearer refuse-me`. */
class BearerIdentity extends Skill {
  static identifiesCaller = true;

  @hook({ lifecycle: 'on_connection', priority: 1 })
  async identify(_data: HookData, context: Context): Promise<void> {
    const header = context.metadata?.authorization as string | undefined;
    const token = header?.startsWith('Bearer ') ? header.slice(7) : undefined;
    if (!token) return;
    if (token === 'refuse-me') throw new AuthenticationError('this bearer is refused');
    context.setAuth({ authenticated: true, user_id: `user:${token}` });
  }
}

interface Fixture {
  agent: BaseAgent;
  llm: EchoLLM;
  skill: A2ATransportSkill;
  fetch: (request: Request) => Promise<Response>;
}

function buildFixture(options: { identity?: boolean; skillConfig?: Record<string, unknown>; basePath?: string } = {}): Fixture {
  const llm = new EchoLLM();
  const skill = new A2ATransportSkill(options.skillConfig ?? {});
  const skills: Skill[] = [llm, new ToolsSkill(), skill];
  if (options.identity !== false) skills.push(new BearerIdentity());
  const agent = new BaseAgent({ name: vectors.agent.name, description: vectors.agent.description, skills });
  const fetchHandler = createFetchHandler(agent, { basePath: options.basePath ?? BASE_PATH, publicUrl: ORIGIN });
  return { agent, llm, skill, fetch: fetchHandler };
}

function credential(name: string): string {
  return vectors.credentials[name];
}

function rpcRequest(caller: string, headers: Record<string, string>, body: unknown): Request {
  return new Request(`${ORIGIN}${BASE_PATH}${vectors.agent.rpc_path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers, authorization: credential(caller) },
    body: JSON.stringify(body),
  });
}

async function rpc(f: Fixture, caller: string, method: string, params: unknown, headers: Record<string, string> = { 'a2a-version': '1.0' }) {
  const res = await f.fetch(rpcRequest(caller, headers, { jsonrpc: '2.0', id: `t-${Math.random()}`, method, params }));
  return { status: res.status, body: (await res.json()) as { result?: Record<string, unknown>; error?: { code: number; data?: Array<{ reason: string }> } } };
}

async function pollTask(f: Fixture, caller: string, id: string): Promise<Task> {
  for (let i = 0; i < 50; i += 1) {
    const { body } = await rpc(f, caller, 'GetTask', { id });
    const task = body.result?.task as Task | undefined;
    if (!task) throw new Error(`GetTask failed: ${JSON.stringify(body)}`);
    if (['TASK_STATE_COMPLETED', 'TASK_STATE_FAILED', 'TASK_STATE_CANCELED', 'TASK_STATE_REJECTED'].includes(task.status.state)) return task;
    await new Promise((r) => setTimeout(r, 10));
  }
  throw new Error('task never settled');
}

function replyOf(task: Task): string {
  const fromArtifacts = task.artifacts.flatMap((a) => a.parts).map((p) => p.text ?? '').join('');
  return fromArtifacts || (task.status.message?.parts ?? []).map((p) => p.text ?? '').join('');
}

async function sseEvents(res: Response): Promise<unknown[]> {
  const text = await res.text();
  return text
    .split('\n\n')
    .filter((chunk) => chunk.startsWith('data: '))
    .map((chunk) => JSON.parse(chunk.slice('data: '.length)));
}

function eventKind(event: StreamResponse): string {
  return Object.keys(event)[0];
}

// ---------------------------------------------------------------------------
// JSON-RPC vectors
// ---------------------------------------------------------------------------

describe('A2A v1.0 JSON-RPC (shared vectors)', () => {
  let f: Fixture;
  beforeEach(() => {
    f = buildFixture();
  });
  afterEach(async () => {
    await f.skill.cleanup();
  });

  for (const c of vectors.jsonrpc as Array<Record<string, any>>) {
    it(c.name, async () => {
      const res = await f.fetch(rpcRequest(c.caller, c.headers, c.body));
      expect(res.status).toBe(c.expect.status);
      const body = await res.json();
      expect(body.jsonrpc).toBe('2.0');
      expect(body.id).toEqual(c.body.id ?? null);
      if (c.expect.error) {
        expect(body.result).toBeUndefined();
        expect(body.error.code).toBe(c.expect.error.code);
        if (c.expect.error.reason) {
          expect(body.error.data[0]['@type']).toBe('type.googleapis.com/google.rpc.ErrorInfo');
          expect(body.error.data[0].reason).toBe(c.expect.error.reason);
          expect(body.error.data[0].domain).toBe('a2a-protocol.org');
        }
        return;
      }
      expect(body.error).toBeUndefined();
      if (c.expect.result_equals) {
        expect(body.result).toEqual(c.expect.result_equals);
        return;
      }
      expect(c.expect.result_kind).toBe('task');
      const task = body.result.task as Task;
      expect(typeof task.id).toBe('string');
      expect(typeof task.contextId).toBe('string');
      expect(c.expect.state_in).toContain(task.status.state);
      if (c.expect.context_id) expect(task.contextId).toBe(c.expect.context_id);
      if (c.expect.reply !== undefined) expect(replyOf(task)).toBe(c.expect.reply);
      if (c.expect.status_message_role) expect(task.status.message?.role).toBe(c.expect.status_message_role);
      if (c.expect.history_length) expect(task.history).toHaveLength(c.expect.history_length);
      if (c.expect.history_first_parts) expect(task.history[0].parts).toEqual(c.expect.history_first_parts);
      if (c.expect.then_get_task) {
        const final = await pollTask(f, c.caller, task.id);
        expect(final.status.state).toBe(c.expect.then_get_task.final_state);
        expect(replyOf(final)).toBe(c.expect.then_get_task.reply);
        expect(final.artifacts[0].parts[0].text).toBe(c.expect.then_get_task.reply);
      }
    });
  }

  for (const flow of vectors.flows as Array<Record<string, any>>) {
    it(`flow: ${flow.name}`, async () => {
      const ids: Record<string, string> = {};
      for (const step of flow.steps) {
        if (step.send !== undefined) {
          const message = { messageId: `m-${Math.random()}`, role: 'ROLE_USER', parts: [{ text: step.send }], ...(step.context_id ? { contextId: step.context_id } : {}) };
          const { body } = await rpc(f, step.caller, 'SendMessage', { message });
          const task = body.result?.task as Task;
          expect(task.status.state).toBe('TASK_STATE_COMPLETED');
          ids[step.as] = task.id;
        } else if (step.get) {
          const { body } = await rpc(f, step.caller, 'GetTask', { id: ids[step.get] });
          if (step.expect_error) {
            expect(body.error?.code).toBe(step.expect_error.code);
            expect(body.error?.data?.[0].reason).toBe(step.expect_error.reason);
          } else {
            const task = body.result?.task as Task;
            if (step.expect_state) expect(task.status.state).toBe(step.expect_state);
            if (step.expect_context_id) expect(task.contextId).toBe(step.expect_context_id);
          }
        } else if (step.cancel) {
          const { body } = await rpc(f, step.caller, 'CancelTask', { id: ids[step.cancel] });
          expect(body.error?.code).toBe(step.expect_error.code);
          expect(body.error?.data?.[0].reason).toBe(step.expect_error.reason);
        } else if (step.list) {
          const { body } = await rpc(f, step.caller, 'ListTasks', step.context_id ? { contextId: step.context_id } : {});
          const listed = (body.result?.tasks as Task[]).map((t) => t.id);
          if (step.expect_contains) expect(listed).toContain(ids[step.expect_contains]);
          if (step.expect_not_contains) expect(listed).not.toContain(ids[step.expect_not_contains]);
          if (step.expect_count !== undefined) expect(listed).toHaveLength(step.expect_count);
        }
      }
    });
  }

  it('SendStreamingMessage streams the task, artifact chunks and the terminal status', async () => {
    for (const body of [vectors.streaming.jsonrpc_body, vectors.streaming.dotted_body]) {
      const res = await f.fetch(rpcRequest('caller_a', {}, body));
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toContain(vectors.streaming.content_type);
      const events = (await sseEvents(res)) as Array<{ jsonrpc: string; id: unknown; result: StreamResponse }>;
      expect(events.length).toBeGreaterThanOrEqual(3);
      for (const e of events) {
        expect(e.jsonrpc).toBe('2.0');
        expect(e.id).toBe(body.id);
      }
      const results = events.map((e) => e.result);
      expect(eventKind(results[0])).toBe(vectors.streaming.expect.first_event);
      expect(vectors.streaming.expect.first_state_in).toContain((results[0] as { task: Task }).task.status.state);
      expect(results.some((r) => 'artifactUpdate' in r)).toBe(vectors.streaming.expect.has_artifact_update);
      const last = results[results.length - 1] as { statusUpdate: { status: { state: string; message: { parts: Array<{ text: string }> } } } };
      expect(eventKind(last as StreamResponse)).toBe(vectors.streaming.expect.last_event);
      expect(last.statusUpdate.status.state).toBe(vectors.streaming.expect.last_state);
      expect(last.statusUpdate.status.message.parts.map((p) => p.text).join('')).toBe(vectors.streaming.expect.reply);
      const chunks = results.filter((r): r is { artifactUpdate: { artifact: { parts: Array<{ text: string }> } } } => 'artifactUpdate' in r);
      expect(chunks.map((c) => c.artifactUpdate.artifact.parts.map((p) => p.text).join('')).join('')).toBe(vectors.streaming.expect.reply);
    }
  });

  it('SubscribeToTask replays a finished task and ends at its terminal status', async () => {
    const { body } = await rpc(f, 'caller_a', 'SendMessage', { message: { messageId: 'sub-1', role: 'ROLE_USER', parts: [{ text: 'replay' }] } });
    const id = (body.result?.task as Task).id;
    const res = await f.fetch(rpcRequest('caller_a', {}, { jsonrpc: '2.0', id: 'sub', method: 'SubscribeToTask', params: { id } }));
    expect(res.status).toBe(200);
    const events = (await sseEvents(res)) as Array<{ result: StreamResponse }>;
    expect(eventKind(events[0].result)).toBe('task');
    const last = events[events.length - 1].result as { statusUpdate: { status: { state: string } } };
    expect(last.statusUpdate.status.state).toBe('TASK_STATE_COMPLETED');
    // Another caller cannot subscribe to it.
    const other = await rpc(f, 'caller_b', 'tasks/resubscribe', { id }, {});
    expect(other.body.error?.code).toBe(-32001);
  });

  it('a running task is WORKING under returnImmediately and can be canceled', async () => {
    f.llm.delayMs = 300;
    const { body } = await rpc(f, 'caller_a', 'SendMessage', {
      message: { messageId: 'slow-1', role: 'ROLE_USER', parts: [{ text: 'slow' }] },
      configuration: { returnImmediately: true },
    });
    const task = body.result?.task as Task;
    expect(task.status.state).toBe('TASK_STATE_WORKING');
    const got = await rpc(f, 'caller_a', 'GetTask', { id: task.id });
    expect((got.body.result?.task as Task).status.state).toBe('TASK_STATE_WORKING');
    const canceled = await rpc(f, 'caller_a', 'CancelTask', { id: task.id });
    expect((canceled.body.result?.task as Task).status.state).toBe('TASK_STATE_CANCELED');
    await new Promise((r) => setTimeout(r, 400));
    const after = await rpc(f, 'caller_a', 'GetTask', { id: task.id });
    expect((after.body.result?.task as Task).status.state).toBe('TASK_STATE_CANCELED');
  });

  it('a blocking send answers the task as it stands after blocking_timeout_seconds', async () => {
    const slow = buildFixture({ skillConfig: { blocking_timeout_seconds: 0.05 } });
    slow.llm.delayMs = 300;
    const { body } = await rpc(slow, 'caller_a', 'SendMessage', { message: { messageId: 'b-1', role: 'ROLE_USER', parts: [{ text: 'later' }] } });
    const task = body.result?.task as Task;
    expect(task.status.state).toBe('TASK_STATE_WORKING');
    const final = await pollTask(slow, 'caller_a', task.id);
    expect(final.status.state).toBe('TASK_STATE_COMPLETED');
    expect(replyOf(final)).toBe('echo: later');
    await slow.skill.cleanup();
  });

  it("an identity skill's refusal is the gate's 401, before any task exists", async () => {
    const res = await f.fetch(
      new Request(`${ORIGIN}${BASE_PATH}/a2a`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer refuse-me' },
        body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'SendMessage', params: { message: { messageId: 'x', role: 'ROLE_USER', parts: [{ text: 'hi' }] } } }),
      }),
    );
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error.code).toBe('unauthorized');
    expect(f.skill.taskCount).toBe(0);
  });

  it('expired tasks are gone after task_ttl_seconds', async () => {
    const short = buildFixture({ skillConfig: { task_ttl_seconds: 0.05 } });
    const { body } = await rpc(short, 'caller_a', 'SendMessage', { message: { messageId: 'ttl-1', role: 'ROLE_USER', parts: [{ text: 'ttl' }] } });
    const id = (body.result?.task as Task).id;
    expect((await rpc(short, 'caller_a', 'GetTask', { id })).body.result).toBeDefined();
    await new Promise((r) => setTimeout(r, 80));
    expect((await rpc(short, 'caller_a', 'GetTask', { id })).body.error?.code).toBe(-32001);
    await short.skill.cleanup();
  });
});

describe('A2A v1.0 task ownership without an identity skill', () => {
  it('keys tasks on the presented credential, so another bearer sees nothing', async () => {
    const f = buildFixture({ identity: false });
    const { body } = await rpc(f, 'caller_a', 'SendMessage', { message: { messageId: 'c-1', role: 'ROLE_USER', parts: [{ text: 'cred' }] } });
    const id = (body.result?.task as Task).id;
    expect((await rpc(f, 'caller_a', 'GetTask', { id })).body.result).toBeDefined();
    expect((await rpc(f, 'caller_b', 'GetTask', { id })).body.error?.code).toBe(-32001);
    await f.skill.cleanup();
  });
});

// ---------------------------------------------------------------------------
// HTTP+JSON vectors
// ---------------------------------------------------------------------------

describe('A2A v1.0 HTTP+JSON binding (shared vectors)', () => {
  let f: Fixture;
  beforeEach(() => {
    f = buildFixture();
  });
  afterEach(async () => {
    await f.skill.cleanup();
  });

  function restRequest(caller: string, method: string, subPath: string, headers: Record<string, string>, body?: unknown): Request {
    return new Request(`${ORIGIN}${BASE_PATH}${subPath}`, {
      method,
      headers: { ...headers, authorization: credential(caller) },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    });
  }

  for (const c of vectors.rest as Array<Record<string, any>>) {
    it(c.name, async () => {
      const res = await f.fetch(restRequest(c.caller, c.method, c.path, c.headers, c.body));
      expect(res.status).toBe(c.expect.status);
      if (c.expect.content_type) expect(res.headers.get('content-type')).toContain(c.expect.content_type);
      const body = await res.json();
      if (c.expect.error) {
        expect(body.error.code).toBe(c.expect.status);
        expect(body.error.status).toBe(c.expect.error.status);
        expect(body.error.details[0].reason).toBe(c.expect.error.reason);
        return;
      }
      if (c.expect.result_equals) {
        expect(body).toEqual(c.expect.result_equals);
        return;
      }
      const task = body.task as Task;
      expect(c.expect.state_in).toContain(task.status.state);
      if (c.expect.reply !== undefined) expect(replyOf(task)).toBe(c.expect.reply);
    });
  }

  it('flow: send, read, cancel and subscribe over REST', async () => {
    const ids: Record<string, string> = {};
    for (const step of vectors.rest_flow.steps as Array<Record<string, any>>) {
      if (step.send !== undefined) {
        const res = await f.fetch(restRequest(step.caller, 'POST', '/a2a/message:send', { 'content-type': 'application/json' }, { message: { messageId: 'rf', role: 'ROLE_USER', parts: [{ text: step.send }] } }));
        ids[step.as] = ((await res.json()).task as Task).id;
      } else if (step.get) {
        const res = await f.fetch(restRequest(step.caller, 'GET', `/a2a/tasks/${ids[step.get]}`, {}));
        expect(res.status).toBe(step.expect_status);
        if (step.expect_state) expect(((await res.json()) as Task).status.state).toBe(step.expect_state);
      } else if (step.cancel) {
        const res = await f.fetch(restRequest(step.caller, 'POST', `/a2a/tasks/${ids[step.cancel]}:cancel`, {}));
        expect(res.status).toBe(step.expect_status);
        if (step.expect_reason) expect((await res.json()).error.details[0].reason).toBe(step.expect_reason);
      } else if (step.subscribe) {
        for (const method of ['POST', 'GET']) {
          const res = await f.fetch(restRequest(step.caller, method, `/a2a/tasks/${ids[step.subscribe]}:subscribe`, {}));
          expect(res.status).toBe(step.expect_status);
          const events = (await sseEvents(res)) as StreamResponse[];
          expect(eventKind(events[0])).toBe(step.expect_first_event);
        }
      }
    }
  });

  it('POST /message:stream streams bare StreamResponse events', async () => {
    const res = await f.fetch(restRequest('caller_a', 'POST', '/a2a/message:stream', { 'content-type': 'application/json', 'a2a-version': '1.0' }, vectors.streaming.rest_body));
    expect(res.status).toBe(200);
    const events = (await sseEvents(res)) as StreamResponse[];
    expect(eventKind(events[0])).toBe('task');
    expect('jsonrpc' in (events[0] as object)).toBe(false);
    const last = events[events.length - 1] as { statusUpdate: { status: { state: string } } };
    expect(last.statusUpdate.status.state).toBe('TASK_STATE_COMPLETED');
  });
});

// ---------------------------------------------------------------------------
// Dispatch through the servers
// ---------------------------------------------------------------------------

describe('A2A v1.0 route dispatch', () => {
  it('getHttpHandler matches the {id} patterns, most specific first', () => {
    const { agent } = buildFixture();
    expect(agent.getHttpHandler('/a2a/tasks/abc', 'GET')?.path).toBe('/a2a/tasks/{id}');
    expect(agent.getHttpHandler('/a2a/tasks/abc:cancel', 'POST')?.path).toBe('/a2a/tasks/{id}:cancel');
    expect(agent.getHttpHandler('/a2a/tasks/abc:subscribe', 'GET')?.path).toBe('/a2a/tasks/{id}:subscribe');
    expect(agent.getHttpHandler('/a2a/tasks/abc:subscribe', 'POST')?.path).toBe('/a2a/tasks/{id}:subscribe');
    expect(agent.getHttpHandler('/a2a/tasks', 'GET')?.path).toBe('/a2a/tasks');
    expect(agent.getHttpHandler('/a2a/tasks/a/b', 'GET')).toBeUndefined();
    expect(agent.getHttpHandler('/a2a/message:send', 'POST')?.path).toBe('/a2a/message:send');
  });

  it('the Hono app serves the literal routes it mounts and the pattern routes through the fallback', async () => {
    const f = buildFixture();
    const { app } = createAgentApp(f.agent, { basePath: BASE_PATH, logging: false, publicUrl: ORIGIN });
    const send = await app.fetch(
      new Request(`${ORIGIN}${BASE_PATH}/a2a/message:send`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: credential('caller_a') },
        body: JSON.stringify({ message: { messageId: 'h-1', role: 'ROLE_USER', parts: [{ text: 'via hono' }] } }),
      }),
    );
    expect(send.status).toBe(200);
    const task = (await send.json()).task as Task;
    expect(replyOf(task)).toBe('echo: via hono');
    const get = await app.fetch(new Request(`${ORIGIN}${BASE_PATH}/a2a/tasks/${task.id}`, { headers: { authorization: credential('caller_a') } }));
    expect(get.status).toBe(200);
    expect(((await get.json()) as Task).id).toBe(task.id);
    // The floor keeps anonymous callers off the task routes and the sends.
    const anon = await app.fetch(new Request(`${ORIGIN}${BASE_PATH}/a2a/tasks/${task.id}`));
    expect(anon.status).toBe(401);
    const anonSend = await app.fetch(new Request(`${ORIGIN}${BASE_PATH}/a2a/message:send`, { method: 'POST', body: '{}' }));
    expect(anonSend.status).toBe(401);
    // And not off the card.
    const card = await app.fetch(new Request(`${ORIGIN}${BASE_PATH}/.well-known/agent-card.json`));
    expect(card.status).toBe(200);
    await f.skill.cleanup();
  });

  it('the pre-v1 card handler is gone: /.well-known/agent.json is the registration card', async () => {
    const f = buildFixture();
    expect(f.agent.getHttpHandler('/.well-known/agent.json', 'GET')).toBeUndefined();
    const res = await f.fetch(new Request(`${ORIGIN}${BASE_PATH}/.well-known/agent.json`));
    const card = await res.json();
    expect(card.client_id).toBe(`${ORIGIN}${BASE_PATH}/.well-known/agent.json`);
    expect(card.url).toBe(`${ORIGIN}${BASE_PATH}`);
  });
});

// ---------------------------------------------------------------------------
// The card
// ---------------------------------------------------------------------------

describe('A2A v1.0 agent card', () => {
  it('serves the fixture card beside agent.json, unsigned without an identity', async () => {
    const f = buildFixture({ skillConfig: { public_url: ORIGIN } });
    const res = await f.fetch(new Request(`${ORIGIN}${BASE_PATH}${vectors.card.well_known_path}`));
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain(vectors.card.content_type);
    expect(res.headers.get('cache-control')).toBe(vectors.card.cache_control);
    expect(res.headers.get('etag')).toBeTruthy();
    const card = await res.json();
    const e = vectors.card.expect;
    expect(card.name).toBe(e.name);
    expect(card.description).toBe(e.description);
    expect(card.supportedInterfaces).toEqual(e.supportedInterfaces);
    expect(card.version).toBe(e.version);
    expect(card.capabilities).toEqual(e.capabilities);
    expect(card.securitySchemes).toEqual(e.securitySchemes);
    expect(card.securityRequirements).toEqual(e.securityRequirements);
    expect(card.defaultInputModes).toEqual(e.defaultInputModes);
    expect(card.defaultOutputModes).toEqual(e.defaultOutputModes);
    const ids = card.skills.map((s: { id: string }) => s.id);
    for (const id of e.skill_ids_include) expect(ids).toContain(id);
    for (const id of e.skill_ids_exclude) expect(ids).not.toContain(id);
    for (const s of card.skills) expect(s.tags).toEqual(e.skill_tags);
    expect(card.signatures).toBeUndefined();
    const again = await f.fetch(new Request(`${ORIGIN}${BASE_PATH}${vectors.card.well_known_path}`, { headers: { 'if-none-match': res.headers.get('etag')! } }));
    expect(again.status).toBe(304);
  });

  it('falls back to the request origin for the interface URL when nothing names the agent URL', async () => {
    const f = buildFixture({ basePath: '/agents/mini' });
    const res = await f.fetch(new Request(`http://127.0.0.1:4242/agents/mini/.well-known/agent-card.json`));
    const card = await res.json();
    expect(card.supportedInterfaces[0].url).toBe('http://127.0.0.1:4242/agents/mini/a2a');
  });

  it('signs the card with the agent identity and the signature verifies against its key set', async () => {
    const f = buildFixture();
    const identity = new AgentIdentity({ agentId: 'a2a-echo', issuer: `${ORIGIN}${BASE_PATH}` });
    await identity.initialize();
    f.agent.identity = identity;
    const res = await f.fetch(new Request(`${ORIGIN}${BASE_PATH}/.well-known/agent-card.json`));
    const card = await res.json();
    expect(card.supportedInterfaces[0].url).toBe(`${ORIGIN}${BASE_PATH}/a2a`);
    expect(card.signatures).toHaveLength(1);
    const header = JSON.parse(new TextDecoder().decode(base64urlDecode(card.signatures[0].protected)));
    expect(header).toEqual({ alg: 'EdDSA', jku: `${ORIGIN}${BASE_PATH}/.well-known/jwks.json`, kid: identity.kid, typ: 'JOSE' });
    const verified = await verifyAgentCard(card, { keys: identity.getJwks().keys as JsonWebKey[] });
    expect(verified).toMatchObject({ ok: true, kid: identity.kid, alg: 'EdDSA', checked: 1 });
    // Through the jku, fetched only from the card's own origin.
    const viaJku = await verifyAgentCard(card, {
      cardUrl: `${ORIGIN}${BASE_PATH}/.well-known/agent-card.json`,
      fetch: (async (url: string) => {
        expect(url).toBe(`${ORIGIN}${BASE_PATH}/.well-known/jwks.json`);
        return new Response(JSON.stringify(identity.getJwks()), { headers: { 'content-type': 'application/json' } });
      }) as unknown as typeof fetch,
    });
    expect(viaJku.ok).toBe(true);
    const tampered = { ...card, description: 'edited after signing' };
    expect((await verifyAgentCard(tampered, { keys: identity.getJwks().keys as JsonWebKey[] })).ok).toBe(false);
    const wrongKey = new AgentIdentity({ agentId: 'other', issuer: ORIGIN });
    await wrongKey.initialize();
    expect((await verifyAgentCard(card, { keys: wrongKey.getJwks().keys as JsonWebKey[] })).ok).toBe(false);
    const foreignJku = { ...card, signatures: [{ ...card.signatures[0], protected: base64url(new TextEncoder().encode(JSON.stringify({ alg: 'EdDSA', kid: identity.kid, typ: 'JOSE', jku: 'https://evil.example/jwks.json' }))) }] };
    const refused = await verifyAgentCard(foreignJku, { cardUrl: `${ORIGIN}/x`, fetch: (async () => { throw new Error('must not fetch'); }) as unknown as typeof fetch });
    expect(refused.ok).toBe(false);
    expect(refused.reason).toContain('evil.example');
  });
});

// ---------------------------------------------------------------------------
// Signature and canonicalisation vectors
// ---------------------------------------------------------------------------

describe('A2A card signature vectors (shared)', () => {
  const v = vectors.signatures.hs256_vector;

  it("reproduces the spec's HS256 byte assembly", async () => {
    expect(canonicalize(cardSigningPayload(v.card))).toBe(v.canonical);
    const digest = await crypto.subtle.digest('SHA-256', cardSigningBytes(v.card) as unknown as ArrayBuffer);
    expect(Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')).toBe(v.sha256);
    const secret = new TextEncoder().encode(v.key);
    const signed = await signAgentCard(v.card, hmacCardSigner(v.protected.kid, secret));
    expect(signed.signatures[0].protected).toBe(v.protected_b64);
    expect(signed.signatures[0].signature).toBe(v.signature);
    expect((await verifyAgentCard(signed, { secret })).ok).toBe(true);
    expect((await verifyAgentCard(signed, { secret: new TextEncoder().encode('other') })).ok).toBe(false);
    // HS256 is never accepted without a secret.
    expect((await verifyAgentCard(signed, {})).ok).toBe(false);
  });

  it('round-trips EdDSA with kid and jku', async () => {
    const identity = new AgentIdentity({ agentId: 'vec', issuer: 'https://agent.example.com' });
    await identity.initialize();
    const signed = await signAgentCard(v.card, identityCardSigner(identity));
    const header = JSON.parse(new TextDecoder().decode(base64urlDecode(signed.signatures[0].protected)));
    expect(header.jku).toBe(vectors.signatures.eddsa_round_trip.jku);
    expect(header.kid).toBe(identity.kid);
    expect((await verifyAgentCard(signed, { keys: identity.getJwks().keys as JsonWebKey[] })).ok).toBe(true);
    expect((await verifyAgentCard({ ...signed, name: 'changed' }, { keys: identity.getJwks().keys as JsonWebKey[] })).ok).toBe(false);
    expect((await verifyAgentCard(signed, { keys: [] })).ok).toBe(false);
    // Two signatures: one bad, one good, is a verified card.
    const twice = { ...signed, signatures: [{ protected: signed.signatures[0].protected, signature: base64url(new Uint8Array(64)) }, ...signed.signatures] };
    expect((await verifyAgentCard(twice, { keys: identity.getJwks().keys as JsonWebKey[] })).checked).toBe(2);
    expect((await verifyAgentCard(twice, { keys: identity.getJwks().keys as JsonWebKey[] })).ok).toBe(true);
  });

  for (const c of vectors.signatures.signing_payload_cases as Array<{ name: string; card: Record<string, unknown>; canonical: string }>) {
    it(`signing payload: ${c.name}`, () => {
      expect(canonicalize(cardSigningPayload(c.card))).toBe(c.canonical);
    });
  }
});

describe('JCS (RFC 8785) vectors (shared)', () => {
  for (const c of vectors.jcs.cases as Array<{ name: string; value: unknown; canonical: string }>) {
    it(c.name, () => {
      expect(canonicalize(c.value)).toBe(c.canonical);
    });
  }

  it('refuses what JSON cannot carry', () => {
    expect(() => canonicalize({ a: Number.NaN })).toThrow(/NaN/);
    expect(() => canonicalize({ a: Number.POSITIVE_INFINITY })).toThrow(/Infinity/);
  });
});

// ---------------------------------------------------------------------------
// Configuration (shared)
// ---------------------------------------------------------------------------

describe('A2A agent-file entry (shared config shapes)', () => {
  for (const shape of configShapes.shapes as Array<{ name: string; config: Record<string, unknown>; settings: unknown }>) {
    it(shape.name, () => {
      expect(new A2ATransportSkill(shape.config).settings).toEqual(shape.settings);
      expect(resolveA2ASettings(shape.config)).toEqual(shape.settings);
    });
  }

  it('peer tokens: the longest configured prefix wins, nothing else matches', () => {
    for (const c of configShapes.peer_token.cases as Array<{ url: string; token: string | null }>) {
      expect(peerTokenFor(c.url, configShapes.peer_token.peers)).toBe(c.token);
    }
  });

  it('camelCase spellings resolve to the same settings', () => {
    expect(resolveA2ASettings({ taskTtlSeconds: 5, blockingTimeoutSeconds: 2, documentationUrl: 'https://d', iconUrl: 'https://i' })).toMatchObject({
      task_ttl_seconds: 5,
      blocking_timeout_seconds: 2,
      documentation_url: 'https://d',
      icon_url: 'https://i',
    });
  });

  it('a2a is nameable in an agent file, with the same keys the Python loader reads', async () => {
    const resolved = await resolveSkillsByName([{ a2a: { peers: { 'https://peer.example': { token: 't' } }, blocking_timeout_seconds: 5 } }]);
    expect(resolved.unknown).toEqual([]);
    expect(resolved.failed).toEqual([]);
    const skill = resolved.byName.get('a2a') as unknown as A2ATransportSkill;
    expect(skill).toBeInstanceOf(A2ATransportSkill);
    expect(skill.settings.peers).toEqual({ 'https://peer.example': { token: 't' } });
    expect(skill.settings.blocking_timeout_seconds).toBe(5);
    const bare = await resolveSkillsByName(['a2a']);
    expect(bare.byName.get('a2a')).toBeInstanceOf(A2ATransportSkill);
  });
});

// ---------------------------------------------------------------------------
// The client (shared vectors)
// ---------------------------------------------------------------------------

describe('A2A client (shared vectors)', () => {
  for (const c of vectors.client.interface_cases as Array<{ name: string; card: Record<string, unknown>; expect: unknown }>) {
    it(`interface: ${c.name}`, () => {
      expect(pickInterface(c.card)).toEqual(c.expect);
    });
  }

  for (const c of vectors.client.result_cases as Array<{ name: string; result: unknown; expect: { kind: string; reply: string } }>) {
    it(`result: ${c.name}`, () => {
      const result = readSendResult(c.result);
      expect(c.expect.kind in result).toBe(true);
      expect(replyOfResult(result)).toBe(c.expect.reply);
    });
  }

  interface Sent {
    url: string;
    method: string;
    headers: Record<string, string>;
    body?: unknown;
  }

  /** A fetch that routes to the fixture agent and records what the client sent. */
  function recordingFetch(f: Fixture, log: Sent[]): typeof fetch {
    return (async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const headers: Record<string, string> = {};
      request.headers.forEach((value, key) => {
        headers[key] = value;
      });
      const body = request.method === 'POST' ? await request.clone().json() : undefined;
      log.push({ url: request.url, method: request.method, headers, body });
      return f.fetch(request);
    }) as typeof fetch;
  }

  it('discovers the card, sends SendMessage with A2A-Version 1.0 and the peer bearer, and reads the task', async () => {
    const peer = `${ORIGIN}${BASE_PATH}`;
    const f = buildFixture({ skillConfig: { public_url: ORIGIN, peers: { [peer]: { token: 'caller-a-token' } } } });
    const log: Sent[] = [];
    const result = await f.skill.callPeer(peer, 'hello peer', { fetch: recordingFetch(f, log) });
    expect(result.reply).toBe('echo: hello peer');
    expect(result.task?.status.state).toBe('TASK_STATE_COMPLETED');
    expect(result.cardUrl).toBe(`${peer}${vectors.client.card_paths[0]}`);
    expect(result.rpcUrl).toBe(`${peer}/a2a`);
    expect(log[0]).toMatchObject({ url: `${peer}${vectors.client.card_paths[0]}`, method: 'GET' });
    const send = log[1];
    expect(send.url).toBe(`${peer}/a2a`);
    expect(send.headers['a2a-version']).toBe(vectors.client.version_header['A2A-Version']);
    expect(send.headers.authorization).toBe('Bearer caller-a-token');
    expect((send.body as { method: string }).method).toBe('SendMessage');
    expect((send.body as { params: { message: unknown } }).params.message).toMatchObject({
      role: 'ROLE_USER',
      parts: [{ text: 'hello peer', mediaType: 'text/plain' }],
    });
    // The task is the caller's on the peer: readable with that bearer, not another.
    const mine = await rpc(f, 'caller_a', 'GetTask', { id: result.task!.id });
    expect((mine.body.result?.task as Task).id).toBe(result.task!.id);
    expect((await rpc(f, 'caller_b', 'GetTask', { id: result.task!.id })).body.error?.code).toBe(-32001);
    await f.skill.cleanup();
  });

  it('retries message/send once on -32601, falls back to agent.json on 404, refuses a v0.3-only card and a redirect', async () => {
    const card = {
      name: 'stub',
      description: '',
      supportedInterfaces: [{ url: 'https://stub.example/rpc', protocolBinding: 'JSONRPC', protocolVersion: '1.0' }],
      version: '1.0.0',
      capabilities: {},
      defaultInputModes: [],
      defaultOutputModes: [],
      skills: [],
    };
    const methods: string[] = [];
    const stub = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const url = new URL(request.url);
      if (url.pathname === vectors.client.card_paths[0]) return new Response('nope', { status: 404 });
      if (url.pathname === vectors.client.card_paths[1]) return Response.json(card);
      const body = (await request.json()) as { id: unknown; method: string };
      methods.push(body.method);
      if (body.method === 'SendMessage') {
        return Response.json({ jsonrpc: '2.0', id: body.id, error: { code: vectors.client.retry_on, message: 'Method not found' } });
      }
      return Response.json({
        jsonrpc: '2.0',
        id: body.id,
        result: { id: 't-legacy', contextId: 'c', status: { state: 'TASK_STATE_COMPLETED' }, artifacts: [{ artifactId: 'a', parts: [{ text: 'from a v0.3 server' }] }], history: [] },
      });
    }) as typeof fetch;
    const result = await callAgent('https://stub.example', 'hi', { fetch: stub });
    expect(result.cardUrl).toBe(`https://stub.example${vectors.client.card_paths[1]}`);
    expect(methods).toEqual(['SendMessage', vectors.client.retry_method]);
    expect(result.reply).toBe('from a v0.3 server');

    const legacy = (async () => Response.json({ url: 'https://old.example/', preferredTransport: 'JSONRPC', protocolVersion: '0.3' })) as unknown as typeof fetch;
    await expect(callAgent('https://old.example', 'hi', { fetch: legacy })).rejects.toThrow(/no A2A 1.0 JSON-RPC interface/);

    const redirecting = (async () => new Response(null, { status: 302, headers: { location: 'https://elsewhere.example/' } })) as unknown as typeof fetch;
    await expect(callAgent('https://moved.example', 'hi', { fetch: redirecting })).rejects.toThrow(/redirect/);
  });
});

describe('A2A part parsing', () => {
  it('reads v1.0 and v0.3 spellings to one shape', () => {
    expect(readPart({ text: 'a', mediaType: 'text/plain' }, 0)).toEqual({ text: 'a', mediaType: 'text/plain' });
    expect(readPart({ kind: 'text', text: 'a' }, 0)).toEqual({ text: 'a' });
    expect(readPart({ kind: 'file', file: { uri: 'https://x/y.png', mimeType: 'image/png', name: 'y.png' } }, 0)).toEqual({ url: 'https://x/y.png', mediaType: 'image/png', filename: 'y.png' });
    expect(readPart({ kind: 'file', file: { bytes: 'AAAA', mimeType: 'application/pdf' } }, 0)).toEqual({ raw: 'AAAA', mediaType: 'application/pdf' });
    expect(readPart({ kind: 'data', data: { k: 1 } }, 0)).toEqual({ data: { k: 1 } });
    expect(() => readPart({ kind: 'text' }, 0)).toThrow(/none of text, raw, url or data/);
  });
});
