/**
 * The schedule runner and the three deliverers (plan item 1.7, 2026-09-26):
 * a one-minute schedule runs a turn of the agent as its owner and delivers
 * to a file and to a local webhook server whose signature verifies, a
 * restart neither double-runs nor drops a missed slot, one run per schedule
 * at a time, the heartbeat sentinel suppresses delivery, and platform-chat
 * delivery goes through a mocked platform client. The words and vectors are
 * the shared fixture's (`python/tests/fixtures/daemon/cron.json`); the
 * Python suite runs the same in `tests/daemon/test_cron_runner_w1daemon.py`.
 */

import { afterAll, describe, expect, it } from 'vitest';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentIdentity } from '../../../src/crypto/identity';
import { MemoryNonceStore, verifyWebBotAuth } from '../../../src/crypto/web-bot-auth-verify';
import type { IAgent } from '../../../src/core/types';
import { parseCronBlock } from '../../../src/agents/schedules';
import {
  chatSessionId,
  chatTurn,
  deliver,
  deliverFile,
  deliverWebhook,
  webhookBody,
  type ChatClient,
  type DeliveryContext,
  type RunResult,
} from '../../../src/daemon/deliver';
import {
  HEARTBEAT_PROMPT,
  HEARTBEAT_SENTINEL,
  ScheduleRunner,
  isQuietHeartbeat,
  nextCronRun,
  parseIso,
  statePath,
} from '../../../src/daemon/schedule-runner';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/cron.json'), 'utf8')) as {
  turn: { caller: Record<string, unknown> };
  heartbeat: { prompt: string; sentinel: string; quiet: string[]; reports: string[] };
  file_entry: string;
  webhook: { body: string; heartbeat_body: string; content_type: string; backoff_seconds: number[] };
  chat: { sessions: Record<string, string>; turn: unknown; heartbeat_turn: unknown };
  details: Record<string, string>;
  next_run: [string, string, string, string][];
  state_example: { schedules: Record<string, Record<string, unknown>> };
};

const tempDir = tempDirs();
const T0 = Date.parse('2026-09-26T10:00:30Z');
const MINUTE = 60_000;
const ISSUER = 'https://agent.example/agents/reporter';

/** A served agent for the runner: answers `reply`, records what it was asked. */
function fakeAgent(reply: string | (() => Promise<string>), identity?: AgentIdentity) {
  const calls: { messages: unknown; options: Record<string, unknown> | undefined }[] = [];
  const agent = {
    name: 'reporter',
    identity,
    async run(messages: unknown, options?: Record<string, unknown>) {
      calls.push({ messages, options });
      return { content: typeof reply === 'string' ? reply : await reply() };
    },
  };
  return { agent: agent as unknown as IAgent, calls };
}

async function signingIdentity(): Promise<AgentIdentity> {
  const identity = new AgentIdentity({ agentId: 'reporter', issuer: ISSUER });
  await identity.initialize();
  return identity;
}

interface Received {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Buffer;
}

/** A local webhook: answers the queued statuses in order (200 once the queue is empty), keeps what it got. */
async function webhookServer(statuses: number[] = []): Promise<{ url: string; authority: string; received: Received[]; server: Server }> {
  const received: Received[] = [];
  const queue = [...statuses];
  const server = createServer((req: IncomingMessage, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const headers: Record<string, string> = {};
      for (const [name, value] of Object.entries(req.headers)) headers[name] = Array.isArray(value) ? value.join(', ') : String(value ?? '');
      received.push({ method: req.method ?? '', url: req.url ?? '', headers, body: Buffer.concat(chunks) });
      res.statusCode = queue.length ? queue.shift()! : 200;
      res.end('{}');
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address() as { port: number };
  const authority = `127.0.0.1:${address.port}`;
  servers.push(server);
  return { url: `http://${authority}/hook`, authority, received, server };
}

const servers: Server[] = [];
afterAll(() => {
  for (const server of servers) server.close();
});

const noSleep = { slept: [] as number[], sleep: async (seconds: number) => void noSleep.slept.push(seconds) };

function result(overrides: Partial<RunResult> = {}): RunResult {
  return {
    agent: 'reporter',
    schedule: 'daily-report',
    kind: 'cron',
    prompt: "Summarize yesterday's activity.",
    content: 'Nothing happened.',
    ranAt: '2026-09-28T07:00:00Z',
    ...overrides,
  };
}

function agentFolder(): string {
  const dir = tempDir('wa-cron-runner-');
  writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: reporter\n---\nYou report.\n');
  return dir;
}

describe('the words shared with Python', () => {
  it('the heartbeat prompt and sentinel are the fixture\'s', () => {
    expect(HEARTBEAT_PROMPT).toBe(FIXTURE.heartbeat.prompt);
    expect(HEARTBEAT_SENTINEL).toBe(FIXTURE.heartbeat.sentinel);
  });

  it('reads a quiet heartbeat however the model dresses the sentinel', () => {
    for (const text of FIXTURE.heartbeat.quiet) expect(isQuietHeartbeat(text), JSON.stringify(text)).toBe(true);
    for (const text of FIXTURE.heartbeat.reports) expect(isQuietHeartbeat(text), JSON.stringify(text)).toBe(false);
  });

  it.each(FIXTURE.next_run)('%s in %s after %s fires at %s', (expression, tz, after, expected) => {
    const next = nextCronRun(expression, tz, parseIso(after)!);
    expect(new Date(next!).toISOString().replace(/\.\d{3}Z$/, 'Z')).toBe(expected);
  });

  it('the webhook body and the chat turn are the fixture\'s', () => {
    expect(webhookBody(result())).toBe(FIXTURE.webhook.body);
    expect(webhookBody(result({ schedule: 'watch', kind: 'every', prompt: undefined, content: 'The queue is stuck.' }))).toBe(
      FIXTURE.webhook.heartbeat_body,
    );
    for (const [key, uuid] of Object.entries(FIXTURE.chat.sessions)) {
      const [agent, schedule] = key.split('/');
      expect(chatSessionId(agent, schedule)).toBe(uuid);
    }
    expect(chatTurn(result())).toEqual(FIXTURE.chat.turn);
    expect(chatTurn(result({ schedule: 'watch', kind: 'every', prompt: undefined, content: 'The queue is stuck.' }))).toEqual(
      FIXTURE.chat.heartbeat_turn,
    );
  });
});

describe('a one-minute schedule, run by the runner', () => {
  it('runs a turn as the owner and delivers to a file and to a signed webhook', async () => {
    const dir = agentFolder();
    const hook = await webhookServer();
    const identity = await signingIdentity();
    const { agent, calls } = fakeAgent('Nothing happened.', identity);
    let now = T0;
    const lines: string[] = [];
    const runner = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: (line) => lines.push(line) });
    const schedules = parseCronBlock([
      { name: 'minute-file', schedule: '* * * * *', prompt: 'Report.', deliver: { file: 'reports/minute.md' } },
      { name: 'minute-hook', schedule: '* * * * *', prompt: 'Report.', deliver: { webhook: hook.url } },
    ]);
    runner.setSchedules('reporter', dir, schedules);

    // Scheduled for the next whole minute, persisted before anything ran.
    const state = JSON.parse(readFileSync(statePath(dir, 'reporter'), 'utf8')) as typeof FIXTURE.state_example;
    expect(Object.keys(state.schedules)).toEqual(['minute-file', 'minute-hook']);
    expect(state.schedules['minute-file']).toEqual({ spec: 'cron * * * * * UTC', nextRun: '2026-09-26T10:01:00Z', lastFire: null, lastRun: null });
    expect(Object.keys(state.schedules['minute-file'])).toEqual(Object.keys(FIXTURE.state_example.schedules['daily-report']));

    // Not due yet: nothing runs.
    await runner.tick();
    expect(calls).toHaveLength(0);

    // Due: both run, concurrently, once.
    now = T0 + MINUTE;
    await runner.tick();
    expect(calls).toHaveLength(2);
    // The owner's own turn (the fixture's `turn.caller`), with the file's prompt.
    expect(calls[0].options?.auth).toMatchObject(FIXTURE.turn.caller);
    expect(calls[0].messages).toEqual([{ role: 'user', content: 'Report.' }]);

    // The file, with the fixture's entry.
    const entry = FIXTURE.file_entry.replace('{schedule}', 'minute-file').replace('{ran_at}', '2026-09-26T10:01:30Z').replace('{content}', 'Nothing happened.');
    expect(readFileSync(path.join(dir, 'reports', 'minute.md'), 'utf8')).toBe(entry);

    // The webhook: the fixture's body, signed as the agent, and the signature verifies.
    expect(hook.received).toHaveLength(1);
    const got = hook.received[0];
    expect(got.method).toBe('POST');
    expect(got.headers['content-type']).toBe(FIXTURE.webhook.content_type);
    expect(JSON.parse(got.body.toString('utf8'))).toEqual({
      agent: 'reporter',
      schedule: 'minute-hook',
      kind: 'cron',
      prompt: 'Report.',
      content: 'Nothing happened.',
      ran_at: '2026-09-26T10:01:30Z',
    });
    const jwk = identity.getJwks().keys[0] as unknown as { kid: string; x: string };
    const outcome = await verifyWebBotAuth(
      { method: got.method, target: '/hook', headers: got.headers, body: new Uint8Array(got.body) },
      {
        authorities: [hook.authority],
        scheme: 'http',
        keySets: { get: async () => ({ ok: true, keys: [{ thumbprint: jwk.kid, x: jwk.x }], ttlS: 300 }) },
        nonces: new MemoryNonceStore(),
      },
    );
    expect(outcome.ok ? 'ok' : outcome.refusal).toBe('ok');
    if (outcome.ok) expect(outcome.agent.principal).toBe(ISSUER);

    // The records, in the state and in the log.
    const listed = runner.list();
    expect(listed.map((s) => [s.name, (s.last_run as { outcome: string; detail: string }).outcome, (s.last_run as { detail: string }).detail])).toEqual([
      ['minute-file', 'delivered', 'file reports/minute.md'],
      ['minute-hook', 'delivered', `webhook ${hook.url} (signed as ${ISSUER})`],
    ]);
    expect(listed[0].next_run).toBe('2026-09-26T10:02:00Z');
    expect(listed[0].last_fire).toBe('2026-09-26T10:01:30Z');
    expect(lines).toContain('[daemon] reporter/minute-file: delivered (file reports/minute.md)');

    // The same minute again: nothing (the next fire is the next minute).
    await runner.tick();
    expect(calls).toHaveLength(2);
  });

  it('a restart does not double-run, and one missed slot catches up once', async () => {
    const dir = agentFolder();
    const { agent, calls } = fakeAgent('Nothing happened.');
    const schedules = parseCronBlock([{ name: 'minute', schedule: '* * * * *', prompt: 'Report.', deliver: { file: 'out.md' } }]);
    let now = T0;
    const first = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: () => {} });
    first.setSchedules('reporter', dir, schedules);
    now = T0 + MINUTE;
    await first.tick();
    expect(calls).toHaveLength(1);

    // Restarted ten seconds later: the slot already ran, nothing runs again.
    now = T0 + MINUTE + 10_000;
    const second = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: () => {} });
    second.setSchedules('reporter', dir, schedules);
    expect(second.list()[0].next_run).toBe('2026-09-26T10:02:00Z');
    expect(second.list()[0].last_run).toEqual({ at: '2026-09-26T10:01:30Z', outcome: 'delivered', detail: 'file out.md' });
    await second.tick();
    expect(calls).toHaveLength(1);

    // Back after sleeping through five slots: one catch-up run, then the schedule resumes from now.
    now = T0 + 6 * MINUTE + 10_000;
    const third = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: () => {} });
    third.setSchedules('reporter', dir, schedules);
    await third.tick();
    expect(calls).toHaveLength(2);
    expect(third.list()[0].next_run).toBe('2026-09-26T10:07:00Z');
    await third.tick();
    expect(calls).toHaveLength(2);

    // A changed timing is rescheduled from now, not from the saved fire.
    const changed = parseCronBlock([{ name: 'minute', schedule: '*/5 * * * *', prompt: 'Report.', deliver: { file: 'out.md' } }]);
    const fourth = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: () => {} });
    fourth.setSchedules('reporter', dir, changed);
    expect(fourth.list()[0].next_run).toBe('2026-09-26T10:10:00Z');
    // A disabled schedule keeps its record and has no next fire.
    const off = parseCronBlock([{ name: 'minute', schedule: '* * * * *', prompt: 'Report.', enabled: false, deliver: { file: 'out.md' } }]);
    fourth.setSchedules('reporter', dir, off);
    expect(fourth.list()[0].next_run).toBeNull();
    await fourth.tick();
    expect(calls).toHaveLength(2);
  });

  it('runs one schedule at a time and says when a slot is skipped', async () => {
    const dir = agentFolder();
    let release: (value: string) => void = () => {};
    const blocked = new Promise<string>((resolve) => {
      release = resolve;
    });
    const { agent, calls } = fakeAgent(() => blocked);
    let now = T0;
    const lines: string[] = [];
    const runner = new ScheduleRunner({ agentFor: () => agent, clock: () => now, log: (line) => lines.push(line) });
    runner.setSchedules('reporter', dir, parseCronBlock([{ name: 'minute', schedule: '* * * * *', prompt: 'Report.', deliver: { file: 'out.md' } }]));
    now = T0 + MINUTE;
    const ticking = runner.tick();
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(calls).toHaveLength(1);
    expect(runner.list()[0].running).toBe(true);
    // `webagents cron run` while it runs: refused with the fixture's words.
    expect(await runner.runNow('reporter', 'minute')).toEqual({ at: '2026-09-26T10:01:30Z', outcome: 'failed', detail: FIXTURE.details.already_running });
    // The next slot comes due while it still runs: skipped, and said.
    now = T0 + 2 * MINUTE;
    await runner.tick();
    expect(calls).toHaveLength(1);
    expect(lines).toContain('[daemon] reporter/minute: due, but the previous run is still going');
    release('Done.');
    await ticking;
    expect(runner.list()[0].running).toBe(false);
    expect((runner.list()[0].last_run as { outcome: string }).outcome).toBe('delivered');
  });

  it('records a turn that fails and an agent that is not served, and runs now on request', async () => {
    const dir = agentFolder();
    const failing = { name: 'reporter', run: async () => { throw new Error('no model key'); } } as unknown as IAgent;
    let served: IAgent | undefined = failing;
    const runner = new ScheduleRunner({ agentFor: () => served, clock: () => T0, log: () => {} });
    runner.setSchedules('reporter', dir, parseCronBlock([{ name: 'daily', schedule: '0 9 * * *', prompt: 'Report.', deliver: { file: 'out.md' } }]));
    expect(await runner.runNow('reporter', 'daily')).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'failed', detail: 'turn failed: no model key' });
    served = undefined;
    expect(await runner.runNow('reporter', 'daily')).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'failed', detail: FIXTURE.details.agent_missing });
    // Running now leaves the schedule's next fire alone.
    expect(runner.list()[0].next_run).toBe('2026-09-27T09:00:00Z');
    await expect(runner.runNow('reporter', 'nope')).rejects.toThrow('reporter/nope: no such schedule');
    expect(existsSync(path.join(dir, 'out.md'))).toBe(false);
  });
});

describe('the heartbeat', () => {
  it('runs the standing instructions with the heartbeat prompt and delivers only a report', async () => {
    const dir = agentFolder();
    let reply = HEARTBEAT_SENTINEL;
    const calls: { messages: unknown }[] = [];
    const agent = { name: 'reporter', async run(messages: unknown) { calls.push({ messages }); return { content: reply }; } } as unknown as IAgent;
    const recorded: unknown[] = [];
    const chat: ChatClient = {
      async target() { return { base: 'https://robutler.test', token: 't', agentId: 'agent-1' }; },
      async record(_target, turn) { recorded.push(turn); return 'chat-1'; },
    };
    const runner = new ScheduleRunner({
      agentFor: () => agent,
      clock: () => T0,
      log: () => {},
      deliver: (target, result, ctx) => deliver(target, result, { ...ctx, chat }),
    });
    runner.setSchedules('reporter', dir, parseCronBlock([{ name: 'watch', every: '1h', heartbeat: true, deliver: { chat: 'owner' } }]));

    // Nothing to report: the sentinel, dressed or bare, and an empty reply.
    for (const quiet of FIXTURE.heartbeat.quiet) {
      reply = quiet;
      expect(await runner.runNow('reporter', 'watch')).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'nothing', detail: FIXTURE.details.quiet });
    }
    expect(recorded).toHaveLength(0);
    expect(calls[0].messages).toEqual([{ role: 'user', content: HEARTBEAT_PROMPT }]);

    // A report: delivered, the report alone, into the schedule's own chat.
    reply = 'The queue is stuck.';
    expect(await runner.runNow('reporter', 'watch')).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'delivered', detail: 'chat owner (chat-1)' });
    expect(recorded).toEqual([FIXTURE.chat.heartbeat_turn]);
  });

  it('an empty reply to a prompt schedule is nothing too', async () => {
    const dir = agentFolder();
    const { agent } = fakeAgent('   ');
    const runner = new ScheduleRunner({ agentFor: () => agent, clock: () => T0, log: () => {} });
    runner.setSchedules('reporter', dir, parseCronBlock([{ name: 'daily', schedule: '0 9 * * *', prompt: 'Report.', deliver: { file: 'out.md' } }]));
    expect(await runner.runNow('reporter', 'daily')).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'nothing', detail: FIXTURE.details.empty });
    expect(existsSync(path.join(dir, 'out.md'))).toBe(false);
  });
});

describe('file delivery', () => {
  it('refuses a path that leaves the agent folder through a link', async () => {
    const dir = agentFolder();
    const outside = tempDir('wa-cron-outside-');
    symlinkSync(outside, path.join(dir, 'reports'));
    const ctx: DeliveryContext = { agentDir: dir };
    expect(await deliverFile({ kind: 'file', path: 'reports/daily.md' }, result(), ctx)).toEqual([
      'failed',
      FIXTURE.details.file_outside.replace('{path}', 'reports/daily.md'),
    ]);
    expect(existsSync(path.join(outside, 'daily.md'))).toBe(false);
    // A link to a file outside, too.
    symlinkSync(path.join(outside, 'log.md'), path.join(dir, 'log.md'));
    expect(await deliverFile({ kind: 'file', path: 'log.md' }, result(), ctx)).toEqual(['failed', "file log.md: outside the agent's folder"]);
    // Inside: appended, twice is two entries.
    mkdirSync(path.join(dir, 'inside'));
    expect(await deliverFile({ kind: 'file', path: 'inside/daily.md' }, result(), ctx)).toEqual(['delivered', 'file inside/daily.md']);
    await deliverFile({ kind: 'file', path: 'inside/daily.md' }, result({ content: 'Again.' }), ctx);
    expect(readFileSync(path.join(dir, 'inside', 'daily.md'), 'utf8')).toBe(
      "## daily-report, 2026-09-28T07:00:00Z\n\nNothing happened.\n\n## daily-report, 2026-09-28T07:00:00Z\n\nAgain.\n\n",
    );
  });
});

describe('webhook delivery', () => {
  it('posts unsigned, and says so, when the agent holds no key', async () => {
    const hook = await webhookServer();
    const ctx: DeliveryContext = { agentDir: agentFolder(), agent: { name: 'reporter' }, sleep: noSleep.sleep };
    expect(await deliverWebhook({ kind: 'webhook', url: hook.url, timeout: 5, retries: 0 }, result(), ctx)).toEqual([
      'delivered',
      `webhook ${hook.url} (unsigned: ${FIXTURE.details.no_identity})`,
    ]);
    expect(hook.received[0].headers.signature).toBeUndefined();
    expect(hook.received[0].body.toString('utf8')).toBe(FIXTURE.webhook.body);
  });

  it('retries a 5xx with the fixture backoff, signing each try afresh', async () => {
    const hook = await webhookServer([500, 503]);
    const identity = await signingIdentity();
    noSleep.slept.length = 0;
    const ctx: DeliveryContext = { agentDir: agentFolder(), agent: { name: 'reporter', identity }, sleep: noSleep.sleep };
    expect(await deliverWebhook({ kind: 'webhook', url: hook.url, timeout: 5, retries: 3 }, result(), ctx)).toEqual([
      'delivered',
      `webhook ${hook.url} (signed as ${ISSUER})`,
    ]);
    expect(hook.received).toHaveLength(3);
    expect(noSleep.slept).toEqual(FIXTURE.webhook.backoff_seconds.slice(0, 2));
    const nonces = hook.received.map((r) => /nonce="([^"]+)"/.exec(r.headers['signature-input'])?.[1]);
    expect(new Set(nonces).size).toBe(3);
  });

  it('gives up after the retries, and does not retry a final answer', async () => {
    const down = await webhookServer([500, 500, 500]);
    noSleep.slept.length = 0;
    const ctx: DeliveryContext = { agentDir: agentFolder(), sleep: noSleep.sleep };
    expect(await deliverWebhook({ kind: 'webhook', url: down.url, timeout: 5, retries: 2 }, result(), ctx)).toEqual([
      'failed',
      `webhook ${down.url}: answered 500 after 3 tries`,
    ]);
    expect(noSleep.slept).toEqual([1, 2]);

    const refusing = await webhookServer([403]);
    expect(await deliverWebhook({ kind: 'webhook', url: refusing.url, timeout: 5, retries: 3 }, result(), ctx)).toEqual([
      'failed',
      `webhook ${refusing.url}: answered 403 after 1 try`,
    ]);
    expect(refusing.received).toHaveLength(1);

    const closed = await webhookServer();
    closed.server.close();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(await deliverWebhook({ kind: 'webhook', url: closed.url, timeout: 5, retries: 1 }, result(), ctx)).toEqual([
      'failed',
      `webhook ${closed.url}: could not be reached after 2 tries`,
    ]);
  });
});

describe('chat delivery, against a mocked platform client', () => {
  it('records the prompt as the owner and the reply as the agent, in the schedule\'s chat', async () => {
    const recorded: { target: unknown; turn: unknown }[] = [];
    const chat: ChatClient = {
      async target(agentDir, agentName) {
        expect(agentName).toBe('reporter');
        expect(existsSync(path.join(agentDir, 'AGENT.md'))).toBe(true);
        return { base: 'https://robutler.test', token: 'tok', agentId: 'agent-1' };
      },
      async record(target, turn) {
        recorded.push({ target, turn });
        return 'chat-42';
      },
    };
    const ctx: DeliveryContext = { agentDir: agentFolder(), chat };
    expect(await deliver({ kind: 'chat', to: 'owner' }, result(), ctx)).toEqual(['delivered', 'chat owner (chat-42)']);
    expect(recorded).toEqual([{ target: { base: 'https://robutler.test', token: 'tok', agentId: 'agent-1' }, turn: FIXTURE.chat.turn }]);
  });

  it('fails with the chat\'s own sentences when signed out, unpublished, or refused', async () => {
    const ctx = (why: 'signed_out' | 'not_published' | 'refused'): DeliveryContext => ({
      agentDir: agentFolder(),
      chat: {
        async target() {
          return why === 'refused' ? { base: 'b', token: 't', agentId: 'a' } : why;
        },
        async record() {
          throw new Error('Robutler answered 500');
        },
      },
    });
    const loginCommand = FIXTURE.details.signed_out.replace('{login}', 'webagents login');
    const publishCommand = FIXTURE.details.not_published.replace('{publish}', 'webagents publish');
    expect(await deliver({ kind: 'chat', to: 'owner' }, result(), ctx('signed_out'))).toEqual(['failed', `chat owner: ${loginCommand}`]);
    expect(await deliver({ kind: 'chat', to: 'owner' }, result(), ctx('not_published'))).toEqual(['failed', `chat owner: ${publishCommand}`]);
    expect(await deliver({ kind: 'chat', to: 'owner' }, result(), ctx('refused'))).toEqual(['failed', 'chat owner: Robutler answered 500']);
  });
});
