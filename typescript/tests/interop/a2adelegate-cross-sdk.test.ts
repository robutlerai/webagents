/**
 * `delegate` across the two SDKs over A2A v1.0 (2026-09-27, the a2a-delegate
 * lane): a TypeScript agent and a Python agent, both served on loopback by
 * this test, each configured as an `a2a` peer of the other, and each one's
 * NLI skill delegates to the other. The hop goes over A2A with the peer's
 * bearer, the card is verified against the key set the other serves, and the
 * reply comes back with the unpaid note from the shared fixture
 * (`python/tests/fixtures/a2a/delegate_routing.json`).
 *
 * The TypeScript agent runs in-process through `serve()`; the Python agent is
 * `python/tests/interop/serve_a2adelegate_echo.py`, started with the
 * interpreter `WEBAGENTS_PYTHON` names, else the repo's
 * `python/.venv/bin/python`; it delegates to the TypeScript agent first, then
 * serves and prints its port. Without an interpreter the suite is SKIPPED with
 * the reason in its name, never passed quietly.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../src/core/agent';
import { Skill } from '../../src/core/skill';
import { handoff } from '../../src/core/decorators';
import type { Context, StructuredToolResult } from '../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../src/uamp/events';
import { serve, type ServeHandle } from '../../src/server/node';
import { A2ATransportSkill } from '../../src/skills/transport/a2a/skill';
import { NLISkill } from '../../src/skills/nli/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_DIR = path.resolve(HERE, '../../../python');
const HELPER = path.join(PYTHON_DIR, 'tests', 'interop', 'serve_a2adelegate_echo.py');
const VENV_PYTHON = path.join(PYTHON_DIR, '.venv', 'bin', 'python');
const PYTHON = process.env.WEBAGENTS_PYTHON || (existsSync(VENV_PYTHON) ? VENV_PYTHON : null);
const FIXTURE = JSON.parse(readFileSync(path.join(PYTHON_DIR, 'tests', 'fixtures', 'a2a', 'delegate_routing.json'), 'utf8')) as { unpaid_note: string };

const TS_TOKEN = 'ts-peer-token-for-python';
const PY_TOKEN = 'py-peer-token-for-typescript';
const TS_BASE_PATH = '/agents/ts-echo';

class EchoLLM extends Skill {
  @handoff({ name: 'echo-llm' })
  async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    const texts: string[] = [];
    for (const e of events) {
      if (e.type === 'input.text' && (e as { text: string }).text) texts.push((e as { text: string }).text);
    }
    const reply = `echo: ${texts.join(' ')}`;
    yield createResponseDeltaEvent('r1', { type: 'text', text: reply });
    yield createResponseDoneEvent('r1', [{ type: 'text', text: reply }]);
  }
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

async function waitFor(url: string, timeoutMs = 20_000): Promise<void> {
  const start = Date.now();
  let last = '';
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
      last = `status ${res.status}`;
    } catch (error) {
      last = (error as Error).message;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error(`${url} did not come up within ${timeoutMs} ms (${last})`);
}

interface HelperReady {
  ready: boolean;
  port: number;
  agent: string;
  delegate: { text?: string; error?: string };
}

/** Start the Python helper and read its one ready line. */
function startHelper(tsUrl: string): Promise<{ child: ChildProcess; ready: HelperReady; log: () => string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON as string, ['-u', HELPER, '--ts-url', tsUrl, '--ts-token', TS_TOKEN], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: PYTHON_DIR, PYTHONDONTWRITEBYTECODE: '1' },
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const log = () => `stdout:\n${stdout}\nstderr:\n${stderr}`;
    child.stdout?.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
      if (settled) return;
      for (const line of stdout.split('\n')) {
        if (!line.startsWith('{')) continue;
        try {
          const parsed = JSON.parse(line) as HelperReady;
          settled = true;
          if (!parsed.ready) reject(new Error(`helper not ready: ${line}\n${log()}`));
          else resolve({ child, ready: parsed, log });
          return;
        } catch {
          // not the ready line
        }
      }
    });
    child.stderr?.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on('exit', (code) => {
      if (!settled) {
        settled = true;
        reject(new Error(`helper exited with ${code} before it was ready\n${log()}`));
      }
    });
    setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error(`helper did not report ready within 60 s\n${log()}`));
      }
    }, 60_000);
  });
}

function makeContext(): Context {
  const store = new Map<string, unknown>();
  return {
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => {
      store.set(key, value);
    },
    delete: (key: string) => {
      store.delete(key);
    },
    signal: new AbortController().signal,
    auth: { authenticated: true, user_id: 'user-1' },
    payment: { token: 'parent-payment-token', agentToken: 'agent-scoped-token' },
    metadata: { authToken: 'caller-auth-token' },
  } as unknown as Context;
}

const suite = PYTHON ? describe : describe.skip;
const title = PYTHON
  ? 'delegate across the SDKs over A2A v1.0'
  : `delegate across the SDKs over A2A v1.0 (SKIPPED: no Python interpreter at ${VENV_PYTHON}; set WEBAGENTS_PYTHON)`;

suite(title, () => {
  let ts: ServeHandle | null = null;
  let tsUrl = '';
  let helper: { child: ChildProcess; ready: HelperReady; log: () => string } | null = null;
  let pyUrl = '';

  beforeAll(async () => {
    const port = await freePort();
    const publicUrl = `http://127.0.0.1:${port}`;
    tsUrl = `${publicUrl}${TS_BASE_PATH}`;
    const agent = new BaseAgent({
      name: 'ts-echo',
      description: 'TypeScript echo agent for the delegate-over-A2A cross-SDK test',
      skills: [new EchoLLM(), new A2ATransportSkill({ peers: {} })],
    });
    ts = await serve(agent, {
      port,
      hostname: '127.0.0.1',
      basePath: TS_BASE_PATH,
      publicUrl,
      keysDir: mkdtempSync(path.join(os.tmpdir(), 'a2adelegate-cross-sdk-ts-keys-')),
      heartbeat: false,
      logging: false,
    } as Parameters<typeof serve>[1]);
    await waitFor(`${tsUrl}/.well-known/agent-card.json`);
    helper = await startHelper(tsUrl);
    pyUrl = `http://127.0.0.1:${helper.ready.port}`;
    await waitFor(`${pyUrl}/.well-known/agent-card.json`);
  }, 120_000);

  afterAll(async () => {
    helper?.child.kill('SIGTERM');
    await ts?.close();
  });

  it("the Python agent's delegate reached the TypeScript agent over A2A, verified its card, and says nobody pays", () => {
    const call = helper!.ready.delegate;
    expect(call.error, helper!.log()).toBeUndefined();
    expect(call.text).toBe(`echo: hello from python\n${FIXTURE.unpaid_note.replace('{url}', tsUrl)}`);
  });

  it("the TypeScript agent's delegate reaches the Python agent over A2A, verifies its card, and says nobody pays", async () => {
    const nli = new NLISkill({ baseUrl: 'https://robutler.ai', transport: 'http', apiKey: 'platform-key-never-forwarded', timeout: 30_000 });
    const callerAgent = new BaseAgent({ name: 'ts-caller', skills: [new A2ATransportSkill({ peers: { [pyUrl]: { token: PY_TOKEN } } }), nli] });
    const result = (await nli.delegate({ agent: pyUrl, message: 'hello from typescript' }, makeContext())) as StructuredToolResult;
    expect(typeof result, JSON.stringify(result)).toBe('object');
    expect(result.text).toBe(`echo: hello from typescript\n${FIXTURE.unpaid_note.replace('{url}', pyUrl)}`);
    expect((result.data as { a2a: Record<string, unknown> }).a2a).toMatchObject({
      url: pyUrl,
      rpcUrl: `${pyUrl}/a2a`,
      state: 'TASK_STATE_COMPLETED',
      verified: true,
    });
    await callerAgent.cleanup();
  });
});
