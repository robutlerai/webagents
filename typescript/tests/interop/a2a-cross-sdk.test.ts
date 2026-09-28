/**
 * A2A v1.0 across the two SDKs (plan item 1.3, 2026-09-26): a TypeScript
 * agent and a Python agent, both served on loopback by this test, each call
 * the other over A2A v1.0 and each verifies the other's signed card against
 * the key set the other serves.
 *
 * The TypeScript agent runs in-process through `serve()` (the documented
 * server: Hono, the Ed25519 identity minted into a temp key directory, the
 * card signed with it). The Python agent is `python/tests/interop/serve_a2a_echo.py`,
 * started with the interpreter `WEBAGENTS_PYTHON` names, else the repo's
 * `python/.venv/bin/python`; it calls the TypeScript agent first, then
 * serves and prints its port. Without an interpreter the suite is SKIPPED
 * with the reason in its name, never passed quietly.
 *
 * THE PYTHON AGENT IS CALLED AT ITS ROOT (2026-09-26, the e2e run): the
 * Python server used to answer A2A only under `/<name>/`, with relative
 * interface URLs and a relative `jku`, so `POST /a2a` at the root was 405
 * and this SDK could not call a served Python agent. The helper now serves
 * its agent as `webagents serve` does (`root_agent` plus `RootMount`), so
 * the card at `/.well-known/agent-card.json` names `{origin}/a2a` and an
 * absolute `jku`, and the named path keeps working.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../src/core/agent';
import { Skill } from '../../src/core/skill';
import { handoff, tool } from '../../src/core/decorators';
import type { Context } from '../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../src/uamp/events';
import { serve, type ServeHandle } from '../../src/server/node';
import { A2ATransportSkill } from '../../src/skills/transport/a2a/skill';
import { callAgent, fetchAgentCard } from '../../src/skills/transport/a2a/a2a-client';
import { verifyAgentCard } from '../../src/skills/transport/a2a/card';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_DIR = path.resolve(HERE, '../../../python');
const HELPER = path.join(PYTHON_DIR, 'tests', 'interop', 'serve_a2a_echo.py');
const VENV_PYTHON = path.join(PYTHON_DIR, '.venv', 'bin', 'python');
const PYTHON = process.env.WEBAGENTS_PYTHON || (existsSync(VENV_PYTHON) ? VENV_PYTHON : null);

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

class ToolsSkill extends Skill {
  @tool({ name: 'public_lookup', description: 'Anyone may call this' })
  async publicLookup(_params: { q: string }, _ctx: Context) {
    return 'ok';
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
  ts_call: { reply?: string; state?: string; card_url?: string; verified?: { ok: boolean; kid?: string; alg?: string; checked: number; reason?: string }; error?: string };
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

const suite = PYTHON ? describe : describe.skip;
const title = PYTHON
  ? 'A2A v1.0 across the SDKs'
  : `A2A v1.0 across the SDKs (SKIPPED: no Python interpreter at ${VENV_PYTHON}; set WEBAGENTS_PYTHON)`;

suite(title, () => {
  let ts: ServeHandle | null = null;
  let tsUrl = '';
  let helper: { child: ChildProcess; ready: HelperReady; log: () => string } | null = null;
  /** The Python agent's ROOT, where `webagents serve` puts its one agent. */
  let pyUrl = '';
  /** The same agent under its name, which keeps working. */
  let pyNamed = '';

  beforeAll(async () => {
    const port = await freePort();
    const publicUrl = `http://127.0.0.1:${port}`;
    tsUrl = `${publicUrl}${TS_BASE_PATH}`;
    const agent = new BaseAgent({
      name: 'ts-echo',
      description: 'TypeScript echo agent for the A2A cross-SDK test',
      skills: [new EchoLLM(), new ToolsSkill(), new A2ATransportSkill({ peers: {} })],
    });
    ts = await serve(agent, {
      port,
      hostname: '127.0.0.1',
      basePath: TS_BASE_PATH,
      publicUrl,
      keysDir: mkdtempSync(path.join(os.tmpdir(), 'a2a-cross-sdk-ts-keys-')),
      heartbeat: false,
      logging: false,
    } as Parameters<typeof serve>[1]);
    await waitFor(`${tsUrl}/.well-known/agent-card.json`);
    helper = await startHelper(tsUrl);
    pyUrl = `http://127.0.0.1:${helper.ready.port}`;
    pyNamed = `${pyUrl}/${helper.ready.agent}`;
    await waitFor(`${pyUrl}/.well-known/agent-card.json`);
  }, 120_000);

  afterAll(async () => {
    helper?.child.kill('SIGTERM');
    await ts?.close();
  });

  it('the Python agent called the TypeScript agent over A2A v1.0 and verified its signed card', () => {
    const call = helper!.ready.ts_call;
    expect(call.error, helper!.log()).toBeUndefined();
    expect(call.reply).toBe('echo: hello from python');
    expect(call.state).toBe('TASK_STATE_COMPLETED');
    expect(call.card_url).toBe(`${tsUrl}/.well-known/agent-card.json`);
    expect(call.verified).toMatchObject({ ok: true, alg: 'EdDSA', checked: 1, kid: ts!.identity.kid });
  });

  it('the TypeScript agent calls the Python agent over A2A v1.0 and verifies its signed card', async () => {
    const result = await callAgent(pyUrl, 'hello from typescript', {
      token: PY_TOKEN,
      verifyCard: true,
      verify: { allowHttp: true },
      timeoutMs: 30_000,
    });
    expect(result.cardUrl).toBe(`${pyUrl}/.well-known/agent-card.json`);
    expect(result.rpcUrl).toBe(`${pyUrl}/a2a`);
    expect(result.task?.status.state).toBe('TASK_STATE_COMPLETED');
    expect(result.reply).toBe('echo: hello from typescript');
    expect(result.verified).toMatchObject({ ok: true, alg: 'EdDSA', checked: 1 });
    // The kid the Python card names is in the key set the Python server publishes.
    const jwks = (await (await fetch(`${pyUrl}/.well-known/jwks.json`)).json()) as { keys: Array<{ kid: string }> };
    expect(jwks.keys.map((k) => k.kid)).toContain(result.verified?.kid);
  });

  it("each card names the other's expectations: JSONRPC first, bearer declared, the public tool as a skill", async () => {
    for (const base of [tsUrl, pyUrl]) {
      const { card } = await fetchAgentCard(base);
      const interfaces = card.supportedInterfaces as Array<{ url: string; protocolBinding: string; protocolVersion: string }>;
      expect(interfaces[0]).toEqual({ url: `${base}/a2a`, protocolBinding: 'JSONRPC', protocolVersion: '1.0' });
      expect(Object.keys(card.securitySchemes as Record<string, unknown>)).toContain('bearer');
      expect((card.skills as Array<{ id: string }>).map((s) => s.id)).toContain('public_lookup');
      const verified = await verifyAgentCard(card, { cardUrl: `${base}/.well-known/agent-card.json`, allowHttp: true });
      expect(verified.ok, `${base}: ${verified.reason}`).toBe(true);
    }
  });

  it('a task is per caller across the SDKs: a bearer cannot read the other bearer’s task', async () => {
    const mine = await callAgent(pyUrl, 'private', { token: PY_TOKEN, timeoutMs: 30_000 });
    const res = await fetch(`${pyUrl}/a2a`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'A2A-Version': '1.0', authorization: 'Bearer someone-else' },
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'GetTask', params: { id: mine.task?.id } }),
    });
    const body = (await res.json()) as { error?: { code: number } };
    expect(body.error?.code).toBe(-32001);
  });

  it('the Python card at the root names absolute interface urls and an absolute jku, and the named path serves the same card', async () => {
    const { card } = await fetchAgentCard(pyUrl);
    const interfaces = card.supportedInterfaces as Array<{ url: string }>;
    expect(interfaces.map((i) => i.url)).toEqual([`${pyUrl}/a2a`, `${pyUrl}/a2a`]);
    const [signature] = card.signatures as Array<{ protected: string }>;
    const header = JSON.parse(Buffer.from(signature.protected, 'base64url').toString('utf8')) as { jku?: string };
    expect(header.jku).toBe(`${pyUrl}/.well-known/jwks.json`);
    // Under its name the agent still answers, with the same root-naming card.
    const named = await fetchAgentCard(pyNamed);
    expect((named.card.supportedInterfaces as Array<{ url: string }>)[0].url).toBe(`${pyUrl}/a2a`);
    const namedCall = await callAgent(pyNamed, 'named path', { token: PY_TOKEN, timeoutMs: 30_000 });
    expect(namedCall.reply).toBe('echo: named path');
  });

  it('the Python registration card at the root self-names the root', async () => {
    const card = (await (await fetch(`${pyUrl}/.well-known/agent.json`)).json()) as { client_id: string; url: string; jwks_uri: string };
    expect(card.client_id).toBe(`${pyUrl}/.well-known/agent.json`);
    expect(card.url).toBe(pyUrl);
    expect(card.jwks_uri).toBe(`${pyUrl}/.well-known/jwks.json`);
  });
});
