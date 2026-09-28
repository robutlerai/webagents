/**
 * A daemon-served agent signs its `cron:` webhooks (2026-09-27, the
 * a2a-delegate-webhooks lane): the daemon gives each agent it builds an
 * Ed25519 identity kept in the store `serve()` uses (`~/.webagents/keys`,
 * 0600; S-309 moved it there out of the agent folder), signs
 * as `{public URL}/agents/{name}`, serves that key set at
 * `/agents/{name}/.well-known/jwks.json`, and a webhook run by the runner
 * verifies with `verifyWebBotAuth` against the set the daemon serves. A
 * restart holds the same key; `webagents cron run` signs with it too; a
 * daemon with no public URL posts unsigned and says why; a key file that
 * cannot be used is said and never replaced. The paths and words are the
 * shared fixture's (`python/tests/fixtures/daemon/signedhooks.json`); the
 * Python daemon runs the same in `tests/daemon/test_daemon_signedhooks.py`.
 */

import { afterAll, afterEach, beforeEach, describe, expect, it } from 'vitest';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { AgentIdentity } from '../../../src/crypto/identity';
import { MemoryNonceStore, verifyWebBotAuth } from '../../../src/crypto/web-bot-auth-verify';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { daemonPublicUrl } from '../../../src/daemon/agent-identity';
import { buildScheduledAgent, cronRunAction } from '../../../src/cli/cron-action';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/signedhooks.json'), 'utf8')) as {
  store: { default: string };
  key_file: string;
  key_mode: string;
  dir_mode: string;
  issuer: string;
  public_url_env: string;
  jwks_route: string;
  jwks_cache_control: string;
  no_identity_status: number;
  signed_detail: string;
  public_url_cases: Array<{ name: string; host: string; port: number; env: string; public_url?: string; expect: string }>;
  agent_file: string;
};

const tempDir = tempDirs();
const PUBLIC_URL = 'https://agent.example';
const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_PUBLIC_URL', 'WEBAGENTS_KEYS_DIR', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY'];
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-signedhooks-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  // The daemon serves an agent only with a model its callers can run on
  // (S-327, 2026-09-28); these tests are about its key and its webhooks.
  process.env.OPENAI_API_KEY = 'sk-test-daemon-model';
});

afterEach(() => {
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

interface Received {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Buffer;
}

const servers: Server[] = [];
afterAll(() => {
  for (const server of servers) server.close();
});

/** A local webhook that keeps what it got and answers 200. */
async function webhookServer(): Promise<{ url: string; authority: string; received: Received[] }> {
  const received: Received[] = [];
  const server = createServer((req: IncomingMessage, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const headers: Record<string, string> = {};
      for (const [name, value] of Object.entries(req.headers)) headers[name] = Array.isArray(value) ? value.join(', ') : String(value ?? '');
      received.push({ method: req.method ?? '', url: req.url ?? '', headers, body: Buffer.concat(chunks) });
      res.statusCode = 200;
      res.end('{}');
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  servers.push(server);
  const { port } = server.address() as { port: number };
  return { url: `http://127.0.0.1:${port}/hook`, authority: `127.0.0.1:${port}`, received };
}

function project(url: string): string {
  const dir = tempDir('wa-signedhooks-project-');
  writeFileSync(path.join(dir, 'AGENT.md'), FIXTURE.agent_file.replace('{url}', url));
  return dir;
}

/** The key, in the store under the test's HOME (the fixture's `store.default`), never in `dir` (S-309). */
function keyFileOf(_dir: string, agent = 'reporter'): string {
  return path.join(process.env.HOME!, FIXTURE.store.default, FIXTURE.key_file.replace('{agent}', agent));
}

function daemonOn(dir: string, publicUrl?: string): WebAgentsDaemon {
  return new WebAgentsDaemon({ port: 0, hostname: '127.0.0.1', watchDir: dir, cron: false, healthChecks: false, ...(publicUrl ? { publicUrl } : {}) });
}

function appOf(daemon: WebAgentsDaemon): { fetch: (request: Request) => Promise<Response> } {
  return (daemon as unknown as { app: { fetch: (request: Request) => Promise<Response> } }).app;
}

function cannedTurn(agent: unknown, reply: string): void {
  (agent as { run: unknown }).run = async () => ({ content: reply });
}

async function verify(got: Received, authority: string, keys: Array<{ kid: string; x: string }>) {
  return verifyWebBotAuth(
    { method: got.method, target: '/hook', headers: got.headers, body: new Uint8Array(got.body) },
    {
      authorities: [authority],
      scheme: 'http',
      keySets: { get: async () => ({ ok: true, keys: keys.map((k) => ({ thumbprint: k.kid, x: k.x })), ttlS: 300 }) },
      nonces: new MemoryNonceStore(),
    },
  );
}

describe('the public URL the daemon signs under (the fixture’s cases)', () => {
  it.each(FIXTURE.public_url_cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    process.env[FIXTURE.public_url_env] = c.env;
    expect(daemonPublicUrl({ hostname: c.host, port: c.port, ...(c.public_url ? { publicUrl: c.public_url } : {}) })).toBe(c.expect);
  });
});

describe('a daemon-served agent signs its webhooks', () => {
  it('keeps a 0600 key in the store, none in the agent folder, serves its key set, and its webhook verifies against that set', async () => {
    const hook = await webhookServer();
    const dir = project(hook.url);
    const issuer = FIXTURE.issuer.replace('{public_url}', PUBLIC_URL).replace('{agent}', 'reporter');
    const daemon = daemonOn(dir, PUBLIC_URL);
    await daemon.discover();
    try {
      const entry = daemon.getRegistry().get('reporter');
      expect(entry?.agent).toBeTruthy();
      const identity = entry!.agent!.identity as AgentIdentity;
      expect(identity.issuer).toBe(issuer);

      // The key, where the fixture says, with the modes the store guarantees; and nothing in the folder (S-309).
      const keyFile = keyFileOf(dir);
      expect(existsSync(keyFile)).toBe(true);
      expect((statSync(keyFile).mode & 0o777).toString(8)).toBe(FIXTURE.key_mode);
      expect((statSync(path.dirname(keyFile)).mode & 0o777).toString(8)).toBe(FIXTURE.dir_mode);
      expect(existsSync(path.join(dir, '.webagents', 'keys'))).toBe(false);

      // The key set, where `serve()` serves one.
      const app = appOf(daemon);
      const res = await app.fetch(new Request(`http://127.0.0.1${FIXTURE.jwks_route.replace('{agent}', 'reporter')}`));
      expect(res.status).toBe(200);
      expect(res.headers.get('cache-control')).toBe(FIXTURE.jwks_cache_control);
      const jwks = (await res.json()) as { keys: Array<{ kid: string; x: string }> };
      expect(jwks.keys.map((k) => k.kid)).toEqual([identity.kid]);
      expect((await app.fetch(new Request(`http://127.0.0.1${FIXTURE.jwks_route.replace('{agent}', 'nobody')}`))).status).toBe(FIXTURE.no_identity_status);

      // The runner's own delivery, on a canned turn: signed as the daemon's identity.
      cannedTurn(entry!.agent, 'Nothing happened.');
      const record = await daemon.getScheduleRunner().runNow('reporter', 'hook');
      expect(record.outcome).toBe('delivered');
      expect(record.detail).toBe(FIXTURE.signed_detail.replace('{url}', hook.url).replace('{issuer}', issuer));
      expect(hook.received).toHaveLength(1);
      const outcome = await verify(hook.received[0], hook.authority, jwks.keys);
      expect(outcome.ok ? 'ok' : outcome.refusal).toBe('ok');
      if (outcome.ok) {
        expect(outcome.agent.principal).toBe(issuer);
        // Signature-Agent names the daemon's own key-set route under the issuer.
        expect(outcome.agent.identifier).toBe(`${issuer}${FIXTURE.jwks_route.replace('{agent}', 'reporter').slice('/agents/reporter'.length)}`);
        expect(outcome.agent.thumbprints).toEqual([identity.kid]);
      }
    } finally {
      daemon.stop();
    }

    // A restart holds the same key: the file is read, never regenerated, and the key set is the same.
    const kid = JSON.parse(readFileSync(keyFileOf(dir), 'utf8')).x as string;
    expect(typeof kid).toBe('string');
    const again = daemonOn(dir, PUBLIC_URL);
    await again.discover();
    try {
      const identity = again.getRegistry().get('reporter')!.agent!.identity as AgentIdentity;
      expect(identity.getJwks().keys[0].x).toBe(kid);
      const res = await appOf(again).fetch(new Request(`http://127.0.0.1${FIXTURE.jwks_route.replace('{agent}', 'reporter')}`));
      expect(((await res.json()) as { keys: Array<{ x: string }> }).keys[0].x).toBe(kid);
    } finally {
      again.stop();
    }
  });

  it('a restart serves the key the first daemon made, and `cron run` signs with it too', async () => {
    const hook = await webhookServer();
    const dir = project(hook.url);
    const issuer = FIXTURE.issuer.replace('{public_url}', PUBLIC_URL).replace('{agent}', 'reporter');
    const first = daemonOn(dir, PUBLIC_URL);
    await first.discover();
    const kid = (first.getRegistry().get('reporter')!.agent!.identity as AgentIdentity).kid;
    first.stop();

    const second = daemonOn(dir, PUBLIC_URL);
    await second.discover();
    expect((second.getRegistry().get('reporter')!.agent!.identity as AgentIdentity).kid).toBe(kid);
    second.stop();

    // `webagents cron run`, with the public URL the daemon would read from the environment.
    process.env[FIXTURE.public_url_env] = PUBLIC_URL;
    const lines: string[] = [];
    const code = await cronRunAction(
      'reporter',
      'hook',
      { watch: dir },
      {
        buildAgent: async (definition) => {
          const agent = await buildScheduledAgent(definition, (line) => lines.push(line));
          cannedTurn(agent, 'Report.');
          return agent;
        },
        log: (line) => lines.push(line),
        error: (line) => lines.push(line),
      },
    );
    expect(code, lines.join('\n')).toBe(0);
    expect(lines).toContain(`reporter/hook: delivered (${FIXTURE.signed_detail.replace('{url}', hook.url).replace('{issuer}', issuer)})`);
    expect(hook.received).toHaveLength(1);
    const keyid = /keyid="([^"]+)"/.exec(hook.received[0].headers['signature-input'])?.[1];
    expect(keyid).toBe(kid);
  });

  it('with no public URL the webhook goes out unsigned, and the record says why', async () => {
    const hook = await webhookServer();
    const dir = project(hook.url);
    const daemon = daemonOn(dir);
    await daemon.discover();
    try {
      const entry = daemon.getRegistry().get('reporter')!;
      expect((entry.agent!.identity as AgentIdentity).issuer).toBe(`http://127.0.0.1:0/agents/reporter`);
      cannedTurn(entry.agent, 'Nothing happened.');
      const record = await daemon.getScheduleRunner().runNow('reporter', 'hook');
      expect(record.outcome).toBe('delivered');
      expect(record.detail.startsWith(`webhook ${hook.url} (unsigned: this agent's key could not sign: `)).toBe(true);
      expect(record.detail).toContain('loopback');
      expect(hook.received[0].headers.signature).toBeUndefined();
    } finally {
      daemon.stop();
    }
  });

  it('a key file that cannot be used is said, never replaced, and the agent serves unsigned', async () => {
    const hook = await webhookServer();
    const dir = project(hook.url);
    const keyFile = keyFileOf(dir);
    mkdirSync(path.dirname(keyFile), { recursive: true });
    writeFileSync(keyFile, 'not a key');
    const errors: string[] = [];
    const original = console.error;
    console.error = (...args: unknown[]) => errors.push(args.map(String).join(' '));
    const daemon = daemonOn(dir, PUBLIC_URL);
    try {
      await daemon.discover();
      const entry = daemon.getRegistry().get('reporter')!;
      expect(entry.agent).toBeTruthy();
      expect(entry.agent!.identity).toBeUndefined();
      expect(errors.some((line) => line.includes(keyFile) && line.includes('served without a signing identity'))).toBe(true);
      expect(readFileSync(keyFile, 'utf8')).toBe('not a key');
      expect((await appOf(daemon).fetch(new Request(`http://127.0.0.1${FIXTURE.jwks_route.replace('{agent}', 'reporter')}`))).status).toBe(FIXTURE.no_identity_status);
      cannedTurn(entry.agent, 'Nothing happened.');
      const record = await daemon.getScheduleRunner().runNow('reporter', 'hook');
      expect(record.detail).toBe(`webhook ${hook.url} (unsigned: this agent holds no signing key)`);
    } finally {
      console.error = original;
      daemon.stop();
    }
  });
});
