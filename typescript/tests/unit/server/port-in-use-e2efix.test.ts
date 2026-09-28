/**
 * A port already in use is one sentence and exit 1 (2026-09-26, the
 * new-developer e2e run): `webagents serve --port <busy>` died with an
 * unhandled 'error' event, a stack trace and `Node.js v24.7.0`. Pinned by
 * `python/tests/fixtures/cli/listen.json`, which the Python suite runs too
 * (`tests/cli/test_port_in_use_e2efix.py`): the three listeners reject with
 * `PortInUseError` carrying the sentence, and the CLI itself, spawned against
 * a port this test holds, prints the sentence alone.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { PORT_IN_USE, PortInUseError, listenError } from '../../../src/server/listen-error';
import { serveMcpHttp } from '../../../src/server/mcp';
import { serve } from '../../../src/server/node';
import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LISTEN = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/listen.json'), 'utf8')) as {
  port_in_use: string;
  exit: number;
  never_printed: string[];
};
const tempDir = tempDirs();
const sentence = (host: string, port: number) => LISTEN.port_in_use.replace('{port}', String(port)).replace('{host}', host);

let holder: net.Server;
let busy = 0;

beforeAll(async () => {
  holder = net.createServer();
  await new Promise<void>((resolve) => holder.listen(0, '127.0.0.1', () => resolve()));
  busy = (holder.address() as { port: number }).port;
});

afterAll(async () => {
  await new Promise<void>((resolve) => holder.close(() => resolve()));
});

describe('the sentence', () => {
  it('is the fixture’s, and only EADDRINUSE becomes it', () => {
    expect(PORT_IN_USE).toBe(LISTEN.port_in_use);
    const error = new PortInUseError('127.0.0.1', 4242);
    expect(error.message).toBe(sentence('127.0.0.1', 4242));
    expect(error.name).toBe('PortInUseError');
    expect(listenError(Object.assign(new Error('x'), { code: 'EADDRINUSE' }), 'h', 1)).toBeInstanceOf(PortInUseError);
    const other = Object.assign(new Error('y'), { code: 'EACCES' });
    expect(listenError(other, 'h', 1)).toBe(other);
  });
});

describe('every listener answers a busy port with the sentence', () => {
  it('serve()', async () => {
    const agent = new BaseAgent({ name: 'a', instructions: 'x', skills: [] });
    await expect(
      serve(agent, { port: busy, hostname: '127.0.0.1', keysDir: mkdtempSync(path.join(os.tmpdir(), 'wa-port-keys-')), heartbeat: false, logging: false } as Parameters<typeof serve>[1]),
    ).rejects.toMatchObject({ name: 'PortInUseError', message: sentence('127.0.0.1', busy) });
  });

  it('the daemon', async () => {
    const daemon = new WebAgentsDaemon({ port: busy, hostname: '127.0.0.1', watch: false, cron: false, healthChecks: false });
    await expect(daemon.start()).rejects.toMatchObject({ name: 'PortInUseError', message: sentence('127.0.0.1', busy) });
  });

  it('mcp serve --http', async () => {
    const agent = new BaseAgent({ name: 'a', instructions: 'x', skills: [] });
    await expect(serveMcpHttp(agent, { port: busy, hostname: '127.0.0.1' })).rejects.toMatchObject({ name: 'PortInUseError', message: sentence('127.0.0.1', busy) });
  });
});

describe('the CLI', () => {
  it('prints the sentence alone and exits 1, no stack trace', () => {
    const dir = tempDir('wa-port-project-');
    writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n');
    const env = { ...process.env, HOME: tempDir('wa-port-home-'), WEBAGENTS_SECRETS_BACKEND: 'file', OPENAI_API_KEY: 'sk-dummy', ROBUTLER_API_URL: 'http://127.0.0.1:9' } as Record<string, string | undefined>;
    for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_DEBUG']) delete env[name];
    const result = spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, 'serve', '--port', String(busy)], { cwd: dir, env: env as NodeJS.ProcessEnv, encoding: 'utf8', timeout: 60_000 });
    expect(result.status).toBe(LISTEN.exit);
    expect(result.stderr).toContain(sentence('127.0.0.1', busy));
    for (const never of LISTEN.never_printed) {
      expect(result.stderr, never).not.toContain(never);
      expect(result.stdout, never).not.toContain(never);
    }
    expect(result.stderr).not.toMatch(/^\s+at /m);
  }, 90_000);
});
