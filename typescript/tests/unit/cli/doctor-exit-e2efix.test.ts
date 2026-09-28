/**
 * `webagents doctor` exits once its report is out (2026-09-26, the new-developer
 * e2e run's HIGH bug): `runChecks` built a chat, initialised it and never
 * cleaned it up, so a connected stdio MCP server's child process kept the
 * event loop alive and doctor sat there until it was killed. It also let the
 * skills print raw lines (`[MCPSkill] Server "x" failed to connect: ...`)
 * above the report; the report carries every one of those findings, so the
 * console is held while the agent starts. The Python doctor is pinned the
 * same way by `tests/cli/test_doctor_exit_e2efix.py`.
 *
 * The CLI is spawned as a person would run it, against the probe MCP server
 * the secrets tests use (`tests/fixtures/mcp-probe-server-mcpsecrets.mjs`),
 * with a scratch HOME and the file secrets backend, never the keychain.
 */

import { describe, expect, it } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROBE_SERVER = path.resolve(HERE, '../../fixtures/mcp-probe-server-mcpsecrets.mjs');
const PROBE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/probe_server_mcpsecrets.json'), 'utf8'),
) as { server_name: string };

const tempDir = tempDirs();
/** Generous for a cold tsx start, far below the harness timeout that caught the hang. */
const EXIT_WITHIN_MS = 60_000;

function project(mcpEntry: Record<string, unknown>): { dir: string; home: string } {
  const dir = tempDir('wa-doctor-exit-project-');
  const home = tempDir('wa-doctor-exit-home-');
  fs.writeFileSync(
    path.join(dir, 'AGENT.md'),
    `---\nname: a\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - mcp:\n      ${PROBE.server_name}: ${JSON.stringify(mcpEntry)}\n---\nBody\n`,
  );
  return { dir, home };
}

/** `webagents doctor` in `dir`, with the outcome or a timeout that means the hang is back. */
function doctor(dir: string, home: string): Promise<{ code: number | null; stdout: string; stderr: string; timedOut: boolean }> {
  const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', OPENAI_API_KEY: 'sk-doctor-exit-dummy', ROBUTLER_API_URL: 'http://127.0.0.1:9' } as Record<string, string | undefined>;
  for (const name of ['WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_DEBUG']) delete env[name];
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, 'doctor'], { cwd: dir, env: env as NodeJS.ProcessEnv, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk: Buffer) => void (stdout += chunk.toString()));
    child.stderr.on('data', (chunk: Buffer) => void (stderr += chunk.toString()));
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      resolve({ code: null, stdout, stderr, timedOut: true });
    }, EXIT_WITHIN_MS);
    child.on('exit', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr, timedOut: false });
    });
  });
}

describe('webagents doctor exits, with its report first', () => {
  it('exits once a stdio MCP server has connected, and the report is the first thing on stdout', async () => {
    const { dir, home } = project({ command: process.execPath, args: [PROBE_SERVER], env: { PROBE_REGION: 'eu-west-9' } });
    const result = await doctor(dir, home);
    expect(result.timedOut, `doctor did not exit within ${EXIT_WITHIN_MS} ms\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`).toBe(false);
    expect(result.stdout.split('\n')[0]).toBe('Checks');
    expect(result.stdout).toContain(`1 server connected: ${PROBE.server_name}`);
    expect(result.stdout).not.toContain('[MCPSkill]');
    expect(result.stderr).not.toContain('[MCPSkill]');
  }, EXIT_WITHIN_MS + 10_000);

  it('a server that cannot start is in the mcp check, not a raw line above the report', async () => {
    const { dir, home } = project({ command: '/nonexistent/mcp-server' });
    const result = await doctor(dir, home);
    expect(result.timedOut, `doctor did not exit\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`).toBe(false);
    expect(result.code).toBe(1);
    expect(result.stdout.split('\n')[0]).toBe('Checks');
    expect(result.stdout).toMatch(/✗ mcp\s+probe: /);
    expect(result.stdout).not.toContain('[MCPSkill]');
    expect(result.stderr).not.toContain('[MCPSkill]');
  }, EXIT_WITHIN_MS + 10_000);
});
