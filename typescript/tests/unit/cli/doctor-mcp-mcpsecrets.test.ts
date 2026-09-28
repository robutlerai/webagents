/**
 * `webagents doctor`'s `mcp` check (S-292, 2026-09-26): the check computed
 * from the MCP skill's report, pinned by `doctor` in
 * `python/tests/fixtures/cli/secrets.json` (the Python suite runs the same
 * cases, `tests/cli/test_doctor_mcp_mcpsecrets.py`), and the check as doctor
 * reaches it through a real agent file: a server whose `${secret:NAME}` is
 * not stored fails with the `webagents secrets set` command, and a stored one
 * connects. Scratch HOME, file backend, never the keychain.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MCP_CHECK_WORDS, mcpCheck, runChecks } from '../../../src/cli/doctor';
import type { McpServerReportRow } from '../../../src/skills/mcp/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const SECRETS = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'cli/secrets.json'), 'utf8')) as {
  doctor: {
    words: Record<string, string>;
    cases: { name: string; report: Array<Record<string, unknown> & { missing_secrets: string[] }>; check: Record<string, string> }[];
  };
};
const PROBE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'mcp_tool/probe_server_mcpsecrets.json'), 'utf8')) as { server_name: string };
const PROBE_SERVER = path.resolve(HERE, '../../fixtures/mcp-probe-server-mcpsecrets.mjs');

const tempDir = tempDirs();

describe('the check from a report', () => {
  it('the words match the fixture', () => {
    expect(MCP_CHECK_WORDS).toEqual(SECRETS.doctor.words);
  });

  it.each(SECRETS.doctor.cases)('$name', ({ report, check }) => {
    delete process.env.WEBAGENTS_PROFILE;
    // The fixture writes the rows in the Python key style.
    const rows = report.map(({ missing_secrets, missing_env, ...rest }) => ({
      ...rest,
      missingSecrets: missing_secrets,
      ...(missing_env ? { missingEnv: missing_env } : {}),
    })) as McpServerReportRow[];
    expect(mcpCheck(rows)).toEqual(check);
  });
});

describe('through a real agent file', () => {
  const ISOLATED = [
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
    'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL',
  ];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();
  let project = '';

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-doctor-mcp-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
    project = tempDir('wa-doctor-mcp-project-');
    process.chdir(project);
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    vi.restoreAllMocks();
  });

  function agentWithProbe(env: Record<string, string>): void {
    const entry = { command: process.execPath, args: [PROBE_SERVER], env };
    fs.writeFileSync(path.join(project, 'AGENT.md'), `---\nname: a\nskills:\n  - mcp:\n      ${PROBE.server_name}: ${JSON.stringify(entry)}\n---\nBody\n`);
  }

  async function mcp() {
    return (await runChecks()).find((c) => c.name === 'mcp')!;
  }

  it('the ten checks, in order', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: a\n---\nBody\n');
    // `keychain` after `keys` since 2026-09-27 (the keychain-ux lane).
    expect((await runChecks()).map((c) => c.name)).toEqual(['runtime', 'agent', 'model', 'sign-in', 'keys', 'keychain', 'sandbox', 'skills', 'mcp', 'config']);
  });

  it('an unstored secret fails the check with the command that stores it', async () => {
    agentWithProbe({ X: '${secret:NOT_STORED}' });
    const check = await mcp();
    expect(check.status).toBe('fail');
    expect(check.detail).toBe(`${PROBE.server_name}: env X of MCP server "${PROBE.server_name}": \${secret:NOT_STORED} is not set: store it with \`webagents secrets set NOT_STORED\``);
    expect(check.fix).toBe('`webagents secrets set NOT_STORED`');
  });

  it('a stored secret connects and the check names the server', async () => {
    const { storeProviderKey } = await import('../../../src/cli/provider-keys');
    await storeProviderKey('STORED_FOR_DOCTOR', 'dummy-not-a-real-credential');
    agentWithProbe({ X: '${secret:STORED_FOR_DOCTOR}' });
    const check = await mcp();
    expect(check.status).toBe('ok');
    expect(check.detail).toBe(`1 server connected: ${PROBE.server_name}`);
  });

  it('a literal that looks like a key is a warning, and the value is not repeated', async () => {
    agentWithProbe({ GH: 'ghp_dummy0123456789abcdefghijklmnop' });
    const check = await mcp();
    expect(check.status).toBe('warn');
    expect(check.detail).not.toContain('ghp_dummy');
    const suggested = `${PROBE.server_name.toUpperCase()}_GH`;
    expect(check.fix).toBe(`\${secret:${suggested}} in the agent file, then \`webagents secrets set ${suggested}\``);
  });

  it('no mcp entry is "not used"', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: a\n---\nBody\n');
    const check = await mcp();
    expect(check.status).toBe('ok');
    expect(check.detail).toBe(MCP_CHECK_WORDS.notUsed);
  });
});
