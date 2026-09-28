/**
 * `${secret:NAME}` and `${env:NAME}` in an MCP server's env, headers and url,
 * and the environment a stdio server gets (S-292, 2026-09-26), pinned by
 * `secret_refs` in `python/tests/fixtures/mcp_tool/config_shapes.json`, which
 * the Python suite runs too (`tests/agents/skills/test_mcp_secret_refs_mcpsecrets.py`).
 *
 * Then the client itself, against the probe server (`tests/fixtures/
 * mcp-probe-server-mcpsecrets.mjs`): a stdio server sees only the MCP SDK's
 * default variables plus its own resolved `env`, never the agent process's
 * `OPENAI_API_KEY`; a remote server's Authorization header built from a
 * `${secret:...}` reaches it; and the value shows up in no console line, no
 * report and no error along the way. The secret store is the CLI's file
 * fallback under a throwaway HOME, never the keychain.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { cliCommand } from '../../../src/cli/config-store';
import { mcpServersFromConfig } from '../../../src/skills/mcp/config';
import { MCPSkill, ownerReferenceSources } from '../../../src/skills/mcp/skill';
import {
  REFERENCE_NAME,
  REFERENCE_SENTENCES,
  SECRET_LOOKING_PATTERNS,
  SECRET_MASK,
  SecretReferenceError,
  atConnectSentence,
  expandReferences,
  literalWarning,
  looksLikeSecret,
  maskText,
  maskUrl,
  maskValue,
  secretsSetHint,
  suggestedSecretName,
} from '../../../src/skills/secrets/references';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const REFS = (JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/config_shapes.json'), 'utf8')) as { secret_refs: SecretRefsFixture }).secret_refs;
const PROBE = JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/probe_server_mcpsecrets.json'), 'utf8')) as {
  server_name: string;
  http_path: string;
  qualified: string[];
  unset: string;
  none: string;
};
const CLI_COMMAND = JSON.parse(readFileSync(path.join(FIXTURES, 'cli/cli_command.json'), 'utf8')) as {
  cases: { profile: string | null; rest: string; expected: string }[];
};
const PROBE_SERVER = path.resolve(HERE, '../../fixtures/mcp-probe-server-mcpsecrets.mjs');
const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };

interface SecretRefsFixture {
  mask: string;
  name_pattern: string;
  sentences: Record<string, string>;
  hints: { profile: string | null; name: string; hint: string }[];
  expansion: {
    name: string;
    value: string;
    secrets?: Record<string, string>;
    env?: Record<string, string>;
    expected?: string;
    values?: string[];
    error?: string;
    missing_secret?: string;
  }[];
  command_line: { name: string; config: unknown; servers: unknown[]; rejected: unknown[] }[];
  literal_warnings: {
    patterns: string[];
    suggested_names: { server: string; key: string; suggested: string }[];
    cases: { name: string; config: unknown; warnings: unknown[] }[];
  };
  masking: {
    values: { value: string; masked: string }[];
    urls: { url: string; masked: string }[];
    text: { text: string; values: string[]; masked: string };
  };
  at_connect: { server: string; field: string; key: string | null; sentence: string; composed: string }[];
}

const tempDir = tempDirs();

// Obvious dummies. Nothing here is or resembles a real credential.
const TOKEN = 'dummy-probe-token-not-a-real-credential-0123456789';
const LEAK = 'sk-dummy-agent-process-key-that-must-not-leak';

describe('the grammar matches the fixture', () => {
  it('sentences, mask, name pattern and secret-looking patterns', () => {
    expect(REFERENCE_SENTENCES).toEqual(REFS.sentences);
    expect(SECRET_MASK).toBe(REFS.mask);
    expect(REFERENCE_NAME.source).toBe(REFS.name_pattern);
    expect([...SECRET_LOOKING_PATTERNS]).toEqual(REFS.literal_warnings.patterns);
  });

  it.each(REFS.hints)('the hint under profile $profile', ({ profile, name, hint }) => {
    const saved = process.env.WEBAGENTS_PROFILE;
    if (profile) process.env.WEBAGENTS_PROFILE = profile;
    else delete process.env.WEBAGENTS_PROFILE;
    try {
      expect(secretsSetHint(name)).toBe(hint);
    } finally {
      if (saved === undefined) delete process.env.WEBAGENTS_PROFILE;
      else process.env.WEBAGENTS_PROFILE = saved;
    }
  });

  it('names a command exactly as the CLI does, under every profile the cli_command fixture tries', () => {
    const saved = process.env.WEBAGENTS_PROFILE;
    try {
      for (const c of CLI_COMMAND.cases) {
        if (c.profile) process.env.WEBAGENTS_PROFILE = c.profile;
        else delete process.env.WEBAGENTS_PROFILE;
        expect(secretsSetHint('X')).toBe(cliCommand('secrets set X'));
      }
    } finally {
      if (saved === undefined) delete process.env.WEBAGENTS_PROFILE;
      else process.env.WEBAGENTS_PROFILE = saved;
    }
  });

  describe('expansion', () => {
    beforeEach(() => {
      delete process.env.WEBAGENTS_PROFILE;
    });

    it.each(REFS.expansion)('$name', async (c) => {
      const lookup = { secret: (name: string) => c.secrets?.[name] ?? null, env: c.env ?? {} };
      if (c.error !== undefined) {
        const failure = await expandReferences(c.value, lookup).then(() => undefined, (e: unknown) => e as SecretReferenceError);
        expect(failure).toBeInstanceOf(SecretReferenceError);
        expect(failure!.message).toBe(c.error);
        if (c.missing_secret) expect(failure!.missingSecret).toBe(c.missing_secret);
        return;
      }
      expect(await expandReferences(c.value, lookup)).toEqual({ text: c.expected, values: c.values });
    });
  });

  it.each(REFS.command_line)('$name', ({ config, servers, rejected }) => {
    const resolution = mcpServersFromConfig(config);
    expect(resolution.servers).toEqual(servers);
    expect(resolution.rejected).toEqual(rejected);
  });

  describe('literal warnings', () => {
    it.each(REFS.literal_warnings.suggested_names)('suggests $suggested for $server $key', ({ server, key, suggested }) => {
      expect(suggestedSecretName(server, key)).toBe(suggested);
    });

    it.each(REFS.literal_warnings.cases)('$name', ({ config, warnings }) => {
      expect(mcpServersFromConfig(config).warnings).toEqual(warnings);
    });

    it('a reference is never a literal, whatever it looks like', () => {
      expect(looksLikeSecret('Bearer ${secret:A_TOKEN_NAME_LONG_ENOUGH}')).toBe(false);
      expect(literalWarning('gh', 'headers', 'Authorization')).toContain('${secret:GH_AUTHORIZATION}');
    });
  });

  describe('masking', () => {
    it.each(REFS.masking.values)('$value', ({ value, masked }) => {
      expect(maskValue(value)).toBe(masked);
    });

    it.each(REFS.masking.urls)('$url', ({ url, masked }) => {
      expect(maskUrl(url)).toBe(masked);
    });

    it('an error text', () => {
      const { text, values, masked } = REFS.masking.text;
      expect(maskText(text, values)).toBe(masked);
    });
  });

  it.each(REFS.at_connect)('$composed', ({ server, field, key, sentence, composed }) => {
    expect(atConnectSentence(server, field, key ?? undefined, sentence)).toBe(composed);
  });
});

// ---------------------------------------------------------------------------
// The client, against the probe server
// ---------------------------------------------------------------------------

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address() as { port: number };
      probe.close(() => resolve(port));
    });
  });
}

async function probeOverHttp(): Promise<{ url: string; child: ChildProcess }> {
  const port = await freePort();
  const child = spawn(process.execPath, [PROBE_SERVER, '--http', String(port)], { stdio: ['ignore', 'ignore', 'pipe'] });
  await new Promise<void>((resolve, reject) => {
    let said = '';
    child.stderr!.on('data', (chunk: Buffer) => {
      said += chunk.toString();
      if (said.includes('probe server on')) resolve();
    });
    child.on('exit', (code) => reject(new Error(`probe server exited with ${code}: ${said}`)));
  });
  return { url: `http://127.0.0.1:${port}${PROBE.http_path}`, child };
}

describe('the client', () => {
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR', 'OPENAI_API_KEY', 'PROBE_FROM_ENV'];
  const saved: Record<string, string | undefined> = {};
  let printed: string[] = [];

  beforeEach(async () => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-mcpsecrets-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    printed = [];
    for (const method of ['log', 'warn', 'error', 'info', 'debug'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        printed.push(args.map((a) => (a instanceof Error ? `${a.message} ${a.stack ?? ''}` : String(a))).join(' '));
      });
    }
    // The CLI's own store, file backend, under the throwaway HOME: what `webagents secrets set PROBE_TOKEN` would write.
    const { storeProviderKey } = await import('../../../src/cli/provider-keys');
    expect(await storeProviderKey('PROBE_TOKEN', TOKEN)).toBe('file');
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });

  async function connected(servers: Record<string, unknown>): Promise<{ agent: BaseAgent; skill: MCPSkill }> {
    // As the agent-file loader builds it (S-295): with the owner's sources.
    // A skill built without them resolves nothing (mcp-reference-gate-s295-e2efix.test.ts).
    const skill = new MCPSkill({ mcp: servers as never, references: ownerReferenceSources() });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    return { agent, skill };
  }

  function nothingPrintedHolds(value: string): void {
    for (const line of printed) expect(line, line).not.toContain(value);
  }

  it('a stdio server gets the default environment plus only its own resolved env, never the agent process variables', async () => {
    process.env.OPENAI_API_KEY = LEAK;
    process.env.PROBE_FROM_ENV = 'from-the-environment';
    const { agent, skill } = await connected({
      [PROBE.server_name]: {
        command: process.execPath,
        args: [PROBE_SERVER],
        env: { PROBE_DECLARED: '${secret:PROBE_TOKEN}', PROBE_COPIED: '${env:PROBE_FROM_ENV}', PROBE_PLAIN: 'plain' },
      },
    });
    try {
      const env = async (name: string) => agent.runTool(`${PROBE.server_name}__env`, { name }, { auth: OWNER });
      expect(await env('OPENAI_API_KEY')).toBe(PROBE.unset);
      expect(await env('PROBE_DECLARED')).toBe(TOKEN);
      expect(await env('PROBE_COPIED')).toBe('from-the-environment');
      expect(await env('PROBE_PLAIN')).toBe('plain');
      expect(await env('PATH')).not.toBe(PROBE.unset);
      // The report shows the references as written and masks the literal; no value anywhere.
      const report = skill.serverReport();
      expect(report).toHaveLength(1);
      expect(report[0]).toMatchObject({
        name: PROBE.server_name,
        transport: 'stdio',
        connected: true,
        tools: PROBE.qualified,
        env: { PROBE_DECLARED: '${secret:PROBE_TOKEN}', PROBE_COPIED: '${env:PROBE_FROM_ENV}', PROBE_PLAIN: SECRET_MASK },
        missingSecrets: [],
        warnings: [],
      });
      expect(JSON.stringify(report)).not.toContain(TOKEN);
      nothingPrintedHolds(TOKEN);
      nothingPrintedHolds(LEAK);
    } finally {
      await agent.cleanup();
    }
  });

  it("a remote server's Authorization header built from ${secret:...} reaches it, and the value is shown nowhere", async () => {
    const { url, child } = await probeOverHttp();
    try {
      const { agent, skill } = await connected({
        [PROBE.server_name]: { url, transport: 'http', headers: { Authorization: 'Bearer ${secret:PROBE_TOKEN}' } },
      });
      try {
        expect(await agent.runTool(`${PROBE.server_name}__authorization`, {}, { auth: OWNER })).toBe(`Bearer ${TOKEN}`);
        const report = skill.serverReport();
        expect(report[0]).toMatchObject({ connected: true, headers: { Authorization: 'Bearer ${secret:PROBE_TOKEN}' }, url });
        expect(JSON.stringify(report)).not.toContain(TOKEN);
        nothingPrintedHolds(TOKEN);
      } finally {
        await agent.cleanup();
      }
    } finally {
      child.kill();
    }
  });

  it('a reference in the url is resolved too, and a url with a literal query is masked in the report', async () => {
    const { url, child } = await probeOverHttp();
    try {
      process.env.PROBE_FROM_ENV = url;
      const { agent, skill } = await connected({
        [PROBE.server_name]: { url: '${env:PROBE_FROM_ENV}?token=${secret:PROBE_TOKEN}', transport: 'http' },
        literal: { url: `${url}?token=literal-not-a-secret`, transport: 'http' },
      });
      try {
        expect(skill.serverReport().map((r) => [r.name, r.connected, r.url])).toEqual([
          [PROBE.server_name, true, '${env:PROBE_FROM_ENV}?token=${secret:PROBE_TOKEN}'],
          ['literal', true, `${url}?token=${SECRET_MASK}`],
        ]);
        nothingPrintedHolds(TOKEN);
      } finally {
        await agent.cleanup();
      }
    } finally {
      child.kill();
    }
  });

  it('an unresolvable reference fails that server with the sentence, names the reference, and the others still load', async () => {
    const { agent, skill } = await connected({
      broken: { command: process.execPath, args: [PROBE_SERVER], env: { X: '${secret:NOT_STORED}' } },
      [PROBE.server_name]: { command: process.execPath, args: [PROBE_SERVER] },
    });
    try {
      const rows = Object.fromEntries(skill.serverReport().map((r) => [r.name, r]));
      expect(rows[PROBE.server_name].connected).toBe(true);
      expect(rows.broken).toMatchObject({
        connected: false,
        tools: [],
        error: 'env X of MCP server "broken": ${secret:NOT_STORED} is not set: store it with `webagents secrets set NOT_STORED`',
        missingSecrets: ['NOT_STORED'],
      });
      expect(printed.some((l) => l.includes('Server "broken" failed to connect: env X of MCP server "broken": ${secret:NOT_STORED} is not set'))).toBe(true);
    } finally {
      await agent.cleanup();
    }
  });

  it('an unset variable and a reference in args are each said by name', async () => {
    const { agent, skill } = await connected({
      novar: { url: 'http://127.0.0.1:9/mcp', transport: 'http', headers: { 'X-Key': '${env:PROBE_NOT_EXPORTED}' } },
      onargs: { command: process.execPath, args: [PROBE_SERVER, '${secret:PROBE_TOKEN}'] },
    });
    try {
      const rows = Object.fromEntries(skill.serverReport().map((r) => [r.name, r]));
      expect(rows.novar.error).toBe('headers X-Key of MCP server "novar": ${env:PROBE_NOT_EXPORTED} is not set in the environment');
      expect(rows.onargs.rejected).toBe(REFS.sentences.commandLine.replace('{field}', 'args'));
      expect(rows.onargs.transport).toBe('unknown');
      nothingPrintedHolds(TOKEN);
    } finally {
      await agent.cleanup();
    }
  });

  it('a literal that looks like a key is warned about once at load, and the server still connects', async () => {
    const { agent, skill } = await connected({
      [PROBE.server_name]: { command: process.execPath, args: [PROBE_SERVER], env: { GH: 'ghp_dummy0123456789abcdefghijklmnop' } },
    });
    try {
      const expected = literalWarning(PROBE.server_name, 'env', 'GH');
      expect(printed.filter((l) => l.includes(expected))).toHaveLength(1);
      expect(skill.serverReport()[0]).toMatchObject({ connected: true, warnings: [expected], env: { GH: SECRET_MASK } });
      nothingPrintedHolds('ghp_dummy0123456789abcdefghijklmnop');
    } finally {
      await agent.cleanup();
    }
  });

  it('a builder may hand in its own sources: a secret reader of its own, with the environment it chooses', async () => {
    const skill = new MCPSkill({
      mcp: { [PROBE.server_name]: { command: process.execPath, args: [PROBE_SERVER], env: { X: '${secret:FROM_HOST}' } } } as never,
      references: { env: {}, secret: async (name) => (name === 'FROM_HOST' ? 'host-supplied-dummy' : null) },
    });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    try {
      expect(await agent.runTool(`${PROBE.server_name}__env`, { name: 'X' }, { auth: OWNER })).toBe('host-supplied-dummy');
    } finally {
      await agent.cleanup();
    }
  });
});
