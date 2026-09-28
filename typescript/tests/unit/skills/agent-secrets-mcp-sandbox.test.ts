/**
 * An MCP server entry that asks for a sandbox this SDK cannot provide does
 * not start (S-313, 2026-09-27, the agent-secrets lane): refused at load, by
 * name, with the shared fixture's sentence
 * (`python/tests/fixtures/mcp_tool/config_shapes.json`, `sandbox_key`), and
 * the other servers still load. The Python suite runs its side in
 * `tests/agents/skills/test_agent_secrets_mcp_sandbox.py`.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SANDBOX_UNAVAILABLE, asksForSandbox, mcpProblemLines, mcpServersFromConfig } from '../../../src/skills/mcp/config';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SHAPES = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/config_shapes.json'), 'utf8')) as {
  sandbox_key: {
    key: string;
    refusal: { typescript: string; python: string };
    config: Record<string, Record<string, unknown>>;
    typescript: { loads: string[]; rejected: string[] };
  };
};
const CASE = SHAPES.sandbox_key;

describe('an MCP entry asking for a sandbox (S-313)', () => {
  it('the sentence is the fixture’s', () => {
    expect(SANDBOX_UNAVAILABLE).toBe(CASE.refusal.typescript);
  });

  it('asks when the key holds anything but false and null', () => {
    expect(asksForSandbox({ sandbox: true })).toBe(true);
    expect(asksForSandbox({ sandbox: 'docker' })).toBe(true);
    expect(asksForSandbox({ sandbox: 1 })).toBe(true);
    expect(asksForSandbox({ sandbox: false })).toBe(false);
    expect(asksForSandbox({ sandbox: null })).toBe(false);
    expect(asksForSandbox({})).toBe(false);
  });

  it('is refused by name at load, and the others still load', () => {
    const resolution = mcpServersFromConfig(CASE.config);
    expect(resolution.servers.map((s) => s.name)).toEqual(CASE.typescript.loads);
    expect(resolution.rejected).toEqual(CASE.typescript.rejected.map((name) => ({ name, reason: CASE.refusal.typescript })));
    for (const name of CASE.typescript.rejected) expect(resolution.configs[name]).toBeUndefined();
    // The chat and `-p` say it the way they say every refused entry.
    const lines = mcpProblemLines(resolution.rejected.map((r) => ({ name: r.name, rejected: r.reason, warnings: [] })));
    expect(lines).toEqual(CASE.typescript.rejected.map((name) => `[MCPSkill] Server "${name}" ${CASE.refusal.typescript}; skipping it.`));
  });

  it('the mcpServers wrapper is read the same way', () => {
    const resolution = mcpServersFromConfig({ mcpServers: CASE.config });
    expect(resolution.rejected.map((r) => r.name)).toEqual(CASE.typescript.rejected);
  });
});
