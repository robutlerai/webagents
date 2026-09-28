/**
 * S-286 addendum (2026-09-26): `enabledTools` and `toolPolicies: block` are
 * applied at discovery in both SDKs. TypeScript always did; this file pins
 * that against the shared fixture `tool_policies.filtering` in
 * `python/tests/fixtures/mcp_tool/config_shapes.json`, which the Python
 * suite now runs too (`tests/agents/skills/test_mcp_tool_filtering_s286_e2efix.py`),
 * so the two SDKs register the same tools from the same entry.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { MCPSkill } from '../../../src/skills/mcp/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FILTERING = (JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/config_shapes.json'), 'utf8')) as {
  tool_policies: { filtering: { cases: { name: string; entry: Record<string, unknown>; listed: string[]; registered: string[] }[] } };
}).tool_policies.filtering;

/** A session that lists `names` and nothing else. */
function sessionListing(names: string[]) {
  return {
    async listTools() {
      return { tools: names.map((name) => ({ name, description: `tool ${name}`, inputSchema: { type: 'object', properties: {} } })) };
    },
    async listResources() {
      return { resources: [] };
    },
    async listPrompts() {
      return { prompts: [] };
    },
  };
}

describe('S-286 addendum: what a server lists is filtered by its entry before anything is registered', () => {
  it.each(FILTERING.cases)('$name', async ({ entry, listed, registered }) => {
    const skill = new MCPSkill({}) as unknown as {
      serverConfigs: Map<string, unknown>;
      _discoverCapabilities(name: string, session: unknown): Promise<void>;
      tools: { name: string }[];
    };
    skill.serverConfigs.set('srv', entry);
    await skill._discoverCapabilities('srv', sessionListing(listed));
    // The skill's own `list_mcp_servers` is always there; the server's tools are the `srv__` ones.
    expect(skill.tools.map((t) => t.name).filter((n) => n.startsWith('srv__')).sort()).toEqual([...registered].sort());
  });
});
