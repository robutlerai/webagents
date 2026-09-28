/**
 * Three parity points pinned against the shared fixtures the Python suite
 * reads too (`python/tests/agents/skills/test_parity_finalfix.py`), 2026-09-27,
 * the final e2e re-run:
 *
 * 1. a bare `- mcp` entry, which reads the folder's mcp.json, resolves the
 *    owner's `${env:NAME}` and `${secret:NAME}` references, as a
 *    `- mcp: {...}` entry does (`mcp_tool/config_shapes.json`,
 *    `folder_mcp_json`); TypeScript always did, Python now does;
 * 2. the `- shell: {allowed_commands: [...]}` block widens the allow-list
 *    here too (`sandbox/srt.json`, `shell_block`): the skill read only
 *    `allowedCommands`, so the file's spelling added nothing;
 * 3. `list_mcp_servers` and the shell tool carry the same model-facing
 *    description in both SDKs (`config_shapes.json` `tools`, `srt.json`
 *    `shell_tool`).
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getTools } from '../../../src/core/decorators';
import { MCPSkill, type MCPServerConfig } from '../../../src/skills/mcp/skill';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { ShellSkill } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const MCP = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'mcp_tool/config_shapes.json'), 'utf8')) as {
  folder_mcp_json: { mcp_json: unknown; environment: Record<string, string>; server: string; resolved_env: Record<string, string> };
  tools: { list_mcp_servers: { description: string } };
};
const SRT = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sandbox/srt.json'), 'utf8')) as {
  shell_block: { case: { declared: Record<string, string[]>; allowed_includes: string[]; blocked_includes: string[] } };
  shell_tool: { name: { python: string; typescript: string }; description: string };
};
const tempDir = tempDirs();

describe('a bare mcp entry resolves the owner’s references from the folder’s mcp.json', () => {
  const saved: Record<string, string | undefined> = {};
  beforeEach(() => {
    for (const [name, value] of Object.entries(MCP.folder_mcp_json.environment)) {
      saved[name] = process.env[name];
      process.env[name] = value;
    }
  });
  afterEach(() => {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });

  it('builds the skill with the owner’s sources and expands the reference for the connection only', async () => {
    const dir = tempDir('wa-parity-mcp-');
    fs.writeFileSync(path.join(dir, 'mcp.json'), JSON.stringify(MCP.folder_mcp_json.mcp_json));
    const resolved = await resolveSkillsByName(['mcp'], { agentDir: dir });
    expect(resolved.unknown).toEqual([]);
    expect(resolved.failed).toEqual([]);
    const skill = resolved.skills[0] as unknown as {
      mcpConfig: { references?: { env?: unknown; secret?: unknown } };
      resolveReferences(name: string, config: MCPServerConfig): Promise<{ live: MCPServerConfig; values: string[] }>;
    };
    expect(skill).toBeInstanceOf(MCPSkill);
    expect(skill.mcpConfig.references).toBeDefined();
    expect(typeof skill.mcpConfig.references!.secret).toBe('function');
    const written = (MCP.folder_mcp_json.mcp_json as { mcpServers: Record<string, MCPServerConfig> }).mcpServers[MCP.folder_mcp_json.server];
    const { live, values } = await skill.resolveReferences(MCP.folder_mcp_json.server, written);
    expect(live.env).toEqual(MCP.folder_mcp_json.resolved_env);
    expect(values).toEqual(Object.values(MCP.folder_mcp_json.resolved_env));
    // The configuration as written keeps the reference, never the value.
    expect(written.env).toEqual((MCP.folder_mcp_json.mcp_json as { mcpServers: Record<string, { env: unknown }> }).mcpServers[MCP.folder_mcp_json.server].env);
  });
});

describe('the shell block’s spellings widen the lists', () => {
  it('reads allowed_commands and blocked_commands as the file writes them', () => {
    const { declared, allowed_includes, blocked_includes } = SRT.shell_block.case;
    const skill = new ShellSkill(declared as never) as unknown as { allowedCommands: Set<string>; blockedCommands: Set<string> };
    for (const name of allowed_includes) expect(skill.allowedCommands.has(name), name).toBe(true);
    for (const name of blocked_includes) expect(skill.blockedCommands.has(name), name).toBe(true);
  });
});

describe('the tool descriptions are the fixtures’', () => {
  it('list_mcp_servers', () => {
    const tools = getTools(new MCPSkill({}));
    const listed = [...tools.values()].find((tool) => tool.name === 'list_mcp_servers');
    expect(listed?.description).toBe(MCP.tools.list_mcp_servers.description);
  });

  it('the shell tool, owner-only', () => {
    const tools = getTools(new ShellSkill({}));
    // Registered under the method's own name here (the fixture records both SDKs' names).
    const run = [...tools.values()].find((tool) => tool.name === SRT.shell_tool.name.typescript);
    expect(run?.description).toBe(SRT.shell_tool.description);
    expect(run?.scopes).toEqual(['owner']);
  });
});
