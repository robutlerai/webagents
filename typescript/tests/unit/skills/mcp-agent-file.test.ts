/**
 * The `mcp` entry of an agent file (plan item 0.3, 2026-09-26): the shapes it
 * accepts and the servers each resolves to, the one name a server's tool
 * gets, and the error when the SDK cannot load, all pinned by
 * `python/tests/fixtures/mcp_tool/config_shapes.json`, which the Python suite
 * runs too (`tests/agents/skills/test_local_mcp_agent_file.py`). Then the
 * client itself, against the fixture echo server (`tests/fixtures/
 * mcp-echo-server.mjs`, `echo_server.json`) over stdio and over Streamable
 * HTTP, and the access block reaching the tools it registers on start.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { accessSkillFor, applyAccessTools } from '../../../src/access/install';
import { BaseAgent } from '../../../src/core/agent';
import type { ISkill } from '../../../src/core/types';
import {
  MCP_TOOL_SEPARATOR,
  mcpSdkMissing,
  mcpServersFromConfig,
  qualifiedToolName,
} from '../../../src/skills/mcp/config';
import { MCPSkill, loadMcpSdk, resetMcpSdkForTests } from '../../../src/skills/mcp/skill';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool');
const SHAPES = JSON.parse(readFileSync(path.join(FIXTURES, 'config_shapes.json'), 'utf8')) as {
  naming: { separator: string; cases: { server: string; tool: string; name: string }[] };
  shapes: { name: string; config: unknown; servers: unknown[]; rejected: unknown[] }[];
  missing_sdk: { starts_with: string; typescript: string };
};
const ECHO = JSON.parse(readFileSync(path.join(FIXTURES, 'echo_server.json'), 'utf8')) as {
  server_name: string;
  http_path: string;
  tools: { name: string; description: string; inputSchema: Record<string, unknown> }[];
  qualified: string[];
  calls: { tool: string; arguments: Record<string, unknown>; text: string }[];
};
const ECHO_SERVER = path.resolve(HERE, '../../fixtures/mcp-echo-server.mjs');
const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };

const tempDir = tempDirs();

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

/** The echo server over Streamable HTTP, once it says it listens. */
async function echoOverHttp(): Promise<{ url: string; child: ChildProcess }> {
  const port = await freePort();
  const child = spawn(process.execPath, [ECHO_SERVER, '--http', String(port)], { stdio: ['ignore', 'ignore', 'pipe'] });
  await new Promise<void>((resolve, reject) => {
    let said = '';
    child.stderr!.on('data', (chunk: Buffer) => {
      said += chunk.toString();
      if (said.includes('echo server on')) resolve();
    });
    child.on('exit', (code) => reject(new Error(`echo server exited with ${code}: ${said}`)));
  });
  return { url: `http://127.0.0.1:${port}${ECHO.http_path}`, child };
}

const stdioServer = { command: process.execPath, args: [ECHO_SERVER] };

describe('the shapes an mcp entry accepts (config_shapes.json)', () => {
  it.each(SHAPES.shapes)('$name', ({ config, servers, rejected }) => {
    const resolution = mcpServersFromConfig(config);
    expect(resolution.servers).toEqual(servers);
    expect(resolution.rejected).toEqual(rejected);
  });

  it("keeps the file's extra keys on the server's config", () => {
    const { configs } = mcpServersFromConfig({ s: { command: 'x', pricing: { creditsPerCall: 1 } } });
    expect(configs.s).toMatchObject({ pricing: { creditsPerCall: 1 }, transport: 'stdio' });
  });
});

describe('one name per tool', () => {
  it('is <server>__<tool>, always', () => {
    expect(MCP_TOOL_SEPARATOR).toBe(SHAPES.naming.separator);
    for (const c of SHAPES.naming.cases) expect(qualifiedToolName(c.server, c.tool)).toBe(c.name);
  });
});

describe('the SDK', () => {
  afterEach(() => resetMcpSdkForTests());

  it('loads, with both remote transports', async () => {
    const sdk = await loadMcpSdk();
    expect(typeof sdk.Client).toBe('function');
    expect(typeof sdk.StreamableHTTPClientTransport).toBe('function');
    expect(typeof sdk.SSEClientTransport).toBe('function');
  });

  it('says what failed when it cannot load, in the fixture words', async () => {
    resetMcpSdkForTests();
    const failing = async () => {
      throw new Error("Cannot find package '@modelcontextprotocol/sdk'");
    };
    await expect(loadMcpSdk({ client: failing })).rejects.toThrow(SHAPES.missing_sdk.starts_with);
    expect(mcpSdkMissing('<reason>')).toBe(SHAPES.missing_sdk.typescript);
  });
});

describe('the agent-file loader', () => {
  it('resolves `mcp` with its config, in either shape', async () => {
    for (const config of [{ demo: { command: 'x' } }, { mcpServers: { demo: { command: 'x' } } }]) {
      const { byName, unknown, failed } = await resolveSkillsByName([{ mcp: config }]);
      expect(unknown).toEqual([]);
      expect(failed).toEqual([]);
      expect(byName.get('mcp')).toBeInstanceOf(MCPSkill);
    }
  });

  it('a bare `mcp` reads mcp.json next to the agent file', async () => {
    const dir = tempDir('wa-mcp-json-');
    writeFileSync(path.join(dir, 'mcp.json'), JSON.stringify({ mcpServers: { [ECHO.server_name]: stdioServer } }));
    const { skills, failed } = await resolveSkillsByName(['mcp'], { agentDir: dir });
    expect(failed).toEqual([]);
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: skills as never });
    await agent.initialize();
    try {
      const names = agent.getToolDefinitions().map((d) => d.function.name).filter((n) => n.startsWith(ECHO.server_name));
      expect(names.sort()).toEqual(ECHO.qualified);
    } finally {
      await agent.cleanup();
    }
  });
});

describe('the client, against the fixture echo server', () => {
  async function connected(servers: Record<string, unknown>): Promise<{ agent: BaseAgent; skill: MCPSkill }> {
    const skill = new MCPSkill({ mcp: servers as never });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    return { agent, skill };
  }

  function echoTools(agent: BaseAgent) {
    return agent
      .getToolDefinitions()
      .filter((d) => d.function.name.startsWith(ECHO.server_name + MCP_TOOL_SEPARATOR))
      .sort((a, b) => a.function.name.localeCompare(b.function.name));
  }

  it('lists its tools by qualified name, with their schemas, and calls them, over stdio', async () => {
    const { agent } = await connected({ [ECHO.server_name]: stdioServer });
    try {
      const tools = echoTools(agent);
      expect(tools.map((d) => d.function.name)).toEqual(ECHO.qualified);
      for (const tool of ECHO.tools) {
        const listed = tools.find((d) => d.function.name === qualifiedToolName(ECHO.server_name, tool.name))!;
        expect(listed.function.description).toBe(tool.description);
        expect(listed.function.parameters).toEqual(tool.inputSchema);
      }
      for (const call of ECHO.calls) {
        expect(await agent.runTool(call.tool, call.arguments, { auth: OWNER })).toBe(call.text);
      }
    } finally {
      await agent.cleanup();
    }
  });

  it('over Streamable HTTP', async () => {
    const { url, child } = await echoOverHttp();
    try {
      const { agent } = await connected({ [ECHO.server_name]: { url, transport: 'http' } });
      try {
        expect(echoTools(agent).map((d) => d.function.name)).toEqual(ECHO.qualified);
        const call = ECHO.calls[0];
        expect(await agent.runTool(call.tool, call.arguments, { auth: OWNER })).toBe(call.text);
      } finally {
        await agent.cleanup();
      }
    } finally {
      child.kill();
    }
  });

  it('a server that fails to connect is reported and the others still load', async () => {
    const { agent } = await connected({
      [ECHO.server_name]: stdioServer,
      broken: { url: 'http://127.0.0.1:9/mcp', transport: 'http' },
    });
    try {
      expect(echoTools(agent).map((d) => d.function.name)).toEqual(ECHO.qualified);
    } finally {
      await agent.cleanup();
    }
  });

  it("access.tools naming the skill restricts the tools it registers on start", async () => {
    const skill = new MCPSkill({ mcp: { [ECHO.server_name]: stdioServer } as never });
    const { skill: access, policy } = accessSkillFor(
      { groups: { friends: ['user:@alice'] }, tools: { friends: ['mcp'] } },
      undefined,
    );
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill, access] });
    applyAccessTools(policy, new Map<string, ISkill>([['mcp', skill as unknown as ISkill]]));
    await agent.initialize();
    try {
      const names = (defs: { function: { name: string } }[]) => defs.map((d) => d.function.name);
      expect(names(await agent.listTools({ auth: OWNER }))).toEqual(expect.arrayContaining(ECHO.qualified));
      expect(names(await agent.listTools({ auth: { authenticated: false, scopes: ['group:everyone'] } }))).not.toContain(
        ECHO.qualified[0],
      );
      expect(names(await agent.listTools({ auth: { authenticated: true, scopes: ['group:friends'] } }))).toEqual(
        expect.arrayContaining(ECHO.qualified),
      );
      await expect(
        agent.runTool(ECHO.calls[0].tool, ECHO.calls[0].arguments, { auth: { authenticated: false, scopes: ['group:everyone'] } }),
      ).rejects.toThrow('Insufficient permissions');
    } finally {
      await agent.cleanup();
    }
  });
});
