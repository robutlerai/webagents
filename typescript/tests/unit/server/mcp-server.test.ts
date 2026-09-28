/**
 * The MCP server (`server/mcp.ts`, plan item 1.8, 2026-09-26): what a real
 * MCP client sees of the fixture agent, in memory as the owner and over
 * Streamable HTTP as a caller the access block places in `everyone`, pinned
 * by `python/tests/fixtures/mcp_tool/serve.json` and run the same way in
 * Python (`tests/cli/test_mcp_serve.py`). The tool definitions are the todo
 * and web fixtures, byte for byte. `tests/unit/cli/mcp-serve.test.ts` runs
 * the same client against the CLI.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createServedAgent } from '../../../src/cli/serve-action';
import type { IAgent } from '../../../src/core/types';
import {
  LOCAL_OWNER_AUTH,
  MCP_HTTP_PATH,
  callToolResult,
  createMcpProtocolServer,
  loadMcpServerSdk,
  mcpToolsOf,
  serveMcpHttp,
  toolNotOpen,
  unknownTool,
  type McpHttpHandle,
} from '../../../src/server/mcp';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const SERVE = JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/serve.json'), 'utf8')) as {
  http_path: string;
  agent_file: string;
  tools: { owner: string[]; everyone: string[] };
  definitions: Record<string, 'todo_tool' | 'web_tool'>;
  call: { name: string; arguments: Record<string, unknown>; contains: string };
  refusals: { not_open: { code: number; message: string }; unknown: { code: number; message: string }; no_credential: { status: number } };
};

type Listed = { name: string; description: string; inputSchema: Record<string, unknown> };

/** The definitions the fixture says each tool has, as MCP lists them. */
function expectedTools(names: string[]): Listed[] {
  const todo = JSON.parse(readFileSync(path.join(FIXTURES, 'todo_tool/definitions.json'), 'utf8')).definitions as {
    function: { name: string; description: string; parameters: Record<string, unknown> };
  }[];
  const web = JSON.parse(readFileSync(path.join(FIXTURES, 'web_tool/definition.json'), 'utf8')) as {
    function: { name: string; description: string; parameters: Record<string, unknown> };
  };
  const all = new Map<string, Listed>();
  for (const { function: f } of [...todo, web]) all.set(f.name, { name: f.name, description: f.description, inputSchema: f.parameters });
  return names.map((name) => {
    const found = all.get(name);
    if (!found) throw new Error(`${name} is in neither tool fixture`);
    return found;
  });
}

function listed(tools: Array<Record<string, unknown>>): Listed[] {
  return tools.map((t) => ({ name: t.name as string, description: t.description as string, inputSchema: t.inputSchema as Record<string, unknown> }));
}

const tempDir = tempDirs();

async function fixtureAgent(): Promise<IAgent> {
  const dir = tempDir('wa-mcp-serve-');
  writeFileSync(path.join(dir, 'AGENT.md'), SERVE.agent_file);
  return createServedAgent(dir);
}

async function mcpClient(): Promise<{ Client: any; InMemoryTransport: any; StreamableHTTPClientTransport: any; McpError: any }> {
  const [{ Client }, { InMemoryTransport }, { StreamableHTTPClientTransport }, { McpError }] = await Promise.all([
    import('@modelcontextprotocol/sdk/client/index.js' as string),
    import('@modelcontextprotocol/sdk/inMemory.js' as string),
    import('@modelcontextprotocol/sdk/client/streamableHttp.js' as string),
    import('@modelcontextprotocol/sdk/types.js' as string),
  ]);
  return { Client, InMemoryTransport, StreamableHTTPClientTransport, McpError };
}

describe('the fixture pins what both SDKs list', () => {
  it('names the owner and everyone lists from the tool fixtures', () => {
    expect(expectedTools(SERVE.tools.owner).map((t) => t.name)).toEqual([...SERVE.tools.owner].sort());
    expect(SERVE.tools.everyone.every((name) => SERVE.tools.owner.includes(name))).toBe(true);
    for (const [name, fixture] of Object.entries(SERVE.definitions)) expect(fixture).toMatch(/^(todo|web)_tool$/), name;
  });

  it('serves the schema as declared, sorted, and `object` when a tool declares no type', () => {
    const tools = mcpToolsOf([
      { type: 'function', function: { name: 'b', description: 'B', parameters: { properties: { x: { type: 'string' } } } } },
      { type: 'function', function: { name: 'a', parameters: { type: 'object', properties: {} } } },
    ]);
    expect(tools).toEqual([
      { name: 'a', description: '', inputSchema: { type: 'object', properties: {} } },
      { name: 'b', description: 'B', inputSchema: { type: 'object', properties: { x: { type: 'string' } } } },
    ]);
  });

  it('turns what a tool returned into MCP content', () => {
    expect(callToolResult('hi')).toEqual({ content: [{ type: 'text', text: 'hi' }] });
    expect(callToolResult(undefined)).toEqual({ content: [{ type: 'text', text: '' }] });
    expect(callToolResult({ a: 1 })).toEqual({ content: [{ type: 'text', text: '{"a":1}' }] });
    expect(callToolResult({ text: 'see', content_items: [{ type: 'image', image: 'data:image/png;base64,AAAA' }] })).toEqual({
      content: [
        { type: 'text', text: 'see' },
        { type: 'image', mimeType: 'image/png', data: 'AAAA' },
      ],
    });
    expect(callToolResult({ error: 'rejected_by_owner', tool_name: 't' }).isError).toBe(true);
  });
});

describe('in memory, as the owner', () => {
  it('lists every tool with the fixture definitions, calls one, and refuses an unknown name', async () => {
    const agent = await fixtureAgent();
    await agent.initialize();
    const sdk = await loadMcpServerSdk();
    const { Client, InMemoryTransport, McpError } = await mcpClient();
    const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
    const server = createMcpProtocolServer(agent, sdk, LOCAL_OWNER_AUTH);
    await server.connect(serverSide);
    const client = new Client({ name: 'test', version: '0' });
    await client.connect(clientSide);
    try {
      const { tools } = await client.listTools();
      expect(listed(tools)).toEqual(expectedTools(SERVE.tools.owner));

      const result = await client.callTool({ name: SERVE.call.name, arguments: SERVE.call.arguments });
      expect(result.isError).toBeFalsy();
      expect(JSON.stringify(result.content)).toContain(SERVE.call.contains);

      const refused = await client.callTool({ name: 'no_such_tool', arguments: {} }).catch((e: unknown) => e);
      expect(refused).toBeInstanceOf(McpError);
      expect((refused as { code: number }).code).toBe(SERVE.refusals.unknown.code);
      expect((refused as Error).message).toContain(SERVE.refusals.unknown.message);
      expect(unknownTool('no_such_tool')).toBe(SERVE.refusals.unknown.message);
    } finally {
      await client.close();
      await server.close();
      await agent.cleanup?.();
    }
  });
});

describe('over Streamable HTTP', () => {
  let handle: McpHttpHandle | undefined;
  let agent: IAgent | undefined;
  afterEach(async () => {
    await handle?.close();
    await agent?.cleanup?.();
    handle = undefined;
    agent = undefined;
  });

  it("a bearer nothing verifies is `everyone`: sees its tools, is refused the friends' one; no credential is 401", async () => {
    agent = await fixtureAgent();
    handle = await serveMcpHttp(agent, { port: 0 });
    expect(handle.url).toBe(`http://127.0.0.1:${handle.port}${MCP_HTTP_PATH}`);
    expect(MCP_HTTP_PATH).toBe(SERVE.http_path);
    const { Client, StreamableHTTPClientTransport, McpError } = await mcpClient();

    const client = new Client({ name: 'test', version: '0' });
    await client.connect(
      new StreamableHTTPClientTransport(new URL(handle.url), { requestInit: { headers: { authorization: 'Bearer anything' } } }),
    );
    try {
      const { tools } = await client.listTools();
      expect(listed(tools)).toEqual(expectedTools(SERVE.tools.everyone));

      const refused = await client.callTool({ name: SERVE.call.name, arguments: SERVE.call.arguments }).catch((e: unknown) => e);
      expect(refused).toBeInstanceOf(McpError);
      expect((refused as { code: number }).code).toBe(SERVE.refusals.not_open.code);
      expect((refused as Error).message).toContain(SERVE.refusals.not_open.message);
      expect(toolNotOpen(SERVE.call.name)).toBe(SERVE.refusals.not_open.message);
    } finally {
      await client.close();
    }

    const anonymous = new Client({ name: 'test', version: '0' });
    const failure = await anonymous.connect(new StreamableHTTPClientTransport(new URL(handle.url))).catch((e: unknown) => e);
    expect((failure as { code?: number }).code).toBe(SERVE.refusals.no_credential.status);
    expect(String((failure as Error).message)).toContain('unauthorized');

    // The floor answers before anything reads the body: a plain request too.
    const response = await fetch(handle.url, { method: 'POST', body: '{' });
    expect(response.status).toBe(SERVE.refusals.no_credential.status);
  });
});
