/**
 * `webagents mcp serve` (plan item 1.8, 2026-09-26): the command's words, its
 * port check, and the real thing: a real MCP client starts the CLI over stdio
 * and connects to it over Streamable HTTP, and sees what
 * `python/tests/fixtures/mcp_tool/serve.json` says a client sees (the Python
 * CLI runs the same cases in `tests/cli/test_mcp_serve.py`). The CLI runs
 * through this repo's tsx with HOME in a temporary folder
 * (`tests/helpers/cli.ts`), so nothing stored on this machine reaches it.
 */

import { describe, expect, it } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { httpPort, mcpServeAction } from '../../../src/cli/mcp-serve-action';
import type { IAgent } from '../../../src/core/types';
import { CLI_ARGS, CLI_SOURCE, cliPrerequisite, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SERVE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/serve.json'), 'utf8')) as {
  cli: {
    group: { name: string; description: string };
    serve: { description: string; arguments: [string, string, string][]; options: [string, string, null][] };
  };
  http_path: string;
  agent_file: string;
  tools: { owner: string[]; everyone: string[] };
  call: { name: string; arguments: Record<string, unknown>; contains: string };
  refusals: { not_open: { code: number; message: string }; no_credential: { status: number } };
  startup: { stdio: string; http: string };
};

const tempDir = tempDirs();

function fixtureProject(): { dir: string; home: string } {
  const dir = tempDir('wa-mcp-cli-');
  writeFileSync(path.join(dir, 'AGENT.md'), SERVE.agent_file);
  return { dir, home: tempDir('wa-mcp-home-') };
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

async function mcpClient() {
  const [{ Client }, { StdioClientTransport }, { StreamableHTTPClientTransport }, { McpError }] = await Promise.all([
    import('@modelcontextprotocol/sdk/client/index.js' as string),
    import('@modelcontextprotocol/sdk/client/stdio.js' as string),
    import('@modelcontextprotocol/sdk/client/streamableHttp.js' as string),
    import('@modelcontextprotocol/sdk/types.js' as string),
  ]);
  return { Client, StdioClientTransport, StreamableHTTPClientTransport, McpError };
}

describe('the words, as the fixture and the Python CLI have them', () => {
  it('declares the group, the command, its argument and its options', () => {
    const cli = readFileSync(CLI_SOURCE, 'utf-8');
    const start = cli.indexOf("program.command('mcp')");
    expect(start, 'the mcp group moved; fix this scan').toBeGreaterThan(-1);
    const block = cli.slice(start, cli.indexOf('.action(', start));
    expect(block).toContain(`.description('${SERVE.cli.group.description}')`);
    expect(block).toContain(`.command('${SERVE.cli.group.name === 'mcp' ? 'serve' : ''}')`);
    expect(block).toContain(`.description("${SERVE.cli.serve.description}")`);
    for (const [term, help, fallback] of SERVE.cli.serve.arguments) {
      expect(block).toContain(`.argument('${term}', '${help}', '${fallback}')`);
    }
    for (const [term, help] of SERVE.cli.serve.options) expect(block).toContain(`.option('${term}', '${help}')`);
  });

  it('--http takes a port, and says so otherwise', () => {
    expect(httpPort('3001')).toBe(3001);
    expect(httpPort('0')).toBe(0);
    for (const bad of ['x', '-1', '65536', '', '80a']) expect(() => httpPort(bad)).toThrow(RangeError);
  });

  it('serves over stdio by default and over HTTP with --http, the agent built once', async () => {
    const agent = { name: 'stub' } as unknown as IAgent;
    const seen: string[] = [];
    await mcpServeAction('.', {}, {
      createAgent: () => agent,
      serveStdio: async (a) => void seen.push(`stdio:${a.name}`),
      serveHttp: async () => void seen.push('http'),
    });
    await mcpServeAction('.', { http: '4040', host: '0.0.0.0' }, {
      createAgent: () => agent,
      serveStdio: async () => void seen.push('stdio'),
      serveHttp: async (a, c) => void seen.push(`http:${a.name}:${c.port}:${c.hostname}`),
    });
    expect(seen).toEqual(['stdio:stub', 'http:stub:4040:0.0.0.0']);
  });
});

// Skipped, with the reason in the title, where the CLI cannot be spawned and
// driven over MCP (tests/helpers/cli.ts, 2026-09-26): from the portal root
// these ran with the portal's tsconfig and failed for that reason alone.
const prerequisite = cliPrerequisite();

describe.skipIf(prerequisite !== null)(
  prerequisite ? `a real MCP client and the CLI (SKIPPED: ${prerequisite})` : 'a real MCP client and the CLI',
  () => {
  it('over stdio: the owner lists every tool and calls one', async () => {
    const { dir, home } = fixtureProject();
    const { Client, StdioClientTransport } = await mcpClient();
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [...CLI_ARGS, 'mcp', 'serve', dir],
      env: { ...process.env, HOME: home },
      stderr: 'pipe',
    });
    let said = '';
    transport.stderr?.on('data', (chunk: Buffer) => {
      said += chunk.toString();
    });
    const client = new Client({ name: 'test', version: '0' });
    await client.connect(transport);
    try {
      const { tools } = await client.listTools();
      expect(tools.map((t: { name: string }) => t.name)).toEqual(SERVE.tools.owner);
      const result = await client.callTool({ name: SERVE.call.name, arguments: SERVE.call.arguments });
      expect(result.isError).toBeFalsy();
      expect(JSON.stringify(result.content)).toContain(SERVE.call.contains);
    } finally {
      await client.close();
    }
    expect(said).toContain(SERVE.startup.stdio);
  }, 90_000);

  it('over Streamable HTTP: a bearer nothing verifies is `everyone`, and no credential is refused', async () => {
    const { dir, home } = fixtureProject();
    const port = await freePort();
    const child: ChildProcess = spawn(process.execPath, [...CLI_ARGS, 'mcp', 'serve', dir, '--http', String(port)], {
      env: { ...process.env, HOME: home },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    const listening = new Promise<void>((resolve, reject) => {
      child.stdout!.on('data', (chunk: Buffer) => {
        out += chunk.toString();
        if (out.includes('MCP on')) resolve();
      });
      child.stderr!.on('data', (chunk: Buffer) => {
        out += chunk.toString();
      });
      child.on('exit', (code) => reject(new Error(`the CLI exited with ${code}: ${out}`)));
    });
    try {
      await listening;
      expect(out).toContain(SERVE.startup.http.replace('<port>', String(port)));
      const url = new URL(`http://127.0.0.1:${port}${SERVE.http_path}`);
      const { Client, StreamableHTTPClientTransport, McpError } = await mcpClient();

      const client = new Client({ name: 'test', version: '0' });
      await client.connect(new StreamableHTTPClientTransport(url, { requestInit: { headers: { authorization: 'Bearer anything' } } }));
      try {
        const { tools } = await client.listTools();
        expect(tools.map((t: { name: string }) => t.name)).toEqual(SERVE.tools.everyone);
        const refused = await client.callTool({ name: SERVE.call.name, arguments: SERVE.call.arguments }).catch((e: unknown) => e);
        expect(refused).toBeInstanceOf(McpError);
        expect((refused as { code: number }).code).toBe(SERVE.refusals.not_open.code);
        expect((refused as Error).message).toContain(SERVE.refusals.not_open.message);
      } finally {
        await client.close();
      }

      const anonymous = new Client({ name: 'test', version: '0' });
      const failure = await anonymous.connect(new StreamableHTTPClientTransport(url)).catch((e: unknown) => e);
      expect((failure as { code?: number }).code).toBe(SERVE.refusals.no_credential.status);
      expect(String((failure as Error).message)).toContain('unauthorized');
    } finally {
      child.kill();
    }
  }, 90_000);
});
