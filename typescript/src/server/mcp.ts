/**
 * An agent's tools, served to an MCP client (plan item 1.8, 2026-09-26).
 *
 * `webagents mcp serve` puts the agent behind the Model Context Protocol, so
 * Claude Code, Codex, OpenCode and any other MCP client call its tools the way
 * they call any MCP server's: over stdio (the client starts this process), or
 * over stateless Streamable HTTP at `/mcp` with `--http <port>`. The Python
 * twin is `python/webagents/server/mcp_server.py`; what a client sees from
 * either is pinned by `python/tests/fixtures/mcp_tool/serve.json`.
 *
 * WHO IS CALLING decides what is listed and what runs, as it does for a chat
 * turn:
 *   - over stdio the caller is the person at this terminal, the agent's
 *     owner, as in the local chat (`LOCAL_OWNER` in `cli/app.ts`);
 *   - over HTTP the rules are `serve()`'s: a request with nothing to
 *     authenticate is refused by the credential floor before its body is read
 *     (`credential-floor.ts`); the agent's auth skills and access block then
 *     say who it is (`BaseAgent.identifyCaller` on the endpoint gate's
 *     context), and a refusal they raise keeps its status and body; a caller
 *     they place in no group is `everyone`, and a bearer nothing verifies
 *     names no one.
 * The tools listed are the ones `getToolDefinitions` gives a run bound to that
 * caller (`BaseAgent.listTools`), and a call goes through `runTool`: the scope
 * check, the posture and confirmation gates, and the `before_tool` and
 * `after_tool` hooks, which is where pricing and payment live. A tool the
 * caller may not use is refused by name with a JSON-RPC invalid-params error;
 * a tool that fails answers an `isError` result, as the protocol wants.
 *
 * SCHEMAS ARE SERVED VERBATIM. The SDK's `McpServer.registerTool` takes Zod
 * shapes; an agent's tools carry JSON Schema, and the Python SDK serves the
 * same schema from the same tool fixtures, so this uses the SDK's low-level
 * `Server` and answers `tools/list` with the definitions as they are.
 *
 * STDOUT IS THE WIRE over stdio: `reserveStdoutForMcp()` sends every
 * `console.log` to stderr before the agent is built, because a skill that
 * prints one line while starting would corrupt the protocol stream.
 *
 * The SDK is imported dynamically (the build handbook's rule 4): the portal
 * typechecks this source against a node_modules that may not carry it, and a
 * load that fails says so (`mcpSdkMissing`).
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createContext } from '../core/context';
import type { IAgent, RunOptions } from '../core/types';
import type { FunctionToolDefinition, ToolDefinition } from '../uamp/types';

/** The lifecycle an agent may have; `IAgent` does not declare it, `BaseAgent` has it. */
type Lifecycle = { initialize?: () => Promise<void>; cleanup?: () => Promise<void> };

/** A tool a caller can name: the function tools; a provider's native tools have no schema to serve. */
function isFunctionTool(definition: ToolDefinition): definition is FunctionToolDefinition {
  return definition.type === 'function' && typeof (definition as FunctionToolDefinition).function?.name === 'string';
}
import { mcpSdkMissing } from '../skills/mcp/config';
import { hasCredential, unauthorizedResponse } from './credential-floor';
import { identificationContext, inboundRequest, refusalResponse } from './endpoint-gate';
import { agentVerifiesCredentials, defaultHostname, loopbackBindLine } from './origin-policy';
import { listenError } from './listen-error';

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Where the Streamable HTTP server answers. */
export const MCP_HTTP_PATH = '/mcp';

/** The caller over stdio: the person at this terminal, the owner (as `cli/app.ts` runs the chat). */
export const LOCAL_OWNER_AUTH: Record<string, unknown> = { authenticated: true, scope: 'owner', provider: 'local' };

/** Every tool the agent has, whoever asks: an admin passes every tier and group scope. */
const ANY_CALLER_AUTH: Record<string, unknown> = { authenticated: true, scope: 'admin', scopes: ['admin', 'owner'], provider: 'local' };

/** The refusal for a tool that exists and is not the caller's to use (fixture `refusals.not_open`). */
export function toolNotOpen(name: string): string {
  return `Tool "${name}" is not open to this caller.`;
}

/** The refusal for a name no tool has (fixture `refusals.unknown`). */
export function unknownTool(name: string): string {
  return `Unknown tool: ${name}`;
}

/** The server half of `@modelcontextprotocol/sdk`, as this module uses it. */
export interface McpServerSdk {
  Server: any;
  StdioServerTransport: any;
  WebStandardStreamableHTTPServerTransport: any;
  ListToolsRequestSchema: any;
  CallToolRequestSchema: any;
  McpError: any;
  ErrorCode: any;
}

/** How each SDK module is imported; a test hands in one that fails. */
export interface McpServerSdkImports {
  server: () => Promise<any>;
  stdio: () => Promise<any>;
  http: () => Promise<any>;
  types: () => Promise<any>;
}

const DEFAULT_IMPORTS: McpServerSdkImports = {
  server: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/server/index.js' as string),
  stdio: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/server/stdio.js' as string),
  http: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js' as string),
  types: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/types.js' as string),
};

let loadedSdk: McpServerSdk | undefined;

/** The SDK, or an error naming what failed (the same sentence as the client side, `skills/mcp/skill.ts`). */
export async function loadMcpServerSdk(imports: Partial<McpServerSdkImports> = {}): Promise<McpServerSdk> {
  if (loadedSdk) return loadedSdk;
  const modules = { ...DEFAULT_IMPORTS, ...imports };
  try {
    const [server, stdio, http, types] = await Promise.all([modules.server(), modules.stdio(), modules.http(), modules.types()]);
    const sdk: McpServerSdk = {
      Server: server?.Server,
      StdioServerTransport: stdio?.StdioServerTransport,
      WebStandardStreamableHTTPServerTransport: http?.WebStandardStreamableHTTPServerTransport,
      ListToolsRequestSchema: types?.ListToolsRequestSchema,
      CallToolRequestSchema: types?.CallToolRequestSchema,
      McpError: types?.McpError,
      ErrorCode: types?.ErrorCode,
    };
    for (const [name, value] of Object.entries(sdk)) {
      if (value === undefined) throw new Error(`the package has no ${name} export`);
    }
    loadedSdk = sdk;
    return sdk;
  } catch (err) {
    throw new Error(mcpSdkMissing((err as Error)?.message ?? String(err)));
  }
}

/** Forget the loaded SDK, so a test can load it again with other imports. */
export function resetMcpServerSdkForTests(): void {
  loadedSdk = undefined;
}

/** Tools as MCP lists them: name, description and the JSON Schema the tool declares, sorted by name. */
export function mcpToolsOf(
  definitions: ToolDefinition[],
): Array<{ name: string; description: string; inputSchema: Record<string, unknown> }> {
  return definitions
    .filter(isFunctionTool)
    .map((definition) => {
      const params = (definition.function.parameters ?? {}) as Record<string, unknown>;
      const inputSchema = typeof params.type === 'string' ? params : { type: 'object', properties: {}, ...params };
      return { name: definition.function.name, description: definition.function.description ?? '', inputSchema };
    })
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/**
 * What a tool returned, as MCP content: a string as text, a structured result
 * (`{ text, content_items }`) as text plus its images, `executeTool`'s own
 * owner refusal as an error, anything else as JSON.
 */
export function callToolResult(result: unknown): { content: Array<Record<string, unknown>>; isError?: boolean } {
  if (result === undefined || result === null) return { content: [{ type: 'text', text: '' }] };
  if (typeof result === 'string') return { content: [{ type: 'text', text: result }] };
  if (typeof result === 'object') {
    const record = result as Record<string, unknown>;
    if (Array.isArray(record.content_items)) {
      const content: Array<Record<string, unknown>> = [];
      if (typeof record.text === 'string' && record.text) content.push({ type: 'text', text: record.text });
      for (const item of record.content_items as Array<Record<string, unknown>>) {
        const image = typeof item?.image === 'string' ? item.image : undefined;
        const match = image ? /^data:([^;,]+);base64,(.*)$/s.exec(image) : null;
        if (match) content.push({ type: 'image', mimeType: match[1], data: match[2] });
        else content.push({ type: 'text', text: JSON.stringify(item) });
      }
      return { content: content.length ? content : [{ type: 'text', text: '' }] };
    }
    if (record.error === 'rejected_by_owner') {
      return { content: [{ type: 'text', text: JSON.stringify(record) }], isError: true };
    }
  }
  return { content: [{ type: 'text', text: JSON.stringify(result) }] };
}

/** The tools `caller` may see, as a run bound to that caller would list them. */
async function toolsFor(agent: IAgent, options: RunOptions): Promise<ToolDefinition[]> {
  if (typeof agent.listTools === 'function') return agent.listTools(options);
  return agent.getToolDefinitions?.() ?? [];
}

/**
 * The protocol server for ONE caller: `tools/list` answers what that caller may
 * use, `tools/call` runs it as that caller through `runTool`. Over stdio there
 * is one caller for the process; over HTTP one per request.
 */
export function createMcpProtocolServer(
  agent: IAgent,
  sdk: McpServerSdk,
  caller: Record<string, unknown>,
  version = packageVersion(),
): any {
  const server = new sdk.Server({ name: agent.name, version }, { capabilities: { tools: {} } });
  const options: RunOptions = { auth: caller };

  server.setRequestHandler(sdk.ListToolsRequestSchema, async () => ({
    tools: mcpToolsOf(await toolsFor(agent, options)),
  }));

  server.setRequestHandler(sdk.CallToolRequestSchema, async (request: any) => {
    const name = String(request?.params?.name ?? '');
    const args = (request?.params?.arguments ?? {}) as Record<string, unknown>;
    const named = (definitions: ToolDefinition[]) => definitions.filter(isFunctionTool).some((d) => d.function.name === name);
    if (!named(await toolsFor(agent, options))) {
      const exists = named(await toolsFor(agent, { auth: ANY_CALLER_AUTH }));
      throw new sdk.McpError(sdk.ErrorCode.InvalidParams, exists ? toolNotOpen(name) : unknownTool(name));
    }
    if (typeof agent.runTool !== 'function') {
      throw new sdk.McpError(sdk.ErrorCode.InternalError, `${agent.name} cannot run tools for a caller.`);
    }
    try {
      return callToolResult(await agent.runTool(name, args, options));
    } catch (err) {
      return { content: [{ type: 'text', text: (err as Error)?.message ?? String(err) }], isError: true };
    }
  });

  return server;
}

/** Reserve stdout for the protocol: everything `console.log` says goes to stderr from now on. */
export function reserveStdoutForMcp(): void {
  const toStderr = (...args: unknown[]) => console.error(...args);
  console.log = toStderr;
  console.info = toStderr;
  console.debug = toStderr;
}

/** This package's version, for the server's own card; `0.0.0` when it cannot be read. */
export function packageVersion(): string {
  try {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(fs.readFileSync(path.join(here, '..', '..', 'package.json'), 'utf-8')) as { version?: unknown };
    return typeof pkg.version === 'string' ? pkg.version : '0.0.0';
  } catch {
    return '0.0.0';
  }
}

/**
 * Serve the agent over stdio, as the owner, until the client closes the
 * stream. Owns `initialize()`, as `serve()` does for HTTP.
 */
export async function serveMcpStdio(agent: IAgent): Promise<void> {
  reserveStdoutForMcp();
  const sdk = await loadMcpServerSdk();
  await (agent as IAgent & Lifecycle).initialize?.();
  const server = createMcpProtocolServer(agent, sdk, LOCAL_OWNER_AUTH);
  const closed = new Promise<void>((resolve) => {
    server.onclose = () => resolve();
  });
  await server.connect(new sdk.StdioServerTransport());
  console.error(`[webagents] ${agent.name}: MCP over stdio`);
  await closed;
  await (agent as IAgent & Lifecycle).cleanup?.();
}

export interface McpHttpConfig {
  port: number;
  /** Unset: loopback unless the agent is meant to be reached (`origin-policy.ts`). */
  hostname?: string;
}

export interface McpHttpHandle {
  port: number;
  hostname: string;
  url: string;
  close(): Promise<void>;
}

/**
 * Serve the agent over stateless Streamable HTTP at `/mcp`. Every request is
 * identified on its own, and answered by a protocol server made for that
 * caller. Owns `initialize()`, as `serve()` does.
 */
export async function serveMcpHttp(agent: IAgent, config: McpHttpConfig): Promise<McpHttpHandle> {
  // A server never waits on a macOS keychain dialog (keychain-ux, 2026-09-27).
  (await import('../skills/secrets/keychain-ux')).forbidKeychainDialogs('serve');
  const sdk = await loadMcpServerSdk();
  const { Hono } = await import('hono');
  const { serve: nodeServe } = await import('@hono/node-server');
  await (agent as IAgent & Lifecycle).initialize?.();

  const verifiesCredentials = agentVerifiesCredentials(agent);
  const hostname = config.hostname || defaultHostname({ publicUrl: process.env.WEBAGENTS_PUBLIC_URL, verifiesCredentials });
  if (!config.hostname && hostname === '127.0.0.1') {
    // Loopback unless the agent verifies its callers (S-327): `serve`'s rule and words.
    console.log(loopbackBindLine(agent.name, process.env.WEBAGENTS_PUBLIC_URL));
  }
  const version = packageVersion();

  const app = new Hono();
  app.all(MCP_HTTP_PATH, async (c) => {
    const request = c.req.raw;
    // The floor (`credential-floor.ts`): nothing to authenticate, nothing
    // read. With the RFC 6750 challenge (2026-09-29): a 401 with no
    // `WWW-Authenticate` told an MCP client nothing about the scheme. The
    // floor's response carries it by default now, as every 401 does.
    if (!hasCredential(request)) return unauthorizedResponse();
    const body = new Uint8Array(await request.clone().arrayBuffer());
    const context = identificationContext(createContext(), inboundRequest(request, body));
    try {
      await agent.identifyCaller?.(context);
    } catch (err) {
      // A credential that was presented and refused: the challenge says
      // `error="invalid_token"`, the one rule every route applies
      // (`refusalResponse`, `credential-floor.ts` `refusalHeaders`).
      const refusal = refusalResponse(err, request);
      if (refusal) return c.json(refusal.body, refusal.status, refusal.headers);
      throw err;
    }
    const caller = { ...(context.auth ?? {}) } as Record<string, unknown>;
    const transport = new sdk.WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    const server = createMcpProtocolServer(agent, sdk, caller, version);
    await server.connect(transport);
    try {
      let parsedBody: unknown;
      if (body.length) {
        try {
          parsedBody = JSON.parse(new TextDecoder().decode(body));
        } catch {
          parsedBody = undefined; // the transport answers the parse error itself
        }
      }
      return await transport.handleRequest(request, parsedBody === undefined ? undefined : { parsedBody });
    } finally {
      await server.close().catch(() => undefined);
    }
  });

  return new Promise<McpHttpHandle>((resolve, reject) => {
    const listener = nodeServe({ fetch: app.fetch, port: config.port, hostname }, (info) => {
      const url = `http://${hostname}:${info.port}${MCP_HTTP_PATH}`;
      console.log(`[webagents] ${agent.name}: MCP on ${url}`);
      resolve({
        port: info.port,
        hostname,
        url,
        close: () =>
          new Promise<void>((done) => {
            listener.close(() => done());
          }),
      });
    });
    // A port already in use is one sentence (`listen-error.ts`, 2026-09-26).
    listener.on('error', (error: unknown) => reject(listenError(error, hostname, config.port)));
  });
}
