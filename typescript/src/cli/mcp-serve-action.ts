/**
 * The `webagents mcp serve` command's action (plan item 1.8, 2026-09-26), out
 * of `cli/index.ts` for the reason `serve-action.ts` is: that module parses
 * argv at import time, so a command can only be tested from its own module.
 *
 * It builds THE SAME AGENT `serve` builds (`createServedAgent`: the file's
 * skills and model, the stored keys, the folder, the access block) and hands
 * it to `server/mcp.ts`: stdio by default, Streamable HTTP with `--http`.
 * Over stdio, stdout is the wire, so it is reserved before the file is even
 * read (`loadAgentConfigFile` prints a line when there is no file). The
 * Python twin is `python/webagents/cli/mcp_serve.py`.
 */

import type { IAgent } from '../core/types';

export interface McpServeCommandOptions {
  /** `--http <port>` as commander hands it over: a string, or unset for stdio. */
  http?: string;
  /** Unset with `--http`: loopback unless the agent is meant to be reached. */
  host?: string;
}

/** Seams for the unit test. Every one defaults to what the CLI really does. */
export interface McpServeCommandDeps {
  createAgent?: (agentPath: string) => Promise<IAgent> | IAgent;
  serveStdio?: (agent: IAgent) => Promise<unknown>;
  serveHttp?: (agent: IAgent, config: { port: number; hostname?: string }) => Promise<unknown>;
}

/** The port `--http` names, or a `RangeError` saying what it should have been. */
export function httpPort(value: string): number {
  const port = /^\d{1,5}$/.test(value.trim()) ? Number(value) : NaN;
  if (!Number.isInteger(port) || port > 65535) {
    throw new RangeError(`--http takes a port number from 0 to 65535, not "${value}".`);
  }
  return port;
}

export async function mcpServeAction(
  agentPath: string,
  options: McpServeCommandOptions,
  deps: McpServeCommandDeps = {},
): Promise<void> {
  let port: number | undefined;
  if (options.http !== undefined) {
    try {
      port = httpPort(options.http);
    } catch (err) {
      console.error((err as Error).message);
      process.exitCode = 1;
      return;
    }
  } else {
    const { reserveStdoutForMcp } = await import('../server/mcp.js');
    reserveStdoutForMcp();
  }

  const createAgent =
    deps.createAgent ??
    (async (target: string) => {
      // Over stdio the caller is the MCP client the owner started, so the
      // owner's turns (the sign-in may pay); over HTTP, whoever reaches the
      // port, so never the sign-in (S-327), as `serve`.
      const { createServedAgent } = await import('./serve-action.js');
      return createServedAgent(target, { forCallers: port !== undefined });
    });
  const agent = await createAgent(agentPath);

  if (port === undefined) {
    const serveStdio =
      deps.serveStdio ??
      (async (built: IAgent) => {
        const { serveMcpStdio } = await import('../server/mcp.js');
        return serveMcpStdio(built);
      });
    await serveStdio(agent);
    return;
  }

  const serveHttp =
    deps.serveHttp ??
    (async (built: IAgent, config: { port: number; hostname?: string }) => {
      const { serveMcpHttp } = await import('../server/mcp.js');
      const handle = await serveMcpHttp(built, config);
      process.on('SIGINT', () => {
        void handle.close().then(() => process.exit(0));
      });
      return handle;
    });
  await serveHttp(agent, { port, hostname: options.host });
}
