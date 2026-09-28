/**
 * The `mcp` entry of an agent file, normalized (plan item 0.3, 2026-09-26).
 *
 * WHAT A FILE MAY WRITE. Two shapes, the ones the Python loader accepted
 * first: the servers at the top level (`- mcp: {sqlite: {command: npx, ...}}`)
 * or under `mcpServers`, the key `mcp.json` files use, so a file copied from
 * another tool's config loads unchanged. A server is stdio (`command`, with
 * `args`, `env`, `cwd`) or remote (`url`, or `httpUrl`; `transport` names
 * `http` for Streamable HTTP, `sse`, or `auto` when unset: Streamable HTTP
 * first, then SSE; `headers` go on every request). The shapes and what each
 * resolves to are pinned by `python/tests/fixtures/mcp_tool/config_shapes.json`,
 * which both SDKs run, so an agent file loads the same servers in each.
 *
 * A server that names neither is rejected BY NAME, and the others still load:
 * before this the TypeScript skill skipped it with a console line and the
 * Python skill raised inside its connect loop, so the same file gave two
 * different agents.
 *
 * ONE NAME PER TOOL. A server's tool is `<server>__<tool>`, always. TypeScript
 * did this from the start; Python used the bare name unless it collided with
 * one already registered, which made a tool's name depend on which OTHER
 * servers the file named and on the order they connected in (this SDK connects
 * them concurrently). An `access.tools` rule, and the model's tool calls, refer
 * to the name, so a name that can change when a server is added is a rule that
 * can stop matching without anyone touching it. Both SDKs now qualify every
 * tool, and the fixture pins the cases.
 *
 * The extra keys a hosted platform sets on a server (`pricing`, `enabledTools`,
 * `toolPolicies`, `auth`, `mcpUrlTemplate`, `urlQuery`, `prompt`) are kept on
 * the server's `configs` entry untouched; only the transport shape is
 * normalized here. `mcpUrlTemplate` counts as a remote address, composed at
 * connect time by the skill.
 *
 * SECRETS (S-292, 2026-09-26). A value in `env`, `headers` or the address may
 * be `${secret:NAME}` or `${env:NAME}` (`skills/secrets/references.ts`),
 * resolved by the skill at connect time and never here, so nothing that
 * reads a resolution sees a value. Two things ARE decided here, at load, so
 * `doctor` and the chat can say them before any server starts: a reference
 * in `command` or `args` rejects that server by name (a command line is
 * readable by every local account), and a literal that looks like a key
 * (`sk-`, `ghp_`, a long bearer) in `env` or `headers` draws a warning that
 * names the reference and the `webagents secrets set` command to use instead.
 * Both sentences are pinned by the fixture's `secret_refs` section.
 */

import { commandLineRefusal, literalWarning, looksLikeSecret, mentionsReference } from '../secrets/references';

export type ResolvedMcpTransport = 'stdio' | 'http' | 'sse' | 'auto';

/** One server as the fixture describes it: keys a config does not set are absent. */
export interface ResolvedMcpServer {
  name: string;
  transport: ResolvedMcpTransport;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;
  headers?: Record<string, string>;
}

export interface RejectedMcpServer {
  name: string;
  reason: string;
}

/**
 * The refusal for an entry that asks for a sandbox (`sandbox: true`, or any
 * value but `false` and `null`) this SDK cannot give an MCP server (S-313,
 * 2026-09-27). The key was ignored here, so a server the file declared
 * confined ran with the owner's permissions and no sign of it: the S-217
 * shape, a declaration that does something other than it says. Refused by
 * name at load, the other servers still load. The Python skill honours the
 * key only with the Docker `sandbox` skill loaded and refuses it otherwise
 * with its own sentence. Pinned by the shared fixture (`sandbox_key`).
 */
export const SANDBOX_UNAVAILABLE =
  'asks for a sandbox (sandbox: true) that this SDK cannot provide for an MCP server, so it did not start: remove the key to run it with your own permissions';

/** Whether an entry asks for a sandbox: the key present with any value but `false` and `null`. */
export function asksForSandbox(entry: Record<string, unknown>): boolean {
  return 'sandbox' in entry && entry.sandbox !== false && entry.sandbox !== null && entry.sandbox !== undefined;
}

export interface McpServersResolution {
  /** The servers that load, in the file's order. */
  servers: ResolvedMcpServer[];
  /** The entries that do not, each with the fixture's reason. */
  rejected: RejectedMcpServer[];
  /** Each loading server's config as written, with `transport` filled in, for the skill's extra keys. */
  configs: Record<string, Record<string, unknown>>;
  /**
   * Servers that load but carry a secret-looking literal in `env` or
   * `headers` (S-292): one entry per offending key, said once at load and
   * again by `doctor`. Never a value.
   */
  warnings: RejectedMcpServer[];
}

/** Between a server's name and its tool's name. */
export const MCP_TOOL_SEPARATOR = '__';

/**
 * What a CLI says on stderr about a server that did not load or connect, or
 * carries a literal that looks like a key (2026-09-26, the e2e run: the
 * Python `-p` was silent about MCP problems while this SDK said them). This
 * skill prints them itself at load and connect; the Python `-p` prints them
 * from the skill's `server_report()` after the agent is built. Pinned by the
 * shared fixture (`problem_lines`).
 */
export const MCP_PROBLEM_LINES = {
  failed: '[MCPSkill] Server "{name}" failed to connect: {error}',
  rejected: '[MCPSkill] Server "{name}" {reason}; skipping it.',
  warning: '[MCPSkill] Server "{name}" {warning}',
} as const;

export function mcpProblemLine(kind: keyof typeof MCP_PROBLEM_LINES, name: string, text: string): string {
  return MCP_PROBLEM_LINES[kind].replace('{name}', name).replace(/\{(error|reason|warning)\}/, text);
}

/**
 * The stderr lines for a `serverReport()` (the Python `problem_lines` twin):
 * a rejected entry, a failed connection, and each literal warning, in the
 * report's order. The skill prints these as it goes; a host that only has
 * the report can print the same lines.
 */
export function mcpProblemLines(
  report: readonly { name: string; rejected?: string; error?: string; warnings: readonly string[] }[],
): string[] {
  const lines: string[] = [];
  for (const row of report) {
    if (row.rejected) lines.push(mcpProblemLine('rejected', row.name, row.rejected));
    else if (row.error) lines.push(mcpProblemLine('failed', row.name, row.error));
    for (const warning of row.warnings) lines.push(mcpProblemLine('warning', row.name, warning));
  }
  return lines;
}

/** The name a server's tool gets in the agent: `<server>__<tool>`, always. */
export function qualifiedToolName(server: string, tool: string): string {
  return `${server}${MCP_TOOL_SEPARATOR}${tool}`;
}

/** The load-time error when the MCP SDK cannot be loaded (fixture `missing_sdk.typescript`). */
export function mcpSdkMissing(reason: string): string {
  return (
    `The mcp skill needs the MCP SDK, which could not be loaded: ${reason}. ` +
    'Install @modelcontextprotocol/sdk next to webagents (npm install @modelcontextprotocol/sdk).'
  );
}

function isMapping(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function stringMap(value: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!isMapping(value)) return out;
  for (const [key, entry] of Object.entries(value)) {
    if (entry !== undefined && entry !== null) out[key] = String(entry);
  }
  return out;
}

/**
 * The servers a `mcp` entry names. Accepts the top-level shape and the
 * `mcpServers` wrapper; anything that is not a mapping names no server.
 */
export function mcpServersFromConfig(raw: unknown): McpServersResolution {
  const resolution: McpServersResolution = { servers: [], rejected: [], configs: {}, warnings: [] };
  if (!isMapping(raw)) return resolution;
  const entries = isMapping(raw.mcpServers) ? raw.mcpServers : raw;
  for (const [name, entry] of Object.entries(entries)) {
    if (name === 'mcpServers') continue;
    if (!isMapping(entry)) {
      resolution.rejected.push({ name, reason: 'is not a mapping' });
      continue;
    }
    // A sandbox this SDK cannot provide is refused, not ignored (S-313).
    if (asksForSandbox(entry)) {
      resolution.rejected.push({ name, reason: SANDBOX_UNAVAILABLE });
      continue;
    }
    const command = typeof entry.command === 'string' && entry.command ? entry.command : undefined;
    const address = [entry.url, entry.httpUrl, entry.mcpUrlTemplate].find((v) => typeof v === 'string' && v) as
      | string
      | undefined;
    if (command) {
      const args = Array.isArray(entry.args) ? entry.args.map(String) : [];
      // A reference on the command line is refused, not resolved (S-292):
      // `ps` shows it to every local account, so no store can keep it secret.
      const onCommandLine = mentionsReference(command) ? 'command' : args.some(mentionsReference) ? 'args' : undefined;
      if (onCommandLine) {
        resolution.rejected.push({ name, reason: commandLineRefusal(onCommandLine) });
        continue;
      }
      const server: ResolvedMcpServer = { name, transport: 'stdio', command, args };
      if (isMapping(entry.env)) server.env = stringMap(entry.env);
      if (typeof entry.cwd === 'string' && entry.cwd) server.cwd = entry.cwd;
      resolution.servers.push(server);
      resolution.configs[name] = { ...entry, transport: 'stdio' };
      for (const [key, value] of Object.entries(server.env ?? {})) {
        if (looksLikeSecret(value)) resolution.warnings.push({ name, reason: literalWarning(name, 'env', key) });
      }
      continue;
    }
    if (!address) {
      resolution.rejected.push({ name, reason: 'has neither command nor url' });
      continue;
    }
    const transport = entry.transport === undefined || entry.transport === null ? 'auto' : entry.transport;
    if (transport !== 'http' && transport !== 'sse' && transport !== 'auto') {
      resolution.rejected.push({ name, reason: 'transport must be http, sse or auto' });
      continue;
    }
    const headers = stringMap(entry.headers);
    resolution.servers.push({ name, transport, url: address, headers });
    resolution.configs[name] = { ...entry, transport };
    for (const [key, value] of Object.entries(headers)) {
      if (looksLikeSecret(value)) resolution.warnings.push({ name, reason: literalWarning(name, 'headers', key) });
    }
  }
  return resolution;
}
