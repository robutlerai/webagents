import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context, Tool, StructuredToolResult, PricingConfig } from '../../core/types';
import type { ContentItem, ImageContent } from '../../uamp/types';
import { ensureContentId } from '../../uamp/content';

// ---------------------------------------------------------------------------
// MCP SDK lazy-loaded references
// ---------------------------------------------------------------------------

let mcpAvailable = false;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let MCPClient: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let StdioTransport: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let SSETransport: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let StreamableHTTPTransport: any;

async function ensureMCP(): Promise<boolean> {
  if (mcpAvailable) return true;
  try {
    // Dynamic imports for optional MCP SDK dependency
    const clientMod = await import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/index.js' as string);
    MCPClient = clientMod.Client;
    const stdioMod = await import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/stdio.js' as string);
    StdioTransport = stdioMod.StdioClientTransport;
    try {
      const sseMod = await import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/sse.js' as string);
      SSETransport = sseMod.SSEClientTransport;
    } catch {
      // SSE transport not available
    }
    try {
      const httpMod = await import(/* @vite-ignore */ '@modelcontextprotocol/sdk/client/streamableHttp.js' as string);
      StreamableHTTPTransport = httpMod.StreamableHTTPClientTransport;
    } catch {
      // Streamable HTTP transport not available
    }
    mcpAvailable = true;
    return true;
  } catch {
    return false;
  }
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

export interface MCPSkillConfig {
  agentName?: string;
  agentPath?: string;
  baseDir?: string;
  mcp?: Record<string, MCPServerConfig> | { mcpServers: Record<string, MCPServerConfig> };
  [key: string]: unknown;
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

export class MCPSkill extends Skill {
  private mcpConfig: MCPSkillConfig;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private sessions: Map<string, any> = new Map();
  private toolsRegistry: Map<string, ToolRegistryEntry> = new Map();
  private resources: Map<string, MCPResource[]> = new Map();
  private mcpPrompts: Map<string, MCPPrompt[]> = new Map();
  private _initialized = false;
  private _cleanupFns: Array<() => Promise<void>> = [];

  constructor(config: MCPSkillConfig = {}) {
    super({ name: 'MCPSkill' });
    this.mcpConfig = config;
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  override async initialize(): Promise<void> {
    if (this._initialized) return;

    const available = await ensureMCP();
    if (!available) {
      console.warn(
        '[MCPSkill] @modelcontextprotocol/sdk is not installed — MCP tools will not be available.',
      );
      return;
    }

    const servers = await this._loadMCPConfig();
    if (!servers || Object.keys(servers).length === 0) {
      return;
    }

    const results = await Promise.allSettled(
      Object.entries(servers).map(([name, cfg]) => this._connectServer(name, cfg)),
    );

    for (const result of results) {
      if (result.status === 'rejected') {
        console.error('[MCPSkill] Server connection failed:', result.reason);
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
    this._initialized = false;
  }

  // =========================================================================
  // Config loading
  // =========================================================================

  private async _loadMCPConfig(): Promise<Record<string, MCPServerConfig>> {
    // 1. Config object passed directly
    if (this.mcpConfig.mcp) {
      const raw = this.mcpConfig.mcp;
      if ('mcpServers' in raw) {
        return (raw as { mcpServers: Record<string, MCPServerConfig> }).mcpServers;
      }
      return raw as Record<string, MCPServerConfig>;
    }

    // 2. Try loading mcp.json from disk (Node environments only)
    const baseDir =
      this.mcpConfig.baseDir ??
      this.mcpConfig.agentPath ??
      (typeof process !== 'undefined' ? process.cwd() : undefined);

    if (!baseDir) return {};

    try {
      const fs = await import('node:fs/promises');
      const path = await import('node:path');
      const configPath = path.join(baseDir, 'mcp.json');
      const raw = await fs.readFile(configPath, 'utf-8');
      const parsed = JSON.parse(raw) as Record<string, unknown>;

      if ('mcpServers' in parsed && typeof parsed.mcpServers === 'object') {
        return parsed.mcpServers as Record<string, MCPServerConfig>;
      }
      return parsed as unknown as Record<string, MCPServerConfig>;
    } catch {
      return {};
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

  private async _connectServer(name: string, config: MCPServerConfig): Promise<void> {
    this.serverConfigs.set(name, config);
    if (config.pricing) {
      this.serverPricing.set(name, config.pricing);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let transport: any;

    if (config.url || config.mcpUrlTemplate) {
      const composed = this._composeServerUrl(name, config);
      if (!composed) return;
      const { url, queryAuth } = composed;

      // Build the Authorization header for header-based auth modes only.
      // - oauth2            → `Authorization: Bearer <token>` (header)
      // - api_key           → custom raw header, or legacy Authorization bearer
      // - api_key_query     → token is in URL; no header
      // - none              → no auth at all
      const headers: Record<string, string> = { ...(config.headers ?? {}) };
      const authType = config.authType ?? (config.auth?.type as MCPAuthType | undefined) ?? 'none';
      if (!queryAuth && config.auth?.token && (authType === 'oauth2' || authType === 'api_key')) {
        if (authType === 'api_key') {
          headers[config.auth.headerName || 'Authorization'] = config.auth.headerName
            ? config.auth.token
            : `Bearer ${config.auth.token}`;
        } else {
          headers[config.auth.headerName || 'Authorization'] = `Bearer ${config.auth.token}`;
        }
      }

      const requestInit: RequestInit = Object.keys(headers).length ? { headers } : {};
      const transportKind = this._resolveTransportKind(config);

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
    } else if (config.command) {
      // Stdio transport
      transport = new StdioTransport({
        command: config.command,
        args: config.args ?? [],
        env: config.env ? { ...process.env, ...config.env } : undefined,
        cwd: config.cwd,
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
      // If `auto` failed via HTTP, fall back to SSE once.
      if (this._resolveTransportKind(config) === 'auto' && transport && SSETransport) {
        try {
          const composed = this._composeServerUrl(name, config);
          if (composed) {
            const headers: Record<string, string> = { ...(config.headers ?? {}) };
            const authType = config.authType ?? 'none';
            if (!composed.queryAuth && config.auth?.token && (authType === 'oauth2' || authType === 'api_key')) {
              if (authType === 'api_key') {
                headers[config.auth.headerName || 'Authorization'] = config.auth.headerName
                  ? config.auth.token
                  : `Bearer ${config.auth.token}`;
              } else {
                headers[config.auth.headerName || 'Authorization'] = `Bearer ${config.auth.token}`;
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
        const qualifiedName = `${name}__${t.name}`;
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
