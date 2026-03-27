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
      // SSE transport not available — stdio-only mode
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
  private prompts: Map<string, MCPPrompt[]> = new Map();
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
    this.toolsRegistry.clear();
    this.resources.clear();
    this.prompts.clear();
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

  private async _connectServer(name: string, config: MCPServerConfig): Promise<void> {
    if (config.pricing) {
      this.serverPricing.set(name, config.pricing);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let transport: any;

    if (config.url) {
      // SSE transport
      if (!SSETransport) {
        console.warn(`[MCPSkill] SSE transport unavailable — skipping server "${name}".`);
        return;
      }
      transport = new SSETransport(new URL(config.url), {
        requestInit: config.headers ? { headers: config.headers } : undefined,
      });
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

    await client.connect(transport);
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
      const tools: MCPToolDef[] = toolsResp?.tools ?? [];
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
        this.prompts.set(name, prompts);
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
        prompts: (this.prompts.get(name) ?? []).map((p) => ({
          name: p.name,
          description: p.description,
          arguments: p.arguments,
        })),
      };
    }

    return { servers, total_servers: this.sessions.size, total_tools: this.toolsRegistry.size };
  }
}
