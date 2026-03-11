/**
 * AgentRuntime, Extension, AgentSource, and SkillFactory interfaces.
 *
 * These define the pluggable architecture for multi-agent serving.
 * The runtime resolves agents by name from registered sources, composes
 * skills via factories, and manages the agent lifecycle.
 *
 * Consumers:
 *   - OSS standalone: LocalDevExtension (file-based agents)
 *   - Portal: PortalExtension (DB-based agents, portal-native skills)
 *   - Browser: BrowserExtension (File System Access API + WebLLM)
 */

import type { BaseAgent } from './agent.js';
import type { ISkill, AgentConfig } from './types.js';
import type { Message, UsageStats, ContentItem } from '../uamp/types.js';
import type { ServerEvent } from '../uamp/events.js';

// ============================================================================
// Agent Info (discovery / listing)
// ============================================================================

export interface AgentInfo {
  /** Unique agent name / username */
  name: string;
  /** Human-readable display name */
  displayName?: string;
  /** Description */
  description?: string;
  /** Where this agent was loaded from */
  source: string;
  /** Whether the agent is currently loaded in memory */
  loaded: boolean;
  /** Model the agent uses (if known) */
  model?: string;
  /** Capabilities list */
  capabilities?: string[];
}

// ============================================================================
// Execute Options & Response
// ============================================================================

export interface ExecuteOptions {
  /** Auth context (user ID, scopes, token) */
  auth?: {
    userId?: string;
    scopes?: string[];
    token?: string;
    claims?: Record<string, unknown>;
  };
  /** Payment token */
  paymentToken?: string;
  /** Model override */
  model?: string;
  /** Max tool iterations override */
  maxToolIterations?: number;
  /** Request metadata */
  metadata?: Record<string, unknown>;
}

export interface ExecuteResponse {
  content: string;
  contentItems?: ContentItem[];
  usage?: UsageStats;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Middleware
// ============================================================================

export type MiddlewareNext = () => Promise<void>;

export interface MiddlewareContext {
  agentName: string;
  messages: Message[];
  options: ExecuteOptions;
  metadata: Record<string, unknown>;
}

export type Middleware = (
  ctx: MiddlewareContext,
  next: MiddlewareNext
) => Promise<void>;

// ============================================================================
// Runtime Hooks
// ============================================================================

export interface RuntimeHooks {
  /** Called when an agent is loaded for the first time */
  onAgentLoaded?(agent: BaseAgent, info: AgentInfo): Promise<void>;
  /** Called when an agent is unloaded / evicted from cache */
  onAgentUnloaded?(name: string): Promise<void>;
  /** Called before agent execution */
  onBeforeExecute?(name: string, messages: Message[], options: ExecuteOptions): Promise<void>;
  /** Called after agent execution */
  onAfterExecute?(name: string, response: ExecuteResponse): Promise<void>;
  /** Called on agent execution error */
  onExecuteError?(name: string, error: Error): Promise<void>;
}

// ============================================================================
// AgentSource: provides agents to the runtime
// ============================================================================

export interface AgentSource {
  /** Unique source type identifier (e.g., 'local-file', 'portal-db', 'remote-ws') */
  readonly type: string;

  /** Resolve an agent by name. Returns null if this source doesn't have it. */
  getAgent(name: string): Promise<BaseAgent | null>;

  /** List all agents available from this source. */
  listAgents(): Promise<AgentInfo[]>;

  /** Search agents by query (optional). */
  searchAgents?(query: string): Promise<AgentInfo[]>;

  /** Invalidate a cached agent (e.g., config changed). */
  invalidate?(name: string): void;

  /** Invalidate all cached agents from this source. */
  invalidateAll?(): void;
}

// ============================================================================
// SkillFactory: composes skills per-agent based on config
// ============================================================================

export interface SkillFactory {
  /** Factory name (e.g., 'portal-auth', 'local-dev', 'browser-llm') */
  readonly name: string;

  /**
   * Create skills for a specific agent configuration.
   * Called each time an agent is instantiated. The factory can inspect
   * the agent config to decide which skills to create and how to configure them.
   */
  createSkills(agentConfig: AgentConfig, runtime: AgentRuntime): ISkill[];
}

// ============================================================================
// Extension: bundles sources, factories, middleware, and hooks
// ============================================================================

export interface Extension {
  /** Extension name */
  readonly name: string;

  /** Initialize the extension (called after registration, before first use) */
  initialize(runtime: AgentRuntime): Promise<void>;

  /** Cleanup extension resources */
  cleanup?(): Promise<void>;

  /** Agent sources provided by this extension */
  getAgentSources(): AgentSource[];

  /** Skill factories provided by this extension */
  getSkillFactories(): SkillFactory[];

  /** Middleware chain provided by this extension */
  getMiddleware?(): Middleware[];

  /** Runtime hooks provided by this extension */
  getHooks?(): RuntimeHooks;
}

// ============================================================================
// AgentRuntime: the core orchestrator
// ============================================================================

export interface AgentRuntime {
  /**
   * Resolve an agent by name. Searches all registered sources in order.
   * Returns null if no source has the agent.
   */
  resolveAgent(name: string): Promise<BaseAgent | null>;

  /**
   * Execute an agent (non-streaming). Resolves the agent, runs the
   * agentic loop, and returns the final response.
   */
  execute(
    agentName: string,
    messages: Message[],
    opts?: ExecuteOptions
  ): Promise<ExecuteResponse>;

  /**
   * Execute an agent with streaming. Yields UAMP server events
   * as the agentic loop runs (text deltas, tool calls, etc.).
   */
  executeStreaming(
    agentName: string,
    messages: Message[],
    opts?: ExecuteOptions
  ): AsyncGenerator<ServerEvent, void, unknown>;

  /** Initialize the runtime and all registered extensions */
  initialize(): Promise<void>;

  /** Cleanup the runtime and all extensions */
  cleanup(): Promise<void>;

  /** Register an extension */
  registerExtension(extension: Extension): void;

  /** List all available agents across all sources */
  listAgents(): Promise<AgentInfo[]>;

  /** Search agents across all sources */
  searchAgents(query: string): Promise<AgentInfo[]>;
}

// ============================================================================
// DefaultAgentRuntime: concrete implementation
// ============================================================================

export class DefaultAgentRuntime implements AgentRuntime {
  private extensions: Extension[] = [];
  private sources: AgentSource[] = [];
  private factories: SkillFactory[] = [];
  private middlewareChain: Middleware[] = [];
  private hooks: RuntimeHooks[] = [];
  private agentCache: Map<string, { agent: BaseAgent; loadedAt: number }> = new Map();
  private initialized = false;

  /** Cache TTL in ms (default: 5 minutes) */
  private cacheTTL: number;

  constructor(options?: { cacheTTL?: number }) {
    this.cacheTTL = options?.cacheTTL ?? 300_000;
  }

  registerExtension(extension: Extension): void {
    this.extensions.push(extension);
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    for (const ext of this.extensions) {
      await ext.initialize(this);
      this.sources.push(...ext.getAgentSources());
      this.factories.push(...ext.getSkillFactories());
      if (ext.getMiddleware) {
        this.middlewareChain.push(...ext.getMiddleware());
      }
      if (ext.getHooks) {
        this.hooks.push(ext.getHooks());
      }
    }

    this.initialized = true;
  }

  async cleanup(): Promise<void> {
    for (const ext of this.extensions) {
      await ext.cleanup?.();
    }
    this.agentCache.clear();
    this.initialized = false;
  }

  async resolveAgent(name: string): Promise<BaseAgent | null> {
    // Check cache first
    const cached = this.agentCache.get(name);
    if (cached && Date.now() - cached.loadedAt < this.cacheTTL) {
      return cached.agent;
    }

    // Search sources in order
    for (const source of this.sources) {
      const agent = await source.getAgent(name);
      if (agent) {
        this.agentCache.set(name, { agent, loadedAt: Date.now() });
        for (const h of this.hooks) {
          await h.onAgentLoaded?.(agent, {
            name,
            source: source.type,
            loaded: true,
          });
        }
        return agent;
      }
    }

    return null;
  }

  async execute(
    agentName: string,
    messages: Message[],
    opts: ExecuteOptions = {}
  ): Promise<ExecuteResponse> {
    const agent = await this.resolveAgent(agentName);
    if (!agent) {
      throw new Error(`Agent not found: ${agentName}`);
    }

    for (const h of this.hooks) {
      await h.onBeforeExecute?.(agentName, messages, opts);
    }

    // Run middleware chain
    const ctx: MiddlewareContext = {
      agentName,
      messages,
      options: opts,
      metadata: {},
    };
    await this.runMiddleware(ctx);

    try {
      const response = await agent.run(messages, {
        model: opts.model,
      });

      const execResponse: ExecuteResponse = {
        content: response.content,
        contentItems: response.content_items,
        usage: response.usage,
        metadata: ctx.metadata,
      };

      for (const h of this.hooks) {
        await h.onAfterExecute?.(agentName, execResponse);
      }

      return execResponse;
    } catch (error) {
      for (const h of this.hooks) {
        await h.onExecuteError?.(agentName, error as Error);
      }
      throw error;
    }
  }

  async *executeStreaming(
    agentName: string,
    messages: Message[],
    opts: ExecuteOptions = {}
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const agent = await this.resolveAgent(agentName);
    if (!agent) {
      throw new Error(`Agent not found: ${agentName}`);
    }

    for (const h of this.hooks) {
      await h.onBeforeExecute?.(agentName, messages, opts);
    }

    const ctx: MiddlewareContext = {
      agentName,
      messages,
      options: opts,
      metadata: {},
    };
    await this.runMiddleware(ctx);

    yield* agent.runStreaming(messages, {
      model: opts.model,
    }) as unknown as AsyncGenerator<ServerEvent, void, unknown>;
  }

  async listAgents(): Promise<AgentInfo[]> {
    const all: AgentInfo[] = [];
    for (const source of this.sources) {
      const agents = await source.listAgents();
      all.push(...agents);
    }
    return all;
  }

  async searchAgents(query: string): Promise<AgentInfo[]> {
    const all: AgentInfo[] = [];
    for (const source of this.sources) {
      if (source.searchAgents) {
        const results = await source.searchAgents(query);
        all.push(...results);
      }
    }
    return all;
  }

  /** Invalidate a cached agent */
  invalidateAgent(name: string): void {
    this.agentCache.delete(name);
    for (const source of this.sources) {
      source.invalidate?.(name);
    }
  }

  /** Invalidate all cached agents */
  invalidateAll(): void {
    this.agentCache.clear();
    for (const source of this.sources) {
      source.invalidateAll?.();
    }
  }

  /** Get registered skill factories (used by agent sources when composing agents) */
  getSkillFactories(): SkillFactory[] {
    return this.factories;
  }

  private async runMiddleware(ctx: MiddlewareContext): Promise<void> {
    let index = 0;
    const chain = this.middlewareChain;

    const next: MiddlewareNext = async () => {
      if (index < chain.length) {
        const mw = chain[index++];
        await mw(ctx, next);
      }
    };

    await next();
  }
}
