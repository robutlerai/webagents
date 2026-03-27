/**
 * Dynamic Routing Skill
 *
 * Enables runtime agent-to-agent routing:
 * - Agent discovery via portal API or local registry
 * - Dynamic delegation (agent calls another agent's tool / handoff)
 * - Cross-agent NLI routing (natural language delegation)
 *
 * The skill exposes tools that let an LLM decide when and how to
 * delegate work to external agents, and a hook that intercepts
 * tool calls targeting remote agents and proxies them.
 */

import { Skill } from '../../core/skill';
import { tool, hook } from '../../core/decorators';
import type { Context, HookData, HookResult } from '../../core/types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DynamicRoutingConfig {
  name?: string;
  enabled?: boolean;
  /** Portal API base URL for agent discovery */
  portalUrl?: string;
  /** API key for portal authentication */
  apiKey?: string;
  /** Static agent registry (name → URL) */
  agents?: Record<string, string>;
  /** Discovery cache TTL in ms (default: 60_000) */
  cacheTtl?: number;
  /** Request timeout in ms (default: 30_000) */
  timeout?: number;
}

export interface AgentEntry {
  name: string;
  url: string;
  description?: string;
  capabilities?: Record<string, unknown>;
  lastSeen?: number;
}

// ---------------------------------------------------------------------------
// DynamicRoutingSkill
// ---------------------------------------------------------------------------

export class DynamicRoutingSkill extends Skill {
  private portalUrl?: string;
  private apiKey?: string;
  private timeout: number;
  private cacheTtl: number;
  private registry = new Map<string, AgentEntry>();
  private cacheTimestamp = 0;

  constructor(config: DynamicRoutingConfig = {}) {
    super({ ...config, name: config.name || 'dynamic-routing' });
    this.portalUrl = config.portalUrl;
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30_000;
    this.cacheTtl = config.cacheTtl ?? 60_000;

    if (config.agents) {
      for (const [name, url] of Object.entries(config.agents)) {
        this.registry.set(name, { name, url });
      }
    }
  }

  // =========================================================================
  // Tools
  // =========================================================================

  @tool({
    name: 'search_agent_registry',
    description: 'Search the local agent registry for available agents that can help with a task. Returns agent names, descriptions, and capabilities.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Natural language description of what you need help with',
        },
      },
    },
  })
  async discoverAgents(
    params: { query?: string },
    _context: Context,
  ): Promise<AgentEntry[]> {
    await this._refreshRegistryIfStale();
    const entries = Array.from(this.registry.values());

    if (!params.query) return entries;

    const q = params.query.toLowerCase();
    return entries.filter((e) => {
      const haystack = [
        e.name,
        e.description ?? '',
        JSON.stringify(e.capabilities ?? {}),
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(q);
    });
  }

  @tool({
    name: 'delegate_to_agent',
    description:
      'Send a message to another agent and receive its response. ' +
      'Use this when you need to delegate a sub-task to a specialized agent.',
    parameters: {
      type: 'object',
      properties: {
        agent_name: {
          type: 'string',
          description: 'Name of the target agent',
        },
        message: {
          type: 'string',
          description: 'The message or task to send to the agent',
        },
        instructions: {
          type: 'string',
          description: 'Optional system instructions for the delegated agent',
        },
      },
      required: ['agent_name', 'message'],
    },
  })
  async delegateToAgent(
    params: { agent_name: string; message: string; instructions?: string },
    context: Context,
  ): Promise<string> {
    const entry = this.registry.get(params.agent_name);
    if (!entry) {
      await this._refreshRegistryIfStale();
      const refreshed = this.registry.get(params.agent_name);
      if (!refreshed) {
        return `Error: Agent "${params.agent_name}" not found. Use search_agent_registry to list available agents.`;
      }
      return this._callAgent(refreshed, params.message, params.instructions, context);
    }
    return this._callAgent(entry, params.message, params.instructions, context);
  }

  @tool({
    name: 'register_agent',
    description: 'Register an external agent URL so it can be discovered and delegated to.',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Agent name' },
        url: { type: 'string', description: 'Agent endpoint URL' },
        description: { type: 'string', description: 'What the agent does' },
      },
      required: ['name', 'url'],
    },
  })
  async registerAgent(
    params: { name: string; url: string; description?: string },
    _context: Context,
  ): Promise<string> {
    this.registry.set(params.name, {
      name: params.name,
      url: params.url,
      description: params.description,
      lastSeen: Date.now(),
    });
    return `Agent "${params.name}" registered at ${params.url}`;
  }

  // =========================================================================
  // Hook: intercept before_tool for cross-agent proxying
  // =========================================================================

  @hook({ lifecycle: 'before_tool', priority: 80 })
  async interceptRemoteToolCall(
    data: HookData,
    _context: Context,
  ): Promise<HookResult | void> {
    // If the tool name matches a known agent prefix (e.g. "agent:research-bot:search"),
    // proxy the call to the remote agent
    const toolName = data.tool_name ?? '';
    if (!toolName.startsWith('agent:')) return;

    const parts = toolName.split(':');
    if (parts.length < 3) return;

    const agentName = parts[1];
    const remoteTool = parts.slice(2).join(':');
    const entry = this.registry.get(agentName);
    if (!entry) return;

    const result = await this._callAgentTool(entry, remoteTool, data.tool_params ?? {});
    return { tool_result: result, skip_remaining: true };
  }

  // =========================================================================
  // Internal: agent communication
  // =========================================================================

  private async _callAgent(
    entry: AgentEntry,
    message: string,
    instructions?: string,
    context?: Context,
  ): Promise<string> {
    const url = entry.url.replace(/\/$/, '');

    const completionsUrl = `${url}/v1/chat/completions`;
    const messages: Array<{ role: string; content: string }> = [];
    if (instructions) {
      messages.push({ role: 'system', content: instructions });
    }
    messages.push({ role: 'user', content: message });

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    const paymentToken =
      (context as any)?.payment?.token ??
      context?.get?.('payment_token') ??
      (context?.metadata?.paymentToken as string);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(completionsUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({ messages, stream: false }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        return `Error calling agent ${entry.name}: ${res.status} ${text}`;
      }

      const body = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      return body.choices?.[0]?.message?.content ?? '(no response)';
    } catch (err) {
      return `Error calling agent ${entry.name}: ${(err as Error).message}`;
    } finally {
      clearTimeout(timer);
    }
  }

  private async _callAgentTool(
    entry: AgentEntry,
    toolName: string,
    params: Record<string, unknown>,
  ): Promise<string> {
    // Wrap tool invocation as a user message asking the agent to use the tool
    return this._callAgent(
      entry,
      `Execute tool "${toolName}" with parameters: ${JSON.stringify(params)}`,
    );
  }

  private async _refreshRegistryIfStale(): Promise<void> {
    if (Date.now() - this.cacheTimestamp < this.cacheTtl) return;
    if (!this.portalUrl) return;

    const headers: Record<string, string> = {};
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5_000);
      const res = await fetch(`${this.portalUrl}/api/agents`, {
        headers,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) return;

      const body = (await res.json()) as {
        agents?: Array<{
          name: string;
          url: string;
          description?: string;
          capabilities?: Record<string, unknown>;
        }>;
      };

      if (body.agents) {
        for (const agent of body.agents) {
          this.registry.set(agent.name, {
            ...agent,
            lastSeen: Date.now(),
          });
        }
      }

      this.cacheTimestamp = Date.now();
    } catch {
      // Discovery failure is not fatal — use stale cache
    }
  }
}
