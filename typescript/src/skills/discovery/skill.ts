/**
 * Portal Discovery Skill
 *
 * Unified search across the Robutler platform:
 * - Intent-based agent search (semantic)
 * - Agent listing and info
 * - Content search (posts, channels, users, tags)
 * - Auto-publish (automatically publish capabilities on init)
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiscoveryConfig {
  /** Portal URL for discovery API */
  portalUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Request timeout in ms */
  timeout?: number;
  /** Agent ID (for auto-publish) */
  agentId?: string;
  /** Auto-publish intents on initialization */
  autoPublish?: boolean;
  /** Intents to auto-publish */
  intents?: string[];
  /** Capabilities to auto-publish */
  capabilities?: string[];
  /** Agent description for auto-publish */
  description?: string;
  /** Agent category */
  category?: string;
  /** Commands this agent supports */
  commands?: AgentCommand[];
}

export interface AgentSearchResult {
  agentUrl: string;
  name: string;
  description: string;
  intents: string[];
  score: number;
  capabilities?: string[];
  category?: string;
  status?: 'online' | 'offline' | 'busy';
  pricing?: { model: string; currency: string };
}

export interface PublishedIntent {
  intent: string;
  description?: string;
  examples?: string[];
}

export interface AgentCommand {
  name: string;
  description: string;
  parameters?: Record<string, { type: string; description: string; required?: boolean }>;
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

export class PortalDiscoverySkill extends Skill {
  private discoveryConfig: DiscoveryConfig;

  constructor(config: DiscoveryConfig = {}) {
    super({ name: 'portal-discovery' });
    this.discoveryConfig = {
      portalUrl: config.portalUrl || 'https://portal.webagents.ai',
      apiKey: config.apiKey || this.getEnvApiKey(),
      timeout: config.timeout || 8000,
      agentId: config.agentId,
      autoPublish: config.autoPublish ?? false,
      intents: config.intents ?? [],
      capabilities: config.capabilities ?? [],
      description: config.description,
      category: config.category,
      commands: config.commands ?? [],
    };
  }

  override async initialize(): Promise<void> {
    await super.initialize();
    if (this.discoveryConfig.autoPublish && this.discoveryConfig.intents?.length) {
      try {
        await this._autoPublish();
      } catch (err) {
        console.warn('[discovery] Auto-publish failed:', (err as Error).message);
      }
    }
  }

  // ============================================================================
  // Consolidated search tool
  // ============================================================================

  @tool({
    name: 'search',
    description:
      'Search the Robutler platform for agents, capabilities, content, and users. ' +
      'Use this when you need to find agents that can perform a task, discover ' +
      'content in channels, or look up users.\n\n' +
      'Returns results grouped by type. Each agent result includes: username, ' +
      'display name, description, capabilities, and URL. Each intent result ' +
      'includes the intent text, agent ID, and similarity score.\n\n' +
      'Examples:\n' +
      '- Find image generation agents: query="generate images", types=["intents","agents"]\n' +
      '- Find posts about AI: query="artificial intelligence", types=["posts"]\n' +
      '- List trending channels: query="popular", types=["channels"]',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'What to search for' },
        types: {
          type: 'array',
          items: { type: 'string', enum: ['intents', 'agents', 'posts', 'channels', 'users', 'tags'] },
          description: 'Result types to include (default: ["intents","agents"])',
        },
        limit: { type: 'number', description: 'Max results per type (default: 10)' },
      },
      required: ['query'],
    },
  })
  async search(
    params: { query: string; types?: string[]; limit?: number },
    _context: Context,
  ): Promise<Record<string, unknown>> {
    const types = params.types ?? ['intents', 'agents'];
    const limit = params.limit ?? 10;
    const results: Record<string, unknown> = {};
    const timeout = this.discoveryConfig.timeout!;
    const headers = this.buildHeaders();
    const base = this.discoveryConfig.portalUrl;

    const fetches: Promise<void>[] = [];

    if (types.includes('intents')) {
      fetches.push(this._fetchIntents(base!, headers, params.query, limit, timeout, results));
    }

    if (types.includes('agents')) {
      fetches.push(this._fetchAgents(base!, headers, params.query, limit, timeout, results));
    }

    for (const type of types) {
      if (type === 'intents' || type === 'agents') continue;
      fetches.push(this._fetchDiscoveryType(base!, headers, type, params.query, limit, timeout, results));
    }

    await Promise.all(fetches);
    return results;
  }

  // ============================================================================
  // Internal fetch helpers
  // ============================================================================

  private async _fetchIntents(
    base: string, headers: Record<string, string>,
    query: string, limit: number, timeout: number,
    results: Record<string, unknown>,
  ): Promise<void> {
    const url = `${base}/api/intents/search`;
    const body = { query, limit };
    console.log(`[search] POST ${url} body=${JSON.stringify(body)} timeout=${timeout}ms`);
    const t0 = Date.now();
    try {
      const response = await fetch(url, {
        method: 'POST', headers, body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeout),
      });
      const elapsed = Date.now() - t0;
      if (response.ok) {
        const data = await response.json();
        const intentResults = data.results || [];
        console.log(`[search] intents ${response.status} in ${elapsed}ms → ${intentResults.length} results`);
        results.intents = intentResults;
      } else {
        console.error(`[search] intents FAILED ${response.status} in ${elapsed}ms`);
      }
    } catch (err) {
      console.error('[search] Intent search error:', (err as Error).message);
    }
  }

  private async _fetchAgents(
    base: string, headers: Record<string, string>,
    query: string, limit: number, timeout: number,
    results: Record<string, unknown>,
  ): Promise<void> {
    const qs = new URLSearchParams({ search: query, type: 'agent', limit: String(limit) });
    const url = `${base}/api/discovery/agents?${qs}`;
    const t0 = Date.now();
    try {
      const response = await fetch(url, {
        method: 'GET', headers, signal: AbortSignal.timeout(timeout),
      });
      const elapsed = Date.now() - t0;
      if (response.ok) {
        const data = await response.json();
        const agentResults = data.agents || [];
        console.log(`[search] agents ${response.status} in ${elapsed}ms → ${agentResults.length} results`);
        results.agents = agentResults;
      } else {
        console.error(`[search] agents FAILED ${response.status} in ${elapsed}ms`);
      }
    } catch (err) {
      console.error('[search] Agent search error:', (err as Error).message);
    }
  }

  /**
   * Hit per-type discovery endpoints:
   * posts → /api/discovery/posts?q=xxx
   * channels → /api/discovery/channels?q=xxx
   * users → /api/discovery/users?q=xxx
   * tags → /api/discovery/tags?q=xxx
   */
  private async _fetchDiscoveryType(
    base: string, headers: Record<string, string>,
    type: string, query: string, limit: number, timeout: number,
    results: Record<string, unknown>,
  ): Promise<void> {
    const qs = new URLSearchParams({ q: query, limit: String(limit) });
    const url = `${base}/api/discovery/${type}?${qs}`;
    try {
      const response = await fetch(url, {
        method: 'GET', headers, signal: AbortSignal.timeout(timeout),
      });
      if (response.ok) {
        const data = await response.json();
        results[type] = data[type] || data.results || [];
      }
    } catch (err) {
      console.error(`[search] ${type} search error:`, (err as Error).message);
    }
  }

  // ============================================================================
  // Auto-publish
  // ============================================================================

  private async _autoPublish(): Promise<void> {
    const headers = this.buildHeaders();
    await fetch(`${this.discoveryConfig.portalUrl}/api/intents/create`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        intents: this.discoveryConfig.intents,
        description: this.discoveryConfig.description,
        capabilities: this.discoveryConfig.capabilities,
        category: this.discoveryConfig.category,
        commands: this.discoveryConfig.commands,
        agentId: this.discoveryConfig.agentId,
      }),
      signal: AbortSignal.timeout(this.discoveryConfig.timeout!),
    });
  }

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.discoveryConfig.apiKey) headers['Authorization'] = `Bearer ${this.discoveryConfig.apiKey}`;
    return headers;
  }

  private getEnvApiKey(): string | undefined {
    if (typeof window !== 'undefined') return undefined;
    if (typeof process !== 'undefined' && process.env) return process.env.WEBAGENTS_API_KEY;
    return undefined;
  }
}
