/**
 * Portal Discovery Skill
 * 
 * Enables intent-based agent discovery across the WebAgents network.
 * Agents can search for other agents by intent and publish their own intents.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';

/**
 * Discovery skill configuration
 */
export interface DiscoveryConfig {
  /** Portal URL for discovery API */
  portalUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Request timeout in ms */
  timeout?: number;
}

/**
 * Agent search result
 */
export interface AgentSearchResult {
  /** Agent URL */
  agentUrl: string;
  /** Agent name */
  name: string;
  /** Agent description */
  description: string;
  /** Intents this agent can handle */
  intents: string[];
  /** Relevance score (0-1) */
  score: number;
  /** Agent capabilities */
  capabilities?: string[];
}

/**
 * Published intent
 */
export interface PublishedIntent {
  /** Intent string */
  intent: string;
  /** Description of what the intent does */
  description?: string;
  /** Examples of queries that match this intent */
  examples?: string[];
}

/**
 * Portal Discovery Skill
 * 
 * @example
 * ```typescript
 * const discovery = new PortalDiscoverySkill({
 *   portalUrl: 'https://portal.webagents.ai',
 *   apiKey: process.env.WEBAGENTS_API_KEY,
 * });
 * 
 * // Search for agents that can analyze emotions
 * const results = await discovery.discoverAgents('analyze emotions in text');
 * console.log(results[0].agentUrl); // https://emotions.webagents.ai
 * ```
 * 
 * @example
 * ```typescript
 * // Publish your agent's intents
 * await discovery.publishIntents(
 *   ['analyze sentiment', 'detect emotions', 'mood analysis'],
 *   'Emotion analysis agent'
 * );
 * ```
 */
export class PortalDiscoverySkill extends Skill {
  private config: DiscoveryConfig;

  constructor(config: DiscoveryConfig = {}) {
    super();
    this.config = {
      portalUrl: config.portalUrl || 'https://portal.webagents.ai',
      apiKey: config.apiKey || this.getEnvApiKey(),
      timeout: config.timeout || 10000,
    };
  }

  get id(): string {
    return 'portal-discovery';
  }

  get name(): string {
    return 'Portal Discovery';
  }

  get description(): string {
    return 'Discover agents by intent across the WebAgents network';
  }

  // ============================================================================
  // Tools
  // ============================================================================

  /**
   * Discover agents by intent
   * 
   * Search the WebAgents network for agents that can handle a specific intent.
   */
  @tool({
    name: 'discover_agents',
    description: 'Discover agents by intent across the WebAgents network',
    parameters: {
      intent: { 
        type: 'string', 
        description: 'The intent or capability to search for (e.g., "analyze emotions", "translate text")' 
      },
      topK: { 
        type: 'number', 
        description: 'Maximum number of results to return (default: 10)' 
      },
    },
  })
  async discoverAgents(intent: string, topK: number = 10): Promise<AgentSearchResult[]> {
    const headers = this.buildHeaders();

    const response = await fetch(`${this.config.portalUrl}/api/intents/search`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ intent, top_k: topK }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Unauthorized: Invalid or missing API key');
      }
      throw new Error(`Discovery request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data.results || [];
  }

  /**
   * Publish agent intents to the platform
   * 
   * Register your agent's capabilities so other agents can discover it.
   */
  @tool({
    name: 'publish_intents',
    description: 'Publish agent intents to the platform for discovery',
    parameters: {
      intents: { 
        type: 'array', 
        items: { type: 'string' },
        description: 'List of intents this agent can handle' 
      },
      description: { 
        type: 'string', 
        description: 'Description of the agent capabilities' 
      },
    },
  })
  async publishIntents(intents: string[], description: string): Promise<void> {
    const headers = this.buildHeaders();

    const response = await fetch(`${this.config.portalUrl}/api/intents/publish`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ intents, description }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Unauthorized: Invalid or missing API key');
      }
      throw new Error(`Publish request failed: ${response.status} ${response.statusText}`);
    }
  }

  /**
   * Get agent details by URL
   * 
   * Fetch detailed information about a specific agent.
   */
  @tool({
    name: 'get_agent_info',
    description: 'Get detailed information about a specific agent',
    parameters: {
      agentUrl: { 
        type: 'string', 
        description: 'The agent URL or @name' 
      },
    },
  })
  async getAgentInfo(agentUrl: string): Promise<AgentSearchResult | null> {
    const normalizedUrl = this.normalizeUrl(agentUrl);
    const headers = this.buildHeaders();

    const response = await fetch(`${this.config.portalUrl}/api/agents/info`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ agent_url: normalizedUrl }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      if (response.status === 404) {
        return null;
      }
      throw new Error(`Agent info request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * List all registered agents (paginated)
   */
  @tool({
    name: 'list_agents',
    description: 'List all registered agents on the platform',
    parameters: {
      page: { 
        type: 'number', 
        description: 'Page number (default: 1)' 
      },
      limit: { 
        type: 'number', 
        description: 'Results per page (default: 20)' 
      },
      category: { 
        type: 'string', 
        description: 'Filter by category (optional)' 
      },
    },
  })
  async listAgents(
    page: number = 1, 
    limit: number = 20, 
    category?: string
  ): Promise<{ agents: AgentSearchResult[]; total: number; page: number }> {
    const headers = this.buildHeaders();
    const params = new URLSearchParams({
      page: String(page),
      limit: String(limit),
    });
    
    if (category) {
      params.set('category', category);
    }

    const response = await fetch(
      `${this.config.portalUrl}/api/agents?${params}`,
      {
        method: 'GET',
        headers,
        signal: AbortSignal.timeout(this.config.timeout!),
      }
    );

    if (!response.ok) {
      throw new Error(`List agents request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Build request headers
   */
  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    return headers;
  }

  /**
   * Normalize agent URL
   */
  private normalizeUrl(agentUrl: string): string {
    if (agentUrl.startsWith('@')) {
      const agentName = agentUrl.slice(1);
      return `${this.config.portalUrl}/agents/${agentName}`;
    }
    return agentUrl;
  }

  /**
   * Get API key from environment
   */
  private getEnvApiKey(): string | undefined {
    // Browser environment
    if (typeof window !== 'undefined') {
      return undefined;
    }
    
    // Node.js environment
    if (typeof process !== 'undefined' && process.env) {
      return process.env.WEBAGENTS_API_KEY;
    }
    
    return undefined;
  }
}
