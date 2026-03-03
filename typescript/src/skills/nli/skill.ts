/**
 * NLI (Natural Language Interface) Skill
 * 
 * Enables agent-to-agent communication via natural language.
 * Can expose custom capabilities that route specific message types
 * to external agents.
 * 
 * Routing capabilities:
 * - When `capability` is specified, subscribes to that custom event type
 * - produces: ['response.delta'] - streams responses from external agents
 */

import { Skill } from '../../core/skill.js';
import { tool, handoff } from '../../core/decorators.js';
import type { ClientEvent, ServerEvent } from '../../uamp/events.js';
import type { Context, Handoff as HandoffType } from '../../core/types.js';

/**
 * NLI skill configuration
 */
export interface NLIConfig {
  /** Portal base URL for agent discovery */
  baseUrl?: string;
  /** Specific agent URL to connect to */
  agentUrl?: string;
  /** Custom capability name (e.g., 'analyze_emotion') */
  capability?: string;
  /** Request timeout in ms */
  timeout?: number;
  /** Maximum retry attempts */
  maxRetries?: number;
  /** API key for authentication */
  apiKey?: string;
}

/**
 * Chat message format
 */
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * NLI response chunk
 */
interface NLIResponseChunk {
  choices?: Array<{
    delta?: {
      content?: string;
    };
    finish_reason?: string;
  }>;
}

/**
 * NLI Skill for agent-to-agent communication
 * 
 * @example
 * ```typescript
 * // Basic usage - dynamic agent communication via tool
 * const nli = new NLISkill();
 * const response = await nli.nli('@emotion-analyzer', 'Analyze: I feel great today!');
 * ```
 * 
 * @example
 * ```typescript
 * // Custom capability routing - messages with type 'analyze_emotion' auto-route
 * const nli = new NLISkill({
 *   agentUrl: 'https://emotions.webagents.ai',
 *   capability: 'analyze_emotion',
 * });
 * agent.addSkill(nli);
 * 
 * // Now 'analyze_emotion' events are automatically handled
 * await agent.router.send({
 *   id: 'msg-1',
 *   type: 'analyze_emotion',
 *   payload: { text: 'I am feeling great today!' }
 * });
 * ```
 */
export class NLISkill extends Skill {
  private config: NLIConfig;
  private _handoffs: HandoffType[] = [];

  constructor(config: NLIConfig = {}) {
    super();
    this.config = {
      baseUrl: config.baseUrl || 'https://portal.webagents.ai',
      agentUrl: config.agentUrl,
      capability: config.capability,
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      apiKey: config.apiKey,
    };

    // If capability is specified, create a handoff for custom routing
    if (this.config.capability && this.config.agentUrl) {
      this._handoffs = [{
        name: `nli-${this.config.capability}`,
        subscribes: [this.config.capability],
        produces: ['response.delta'],
        priority: 50,
        enabled: true,
        handler: this.routeToAgent.bind(this),
      }];
    }
  }

  get id(): string {
    return `nli${this.config.capability ? `-${this.config.capability}` : ''}`;
  }

  get name(): string {
    return this.config.capability 
      ? `NLI (${this.config.capability})`
      : 'Natural Language Interface';
  }

  get description(): string {
    return this.config.capability
      ? `Route ${this.config.capability} messages to external agent`
      : 'Communicate with other WebAgents via natural language';
  }

  /**
   * Get handoffs registered by this skill
   */
  get handoffs(): HandoffType[] {
    return this._handoffs;
  }

  // ============================================================================
  // Tools
  // ============================================================================

  /**
   * Communicate with another WebAgent via natural language
   */
  @tool({
    name: 'nli',
    description: 'Communicate with other WebAgents via natural language',
    parameters: {
      agentUrl: { 
        type: 'string', 
        description: 'Agent URL or @name (e.g., "@emotion-analyzer" or "https://agent.example.com")' 
      },
      message: { 
        type: 'string', 
        description: 'Message to send to the agent' 
      },
      stream: {
        type: 'boolean',
        description: 'Whether to stream the response',
      },
    },
  })
  async nli(
    agentUrl: string,
    message: string,
    options?: { authorizedAmount?: number; stream?: boolean }
  ): Promise<string> {
    const fullUrl = this.normalizeUrl(agentUrl);
    
    if (options?.stream) {
      let result = '';
      for await (const chunk of this.streamMessage(fullUrl, [{ role: 'user', content: message }])) {
        result += chunk;
      }
      return result;
    }
    
    return this.sendMessage(fullUrl, message, options);
  }

  // ============================================================================
  // Handoff Handler
  // ============================================================================

  /**
   * Handoff handler for custom capability routing
   */
  private async *routeToAgent(
    events: ClientEvent[],
    context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      // Extract message from event payload
      const payload = event as unknown as { text?: string; content?: string };
      const message = payload.text || payload.content || '';
      
      if (message && this.config.agentUrl) {
        try {
          for await (const chunk of this.streamMessage(
            this.config.agentUrl,
            [{ role: 'user', content: message }],
            context
          )) {
            yield {
              type: 'response.delta',
              event_id: `nli-${Date.now()}`,
              delta: { text: chunk },
            } as unknown as ServerEvent;
          }
          
          yield {
            type: 'response.done',
            event_id: `nli-done-${Date.now()}`,
            response: { output: [] },
          } as unknown as ServerEvent;
        } catch (error) {
          yield {
            type: 'response.error',
            event_id: `error-${Date.now()}`,
            error: {
              type: 'nli_error',
              message: (error as Error).message,
            },
          } as unknown as ServerEvent;
        }
      }
    }
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Send a non-streaming message to an agent
   */
  private async sendMessage(
    agentUrl: string,
    message: string,
    options?: { authorizedAmount?: number }
  ): Promise<string> {
    const headers = this.buildHeaders();
    
    const response = await fetch(`${agentUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        messages: [{ role: 'user', content: message }],
        stream: false,
      }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`NLI request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  }

  /**
   * Stream messages from an agent via SSE
   */
  async *streamMessage(
    agentUrl: string,
    messages: ChatMessage[],
    context?: Context
  ): AsyncGenerator<string, void, unknown> {
    const headers = this.buildHeaders(context);

    const response = await fetch(`${agentUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ messages, stream: true }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`NLI request failed: ${response.status} ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    // Parse SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed: NLIResponseChunk = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                yield content;
              }
            } catch {
              // Ignore parse errors for malformed chunks
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Normalize agent URL
   * @param agentUrl - URL or @name format
   * @returns Full URL
   */
  private normalizeUrl(agentUrl: string): string {
    if (agentUrl.startsWith('@')) {
      const agentName = agentUrl.slice(1);
      return `${this.config.baseUrl}/agents/${agentName}`;
    }
    
    // Ensure URL has protocol
    if (!agentUrl.startsWith('http://') && !agentUrl.startsWith('https://')) {
      return `https://${agentUrl}`;
    }
    
    return agentUrl;
  }

  /**
   * Build request headers
   */
  private buildHeaders(context?: Context): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Use API key from config or context
    const apiKey = this.config.apiKey || (context?.metadata?.apiKey as string);
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    // Forward auth token from context if available
    if (context?.auth?.authenticated) {
      const token = context.metadata?.authToken as string;
      if (token) {
        headers['X-Forwarded-Auth'] = token;
      }
    }

    return headers;
  }
}
