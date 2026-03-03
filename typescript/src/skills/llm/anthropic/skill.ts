/**
 * Anthropic Skill
 * 
 * Cloud LLM inference using Anthropic's Claude API.
 * 
 * @see https://docs.anthropic.com/en/api
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// Anthropic client type (simplified)
interface AnthropicClient {
  messages: {
    stream(params: MessageParams): AsyncIterable<MessageStreamEvent>;
  };
}

interface MessageParams {
  model: string;
  messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  system?: string;
  max_tokens: number;
  temperature?: number;
  stream: boolean;
}

interface MessageStreamEvent {
  type: string;
  delta?: { type: string; text?: string };
  message?: { usage?: { input_tokens: number; output_tokens: number } };
  usage?: { output_tokens: number };
}

type AnthropicConstructor = new (config: { apiKey: string }) => AnthropicClient;

/**
 * Anthropic skill configuration
 */
export interface AnthropicSkillConfig extends SkillConfig {
  /** API key (defaults to ANTHROPIC_API_KEY env var) */
  apiKey?: string;
  /** Model ID (e.g., 'claude-3-5-sonnet-20241022') */
  model?: string;
  /** Temperature for generation */
  temperature?: number;
  /** Max tokens to generate */
  max_tokens?: number;
}

/**
 * Anthropic Skill for Claude models
 */
export class AnthropicSkill extends Skill {
  private client: AnthropicClient | null = null;
  private AnthropicClass: AnthropicConstructor | null = null;
  private modelConfig: AnthropicSkillConfig;
  
  constructor(config: AnthropicSkillConfig = {}) {
    super({ ...config, name: config.name || 'anthropic' });
    this.modelConfig = config;
  }
  
  async initialize(): Promise<void> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const anthropic = await import('@anthropic-ai/sdk' as any);
      this.AnthropicClass = anthropic.default as unknown as AnthropicConstructor;
      
      const apiKey = this.modelConfig.apiKey || 
        (typeof process !== 'undefined' ? process.env?.ANTHROPIC_API_KEY : undefined);
      
      if (apiKey) {
        this.client = new this.AnthropicClass({ apiKey });
      }
    } catch {
      console.warn('Anthropic SDK not available - @anthropic-ai/sdk not installed');
    }
  }
  
  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'claude-3-5-sonnet-20241022';
    return {
      id: model,
      provider: 'anthropic',
      modalities: ['text', 'image'],
      supports_streaming: true,
      supports_thinking: model.includes('opus'),
      supports_caching: true,
      tools: {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: true,
        built_in_tools: [],
      },
      context_window: 200000,
    };
  }
  
  private extractSystemAndMessages(events: ClientEvent[]): {
    system?: string;
    messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  } {
    let system: string | undefined;
    const messages: Array<{ role: 'user' | 'assistant'; content: string }> = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          system = createEvent.session.instructions;
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        if (inputEvent.role === 'system') {
          system = (system ? system + '\n\n' : '') + inputEvent.text;
        } else {
          messages.push({ role: 'user', content: inputEvent.text });
        }
      }
    }
    
    return { system, messages };
  }
  
  @handoff({ name: 'anthropic', priority: 9 })
  async *processUAMP(
    events: ClientEvent[],
    _context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();
    
    yield {
      type: 'response.created',
      event_id: generateEventId(),
      response_id: responseId,
    };
    
    try {
      if (!this.client) {
        throw new Error('Anthropic client not initialized');
      }
      
      const { system, messages } = this.extractSystemAndMessages(events);
      
      if (messages.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' },
        };
        return;
      }
      
      const stream = this.client.messages.stream({
        model: this.modelConfig.model || 'claude-3-5-sonnet-20241022',
        messages,
        system,
        max_tokens: this.modelConfig.max_tokens ?? 4096,
        temperature: this.modelConfig.temperature ?? 0.7,
        stream: true,
      });
      
      let fullContent = '';
      let usage: UsageStats | undefined;
      
      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta?.text) {
          fullContent += event.delta.text;
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: { type: 'text', text: event.delta.text },
          };
        }
        
        if (event.type === 'message_start' && event.message?.usage) {
          usage = {
            input_tokens: event.message.usage.input_tokens,
            output_tokens: 0,
            total_tokens: event.message.usage.input_tokens,
          };
        }
        
        if (event.type === 'message_delta' && event.usage) {
          if (usage) {
            usage.output_tokens = event.usage.output_tokens;
            usage.total_tokens = usage.input_tokens + event.usage.output_tokens;
          }
        }
      }
      
      const output: ContentItem[] = fullContent ? [{ type: 'text', text: fullContent }] : [];
      
      yield {
        type: 'response.done',
        event_id: generateEventId(),
        response_id: responseId,
        response: { id: responseId, status: 'completed', output, usage },
      };
    } catch (error) {
      yield {
        type: 'response.error',
        event_id: generateEventId(),
        response_id: responseId,
        error: { code: 'anthropic_error', message: (error as Error).message },
      };
    }
  }
}
