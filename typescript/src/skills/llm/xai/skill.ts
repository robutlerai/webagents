/**
 * xAI Skill
 * 
 * Cloud LLM inference using xAI's Grok API.
 * Uses OpenAI-compatible API.
 * 
 * @see https://docs.x.ai/api
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// xAI uses OpenAI-compatible API
interface XAIClient {
  chat: {
    completions: {
      create(params: ChatCompletionParams): Promise<AsyncIterable<ChatCompletionChunk>>;
    };
  };
}

interface ChatCompletionParams {
  model: string;
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  stream_options?: { include_usage?: boolean };
}

interface ChatCompletionChunk {
  choices: Array<{
    delta: { content?: string };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

type OpenAIConstructor = new (config: { apiKey: string; baseURL: string }) => XAIClient;

/**
 * xAI skill configuration
 */
export interface XAISkillConfig extends SkillConfig {
  /** API key (defaults to XAI_API_KEY env var) */
  apiKey?: string;
  /** Model ID (e.g., 'grok-2-1212') */
  model?: string;
  /** Temperature */
  temperature?: number;
  /** Max tokens */
  max_tokens?: number;
}

/**
 * xAI Skill for Grok models
 */
export class XAISkill extends Skill {
  private client: XAIClient | null = null;
  private OpenAIClass: OpenAIConstructor | null = null;
  private modelConfig: XAISkillConfig;
  
  constructor(config: XAISkillConfig = {}) {
    super({ ...config, name: config.name || 'xai' });
    this.modelConfig = config;
  }
  
  async initialize(): Promise<void> {
    try {
      // xAI uses OpenAI SDK with custom base URL
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const openai = await import('openai' as any);
      this.OpenAIClass = openai.default as unknown as OpenAIConstructor;
      
      const apiKey = this.modelConfig.apiKey || 
        (typeof process !== 'undefined' ? process.env?.XAI_API_KEY : undefined);
      
      if (apiKey) {
        this.client = new this.OpenAIClass({
          apiKey,
          baseURL: 'https://api.x.ai/v1',
        });
      }
    } catch {
      console.warn('OpenAI SDK not available (required for xAI) - openai not installed');
    }
  }
  
  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'grok-2-1212';
    return {
      id: model,
      provider: 'xai',
      modalities: ['text'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: false,
      tools: {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: false,
        built_in_tools: [],
      },
      context_window: 131072,
    };
  }
  
  private extractMessages(events: ClientEvent[]): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          messages.push({ role: 'system', content: createEvent.session.instructions });
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        messages.push({ role: inputEvent.role || 'user', content: inputEvent.text });
      }
    }
    
    return messages;
  }
  
  @handoff({ name: 'xai', priority: 7 })
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
        throw new Error('xAI client not initialized');
      }
      
      const messages = this.extractMessages(events);
      
      if (messages.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' },
        };
        return;
      }
      
      const stream = await this.client.chat.completions.create({
        model: this.modelConfig.model || 'grok-2-1212',
        messages,
        stream: true,
        temperature: this.modelConfig.temperature ?? 0.7,
        max_tokens: this.modelConfig.max_tokens ?? 4096,
        stream_options: { include_usage: true },
      });
      
      let fullContent = '';
      let usage: UsageStats | undefined;
      
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) {
          fullContent += delta;
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: { type: 'text', text: delta },
          };
        }
        
        if (chunk.usage) {
          usage = {
            input_tokens: chunk.usage.prompt_tokens,
            output_tokens: chunk.usage.completion_tokens,
            total_tokens: chunk.usage.total_tokens,
          };
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
        error: { code: 'xai_error', message: (error as Error).message },
      };
    }
  }
}
