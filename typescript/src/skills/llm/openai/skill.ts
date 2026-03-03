/**
 * OpenAI Skill
 * 
 * Cloud LLM inference using OpenAI's API.
 * 
 * @see https://platform.openai.com/docs/api-reference
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats, ToolCall } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// Types for openai (peer dependency)
interface OpenAI {
  chat: {
    completions: {
      create(params: ChatCompletionParams): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
    };
  };
}

interface ChatCompletionParams {
  model: string;
  messages: Array<{ role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: Array<{
    type: 'function';
    function: { name: string; description?: string; parameters?: unknown };
  }>;
  stream_options?: { include_usage?: boolean };
}

interface ChatCompletion {
  id: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ChatCompletionChunk {
  id: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

type OpenAIConstructor = new (config: { apiKey: string; baseURL?: string }) => OpenAI;

/**
 * OpenAI skill configuration
 */
export interface OpenAISkillConfig extends SkillConfig {
  /** API key (defaults to OPENAI_API_KEY env var) */
  apiKey?: string;
  /** Model ID (e.g., 'gpt-4o', 'gpt-4o-mini') */
  model?: string;
  /** Base URL for API (for proxies or compatible APIs) */
  baseURL?: string;
  /** Temperature for generation */
  temperature?: number;
  /** Max tokens to generate */
  max_tokens?: number;
  /** Top-p sampling */
  top_p?: number;
}

/**
 * OpenAI Skill for cloud LLM inference
 * 
 * @example
 * ```typescript
 * import { OpenAISkill } from 'webagents/skills/llm/openai';
 * 
 * const skill = new OpenAISkill({
 *   model: 'gpt-4o',
 *   apiKey: process.env.OPENAI_API_KEY
 * });
 * 
 * const agent = new BaseAgent({ skills: [skill] });
 * ```
 */
export class OpenAISkill extends Skill {
  private client: OpenAI | null = null;
  private OpenAIClass: OpenAIConstructor | null = null;
  private modelConfig: OpenAISkillConfig;
  
  constructor(config: OpenAISkillConfig = {}) {
    super({ ...config, name: config.name || 'openai' });
    this.modelConfig = config;
  }
  
  /**
   * Initialize the skill - loads the OpenAI library
   */
  async initialize(): Promise<void> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const openai = await import('openai' as any);
      this.OpenAIClass = openai.default as unknown as OpenAIConstructor;
      
      const apiKey = this.modelConfig.apiKey || 
        (typeof process !== 'undefined' ? process.env?.OPENAI_API_KEY : undefined);
      
      if (apiKey) {
        this.client = new this.OpenAIClass({
          apiKey,
          baseURL: this.modelConfig.baseURL,
        });
      }
    } catch {
      console.warn('OpenAI SDK not available - openai not installed');
    }
  }
  
  /**
   * Get OpenAI capabilities
   */
  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'gpt-4o';
    const isMultimodal = model.includes('gpt-4') || model.includes('vision');
    
    return {
      id: model,
      provider: 'openai',
      modalities: isMultimodal ? ['text', 'image'] : ['text'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: false,
      tools: {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: true,
        built_in_tools: [],
      },
      context_window: model.includes('gpt-4o') ? 128000 : 16384,
    };
  }
  
  /**
   * Ensure client is initialized
   */
  private ensureClient(): OpenAI {
    if (!this.client) {
      throw new Error('OpenAI client not initialized - call initialize() first or provide apiKey');
    }
    return this.client;
  }
  
  /**
   * Extract messages from UAMP events
   */
  private extractMessages(events: ClientEvent[]): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          messages.push({
            role: 'system',
            content: createEvent.session.instructions,
          });
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        messages.push({
          role: inputEvent.role || 'user',
          content: inputEvent.text,
        });
      }
    }
    
    return messages;
  }
  
  /**
   * Extract tools from session create event
   */
  private extractTools(events: ClientEvent[]): Array<{
    type: 'function';
    function: { name: string; description?: string; parameters?: unknown };
  }> | undefined {
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.tools && createEvent.session.tools.length > 0) {
          return createEvent.session.tools.map(t => ({
            type: 'function' as const,
            function: {
              name: t.function.name,
              description: t.function.description,
              parameters: t.function.parameters,
            },
          }));
        }
      }
    }
    return undefined;
  }
  
  /**
   * Process UAMP events with OpenAI
   */
  @handoff({ name: 'openai', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    _context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();
    
    // Yield response started
    yield {
      type: 'response.created',
      event_id: generateEventId(),
      response_id: responseId,
    };
    
    try {
      const client = this.ensureClient();
      
      // Extract messages and tools
      const messages = this.extractMessages(events);
      const tools = this.extractTools(events);
      
      if (messages.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: {
            code: 'no_input',
            message: 'No input messages provided',
          },
        };
        return;
      }
      
      // Call OpenAI with streaming
      const stream = await client.chat.completions.create({
        model: this.modelConfig.model || 'gpt-4o',
        messages,
        stream: true,
        temperature: this.modelConfig.temperature ?? 0.7,
        max_tokens: this.modelConfig.max_tokens ?? 4096,
        top_p: this.modelConfig.top_p,
        tools: tools,
        stream_options: { include_usage: true },
      });
      
      let fullContent = '';
      let toolCalls: ToolCall[] = [];
      let usage: UsageStats | undefined;
      
      // Track tool call deltas
      const toolCallDeltas: Map<number, { id: string; name: string; arguments: string }> = new Map();
      
      // Stream response chunks
      for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
        const choice = chunk.choices[0];
        if (!choice) continue;
        
        // Handle text content
        const delta = choice.delta?.content;
        if (delta) {
          fullContent += delta;
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: {
              type: 'text',
              text: delta,
            },
          };
        }
        
        // Handle tool calls
        if (choice.delta?.tool_calls) {
          for (const tc of choice.delta.tool_calls) {
            let existing = toolCallDeltas.get(tc.index);
            if (!existing) {
              existing = { id: tc.id || '', name: '', arguments: '' };
              toolCallDeltas.set(tc.index, existing);
            }
            
            if (tc.id) existing.id = tc.id;
            if (tc.function?.name) existing.name = tc.function.name;
            if (tc.function?.arguments) existing.arguments += tc.function.arguments;
            
            // Yield tool call delta
            yield {
              type: 'response.delta',
              event_id: generateEventId(),
              response_id: responseId,
              delta: {
                type: 'tool_call',
                tool_call: {
                  id: existing.id,
                  name: existing.name,
                  arguments: existing.arguments,
                },
              },
            };
          }
        }
        
        // Capture usage from last chunk
        if (chunk.usage) {
          usage = {
            input_tokens: chunk.usage.prompt_tokens,
            output_tokens: chunk.usage.completion_tokens,
            total_tokens: chunk.usage.total_tokens,
          };
        }
      }
      
      // Collect final tool calls
      for (const tc of toolCallDeltas.values()) {
        if (tc.id && tc.name) {
          toolCalls.push(tc);
        }
      }
      
      // Build output
      const output: ContentItem[] = [];
      if (fullContent) {
        output.push({ type: 'text', text: fullContent });
      }
      for (const tc of toolCalls) {
        output.push({
          type: 'tool_call',
          tool_call: tc,
        });
      }
      
      // Yield response done
      yield {
        type: 'response.done',
        event_id: generateEventId(),
        response_id: responseId,
        response: {
          id: responseId,
          status: 'completed',
          output,
          usage,
        },
      };
    } catch (error) {
      yield {
        type: 'response.error',
        event_id: generateEventId(),
        response_id: responseId,
        error: {
          code: 'openai_error',
          message: (error as Error).message,
        },
      };
    }
  }
}
