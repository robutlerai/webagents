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
  messages: Array<{ role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  tools?: Array<{ type: 'function'; function: { name: string; description?: string; parameters?: unknown } }>;
  stream_options?: { include_usage?: boolean };
}

interface ChatCompletionChunk {
  choices: Array<{
    delta: {
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
  
  private extractMessages(
    events: ClientEvent[],
    context?: Context,
  ): Array<{ role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }> {
    if (context?.get) {
      const agenticMessages = context.get<Array<{
        role: string; content: string | null;
        tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
        tool_call_id?: string;
      }>>('_agentic_messages');
      if (agenticMessages && agenticMessages.length > 0) return agenticMessages;
    }

    const messages: Array<{ role: string; content: string | null }> = [];
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
            function: { name: t.function.name, description: t.function.description, parameters: t.function.parameters },
          }));
        }
      }
    }
    return undefined;
  }
  
  @handoff({ name: 'xai', priority: 7 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context
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
      
      const messages = this.extractMessages(events, context);

      // Prefer tools from context (agentic loop), fall back to session.create
      let tools = this.extractTools(events);
      const contextTools = context?.get ? context.get<Array<{
        type: string; function: { name: string; description?: string; parameters?: unknown };
      }>>('_agentic_tools') : undefined;
      if (contextTools && contextTools.length > 0) {
        tools = contextTools.map(t => ({
          type: 'function' as const,
          function: { name: t.function.name, description: t.function.description, parameters: t.function.parameters },
        }));
      }
      
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
        tools,
        stream_options: { include_usage: true },
      });
      
      let fullContent = '';
      let usage: UsageStats | undefined;
      const toolCallDeltas: Map<number, { id: string; name: string; arguments: string }> = new Map();
      
      for await (const chunk of stream) {
        const choice = chunk.choices[0];
        if (!choice) continue;

        const delta = choice.delta?.content;
        if (delta) {
          fullContent += delta;
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: { type: 'text', text: delta },
          };
        }

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

            yield {
              type: 'response.delta',
              event_id: generateEventId(),
              response_id: responseId,
              delta: {
                type: 'tool_call',
                tool_call: { id: existing.id, name: existing.name, arguments: existing.arguments },
              },
            };
          }
        }
        
        if (chunk.usage) {
          usage = {
            input_tokens: chunk.usage.prompt_tokens,
            output_tokens: chunk.usage.completion_tokens,
            total_tokens: chunk.usage.total_tokens,
          };
        }
      }

      const toolCalls = [...toolCallDeltas.values()].filter(tc => tc.id && tc.name);
      const output: ContentItem[] = [];
      if (fullContent) output.push({ type: 'text', text: fullContent });
      for (const tc of toolCalls) {
        output.push({ type: 'tool_call', tool_call: { id: tc.id, name: tc.name, arguments: tc.arguments } });
      }
      
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
