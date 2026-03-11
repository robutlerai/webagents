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

// Anthropic client types
type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean };

interface AnthropicToolDef {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

interface AnthropicClient {
  messages: {
    stream(params: MessageParams): AsyncIterable<MessageStreamEvent>;
  };
}

interface MessageParams {
  model: string;
  messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }>;
  system?: string;
  max_tokens: number;
  temperature?: number;
  stream: boolean;
  tools?: AnthropicToolDef[];
}

interface MessageStreamEvent {
  type: string;
  index?: number;
  content_block?: AnthropicContentBlock;
  delta?: { type: string; text?: string; partial_json?: string };
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
  
  private extractToolDefinitions(events: ClientEvent[]): AnthropicToolDef[] {
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.tools && createEvent.session.tools.length > 0) {
          return createEvent.session.tools.map(t => ({
            name: t.function.name,
            description: t.function.description,
            input_schema: (t.function.parameters || { type: 'object', properties: {} }) as Record<string, unknown>,
          }));
        }
      }
    }
    return [];
  }

  private extractSystemAndMessages(events: ClientEvent[]): {
    system?: string;
    messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }>;
  } {
    let system: string | undefined;
    const messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }> = [];
    
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

  /**
   * Convert agentic messages (with tool_calls/tool_results) to Anthropic format.
   */
  private agenticToAnthropicMessages(
    agenticMessages: Array<{
      role: string;
      content: string | null;
      tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
      tool_call_id?: string;
    }>
  ): { system?: string; messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }> } {
    let system: string | undefined;
    const messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }> = [];

    for (const msg of agenticMessages) {
      if (msg.role === 'system') {
        system = (system ? system + '\n\n' : '') + (msg.content || '');
      } else if (msg.role === 'assistant') {
        const blocks: AnthropicContentBlock[] = [];
        if (msg.content) blocks.push({ type: 'text', text: msg.content });
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            let input: Record<string, unknown> = {};
            try { input = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
            blocks.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
          }
        }
        messages.push({ role: 'assistant', content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0] as { text: string }).text : blocks });
      } else if (msg.role === 'tool') {
        messages.push({
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: msg.tool_call_id || '', content: msg.content || '' }],
        });
      } else {
        messages.push({ role: 'user', content: msg.content || '' });
      }
    }

    return { system, messages };
  }
  
  @handoff({ name: 'anthropic', priority: 9 })
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
        throw new Error('Anthropic client not initialized');
      }
      
      let system: string | undefined;
      let messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }>;

      const agenticMessages = context?.get ? context.get<Array<{
        role: string;
        content: string | null;
        tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
        tool_call_id?: string;
      }>>('_agentic_messages') : undefined;

      if (agenticMessages && agenticMessages.length > 0) {
        ({ system, messages } = this.agenticToAnthropicMessages(agenticMessages));
      } else {
        ({ system, messages } = this.extractSystemAndMessages(events));
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

      // Extract tools from session.create or context
      let toolDefs = this.extractToolDefinitions(events);
      const contextTools = context?.get ? context.get<Array<{
        type: string; function: { name: string; description?: string; parameters?: Record<string, unknown> };
      }>>('_agentic_tools') : undefined;
      if (contextTools && contextTools.length > 0) {
        toolDefs = contextTools.map(t => ({
          name: t.function.name,
          description: t.function.description,
          input_schema: (t.function.parameters || { type: 'object', properties: {} }) as Record<string, unknown>,
        }));
      }

      const params: MessageParams = {
        model: this.modelConfig.model || 'claude-3-5-sonnet-20241022',
        messages,
        system,
        max_tokens: this.modelConfig.max_tokens ?? 4096,
        temperature: this.modelConfig.temperature ?? 0.7,
        stream: true,
      };
      if (toolDefs.length > 0) params.tools = toolDefs;
      
      const stream = this.client.messages.stream(params);
      
      let fullContent = '';
      let usage: UsageStats | undefined;
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      let currentToolUse: { id: string; name: string; jsonBuf: string } | null = null;
      
      for await (const event of stream) {
        if (event.type === 'content_block_start' && event.content_block?.type === 'tool_use') {
          const block = event.content_block as { type: 'tool_use'; id: string; name: string };
          currentToolUse = { id: block.id, name: block.name, jsonBuf: '' };
        }

        if (event.type === 'content_block_delta') {
          if (event.delta?.type === 'text_delta' && event.delta?.text) {
            fullContent += event.delta.text;
            yield {
              type: 'response.delta',
              event_id: generateEventId(),
              response_id: responseId,
              delta: { type: 'text', text: event.delta.text },
            };
          } else if (event.delta?.type === 'input_json_delta' && event.delta?.partial_json && currentToolUse) {
            currentToolUse.jsonBuf += event.delta.partial_json;
          }
        }

        if (event.type === 'content_block_stop' && currentToolUse) {
          toolCalls.push({
            id: currentToolUse.id,
            name: currentToolUse.name,
            arguments: currentToolUse.jsonBuf || '{}',
          });
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: {
              type: 'tool_call',
              tool_call: { id: currentToolUse.id, name: currentToolUse.name, arguments: currentToolUse.jsonBuf || '{}' },
            },
          };
          currentToolUse = null;
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
        error: { code: 'anthropic_error', message: (error as Error).message },
      };
    }
  }
}
