/**
 * Fireworks AI Skill
 *
 * Cloud LLM inference via the Fireworks AI platform.
 * Uses the OpenAI-compatible API (same SDK, different base URL).
 * Supports deepseek, glm, kimi, minimax, qwen, cogito, llama, and more.
 *
 * @see https://docs.fireworks.ai/api-reference
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

interface FireworksClient {
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

type OpenAIConstructor = new (config: { apiKey: string; baseURL: string }) => FireworksClient;

export interface FireworksModelDef {
  name: string;
  maxOutputTokens: number;
  supportsTools: boolean;
  supportsVision: boolean;
  contextWindow: number;
}

const DEFAULT_MODELS: Record<string, FireworksModelDef> = {
  'deepseek-v3p2':             { name: 'deepseek-v3p2', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 163840 },
  'deepseek-v3p1':             { name: 'deepseek-v3p1', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 163840 },
  'glm-5':                     { name: 'glm-5', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 202752 },
  'glm-4p7':                   { name: 'glm-4p7', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'kimi-k2p5':                 { name: 'kimi-k2p5', maxOutputTokens: 131072, supportsTools: true, supportsVision: true, contextWindow: 262144 },
  'kimi-k2-thinking':          { name: 'kimi-k2-thinking', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'kimi-k2-instruct-0905':     { name: 'kimi-k2-instruct-0905', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'minimax-m2p5':              { name: 'minimax-m2p5', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 196608 },
  'minimax-m2p1':              { name: 'minimax-m2p1', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'gpt-oss-120b':              { name: 'gpt-oss-120b', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'gpt-oss-20b':               { name: 'gpt-oss-20b', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'llama-v3p3-70b-instruct':   { name: 'llama-v3p3-70b-instruct', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'qwen3-8b':                  { name: 'qwen3-8b', maxOutputTokens: 32768, supportsTools: true, supportsVision: false, contextWindow: 131072 },
  'qwen3-vl-30b-a3b-thinking': { name: 'qwen3-vl-30b-a3b-thinking', maxOutputTokens: 131072, supportsTools: true, supportsVision: true, contextWindow: 131072 },
  'qwen3-vl-30b-a3b-instruct': { name: 'qwen3-vl-30b-a3b-instruct', maxOutputTokens: 131072, supportsTools: true, supportsVision: true, contextWindow: 131072 },
  'cogito-671b-v2':            { name: 'cogito-671b-v2', maxOutputTokens: 131072, supportsTools: true, supportsVision: false, contextWindow: 131072 },
};

export interface FireworksSkillConfig extends SkillConfig {
  apiKey?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  baseUrl?: string;
}

export class FireworksSkill extends Skill {
  private client: FireworksClient | null = null;
  private OpenAIClass: OpenAIConstructor | null = null;
  private modelConfig: FireworksSkillConfig;
  static readonly BASE_URL = 'https://api.fireworks.ai/inference/v1';
  static readonly DEFAULT_MODELS = DEFAULT_MODELS;

  constructor(config: FireworksSkillConfig = {}) {
    super({ ...config, name: config.name || 'fireworks' });
    this.modelConfig = config;
  }

  async initialize(): Promise<void> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const openai = await import('openai' as any);
      this.OpenAIClass = openai.default as unknown as OpenAIConstructor;

      const apiKey = this.modelConfig.apiKey ||
        (typeof process !== 'undefined' ? process.env?.FIREWORKS_API_KEY : undefined);

      if (apiKey) {
        this.client = new this.OpenAIClass({
          apiKey,
          baseURL: this.modelConfig.baseUrl || FireworksSkill.BASE_URL,
        });
      }
    } catch {
      console.warn('OpenAI SDK not available (required for Fireworks AI) - openai not installed');
    }
  }

  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'deepseek-v3p2';
    const modelDef = DEFAULT_MODELS[model];
    return {
      id: model,
      provider: 'fireworks',
      modalities: modelDef?.supportsVision ? ['text', 'image'] : ['text'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: false,
      tools: {
        supports_tools: modelDef?.supportsTools ?? true,
        supports_parallel_tools: true,
        supports_streaming_tools: false,
        built_in_tools: [],
      },
      context_window: modelDef?.contextWindow ?? 131072,
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

  @handoff({ name: 'fireworks', priority: 6 })
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
        throw new Error('Fireworks AI client not initialized — set FIREWORKS_API_KEY');
      }

      const messages = this.extractMessages(events, context);

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

      const modelShort = this.modelConfig.model || 'deepseek-v3p2';
      const fireworksModel = `accounts/fireworks/models/${modelShort}`;

      const stream = await this.client.chat.completions.create({
        model: fireworksModel,
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

          // Append usage record for PaymentSkill
          const usageRecords = context.get<Array<Record<string, unknown>>>('usage') ?? [];
          usageRecords.push({
            type: 'llm',
            model: `fireworks/${modelShort}`,
            prompt_tokens: chunk.usage.prompt_tokens,
            completion_tokens: chunk.usage.completion_tokens,
          });
          context.set('usage', usageRecords);
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
        error: { code: 'fireworks_error', message: (error as Error).message },
      };
    }
  }
}
