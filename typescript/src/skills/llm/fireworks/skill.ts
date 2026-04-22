/**
 * Fireworks AI Skill
 *
 * Cloud LLM inference via Fireworks AI using the shared OpenAI-compatible adapter.
 * Uses raw fetch + SSE (no SDK dependency).
 * Supports deepseek, glm, kimi, minimax, qwen, cogito, llama, and more.
 *
 * @see https://docs.fireworks.ai/api-reference
 */

import { Skill } from '../../../core/skill';
import { handoff } from '../../../core/decorators';
import type { SkillConfig, Context } from '../../../core/types';
import type { Capabilities, ContentItem, FunctionToolDefinition, UsageStats } from '../../../uamp/types';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events';
import { generateEventId } from '../../../uamp/events';
import { fireworksAdapter } from '../../../adapters/openai';
import type { AdapterChunk, Message, ToolDefinition, UAMPUsage } from '../../../adapters/types';

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
  'kimi-k2p6':                 { name: 'kimi-k2p6', maxOutputTokens: 131072, supportsTools: true, supportsVision: true, contextWindow: 262144 },
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
  private modelConfig: FireworksSkillConfig;
  static readonly BASE_URL = 'https://api.fireworks.ai/inference/v1';
  static readonly DEFAULT_MODELS = DEFAULT_MODELS;

  constructor(config: FireworksSkillConfig = {}) {
    super({ ...config, name: config.name || 'fireworks' });
    this.modelConfig = config;
  }

  private get apiKey(): string | undefined {
    return this.modelConfig.apiKey
      || (typeof process !== 'undefined' ? process.env?.FIREWORKS_API_KEY : undefined);
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

  @handoff({ name: 'fireworks', priority: 6 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId };

    try {
      const key = this.apiKey;
      if (!key) throw new Error('Fireworks API key not configured — set FIREWORKS_API_KEY');

      const { messages, tools } = extractInput(events, context);
      if (messages.length === 0) {
        yield { type: 'response.error', event_id: generateEventId(), response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' } };
        return;
      }

      const modelShort = this.modelConfig.model || 'deepseek-v3p2';
      const fireworksModel = `accounts/fireworks/models/${modelShort}`;

      context.set?.('_llm_capabilities', {
        model: fireworksModel,
        provider: 'fireworks',
        maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        pricing: { inputPer1k: 0, outputPer1k: 0 },
      });

      const request = fireworksAdapter.buildRequest({
        messages,
        model: fireworksModel,
        tools: tools.length > 0 ? tools : undefined,
        temperature: this.modelConfig.temperature ?? 0.7,
        maxTokens: this.modelConfig.max_tokens ?? 4096,
        apiKey: key,
      });

      const response = await fetch(request.url, {
        method: 'POST',
        headers: request.headers,
        body: request.body,
        signal: context.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Fireworks API returned ${response.status}: ${errorText.slice(0, 200)}`);
      }

      let fullContent = '';
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      let usageInput = 0;
      let usageOutput = 0;

      for await (const chunk of fireworksAdapter.parseStream(response)) {
        const event = chunkToEvent(responseId, chunk);
        if (event) yield event;

        if (chunk.type === 'text') fullContent += chunk.text;
        if (chunk.type === 'tool_call') toolCalls.push({ id: chunk.id, name: chunk.name, arguments: chunk.arguments });
        if (chunk.type === 'usage') { usageInput = chunk.input; usageOutput = chunk.output; }
      }

      const usage: UsageStats = {
        input_tokens: usageInput,
        output_tokens: usageOutput,
        total_tokens: usageInput + usageOutput,
      };

      context.set?.('_llm_usage', {
        model: fireworksModel,
        provider: 'fireworks',
        input_tokens: usageInput,
        output_tokens: usageOutput,
        is_byok: false,
      } satisfies UAMPUsage);

      const output: ContentItem[] = [];
      if (fullContent) output.push({ type: 'text', text: fullContent });
      for (const tc of toolCalls) {
        output.push({ type: 'tool_call', tool_call: tc });
      }

      yield {
        type: 'response.done', event_id: generateEventId(), response_id: responseId,
        response: { id: responseId, status: 'completed', output, usage },
      };
    } catch (error) {
      yield {
        type: 'response.error', event_id: generateEventId(), response_id: responseId,
        error: { code: 'fireworks_error', message: (error as Error).message },
      };
    }
  }
}

function extractInput(events: ClientEvent[], context: Context): {
  messages: Message[];
  tools: ToolDefinition[];
} {
  const agenticMessages = context?.get ? context.get<Message[]>('_agentic_messages') : undefined;

  if (agenticMessages && agenticMessages.length > 0) {
    const contextTools = context?.get ? context.get<ToolDefinition[]>('_agentic_tools') ?? [] : [];
    return { messages: agenticMessages, tools: contextTools };
  }

  const messages: Message[] = [];
  let tools: ToolDefinition[] = [];

  for (const event of events) {
    if (event.type === 'session.create') {
      const e = event as SessionCreateEvent;
      if (e.session.instructions) messages.push({ role: 'system', content: e.session.instructions });
      if (e.session.tools && e.session.tools.length > 0) {
        tools = e.session.tools
          .filter((t): t is FunctionToolDefinition => t.type === 'function')
          .map(t => ({
            type: 'function' as const,
            function: { name: t.function.name, description: t.function.description, parameters: t.function.parameters },
          }));
      }
    } else if (event.type === 'input.text') {
      const e = event as InputTextEvent;
      messages.push({ role: (e as { role?: string }).role || 'user', content: e.text });
    }
  }

  return { messages, tools };
}

function chunkToEvent(responseId: string, chunk: AdapterChunk): ServerEvent | null {
  if (chunk.type === 'text') {
    return {
      type: 'response.delta', event_id: generateEventId(), response_id: responseId,
      delta: { type: 'text', text: chunk.text },
    };
  }
  if (chunk.type === 'tool_call') {
    return {
      type: 'response.delta', event_id: generateEventId(), response_id: responseId,
      delta: { type: 'tool_call', tool_call: { id: chunk.id, name: chunk.name, arguments: chunk.arguments } },
    };
  }
  return null;
}
