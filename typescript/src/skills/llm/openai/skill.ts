/**
 * OpenAI Skill
 *
 * Cloud LLM inference using OpenAI's API via the shared OpenAIAdapter.
 * Uses raw fetch + SSE (no SDK dependency).
 *
 * @see https://platform.openai.com/docs/api-reference
 */

import { Skill } from '../../../core/skill';
import { handoff } from '../../../core/decorators';
import type { SkillConfig, Context } from '../../../core/types';
import type { Capabilities, ContentItem, FunctionToolDefinition, UsageStats } from '../../../uamp/types';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events';
import { generateEventId } from '../../../uamp/events';
import { openaiAdapter, createOpenAICompatibleAdapter } from '../../../adapters/openai';
import type { LLMAdapter, AdapterChunk, Message, ToolDefinition, UAMPUsage } from '../../../adapters/types';

export interface OpenAISkillConfig extends SkillConfig {
  apiKey?: string;
  model?: string;
  baseURL?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}

export class OpenAISkill extends Skill {
  private modelConfig: OpenAISkillConfig;
  private adapter: LLMAdapter;

  constructor(config: OpenAISkillConfig = {}) {
    super({ ...config, name: config.name || 'openai' });
    this.modelConfig = config;
    this.adapter = config.baseURL
      ? createOpenAICompatibleAdapter({ name: 'openai', baseUrl: config.baseURL })
      : openaiAdapter;
  }

  private get apiKey(): string | undefined {
    return this.modelConfig.apiKey
      || (typeof process !== 'undefined' ? process.env?.OPENAI_API_KEY : undefined);
  }

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

  @handoff({ name: 'openai', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId };

    try {
      const key = this.apiKey;
      if (!key) throw new Error('OpenAI API key not configured');

      const { messages, tools } = extractInput(events, context);
      if (messages.length === 0) {
        yield { type: 'response.error', event_id: generateEventId(), response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' } };
        return;
      }

      const model = this.modelConfig.model || 'gpt-4o';

      context.set?.('_llm_capabilities', {
        model,
        provider: 'openai',
        maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        pricing: { inputPer1k: 0, outputPer1k: 0 },
      });

      const request = this.adapter.buildRequest({
        messages,
        model,
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
        throw new Error(`OpenAI API returned ${response.status}: ${errorText.slice(0, 200)}`);
      }

      let fullContent = '';
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      let usageInput = 0;
      let usageOutput = 0;

      for await (const chunk of this.adapter.parseStream(response)) {
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
        model,
        provider: 'openai',
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
        error: { code: 'openai_error', message: (error as Error).message },
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
