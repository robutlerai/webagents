/**
 * Anthropic Skill
 *
 * Cloud LLM inference using Anthropic's Claude API via the shared AnthropicAdapter.
 * Uses raw fetch + SSE (no SDK dependency).
 *
 * @see https://docs.anthropic.com/en/api
 */

import { Skill } from '../../../core/skill';
import { handoff } from '../../../core/decorators';
import type { SkillConfig, Context } from '../../../core/types';
import type { Capabilities, ContentItem, FunctionToolDefinition, UsageStats } from '../../../uamp/types';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events';
import { generateEventId } from '../../../uamp/events';
import { anthropicAdapter } from '../../../adapters/anthropic';
import type { AdapterChunk, Message, ToolDefinition, UAMPUsage } from '../../../adapters/types';

export interface AnthropicSkillConfig extends SkillConfig {
  apiKey?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export class AnthropicSkill extends Skill {
  private modelConfig: AnthropicSkillConfig;

  constructor(config: AnthropicSkillConfig = {}) {
    super({ ...config, name: config.name || 'anthropic' });
    this.modelConfig = config;
  }

  private get apiKey(): string | undefined {
    return this.modelConfig.apiKey
      || (typeof process !== 'undefined' ? process.env?.ANTHROPIC_API_KEY : undefined);
  }

  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'claude-sonnet-4-20250514';
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

  @handoff({ name: 'anthropic', priority: 9 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId };

    try {
      const key = this.apiKey;
      if (!key) throw new Error('Anthropic API key not configured');

      const { messages, tools } = extractInput(events, context);
      if (messages.length === 0) {
        yield { type: 'response.error', event_id: generateEventId(), response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' } };
        return;
      }

      const model = this.modelConfig.model || 'claude-sonnet-4-20250514';

      context.set?.('_llm_capabilities', {
        model,
        provider: 'anthropic',
        maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        pricing: { inputPer1k: 0, outputPer1k: 0 },
      });

      const request = anthropicAdapter.buildRequest({
        messages,
        model,
        tools,
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
        throw new Error(`Anthropic API returned ${response.status}: ${errorText.slice(0, 200)}`);
      }

      let fullContent = '';
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      let usageInput = 0;
      let usageOutput = 0;

      for await (const chunk of anthropicAdapter.parseStream(response)) {
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
        provider: 'anthropic',
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
        error: { code: 'anthropic_error', message: (error as Error).message },
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
