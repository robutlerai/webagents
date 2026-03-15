/**
 * Google AI Skill
 *
 * Cloud LLM inference using Google's Gemini API via the shared GoogleAdapter.
 * Uses raw fetch + SSE (no SDK dependency).
 *
 * @see https://ai.google.dev/gemini-api/docs
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';
import { googleAdapter } from '../../../adapters/google.js';
import type { AdapterChunk, Message, ToolDefinition, UAMPUsage } from '../../../adapters/types.js';

export interface GoogleSkillConfig extends SkillConfig {
  apiKey?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export class GoogleSkill extends Skill {
  private modelConfig: GoogleSkillConfig;

  constructor(config: GoogleSkillConfig = {}) {
    super({ ...config, name: config.name || 'google' });
    this.modelConfig = config;
  }

  private get apiKey(): string | undefined {
    return this.modelConfig.apiKey
      || (typeof process !== 'undefined' ? process.env?.GOOGLE_API_KEY : undefined);
  }

  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'gemini-2.5-flash';
    return {
      id: model,
      provider: 'google',
      modalities: ['text', 'image', 'audio', 'video'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: true,
      tools: {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: false,
        built_in_tools: ['code_execution'],
      },
      context_window: model.includes('1.5') ? 1000000 : 128000,
    };
  }

  @handoff({ name: 'google', priority: 8 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId };

    try {
      const key = this.apiKey;
      if (!key) throw new Error('Google API key not configured');

      const { messages, tools } = extractInput(events, context);
      if (messages.length === 0) {
        yield { type: 'response.error', event_id: generateEventId(), response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' } };
        return;
      }

      const resolvedMedia = context?.get
        ? context.get<Map<string, { mimeType: string; base64: string }>>('_resolved_images')
        : undefined;

      const model = this.modelConfig.model || 'gemini-2.5-flash';

      // Set capabilities so PaymentSkill can size the lock
      context.set?.('_llm_capabilities', {
        model,
        provider: 'google',
        maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        pricing: { inputPer1k: 0, outputPer1k: 0 },
      });

      const request = googleAdapter.buildRequest({
        messages,
        model,
        tools,
        temperature: this.modelConfig.temperature ?? 0.7,
        maxTokens: this.modelConfig.max_tokens ?? 4096,
        apiKey: key,
        resolvedMedia,
      });

      const response = await fetch(request.url, {
        method: 'POST',
        headers: request.headers,
        body: request.body,
        signal: context.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Google API returned ${response.status}: ${errorText.slice(0, 200)}`);
      }

      let fullContent = '';
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      let usageInput = 0;
      let usageOutput = 0;

      for await (const chunk of googleAdapter.parseStream(response)) {
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

      // Set usage so PaymentSkill can settle
      context.set?.('_llm_usage', {
        model,
        provider: 'google',
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
        error: { code: 'google_error', message: (error as Error).message },
      };
    }
  }
}

function extractInput(events: ClientEvent[], context: Context): {
  messages: Message[];
  tools: ToolDefinition[];
} {
  const agenticMessages = context?.get
    ? context.get<Message[]>('_agentic_messages')
    : undefined;

  if (agenticMessages && agenticMessages.length > 0) {
    const contextTools = context?.get
      ? context.get<ToolDefinition[]>('_agentic_tools') ?? []
      : [];
    return { messages: agenticMessages, tools: contextTools };
  }

  const messages: Message[] = [];
  let tools: ToolDefinition[] = [];

  for (const event of events) {
    if (event.type === 'session.create') {
      const e = event as SessionCreateEvent;
      if (e.session.instructions) {
        messages.push({ role: 'system', content: e.session.instructions });
      }
      if (e.session.tools && e.session.tools.length > 0) {
        tools = e.session.tools.map(t => ({
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
  if (chunk.type === 'image') {
    return {
      type: 'response.delta', event_id: generateEventId(), response_id: responseId,
      delta: { type: 'text', text: `[inline image: ${chunk.mimeType}]` },
    };
  }
  return null;
}
