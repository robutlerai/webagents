/**
 * OpenAI-Compatible LLM Adapter
 *
 * Handles request building, SSE stream parsing with choices[].delta,
 * tool call accumulation by index, and usage reporting.
 *
 * Also used by xAI (Grok) and Fireworks with different base URLs.
 *
 * Extracted from the battle-tested proxy implementation in lib/llm/uamp-proxy.ts.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport } from './types.js';
import { readSSEStream } from './sse.js';

const OPENAI_BASE_URL = 'https://api.openai.com/v1';

export function createOpenAICompatibleAdapter(config: {
  name: string;
  baseUrl: string;
  mediaSupport?: Partial<MediaSupport>;
  modelAliases?: Record<string, string>;
}): LLMAdapter {
  return {
    name: config.name,

    mediaSupport: {
      image: 'url',
      audio: 'base64',
      video: 'none',
      document: 'none',
      ...config.mediaSupport,
    },

    buildRequest(params: AdapterRequestParams): AdapterRequest {
      const rawName = params.model.includes('/') ? params.model.split('/').pop()! : params.model;
      const modelName = config.modelAliases?.[rawName] ?? rawName;
      const stream = params.stream !== false;

      const body: Record<string, unknown> = {
        model: modelName,
        messages: params.messages,
        stream,
      };
      if (params.temperature != null) body.temperature = params.temperature;
      if (params.maxTokens != null) body.max_tokens = params.maxTokens;
      if (params.tools && params.tools.length > 0) body.tools = params.tools;
      if (stream) body.stream_options = { include_usage: true };

      return {
        url: `${config.baseUrl}/chat/completions`,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${params.apiKey}`,
        },
        body: JSON.stringify(body),
      };
    },

    async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
      let inputTokens = 0;
      let outputTokens = 0;
      const pendingToolCalls = new Map<number, { id: string; name: string; arguments: string }>();

      for await (const chunk of readSSEStream(response)) {
        const data = chunk as Record<string, unknown>;
        const choices = data.choices as Array<Record<string, unknown>> | undefined;
        const choice = choices?.[0];
        if (!choice && !data.usage) continue;

        const delta = choice?.delta as Record<string, unknown> | undefined;

        if (delta?.content) {
          yield { type: 'text', text: delta.content as string };
        }

        const toolCallDeltas = delta?.tool_calls as Array<{
          index: number;
          id?: string;
          function?: { name?: string; arguments?: string };
        }> | undefined;

        if (toolCallDeltas) {
          for (const tc of toolCallDeltas) {
            const idx = tc.index ?? 0;
            if (!pendingToolCalls.has(idx)) {
              pendingToolCalls.set(idx, { id: tc.id ?? '', name: '', arguments: '' });
            }
            const entry = pendingToolCalls.get(idx)!;
            if (tc.id) entry.id = tc.id;
            if (tc.function?.name) entry.name += tc.function.name;
            if (tc.function?.arguments) entry.arguments += tc.function.arguments;
          }
        }

        const finishReason = choice?.finish_reason as string | null;
        if (finishReason === 'tool_calls' || finishReason === 'stop') {
          for (const [, tc] of pendingToolCalls) {
            if (tc.id && tc.name) {
              yield { type: 'tool_call', id: tc.id, name: tc.name, arguments: tc.arguments };
            }
          }
          pendingToolCalls.clear();
        }

        const usage = data.usage as { prompt_tokens?: number; completion_tokens?: number } | undefined;
        if (usage) {
          inputTokens = usage.prompt_tokens ?? inputTokens;
          outputTokens = usage.completion_tokens ?? outputTokens;
        }
      }

      if (inputTokens > 0 || outputTokens > 0) {
        yield { type: 'usage', input: inputTokens, output: outputTokens };
      }
    },
  };
}

export const openaiAdapter = createOpenAICompatibleAdapter({
  name: 'openai',
  baseUrl: OPENAI_BASE_URL,
  mediaSupport: {
    image: 'url',
    audio: 'base64',
    video: 'none',
    document: 'none',
  },
});

export const xaiAdapter = createOpenAICompatibleAdapter({
  name: 'xai',
  baseUrl: 'https://api.x.ai/v1',
  mediaSupport: {
    image: 'none',
    audio: 'none',
    video: 'none',
    document: 'none',
  },
  modelAliases: {
    'grok-4.20-beta': 'grok-4.20-beta-latest',
  },
});

export const fireworksAdapter = createOpenAICompatibleAdapter({
  name: 'fireworks',
  baseUrl: 'https://api.fireworks.ai/inference/v1',
  mediaSupport: {
    image: 'none',
    audio: 'none',
    video: 'none',
    document: 'none',
  },
});

export default openaiAdapter;
