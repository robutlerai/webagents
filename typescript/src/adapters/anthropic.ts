/**
 * Anthropic Claude LLM Adapter
 *
 * Handles message conversion (OpenAI -> Anthropic format), request building,
 * SSE stream parsing with content_block events, tool_use/tool_result handling,
 * and usage reporting.
 *
 * Extracted from the battle-tested proxy implementation in lib/llm/uamp-proxy.ts.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message } from './types.js';
import { readSSEStream } from './sse.js';

const BASE_URL = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';

export const anthropicAdapter: LLMAdapter = {
  name: 'anthropic',

  mediaSupport: {
    image: 'base64',
    audio: 'none',
    video: 'none',
    document: 'base64',
  } satisfies MediaSupport,

  buildRequest(params: AdapterRequestParams): AdapterRequest {
    const modelName = params.model.includes('/') ? params.model.split('/').pop()! : params.model;
    const stream = params.stream !== false;

    const { system, messages } = convertMessages(params.messages);

    const body: Record<string, unknown> = {
      model: modelName,
      messages,
      stream,
      max_tokens: params.maxTokens ?? 4096,
    };
    if (params.temperature != null) body.temperature = params.temperature;
    if (system) body.system = system;

    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools.map(t => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters || { type: 'object', properties: {} },
      }));
    }

    return {
      url: `${BASE_URL}/messages`,
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': params.apiKey,
        'anthropic-version': ANTHROPIC_VERSION,
      },
      body: JSON.stringify(body),
    };
  },

  async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
    let inputTokens = 0;
    let outputTokens = 0;
    let currentToolId = '';
    let currentToolName = '';
    let currentToolArgs = '';
    let inToolBlock = false;

    for await (const chunk of readSSEStream(response)) {
      const data = chunk as Record<string, unknown>;

      if (data.type === 'content_block_start') {
        const block = data.content_block as { type: string; id?: string; name?: string } | undefined;
        if (block?.type === 'tool_use') {
          inToolBlock = true;
          currentToolId = block.id ?? '';
          currentToolName = block.name ?? '';
          currentToolArgs = '';
        }
      }

      if (data.type === 'content_block_delta') {
        const delta = data.delta as { type?: string; text?: string; partial_json?: string } | undefined;
        if (delta?.text) {
          yield { type: 'text', text: delta.text };
        }
        if (delta?.type === 'input_json_delta' && delta.partial_json != null) {
          currentToolArgs += delta.partial_json;
        }
      }

      if (data.type === 'content_block_stop' && inToolBlock) {
        inToolBlock = false;
        if (currentToolId && currentToolName) {
          yield {
            type: 'tool_call',
            id: currentToolId,
            name: currentToolName,
            arguments: currentToolArgs || '{}',
          };
        }
        currentToolId = '';
        currentToolName = '';
        currentToolArgs = '';
      }

      if (data.type === 'message_start') {
        const msg = data.message as { usage?: { input_tokens: number } } | undefined;
        if (msg?.usage) {
          inputTokens = msg.usage.input_tokens ?? 0;
        }
      }

      if (data.type === 'message_delta') {
        const usage = data.usage as { output_tokens: number } | undefined;
        if (usage) {
          outputTokens = usage.output_tokens ?? 0;
        }
      }
    }

    if (inputTokens > 0 || outputTokens > 0) {
      yield { type: 'usage', input: inputTokens, output: outputTokens };
    }
  },
};

type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean };

/**
 * Convert OpenAI-format messages to Anthropic format.
 * Extracts system message as top-level, converts tool_calls to tool_use blocks,
 * and tool results to tool_result blocks.
 */
function convertMessages(
  messages: Message[],
): {
  system?: string;
  messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }>;
} {
  let system: string | undefined;
  const result: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }> = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      const text = typeof msg.content === 'string' ? msg.content : '';
      system = (system ? system + '\n\n' : '') + text;
      continue;
    }

    if (msg.role === 'assistant') {
      const blocks: AnthropicContentBlock[] = [];
      const text = typeof msg.content === 'string' ? msg.content : '';
      if (text) blocks.push({ type: 'text', text });
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: Record<string, unknown> = {};
          try { input = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
          blocks.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
        }
      }
      if (blocks.length === 1 && blocks[0].type === 'text') {
        result.push({ role: 'assistant', content: (blocks[0] as { text: string }).text });
      } else {
        result.push({ role: 'assistant', content: blocks });
      }
      continue;
    }

    if (msg.role === 'tool') {
      const content = typeof msg.content === 'string' ? msg.content : '';
      result.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.tool_call_id || '',
          content,
        }],
      });
      continue;
    }

    // User message
    const content = typeof msg.content === 'string' ? msg.content : '';
    result.push({ role: 'user', content });
  }

  return { system, messages: result };
}

export default anthropicAdapter;
