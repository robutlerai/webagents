/**
 * Anthropic Claude LLM Adapter
 *
 * Handles message conversion (OpenAI -> Anthropic format), request building,
 * SSE stream parsing with content_block events, tool_use/tool_result handling,
 * UAMP content_items → Anthropic blocks conversion, and usage reporting.
 *
 * Source of truth for all Anthropic-specific conversion logic.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message } from './types';
import { isFunctionTool } from './types';
import { readSSEStream } from './sse';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, type ResolvedMediaMap } from './content';

const BASE_URL = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';

// MIME types Anthropic accepts natively via document blocks
const ANTHROPIC_DOCUMENT_TYPES = new Set([
  'application/pdf',
  'text/plain', 'text/html', 'text/csv', 'text/markdown',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
]);

/**
 * Convert UAMP content items to Anthropic content blocks.
 * Handles image (base64), file/document (native types via base64, others via _extracted_text),
 * and adds placeholders for unsupported modalities (audio, video).
 */
function uampToAnthropicBlocks(
  items: Array<Record<string, unknown>>,
  resolvedMedia?: ResolvedMediaMap,
): AnthropicContentBlock[] {
  const blocks: AnthropicContentBlock[] = [];
  for (const item of items) {
    if (item.type === 'text' && item.text) {
      blocks.push({ type: 'text', text: item.text as string });
    } else if (item.type === 'image') {
      const url = extractContentRef(item.image);
      if (url) {
        const canonical = canonicalContentUrl(url);
        const media = canonical ? resolvedMedia?.get(canonical) : undefined;
        if (media) {
          blocks.push({ type: 'image', source: { type: 'base64', media_type: media.mimeType, data: media.base64 } });
        }
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media && ANTHROPIC_DOCUMENT_TYPES.has(media.mimeType)) {
        blocks.push({ type: 'document', source: { type: 'base64', media_type: media.mimeType, data: media.base64 } });
      } else if ((item as Record<string, unknown>)._extracted_text) {
        blocks.push({ type: 'text', text: (item as Record<string, unknown>)._extracted_text as string });
      } else {
        const fname = (item.filename as string) || 'file';
        const mime = (item.mime_type as string) || 'unknown';
        blocks.push({ type: 'text', text: `[Attached file: ${fname} (${mime}) — content not available inline]` });
      }
    } else if (item.type === 'audio' || item.type === 'video') {
      const modality = item.type as string;
      blocks.push({ type: 'text', text: `[Attached ${modality} — not supported by this model]` });
    }
  }
  return blocks.length > 0 ? blocks : [{ type: 'text', text: '(no content)' }];
}

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

    const { system, messages } = convertMessages(params.messages, params.resolvedMedia);

    const body: Record<string, unknown> = {
      model: modelName,
      messages,
      stream,
      max_tokens: params.maxTokens ?? 4096,
    };
    if (params.temperature != null) body.temperature = params.temperature;
    if (system) body.system = system;

    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools.map(t => {
        if (isFunctionTool(t)) {
          return {
            name: t.function.name,
            description: t.function.description,
            input_schema: t.function.parameters || { type: 'object', properties: {} },
          };
        }
        const { type: _type, ...rest } = t;
        return { type: t.type, ...rest };
      });
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
        const block = data.content_block as Record<string, unknown> | undefined;
        if (block?.type === 'tool_use') {
          inToolBlock = true;
          currentToolId = (block.id as string) ?? '';
          currentToolName = (block.name as string) ?? '';
          currentToolArgs = '';
        }
        if (block?.type === 'web_search_tool_result') {
          const content = block.content as Array<{ type?: string; url?: string; title?: string; text?: string }> | undefined;
          const summary = content?.map(c => `[${c.title ?? ''}](${c.url ?? ''}): ${c.text ?? ''}`).join('\n') ?? '';
          yield { type: 'tool_result', call_id: 'web_search', result: summary };
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
  | { type: 'image'; source: { type: 'base64'; media_type: string; data: string } }
  | { type: 'document'; source: { type: 'base64'; media_type: string; data: string } }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean };

/**
 * Convert OpenAI-format messages to Anthropic format.
 * Extracts system message as top-level, converts tool_calls to tool_use blocks,
 * tool results to tool_result blocks, and UAMP content_items to Anthropic blocks.
 */
function convertMessages(
  messages: Message[],
  resolvedMedia?: ResolvedMediaMap,
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

    // Detect UAMP content items on the message (content array or content_items field)
    const uampItems = (Array.isArray(msg.content) && isUAMPContentArray(msg.content))
      ? msg.content as Array<Record<string, unknown>>
      : (Array.isArray(msg.content_items) && msg.content_items.length > 0
          && msg.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? msg.content_items
        : null;

    if (msg.role === 'assistant') {
      const blocks: AnthropicContentBlock[] = [];
      const text = typeof msg.content === 'string' ? msg.content : '';
      if (text) blocks.push({ type: 'text', text });
      if (uampItems) blocks.push(...uampToAnthropicBlocks(uampItems, resolvedMedia));
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: Record<string, unknown> = {};
          try { input = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
          blocks.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
        }
      }
      if (blocks.length === 1 && blocks[0].type === 'text') {
        result.push({ role: 'assistant', content: (blocks[0] as { text: string }).text });
      } else if (blocks.length > 0) {
        result.push({ role: 'assistant', content: blocks });
      } else {
        result.push({ role: 'assistant', content: text });
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

    // User message — convert UAMP content items if present
    if (uampItems) {
      result.push({ role: 'user', content: uampToAnthropicBlocks(uampItems, resolvedMedia) });
    } else {
      const content = typeof msg.content === 'string' ? msg.content : '';
      result.push({ role: 'user', content });
    }
  }

  return { system, messages: result };
}

export default anthropicAdapter;
