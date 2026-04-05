/**
 * OpenAI-Compatible LLM Adapter
 *
 * Handles request building, UAMP content_items conversion, SSE stream parsing
 * with choices[].delta, tool call accumulation by index, and usage reporting.
 *
 * Also used by xAI (Grok) and Fireworks with different base URLs.
 *
 * Source of truth for all OpenAI-compatible conversion logic.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message } from './types';
import { readSSEStream } from './sse';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, type ResolvedMediaMap } from './content';

const OPENAI_BASE_URL = 'https://api.openai.com/v1';

function isReasoningModel(model: string): boolean {
  return /^o[1-9]/.test(model);
}

// MIME types OpenAI accepts natively via file content parts
const OPENAI_FILE_TYPES = new Set([
  'application/pdf',
  'text/plain', 'text/html', 'text/css', 'text/csv', 'text/markdown',
  'text/javascript', 'text/x-python', 'text/x-c', 'text/x-c++', 'text/x-java',
  'application/json',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
]);

const MIME_TO_DEFAULT_EXT: Record<string, string> = {
  'application/pdf': '.pdf',
  'text/plain': '.txt', 'text/html': '.html', 'text/css': '.css',
  'text/csv': '.csv', 'text/markdown': '.md', 'text/javascript': '.js',
  'application/json': '.json',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
};

/**
 * Convert UAMP content items to OpenAI multimodal parts.
 * Handles image → image_url (data URI), audio → input_audio, file → native file part,
 * and adds placeholders for unsupported modalities (video).
 */
function uampToOpenAIParts(
  items: Array<Record<string, unknown>>,
  resolvedMedia?: ResolvedMediaMap,
): unknown[] {
  const parts: unknown[] = [];
  for (const item of items) {
    if (item.type === 'text' && item.text) {
      parts.push({ type: 'text', text: item.text });
    } else if (item.type === 'image') {
      const url = extractContentRef(item.image);
      if (url) {
        const canonical = canonicalContentUrl(url);
        const media = canonical ? resolvedMedia?.get(canonical) : undefined;
        if (media) {
          parts.push({ type: 'image_url', image_url: { url: `data:${media.mimeType};base64,${media.base64}` } });
        } else {
          parts.push({ type: 'image_url', image_url: { url } });
        }
      }
    } else if (item.type === 'audio') {
      const url = extractContentRef(item.audio);
      if (url) {
        const canonical = canonicalContentUrl(url);
        const media = canonical ? resolvedMedia?.get(canonical) : undefined;
        if (media) {
          const fmt = media.mimeType.split('/')[1] || 'wav';
          parts.push({ type: 'input_audio', input_audio: { data: media.base64, format: fmt } });
        }
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media && OPENAI_FILE_TYPES.has(media.mimeType)) {
        const filename = (item.filename as string) || `document${MIME_TO_DEFAULT_EXT[media.mimeType] || ''}`;
        parts.push({ type: 'file', file: { filename, file_data: `data:${media.mimeType};base64,${media.base64}` } });
      } else if ((item as Record<string, unknown>)._extracted_text) {
        parts.push({ type: 'text', text: (item as Record<string, unknown>)._extracted_text });
      } else {
        const fname = (item.filename as string) || 'file';
        const mime = (item.mime_type as string) || 'unknown';
        parts.push({ type: 'text', text: `[Attached file: ${fname} (${mime}) — content not available inline]` });
      }
    } else if (item.type === 'video') {
      parts.push({ type: 'text', text: '[Attached video — not supported by this model]' });
    }
  }
  return parts.length > 0 ? parts : [{ type: 'text', text: '(no content)' }];
}

/**
 * Convert messages: detect UAMP content_items and convert them to OpenAI parts,
 * strip UAMP-specific fields (content_items) from all messages.
 */
function convertMessages(
  messages: Message[],
  resolvedMedia?: ResolvedMediaMap,
): Array<Record<string, unknown>> {
  return messages.map(m => {
    const uampItems = (Array.isArray(m.content) && isUAMPContentArray(m.content))
      ? m.content as Array<Record<string, unknown>>
      : (Array.isArray(m.content_items) && m.content_items.length > 0
          && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? m.content_items
        : null;

    // Build a clean message without content_items
    const clean: Record<string, unknown> = { role: m.role };
    if (uampItems) {
      clean.content = uampToOpenAIParts(uampItems, resolvedMedia);
    } else {
      clean.content = m.content;
    }
    if (m.tool_calls) clean.tool_calls = m.tool_calls;
    if (m.tool_call_id) clean.tool_call_id = m.tool_call_id;
    if (m.name) clean.name = m.name;
    return clean;
  });
}

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

      const messages = convertMessages(params.messages, params.resolvedMedia);

      const body: Record<string, unknown> = {
        model: modelName,
        messages,
        stream,
      };
      if (params.temperature != null) body.temperature = params.temperature;
      if (params.maxTokens != null) {
        if (isReasoningModel(modelName)) {
          body.max_completion_tokens = params.maxTokens;
        } else {
          body.max_tokens = params.maxTokens;
        }
      }
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

        if (delta?.reasoning_content) {
          yield { type: 'thinking', text: delta.reasoning_content as string };
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

        const annotations = delta?.annotations as Array<{
          type?: string;
          url?: string;
          title?: string;
          file_id?: string;
          filename?: string;
          text?: string;
        }> | undefined;
        if (annotations) {
          for (const ann of annotations) {
            if (ann.type === 'url_citation' && ann.url) {
              yield {
                type: 'tool_result',
                call_id: 'web_search',
                result: JSON.stringify({ url: ann.url, title: ann.title ?? '' }),
              };
            }
            if (ann.type === 'file_citation' && ann.file_id) {
              yield {
                type: 'tool_result',
                call_id: 'file_search',
                result: JSON.stringify({ file_id: ann.file_id, filename: ann.filename ?? '', text: ann.text ?? '' }),
              };
            }
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
    document: 'base64',
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
