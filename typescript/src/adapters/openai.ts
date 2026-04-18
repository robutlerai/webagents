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
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime, type ResolvedMediaMap, type DescribeContentOptions } from './content';

const OPENAI_BASE_URL = 'https://api.openai.com/v1';

function usesMaxCompletionTokens(model: string): boolean {
  return /^(o[1-9]|gpt-4o|gpt-5)/.test(model);
}

// MIME types OpenAI Chat Completions accepts as a `file` part with inline
// `file_data` (base64 data URI). Only PDFs are accepted inline; other types
// must either be uploaded via the Files API (file_id) or inlined as text.
// We choose the latter for text-bearing files (see uampToOpenAIParts).
const OPENAI_FILE_BASE64_TYPES = new Set(['application/pdf']);

function inlineFileAsTextOpenAI(filename: string | undefined, mime: string, text: string): string {
  const safeName = (filename || 'file').replace(/[<>"]/g, '');
  return `<file name="${safeName}" mime="${mime}">\n${text}\n</file>`;
}

const MIME_TO_DEFAULT_EXT: Record<string, string> = {
  'application/pdf': '.pdf',
  'text/plain': '.txt', 'text/html': '.html', 'text/css': '.css',
  'text/csv': '.csv', 'text/markdown': '.md', 'text/javascript': '.js',
  'application/json': '.json',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
};

const OPENAI_DESCRIBE_OPTIONS: DescribeContentOptions = {
  supportedModalities: new Set(['image', 'audio']),
  supportedDocMimes: OPENAI_FILE_BASE64_TYPES,
  textDecodableMime: isTextDecodableMime,
};

/**
 * Convert UAMP content items to OpenAI multimodal parts.
 * Handles image → image_url (data URI), audio → input_audio, file → native file part.
 * Unresolved media falls back to describeContentItem text placeholder.
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
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media?.kind === 'binary') {
        parts.push({ type: 'image_url', image_url: { url: `data:${media.mimeType};base64,${media.base64}` } });
      } else {
        parts.push({ type: 'text', text: describeContentItem(item, OPENAI_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'audio') {
      const url = extractContentRef(item.audio);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media?.kind === 'binary') {
        const fmt = media.mimeType.split('/')[1] || 'wav';
        parts.push({ type: 'input_audio', input_audio: { data: media.base64, format: fmt } });
      } else {
        parts.push({ type: 'text', text: describeContentItem(item, OPENAI_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      const filename = (item.filename as string) || undefined;
      const extractedText = (item as Record<string, unknown>)._extracted_text as string | undefined;
      if (media?.kind === 'binary' && OPENAI_FILE_BASE64_TYPES.has(media.mimeType)) {
        const fname = filename || `document${MIME_TO_DEFAULT_EXT[media.mimeType] || ''}`;
        parts.push({ type: 'file', file: { filename: fname, file_data: `data:${media.mimeType};base64,${media.base64}` } });
      } else if (media?.kind === 'text') {
        parts.push({ type: 'text', text: inlineFileAsTextOpenAI(filename, media.mimeType, media.text) });
      } else if (extractedText) {
        const mime = (item as Record<string, unknown>).mime_type as string | undefined ?? 'application/octet-stream';
        parts.push({ type: 'text', text: inlineFileAsTextOpenAI(filename, mime, extractedText) });
      } else {
        parts.push({ type: 'text', text: describeContentItem(item, OPENAI_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'video') {
      parts.push({ type: 'text', text: describeContentItem(item, OPENAI_DESCRIBE_OPTIONS) });
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

    // Tool results: never inline media, append text metadata instead
    if (m.role === 'tool') {
      const clean: Record<string, unknown> = { role: m.role };
      let content = typeof m.content === 'string' ? m.content : '';
      if (uampItems) {
        for (const item of uampItems) {
          if (['image', 'audio', 'video', 'file'].includes(item.type as string)) {
            content += '\n' + describeContentItem(item, OPENAI_DESCRIBE_OPTIONS);
          }
        }
      }
      clean.content = content;
      if (m.tool_call_id) clean.tool_call_id = m.tool_call_id;
      if (m.name) clean.name = m.name;
      return clean;
    }

    // Build a clean message without content_items
    const clean: Record<string, unknown> = { role: m.role };
    if (uampItems) {
      const parts = uampToOpenAIParts(uampItems, resolvedMedia);
      // Backstop: if the caller also supplied a string `content` (e.g. an
      // assistant turn that mixes text + media but the text is not in the
      // items array), prepend it as a text part so we don't drop context.
      // Only prepend when the text is not already represented as the first
      // text part to avoid duplication.
      if (typeof m.content === 'string' && m.content.trim()) {
        const firstTextPart = parts.find(
          (p): p is { type: 'text'; text: string } =>
            (p as { type?: string }).type === 'text',
        );
        if (!firstTextPart || firstTextPart.text !== m.content) {
          parts.unshift({ type: 'text', text: m.content });
        }
      }
      clean.content = parts;
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
  modelTransform?: (rawName: string) => string;
  /** Extra headers derived from request params, e.g. session-affinity for Fireworks. */
  extraHeaders?: (params: AdapterRequestParams) => Record<string, string>;
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
      const aliased = config.modelAliases?.[rawName] ?? rawName;
      const modelName = config.modelTransform ? config.modelTransform(aliased) : aliased;
      const stream = params.stream !== false;

      const messages = convertMessages(params.messages, params.resolvedMedia);

      const body: Record<string, unknown> = {
        model: modelName,
        messages,
        stream,
      };
      if (params.temperature != null) body.temperature = params.temperature;
      if (params.maxTokens != null) {
        if (usesMaxCompletionTokens(modelName)) {
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
          ...config.extraHeaders?.(params),
        },
        body: JSON.stringify(body),
      };
    },

    async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
      let inputTokens = 0;
      let outputTokens = 0;
      let cacheReadInputTokens = 0;
      const pendingToolCalls = new Map<number, { id: string; name: string; arguments: string }>();
      const startedToolCalls = new Set<number>();
      const lastProgressBytes = new Map<number, number>();
      const PROGRESS_INTERVAL = 2048;

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
              lastProgressBytes.set(idx, 0);
            }
            const entry = pendingToolCalls.get(idx)!;
            if (tc.id) entry.id = tc.id;
            if (tc.function?.name) entry.name += tc.function.name;
            if (tc.function?.arguments) entry.arguments += tc.function.arguments;

            if (!startedToolCalls.has(idx) && entry.id && entry.name) {
              startedToolCalls.add(idx);
              yield { type: 'tool_call_start' as const, id: entry.id, name: entry.name };
            }

            if (tc.function?.arguments) {
              const prev = lastProgressBytes.get(idx) ?? 0;
              if (entry.arguments.length - prev >= PROGRESS_INTERVAL) {
                lastProgressBytes.set(idx, entry.arguments.length);
                yield { type: 'tool_call_progress' as const, id: entry.id, bytes: entry.arguments.length };
              }
            }
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

        const usage = data.usage as {
          prompt_tokens?: number;
          completion_tokens?: number;
          prompt_tokens_details?: { cached_tokens?: number };
        } | undefined;
        if (usage) {
          inputTokens = usage.prompt_tokens ?? inputTokens;
          outputTokens = usage.completion_tokens ?? outputTokens;
          cacheReadInputTokens = usage.prompt_tokens_details?.cached_tokens ?? cacheReadInputTokens;
        }
      }

      if (inputTokens > 0 || outputTokens > 0) {
        yield {
          type: 'usage',
          input: inputTokens,
          output: outputTokens,
          ...(cacheReadInputTokens > 0 && { cache_read_input: cacheReadInputTokens }),
        };
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
  modelTransform: (name) =>
    name.startsWith('accounts/') ? name : `accounts/fireworks/models/${name}`,
  extraHeaders: (params): Record<string, string> =>
    params.sessionId ? { 'x-session-affinity': params.sessionId } : {},
});

export default openaiAdapter;
