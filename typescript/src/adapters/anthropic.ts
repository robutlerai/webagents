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
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime, type ResolvedMediaMap, type DescribeContentOptions } from './content';

const BASE_URL = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';
const THINKING_BUDGET_TOKENS = 10_000;

const MODEL_ALIASES: Record<string, string> = {
};

function resolveModel(raw: string): string {
  return MODEL_ALIASES[raw] ?? raw;
}

function isThinkingModel(model: string): boolean {
  return /^claude-(3-7-sonnet|sonnet-4|opus-4)/.test(model);
}

/**
 * Newer Anthropic models (claude-opus-4-7 and beyond) have replaced the
 * `thinking: { type: 'enabled', budget_tokens }` API with an adaptive variant
 * controlled via `output_config.effort`. Sending the legacy shape against
 * these models returns:
 *   "thinking.type.enabled" is not supported for this model.
 *   Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.
 *
 * Detect any opus/sonnet/haiku-4-7-or-later model so we automatically opt into
 * the new shape as additional minor versions ship.
 */
function usesAdaptiveThinking(model: string): boolean {
  const m = /^claude-(?:opus|sonnet|haiku)-(\d+)-(\d+)/.exec(model);
  if (!m) return false;
  const major = Number(m[1]);
  const minor = Number(m[2]);
  return major > 4 || (major === 4 && minor >= 7);
}

// ────────────────────────────────────────────────────────────────
// Anthropic-specific tool variant resolution
// ────────────────────────────────────────────────────────────────
//
// Anthropic ships text_editor under multiple (type, name) pairs that change
// per model family, and the API rejects any mismatched combination:
//   text_editor_20241022 / text_editor_20250124 → name: "str_replace_editor"
//   text_editor_20250429 / text_editor_20250728 → name: "str_replace_based_edit_tool"
//
// We keep this leak fully contained inside the adapter:
//   - buildRequest maps the canonical { type: 'native', name: 'text_editor' | 'bash' }
//     marker to the Anthropic-specific (type, name) pair for the active model.
//   - convertMessages translates assistant `tool_calls` from canonical names
//     back to the Anthropic-specific name for the same active model.
//   - parseStream normalizes inbound tool_use names (`str_replace_editor` /
//     `str_replace_based_edit_tool` → `text_editor`) so the canonical name is
//     all the rest of the system ever sees on the wire.

type AnthropicNativeKind = 'text_editor' | 'bash';
type AnthropicNativeVariant = { type: string; name: string };

const TEXT_EDITOR_LEGACY: AnthropicNativeVariant = { type: 'text_editor_20250124', name: 'str_replace_editor' };
const TEXT_EDITOR_MODERN: AnthropicNativeVariant = { type: 'text_editor_20250728', name: 'str_replace_based_edit_tool' };
const BASH_DEFAULT:       AnthropicNativeVariant = { type: 'bash_20250124', name: 'bash' };

function resolveAnthropicNative(
  modelName: string,
  kind: AnthropicNativeKind,
): AnthropicNativeVariant {
  if (kind === 'bash') return BASH_DEFAULT;
  // text_editor: claude-3-x families use the legacy variant; everything 4.x+
  // uses the modern one. Fall back to legacy for unknown models so calls don't
  // fail outright (Anthropic still accepts the older variant on most families).
  if (/^claude-3(?:-|$)/.test(modelName)) return TEXT_EDITOR_LEGACY;
  if (/^claude-(?:opus|sonnet|haiku)-/.test(modelName)) return TEXT_EDITOR_MODERN;
  return TEXT_EDITOR_LEGACY;
}

/**
 * Normalize an inbound Anthropic tool_use name back to the canonical UAMP name.
 * Mapping is many-to-one and model-independent, so the adapter doesn't need to
 * know which model produced the response.
 */
function canonicalToolName(rawAnthropicName: string): string {
  if (rawAnthropicName === 'str_replace_editor' || rawAnthropicName === 'str_replace_based_edit_tool') {
    return 'text_editor';
  }
  return rawAnthropicName;
}

/**
 * Map a canonical UAMP tool name (as used in stored assistant messages and over
 * the UAMP wire) to the Anthropic-specific name for `modelName`. Names that
 * aren't part of the Anthropic native-tool catalog are passed through.
 */
function anthropicToolNameFor(modelName: string, canonicalName: string): string {
  if (canonicalName === 'text_editor') return resolveAnthropicNative(modelName, 'text_editor').name;
  if (canonicalName === 'bash') return resolveAnthropicNative(modelName, 'bash').name;
  return canonicalName;
}

// MIME types Anthropic accepts natively via `document` blocks with
// `source: { type: 'base64', ... }`. The Messages API only allows
// `application/pdf` here — sending text/html, docx, etc. as base64 returns:
//   400 messages.0.content.N.document.source.base64.media_type:
//   Input should be 'application/pdf'.
// All other text-bearing files are inlined as `text` blocks below (the
// proxy resolves them as `kind: 'text'` so we never base64-round-trip
// plain UTF-8). Binary docs that aren't PDF (.docx, .xlsx, …) fall back
// to `_extracted_text` if present, else describeContentItem.
const ANTHROPIC_DOCUMENT_BASE64_TYPES = new Set(['application/pdf']);

function inlineFileAsText(filename: string | undefined, mime: string, text: string): string {
  const safeName = (filename || 'file').replace(/[<>"]/g, '');
  return `<file name="${safeName}" mime="${mime}">\n${text}\n</file>`;
}

const ANTHROPIC_DESCRIBE_OPTIONS: DescribeContentOptions = {
  supportedModalities: new Set(['image']),
  supportedDocMimes: ANTHROPIC_DOCUMENT_BASE64_TYPES,
  textDecodableMime: isTextDecodableMime,
};

/**
 * Convert UAMP content items to Anthropic content blocks.
 * Handles image (base64), file/document (native types via base64, others via _extracted_text).
 * Unresolved media falls back to describeContentItem text placeholder.
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
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media && media.kind === 'binary') {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: media.mimeType, data: media.base64 } });
      } else {
        blocks.push({ type: 'text', text: describeContentItem(item, ANTHROPIC_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      const filename = (item as Record<string, unknown>).filename as string | undefined;
      const extractedText = (item as Record<string, unknown>)._extracted_text as string | undefined;
      if (media?.kind === 'binary' && ANTHROPIC_DOCUMENT_BASE64_TYPES.has(media.mimeType)) {
        blocks.push({ type: 'document', source: { type: 'base64', media_type: media.mimeType, data: media.base64 } });
      } else if (media?.kind === 'text') {
        blocks.push({ type: 'text', text: inlineFileAsText(filename, media.mimeType, media.text) });
      } else if (extractedText) {
        const mime = (item as Record<string, unknown>).mime_type as string | undefined ?? 'application/octet-stream';
        blocks.push({ type: 'text', text: inlineFileAsText(filename, mime, extractedText) });
      } else {
        blocks.push({ type: 'text', text: describeContentItem(item, ANTHROPIC_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'audio' || item.type === 'video') {
      blocks.push({ type: 'text', text: describeContentItem(item, ANTHROPIC_DESCRIBE_OPTIONS) });
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
    const rawName = params.model.includes('/') ? params.model.split('/').pop()! : params.model;
    const modelName = resolveModel(rawName);
    const stream = params.stream !== false;

    const { system, messages } = convertMessages(params.messages, modelName, params.resolvedMedia);

    const thinking = params.thinking !== false && isThinkingModel(modelName);
    const defaultMaxTokens = thinking ? 16_000 : 4096;
    const maxTokens = Math.max(params.maxTokens ?? defaultMaxTokens, thinking ? THINKING_BUDGET_TOKENS + 1 : 0);

    const body: Record<string, unknown> = {
      model: modelName,
      messages,
      stream,
      max_tokens: maxTokens,
      cache_control: { type: 'ephemeral' },
    };
    if (thinking) {
      if (usesAdaptiveThinking(modelName)) {
        body.thinking = { type: 'adaptive' };
        body.output_config = { effort: 'medium' };
      } else {
        body.thinking = { type: 'enabled', budget_tokens: THINKING_BUDGET_TOKENS };
      }
    }
    if (params.temperature != null && !thinking) body.temperature = params.temperature;
    if (system) body.system = system;

    // Beta-header markers travel as `beta` on each native tool entry (see
    // lib/models/tool-support.ts). Collect, dedupe and emit as a single
    // `anthropic-beta` header. The `beta` field is stripped from the per-tool
    // body below — Anthropic 400s on unknown fields inside the tool object.
    const betas = new Set<string>();
    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools.map(t => {
        if (isFunctionTool(t)) {
          return {
            name: t.function.name,
            description: t.function.description,
            input_schema: t.function.parameters || { type: 'object', properties: {} },
          };
        }
        // Canonical native-tool marker: { type: 'native', name: 'text_editor' | 'bash' }.
        // Resolve to the Anthropic-specific (type, name) pair for this model.
        if ((t as { type?: string }).type === 'native') {
          const canonicalName = (t as { name?: string }).name;
          if (canonicalName === 'text_editor' || canonicalName === 'bash') {
            const variant = resolveAnthropicNative(modelName, canonicalName);
            return { type: variant.type, name: variant.name };
          }
        }
        const { type: _type, beta, ...rest } = t as { type: string; beta?: string; [k: string]: unknown };
        if (typeof beta === 'string' && beta.length > 0) betas.add(beta);
        return { type: t.type, ...rest };
      });
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-api-key': params.apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    };
    if (betas.size > 0) {
      headers['anthropic-beta'] = Array.from(betas).join(',');
    }

    return {
      url: `${BASE_URL}/messages`,
      headers,
      body: JSON.stringify(body),
    };
  },

  async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
    let inputTokens = 0;
    let outputTokens = 0;
    let cacheReadInputTokens = 0;
    let cacheCreationInputTokens = 0;
    let currentToolId = '';
    let currentToolName = '';
    let currentToolArgs = '';
    let lastProgressBytes = 0;
    const PROGRESS_INTERVAL = 2048;
    let inToolBlock = false;
    let inThinkingBlock = false;

    for await (const chunk of readSSEStream(response)) {
      const data = chunk as Record<string, unknown>;

      if (data.type === 'content_block_start') {
        const block = data.content_block as Record<string, unknown> | undefined;
        if (block?.type === 'thinking') {
          inThinkingBlock = true;
        }
        if (block?.type === 'tool_use') {
          inToolBlock = true;
          currentToolId = (block.id as string) ?? '';
          // Normalize Anthropic's per-version text_editor name back to the
          // canonical UAMP "text_editor" so the rest of the system never sees
          // str_replace_editor / str_replace_based_edit_tool on the wire.
          currentToolName = canonicalToolName((block.name as string) ?? '');
          currentToolArgs = '';
          lastProgressBytes = 0;
          if (currentToolId && currentToolName) {
            yield { type: 'tool_call_start', id: currentToolId, name: currentToolName };
          }
        }
        if (block?.type === 'web_search_tool_result') {
          const content = block.content as Array<{ type?: string; url?: string; title?: string; text?: string }> | undefined;
          const summary = content?.map(c => `[${c.title ?? ''}](${c.url ?? ''}): ${c.text ?? ''}`).join('\n') ?? '';
          yield { type: 'tool_result', call_id: 'web_search', result: summary };
        }
        if (block?.type === 'web_fetch_tool_result') {
          const content = block.content as Array<{ type?: string; text?: string; url?: string }> | undefined;
          const text = content?.map(c => c.text ?? '').join('\n') ?? '';
          yield { type: 'tool_result', call_id: 'web_fetch', result: text };
        }
        if (block?.type === 'bash_result') {
          const stdout = (block.content as string) ?? (block.output as string) ?? '';
          yield { type: 'tool_result', call_id: 'bash', result: stdout };
        }
        if (block?.type === 'code_execution_result') {
          const output = (block.content as string) ?? (block.output as string) ?? '';
          yield { type: 'tool_result', call_id: 'code_execution', result: output };
        }
        if (block?.type === 'memory_result') {
          const content = (block.content as string) ?? JSON.stringify(block);
          yield { type: 'tool_result', call_id: 'memory', result: content };
        }
      }

      if (data.type === 'content_block_delta') {
        const delta = data.delta as { type?: string; text?: string; thinking?: string; partial_json?: string } | undefined;
        if (delta?.type === 'thinking_delta' && delta.thinking) {
          yield { type: 'thinking', text: delta.thinking };
        } else if (delta?.text) {
          yield { type: 'text', text: delta.text };
        }
        if (delta?.type === 'input_json_delta' && delta.partial_json != null) {
          currentToolArgs += delta.partial_json;
          if (currentToolId && currentToolArgs.length - lastProgressBytes >= PROGRESS_INTERVAL) {
            lastProgressBytes = currentToolArgs.length;
            yield { type: 'tool_call_progress', id: currentToolId, bytes: currentToolArgs.length };
          }
        }
      }

      if (data.type === 'content_block_stop' && inThinkingBlock) {
        inThinkingBlock = false;
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
        const msg = data.message as { usage?: {
          input_tokens: number;
          cache_read_input_tokens?: number;
          cache_creation_input_tokens?: number;
        } } | undefined;
        if (msg?.usage) {
          inputTokens = msg.usage.input_tokens ?? 0;
          cacheReadInputTokens = msg.usage.cache_read_input_tokens ?? 0;
          cacheCreationInputTokens = msg.usage.cache_creation_input_tokens ?? 0;
        }
      }

      if (data.type === 'message_delta') {
        const usage = data.usage as { output_tokens: number } | undefined;
        if (usage) {
          outputTokens = usage.output_tokens ?? 0;
        }
      }
    }

    if (inputTokens > 0 || outputTokens > 0 || cacheReadInputTokens > 0 || cacheCreationInputTokens > 0) {
      yield {
        type: 'usage',
        input: inputTokens,
        output: outputTokens,
        ...(cacheReadInputTokens > 0 && { cache_read_input: cacheReadInputTokens }),
        ...(cacheCreationInputTokens > 0 && { cache_creation_input: cacheCreationInputTokens }),
      };
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
  modelName: string,
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
          // Translate canonical UAMP tool names ("text_editor", "bash") back to
          // the Anthropic-specific name for this model. Other names pass through.
          const name = anthropicToolNameFor(modelName, tc.function.name);
          blocks.push({ type: 'tool_use', id: tc.id, name, input });
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
      let content = typeof msg.content === 'string' ? msg.content : '';
      if (uampItems) {
        for (const item of uampItems) {
          if (['image', 'audio', 'video', 'file'].includes(item.type as string)) {
            content += '\n' + describeContentItem(item, ANTHROPIC_DESCRIBE_OPTIONS);
          }
        }
      }
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
      const blocks = uampToAnthropicBlocks(uampItems, resolvedMedia);
      // Backstop: prepend `m.content` text if not already represented by the
      // first text block. Without this, a delegate call carrying both a
      // prompt body AND an attached image (e.g. "Create unicorn.html using
      // the attached image") was sent to Claude as image-only — the model
      // then had no instructions and just described the picture instead of
      // running text_editor. Mirrors the openai/google adapters; see
      // data/logs/llm-payloads/2026-04-19T04-13-55-236Z_anthropic_*claude-sonnet-4-6_*
      // for the symptom (msg.content="Create a file…", outgoing user msg
      // had only `[Available image: …]`).
      if (typeof msg.content === 'string' && msg.content.trim()) {
        const firstText = blocks.find(
          (b): b is AnthropicContentBlock & { type: 'text'; text: string } =>
            b.type === 'text' && typeof (b as { text?: unknown }).text === 'string',
        );
        if (!firstText || firstText.text !== msg.content) {
          blocks.unshift({ type: 'text', text: msg.content });
        }
      }
      result.push({ role: 'user', content: blocks });
    } else {
      const content = typeof msg.content === 'string' ? msg.content : '';
      result.push({ role: 'user', content });
    }
  }

  // Post-process: enforce Anthropic's "tool_use must be immediately followed
  // by tool_result" rule. After an assistant message with `tool_use` blocks,
  // the very next user message MUST contain `tool_result` blocks for those
  // ids. If intervening user messages slipped in (e.g. read_content's
  // `_inline_for_llm` follow-up), rebuild that segment so the merged
  // tool_result user message comes first, then any remaining user blocks.
  const reordered: typeof result = [];
  for (let i = 0; i < result.length; i++) {
    const msg = result[i];
    reordered.push(msg);
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;
    const requiredIds = new Set<string>();
    for (const b of msg.content as AnthropicContentBlock[]) {
      if (b.type === 'tool_use' && (b as { id?: string }).id) {
        requiredIds.add((b as { id: string }).id);
      }
    }
    if (requiredIds.size === 0) continue;
    const toolResultBlocks: AnthropicContentBlock[] = [];
    const trailingBlocks: AnthropicContentBlock[] = [];
    let j = i + 1;
    while (j < result.length && result[j].role === 'user' && requiredIds.size > 0) {
      const next = result[j];
      const blocks = Array.isArray(next.content)
        ? (next.content as AnthropicContentBlock[])
        : [{ type: 'text', text: typeof next.content === 'string' ? next.content : '' } as AnthropicContentBlock];
      for (const b of blocks) {
        if (b.type === 'tool_result') {
          const tid = (b as { tool_use_id?: string }).tool_use_id;
          if (tid && requiredIds.has(tid)) {
            requiredIds.delete(tid);
            toolResultBlocks.push(b);
          } else {
            // tool_result for some other call — keep as trailing to preserve later pairings
            trailingBlocks.push(b);
          }
        } else {
          trailingBlocks.push(b);
        }
      }
      j++;
    }
    // Only rewrite if we actually consumed forward messages (i.e. there was
    // something to merge or reorder). When the first user already covered all
    // required ids without intervening blocks, j === i+2 and trailingBlocks
    // is empty — leaving the original message untouched is fine.
    if (j > i + 1 && (toolResultBlocks.length > 0 || trailingBlocks.length > 0)) {
      if (toolResultBlocks.length > 0) {
        reordered.push({ role: 'user', content: toolResultBlocks });
      }
      if (trailingBlocks.length > 0) {
        reordered.push({ role: 'user', content: trailingBlocks });
      }
      i = j - 1; // skip the forward messages we just merged
    }
  }

  return { system, messages: reordered };
}

export default anthropicAdapter;
