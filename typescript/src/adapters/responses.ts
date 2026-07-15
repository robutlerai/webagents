/**
 * Responses API LLM Adapter (OpenAI `/v1/responses` and xAI `/v1/responses`)
 *
 * The Responses API is a structurally different endpoint from chat-completions:
 *   - separate `instructions` (system) string, separate `input` array of
 *     typed items (`message`, `function_call`, `function_call_output`,
 *     `reasoning`, …) instead of a flat `messages` array
 *   - flat function tool shape `{ type: 'function', name, description,
 *     parameters }` rather than chat-completions' `{ type: 'function',
 *     function: { name, description, parameters } }` nested wrapper
 *   - `reasoning: { effort, summary }` instead of `reasoning_effort` /
 *     `thinking_config`
 *   - `max_output_tokens` instead of `max_tokens` / `max_completion_tokens`
 *   - SSE event-typed protocol (`response.output_text.delta`,
 *     `response.function_call_arguments.delta`, `response.completed`, …)
 *     instead of `choices[].delta`
 *
 * The chat-completions and Responses surfaces share enough structure that we
 * could overload `createChatCompletionsAdapter`, but the differences are
 * pervasive enough that a separate factory is materially cleaner.
 *
 * Compatibility with the proxy stays:
 *   - `streamAdapterResponse` in `lib/llm/uamp-proxy.ts` only consumes the
 *     `AdapterChunk` union, so we emit the same chunk vocabulary
 *     (`text | thinking | tool_call_start | tool_call_progress | tool_call |
 *     tool_result | image | usage`).
 *   - Function-call dispatch needs `{ id, name, arguments }`. Responses uses
 *     `call_id`; the factory normalises `id ← item.call_id` on the way in
 *     and back to `call_id` on the way out (`function_call_output`).
 *   - Native built-in tool calls (`web_search_call`, `code_interpreter_call`,
 *     `image_generation_call`) ran server-side; we surface them as
 *     `tool_result` chunks tagged with the canonical UAMP tool name so the
 *     UI's existing chip rendering picks them up.
 *
 * Multi-turn reasoning replay: when the previous assistant turn carried
 * `_encryptedReasoning` (set by the proxy from `include:
 * ['reasoning.encrypted_content']`), we serialise those items back into
 * `input` so gpt-5*+tools doesn't lose chain-of-thought between calls. With
 * `store: false` the API is stateless, so this is the only way to preserve
 * reasoning across function-call rounds.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message, ThinkingLevel, ToolDefinition } from './types';
import { isFunctionTool, normalizeThinking } from './types';
import { readSSEStream } from './sse';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime, parseDataUrl, type ResolvedMediaMap, type DescribeContentOptions } from './content';

const OPENAI_RESPONSES_BASE_URL = 'https://api.openai.com/v1';
const XAI_RESPONSES_BASE_URL = 'https://api.x.ai/v1';

// MIME types Responses API accepts inline as a `file` part (PDF only).
// All other text-bearing files are inlined as text blocks; binary non-PDF
// files fall back to a description placeholder.
const RESPONSES_FILE_BASE64_TYPES = new Set(['application/pdf']);

const MIME_TO_DEFAULT_EXT: Record<string, string> = {
  'application/pdf': '.pdf',
  'text/plain': '.txt', 'text/html': '.html', 'text/css': '.css',
  'text/csv': '.csv', 'text/markdown': '.md', 'text/javascript': '.js',
  'application/json': '.json',
};

const RESPONSES_DESCRIBE_OPTIONS: DescribeContentOptions = {
  supportedModalities: new Set(['image', 'audio']),
  supportedDocMimes: RESPONSES_FILE_BASE64_TYPES,
  textDecodableMime: isTextDecodableMime,
};

function inlineFileAsText(filename: string | undefined, mime: string, text: string): string {
  const safeName = (filename || 'file').replace(/[<>"]/g, '');
  return `<file name="${safeName}" mime="${mime}">\n${text}\n</file>`;
}

/**
 * Responses-API GPT-5.x and o-series reasoning models reject any
 * `temperature` other than the default (1). Mirrors the chat-completions
 * factory — Responses behaves the same way for these families.
 */
function rejectsCustomTemperature(model: string): boolean {
  return /^(o[1-9]|gpt-5)/.test(model);
}

/**
 * Convert a UAMP content_items array into Responses-API content parts.
 * Output items use `input_text`, `input_image`, `input_file`. Unresolved or
 * unsupported media falls back to a `describeContentItem` text placeholder.
 */
function uampToResponseInputContent(
  items: Array<Record<string, unknown>>,
  resolvedMedia?: ResolvedMediaMap,
): unknown[] {
  const parts: unknown[] = [];
  for (const item of items) {
    if (item.type === 'text' && item.text) {
      parts.push({ type: 'input_text', text: item.text });
    } else if (item.type === 'image') {
      const url = extractContentRef(item.image);
      // Ephemeral data URLs (tool screenshots) inline directly — no
      // content-library resolution needed.
      const dataMedia = parseDataUrl(url);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = dataMedia
        ? { kind: 'binary' as const, mimeType: dataMedia.mimeType, base64: dataMedia.base64 }
        : (canonical ? resolvedMedia?.get(canonical) : undefined);
      if (media?.kind === 'binary') {
        parts.push({
          type: 'input_image',
          image_url: `data:${media.mimeType};base64,${media.base64}`,
        });
      } else {
        parts.push({ type: 'input_text', text: describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'audio') {
      // Responses API doesn't natively accept audio inputs (Realtime is the
      // audio-native endpoint). Render as a description placeholder so the
      // model sees the metadata but the request body stays accepted.
      parts.push({ type: 'input_text', text: describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS) });
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      const filename = (item.filename as string) || undefined;
      const extractedText = (item as Record<string, unknown>)._extracted_text as string | undefined;
      if (media?.kind === 'binary' && RESPONSES_FILE_BASE64_TYPES.has(media.mimeType)) {
        const fname = filename || `document${MIME_TO_DEFAULT_EXT[media.mimeType] || ''}`;
        parts.push({
          type: 'input_file',
          filename: fname,
          file_data: `data:${media.mimeType};base64,${media.base64}`,
        });
      } else if (media?.kind === 'text') {
        parts.push({ type: 'input_text', text: inlineFileAsText(filename, media.mimeType, media.text) });
      } else if (extractedText) {
        const mime = (item as Record<string, unknown>).mime_type as string | undefined ?? 'application/octet-stream';
        parts.push({ type: 'input_text', text: inlineFileAsText(filename, mime, extractedText) });
      } else {
        parts.push({ type: 'input_text', text: describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'video') {
      parts.push({ type: 'input_text', text: describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS) });
    }
  }
  return parts.length > 0 ? parts : [{ type: 'input_text', text: '(no content)' }];
}

/**
 * Pull leading and trailing system messages out of the message array and
 * concatenate their text content into a single `instructions` string.
 *
 * Responses uses `instructions` separately from `input`; system role inside
 * `input` would be re-typed as `developer` (which is fine but inconsistent),
 * so we lift them out cleanly.
 */
function extractInstructions(messages: Message[]): { instructions: string | null; rest: Message[] } {
  const systemTexts: string[] = [];
  const rest: Message[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const txt = typeof m.content === 'string'
        ? m.content
        : Array.isArray(m.content)
          ? m.content.filter((p) => (p as { type?: string }).type === 'text').map((p) => (p as { text?: string }).text ?? '').join('')
          : '';
      if (txt) systemTexts.push(txt);
    } else {
      rest.push(m);
    }
  }
  return {
    instructions: systemTexts.length > 0 ? systemTexts.join('\n\n') : null,
    rest,
  };
}

/**
 * Convert non-system UAMP messages into Responses-API `input` items.
 *
 *   user / assistant text → `{ type: 'message', role, content: [...] }`
 *   assistant tool_calls → one `function_call` item per call (carries
 *     `call_id`, `name`, `arguments` as a string)
 *   tool result → `{ type: 'function_call_output', call_id, output: string }`
 *   `_encryptedReasoning` → `reasoning` items with `encrypted_content` so the
 *     model can replay its CoT across multi-turn tool flows when `store: false`
 */
function convertMessagesToInput(
  messages: Message[],
  resolvedMedia?: ResolvedMediaMap,
): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];

  for (const m of messages) {
    // Tool result → function_call_output. Responses requires the literal
    // `call_id` the model emitted (not the chat-completions `tool_call_id`),
    // and we round-trip it transparently — the proxy uses `call_id` as the
    // adapter chunk's `id` field, so this works for either source.
    if (m.role === 'tool') {
      const callId = m.tool_call_id ?? '';
      const output = typeof m.content === 'string' ? m.content : '';

      // If the tool result carried media items, append their textual
      // metadata to `output` so the planner sees them. Tool results never
      // round-trip media inline in Responses (same posture as the
      // chat-completions adapter today).
      let stitched = output;
      const items = (Array.isArray(m.content) && isUAMPContentArray(m.content))
        ? m.content as Array<Record<string, unknown>>
        : (Array.isArray(m.content_items) && m.content_items.length > 0)
          ? m.content_items
          : null;
      if (items) {
        for (const item of items) {
          if (['image', 'audio', 'video', 'file'].includes(item.type as string)) {
            stitched += '\n' + describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS);
          }
        }
      }

      out.push({
        type: 'function_call_output',
        call_id: callId,
        output: stitched,
      });
      continue;
    }

    // Assistant message: it can carry text (for narration), tool_calls
    // (function_call items), `_encryptedReasoning` (reasoning items), and
    // historical `content_items` that captured prior media outputs (e.g.
    // images returned from generate_image). Media in assistant turns is
    // surfaced as describe-style text markers — the Responses API doesn't
    // accept inline image bytes inside an assistant `output_text` block,
    // and the assistant's "produced" media was actually a tool side-effect
    // that doesn't need replay byte-perfect.
    if (m.role === 'assistant') {
      // Replay encrypted reasoning items first — they precede the visible
      // function_call / message item in the original output stream, so
      // restoring them in this order matches the API's expectations.
      if (m._encryptedReasoning && m._encryptedReasoning.length > 0) {
        for (const enc of m._encryptedReasoning) {
          out.push({
            type: 'reasoning',
            encrypted_content: enc,
            // Responses requires `summary: []` (or omitted); empty array is
            // the safest replay form when we never received a summary
            // (typical when reasoning.summary != 'auto' or when the original
            // request didn't include it).
            summary: [],
          });
        }
      }

      // Visible text content of the assistant turn (if any).
      const baseText = typeof m.content === 'string' ? m.content.trim() : '';

      // Append describe-style markers for any assistant-side content_items
      // (historical media). Defensive against an upstream caller already
      // baking the text into `content_items` (we'd otherwise duplicate).
      const itemMarkers: string[] = [];
      const items = Array.isArray(m.content_items) ? m.content_items : null;
      if (items) {
        for (const item of items) {
          const t = item.type as string | undefined;
          if (t === 'text') {
            // Skip if it's the same text already in `m.content` (dedup).
            const txt = (item.text as string | undefined) ?? '';
            if (txt && txt !== baseText) itemMarkers.push(txt);
          } else if (t && ['image', 'audio', 'video', 'file'].includes(t)) {
            itemMarkers.push(describeContentItem(item, RESPONSES_DESCRIBE_OPTIONS));
          }
        }
      }

      const stitched = [baseText, ...itemMarkers].filter(Boolean).join('\n\n');
      if (stitched) {
        out.push({
          type: 'message',
          role: 'assistant',
          content: [{ type: 'output_text', text: stitched, annotations: [] }],
        });
      }

      // Function calls.
      if (m.tool_calls && m.tool_calls.length > 0) {
        for (const tc of m.tool_calls) {
          out.push({
            type: 'function_call',
            call_id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
          });
        }
      }
      continue;
    }

    // user (and any other non-system, non-assistant role) → message
    const uampItems = (Array.isArray(m.content) && isUAMPContentArray(m.content))
      ? m.content as Array<Record<string, unknown>>
      : (Array.isArray(m.content_items) && m.content_items.length > 0
          && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? m.content_items
        : null;

    let content: unknown[];
    if (uampItems) {
      content = uampToResponseInputContent(uampItems, resolvedMedia);
      // Backstop: if the caller also supplied a string `content`, prepend it.
      if (typeof m.content === 'string' && m.content.trim()) {
        const firstText = content.find(
          (p): p is { type: 'input_text'; text: string } =>
            (p as { type?: string }).type === 'input_text',
        );
        if (!firstText || firstText.text !== m.content) {
          content.unshift({ type: 'input_text', text: m.content });
        }
      }
    } else if (typeof m.content === 'string') {
      content = [{ type: 'input_text', text: m.content }];
    } else if (m.content == null) {
      content = [{ type: 'input_text', text: '' }];
    } else {
      // Best-effort: stringify
      content = [{ type: 'input_text', text: JSON.stringify(m.content) }];
    }

    out.push({
      type: 'message',
      role: m.role,
      content,
    });
  }

  return out;
}

/**
 * Convert a mixed tool list (function tools in chat-completions wrapper
 * shape, plus pre-built native tools) into the Responses-API tool list.
 */
function convertToolsForResponses(tools: ToolDefinition[] | undefined): unknown[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((t) => {
    if (isFunctionTool(t)) {
      // Flatten { type: 'function', function: { name, description, parameters } }
      // → { type: 'function', name, description, parameters, strict: false }
      const fn = t.function;
      return {
        type: 'function',
        name: fn.name,
        description: fn.description,
        parameters: fn.parameters ?? { type: 'object', properties: {} },
        strict: false,
      };
    }
    // Native tool — pass through verbatim. PROVIDER_TOOL_SUPPORT is the
    // source of truth for the body shape (`{ type: 'web_search' }`,
    // `{ type: 'code_interpreter', container: { type: 'auto' } }`, etc.).
    return t;
  });
}

/**
 * Map a canonical native tool item type emitted by the Responses stream to
 * the canonical UAMP tool name. Used so `tool_result` chunks land under the
 * same call_id convention the chat-completions annotation handler used.
 */
const NATIVE_ITEM_TYPE_TO_UAMP_NAME: Record<string, string> = {
  web_search_call: 'web_search',
  code_interpreter_call: 'code_execution',
  image_generation_call: 'image_generation',
  file_search_call: 'file_search',
  computer_call: 'computer_use',
};

export function createResponsesApiAdapter(config: {
  name: string;
  baseUrl: string;
  mediaSupport?: Partial<MediaSupport>;
  modelAliases?: Record<string, string>;
  modelTransform?: (rawName: string) => string;
  /** Extra request body fields keyed off the params (e.g. xAI image understanding). */
  extraBody?: (params: AdapterRequestParams, modelName: string) => Record<string, unknown>;
  /** Extra HTTP headers derived from request params. */
  extraHeaders?: (params: AdapterRequestParams) => Record<string, string>;
  /**
   * Map a canonical ThinkingLevel onto the Responses API's `reasoning` block.
   * Called with the resolved (post-alias / post-transform) model name. Return
   * the full `reasoning` object (`{ effort, summary }`) or `null` to omit
   * the block entirely.
   */
  thinkingMapper?: (
    modelName: string,
    level: ThinkingLevel | undefined,
    ctx: { hasTools: boolean },
  ) => Record<string, unknown> | null;
  /**
   * Optional remap of inbound SSE event names for providers whose Responses
   * dialect drifts from OpenAI's. Maps `from` → `to` on `event.type` before
   * the parser dispatches. Default identity.
   */
  eventNameMap?: Record<string, string>;
}): LLMAdapter {
  return {
    name: config.name,

    mediaSupport: {
      image: 'url',
      audio: 'none',
      video: 'none',
      document: 'base64',
      ...config.mediaSupport,
    },

    buildRequest(params: AdapterRequestParams): AdapterRequest {
      const rawName = params.model.includes('/') ? params.model.split('/').pop()! : params.model;
      const aliased = config.modelAliases?.[rawName] ?? rawName;
      const modelName = config.modelTransform ? config.modelTransform(aliased) : aliased;

      const stream = params.stream !== false;

      const { instructions, rest } = extractInstructions(params.messages);
      const input = convertMessagesToInput(rest, params.resolvedMedia);

      const tools = convertToolsForResponses(params.tools);

      const body: Record<string, unknown> = {
        model: modelName,
        input,
        stream,
        // Stateless: no server-side conversation. Encrypted reasoning items
        // are replayed from `_encryptedReasoning` on the next turn.
        store: false,
        // Pull encrypted reasoning back so we can replay it. Without this
        // multi-turn gpt-5+tools loses chain-of-thought between calls.
        include: ['reasoning.encrypted_content'],
        // Match chat-completions default: serialize tool calls. The proxy's
        // agent loop handles them one at a time anyway.
        parallel_tool_calls: false,
      };

      if (instructions) body.instructions = instructions;
      if (tools && tools.length > 0) body.tools = tools;

      if (params.temperature != null && !rejectsCustomTemperature(modelName)) {
        body.temperature = params.temperature;
      }
      if (params.maxTokens != null) {
        body.max_output_tokens = params.maxTokens;
      }

      if (config.thinkingMapper) {
        const hasTools = !!(tools && tools.length > 0);
        const reasoning = config.thinkingMapper(modelName, normalizeThinking(params.thinking), { hasTools });
        if (reasoning) body.reasoning = reasoning;
      }

      if (config.extraBody) {
        Object.assign(body, config.extraBody(params, modelName));
      }

      return {
        url: `${config.baseUrl}/responses`,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${params.apiKey}`,
          ...config.extraHeaders?.(params),
        },
        body: JSON.stringify(body),
      };
    },

    async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
      // Per-item function call accumulator. Keyed by `item_id` because that's
      // what the streaming events reference; we map back to the visible
      // `call_id` once `output_item.added` arrives.
      const pendingFn = new Map<string, { callId: string; name: string; arguments: string; lastProgressBytes: number }>();
      const itemIdToCallId = new Map<string, string>();
      const PROGRESS_INTERVAL = 2048;

      // Last seen usage on `response.completed`.
      let inputTokens = 0;
      let outputTokens = 0;
      let cacheReadInputTokens = 0;

      const eventNameMap = config.eventNameMap ?? {};

      for await (const data of readSSEStream(response)) {
        const evt = data as Record<string, unknown>;
        const rawType = (evt.type as string) ?? '';
        const type = eventNameMap[rawType] ?? rawType;
        if (!type) continue;

        // ---- Errors ----------------------------------------------------
        if (type === 'response.error' || type === 'response.failed' || type === 'error') {
          const err = (evt.error ?? evt.response ?? {}) as Record<string, unknown>;
          const msg = (err.message as string) || (evt.message as string) || `Responses API error (${type})`;
          throw new Error(msg);
        }

        // ---- Visible text ---------------------------------------------
        if (type === 'response.output_text.delta') {
          const delta = evt.delta as string | undefined;
          if (delta) yield { type: 'text', text: delta };
          continue;
        }

        // ---- Reasoning summary / reasoning text -----------------------
        if (
          type === 'response.reasoning_summary_text.delta' ||
          type === 'response.reasoning.delta' ||
          type === 'response.reasoning_text.delta'
        ) {
          const delta = evt.delta as string | undefined;
          if (delta) yield { type: 'thinking', text: delta };
          continue;
        }

        // ---- New output item ------------------------------------------
        if (type === 'response.output_item.added') {
          const item = (evt.item ?? {}) as Record<string, unknown>;
          const itemType = item.type as string | undefined;
          const itemId = item.id as string | undefined;

          if (itemType === 'function_call' && itemId) {
            const callId = (item.call_id as string) ?? itemId;
            const name = (item.name as string) ?? '';
            pendingFn.set(itemId, { callId, name, arguments: '', lastProgressBytes: 0 });
            itemIdToCallId.set(itemId, callId);
            if (name) {
              yield { type: 'tool_call_start', id: callId, name };
            }
          }
          // Native built-in tool calls — surface as tool_call_start so the
          // UI shows the chip immediately. Final tool_result is emitted on
          // `output_item.done` (server-side execution; we're spectators).
          else if (itemType && NATIVE_ITEM_TYPE_TO_UAMP_NAME[itemType] && itemId) {
            const canonicalName = NATIVE_ITEM_TYPE_TO_UAMP_NAME[itemType];
            yield { type: 'tool_call_start', id: canonicalName, name: canonicalName };
          }
          continue;
        }

        // ---- Function call argument streaming -------------------------
        if (type === 'response.function_call_arguments.delta') {
          const itemId = evt.item_id as string | undefined;
          const delta = evt.delta as string | undefined;
          if (!itemId || !delta) continue;
          const entry = pendingFn.get(itemId);
          if (!entry) continue;
          entry.arguments += delta;
          if (entry.arguments.length - entry.lastProgressBytes >= PROGRESS_INTERVAL) {
            entry.lastProgressBytes = entry.arguments.length;
            yield { type: 'tool_call_progress', id: entry.callId, bytes: entry.arguments.length };
          }
          continue;
        }

        // ---- Output item finalised ------------------------------------
        if (type === 'response.output_item.done') {
          const item = (evt.item ?? {}) as Record<string, unknown>;
          const itemType = item.type as string | undefined;
          const itemId = item.id as string | undefined;

          if (itemType === 'function_call' && itemId) {
            const entry = pendingFn.get(itemId);
            // Prefer the model's final `arguments` string from the item
            // payload — it's authoritative even if some deltas were
            // dropped/coalesced upstream.
            const finalArgs = (item.arguments as string) ?? entry?.arguments ?? '';
            const callId = (item.call_id as string) ?? entry?.callId ?? itemId;
            const name = (item.name as string) ?? entry?.name ?? '';
            if (callId && name) {
              yield { type: 'tool_call', id: callId, name, arguments: finalArgs };
            }
            pendingFn.delete(itemId);
          }
          // Native built-in tool calls finished → emit a tool_result so the
          // UI's chip rendering picks them up. The proxy's agent-loop
          // dispatcher must NOT route these back as function calls (they
          // ran server-side); it already filters by `chunk.type ===
          // 'tool_call'` only, so `tool_result` is a no-op for it.
          else if (itemType && NATIVE_ITEM_TYPE_TO_UAMP_NAME[itemType]) {
            const canonicalName = NATIVE_ITEM_TYPE_TO_UAMP_NAME[itemType];
            const status = (item.status as string) ?? 'completed';
            // Slim payload — we don't include the full server-side blob.
            const resultPayload: Record<string, unknown> = { status };
            if (itemType === 'web_search_call') {
              const action = (item.action as Record<string, unknown> | undefined);
              if (action?.queries) resultPayload.queries = action.queries;
            } else if (itemType === 'image_generation_call') {
              const result = item.result as string | undefined;
              if (result) {
                yield { type: 'image', base64: result, mimeType: 'image/png' };
              }
            }
            yield { type: 'tool_result', call_id: canonicalName, result: JSON.stringify(resultPayload), status };
          }
          // Reasoning item: nothing to surface in the stream — the
          // `_encryptedReasoning` round-trip is plumbed by the proxy when
          // it persists the assistant message (see uamp-proxy.ts integration).
          continue;
        }

        // ---- URL citations / annotations ------------------------------
        if (type === 'response.output_text.annotation.added') {
          const ann = (evt.annotation ?? {}) as Record<string, unknown>;
          const annType = ann.type as string | undefined;
          if (annType === 'url_citation' && ann.url) {
            yield {
              type: 'tool_result',
              call_id: 'web_search',
              result: JSON.stringify({ url: ann.url, title: ann.title ?? '' }),
            };
          } else if (annType === 'file_citation' && ann.file_id) {
            yield {
              type: 'tool_result',
              call_id: 'file_search',
              result: JSON.stringify({ file_id: ann.file_id, filename: ann.filename ?? '' }),
            };
          }
          continue;
        }

        // ---- Image generation streaming -------------------------------
        if (type === 'response.image_generation_call.partial_image') {
          const b64 = (evt.partial_image_b64 as string) ?? (evt.b64_json as string) ?? '';
          if (b64) {
            yield { type: 'image', base64: b64, mimeType: 'image/png' };
          }
          continue;
        }

        // ---- Final usage from response.completed ---------------------
        if (type === 'response.completed') {
          const resp = (evt.response ?? {}) as Record<string, unknown>;
          const usage = (resp.usage ?? {}) as {
            input_tokens?: number;
            output_tokens?: number;
            input_tokens_details?: { cached_tokens?: number };
          };
          inputTokens = usage.input_tokens ?? inputTokens;
          outputTokens = usage.output_tokens ?? outputTokens;
          cacheReadInputTokens = usage.input_tokens_details?.cached_tokens ?? cacheReadInputTokens;
          continue;
        }

        // Unhandled event types are silently ignored — Responses API ships
        // a constant trickle of new events (delta-of-deltas, hand-shake
        // pings, …). The set above covers everything we currently surface
        // through UAMP.
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

// ────────────────────────────────────────────────────────────────
// Concrete adapters
// ────────────────────────────────────────────────────────────────

/**
 * Translate ThinkingLevel into Responses API `reasoning.effort`. OpenAI's
 * vocabulary uses `minimal` (not `off`) for the lowest budget. Unlike
 * chat-completions, Responses accepts the combination with function tools,
 * so the gpt-5-+-tools workaround is dropped here.
 */
function openaiResponsesThinkingMapper(
  _modelName: string,
  level: ThinkingLevel | undefined,
): Record<string, unknown> | null {
  if (level === undefined) return null;
  const native = level === 'off' ? 'minimal' : level;
  // `summary: 'auto'` is required to stream reasoning summary text events
  // that we map onto the `thinking` AdapterChunk. Without it we'd still get
  // reasoning output items but no incremental events for the UI.
  return { effort: native, summary: 'auto' };
}

export const openaiAdapter: LLMAdapter = createResponsesApiAdapter({
  name: 'openai',
  baseUrl: OPENAI_RESPONSES_BASE_URL,
  mediaSupport: {
    image: 'url',
    audio: 'base64',
    video: 'none',
    document: 'base64',
  },
  thinkingMapper: openaiResponsesThinkingMapper,
});

/**
 * xAI Responses API uses the same `reasoning.effort` shape but xAI's native
 * vocabulary is `low|high` only (no `medium`, no native `off`). Collapse:
 * medium → high, off → low. The catalog gates which models receive a level.
 */
function xaiResponsesThinkingMapper(
  _modelName: string,
  level: ThinkingLevel | undefined,
): Record<string, unknown> | null {
  if (level === undefined) return null;
  const native: 'low' | 'high' = (level === 'high' || level === 'medium') ? 'high' : 'low';
  return { effort: native };
}

export const xaiAdapter: LLMAdapter = createResponsesApiAdapter({
  name: 'xai',
  baseUrl: XAI_RESPONSES_BASE_URL,
  mediaSupport: {
    image: 'url',
    audio: 'none',
    video: 'none',
    document: 'none',
  },
  modelAliases: {
    // Per https://docs.x.ai/developers/models. `grok-4.3` accepted directly.
    'grok-4.20-reasoning': 'grok-4.20-reasoning-latest',
    'grok-4.20-non-reasoning': 'grok-4.20-non-reasoning-latest',
  },
  thinkingMapper: xaiResponsesThinkingMapper,
  // Reserved for `enable_image_understanding: true` on vision-capable Grok
  // models. No-op until/unless we add such a model to the catalog.
  extraBody: () => ({}),
});

export default openaiAdapter;
