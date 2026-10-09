/**
 * Anthropic Claude LLM Adapter
 *
 * Handles message conversion (OpenAI -> Anthropic format), request building,
 * SSE stream parsing with content_block events, tool_use/tool_result handling,
 * UAMP content_items → Anthropic blocks conversion, and usage reporting.
 *
 * Source of truth for all Anthropic-specific conversion logic.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message, ThinkingLevel } from './types';
import { isFunctionTool, normalizeThinking } from './types';
import { readSSEStream } from './sse';
import { normalizeToolHistory } from './tool-ids';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime, parseDataUrl, type ResolvedMediaMap, type DescribeContentOptions } from './content';

const BASE_URL = 'https://api.anthropic.com/v1';
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * Token budgets per canonical ThinkingLevel for the legacy
 * `thinking: { type: 'enabled', budget_tokens: N }` shape.
 *
 * `off` is intentionally absent — at level 'off' we omit the `thinking` block
 * entirely so the model behaves as a chat model.
 */
const ANTHROPIC_THINKING_BUDGETS: Record<Exclude<ThinkingLevel, 'off'>, number> = {
  low: 2_000,
  medium: 8_000,
  high: 16_000,
};

/**
 * Default budget when the caller didn't specify a level but the model is a
 * thinking-capable family. Mirrors the historical behaviour of this adapter
 * so prior callers see no change.
 */
const ANTHROPIC_DEFAULT_BUDGET = 10_000;

const MODEL_ALIASES: Record<string, string> = {
};

function resolveModel(raw: string): string {
  return MODEL_ALIASES[raw] ?? raw;
}

/**
 * Split a Claude id into family + version.
 *
 * Anthropic dropped the minor segment with the current generation:
 * `claude-opus-5` / `claude-sonnet-5`, not `claude-opus-5-0`. Every gate below
 * used to be a prefix or two-group regex, so the whole generation fell through
 * to "not a thinking model" and "not adaptive" — Opus 5 requests went out with
 * no `thinking` field at all, and anything that did enable thinking sent the
 * legacy `thinking.type.enabled` shape the API rejects. Parse the version once
 * and let the gates ask questions of it.
 *
 * `minor` is 0 when the id carries no minor segment, so `claude-opus-5` reads
 * as 5.0 and compares correctly against the 4.7 adaptive cutover.
 */
function parseClaudeVersion(model: string): { family: string; major: number; minor: number } | null {
  const m = /^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?/.exec(model);
  if (!m) return null;
  return { family: m[1], major: Number(m[2]), minor: m[3] === undefined ? 0 : Number(m[3]) };
}

/**
 * What a Claude family and version accept on the wire. Keyed by the PARSED
 * family and version, never by a literal model string, so a new minor
 * release inherits its family's shape instead of falling through to the
 * oldest one.
 *
 *  - `thinking`: `none` (the model never thinks; Haiku 4.5 and earlier,
 *    Claude 3), `budget` (the legacy `{ type: 'enabled', budget_tokens }`
 *    shape; Claude 4 up to 4.6 and 3.7 Sonnet) or `adaptive`
 *    (`{ type: 'adaptive' }` plus `output_config.effort`; Opus 4.7 and later,
 *    the 5.x generation, Fable and Mythos). Sending the legacy shape to an
 *    adaptive model returns:
 *      "thinking.type.enabled" is not supported for this model.
 *      Use "thinking.type.adaptive" and "output_config.effort" ...
 *  - `thinksByDefault`: omitting `thinking` still runs adaptive thinking
 *    (the 5.x generation, Haiku 5 and later, Fable, Mythos). A caller's `off`
 *    therefore has to be sent explicitly on these models; omitting the field
 *    would silently leave thinking on.
 *  - `acceptsDisabled`: `{ type: 'disabled' }` is accepted (at effort high or
 *    below). Opus 5.5 and Sonnet 5.5 reject it with a 400, as do Fable and
 *    Mythos, so `off` on those keeps the model's own default rather than
 *    sending a shape that fails the whole request.
 *  - `rejectsSampling`: any `temperature` other than the default is a 400
 *    ("temperature may only be set to 1"), so the field is never sent. This
 *    is the same set as the adaptive models today, Haiku 5 included; a
 *    forwarded default temperature of 0.7 used to fail every call on them.
 */
interface ClaudeCapabilities {
  thinking: 'none' | 'budget' | 'adaptive';
  thinksByDefault: boolean;
  acceptsDisabled: boolean;
  rejectsSampling: boolean;
}

const CLAUDE_NO_THINKING: ClaudeCapabilities = { thinking: 'none', thinksByDefault: false, acceptsDisabled: false, rejectsSampling: false };
const CLAUDE_BUDGET_THINKING: ClaudeCapabilities = { thinking: 'budget', thinksByDefault: false, acceptsDisabled: false, rejectsSampling: false };

function claudeCapabilities(model: string): ClaudeCapabilities {
  if (/^claude-3-7-sonnet/.test(model)) return CLAUDE_BUDGET_THINKING;
  const v = parseClaudeVersion(model);
  if (!v) return CLAUDE_NO_THINKING;
  // Fable / Mythos are thinking-only by construction: adaptive, never off.
  if (v.family === 'fable' || v.family === 'mythos') {
    return { thinking: 'adaptive', thinksByDefault: true, acceptsDisabled: false, rejectsSampling: true };
  }
  if (v.family === 'haiku') {
    // Haiku 4.5 and earlier never think and take a temperature. Haiku 5 and
    // later think by default, accept `{ type: 'disabled' }` and refuse a
    // temperature.
    if (v.major < 5) return CLAUDE_NO_THINKING;
    return { thinking: 'adaptive', thinksByDefault: true, acceptsDisabled: true, rejectsSampling: true };
  }
  if (v.major < 4) return CLAUDE_NO_THINKING;
  const adaptive = v.major > 4 || (v.major === 4 && v.minor >= 7);
  if (!adaptive) return CLAUDE_BUDGET_THINKING;
  // Opus 4.7 and 4.8 are adaptive but off unless asked; the 5.x generation
  // thinks by default. Opus 5 and Sonnet 5 still accept `disabled`; the 5.5
  // pair and anything later do not.
  return {
    thinking: 'adaptive',
    thinksByDefault: v.major >= 5,
    acceptsDisabled: v.major < 5 || (v.major === 5 && v.minor === 0),
    rejectsSampling: true,
  };
}


/**
 * Anthropic's tool `input_schema` rejects `oneOf`/`allOf`/`anyOf` at the top
 * level — the request 400s with:
 *   tools.N.custom.input_schema: input_schema does not support oneOf, allOf,
 *   or anyOf at the top level
 *
 * Strip them defensively so a single misconfigured tool can't take down the
 * whole agent (we lose the schema-level "exactly one of these shapes"
 * validation, but the runtime handler validates per-command anyway). Nested
 * `oneOf`/`anyOf`/`allOf` inside `properties.*` are NOT touched — Anthropic
 * accepts those.
 */
function sanitizeAnthropicInputSchema(schema: unknown): Record<string, unknown> {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
    return { type: 'object', properties: {} };
  }
  const { oneOf: _oneOf, allOf: _allOf, anyOf: _anyOf, ...rest } = schema as Record<string, unknown>;
  return rest;
}

/**
 * Anthropic's wire format requires both `tool_use.id` (assistant side) and
 * `tool_result.tool_use_id` (user side) to match this regex. Empty strings,
 * `|ts:`-suffixed Gemini IDs, dotted IDs, etc. all 400 the entire request.
 * Used by `convertMessages` to drop tool_call / tool_result rows whose ids
 * wouldn't survive the wire check.
 */
const ANTHROPIC_TOOL_ID_RE = /^[a-zA-Z0-9_-]+$/;

// ────────────────────────────────────────────────────────────────
// Prompt caching
// ────────────────────────────────────────────────────────────────
//
// Caching is a prefix match over `tools` -> `system` -> `messages`. The
// markers below are placed only when the caller sets `promptCache`, and only
// on block-level fields: a top-level `cache_control` (the API's automatic
// mode) is accepted by the direct API but refused with a 400 by the legacy
// Bedrock integration, which reuses this body for the InvokeModel wire, so
// this adapter never emits one. The rules the API enforces and this code
// guards: at most four breakpoints per request, never on an empty block, and
// a breakpoint looks back at most 20 positions for the previous entry.
//
// The four breakpoints, in prefix order:
//   BP1  the last tool definition: a read point for the tool list on its own.
//   BP2  the last leading system block marked `stable` (or the first system
//        block): tools plus the stable instructions, shared by every request
//        of the same agent.
//   BP3  the last non-empty block of the final message: the growing
//        conversation. Placed only when the request carries tools or history,
//        so a one-shot request pays no write premium with nothing to read it.
//   BP4  an anchor on the last human user message when more than
//        LOOKBACK_ANCHOR_POSITIONS positions follow it, so the next request's
//        BP3 still finds a cache entry inside the 20-position window.

const EPHEMERAL_CACHE = { type: 'ephemeral' } as const;
const MAX_CACHE_BREAKPOINTS = 4;
const LOOKBACK_ANCHOR_POSITIONS = 15;

/**
 * Tool definitions a breakpoint may sit on: function tools (no `type` once
 * mapped) and the Anthropic-defined client tools. A server tool (web search,
 * web fetch, code execution) is left unmarked; BP2 still covers the whole
 * tool list because system renders after tools.
 */
function cacheableToolDefinition(tool: Record<string, unknown>): boolean {
  const type = tool.type;
  if (type === undefined) return true;
  return typeof type === 'string' && /^(bash|text_editor|computer|memory)_\d{8}$/.test(type);
}

function isNonEmptyBlock(block: AnthropicContentBlock): boolean {
  switch (block.type) {
    case 'text': return block.text.length > 0;
    case 'tool_result': return typeof block.content === 'string' ? block.content.length > 0 : true;
    case 'tool_use': return true;
    case 'image':
    case 'document': return block.source.data.length > 0;
    default: return false;
  }
}

type AnthropicWireMessage = { role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] };

/**
 * Put a breakpoint on the last non-empty block of `message`, converting
 * string content to a text block first. False when nothing qualifies, so
 * an empty block is never marked.
 */
function markLastNonEmptyBlock(message: AnthropicWireMessage): boolean {
  if (typeof message.content === 'string') {
    if (message.content.length === 0) return false;
    message.content = [{ type: 'text', text: message.content, cache_control: EPHEMERAL_CACHE } as AnthropicContentBlock];
    return true;
  }
  for (let i = message.content.length - 1; i >= 0; i--) {
    const block = message.content[i];
    if (!isNonEmptyBlock(block)) continue;
    message.content[i] = { ...block, cache_control: EPHEMERAL_CACHE } as AnthropicContentBlock;
    return true;
  }
  return false;
}

/** A user message a person wrote: not a tool-result carrier. */
function isHumanUserMessage(message: AnthropicWireMessage): boolean {
  if (message.role !== 'user') return false;
  if (typeof message.content === 'string') return true;
  return !message.content.some((b) => b.type === 'tool_result');
}

/**
 * Place the breakpoints on an already-built body. Mutates `body.tools`,
 * `body.system` and `body.messages` in place; the count is bounded by
 * construction (one marker per rule) and checked again at the end.
 */
function applyPromptCacheBreakpoints(
  body: Record<string, unknown>,
  opts: { stableSystemIndex: number; hasTools: boolean; hasHistory: boolean },
): void {
  let placed = 0;
  const tools = body.tools as Array<Record<string, unknown>> | undefined;
  if (tools && tools.length > 0 && cacheableToolDefinition(tools[tools.length - 1]) && placed < MAX_CACHE_BREAKPOINTS) {
    tools[tools.length - 1] = { ...tools[tools.length - 1], cache_control: EPHEMERAL_CACHE };
    placed++;
  }
  const system = body.system as Array<{ type: 'text'; text: string; cache_control?: unknown }> | undefined;
  if (system && system.length > 0 && placed < MAX_CACHE_BREAKPOINTS) {
    const index = opts.stableSystemIndex >= 0 && opts.stableSystemIndex < system.length ? opts.stableSystemIndex : 0;
    if (system[index].text.length > 0) {
      system[index] = { ...system[index], cache_control: EPHEMERAL_CACHE };
      placed++;
    }
  }
  const messages = body.messages as AnthropicWireMessage[];
  if (messages.length > 0 && (opts.hasTools || opts.hasHistory) && placed < MAX_CACHE_BREAKPOINTS) {
    if (markLastNonEmptyBlock(messages[messages.length - 1])) placed++;
  }
  if (placed < MAX_CACHE_BREAKPOINTS) {
    for (let i = messages.length - 2; i >= 0; i--) {
      if (!isHumanUserMessage(messages[i])) continue;
      if (messages.length - 1 - i > LOOKBACK_ANCHOR_POSITIONS && markLastNonEmptyBlock(messages[i])) placed++;
      break;
    }
  }
  // Belt and braces: the rules above place at most one marker each.
  if (placed > MAX_CACHE_BREAKPOINTS) throw new Error(`prompt cache: ${placed} breakpoints placed, the API allows ${MAX_CACHE_BREAKPOINTS}`);
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
      // Ephemeral data URLs (tool screenshots) inline directly — no
      // content-library resolution needed.
      const dataMedia = parseDataUrl(url);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = dataMedia
        ? { kind: 'binary' as const, mimeType: dataMedia.mimeType, base64: dataMedia.base64 }
        : (canonical ? resolvedMedia?.get(canonical) : undefined);
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

    const { system, stableSystemIndex, messages } = convertMessages(params.messages, modelName, params.resolvedMedia);

    const caps = claudeCapabilities(modelName);
    const level = normalizeThinking(params.thinking);
    // Thinking is "on" when (a) the model supports it AND (b) the caller
    // didn't explicitly say `off`. Undefined means "use the default".
    const thinking = caps.thinking !== 'none' && level !== 'off';
    const budget = thinking
      ? (level === undefined ? ANTHROPIC_DEFAULT_BUDGET : ANTHROPIC_THINKING_BUDGETS[level])
      : 0;
    const defaultMaxTokens = thinking ? 16_000 : 4096;
    const maxTokens = Math.max(params.maxTokens ?? defaultMaxTokens, thinking ? budget + 1 : 0);

    // No top-level `cache_control` here, ever: that is the API's automatic
    // caching, which the legacy Bedrock integration (the InvokeModel wire
    // reuses this body) refuses with a 400. Caching is explicit, block-level
    // and opt-in through `params.promptCache` (see the prompt caching section
    // above). A stream-level error of any kind is thrown by parseStream, so an
    // unknown field can no longer resolve as a silent empty reply.
    const body: Record<string, unknown> = {
      model: modelName,
      messages,
      stream,
      max_tokens: maxTokens,
    };
    if (thinking) {
      if (caps.thinking === 'adaptive') {
        body.thinking = { type: 'adaptive' };
        // Adaptive thinking uses `output_config.effort` instead of a token
        // budget. Map our canonical level: undefined defaults to medium for
        // parity with the previous behaviour.
        const effort = level === undefined ? 'medium' : level;
        body.output_config = { effort };
      } else {
        body.thinking = { type: 'enabled', budget_tokens: budget };
      }
    } else if (level === 'off' && caps.thinksByDefault && caps.acceptsDisabled) {
      // A model that thinks when the field is omitted needs an explicit
      // `disabled`, which it accepts only at effort high or below; `low` is
      // the fastest first token with thinking off. Models that think by
      // default and reject `disabled` keep their own default instead.
      body.thinking = { type: 'disabled' };
      body.output_config = { effort: 'low' };
    }
    if (params.temperature != null && !thinking && !caps.rejectsSampling) body.temperature = params.temperature;
    if (system.length > 0) body.system = system;

    // Beta-header markers travel as `beta` on each native tool entry.
    // Collect, dedupe and emit as a single
    // `anthropic-beta` header. The `beta` field is stripped from the per-tool
    // body below — Anthropic 400s on unknown fields inside the tool object.
    const betas = new Set<string>();
    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools.map(t => {
        if (isFunctionTool(t)) {
          return {
            name: t.function.name,
            description: t.function.description,
            input_schema: sanitizeAnthropicInputSchema(t.function.parameters),
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

    if (params.promptCache) {
      applyPromptCacheBreakpoints(body, {
        stableSystemIndex,
        hasTools: !!(params.tools && params.tools.length > 0),
        hasHistory: params.messages.filter((m) => m.role !== 'system').length > 1,
      });
    }
    // Blocks are needed only to carry a cache marker. Without one, send the
    // system prompt as the single joined string this adapter always sent, so
    // a request that does not ask for caching is byte-identical to before
    // (and matches the other SDKs' adapters).
    const systemBlocks = body.system as Array<{ type: 'text'; text: string; cache_control?: unknown }> | undefined;
    if (systemBlocks && !systemBlocks.some((b) => b.cache_control)) {
      body.system = systemBlocks.map((b) => b.text).join('\n\n');
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
    // Why the model stopped, from `message_delta.delta.stop_reason`, reported
    // once after the stream as a `finish` chunk, as the other adapters
    // (`completions.ts`, `google.ts`) report theirs. Without it a Claude
    // refusal (HTTP 200 with `stop_reason: "refusal"`, on Anthropic's API and
    // on Amazon Bedrock alike) would reach the caller as an ordinary empty
    // answer with no reason. `blocked` marks the refusal the same way
    // `content_filter` is marked in those adapters.
    let stopReason: string | null = null;

    for await (const chunk of readSSEStream(response)) {
      const data = chunk as Record<string, unknown>;

      // Anthropic emits errors mid-stream as `event: error\ndata: {"type":"error","error":{...}}`
      // on an HTTP 200 response. Surface them as thrown errors instead of
      // silently terminating with zero output — otherwise an invalid body
      // field (e.g. an unknown top-level key, an unsupported `thinking` shape
      // for this model) looks indistinguishable from a successful empty reply.
      if (data.type === 'error' || data.__sseEvent === 'error') {
        const err = (data.error ?? data) as { type?: string; message?: string };
        const kind = err?.type ?? 'unknown_error';
        const message = err?.message ?? JSON.stringify(data).slice(0, 500);
        throw new Error(`Anthropic stream error: ${kind}: ${message}`);
      }

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
        // The usage on `message_delta` is CUMULATIVE for the whole message
        // and the final word on it: a server tool run (web search) adds
        // input after `message_start` has reported its count, so the
        // documented shape is 2,679 input tokens at the start and 10,682 at
        // the end. Every field present here overrides the start value; a
        // field the event leaves out (Bedrock's deltas carry output only)
        // keeps what `message_start` said.
        const usage = data.usage as {
          output_tokens?: number;
          input_tokens?: number;
          cache_read_input_tokens?: number;
          cache_creation_input_tokens?: number;
        } | undefined;
        if (usage) {
          if (typeof usage.output_tokens === 'number') outputTokens = usage.output_tokens;
          if (typeof usage.input_tokens === 'number') inputTokens = usage.input_tokens;
          if (typeof usage.cache_read_input_tokens === 'number') cacheReadInputTokens = usage.cache_read_input_tokens;
          if (typeof usage.cache_creation_input_tokens === 'number') cacheCreationInputTokens = usage.cache_creation_input_tokens;
        }
        const delta = data.delta as { stop_reason?: unknown } | undefined;
        if (typeof delta?.stop_reason === 'string' && delta.stop_reason) stopReason = delta.stop_reason;
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

    if (stopReason) {
      yield { type: 'finish', reason: stopReason, ...(stopReason === 'refusal' ? { blocked: true } : {}) };
    }
  },
};

type AnthropicCacheControl = { type: 'ephemeral' };

type AnthropicContentBlock = (
  | { type: 'text'; text: string }
  | { type: 'image'; source: { type: 'base64'; media_type: string; data: string } }
  | { type: 'document'; source: { type: 'base64'; media_type: string; data: string } }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean }
) & { cache_control?: AnthropicCacheControl };

type AnthropicSystemBlock = { type: 'text'; text: string; cache_control?: AnthropicCacheControl };

/**
 * Convert OpenAI-format messages to Anthropic format.
 *
 * System messages: every LEADING one (before the first non-system message)
 * becomes its own top-level `system` text block, in order, so the first
 * block can stay byte-identical across requests while later blocks change
 * (a cache breakpoint on a joined string would move with every change
 * anywhere in it). A system message that arrives mid-conversation is
 * rendered IN PLACE, as a text block appended to the adjacent user turn
 * after any tool_result blocks (or as a user message of its own when the
 * previous turn is the assistant's), instead of being hoisted into
 * `system`: hoisting it would change the prefix ahead of the whole history
 * and invalidate every cached turn. Empty system text is dropped: an empty
 * block is a 400.
 *
 * Also converts tool_calls to tool_use blocks, tool results to tool_result
 * blocks, and UAMP content_items to Anthropic blocks.
 */
function convertMessages(
  messages: Message[],
  modelName: string,
  resolvedMedia?: ResolvedMediaMap,
): {
  system: AnthropicSystemBlock[];
  /** Index into `system` of the last leading block marked `stable`, or -1. */
  stableSystemIndex: number;
  messages: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }>;
} {
  const system: AnthropicSystemBlock[] = [];
  let stableSystemIndex = -1;
  let leading = true;
  const result: Array<{ role: 'user' | 'assistant'; content: string | AnthropicContentBlock[] }> = [];

  // Id-less calls and outputs get stable, paired ids first (./tool-ids.ts),
  // so a turn the regex guards below used to drop is replayed instead.
  for (const msg of normalizeToolHistory(messages)) {
    if (msg.role === 'system') {
      const text = typeof msg.content === 'string' ? msg.content : '';
      if (!text) continue;
      if (leading) {
        system.push({ type: 'text', text });
        if (msg.stable) stableSystemIndex = system.length - 1;
        continue;
      }
      const prev = result[result.length - 1];
      if (prev && prev.role === 'user') {
        const blocks: AnthropicContentBlock[] = Array.isArray(prev.content)
          ? prev.content
          : (prev.content ? [{ type: 'text', text: prev.content }] : []);
        blocks.push({ type: 'text', text });
        prev.content = blocks;
      } else {
        result.push({ role: 'user', content: [{ type: 'text', text }] });
      }
      continue;
    }
    leading = false;

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
          // Anthropic enforces tool_use.id matches ^[a-zA-Z0-9_-]+$. Skip
          // tool_calls whose id wouldn't survive the wire-format check —
          // emitting them would 400 the entire request. The matching tool
          // result is dropped below for the same reason, so the assistant /
          // user pairing stays consistent.
          if (!tc.id || !ANTHROPIC_TOOL_ID_RE.test(tc.id)) continue;
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
      // Same wire-format guard as the assistant tool_use branch above. A
      // tool_result with an empty / non-conforming tool_use_id would 400
      // every request with
      //   messages.N.content.0.tool_result.tool_use_id: String should match
      //   pattern '^[a-zA-Z0-9_-]+$'
      // Drop the entire row — the corresponding assistant tool_use was also
      // dropped (it failed the same regex), so there is nothing to pair
      // against and the LLM would be unable to reference this turn anyway.
      if (!msg.tool_call_id || !ANTHROPIC_TOOL_ID_RE.test(msg.tool_call_id)) {
        continue;
      }
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
          tool_use_id: msg.tool_call_id,
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
      // running text_editor. Mirrors the openai/google adapters. The symptom:
      // msg.content="Create a file…" while the outgoing user message carried
      // only `[Available image: …]`.
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
    // something to merge or reorder). When the single next user message
    // already covers all required ids with its tool_result blocks FIRST, it
    // is left untouched, trailing text included: a mid-conversation system
    // message rendered after the tool results (convertMessages above) must
    // stay in that message, not be split into a message of its own.
    const resultsFirst = (blocks: AnthropicContentBlock[]): boolean => {
      const firstOther = blocks.findIndex((b) => b.type !== 'tool_result');
      return firstOther < 0 || !blocks.slice(firstOther).some((b) => b.type === 'tool_result');
    };
    const singleWellFormed = j === i + 2 && requiredIds.size === 0
      && Array.isArray(result[i + 1].content) && resultsFirst(result[i + 1].content as AnthropicContentBlock[]);
    if (!singleWellFormed && j > i + 1 && (toolResultBlocks.length > 0 || trailingBlocks.length > 0)) {
      if (toolResultBlocks.length > 0) {
        reordered.push({ role: 'user', content: toolResultBlocks });
      }
      if (trailingBlocks.length > 0) {
        reordered.push({ role: 'user', content: trailingBlocks });
      }
      i = j - 1; // skip the forward messages we just merged
    }
  }

  return { system, stableSystemIndex, messages: reordered };
}

export default anthropicAdapter;
