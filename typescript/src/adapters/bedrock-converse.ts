/**
 * Amazon Bedrock's Converse API as an SDK adapter, plus the transcoders that
 * turn a Converse answer into Chat Completions SSE or JSON.
 *
 * Converse is the only chat API Bedrock offers for several model families
 * (Llama 3.x and 4, DeepSeek R1, Nova, Pixtral, Palmyra and the older Mistral
 * models, among others). This adapter calls it with a Bedrock API key sent as
 * a bearer token and a region, without the AWS SDK.
 *
 * Design: transcode at the boundary and reuse an existing parser. The answer
 * is decoded from Bedrock's binary event stream (`./bedrock-eventstream.ts`)
 * and re-emitted as OpenAI Chat Completions SSE, which the SDK's own Chat
 * Completions parser (`./completions.ts`) then reads. Tool-call assembly,
 * progress chunks, `reasoning_content` as thinking and the finish chunk are
 * therefore the same code every OpenAI-compatible provider runs through, and
 * an HTTP client handed the transcoded stream reads a format it already
 * knows.
 *
 * Request mapping (`buildConverseBody`), from OpenAI-shaped messages:
 *   - `system` messages become `system: [{text}]`, one block each, in order.
 *   - user text becomes `{text}`; blank text is skipped (Converse rejects it).
 *     Media is described as text (this adapter is text only), or refused with
 *     `media: 'refuse'` when the caller can route it to another provider.
 *   - assistant `tool_calls` become `toolUse` blocks; `''` or unparseable
 *     arguments become `{}`, as the Anthropic adapter does.
 *   - `tool` messages become `toolResult` blocks in a user turn.
 *   - consecutive same-role turns merge; in a user turn the `toolResult`
 *     blocks go first.
 *   - tool ids pass through when they match AWS's pattern
 *     `^[a-zA-Z0-9_.:-]{1,64}$`, else map to `t_` + 32 hex of their SHA-256,
 *     deterministically, so a call and its result always meet.
 *   - `toolConfig` only when the request carries tools; no `toolChoice`.
 *
 * Refused at build time (`BedrockRequestUnsupported`), each one an AWS 400
 * otherwise: a non-function tool; tools when `toolCalling` is not set; a tool
 * name outside `^[a-zA-Z0-9_-]{1,64}$` among the request's tools (a past
 * call's name is made to fit instead, `converseHistoryToolName`, so a tool
 * loop's re-request is never refused halfway); no messages, or a first turn
 * not from the user; a thinking level the model cannot honour; `maxTokens`
 * over `maxOutputTokens` when `overCap: 'refuse'` (otherwise it is clamped).
 * With `toolHistory: 'refuse'` also tool blocks in the history with no tools
 * in the request (Converse demands `toolConfig` then) and an unpaired tool
 * call or result; with `adapt` (the default) the first is written as text and
 * the second is sent for AWS to judge (see `buildConverseBody`).
 *
 * Usage accounting differs between the two shapes. Converse reports Anthropic
 * semantics for every model: `inputTokens` EXCLUDES cached tokens ("total
 * input tokens = inputTokens + cacheReadInputTokens + cacheWriteInputTokens",
 * AWS prompt caching documentation). Chat Completions `prompt_tokens`
 * INCLUDES them, with the cached share in `prompt_tokens_details.cached_tokens`.
 * The transcoders add the cache counts back, otherwise any consumer metering
 * the Chat Completions shape would under-count every cached token.
 */
import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message, ToolDefinition } from './types';
import { normalizeThinking } from './types';
import { createChatCompletionsAdapter } from './completions';
import { normalizeToolHistory } from './tool-ids';
import { extractContentRef, canonicalContentUrl, describeContentItem, isUAMPContentArray, parseDataUrl, type ResolvedMediaMap } from './content';
import {
  bedrockEventStreamToSse,
  isEventStream,
  bedrockBaseUrl,
  BedrockRequestUnsupported,
  BedrockStreamError,
  type BedrockSseEncoder,
} from './bedrock-eventstream';

export interface BedrockConverseOptions {
  /** Adapter name. Default `bedrock-converse`. */
  name?: string;
  /** The bedrock-runtime base URL; wins over `region`. */
  baseUrl?: string;
  /** Region for the default base URL. Default `us-east-1`. */
  region?: string;
  /**
   * The id in the URL path (a base model id, an inference profile id such as
   * `us.meta.llama3-3-70b-instruct-v1:0`, or an ARN). Default: the last
   * segment of `params.model`. Never a string from an untrusted request.
   */
  modelId?: string | ((params: AdapterRequestParams) => string);
  /** The model takes client tools on Converse. Default false: a request with tools is refused. */
  toolCalling?: boolean;
  /** The model's output ceiling on Bedrock. */
  maxOutputTokens?: number;
  /** Over `maxOutputTokens`: clamp (default) or refuse. */
  overCap?: 'clamp' | 'refuse';
  /** `always`: the model reasons whatever is asked, so `off` is refused. Otherwise a requested level is refused (this adapter sends no reasoning configuration). */
  reasoning?: 'none' | 'always';
  /** Media in the conversation: described as text (default), or refused when the caller can route it to a provider that can see it. */
  media?: 'describe' | 'refuse';
  /**
   * Tool history Converse would reject. `adapt` (default): with no tools in
   * the request, past tool calls and results are written as text (Converse
   * demands `toolConfig` beside any `toolUse` or `toolResult` block); with
   * tools, the pairing of calls and results is left to AWS, which answers an
   * unpaired call with the same 400 other providers return. `refuse`: both
   * are refused at build time, for a caller that can route the request to
   * another provider. See `buildConverseBody` for why a tool loop needs
   * `adapt`.
   */
  toolHistory?: 'adapt' | 'refuse';
}

/** AWS's pattern for `toolUseId`. */
const TOOL_ID_RE = /^[a-zA-Z0-9_.:-]{1,64}$/;
/** AWS's pattern for tool names (`toolSpec.name`, `toolUse.name`). */
const TOOL_NAME_RE = /^[a-zA-Z0-9_-]{1,64}$/;

const CONVERSE_MEDIA: MediaSupport = { image: 'none', audio: 'none', video: 'none', document: 'none' };
const DESCRIBE_TEXT_ONLY = { supportedModalities: new Set<string>() };

// --- SHA-256, synchronous and platform-neutral (for `converseToolUseId`) ----

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** SHA-256 of a string's UTF-8 bytes, as lowercase hex. */
export function sha256Hex(text: string): string {
  const msg = new TextEncoder().encode(text);
  const bitLength = msg.length * 8;
  const padded = new Uint8Array(((msg.length + 9 + 63) >> 6) << 6);
  padded.set(msg);
  padded[msg.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(bitLength / 0x100000000), false);
  view.setUint32(padded.length - 4, bitLength >>> 0, false);
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K[i] + w[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + hh) >>> 0;
  }
  return Array.from(h, (x) => x.toString(16).padStart(8, '0')).join('');
}

/**
 * The name a PAST tool call is replayed under. A name the model itself
 * produced can fall outside AWS's pattern (a dotted `functions.get_time`, a
 * space), and the request's own tools already passed it (a tool whose name
 * fails is refused at build, `toolConfigOf`), so such a name was never one
 * of the request's tools. Refusing it would be wrong in one case that
 * matters: a caller's tool loop re-sends the history after every tool round,
 * and a refusal there ends the turn halfway, whereas a refusal of the first
 * request lets the caller route it elsewhere. So the name is made to fit:
 * every character outside the pattern becomes `_`, cut to 64, `tool` when
 * nothing is left. Deterministic, like `converseToolUseId`.
 */
export function converseHistoryToolName(name: string): string {
  if (TOOL_NAME_RE.test(name)) return name;
  return name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'tool';
}

/**
 * The `toolUseId` Converse receives for a tool call id: the id itself when it
 * fits AWS's pattern, else `t_` + 32 hex of its SHA-256. Deterministic, so a
 * call and its result map to the same id on every request.
 */
export function converseToolUseId(id: string): string {
  return TOOL_ID_RE.test(id) ? id : `t_${sha256Hex(id).slice(0, 32)}`;
}

// --- Request -----------------------------------------------------------------

type ConverseBlock = Record<string, unknown>;
interface ConverseTurn { role: 'user' | 'assistant'; content: ConverseBlock[] }

function refuse(reason: string): never {
  throw new BedrockRequestUnsupported(reason);
}

function textOf(content: Message['content']): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((p) => (p && (p.type === 'text' || p.type === undefined) && typeof p.text === 'string' ? p.text : ''))
    .filter(Boolean)
    .join('\n');
}

function uampItemsOf(m: Message): Array<Record<string, unknown>> | null {
  if (Array.isArray(m.content) && isUAMPContentArray(m.content)) return m.content as Array<Record<string, unknown>>;
  if (Array.isArray(m.content_items) && m.content_items.length > 0
    && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string')) {
    return m.content_items;
  }
  return null;
}

/** Whether an item carries media a vision-capable provider would actually see (a data URL or resolved bytes). */
function inlineableMedia(item: Record<string, unknown>, resolved?: ResolvedMediaMap): boolean {
  const type = String(item.type ?? '');
  if (type !== 'image' && type !== 'audio' && type !== 'video' && type !== 'file') return false;
  const url = extractContentRef(item[type]);
  if (parseDataUrl(url)) return type !== 'file';
  const canonical = url ? canonicalContentUrl(url, item.content_id as string | undefined) : null;
  const media = canonical ? resolved?.get(canonical) : undefined;
  return media?.kind === 'binary';
}

function fileAsText(item: Record<string, unknown>, resolved?: ResolvedMediaMap): string | null {
  const url = extractContentRef(item.file);
  const canonical = url ? canonicalContentUrl(url, item.content_id as string | undefined) : null;
  const media = canonical ? resolved?.get(canonical) : undefined;
  const name = String(item.filename ?? 'file').replace(/[<>"]/g, '');
  if (media?.kind === 'text') return `<file name="${name}" mime="${media.mimeType}">\n${media.text}\n</file>`;
  const extracted = item._extracted_text;
  if (typeof extracted === 'string' && extracted) {
    return `<file name="${name}" mime="${String(item.mime_type ?? 'application/octet-stream')}">\n${extracted}\n</file>`;
  }
  return null;
}

function itemBlocks(items: Array<Record<string, unknown>>, opts: BedrockConverseOptions, resolved?: ResolvedMediaMap): ConverseBlock[] {
  const out: ConverseBlock[] = [];
  for (const item of items) {
    if (item.type === 'text') {
      if (typeof item.text === 'string' && item.text.trim()) out.push({ text: item.text });
      continue;
    }
    if (item.type === 'file') {
      const inlined = fileAsText(item, resolved);
      if (inlined) { out.push({ text: inlined }); continue; }
    }
    if (opts.media === 'refuse' && inlineableMedia(item, resolved)) refuse('media input');
    out.push({ text: describeContentItem(item, DESCRIBE_TEXT_ONLY) });
  }
  return out;
}

/** Content blocks of a user or assistant message's own content (not its tool calls). */
function contentBlocks(m: Message, opts: BedrockConverseOptions, resolved?: ResolvedMediaMap): ConverseBlock[] {
  const items = uampItemsOf(m);
  if (items) {
    const blocks = itemBlocks(items, opts, resolved);
    if (typeof m.content === 'string' && m.content.trim()) {
      const first = blocks.find((b) => typeof b.text === 'string');
      if (!first || first.text !== m.content) blocks.unshift({ text: m.content });
    }
    return blocks;
  }
  if (Array.isArray(m.content)) {
    const blocks: ConverseBlock[] = [];
    for (const part of m.content) {
      if (!part) continue;
      if ((part.type === 'text' || part.type === undefined) && typeof part.text === 'string') {
        if (part.text.trim()) blocks.push({ text: part.text });
      } else if (part.type === 'image_url' || part.type === 'input_audio' || part.type === 'file') {
        if (opts.media === 'refuse') refuse('media input');
        blocks.push({ text: '[An attachment was here; this model reads text only.]' });
      }
    }
    return blocks;
  }
  return typeof m.content === 'string' && m.content.trim() ? [{ text: m.content }] : [];
}

function parseArguments(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function toolSchema(parameters: unknown): Record<string, unknown> {
  const schema = parameters && typeof parameters === 'object' && !Array.isArray(parameters)
    ? { ...(parameters as Record<string, unknown>) }
    : {};
  delete schema.oneOf;
  delete schema.anyOf;
  delete schema.allOf;
  if (schema.type === undefined) schema.type = 'object';
  if (schema.type === 'object' && schema.properties === undefined) schema.properties = {};
  return schema;
}

function toolConfigOf(tools: ToolDefinition[] | undefined, opts: BedrockConverseOptions): Record<string, unknown> | null {
  if (!tools || tools.length === 0) return null;
  for (const t of tools) {
    const type = (t as { type?: string }).type ?? 'function';
    if (type !== 'function' || !('function' in t)) refuse(`tool ${type}`);
  }
  if (opts.toolCalling !== true) refuse('tools on a model without tool calling on Converse');
  return {
    tools: tools.map((t) => {
      const fn = (t as { function: { name: string; description?: string; parameters?: unknown } }).function;
      if (!TOOL_NAME_RE.test(fn.name ?? '')) refuse(`tool name ${String(fn.name).slice(0, 80)}`);
      return {
        toolSpec: {
          name: fn.name,
          ...(fn.description && fn.description.trim() ? { description: fn.description } : {}),
          inputSchema: { json: toolSchema(fn.parameters) },
        },
      };
    }),
  };
}

/**
 * Past tool calls and results as text blocks, in place (`toolHistory:
 * 'adapt'` with no tools in the request). The model still reads what it did
 * and what came back; Converse sees no tool block, so it needs no
 * `toolConfig`. Names are the ones the caller sent (text takes any name).
 */
function flattenToolHistory(turns: ConverseTurn[], callNames: Map<string, string>): void {
  for (const turn of turns) {
    turn.content = turn.content.map((b) => {
      const use = b.toolUse as { toolUseId: string; name: string; input: unknown } | undefined;
      if (use) return { text: `[called ${callNames.get(use.toolUseId) ?? use.name}(${JSON.stringify(use.input)})]` };
      const result = b.toolResult as { toolUseId: string; content: Array<{ text: string }> } | undefined;
      if (result) {
        const name = callNames.get(result.toolUseId);
        return { text: `[result${name ? ` of ${name}` : ''}]\n${result.content.map((c) => c.text).join('\n')}` };
      }
      return b;
    });
  }
}

/**
 * The Converse request body for `params` (see the header for the mapping and
 * the refusals).
 *
 * Why `toolHistory: 'adapt'` is the default: a tool loop itself produces
 * history that strict validation would refuse.
 *   - A host that executes some tools server-side and leaves others to its
 *     client may, when one round calls both kinds, answer only its own calls
 *     and replay the round, so the re-request carries an unpaired call. Other
 *     providers answer that with a 400 the host already handles through its
 *     normal error path; a refusal thrown at build time would bypass that
 *     path instead.
 *   - An agent's wrap-up call after its tool budget is spent sends the full
 *     history with no tools, which Converse rejects unless the history is
 *     written as text.
 * A caller with no other provider is better served by both behaviours than
 * by a refusal it cannot route anywhere; a caller with another provider
 * passes `refuse` and moves the request at build time.
 */
export function buildConverseBody(params: AdapterRequestParams, opts: BedrockConverseOptions = {}): Record<string, unknown> {
  const level = normalizeThinking(params.thinking);
  if (opts.reasoning === 'always' && level === 'off') refuse('thinking off on a model that always reasons');
  if (opts.reasoning !== 'always' && level !== undefined && level !== 'off') refuse(`thinking ${level}`);

  const system: ConverseBlock[] = [];
  const turns: ConverseTurn[] = [];
  /** `toolUseId` to the name the caller sent, for history written as text. */
  const callNames = new Map<string, string>();
  // An id-less call or output gets a stable id (./tool-ids.ts); what stays
  // unpaired is this adapter's own `toolHistory` decision, so it is kept.
  for (const m of normalizeToolHistory(params.messages, { unpaired: 'keep' })) {
    if (m.role === 'system' || m.role === 'developer') {
      const text = textOf(m.content);
      if (text.trim()) system.push({ text });
      continue;
    }
    if (m.role === 'user') {
      turns.push({ role: 'user', content: contentBlocks(m, opts, params.resolvedMedia) });
      continue;
    }
    if (m.role === 'assistant') {
      const blocks = contentBlocks(m, opts, params.resolvedMedia);
      for (const tc of m.tool_calls ?? []) {
        if (!tc.id) refuse('a tool call without an id');
        const toolUseId = converseToolUseId(tc.id);
        const name = tc.function?.name ?? '';
        if (name) callNames.set(toolUseId, name);
        blocks.push({ toolUse: { toolUseId, name: converseHistoryToolName(name), input: parseArguments(tc.function?.arguments) } });
      }
      turns.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (m.role === 'tool') {
      if (!m.tool_call_id) refuse('a tool result without an id');
      let text = textOf(m.content);
      for (const item of uampItemsOf(m) ?? []) {
        if (item.type === 'image' || item.type === 'audio' || item.type === 'video' || item.type === 'file') {
          text += `\n${describeContentItem(item, DESCRIBE_TEXT_ONLY)}`;
        }
      }
      turns.push({
        role: 'user',
        content: [{ toolResult: { toolUseId: converseToolUseId(m.tool_call_id), content: [{ text: text.trim() ? text : '(no output)' }] } }],
      });
      continue;
    }
    refuse(`message role ${String(m.role).slice(0, 40)}`);
  }

  // Merge same-role neighbours (after dropping empty turns), tool results first in a user turn.
  const merged: ConverseTurn[] = [];
  for (const turn of turns) {
    if (turn.content.length === 0) continue;
    const last = merged[merged.length - 1];
    if (last && last.role === turn.role) last.content.push(...turn.content);
    else merged.push({ role: turn.role, content: [...turn.content] });
  }
  for (const turn of merged) {
    if (turn.role !== 'user') continue;
    const results = turn.content.filter((b) => b.toolResult);
    if (results.length > 0) turn.content = [...results, ...turn.content.filter((b) => !b.toolResult)];
  }

  if (merged.length === 0) refuse('no messages');
  if (merged[0].role !== 'user') refuse('the first message is not from the user');

  // Every assistant tool call answered in full by the next user turn, and no
  // result without its call. Checked only under `refuse`: under `adapt`
  // (see above) the history goes as built and AWS judges the pairing.
  const refuseHistory = opts.toolHistory === 'refuse';
  let historyHasTools = false;
  for (let i = 0; i < merged.length; i++) {
    const turn = merged[i];
    if (turn.role === 'assistant') {
      const calls = turn.content.filter((b) => b.toolUse).map((b) => (b.toolUse as { toolUseId: string }).toolUseId);
      if (calls.length === 0) continue;
      historyHasTools = true;
      if (!refuseHistory) continue;
      const next = merged[i + 1];
      const answered = new Set((next?.content ?? []).filter((b) => b.toolResult).map((b) => (b.toolResult as { toolUseId: string }).toolUseId));
      for (const id of calls) if (!answered.has(id)) refuse('a tool call without its result');
    } else {
      const results = turn.content.filter((b) => b.toolResult).map((b) => (b.toolResult as { toolUseId: string }).toolUseId);
      if (results.length === 0) continue;
      historyHasTools = true;
      if (!refuseHistory) continue;
      const prev = merged[i - 1];
      const called = new Set((prev?.role === 'assistant' ? prev.content : []).filter((b) => b.toolUse).map((b) => (b.toolUse as { toolUseId: string }).toolUseId));
      for (const id of results) if (!called.has(id)) refuse('a tool result without its call');
    }
  }

  const toolConfig = toolConfigOf(params.tools, opts);
  if (historyHasTools && !toolConfig) {
    if (refuseHistory) refuse('tool history without tools in the request');
    flattenToolHistory(merged, callNames);
  }

  const inference: Record<string, unknown> = {};
  if (params.maxTokens != null) {
    let max = params.maxTokens;
    if (opts.maxOutputTokens && max > opts.maxOutputTokens) {
      if (opts.overCap === 'refuse') refuse(`max_tokens ${max} over the model's ${opts.maxOutputTokens}`);
      max = opts.maxOutputTokens;
    }
    inference.maxTokens = max;
  }
  if (params.temperature != null) inference.temperature = params.temperature;

  const body: Record<string, unknown> = { messages: merged };
  if (system.length > 0) body.system = system;
  if (Object.keys(inference).length > 0) body.inferenceConfig = inference;
  if (toolConfig) body.toolConfig = toolConfig;
  return body;
}

function resolveModelId(params: AdapterRequestParams, opt: BedrockConverseOptions['modelId']): string {
  if (typeof opt === 'function') return opt(params);
  if (typeof opt === 'string') return opt;
  return params.model.includes('/') ? params.model.split('/').pop()! : params.model;
}

/** The Converse request: URL `.../model/{id}/converse-stream` (or `/converse`), bearer key, JSON body. */
export function buildBedrockConverseRequest(params: AdapterRequestParams, opts: BedrockConverseOptions = {}): AdapterRequest {
  const stream = params.stream !== false;
  const body = buildConverseBody(params, opts);
  const base = bedrockBaseUrl(opts);
  return {
    url: `${base}/model/${encodeURIComponent(resolveModelId(params, opts.modelId))}/${stream ? 'converse-stream' : 'converse'}`,
    headers: {
      'content-type': 'application/json',
      accept: stream ? 'application/vnd.amazon.eventstream' : 'application/json',
      authorization: `Bearer ${params.apiKey}`,
    },
    body: JSON.stringify(body),
  };
}

// --- Response ----------------------------------------------------------------

/**
 * Converse stop reason to Chat Completions `finish_reason`. `tool_calls` is
 * required for a tool use: the Chat Completions parser releases buffered tool
 * calls only on `tool_calls` or `stop`. A malformed tool use keeps its own
 * name, so no half-formed call is released.
 */
export function converseFinishReason(stopReason: unknown): string {
  switch (stopReason) {
    case 'end_turn':
    case 'stop_sequence': return 'stop';
    case 'tool_use': return 'tool_calls';
    case 'max_tokens':
    case 'model_context_window_exceeded': return 'length';
    case 'guardrail_intervened':
    case 'content_filtered': return 'content_filter';
    default: return typeof stopReason === 'string' && stopReason ? stopReason : 'stop';
  }
}

const count = (n: unknown): number => (typeof n === 'number' && Number.isFinite(n) && n > 0 ? Math.floor(n) : 0);

/** Converse `usage` in the Chat Completions shape: the cache legs added back into `prompt_tokens` (see the header). */
export function converseUsageToChat(usage: unknown): { prompt_tokens: number; completion_tokens: number; total_tokens: number; prompt_tokens_details: { cached_tokens: number } } {
  const u = (usage && typeof usage === 'object' ? usage : {}) as Record<string, unknown>;
  const cacheRead = count(u.cacheReadInputTokens);
  const prompt = count(u.inputTokens) + cacheRead + count(u.cacheWriteInputTokens);
  const completion = count(u.outputTokens);
  return { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion, prompt_tokens_details: { cached_tokens: cacheRead } };
}

export interface ConverseTranscodeOptions {
  /** The `model` written into every chunk (the id the request named). */
  model: string;
  /** Default `chatcmpl-bedrock-<uuid>`. */
  id?: string;
  /** Unix seconds. Default now. */
  created?: number;
  /**
   * Some models (Amazon Nova) write their reasoning inline, as a
   * `<thinking>...</thinking>` span inside ordinary text, instead of in a
   * reasoning block. When set, such spans are sent as `reasoning_content`
   * rather than `content`, and the whitespace a removed span leaves at the
   * start of the answer is trimmed. Off by default: other models' text is
   * passed through untouched.
   */
  inlineThinkingTags?: boolean;
}

const THINKING_OPEN = '<thinking>';
const THINKING_CLOSE = '</thinking>';

/** Text split into what the reader sees and what the model reasoned. */
export interface ThinkingSplit {
  content: string;
  reasoning: string;
}

/**
 * Splits streamed text into answer and inline reasoning, for models that wrap
 * their reasoning in `<thinking>...</thinking>`. A tag may arrive split across
 * chunks, so a tail that could still become a tag is held back until the next
 * chunk (or `end`). Text inside an unclosed span at the end counts as
 * reasoning. Leading whitespace of the answer is dropped until its first
 * visible character, so "<thinking>...</thinking> Hello" reads "Hello".
 */
export function createInlineThinkingSplitter(): { push(text: string): ThinkingSplit; end(): ThinkingSplit } {
  let inside = false;
  let held = '';
  let answerStarted = false;
  const answer = (text: string): string => {
    if (answerStarted) return text;
    const trimmed = text.replace(/^\s+/, '');
    if (trimmed) answerStarted = true;
    return trimmed;
  };
  return {
    push(text) {
      held += text;
      let content = '';
      let reasoning = '';
      for (;;) {
        const tag = inside ? THINKING_CLOSE : THINKING_OPEN;
        const at = held.indexOf(tag);
        if (at >= 0) {
          if (inside) reasoning += held.slice(0, at);
          else content += answer(held.slice(0, at));
          held = held.slice(at + tag.length);
          inside = !inside;
          continue;
        }
        let keep = 0;
        for (let k = Math.min(tag.length - 1, held.length); k > 0; k--) {
          if (tag.startsWith(held.slice(held.length - k))) { keep = k; break; }
        }
        const out = held.slice(0, held.length - keep);
        if (inside) reasoning += out;
        else content += answer(out);
        held = held.slice(held.length - keep);
        return { content, reasoning };
      }
    },
    end() {
      const rest = held;
      held = '';
      return inside ? { content: '', reasoning: rest } : { content: answer(rest), reasoning: '' };
    },
  };
}

function newId(): string {
  const uuid = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto?.randomUUID?.()
    ?? `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
  return `chatcmpl-bedrock-${uuid}`;
}

/** The encoder from ConverseStream events to Chat Completions SSE. */
export function createConverseChatSseEncoder(opts: ConverseTranscodeOptions): BedrockSseEncoder {
  const head = { id: opts.id ?? newId(), object: 'chat.completion.chunk', created: opts.created ?? Math.floor(Date.now() / 1000), model: opts.model };
  // The n-th tool block of the message is `index: n`, keyed by its contentBlockIndex.
  const toolIndex = new Map<number, number>();
  const line = (rest: Record<string, unknown>) => `data: ${JSON.stringify({ ...head, ...rest })}\n\n`;
  const delta = (d: Record<string, unknown>) => line({ choices: [{ index: 0, delta: d, finish_reason: null }] });
  const splitter = opts.inlineThinkingTags ? createInlineThinkingSplitter() : null;
  const splitDeltas = (part: ThinkingSplit): string =>
    (part.reasoning ? delta({ reasoning_content: part.reasoning }) : '') + (part.content ? delta({ content: part.content }) : '');
  let flushed = false;
  const flush = (): string => {
    if (!splitter || flushed) return '';
    flushed = true;
    return splitDeltas(splitter.end());
  };
  return {
    frame(frame) {
      const j = frame.json;
      switch (frame.type) {
        case 'messageStart':
          return delta({ role: 'assistant' });
        case 'contentBlockStart': {
          const start = (j.start ?? {}) as { toolUse?: { toolUseId?: string; name?: string } };
          if (!start.toolUse) return '';
          const block = count(j.contentBlockIndex);
          const index = toolIndex.size;
          toolIndex.set(block, index);
          return delta({ tool_calls: [{ index, id: start.toolUse.toolUseId ?? '', type: 'function', function: { name: start.toolUse.name ?? '', arguments: '' } }] });
        }
        case 'contentBlockDelta': {
          const d = (j.delta ?? {}) as { text?: unknown; reasoningContent?: { text?: unknown }; toolUse?: { input?: unknown } };
          if (typeof d.text === 'string') {
            if (!d.text) return '';
            return splitter ? splitDeltas(splitter.push(d.text)) : delta({ content: d.text });
          }
          if (d.reasoningContent && typeof d.reasoningContent.text === 'string') {
            return d.reasoningContent.text ? delta({ reasoning_content: d.reasoningContent.text }) : '';
          }
          if (d.toolUse && typeof d.toolUse.input === 'string') {
            const block = count(j.contentBlockIndex);
            let index = toolIndex.get(block);
            if (index === undefined) { index = toolIndex.size; toolIndex.set(block, index); }
            return d.toolUse.input ? delta({ tool_calls: [{ index, function: { arguments: d.toolUse.input } }] }) : '';
          }
          return '';
        }
        case 'messageStop':
          return flush() + line({ choices: [{ index: 0, delta: {}, finish_reason: converseFinishReason(j.stopReason) }] });
        case 'metadata':
          return j.usage ? line({ choices: [], usage: converseUsageToChat(j.usage) }) : '';
        default:
          return '';
      }
    },
    // `messageStart` says only `role: assistant` and `contentBlockStart`
    // only opens a tool block (its id and name, none of its input): priming
    // holds them, so an exception straight after them still falls back
    // (./bedrock-eventstream.ts).
    opening(frame) {
      return frame.type === 'messageStart' || frame.type === 'contentBlockStart';
    },
    // After the first output a Converse exception ends the stream as an error:
    // Chat Completions has no in-band error event every client reads.
    exception() {
      return null;
    },
    end() {
      return flush() + 'data: [DONE]\n\n';
    },
  };
}

/**
 * A 2xx ConverseStream body as Chat Completions SSE, primed: an exception (or
 * a malformed or empty stream) before the first output resolves to a
 * synthetic non-2xx `Response` instead.
 */
export function converseStreamToChatCompletionsSSE(res: Response, opts: ConverseTranscodeOptions): Promise<Response> {
  return bedrockEventStreamToSse(res, createConverseChatSseEncoder(opts));
}

/** A non-stream Converse answer as a Chat Completions `chat.completion` object. */
export function converseJsonToChatCompletion(json: unknown, opts: ConverseTranscodeOptions): Record<string, unknown> {
  const j = (json && typeof json === 'object' ? json : {}) as Record<string, unknown>;
  const message = ((j.output as Record<string, unknown> | undefined)?.message ?? {}) as { content?: unknown };
  const blocks = Array.isArray(message.content) ? message.content as Array<Record<string, unknown>> : [];
  const text: string[] = [];
  const reasoning: string[] = [];
  const toolCalls: Array<Record<string, unknown>> = [];
  for (const b of blocks) {
    if (typeof b.text === 'string') text.push(b.text);
    const r = b.reasoningContent as { reasoningText?: { text?: unknown } } | undefined;
    if (r?.reasoningText && typeof r.reasoningText.text === 'string') reasoning.push(r.reasoningText.text);
    const tu = b.toolUse as { toolUseId?: string; name?: string; input?: unknown } | undefined;
    if (tu) toolCalls.push({ id: tu.toolUseId ?? '', type: 'function', function: { name: tu.name ?? '', arguments: JSON.stringify(tu.input ?? {}) } });
  }
  let answer = text.length > 0 ? text.join('') : null;
  if (answer !== null && opts.inlineThinkingTags) {
    const splitter = createInlineThinkingSplitter();
    const first = splitter.push(answer);
    const last = splitter.end();
    if (first.reasoning || last.reasoning) reasoning.unshift(first.reasoning + last.reasoning);
    answer = first.content + last.content;
  }
  const out: Record<string, unknown> = { role: 'assistant', content: answer };
  if (reasoning.length > 0) out.reasoning_content = reasoning.join('');
  if (toolCalls.length > 0) out.tool_calls = toolCalls;
  return {
    id: opts.id ?? newId(),
    object: 'chat.completion',
    created: opts.created ?? Math.floor(Date.now() / 1000),
    model: opts.model,
    choices: [{ index: 0, message: out, finish_reason: converseFinishReason(j.stopReason) }],
    usage: converseUsageToChat(j.usage),
  };
}

/** The model-independent Chat Completions parser the transcoded stream is read with. */
const chatParser = createChatCompletionsAdapter({ name: 'bedrock-converse', baseUrl: 'https://bedrock-runtime.invalid' });

/** Parse a Converse stream (raw event stream or already transcoded Chat Completions SSE). */
export async function* parseBedrockConverseStream(
  response: Response,
  model = 'bedrock',
  opts: Omit<ConverseTranscodeOptions, 'model'> = {},
): AsyncGenerator<AdapterChunk> {
  let sse = response;
  if (isEventStream(response)) {
    sse = await converseStreamToChatCompletionsSSE(response, { ...opts, model });
    if (!sse.ok) {
      const said = await sse.json().catch(() => ({})) as { type?: string; message?: string };
      throw new BedrockStreamError(said.type ?? 'exception', said.message ?? `HTTP ${sse.status}`);
    }
  }
  yield* chatParser.parseStream(sse);
}

/**
 * An SDK adapter for Bedrock ConverseStream: give it a region (or a base
 * URL), the model id and what the model supports there, and pass the Bedrock
 * API key as `apiKey`. Text only (`mediaSupport` all `none`).
 */
export function createBedrockConverseAdapter(opts: BedrockConverseOptions = {}): LLMAdapter {
  return {
    name: opts.name ?? 'bedrock-converse',
    mediaSupport: CONVERSE_MEDIA,
    buildRequest(params: AdapterRequestParams): AdapterRequest {
      return buildBedrockConverseRequest(params, opts);
    },
    parseStream(response: Response) {
      return parseBedrockConverseStream(response, typeof opts.modelId === 'string' ? opts.modelId : 'bedrock');
    },
  };
}
