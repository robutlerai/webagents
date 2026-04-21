/**
 * Google Gemini LLM Adapter
 *
 * Handles message conversion (OpenAI -> Gemini contents), request building,
 * SSE stream parsing, inline image extraction, function calls with thought_signature,
 * and usage reporting.
 *
 * Extracted from the battle-tested proxy implementation in lib/llm/uamp-proxy.ts.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message } from './types';
import { isFunctionTool } from './types';
import { readSSEStream } from './sse';
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, type ResolvedMediaMap, type DescribeContentOptions } from './content';

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

const MODEL_API_ALIASES: Record<string, string> = {
  'gemini-3.1-pro': 'gemini-3.1-pro-preview',
  'gemini-3-flash': 'gemini-3-flash-preview',
  'gemini-3.1-flash-image': 'gemini-3.1-flash-image-preview',
  'gemini-3-pro-image': 'gemini-3-pro-image-preview',
  'gemini-3.1-flash-lite': 'gemini-3.1-flash-lite-preview',
};

/**
 * Gemini's function_declarations use a restricted OpenAPI 3.0 dialect over
 * Protocol Buffer typing. This sanitizer applies six rules in one pass:
 *
 *   1. Coerce numeric/boolean `enum` values to strings — Gemini requires
 *      string enums and `type: STRING` whenever `enum` is present.
 *   2. Drop fields the `parameters` Schema parser rejects outright. Sending
 *      any returns HTTP 400 ("Unknown name 'additionalProperties'", etc.)
 *      and aborts the whole generateContent call.
 *   3. Reconcile unions with sibling schemas. Per the Vertex function-
 *      calling docs (and matching OpenCode anomalyco/opencode#14509),
 *      when a node has `anyOf` / `oneOf`, sibling keys are rejected.
 *      But blindly stripping siblings nukes the top-level parameters
 *      schema (`type: 'object', properties: {…}, oneOf: […]` → bare
 *      `{anyOf: […]}`), which Gemini also rejects with a path-less
 *      `INVALID_ARGUMENT`. So: if the node has a useful object/array
 *      shape (type, properties, or items), drop the union and keep the
 *      shape. Otherwise, keep the union and strip siblings.
 *   4. Rewrite `const: X` to `enum: [X]` with `type: 'string'`. Gemini
 *      drops `const` entirely (rule 2), so without this rewrite a
 *      discriminator like `command: { const: 'create' }` becomes the
 *      empty schema `command: {}`, which makes the parent property look
 *      typeless and triggers another path-less `INVALID_ARGUMENT`.
 *   5. Infer missing `type` from sibling fields (`properties` ⇒ object,
 *      `items` ⇒ array) — Gemini rejects those keys without an explicit
 *      `type`, even though they're sufficient evidence in JSON Schema.
 *   6. Filter `required` to keys actually declared in the same node's
 *      `properties`. JSON Schema permits cross-node references in
 *      discriminated unions; Gemini insists every entry be local.
 *
 * Source schemas keep the dropped fields, so OpenAI strict mode and
 * Anthropic still get the tighter contract — the rewrite is purely on the
 * Gemini wire boundary.
 *
 * Unsupported as of the Nov 2025 expanded-schema update (verified against
 * `googleapis/googleapis google/ai/generativelanguage/v1beta` proto and
 * the live `generativelanguage.googleapis.com/v1beta` parameters path):
 *   additionalProperties, allOf, not, const, $ref, $defs, definitions.
 *
 * Notably NOT in this list (these are now supported):
 *   anyOf, oneOf — but see rules (3)/(4) above for the reconciliation.
 */
const GEMINI_UNSUPPORTED_KEYS = new Set([
  'additionalProperties',
  'allOf',
  'not',
  'const',
  '$ref',
  '$defs',
  'definitions',
]);

const GEMINI_UNION_KEYS = ['anyOf', 'oneOf'] as const;

function sanitizeSchemaForGemini(schema: unknown): unknown {
  if (schema == null || typeof schema !== 'object') return schema;
  if (Array.isArray(schema)) return schema.map(sanitizeSchemaForGemini);

  const obj = schema as Record<string, unknown>;
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (GEMINI_UNSUPPORTED_KEYS.has(key)) continue;
    if (key === 'enum' && Array.isArray(value)) {
      out[key] = value.map(v => typeof v === 'string' ? v : String(v));
    } else if (typeof value === 'object' && value !== null) {
      out[key] = sanitizeSchemaForGemini(value);
    } else {
      out[key] = value;
    }
  }

  // Rule (4): rewrite `const: X` (which we just stripped above) to
  // `enum: [X]` with explicit `type: 'string'`. Gemini doesn't support
  // `const` directly, but a single-element string `enum` carries the
  // exact "must equal X" semantic and IS supported. Without this,
  // discriminators inside `oneOf` branches (`command: { const: 'create' }`)
  // collapse to type-less empty schemas (`command: {}`), and Gemini
  // rejects the whole request with a path-less INVALID_ARGUMENT.
  if ('const' in obj && obj.const !== undefined && obj.const !== null) {
    const c = obj.const;
    out.enum = [typeof c === 'string' ? c : String(c)];
    if (out.type === undefined) out.type = 'string';
  }

  // Rule (5): infer missing `type` from sibling fields. Gemini's validator
  // rejects `properties` / `required` unless `type` is `object`, and
  // `items` / `prefixItems` unless `type` is `array`. Schemas that omit
  // `type` (legal in JSON Schema, common in MCP tools) would otherwise
  // 400 the entire request. See OpenCode anomalyco/opencode#14509.
  // (Done before Rule 3 so the union-vs-shape decision sees inferred types.)
  if (out.type === undefined) {
    if ('properties' in out || 'required' in out || 'propertyOrdering' in out) {
      out.type = 'object';
    } else if ('items' in out || 'prefixItems' in out) {
      out.type = 'array';
    }
  }

  // Rule (3): reconcile unions with sibling schemas. Gemini rejects
  // siblings of `anyOf`/`oneOf`, but it ALSO rejects a top-level
  // parameters schema reduced to bare `{anyOf: [...]}` with no type
  // (path-less INVALID_ARGUMENT). When the node has a useful object/array
  // shape we drop the union and keep the shape; otherwise we keep the
  // union and strip siblings.
  const presentUnionKey = GEMINI_UNION_KEYS.find((k) => Array.isArray(out[k]));
  if (presentUnionKey) {
    const hasObjectShape = out.type === 'object' || 'properties' in out;
    const hasArrayShape = out.type === 'array' || 'items' in out;
    if (hasObjectShape || hasArrayShape) {
      delete out[presentUnionKey];
    } else {
      const union = out[presentUnionKey];
      return { [presentUnionKey === 'oneOf' ? 'anyOf' : presentUnionKey]: union };
    }
  }

  // Rule (6): every entry in `required` must be declared in `properties`
  // at the same node. JSON Schema permits the discriminated-union pattern
  // we use in MEMORY_TOOL (parent declares all props, each oneOf branch
  // re-asserts only its own discriminator + the subset that becomes
  // required), but Gemini's validator rejects it with
  //   "parameters.any_of[N].required[i]: property is not defined".
  // Filter `required` to known props so the call goes through; the parent
  // schema still carries the full `properties` map so the model has the
  // type info it needs to fill the call.
  if (Array.isArray(out.required)) {
    const props = (out.properties && typeof out.properties === 'object')
      ? new Set(Object.keys(out.properties as Record<string, unknown>))
      : null;
    const filtered = props
      ? (out.required as unknown[]).filter((k) => typeof k === 'string' && props.has(k))
      : [];
    if (filtered.length > 0) {
      out.required = filtered;
    } else {
      delete out.required;
    }
  }

  // Rule (1) cont.: enum requires STRING type.
  if (Array.isArray(out.enum) && out.type && out.type !== 'string') {
    out.type = 'string';
  }

  return out;
}

export const googleAdapter: LLMAdapter = {
  name: 'google',

  mediaSupport: {
    image: 'base64',
    audio: 'base64',
    video: 'base64',
    document: 'base64',
  } satisfies MediaSupport,

  buildRequest(params: AdapterRequestParams): AdapterRequest {
    const rawName = params.model.includes('/') ? params.model.split('/').pop()! : params.model;
    const modelName = MODEL_API_ALIASES[rawName] ?? rawName;
    const stream = params.stream !== false;
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    const sseParam = stream ? '&alt=sse' : '';

    const { systemParts, contents } = convertMessages(params.messages, params.resolvedMedia);

    const generationConfig: Record<string, unknown> = {};
    if (params.temperature != null) generationConfig.temperature = params.temperature;
    if (params.maxTokens != null) generationConfig.maxOutputTokens = params.maxTokens;

    if (rawName.includes('-image')) {
      generationConfig.responseModalities = ['TEXT', 'IMAGE'];
    }
    if (params.responseModalities) {
      generationConfig.responseModalities = params.responseModalities;
    }

    const isGemini3 = /^gemini-3/.test(modelName);
    const isGemini25 = /^gemini-2\.5/.test(modelName);

    if (params.thinking !== false && (isGemini3 || isGemini25)) {
      const thinkingConfig: Record<string, unknown> = { includeThoughts: true };
      if (isGemini3) {
        // Flash-Lite defaults to 'minimal' (effectively no thinking);
        // explicitly set 'low' so thinking actually engages.
        // Flash and Pro default to 'high' (dynamic) which is fine as-is.
        if (modelName.includes('flash-lite')) {
          thinkingConfig.thinkingLevel = 'low';
        }
      }
      generationConfig.thinkingConfig = thinkingConfig;
    } else if (params.thinking === false) {
      if (isGemini3) {
        generationConfig.thinkingConfig = { thinkingLevel: 'minimal' };
      } else if (isGemini25) {
        generationConfig.thinkingConfig = { thinkingBudget: 0 };
      }
    }

    const body: Record<string, unknown> = { contents };
    if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
    if (systemParts.length > 0) {
      body.system_instruction = { parts: systemParts };
    }

    if (params.tools && params.tools.length > 0) {
      const toolsArray: Record<string, unknown>[] = [];

      const funcTools = params.tools.filter(isFunctionTool);
      if (funcTools.length > 0) {
        toolsArray.push({
          function_declarations: funcTools.map(t => ({
            name: t.function.name,
            description: t.function.description || '',
            parameters: sanitizeSchemaForGemini(t.function.parameters || { type: 'object', properties: {} }),
          })),
        });
      }

      for (const t of params.tools) {
        if (isFunctionTool(t)) continue;
        const { type, ...config } = t;
        toolsArray.push({ [type]: Object.keys(config).length > 0 ? config : {} });
      }

      if (toolsArray.length > 0) {
        body.tools = toolsArray;
        const hasFunc = funcTools.length > 0;
        const hasBuiltIn = params.tools.some(t => !isFunctionTool(t));
        if (hasFunc && hasBuiltIn) {
          body.tool_config = {
            function_calling_config: { mode: 'AUTO' },
            include_server_side_tool_invocations: true,
          };
        }
      }
    }

    return {
      url: `${BASE_URL}/models/${modelName}:${action}?key=${params.apiKey}${sseParam}`,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    };
  },

  async *parseStream(response: Response): AsyncGenerator<AdapterChunk> {
    let toolCallIndex = 0;
    const nonce = Math.random().toString(36).slice(2, 8);

    for await (const chunk of readSSEStream(response)) {
      const data = chunk as Record<string, unknown>;
      const candidates = data.candidates as Array<Record<string, unknown>> | undefined;

      if (Array.isArray(candidates)) {
        for (const candidate of candidates) {
          const content = candidate.content as { parts?: Array<Record<string, unknown>> } | undefined;
          const parts = content?.parts;
          if (!Array.isArray(parts)) continue;

          for (const part of parts) {
            if (part.thought) {
              yield { type: 'thinking', text: (part.text as string) || '' };
              continue;
            }

            if (part.text) {
              yield { type: 'text', text: part.text as string };
            }

            if (part.inlineData) {
              const inline = part.inlineData as { mimeType: string; data: string };
              if (inline.mimeType && inline.data) {
                yield {
                  type: 'image',
                  base64: inline.data,
                  mimeType: inline.mimeType,
                  ...((part.thought_signature || part.thoughtSignature) ? { thoughtSignature: (part.thought_signature ?? part.thoughtSignature) as string } : {}),
                };
              }
            }

            if (part.functionCall) {
              const fc = part.functionCall as {
                name: string;
                args?: Record<string, unknown>;
              };
              const ts = (part.thought_signature || part.thoughtSignature) as string | undefined;
              const baseCid = `call_${nonce}_${toolCallIndex++}`;
              const callId = ts
                ? `${baseCid}|ts:${ts}`
                : baseCid;
              yield {
                type: 'tool_call',
                id: callId,
                name: fc.name,
                arguments: JSON.stringify(fc.args ?? {}),
              };
            }

            if (part.executableCode) {
              const ec = part.executableCode as { language?: string; code: string };
              yield {
                type: 'tool_progress',
                call_id: 'code_execution',
                text: `\`\`\`${ec.language ?? 'python'}\n${ec.code}\n\`\`\``,
              };
            }

            if (part.codeExecutionResult) {
              const cer = part.codeExecutionResult as { outcome?: string; output?: string };
              yield {
                type: 'tool_result',
                call_id: 'code_execution',
                result: cer.output ?? '',
                status: cer.outcome,
              };
            }
          }
        }
      }

      const grounding = data.groundingMetadata as Record<string, unknown> | undefined;
      if (grounding?.searchEntryPoint) {
        const sep = grounding.searchEntryPoint as { renderedContent?: string };
        if (sep.renderedContent) {
          yield {
            type: 'tool_result',
            call_id: 'web_search',
            result: sep.renderedContent,
          };
        }
      }

      const usage = data.usageMetadata as Record<string, number> | undefined;
      if (usage) {
        const cachedTokens = usage.cachedContentTokenCount ?? usage.cached_content_token_count ?? 0;
        yield {
          type: 'usage',
          input: usage.promptTokenCount ?? usage.prompt_token_count ?? 0,
          output: usage.candidatesTokenCount ?? usage.candidates_token_count ?? 0,
          ...(cachedTokens > 0 && { cache_read_input: cachedTokens }),
        };
      }
    }
  },
};

const GOOGLE_DESCRIBE_OPTIONS: DescribeContentOptions = {
  supportedModalities: new Set(['image', 'audio', 'video']),
};

/**
 * Convert UAMP content items to Gemini parts.
 */
function uampToGeminiParts(
  items: Array<Record<string, unknown>>,
  resolvedMedia?: ResolvedMediaMap,
): unknown[] {
  const parts: unknown[] = [];
  for (const item of items) {
    if (item.type === 'text' && item.text) {
      parts.push({ text: item.text });
    } else if (item.type === 'image' || item.type === 'video' || item.type === 'audio') {
      const ref = item.image || item.video || item.audio;
      const url = extractContentRef(ref);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media?.kind === 'binary') {
        const mediaPart: Record<string, unknown> = {
          inlineData: { mimeType: media.mimeType, data: media.base64 },
        };
        if (media.thoughtSignature) mediaPart.thought_signature = media.thoughtSignature;
        parts.push(mediaPart);
        const desc = (item as { description?: string }).description;
        const cid = (item as { content_id?: string }).content_id;
        if (desc) {
          parts.push({ text: `(${desc}${cid ? ` [${cid}]` : ''})` });
        } else if (cid) {
          parts.push({ text: `(content: ${cid})` });
        }
      } else {
        parts.push({ text: describeContentItem(item, GOOGLE_DESCRIBE_OPTIONS) });
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media?.kind === 'binary') {
        parts.push({ inlineData: { mimeType: media.mimeType, data: media.base64 } });
      } else if (media?.kind === 'text') {
        // Gemini's `inlineData` requires base64. For text-bearing files we
        // emit a plain `text` part wrapped in <file> tags so the model gets
        // the same structural cue without paying the 33% base64 inflation
        // tax through the proxy → adapter pipeline.
        const filename = (item as { filename?: string }).filename;
        const safeName = (filename || 'file').replace(/[<>"]/g, '');
        parts.push({ text: `<file name="${safeName}" mime="${media.mimeType}">\n${media.text}\n</file>` });
      } else if ((item as Record<string, unknown>)._extracted_text) {
        parts.push({ text: (item as Record<string, unknown>)._extracted_text as string });
      } else {
        parts.push({ text: describeContentItem(item, GOOGLE_DESCRIBE_OPTIONS) });
      }
    }
  }
  return parts.length > 0 ? parts : [{ text: '(no content)' }];
}

/**
 * Convert OpenAI-format messages to Gemini contents format.
 * Handles system messages, UAMP content_items, tool calls with thought_signature,
 * tool results, and inline image resolution from resolved media.
 */
function convertMessages(
  messages: Message[],
  resolvedMedia?: ResolvedMediaMap,
): { systemParts: Array<{ text: string }>; contents: Array<{ role: string; parts: unknown[] }> } {
  const systemParts: Array<{ text: string }> = [];
  const contents: Array<{ role: string; parts: unknown[] }> = [];

  for (const m of messages) {
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : '';
      if (text) systemParts.push({ text });
      continue;
    }

    // Assistant with tool calls -> native functionCall parts (always)
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const tsMarker = '|ts:';
      const parts: unknown[] = [];
      if (m.content && typeof m.content === 'string') parts.push({ text: m.content });
      for (const tc of m.tool_calls) {
        let args: Record<string, unknown> = {};
        try { args = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
        const part: Record<string, unknown> = { functionCall: { name: tc.function.name, args } };
        if (tc.id.includes(tsMarker)) {
          part.thought_signature = tc.id.slice(tc.id.indexOf(tsMarker) + tsMarker.length);
        }
        parts.push(part);
      }
      contents.push({ role: 'model', parts });
      continue;
    }

    // Tool result -> native functionResponse (no media inlined, text metadata only)
    if (m.role === 'tool') {
      const toolUampItems = (Array.isArray(m.content_items) && m.content_items.length > 0
        && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? m.content_items
        : null;

      const mediaDescParts: unknown[] = [];
      if (toolUampItems) {
        for (const item of toolUampItems) {
          if (item.type === 'text' && (item as { text?: string }).text) {
            mediaDescParts.push({ text: (item as { text: string }).text });
            continue;
          }
          if (['image', 'audio', 'video', 'file'].includes(item.type as string)) {
            mediaDescParts.push({ text: describeContentItem(item, GOOGLE_DESCRIBE_OPTIONS) });
          }
        }
      }

      let response: unknown;
      const text = typeof m.content === 'string' ? m.content : '';
      try { response = JSON.parse(text || '""'); } catch { response = text || ''; }
      const toolName = m.name || m.tool_call_id || 'unknown';
      const parts: unknown[] = [{
        functionResponse: { name: toolName, id: m.tool_call_id, response: { result: response } },
      }];
      if (mediaDescParts.length > 0) parts.push(...mediaDescParts);
      contents.push({ role: 'user', parts });
      continue;
    }

    // Regular user/assistant message
    const role = m.role === 'assistant' ? 'model' : 'user';

    // Detect UAMP content_items (content array or content_items field)
    const uampItems = (Array.isArray(m.content) && isUAMPContentArray(m.content))
      ? m.content as Array<Record<string, unknown>>
      : (Array.isArray(m.content_items) && m.content_items.length > 0
          && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? m.content_items
        : null;

    if (uampItems) {
      const geminiParts = uampToGeminiParts(uampItems, resolvedMedia);
      if (typeof m.content === 'string' && m.content.trim()) {
        geminiParts.unshift({ text: m.content });
      }
      contents.push({ role, parts: geminiParts });
      continue;
    }

    const content = typeof m.content === 'string'
      ? m.content
      : Array.isArray(m.content)
        ? (m.content as Array<{ type?: string; text?: string }>)
            .filter(p => p.type === 'text' && p.text)
            .map(p => p.text!)
            .join('\n')
        : (m.content || '') as string;

    contents.push({ role, parts: [{ text: content }] });
  }

  return { systemParts, contents };
}

export default googleAdapter;
