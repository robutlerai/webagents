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
import { extractContentRef, isUAMPContentArray, canonicalContentUrl, type ResolvedMediaMap } from './content';

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

const MODEL_API_ALIASES: Record<string, string> = {
  'gemini-3.1-pro': 'gemini-3.1-pro-preview',
  'gemini-3-flash': 'gemini-3-flash-preview',
  'gemini-3.1-flash-image': 'gemini-3.1-flash-image-preview',
  'gemini-3-pro-image': 'gemini-3-pro-image-preview',
  'gemini-3.1-flash-lite': 'gemini-3.1-flash-lite-preview',
};

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
            parameters: t.function.parameters || { type: 'object', properties: {} },
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
      if (url) {
        const canonical = canonicalContentUrl(url);
        const media = canonical ? resolvedMedia?.get(canonical) : undefined;
        if (media) {
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
        }
      }
    } else if (item.type === 'file') {
      const url = extractContentRef(item.file);
      const canonical = url ? canonicalContentUrl(url) : null;
      const media = canonical ? resolvedMedia?.get(canonical) : undefined;
      if (media) {
        parts.push({ inlineData: { mimeType: media.mimeType, data: media.base64 } });
      } else if ((item as Record<string, unknown>)._extracted_text) {
        parts.push({ text: (item as Record<string, unknown>)._extracted_text as string });
      } else {
        const fname = (item.filename as string) || 'file';
        const mime = (item.mime_type as string) || 'unknown';
        parts.push({ text: `[Attached file: ${fname} (${mime}) — content not available inline]` });
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

    // Tool result -> native functionResponse (always)
    if (m.role === 'tool') {
      // Detect UAMP content_items on tool result messages
      const toolUampItems = (Array.isArray(m.content_items) && m.content_items.length > 0
        && m.content_items.every((i: Record<string, unknown>) => i && typeof i.type === 'string'))
        ? m.content_items
        : null;
      const mediaParts = toolUampItems
        ? uampToGeminiParts(toolUampItems, resolvedMedia).filter((p: unknown) => (p as Record<string, unknown>).inlineData)
        : [];

      let response: unknown;
      const text = typeof m.content === 'string' ? m.content : '';
      try { response = JSON.parse(text || '""'); } catch { response = text || ''; }
      const toolName = m.name || m.tool_call_id || 'unknown';
      const parts: unknown[] = [{ functionResponse: { name: toolName, response: { result: response } } }];
      if (mediaParts.length > 0) parts.push(...mediaParts);
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
