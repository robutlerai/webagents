/**
 * Google Gemini LLM Adapter
 *
 * Handles message conversion (OpenAI -> Gemini contents), request building,
 * SSE stream parsing, inline image extraction, function calls with thought_signature,
 * and usage reporting.
 *
 * Extracted from the battle-tested proxy implementation in lib/llm/uamp-proxy.ts.
 */

import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk, MediaSupport, Message } from './types.js';
import { readSSEStream } from './sse.js';

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';
const MD_IMAGE_RE = /!\[([^\]]*)\]\((\/api\/content\/[0-9a-f-]{36})\)/g;
// Matches content URLs in any form: md image with relative/absolute URL, or bare relative/absolute URL
const COMBINED_MEDIA_RE = /(?:!\[([^\]]*)\]\(((?:https?:\/\/[^)]+)?\/api\/content\/[0-9a-f-]{36})\))|((?:https?:\/\/[^\s]+)?\/api\/content\/[0-9a-f-]{36})/g;
const UUID_EXTRACT = /\/api\/content\/([0-9a-f-]{36})/;

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

    const body: Record<string, unknown> = { contents };
    if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
    if (systemParts.length > 0) {
      body.system_instruction = { parts: systemParts };
    }

    if (params.tools && params.tools.length > 0) {
      const declarations = params.tools
        .filter(t => t.function)
        .map(t => ({
          name: t.function.name,
          description: t.function.description || '',
          parameters: t.function.parameters || { type: 'object', properties: {} },
        }));
      if (declarations.length > 0) {
        body.tools = [{ function_declarations: declarations }];
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

    for await (const chunk of readSSEStream(response)) {
      const data = chunk as Record<string, unknown>;
      const candidates = data.candidates as Array<Record<string, unknown>> | undefined;

      if (Array.isArray(candidates)) {
        for (const candidate of candidates) {
          const content = candidate.content as { parts?: Array<Record<string, unknown>> } | undefined;
          const parts = content?.parts;
          if (!Array.isArray(parts)) continue;

          for (const part of parts) {
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
                thought_signature?: string;
              };
              const baseCid = `call_gemini_${toolCallIndex++}`;
              const callId = fc.thought_signature
                ? `${baseCid}|ts:${fc.thought_signature}`
                : baseCid;
              yield {
                type: 'tool_call',
                id: callId,
                name: fc.name,
                arguments: JSON.stringify(fc.args ?? {}),
              };
            }
          }
        }
      }

      const usage = data.usageMetadata as Record<string, number> | undefined;
      if (usage) {
        yield {
          type: 'usage',
          input: usage.promptTokenCount ?? usage.prompt_token_count ?? 0,
          output: usage.candidatesTokenCount ?? usage.candidates_token_count ?? 0,
        };
      }
    }
  },
};

/**
 * Convert OpenAI-format messages to Gemini contents format.
 * Handles system messages, tool calls with thought_signature, tool results,
 * and inline image resolution from resolved media.
 */
function convertMessages(
  messages: Message[],
  resolvedMedia?: Map<string, { mimeType: string; base64: string; thoughtSignature?: string }>,
): { systemParts: Array<{ text: string }>; contents: Array<{ role: string; parts: unknown[] }> } {
  const systemParts: Array<{ text: string }> = [];
  const contents: Array<{ role: string; parts: unknown[] }> = [];

  for (const m of messages) {
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content : '';
      if (text) systemParts.push({ text });
      continue;
    }

    // Assistant with tool calls -> functionCall parts
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const tsMarker = '|ts:';
      const hasSignatures = m.tool_calls.every(tc => tc.id.includes(tsMarker));

      if (hasSignatures) {
        const parts: unknown[] = [];
        if (m.content && typeof m.content === 'string') parts.push({ text: m.content });
        for (const tc of m.tool_calls) {
          let args: Record<string, unknown> = {};
          try { args = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
          const fcPart: Record<string, unknown> = { name: tc.function.name, args };
          fcPart.thought_signature = tc.id.slice(tc.id.indexOf(tsMarker) + tsMarker.length);
          parts.push({ functionCall: fcPart });
        }
        contents.push({ role: 'model', parts });
      } else {
        const summary = m.tool_calls
          .map(tc => `[Called tool ${tc.function.name}(${tc.function.arguments})]`)
          .join('\n');
        const text = typeof m.content === 'string' ? m.content : '';
        contents.push({
          role: 'model',
          parts: [{ text: (text ? text + '\n' : '') + summary }],
        });
      }
      continue;
    }

    // Tool result -> functionResponse
    if (m.role === 'tool') {
      const matchingAssistant = messages.slice(0, messages.indexOf(m)).reverse()
        .find(prev => prev.role === 'assistant' && prev.tool_calls?.length);
      const hasSig = matchingAssistant?.tool_calls?.some(tc => tc.id.includes('|ts:'));

      if (hasSig) {
        let response: unknown;
        const text = typeof m.content === 'string' ? m.content : '';
        try { response = JSON.parse(text || '""'); } catch { response = text || ''; }
        const toolName = m.name || m.tool_call_id || 'unknown';
        contents.push({
          role: 'user',
          parts: [{ functionResponse: { name: toolName, response: { result: response } } }],
        });
      } else {
        const toolName = m.name || m.tool_call_id || 'tool';
        const text = typeof m.content === 'string' ? m.content : '';
        const truncated = text.slice(0, 2000);
        contents.push({
          role: 'user',
          parts: [{ text: `[Result from ${toolName}]: ${truncated}` }],
        });
      }
      continue;
    }

    // Regular user/assistant message
    const role = m.role === 'assistant' ? 'model' : 'user';
    const content = typeof m.content === 'string'
      ? m.content
      : Array.isArray(m.content)
        ? (m.content as Array<{ type?: string; text?: string }>)
            .filter(p => p.type === 'text' && p.text)
            .map(p => p.text!)
            .join('\n')
        : (m.content || '') as string;

    const hasMedia = resolvedMedia && resolvedMedia.size > 0 && /\/api\/content\/[0-9a-f-]{36}/.test(content);
    if (hasMedia) {
      COMBINED_MEDIA_RE.lastIndex = 0;
      const parts: unknown[] = [];
      let lastIdx = 0;
      let mediaMatch: RegExpExecArray | null;
      while ((mediaMatch = COMBINED_MEDIA_RE.exec(content)) !== null) {
        const rawUrl = mediaMatch[2] || mediaMatch[3];
        const uuidMatch = UUID_EXTRACT.exec(rawUrl);
        if (!uuidMatch) continue;
        const canonicalUrl = `/api/content/${uuidMatch[1]}`;
        const mediaData = resolvedMedia!.get(canonicalUrl);
        if (!mediaData) continue;
        const textBefore = content.slice(lastIdx, mediaMatch.index);
        if (textBefore.trim()) parts.push({ text: textBefore });
        const mediaCategory = mediaData.mimeType.split('/')[0];
        if (role === 'model' && !mediaData.thoughtSignature && mediaCategory === 'image') {
          parts.push({ text: '[Previously generated image]' });
        } else {
          const mediaPart: Record<string, unknown> = {
            inlineData: { mimeType: mediaData.mimeType, data: mediaData.base64 },
          };
          if (mediaData.thoughtSignature && role === 'model') {
            mediaPart.thought_signature = mediaData.thoughtSignature;
          }
          parts.push(mediaPart);
        }
        lastIdx = mediaMatch.index + mediaMatch[0].length;
      }
      const textAfter = content.slice(lastIdx);
      if (textAfter.trim()) parts.push({ text: textAfter });
      if (parts.length === 0) parts.push({ text: content });
      contents.push({ role, parts });
    } else {
      contents.push({ role, parts: [{ text: content }] });
    }
  }

  return { systemParts, contents };
}

export default googleAdapter;
