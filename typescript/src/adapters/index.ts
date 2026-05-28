/**
 * Shared LLM Provider Adapters
 *
 * Single source of truth for provider-specific logic (message conversion,
 * request building, SSE stream parsing). Used by both direct LLM skills
 * and the UAMP proxy.
 *
 * Routing:
 *   - `openai`, `xai`  →  Responses API (`./responses.ts`)
 *   - `anthropic`      →  `./anthropic.ts` (`/v1/messages`)
 *   - `google`         →  `./google.ts`
 *   - `fireworks`      →  Chat Completions (`./completions.ts`)
 *
 * Rollback flags (env-gated, default off):
 *   - `OPENAI_USE_CHAT_COMPLETIONS=1` swaps the OpenAI adapter back to
 *     `/v1/chat/completions`. Lets us roll back without redeploy if the
 *     Responses migration regresses something.
 *   - `XAI_USE_CHAT_COMPLETIONS=1` ditto for xAI.
 */

export type {
  LLMAdapter,
  AdapterRequestParams,
  AdapterRequest,
  AdapterChunk,
  MediaSupport,
  MediaMode,
  Message,
  ToolDefinition,
  UAMPUsage,
  AdapterCapabilities,
  ThinkingLevel,
} from './types';
export { normalizeThinking } from './types';

export { readSSEStream } from './sse';
export { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem, isTextDecodableMime } from './content';
export type { ResolvedMediaMap, ResolvedMediaEntry } from './content';
export { googleAdapter } from './google';
export {
  openGeminiLiveSession,
  GEMINI_LIVE_INPUT_RATE,
  GEMINI_LIVE_OUTPUT_RATE,
} from './google-live';
export type { GeminiLiveSession, GeminiLiveSessionOptions } from './google-live';
export { anthropicAdapter } from './anthropic';
export {
  fireworksAdapter,
  createChatCompletionsAdapter,
  createOpenAICompatibleAdapter,
  createOpenAICompletionsAdapter,
  createXAICompletionsAdapter,
} from './completions';
export {
  openaiAdapter,
  xaiAdapter,
  createResponsesApiAdapter,
} from './responses';

import { googleAdapter } from './google';
import { anthropicAdapter } from './anthropic';
import { fireworksAdapter, createOpenAICompletionsAdapter, createXAICompletionsAdapter } from './completions';
import { openaiAdapter as openaiResponsesAdapter, xaiAdapter as xaiResponsesAdapter } from './responses';
import type { LLMAdapter } from './types';

/**
 * Resolve the OpenAI adapter, respecting the `OPENAI_USE_CHAT_COMPLETIONS=1`
 * rollback flag. Built lazily so the env can be flipped at process start.
 */
function resolveOpenAIAdapter(): LLMAdapter {
  if (typeof process !== 'undefined' && process.env?.OPENAI_USE_CHAT_COMPLETIONS === '1') {
    return createOpenAICompletionsAdapter();
  }
  return openaiResponsesAdapter;
}

function resolveXAIAdapter(): LLMAdapter {
  if (typeof process !== 'undefined' && process.env?.XAI_USE_CHAT_COMPLETIONS === '1') {
    return createXAICompletionsAdapter();
  }
  return xaiResponsesAdapter;
}

/**
 * Get the adapter for a provider name.
 * @throws Error if provider is unknown
 */
export function getAdapter(provider: string): LLMAdapter {
  switch (provider) {
    case 'google':    return googleAdapter;
    case 'anthropic': return anthropicAdapter;
    case 'openai':    return resolveOpenAIAdapter();
    case 'xai':       return resolveXAIAdapter();
    case 'fireworks': return fireworksAdapter;
    default:
      throw new Error(`Unknown LLM provider: ${provider}. Available: google, anthropic, openai, xai, fireworks`);
  }
}
