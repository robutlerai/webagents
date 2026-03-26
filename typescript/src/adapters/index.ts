/**
 * Shared LLM Provider Adapters
 *
 * Single source of truth for provider-specific logic (message conversion,
 * request building, SSE stream parsing). Used by both direct LLM skills
 * and the UAMP proxy.
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
} from './types.js';

export { readSSEStream } from './sse.js';
export { extractContentRef, isUAMPContentArray, canonicalContentUrl } from './content.js';
export type { ResolvedMediaMap } from './content.js';
export { googleAdapter } from './google.js';
export { anthropicAdapter } from './anthropic.js';
export {
  openaiAdapter,
  xaiAdapter,
  fireworksAdapter,
  createOpenAICompatibleAdapter,
} from './openai.js';

import { googleAdapter } from './google.js';
import { anthropicAdapter } from './anthropic.js';
import { openaiAdapter, xaiAdapter, fireworksAdapter } from './openai.js';
import type { LLMAdapter } from './types.js';

const adapters: Record<string, LLMAdapter> = {
  google: googleAdapter,
  anthropic: anthropicAdapter,
  openai: openaiAdapter,
  xai: xaiAdapter,
  fireworks: fireworksAdapter,
};

/**
 * Get the adapter for a provider name.
 * @throws Error if provider is unknown
 */
export function getAdapter(provider: string): LLMAdapter {
  const adapter = adapters[provider];
  if (!adapter) {
    throw new Error(`Unknown LLM provider: ${provider}. Available: ${Object.keys(adapters).join(', ')}`);
  }
  return adapter;
}
