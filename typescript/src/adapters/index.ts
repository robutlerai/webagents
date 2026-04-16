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
} from './types';

export { readSSEStream } from './sse';
export { extractContentRef, isUAMPContentArray, canonicalContentUrl, describeContentItem } from './content';
export type { ResolvedMediaMap } from './content';
export { googleAdapter } from './google';
export { anthropicAdapter } from './anthropic';
export {
  openaiAdapter,
  xaiAdapter,
  fireworksAdapter,
  createOpenAICompatibleAdapter,
} from './openai';

import { googleAdapter } from './google';
import { anthropicAdapter } from './anthropic';
import { openaiAdapter, xaiAdapter, fireworksAdapter } from './openai';
import type { LLMAdapter } from './types';

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
