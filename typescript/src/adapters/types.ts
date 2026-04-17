/**
 * Shared LLM Provider Adapter Types
 *
 * These types define the uniform contract that all provider adapters implement.
 * Both direct LLM skills and the UAMP proxy import these same adapters.
 */

export interface Message {
  role: string;
  content: string | null | Array<{ type?: string; text?: string; [key: string]: unknown }>;
  content_items?: Array<Record<string, unknown>>;
  tool_calls?: Array<{
    id: string;
    type?: string;
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
  /** Ephemeral: when true, media in this message should be inlined for the LLM. */
  _inline_for_llm?: boolean;
}

export interface FunctionToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: unknown;
  };
}

export interface NativeToolDefinition {
  type: string;
  [key: string]: unknown;
}

export type ToolDefinition = FunctionToolDefinition | NativeToolDefinition;

/** Type guard for function tool definitions */
export function isFunctionTool(t: ToolDefinition): t is FunctionToolDefinition {
  return t.type === 'function' && 'function' in t;
}

export interface AdapterRequestParams {
  messages: Message[];
  model: string;
  tools?: ToolDefinition[];
  temperature?: number;
  maxTokens?: number;
  apiKey: string;
  resolvedMedia?: Map<string, { mimeType: string; base64: string; thoughtSignature?: string }>;
  responseModalities?: string[];
  stream?: boolean;
  /** When explicitly false, adapters should not request thinking/reasoning even for capable models. */
  thinking?: boolean;
  /** Session/chat identifier used by Fireworks for replica-affinity prompt caching. */
  sessionId?: string;
}

export interface AdapterRequest {
  url: string;
  headers: Record<string, string>;
  body: string;
}

export type AdapterChunk =
  | { type: 'text'; text: string }
  | { type: 'tool_call'; id: string; name: string; arguments: string }
  | { type: 'tool_result'; call_id: string; result: string; status?: string }
  | { type: 'tool_progress'; call_id: string; text: string }
  | { type: 'image'; base64: string; mimeType: string; thoughtSignature?: string }
  | { type: 'thinking'; text: string; signature?: string }
  | { type: 'usage'; input: number; output: number; cache_read_input?: number; cache_creation_input?: number }
  | { type: 'done' };

export type MediaMode = 'base64' | 'url' | 'none';

export interface MediaSupport {
  image: MediaMode;
  audio: MediaMode;
  video: MediaMode;
  document: MediaMode;
}

export interface LLMAdapter {
  name: string;
  mediaSupport: MediaSupport;
  buildRequest(params: AdapterRequestParams): AdapterRequest;
  parseStream(response: Response): AsyncGenerator<AdapterChunk>;
}

/**
 * Standard UAMP usage format reported by all adapters.
 * PaymentSkill reads this from context._llm_usage to settle charges.
 */
export interface UAMPUsage {
  model: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  cached_tokens?: number;
  image_count?: number;
  audio_seconds?: number;
  is_byok: boolean;
}

/**
 * Adapter capabilities declared before the LLM call.
 * PaymentSkill reads this from context._llm_capabilities to size locks.
 */
export interface AdapterCapabilities {
  model: string;
  provider: string;
  maxOutputTokens: number;
  pricing: {
    inputPer1k: number;
    outputPer1k: number;
    cacheReadPer1k?: number;
    cacheWritePer1k?: number;
  };
}
