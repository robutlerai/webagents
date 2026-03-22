/**
 * UAMP Protocol Types
 * 
 * Type definitions for the Universal Agentic Message Protocol.
 * These types are designed to be compatible with the Python SDK.
 */

// ============================================================================
// Basic Types
// ============================================================================

/**
 * Extensible modality type for content formats
 */
export type Modality = 'text' | 'audio' | 'image' | 'video' | 'file' | string;

/**
 * Common audio formats for voice interfaces
 */
export type AudioFormat =
  | 'pcm16'
  | 'g711_ulaw'
  | 'g711_alaw'
  | 'mp3'
  | 'opus'
  | 'wav'
  | 'webm'
  | 'aac'
  | string;

// ============================================================================
// Voice and Audio Configuration
// ============================================================================

/**
 * Provider-agnostic voice configuration
 */
export interface VoiceConfig {
  /** Provider identifier: 'openai', 'elevenlabs', 'google', etc. */
  provider?: string;
  /** Provider-specific voice ID */
  voice_id?: string;
  /** Human-readable voice name */
  name?: string;
  /** Speed multiplier (0.5 - 2.0) */
  speed?: number;
  /** Voice pitch adjustment */
  pitch?: number;
  /** BCP-47 language tag */
  language?: string;
  /** Provider-specific extensions */
  extensions?: Record<string, unknown>;
}

/**
 * Turn detection configuration for voice interfaces
 */
export interface TurnDetectionConfig {
  /** Detection type */
  type: 'server_vad' | 'client_vad' | 'push_to_talk' | 'none';
  /** VAD sensitivity threshold (0.0 - 1.0) */
  threshold?: number;
  /** Silence duration before end of turn (ms) */
  silence_duration_ms?: number;
  /** Audio padding before speech (ms) */
  prefix_padding_ms?: number;
  /** Provider-specific extensions */
  extensions?: Record<string, unknown>;
}

// ============================================================================
// Session Configuration
// ============================================================================

/**
 * JSON Schema type for tool parameters
 */
export interface JSONSchema {
  type?: string;
  properties?: Record<string, JSONSchema>;
  required?: string[];
  items?: JSONSchema;
  description?: string;
  enum?: unknown[];
  default?: unknown;
  [key: string]: unknown;
}

/**
 * Standard tool/function definition (OpenAI-compatible)
 */
export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: JSONSchema;
  };
}

/**
 * Full session configuration
 */
export interface SessionConfig {
  /** Active modalities for this session */
  modalities: Modality[];
  /** System instructions */
  instructions?: string;
  /** Available tools */
  tools?: ToolDefinition[];
  /** Voice configuration */
  voice?: VoiceConfig;
  /** Input audio format */
  input_audio_format?: AudioFormat;
  /** Output audio format */
  output_audio_format?: AudioFormat;
  /** Turn detection settings */
  turn_detection?: TurnDetectionConfig;
  /** Provider-specific extensions */
  extensions?: Record<string, unknown>;
}

// ============================================================================
// Content Types
// ============================================================================

/**
 * Tool call in content
 */
export interface ToolCall {
  /** Unique tool call ID */
  id: string;
  /** Tool function name */
  name: string;
  /** JSON string of arguments */
  arguments: string;
}

/**
 * Tool result in content
 */
export interface ToolResult {
  /** ID of the tool call this responds to */
  call_id: string;
  /** Result as string */
  result: string;
  /** Whether the tool execution errored */
  is_error?: boolean;
  /** Multimodal content from tool execution */
  content_items?: ContentItem[];
}

// ============================================================================
// Content Item Types (discriminated union)
// ============================================================================

export interface TextContent {
  type: 'text';
  text: string;
}

export interface AudioContent {
  type: 'audio';
  audio: string | { url: string };
  format?: AudioFormat;
  duration_ms?: number;
  content_id?: string;
}

export interface ImageContent {
  type: 'image';
  image: string | { url: string };
  format?: string;
  detail?: 'low' | 'high' | 'auto';
  alt_text?: string;
  content_id?: string;
}

export interface VideoContent {
  type: 'video';
  video: string | { url: string };
  format?: string;
  duration_ms?: number;
  thumbnail?: string;
  content_id?: string;
}

export interface FileContent {
  type: 'file';
  file: string | { url: string };
  filename: string;
  mime_type: string;
  size_bytes?: number;
  content_id?: string;
}

export interface ToolCallContent {
  type: 'tool_call';
  tool_call: ToolCall;
}

export interface ToolResultContent {
  type: 'tool_result';
  tool_result: ToolResult;
}

export type ContentItem =
  | TextContent
  | AudioContent
  | ImageContent
  | VideoContent
  | FileContent
  | ToolCallContent
  | ToolResultContent;

// ============================================================================
// Usage Statistics
// ============================================================================

/**
 * Cost tracking information
 */
export interface CostInfo {
  /** Cost for input tokens */
  input_cost: number;
  /** Cost for output tokens */
  output_cost: number;
  /** Total cost */
  total_cost: number;
  /** Currency code (e.g., 'USD') */
  currency: string;
}

/**
 * Audio usage information
 */
export interface AudioUsage {
  /** Input audio duration in seconds */
  input_seconds?: number;
  /** Output audio duration in seconds */
  output_seconds?: number;
}

/**
 * Token and cost tracking
 */
export interface UsageStats {
  /** Number of input tokens */
  input_tokens: number;
  /** Number of output tokens */
  output_tokens: number;
  /** Total tokens */
  total_tokens: number;
  /** Tokens served from cache */
  cached_tokens?: number;
  /** Cost information */
  cost?: CostInfo;
  /** Audio usage information */
  audio?: AudioUsage;
  /** Provider-specific details */
  details?: Record<string, unknown>;
}

// ============================================================================
// Session
// ============================================================================

/**
 * Session status
 */
export type SessionStatus = 'active' | 'closed';

/**
 * Session object returned by server
 */
export interface Session {
  /** Unique session ID */
  id: string;
  /** Creation timestamp (Unix seconds) */
  created_at: number;
  /** Session configuration */
  config: SessionConfig;
  /** Current status */
  status: SessionStatus;
}

// ============================================================================
// Capability Types
// ============================================================================

/**
 * Image processing capabilities
 */
export interface ImageCapabilities {
  /** Supported image formats */
  formats: string[];
  /** Maximum file size in bytes */
  max_size_bytes?: number;
  /** Maximum total pixels */
  max_pixels?: number;
  /** Supported detail levels */
  detail_levels?: string[];
  /** Maximum images per request */
  max_images_per_request?: number;
}

/**
 * Audio processing capabilities
 */
export interface AudioCapabilities {
  /** Supported input audio formats */
  input_formats: AudioFormat[];
  /** Supported output audio formats */
  output_formats: AudioFormat[];
  /** Supported sample rates */
  sample_rates?: number[];
  /** Maximum audio duration in seconds */
  max_duration_seconds?: number;
  /** Whether realtime audio is supported */
  supports_realtime: boolean;
  /** Available voice IDs */
  voices?: string[];
}

/**
 * File processing capabilities
 */
export interface FileCapabilities {
  /** Supported MIME types */
  supported_mime_types: string[];
  /** Maximum file size in bytes */
  max_size_bytes?: number;
  /** Whether PDF files are supported */
  supports_pdf: boolean;
  /** Whether code files are supported */
  supports_code: boolean;
  /** Whether structured data (JSON, CSV) is supported */
  supports_structured_data: boolean;
}

/**
 * Tool/function calling capabilities
 */
export interface ToolCapabilities {
  /** Whether tools are supported */
  supports_tools: boolean;
  /** Whether parallel tool calls are supported */
  supports_parallel_tools: boolean;
  /** Whether streaming tool calls are supported */
  supports_streaming_tools: boolean;
  /** Maximum tools per request */
  max_tools_per_request?: number;
  /** Built-in tools available */
  built_in_tools: string[];
}

/**
 * Unified capabilities for models, clients, and agents
 */
export interface Capabilities {
  // Identity
  /** Model ID, client ID, or agent ID */
  id: string;
  /** Provider name */
  provider: string;

  // Core modalities
  /** Supported modalities */
  modalities: Modality[];

  // Detailed capabilities
  /** Image capabilities */
  image?: ImageCapabilities;
  /** Audio capabilities */
  audio?: AudioCapabilities;
  /** File capabilities */
  file?: FileCapabilities;
  /** Tool capabilities */
  tools?: ToolCapabilities;

  // Features
  /** Whether streaming is supported */
  supports_streaming: boolean;
  /** Whether thinking/reasoning is supported */
  supports_thinking: boolean;
  /** Whether caching is supported */
  supports_caching: boolean;
  /** Context window size in tokens */
  context_window?: number;
  /** Maximum output tokens */
  max_output_tokens?: number;

  // Agent/client extensions
  /** Capabilities provided by this agent/client */
  provides?: string[];
  /** Available widgets */
  widgets?: string[];
  /** HTTP/WebSocket endpoints */
  endpoints?: string[];

  // Custom extensions
  /** Provider-specific extensions */
  extensions?: Record<string, unknown>;
}

// ============================================================================
// Message Types (for convenience)
// ============================================================================

/**
 * Message role
 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/**
 * Standard message format
 */
export interface Message {
  /** Message role */
  role: MessageRole;
  /** Text content (simple format) */
  content?: string;
  /** Multimodal content items */
  content_items?: ContentItem[];
  /** Name for multi-participant conversations */
  name?: string;
  /** Tool calls made by assistant */
  tool_calls?: ToolCall[];
  /** Tool call ID (for tool responses) */
  tool_call_id?: string;
}
