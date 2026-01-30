/**
 * UAMP Protocol Events
 * 
 * All UAMP communication is event-based. Events flow bidirectionally
 * between clients and servers.
 */

import type {
  Modality,
  AudioFormat,
  ToolDefinition,
  VoiceConfig,
  TurnDetectionConfig,
  Session,
  Capabilities,
  UsageStats,
  ContentItem,
} from './types.js';

// ============================================================================
// Base Event
// ============================================================================

/**
 * Base structure for all UAMP events
 */
export interface BaseEvent {
  /** Event type identifier */
  type: string;
  /** Unique event ID (UUID) */
  event_id: string;
  /** Unix timestamp in milliseconds */
  timestamp?: number;
}

/**
 * Generate a unique event ID
 */
export function generateEventId(): string {
  return crypto.randomUUID();
}

/**
 * Create base event properties
 */
export function createBaseEvent(type: string): BaseEvent {
  return {
    type,
    event_id: generateEventId(),
    timestamp: Date.now(),
  };
}

// ============================================================================
// Session Events
// ============================================================================

/**
 * Session configuration in create event
 */
export interface SessionCreateConfig {
  /** Client-suggested session ID */
  id?: string;
  /** Active modalities */
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
  extensions?: {
    openai?: { model?: string; temperature?: number; max_tokens?: number };
    anthropic?: { thinking?: boolean; max_tokens?: number };
    google?: { safety_settings?: unknown };
    [provider: string]: unknown;
  };
}

/**
 * Client → Server: Create new session
 */
export interface SessionCreateEvent extends BaseEvent {
  type: 'session.create';
  /** UAMP version */
  uamp_version: string;
  /** Session configuration */
  session: SessionCreateConfig;
  /** Optional client capabilities */
  client_capabilities?: Capabilities;
}

/**
 * Server → Client: Session created confirmation
 */
export interface SessionCreatedEvent extends BaseEvent {
  type: 'session.created';
  /** Server's UAMP version */
  uamp_version: string;
  /** Full session object */
  session: Session;
}

/**
 * Client → Server: Update session configuration
 */
export interface SessionUpdateEvent extends BaseEvent {
  type: 'session.update';
  /** Partial session update */
  session: Partial<SessionCreateConfig>;
}

/**
 * Server → Client: Session updated confirmation
 */
export interface SessionUpdatedEvent extends BaseEvent {
  type: 'session.updated';
  /** Updated session */
  session: Session;
}

// ============================================================================
// Capabilities Events
// ============================================================================

/**
 * Client → Server: Query server capabilities
 */
export interface CapabilitiesQueryEvent extends BaseEvent {
  type: 'capabilities.query';
  /** Query for specific model */
  model?: string;
}

/**
 * Client → Server: Announce client capabilities
 */
export interface ClientCapabilitiesEvent extends BaseEvent {
  type: 'client.capabilities';
  /** Client capabilities (same format as model) */
  capabilities: Capabilities;
}

/**
 * Server → Client: Server/model capabilities
 */
export interface CapabilitiesEvent extends BaseEvent {
  type: 'capabilities';
  /** Model/agent capabilities */
  capabilities: Capabilities;
}

// ============================================================================
// Input Events
// ============================================================================

/**
 * Client → Server: Text input
 */
export interface InputTextEvent extends BaseEvent {
  type: 'input.text';
  /** Text content */
  text: string;
  /** Role for the input */
  role?: 'user' | 'system';
}

/**
 * Client → Server: Audio input
 */
export interface InputAudioEvent extends BaseEvent {
  type: 'input.audio';
  /** Base64 encoded audio */
  audio: string;
  /** Audio format */
  format: AudioFormat;
  /** End of audio stream */
  is_final?: boolean;
}

/**
 * Client → Server: Image input
 */
export interface InputImageEvent extends BaseEvent {
  type: 'input.image';
  /** Base64 data or URL object */
  image: string | { url: string };
  /** Image format */
  format?: 'jpeg' | 'png' | 'webp' | 'gif';
  /** Detail level */
  detail?: 'low' | 'high' | 'auto';
}

/**
 * Client → Server: Video input
 */
export interface InputVideoEvent extends BaseEvent {
  type: 'input.video';
  /** Base64 data or URL object */
  video: string | { url: string };
  /** Video format */
  format?: 'mp4' | 'webm';
}

/**
 * Client → Server: File input
 */
export interface InputFileEvent extends BaseEvent {
  type: 'input.file';
  /** Base64 data or URL object */
  file: string | { url: string };
  /** Filename */
  filename: string;
  /** MIME type */
  mime_type: string;
}

// ============================================================================
// Response Events
// ============================================================================

/**
 * Response configuration override
 */
export interface ResponseConfig {
  /** Override modalities */
  modalities?: Modality[];
  /** Override instructions */
  instructions?: string;
  /** Override tools */
  tools?: ToolDefinition[];
}

/**
 * Client → Server: Request response generation
 */
export interface ResponseCreateEvent extends BaseEvent {
  type: 'response.create';
  /** Optional response configuration override */
  response?: ResponseConfig;
}

/**
 * Client → Server: Cancel in-progress response
 */
export interface ResponseCancelEvent extends BaseEvent {
  type: 'response.cancel';
  /** Response ID to cancel (current if omitted) */
  response_id?: string;
}

/**
 * Server → Client: Response started
 */
export interface ResponseCreatedEvent extends BaseEvent {
  type: 'response.created';
  /** Unique response ID */
  response_id: string;
}

/**
 * Delta content types
 */
export interface ResponseDelta {
  type: 'text' | 'audio' | 'tool_call';
  /** Text delta */
  text?: string;
  /** Base64 audio chunk */
  audio?: string;
  /** Tool call delta */
  tool_call?: {
    id: string;
    name: string;
    arguments: string;
  };
}

/**
 * Server → Client: Streaming content delta
 */
export interface ResponseDeltaEvent extends BaseEvent {
  type: 'response.delta';
  /** Response ID */
  response_id: string;
  /** Delta content */
  delta: ResponseDelta;
}

/**
 * Response status
 */
export type ResponseStatus = 'completed' | 'cancelled' | 'failed';

/**
 * Complete response object
 */
export interface ResponseOutput {
  id: string;
  status: ResponseStatus;
  output: ContentItem[];
  usage?: UsageStats;
}

/**
 * Server → Client: Response completed
 */
export interface ResponseDoneEvent extends BaseEvent {
  type: 'response.done';
  /** Response ID */
  response_id: string;
  /** Complete response */
  response: ResponseOutput;
}

/**
 * Error details
 */
export interface ErrorDetails {
  code: string;
  message: string;
  details?: unknown;
}

/**
 * Server → Client: Response error
 */
export interface ResponseErrorEvent extends BaseEvent {
  type: 'response.error';
  /** Response ID (if applicable) */
  response_id?: string;
  /** Error details */
  error: ErrorDetails;
}

// ============================================================================
// Tool Events
// ============================================================================

/**
 * Server → Client: Tool execution request
 */
export interface ToolCallEvent extends BaseEvent {
  type: 'tool.call';
  /** Unique call ID */
  call_id: string;
  /** Tool name */
  name: string;
  /** JSON string arguments */
  arguments: string;
}

/**
 * Client → Server: Tool execution result
 */
export interface ToolResultEvent extends BaseEvent {
  type: 'tool.result';
  /** Call ID this responds to */
  call_id: string;
  /** JSON string result */
  result: string;
  /** Whether execution errored */
  is_error?: boolean;
}

/**
 * Server → Client: Tool call completed
 */
export interface ToolCallDoneEvent extends BaseEvent {
  type: 'tool.call_done';
  /** Call ID */
  call_id: string;
}

// ============================================================================
// Progress Events
// ============================================================================

/**
 * Progress target types
 */
export type ProgressTarget = 'tool' | 'response' | 'upload' | 'reasoning';

/**
 * Server → Client: Progress update
 */
export interface ProgressEvent extends BaseEvent {
  type: 'progress';
  /** What's being progressed */
  target: ProgressTarget;
  /** Target ID (tool_call_id, response_id) */
  target_id?: string;
  /** Current stage */
  stage?: string;
  /** Human-readable message */
  message?: string;
  /** Progress percentage (0-100) */
  percent?: number;
  /** Current step number */
  step?: number;
  /** Total steps */
  total_steps?: number;
}

/**
 * Server → Client: Reasoning/thinking content
 */
export interface ThinkingEvent extends BaseEvent {
  type: 'thinking';
  /** Reasoning text */
  content: string;
  /** Thinking stage */
  stage?: string;
  /** Content hidden for safety */
  redacted?: boolean;
  /** true = append, false = complete thought */
  is_delta?: boolean;
}

// ============================================================================
// Audio Events
// ============================================================================

/**
 * Server → Client: Streaming audio output
 */
export interface AudioDeltaEvent extends BaseEvent {
  type: 'audio.delta';
  /** Response ID */
  response_id: string;
  /** Base64 audio chunk */
  audio: string;
}

/**
 * Server → Client: Real-time transcription
 */
export interface TranscriptDeltaEvent extends BaseEvent {
  type: 'transcript.delta';
  /** Response ID */
  response_id: string;
  /** Transcript text */
  transcript: string;
}

// ============================================================================
// Usage Events
// ============================================================================

/**
 * Server → Client: Incremental usage update
 */
export interface UsageDeltaEvent extends BaseEvent {
  type: 'usage.delta';
  /** Response ID */
  response_id: string;
  /** Usage delta */
  delta: Partial<UsageStats>;
}

// ============================================================================
// Utility Events
// ============================================================================

/**
 * Client → Server: Connection keepalive
 */
export interface PingEvent extends BaseEvent {
  type: 'ping';
}

/**
 * Server → Client: Keepalive response
 */
export interface PongEvent extends BaseEvent {
  type: 'pong';
}

/**
 * Server → Client: Rate limit notification
 */
export interface RateLimitEvent extends BaseEvent {
  type: 'rate_limit';
  /** Request limit */
  limit: number;
  /** Remaining requests */
  remaining: number;
  /** Reset timestamp (Unix seconds) */
  reset_at: number;
}

// ============================================================================
// Event Type Unions
// ============================================================================

/**
 * All client → server events
 */
export type ClientEvent =
  | SessionCreateEvent
  | SessionUpdateEvent
  | CapabilitiesQueryEvent
  | ClientCapabilitiesEvent
  | InputTextEvent
  | InputAudioEvent
  | InputImageEvent
  | InputVideoEvent
  | InputFileEvent
  | ResponseCreateEvent
  | ResponseCancelEvent
  | ToolResultEvent
  | PingEvent;

/**
 * All server → client events
 */
export type ServerEvent =
  | SessionCreatedEvent
  | SessionUpdatedEvent
  | CapabilitiesEvent
  | ResponseCreatedEvent
  | ResponseDeltaEvent
  | ResponseDoneEvent
  | ResponseErrorEvent
  | ToolCallEvent
  | ToolCallDoneEvent
  | ProgressEvent
  | ThinkingEvent
  | AudioDeltaEvent
  | TranscriptDeltaEvent
  | UsageDeltaEvent
  | RateLimitEvent
  | PongEvent;

/**
 * All UAMP events
 */
export type UAMPEvent = ClientEvent | ServerEvent;

// ============================================================================
// Event Factories
// ============================================================================

/**
 * Create a session.create event
 */
export function createSessionCreateEvent(
  session: SessionCreateConfig,
  clientCapabilities?: Capabilities,
  uampVersion = '1.0'
): SessionCreateEvent {
  return {
    ...createBaseEvent('session.create'),
    type: 'session.create',
    uamp_version: uampVersion,
    session,
    client_capabilities: clientCapabilities,
  };
}

/**
 * Create an input.text event
 */
export function createInputTextEvent(
  text: string,
  role: 'user' | 'system' = 'user'
): InputTextEvent {
  return {
    ...createBaseEvent('input.text'),
    type: 'input.text',
    text,
    role,
  };
}

/**
 * Create a response.create event
 */
export function createResponseCreateEvent(
  config?: ResponseConfig
): ResponseCreateEvent {
  return {
    ...createBaseEvent('response.create'),
    type: 'response.create',
    response: config,
  };
}

/**
 * Create a tool.result event
 */
export function createToolResultEvent(
  callId: string,
  result: unknown,
  isError = false
): ToolResultEvent {
  return {
    ...createBaseEvent('tool.result'),
    type: 'tool.result',
    call_id: callId,
    result: typeof result === 'string' ? result : JSON.stringify(result),
    is_error: isError,
  };
}

/**
 * Create a response.delta event (for servers)
 */
export function createResponseDeltaEvent(
  responseId: string,
  delta: ResponseDelta
): ResponseDeltaEvent {
  return {
    ...createBaseEvent('response.delta'),
    type: 'response.delta',
    response_id: responseId,
    delta,
  };
}

/**
 * Create a response.done event (for servers)
 */
export function createResponseDoneEvent(
  responseId: string,
  output: ContentItem[],
  status: ResponseStatus = 'completed',
  usage?: UsageStats
): ResponseDoneEvent {
  return {
    ...createBaseEvent('response.done'),
    type: 'response.done',
    response_id: responseId,
    response: {
      id: responseId,
      status,
      output,
      usage,
    },
  };
}

/**
 * Create a response.error event (for servers)
 */
export function createResponseErrorEvent(
  code: string,
  message: string,
  responseId?: string,
  details?: unknown
): ResponseErrorEvent {
  return {
    ...createBaseEvent('response.error'),
    type: 'response.error',
    response_id: responseId,
    error: {
      code,
      message,
      details,
    },
  };
}

/**
 * Create a progress event (for servers)
 */
export function createProgressEvent(
  target: ProgressTarget,
  message?: string,
  percent?: number,
  targetId?: string
): ProgressEvent {
  return {
    ...createBaseEvent('progress'),
    type: 'progress',
    target,
    target_id: targetId,
    message,
    percent,
  };
}

// ============================================================================
// Event Parsing
// ============================================================================

/**
 * Parse a JSON string into a typed event
 */
export function parseEvent(json: string): UAMPEvent {
  const data = JSON.parse(json);
  if (!data.type || !data.event_id) {
    throw new Error('Invalid UAMP event: missing type or event_id');
  }
  return data as UAMPEvent;
}

/**
 * Serialize an event to JSON
 */
export function serializeEvent(event: UAMPEvent): string {
  return JSON.stringify(event);
}

/**
 * Check if event is a client event
 */
export function isClientEvent(event: UAMPEvent): event is ClientEvent {
  const clientTypes = [
    'session.create',
    'session.update',
    'capabilities.query',
    'client.capabilities',
    'input.text',
    'input.audio',
    'input.image',
    'input.video',
    'input.file',
    'response.create',
    'response.cancel',
    'tool.result',
    'ping',
  ];
  return clientTypes.includes(event.type);
}

/**
 * Check if event is a server event
 */
export function isServerEvent(event: UAMPEvent): event is ServerEvent {
  return !isClientEvent(event);
}
