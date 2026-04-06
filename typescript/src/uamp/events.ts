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
} from './types';

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
  /** Base64 encoded audio or URL reference */
  audio: string | { url: string };
  /** Audio format */
  format: AudioFormat;
  /** End of audio stream */
  is_final?: boolean;
  /** Content ID for cross-agent reference */
  content_id?: string;
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
  /** Content ID for cross-agent reference */
  content_id?: string;
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
  /** Content ID for cross-agent reference */
  content_id?: string;
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
  /** Content ID for cross-agent reference */
  content_id?: string;
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
  /** Conversation messages to use for this response */
  messages?: Array<{ role: string; content?: string | null; tool_calls?: unknown[]; tool_call_id?: string }>;
  /** Model to use for this response */
  model?: string;
  /** Temperature */
  temperature?: number;
  /** Max output tokens */
  max_tokens?: number;
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
  type: 'text' | 'audio' | 'image' | 'video' | 'html' | 'tool_call' | 'tool_result' | 'tool_progress' | 'content_updated' | 'file';
  /** Text delta */
  text?: string;
  /** Base64 audio chunk */
  audio?: string;
  /** Image content (from present tool) */
  image?: string | { url: string };
  /** Video content (from present tool) */
  video?: string | { url: string };
  /** HTML content (from present tool) */
  html?: string | { url: string };
  /** Tool call delta */
  tool_call?: {
    id: string;
    name: string;
    arguments: string;
  };
  /** Tool result delta */
  tool_result?: {
    call_id: string;
    result: string;
    status?: string;
    content_items?: ContentItem[];
  };
  /** Incremental tool execution progress with optional typed fields */
  tool_progress?: {
    call_id: string;
    text?: string;
    media_type?: 'image' | 'video' | 'audio' | 'file' | '3d' | 'html';
    status?: 'queued' | 'generating' | 'processing' | 'uploading' | 'downloading' | 'complete' | 'failed';
    progress_percent?: number;
    estimated_duration_ms?: number;
    dimensions?: { width: number; height: number };
    thumbnail_url?: string;
  };
  /** Content update delta (edit to existing content) */
  content_updated?: {
    content_id: string;
    command: string;
    diff?: {
      old_str?: string;
      new_str?: string;
      insert_line?: number;
    };
    timestamp: number;
  };
  /** File content delta */
  file?: string | { url: string };
  filename?: string;
  mime_type?: string;
  content_id?: string;
  /** Display hint (set by present tool) */
  display_hint?: import('./types').DisplayHint;
  /** Description from producing tool */
  description?: string;
  /** Content dimensions */
  dimensions?: { width: number; height: number };
  /** Title (for HTML content) */
  title?: string;
  /** Sandbox flag (for HTML content) */
  sandbox?: boolean;
  /** Alt text (for images) */
  alt_text?: string;
  metadata?: Record<string, unknown>;
  /** Allow additional properties for extensibility */
  [key: string]: unknown;
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
// Presence Events
// ============================================================================

/**
 * Client → Server: User typing indicator
 */
export interface InputTypingEvent extends BaseEvent {
  type: 'input.typing';
  /** true = started typing, false = stopped */
  is_typing: boolean;
  /** Optional conversation ID for multi-conversation contexts */
  conversation_id?: string;
}

/**
 * Server → Client: Typing indicator from another participant
 */
export interface PresenceTypingEvent extends BaseEvent {
  type: 'presence.typing';
  /** ID of the user who is typing */
  user_id: string;
  /** Optional display name */
  username?: string;
  /** true = typing, false = stopped */
  is_typing: boolean;
  /** Optional conversation ID */
  conversation_id?: string;
}

// ============================================================================
// Payment Events
// ============================================================================

/**
 * Payment scheme option
 */
export interface PaymentScheme {
  /** Payment scheme type */
  scheme: 'token' | 'crypto' | 'card' | 'ap2';
  /** Network identifier (e.g., 'robutler', 'ethereum', 'base') */
  network?: string;
  /** Address for crypto payments */
  address?: string;
  /** Minimum amount */
  min_amount?: string;
  /** Maximum amount */
  max_amount?: string;
}

/**
 * Payment requirements
 */
export interface PaymentRequirements {
  /** Required amount (string for precision) */
  amount: string;
  /** Currency code */
  currency: string;
  /** Accepted payment schemes */
  schemes: PaymentScheme[];
  /** Expiry timestamp */
  expires_at?: number;
  /** Reason for payment (e.g., 'llm_usage', 'tool_call', 'api_access') */
  reason?: string;
  /** UCP/AP2 extension for commerce flows */
  ap2?: {
    mandate_uri?: string;
    credential_types?: string[];
    checkout_session_uri?: string;
  };
}

/**
 * Server → Client: Payment required to continue
 */
export interface PaymentRequiredEvent extends BaseEvent {
  type: 'payment.required';
  /** Response ID if blocking a specific response */
  response_id?: string;
  /** Payment requirements */
  requirements: PaymentRequirements;
}

/**
 * Payment submission data
 */
export interface PaymentData {
  /** Payment scheme used */
  scheme: string;
  /** Amount being paid */
  amount: string;
  /** Network identifier */
  network?: string;
  /** Payment token (for token-based payments) */
  token?: string;
  /** Payment proof (for crypto/verifiable payments) */
  proof?: string;
  /** UCP/AP2 verifiable credential */
  ap2_credential?: Record<string, unknown>;
}

/**
 * Client → Server: Submit payment token or proof
 */
export interface PaymentSubmitEvent extends BaseEvent {
  type: 'payment.submit';
  /** Payment data */
  payment: PaymentData;
}

/**
 * Server → Client: Payment accepted
 */
export interface PaymentAcceptedEvent extends BaseEvent {
  type: 'payment.accepted';
  /** Unique payment ID */
  payment_id: string;
  /** Remaining balance */
  balance_remaining?: string;
  /** Token expiry timestamp */
  expires_at?: number;
}

/**
 * Server → Client: Balance update notification
 */
export interface PaymentBalanceEvent extends BaseEvent {
  type: 'payment.balance';
  /** Current balance */
  balance: string;
  /** Currency code */
  currency: string;
  /** Warning when balance is low */
  low_balance_warning?: boolean;
  /** Estimated remaining messages/requests */
  estimated_remaining?: number;
  /** Token expiry timestamp */
  expires_at?: number;
}

/**
 * Payment error codes
 */
export type PaymentErrorCode =
  | 'insufficient_balance'
  | 'token_expired'
  | 'token_invalid'
  | 'payment_failed'
  | 'rate_limited'
  | 'mandate_revoked';

/**
 * Server → Client: Payment error
 */
export interface PaymentErrorEvent extends BaseEvent {
  type: 'payment.error';
  /** Error code */
  code: PaymentErrorCode;
  /** Error message */
  message: string;
  /** Required balance */
  balance_required?: string;
  /** Current balance */
  balance_current?: string;
  /** Whether the client can retry */
  can_retry: boolean;
}

// ============================================================================
// Session Lifecycle Events
// ============================================================================

/**
 * Client → Server: End session gracefully
 */
export interface SessionEndEvent extends BaseEvent {
  type: 'session.end';
  /** Reason for ending */
  reason?: 'user_closed' | 'timeout' | 'error' | string;
}

/**
 * Server → Client: Session-level error
 */
export interface SessionErrorEvent extends BaseEvent {
  type: 'session.error';
  /** Error details */
  error: ErrorDetails;
}

/**
 * Server → Client: Confirmation that response was cancelled
 */
export interface ResponseCancelledEvent extends BaseEvent {
  type: 'response.cancelled';
  /** Response ID that was cancelled */
  response_id: string;
  /** Partial output collected before cancellation */
  partial_output?: ContentItem[];
}

// ============================================================================
// Audio Lifecycle Events
// ============================================================================

/**
 * Client → Server: Audio input buffer committed (realtime mode)
 */
export interface InputAudioCommittedEvent extends BaseEvent {
  type: 'input.audio_committed';
  /** Total audio duration in ms */
  duration_ms?: number;
}

/**
 * Server → Client: Audio output completed for a response
 */
export interface AudioDoneEvent extends BaseEvent {
  type: 'audio.done';
  /** Response ID */
  response_id: string;
  /** Total audio duration in ms */
  duration_ms?: number;
}

/**
 * Server → Client: Transcript completed
 */
export interface TranscriptDoneEvent extends BaseEvent {
  type: 'transcript.done';
  /** Response ID */
  response_id: string;
  /** Full transcript text */
  transcript: string;
}

// ============================================================================
// Conversation Item Events (OpenAI Realtime compatibility)
// ============================================================================

/**
 * Client → Server: Add item to conversation
 */
export interface ConversationItemCreateEvent extends BaseEvent {
  type: 'conversation.item.create';
  /** Item to add */
  item: {
    type: 'message' | 'function_call' | 'function_call_output';
    role?: 'user' | 'assistant' | 'system';
    content?: ContentItem[];
    call_id?: string;
    output?: string;
  };
}

/**
 * Client → Server: Delete item from conversation
 */
export interface ConversationItemDeleteEvent extends BaseEvent {
  type: 'conversation.item.delete';
  /** Item ID to delete */
  item_id: string;
}

/**
 * Client → Server: Truncate conversation item
 */
export interface ConversationItemTruncateEvent extends BaseEvent {
  type: 'conversation.item.truncate';
  /** Item ID to truncate */
  item_id: string;
  /** Content index to truncate at */
  content_index: number;
  /** Audio end (ms) for audio content */
  audio_end_ms?: number;
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
  | SessionEndEvent
  | CapabilitiesQueryEvent
  | ClientCapabilitiesEvent
  | InputTextEvent
  | InputAudioEvent
  | InputImageEvent
  | InputVideoEvent
  | InputFileEvent
  | InputTypingEvent
  | InputAudioCommittedEvent
  | ResponseCreateEvent
  | ResponseCancelEvent
  | ToolResultEvent
  | PaymentSubmitEvent
  | ConversationItemCreateEvent
  | ConversationItemDeleteEvent
  | ConversationItemTruncateEvent
  | PingEvent;

/**
 * All server → client events
 */
export type ServerEvent =
  | SessionCreatedEvent
  | SessionUpdatedEvent
  | SessionErrorEvent
  | CapabilitiesEvent
  | ResponseCreatedEvent
  | ResponseDeltaEvent
  | ResponseDoneEvent
  | ResponseErrorEvent
  | ResponseCancelledEvent
  | ToolCallEvent
  | ToolCallDoneEvent
  | ProgressEvent
  | ThinkingEvent
  | AudioDeltaEvent
  | AudioDoneEvent
  | TranscriptDeltaEvent
  | TranscriptDoneEvent
  | UsageDeltaEvent
  | PresenceTypingEvent
  | PaymentRequiredEvent
  | PaymentAcceptedEvent
  | PaymentBalanceEvent
  | PaymentErrorEvent
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

/**
 * Create an input.typing event
 */
export function createInputTypingEvent(
  isTyping: boolean,
  conversationId?: string
): InputTypingEvent {
  return {
    ...createBaseEvent('input.typing'),
    type: 'input.typing',
    is_typing: isTyping,
    conversation_id: conversationId,
  };
}

/**
 * Create a presence.typing event (for servers)
 */
export function createPresenceTypingEvent(
  userId: string,
  isTyping: boolean,
  username?: string,
  conversationId?: string
): PresenceTypingEvent {
  return {
    ...createBaseEvent('presence.typing'),
    type: 'presence.typing',
    user_id: userId,
    is_typing: isTyping,
    username,
    conversation_id: conversationId,
  };
}

/**
 * Create a payment.submit event
 */
export function createPaymentSubmitEvent(
  payment: PaymentData
): PaymentSubmitEvent {
  return {
    ...createBaseEvent('payment.submit'),
    type: 'payment.submit',
    payment,
  };
}

/**
 * Create a payment.required event (for servers)
 */
export function createPaymentRequiredEvent(
  requirements: PaymentRequirements,
  responseId?: string
): PaymentRequiredEvent {
  return {
    ...createBaseEvent('payment.required'),
    type: 'payment.required',
    response_id: responseId,
    requirements,
  };
}

/**
 * Create a payment.accepted event (for servers)
 */
export function createPaymentAcceptedEvent(
  paymentId: string,
  balanceRemaining?: string,
  expiresAt?: number
): PaymentAcceptedEvent {
  return {
    ...createBaseEvent('payment.accepted'),
    type: 'payment.accepted',
    payment_id: paymentId,
    balance_remaining: balanceRemaining,
    expires_at: expiresAt,
  };
}

/**
 * Create a payment.balance event (for servers)
 */
export function createPaymentBalanceEvent(
  balance: string,
  currency: string,
  options?: {
    lowBalanceWarning?: boolean;
    estimatedRemaining?: number;
    expiresAt?: number;
  }
): PaymentBalanceEvent {
  return {
    ...createBaseEvent('payment.balance'),
    type: 'payment.balance',
    balance,
    currency,
    low_balance_warning: options?.lowBalanceWarning,
    estimated_remaining: options?.estimatedRemaining,
    expires_at: options?.expiresAt,
  };
}

/**
 * Create a payment.error event (for servers)
 */
export function createPaymentErrorEvent(
  code: PaymentErrorCode,
  message: string,
  canRetry: boolean,
  balanceRequired?: string,
  balanceCurrent?: string
): PaymentErrorEvent {
  return {
    ...createBaseEvent('payment.error'),
    type: 'payment.error',
    code,
    message,
    can_retry: canRetry,
    balance_required: balanceRequired,
    balance_current: balanceCurrent,
  };
}

/**
 * Create a session.end event
 */
export function createSessionEndEvent(
  reason?: SessionEndEvent['reason']
): SessionEndEvent {
  return {
    ...createBaseEvent('session.end'),
    type: 'session.end',
    reason,
  };
}

/**
 * Create a session.error event (for servers)
 */
export function createSessionErrorEvent(
  code: string,
  message: string,
  details?: unknown
): SessionErrorEvent {
  return {
    ...createBaseEvent('session.error'),
    type: 'session.error',
    error: { code, message, details },
  };
}

/**
 * Create a response.cancelled event (for servers)
 */
export function createResponseCancelledEvent(
  responseId: string,
  partialOutput?: ContentItem[]
): ResponseCancelledEvent {
  return {
    ...createBaseEvent('response.cancelled'),
    type: 'response.cancelled',
    response_id: responseId,
    partial_output: partialOutput,
  };
}

/**
 * Create an audio.done event (for servers)
 */
export function createAudioDoneEvent(
  responseId: string,
  durationMs?: number
): AudioDoneEvent {
  return {
    ...createBaseEvent('audio.done'),
    type: 'audio.done',
    response_id: responseId,
    duration_ms: durationMs,
  };
}

/**
 * Create a rate_limit event (for servers)
 */
export function createRateLimitEvent(
  limit: number,
  remaining: number,
  resetAt: number
): RateLimitEvent {
  return {
    ...createBaseEvent('rate_limit'),
    type: 'rate_limit',
    limit,
    remaining,
    reset_at: resetAt,
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
    'session.end',
    'capabilities.query',
    'client.capabilities',
    'input.text',
    'input.audio',
    'input.image',
    'input.video',
    'input.file',
    'input.typing',
    'input.audio_committed',
    'response.create',
    'response.cancel',
    'tool.result',
    'payment.submit',
    'conversation.item.create',
    'conversation.item.delete',
    'conversation.item.truncate',
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
