/**
 * UAMP Protocol Module
 * 
 * Universal Agentic Message Protocol types and events.
 * @see https://uamp.dev
 */

// Types
export * from './types.js';

// Events
export * from './events.js';

// Content helpers
export { getContentItemUrl, isMediaContent, ensureContentId } from './content.js';

// Re-export commonly used types for convenience
export type {
  // Basic types
  Modality,
  AudioFormat,
  
  // Configuration
  VoiceConfig,
  TurnDetectionConfig,
  SessionConfig,
  ToolDefinition,
  JSONSchema,
  
  // Content
  ContentItem,
  ToolCall,
  ToolResult,
  
  // Usage
  UsageStats,
  CostInfo,
  
  // Session
  Session,
  SessionStatus,
  
  // Capabilities
  Capabilities,
  ImageCapabilities,
  AudioCapabilities,
  FileCapabilities,
  ToolCapabilities,
  
  // Messages
  Message,
  MessageRole,
} from './types.js';

export type {
  // Base
  BaseEvent,
  ErrorDetails,
  
  // Client events
  ClientEvent,
  SessionCreateEvent,
  SessionCreateConfig,
  SessionUpdateEvent,
  SessionEndEvent,
  CapabilitiesQueryEvent,
  ClientCapabilitiesEvent,
  InputTextEvent,
  InputAudioEvent,
  InputImageEvent,
  InputVideoEvent,
  InputFileEvent,
  InputTypingEvent,
  InputAudioCommittedEvent,
  ResponseCreateEvent,
  ResponseCancelEvent,
  ResponseConfig,
  ToolResultEvent,
  PaymentSubmitEvent,
  ConversationItemCreateEvent,
  ConversationItemDeleteEvent,
  ConversationItemTruncateEvent,
  PingEvent,
  
  // Server events
  ServerEvent,
  SessionCreatedEvent,
  SessionUpdatedEvent,
  SessionErrorEvent,
  CapabilitiesEvent,
  ResponseCreatedEvent,
  ResponseDelta,
  ResponseDeltaEvent,
  ResponseDoneEvent,
  ResponseOutput,
  ResponseStatus,
  ResponseErrorEvent,
  ResponseCancelledEvent,
  ToolCallEvent,
  ToolCallDoneEvent,
  ProgressEvent,
  ProgressTarget,
  ThinkingEvent,
  AudioDeltaEvent,
  AudioDoneEvent,
  TranscriptDeltaEvent,
  TranscriptDoneEvent,
  UsageDeltaEvent,
  PresenceTypingEvent,
  PaymentRequiredEvent,
  PaymentAcceptedEvent,
  PaymentBalanceEvent,
  PaymentErrorEvent,
  RateLimitEvent,
  PongEvent,
  
  // Payment types
  PaymentScheme,
  PaymentRequirements,
  PaymentData,
  PaymentErrorCode,
  
  // Union
  UAMPEvent,
} from './events.js';

export { UAMPClient, type UAMPClientConfig, type UAMPClientEvents } from './client.js';
