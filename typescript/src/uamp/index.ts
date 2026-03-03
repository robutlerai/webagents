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
  
  // Client events
  ClientEvent,
  SessionCreateEvent,
  InputTextEvent,
  InputAudioEvent,
  InputImageEvent,
  InputTypingEvent,
  ResponseCreateEvent,
  ToolResultEvent,
  PaymentSubmitEvent,
  
  // Server events
  ServerEvent,
  SessionCreatedEvent,
  CapabilitiesEvent,
  ResponseDeltaEvent,
  ResponseDoneEvent,
  ResponseErrorEvent,
  ToolCallEvent,
  ProgressEvent,
  ThinkingEvent,
  PresenceTypingEvent,
  PaymentRequiredEvent,
  PaymentAcceptedEvent,
  PaymentBalanceEvent,
  PaymentErrorEvent,
  
  // Payment types
  PaymentScheme,
  PaymentRequirements,
  PaymentData,
  PaymentErrorCode,
  
  // Union
  UAMPEvent,
} from './events.js';
