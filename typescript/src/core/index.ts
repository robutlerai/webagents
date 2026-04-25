/**
 * Core Framework Module
 * 
 * Base classes and utilities for building agents and skills.
 */

// Types
export * from './types';

// Decorators
export {
  tool,
  hook,
  handoff,
  observe,
  prompt,
  http,
  websocket,
  getTools,
  getHooks,
  getHandoffs,
  getObservers,
  getPrompts,
  getHttpEndpoints,
  getWebSocketEndpoints,
  TOOLS_KEY,
  HOOKS_KEY,
  HANDOFFS_KEY,
  OBSERVERS_KEY,
  PROMPTS_KEY,
  HTTP_KEY,
  WEBSOCKET_KEY,
  PRICING_KEY,
  pricing,
  getPricing,
  getPricingForTool,
} from './decorators';

// Router
export {
  MessageRouter,
  matchesSubscription,
  matchesScope,
  SystemEvents,
  UAMPEventTypes,
  type UAMPEvent as RouterUAMPEvent,
  type ServerEvent as RouterServerEvent,
  type RouterContext,
  type Route,
  type Observer as RouterObserver,
  type TransportSink,
  type Handler,
  type UnroutableHandler,
  type ErrorHandler,
  type RouteInterceptor,
  type UAMPEventType,
} from './router';

// Transport Sinks
export {
  WebSocketSink,
  SSESink,
  WebStreamSink,
  CallbackSink,
  BufferSink,
  type SSEResponseWriter,
  type StreamController,
  type EventCallback,
} from './transport';

// Skill base class
export { Skill } from './skill';

// Context
export {
  ContextImpl,
  createContext,
  createSessionState,
  createDefaultAuthInfo,
  createDefaultPaymentInfo,
} from './context';

// Agent
export { BaseAgent } from './agent';

// Runtime
export {
  DefaultAgentRuntime,
  type AgentRuntime,
  type Extension,
  type AgentSource,
  type SkillFactory,
  type AgentInfo,
  type ExecuteOptions,
  type ExecuteResponse,
  type Middleware,
  type MiddlewareContext,
  type MiddlewareNext,
  type RuntimeHooks,
} from './runtime';

// Extensions
export {
  LocalDevExtension,
  LocalFileSource,
  LocalDevSkillFactory,
  type LocalDevExtensionOptions,
  type LocalFileSourceOptions,
} from './extensions/local-dev';

// Re-export commonly used types
export type {
  // Configuration
  ToolConfig,
  HookConfig,
  HandoffConfig,
  ObserveConfig,
  PromptConfig,
  HttpConfig,
  WebSocketConfig,
  SkillConfig,
  AgentConfig,
  RunOptions,
  
  // Registered items
  Tool,
  Hook,
  Handoff,
  Prompt,
  Observer,
  HttpEndpoint,
  WebSocketEndpoint,
  
  // Handlers
  ToolHandler,
  HookHandler,
  HandoffHandler,
  ObserverHandler,
  HttpHandler,
  WebSocketHandler,
  
  // Context
  Context,
  AuthInfo,
  PaymentInfo,
  SessionState,
  BridgeContext,
  
  // Hook types
  HookLifecycle,
  HookData,
  HookResult,
  
  // Response types
  RunResponse,
  StreamChunk,
  
  // Interfaces
  ISkill,
  IAgent,

  // Agentic loop
  AgenticMessage,

  // Pricing
  PricingConfig,
  PricingInfo,
} from './types';
