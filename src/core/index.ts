/**
 * Core Framework Module
 * 
 * Base classes and utilities for building agents and skills.
 */

// Types
export * from './types.js';

// Decorators
export {
  tool,
  hook,
  handoff,
  observe,
  http,
  websocket,
  getTools,
  getHooks,
  getHandoffs,
  getObservers,
  getHttpEndpoints,
  getWebSocketEndpoints,
  TOOLS_KEY,
  HOOKS_KEY,
  HANDOFFS_KEY,
  OBSERVERS_KEY,
  HTTP_KEY,
  WEBSOCKET_KEY,
} from './decorators.js';

// Router
export {
  MessageRouter,
  matchesSubscription,
  SystemEvents,
  UAMPEventTypes,
  type UAMPEvent,
  type ServerEvent,
  type RouterContext,
  type Route,
  type Observer as RouterObserver,
  type TransportSink,
  type Handler,
  type UnroutableHandler,
  type ErrorHandler,
  type RouteInterceptor,
  type UAMPEventType,
} from './router.js';

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
} from './transport.js';

// Skill base class
export { Skill } from './skill.js';

// Context
export {
  ContextImpl,
  createContext,
  createSessionState,
  createDefaultAuthInfo,
  createDefaultPaymentInfo,
} from './context.js';

// Agent
export { BaseAgent } from './agent.js';

// Re-export commonly used types
export type {
  // Configuration
  ToolConfig,
  HookConfig,
  HandoffConfig,
  ObserveConfig,
  HttpConfig,
  WebSocketConfig,
  SkillConfig,
  AgentConfig,
  RunOptions,
  
  // Registered items
  Tool,
  Hook,
  Handoff,
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
} from './types.js';
