/**
 * webagents-ts - TypeScript SDK for in-browser AI agents
 * 
 * @packageDocumentation
 */

// UAMP Protocol
export * from './uamp/index.js';

// Core Framework
export * from './core/index.js';

// Skills
export * from './skills/index.js';

// Server
export * from './server/index.js';

// Daemon
export * from './daemon/index.js';

// Re-export commonly used types
export type {
  Capabilities,
  Modality,
  AudioFormat,
  SessionConfig,
} from './uamp/types.js';

export type {
  ToolConfig,
  HookConfig,
  HandoffConfig,
  HttpConfig,
  WebSocketConfig,
} from './core/types.js';
