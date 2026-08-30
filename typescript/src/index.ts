/**
 * webagents-ts - TypeScript SDK for in-browser AI agents
 * 
 * @packageDocumentation
 */

// UAMP Protocol
export * from './uamp/index';

// Core Framework
export * from './core/index';

// Crypto (JWKS, JWT verification)
export * from './crypto/index';

// Skills
export * from './skills/index';

// Server
export * from './server/index';

// Daemon
export * from './daemon/index';

// Portal transport internals: the reverse-WebSocket bridge behind
// PortalConnectSkill, plus the credential guard (checkAgentToken /
// PortalCredentialError). Not an entry point — attach the skill and serve().
export * from './portal/index';

// Re-export commonly used types
export type {
  Capabilities,
  Modality,
  AudioFormat,
  SessionConfig,
} from './uamp/types';

export type {
  ToolConfig,
  HookConfig,
  HandoffConfig,
  HttpConfig,
  WebSocketConfig,
} from './core/types';
