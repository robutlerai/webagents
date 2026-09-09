/**
 * Server Module
 * 
 * HTTP/WebSocket server implementations.
 */

export { createAgentApp, serve } from './node';
export type { ServerConfig } from './node';

export { createFetchHandler } from './handler';
export type { HandlerOptions } from './handler';

export { WebAgentsServer } from './multi';
export type { WebAgentsServerConfig, RateLimitConfig, ExtensionLoader } from './multi';

// The platform registration surface. `registerWithPlatform` was reachable
// only by importing the module path directly, which is the same as not
// existing: the one call that turns a served card into a registered agent was
// absent from the package's public API.
export {
  startHeartbeat,
  registerWithPlatform,
  resolvePlatformBaseUrl,
  resolvePortalApiUrl,
  resolveAgentToken,
  HEARTBEAT_INTERVAL_MS,
} from './registration';
export type {
  HeartbeatHandle,
  RegisterWithPlatformOptions,
  PlatformRegistrationResult,
} from './registration';
