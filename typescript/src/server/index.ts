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

// The well-known signatures directory a `legacy-string` signer's bare origin
// resolves to, for anyone serving an agent through a framework of their own.
export {
  DIRECTORY_WELL_KNOWN_PATH,
  DIRECTORY_MEDIA_TYPE,
  DIRECTORY_MAX_KEYS,
  directoryKeys,
  isKeyDirectoryRequest,
  keyDirectoryResponse,
} from './key-directory';
export type { DirectoryKeySource } from './key-directory';

// The platform registration surface. `registerWithPlatform` was reachable
// only by importing the module path directly, which is the same as not
// existing: the one call that turns a served card into a registered agent was
// absent from the package's public API.
export {
  startHeartbeat,
  registerWithPlatform,
  // The claim link an ownerless agent prints for its operator. Exported for
  // the same reason `registerWithPlatform` had to be: a call reachable only by
  // deep import is the same as a call that does not exist.
  claimUrl,
  resolvePlatformBaseUrl,
  resolvePortalApiUrl,
  resolveAgentToken,
  HEARTBEAT_INTERVAL_MS,
  OWNER_KEY_HEADER,
} from './registration';
export type {
  HeartbeatHandle,
  RegisterWithPlatformOptions,
  PlatformRegistrationResult,
} from './registration';
