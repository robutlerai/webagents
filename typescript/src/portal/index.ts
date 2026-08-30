/**
 * Portal transport internals.
 *
 * The `connect()` / `host()` entry points that used to live here are GONE.
 * They were wrappers: everything they added belongs to the normal path and
 * now lives there — the agent card at the origin with `metadata.publicKey`,
 * the persisted signing key and the presence heartbeat are `serve()`'s
 * (src/server/node.ts), and starting the reverse bridge is
 * `PortalConnectSkill`'s (src/skills/transport/portal-connect).
 *
 * What remains is the transport itself plus the credential guard worth
 * keeping.
 */
export {
  runPortalBridge,
  checkAgentToken,
  resolvePortalWsUrl,
  sanitizeMessages,
  PortalCredentialError,
  type PortalBridgeOptions,
  type TerminalRouterLike,
} from './connect';
