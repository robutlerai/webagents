/**
 * Workspace Terminal transport — `workspace.terminal` namespace.
 *
 * Bridges the `extension.message` UAMP envelope to a Tauri PTY surface.
 * Not a Skill; not a Capability. The portal-transport pulls in the
 * router and hands it sub-protocol payloads it has unwrapped from the
 * envelope.
 */

export { TerminalRouter } from './router.js';
export type { TerminalRouterOptions, Host } from './router.js';
export {
  NAMESPACE,
  VERSION,
  MAX_DIM,
  MIN_DIM,
  MAX_SESSIONS_PER_ROUTER,
  PAUSED_BUFFER_BYTES,
} from './types.js';
export type {
  EnvelopeSender,
  ErrPayload,
  ExitPayload,
  IncomingPayload,
  InPayload,
  MaybeEnvelope,
  OpenPayload,
  OutgoingPayload,
  OutPayload,
  PausePayload,
  ReadyPayload,
  ResizePayload,
  ResumePayload,
  ClosePayload,
  TerminalErrorCode,
} from './types.js';
