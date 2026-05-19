/**
 * Workspace Terminal sub-protocol types.
 *
 * This is the v1 wire vocabulary carried inside an `extension.message`
 * envelope:
 *
 * ```json
 * {
 *   "type": "extension.message",
 *   "namespace": "workspace.terminal",
 *   "extension_version": 1,
 *   "payload": { "type": "open", "session_id": "...", ... }
 * }
 * ```
 *
 * The envelope is owned by UAMP; everything in this file is owned by the
 * `workspace.terminal` namespace and is opaque to UAMP itself.
 *
 * `session_id` here is the per-PTY session id minted by the portal-side
 * gateway. It is NOT the UAMP `session_id` (which is omitted from
 * extension envelopes by design).
 */

/** Owned namespace name carried on the UAMP envelope. */
export const NAMESPACE = 'workspace.terminal' as const;

/**
 * Sub-protocol version. Bumped when the payload schema changes in a way
 * that older routers can't parse. Negotiation is receiver-validated.
 */
export const VERSION = 1 as const;

/** Per-router cap. Independent of the per-user cap on the portal side. */
export const MAX_SESSIONS_PER_ROUTER = 8 as const;

/** Rolling buffer for paused sessions, in bytes. */
export const PAUSED_BUFFER_BYTES = 256 * 1024;

/** Cell-dimension clamps for `resize`. */
export const MIN_DIM = 1;
export const MAX_DIM = 1024;

// ---------------------------------------------------------------------------
// Incoming payloads (portal → daemon)
// ---------------------------------------------------------------------------

export interface OpenPayload {
  type: 'open';
  session_id: string;
  /** Caller user-id (audit, not auth — auth happens on the portal). */
  peer_id: string;
  cols: number;
  rows: number;
}

export interface InPayload {
  type: 'in';
  session_id: string;
  /** Base64-encoded stdin bytes. */
  data: string;
}

export interface ResizePayload {
  type: 'resize';
  session_id: string;
  cols: number;
  rows: number;
}

export interface PausePayload {
  type: 'pause';
  session_id: string;
}

export interface ResumePayload {
  type: 'resume';
  session_id: string;
}

export interface ClosePayload {
  type: 'close';
  session_id: string;
  reason?: string;
}

export type IncomingPayload =
  | OpenPayload
  | InPayload
  | ResizePayload
  | PausePayload
  | ResumePayload
  | ClosePayload;

// ---------------------------------------------------------------------------
// Outgoing payloads (daemon → portal)
// ---------------------------------------------------------------------------

export interface ReadyPayload {
  type: 'ready';
  session_id: string;
}

export interface OutPayload {
  type: 'out';
  session_id: string;
  /** Base64-encoded stdout/stderr bytes. */
  data: string;
}

export interface ExitPayload {
  type: 'exit';
  session_id: string;
  code: number | null;
  signal?: string | null;
}

/**
 * Sub-protocol-typed errors. Each code is additive — once shipped, code
 * meaning never changes; new codes only get added. UAMP carries no
 * generic `extension.unsupported`; this is the namespace's error model.
 */
export type TerminalErrorCode =
  | 'not_supported'
  | 'bad_payload'
  | 'pty_open_failed'
  | 'pty_write_failed'
  | 'pty_resize_failed'
  | 'pty_close_failed'
  | 'unsupported_version'
  | 'concurrency_limit'
  | 'duplicate_session';

export interface ErrPayload {
  type: 'err';
  session_id: string;
  code: TerminalErrorCode;
  message: string;
}

export type OutgoingPayload = ReadyPayload | OutPayload | ExitPayload | ErrPayload;

// ---------------------------------------------------------------------------
// Envelope sender contract
// ---------------------------------------------------------------------------

/**
 * Contract implemented by the transport that owns the WebSocket. The
 * router calls this to emit outgoing sub-protocol payloads; the
 * transport wraps each one in an `extension.message` envelope before
 * sending it on the wire. Decoupling the router from the wire lets us
 * unit-test it without a live socket.
 */
export type EnvelopeSender = (payload: OutgoingPayload) => void;

// ---------------------------------------------------------------------------
// Frame predicate inputs
// ---------------------------------------------------------------------------

/**
 * Minimal shape the router needs to decide whether a UAMP frame belongs
 * to it. Kept loose so callers don't have to import the full
 * `ExtensionMessageEvent` type — most receivers only know they have a
 * parsed JSON object.
 */
export interface MaybeEnvelope {
  type?: string;
  namespace?: string;
  payload?: unknown;
  extension_version?: number;
}
