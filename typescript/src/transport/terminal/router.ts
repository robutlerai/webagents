/**
 * Workspace Terminal Router
 *
 * Bridges the `workspace.terminal` UAMP sub-protocol to a Tauri-hosted
 * PTY surface. NOT a Skill — terminals are peer-machine control, not
 * an agent capability.
 *
 * Wire (browser↔portal binary, portal↔daemon UAMP envelope):
 *   browser ──[binary]──► portal ──[ext-msg]──► daemon ──► TerminalRouter
 *                                                                │
 *                                            invoke('pty_*')     ▼
 *                                                              Tauri
 *
 * Hosts:
 *   - Tauri (`globalThis.__TAURI_INTERNALS__` present): real PTYs.
 *   - Anything else: every `open` returns `{ code: 'not_supported' }`.
 *
 * Per-router invariants:
 *   - 8 concurrent sessions max (`MAX_SESSIONS_PER_ROUTER`).
 *   - 256 KiB rolling buffer per paused session.
 *   - Every Tauri `invoke` is wrapped with an explicit error contract.
 *   - `shutdown(reason)` releases all sessions and listeners.
 */

import {
  MAX_DIM,
  MAX_SESSIONS_PER_ROUTER,
  MIN_DIM,
  NAMESPACE,
  PAUSED_BUFFER_BYTES,
  VERSION,
  type EnvelopeSender,
  type ErrPayload,
  type IncomingPayload,
  type MaybeEnvelope,
  type TerminalErrorCode,
} from './types.js';

// ---------------------------------------------------------------------------
// Tauri surface (typed minimally; loaded dynamically only when present)
// ---------------------------------------------------------------------------

interface TauriInvoke {
  <T>(cmd: string, args?: Record<string, unknown>): Promise<T>;
}

interface TauriEvent<T> {
  payload: T;
}

interface TauriUnlistenFn {
  (): void;
}

interface TauriEventApi {
  listen<T>(
    event: string,
    handler: (event: TauriEvent<T>) => void,
  ): Promise<TauriUnlistenFn>;
}

interface TauriBundle {
  invoke: TauriInvoke;
  events: TauriEventApi;
}

interface PtyDataPayload {
  ptySid: string;
  data: string; // base64
}

interface PtyExitPayload {
  ptySid: string;
  code: number | null;
  signal?: string | null;
}

// ---------------------------------------------------------------------------
// Per-session bookkeeping
// ---------------------------------------------------------------------------

interface SessionEntry {
  ptySid: string;
  paused: boolean;
  /** Rolling buffer of base64-encoded chunks while paused. */
  pausedBuf: string[];
  pausedBufBytes: number;
  unlistenData: TauriUnlistenFn | null;
  unlistenExit: TauriUnlistenFn | null;
}

// ---------------------------------------------------------------------------
// Host detection
// ---------------------------------------------------------------------------

export type Host = 'tauri' | 'unsupported';

function detectHost(): Host {
  // Tauri injects `__TAURI_INTERNALS__` into the webview's global before
  // any user JS runs. This is the canonical signal — `window.__TAURI__`
  // is also set on classic builds, but `__TAURI_INTERNALS__` is the
  // forward-compatible one.
  const g = globalThis as unknown as Record<string, unknown>;
  if (g.__TAURI_INTERNALS__) return 'tauri';
  return 'unsupported';
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

export interface TerminalRouterOptions {
  /**
   * Override host detection. Useful for tests (`'unsupported'`) and
   * future hosts (e.g. when we add a Node fallback).
   */
  host?: Host;
  /**
   * Override Tauri import. Tests inject a mock; production calls
   * `importTauri()` lazily on first `open`.
   */
  importTauri?: () => Promise<TauriBundle>;
}

/**
 * Plain class. Not a Skill, not a Capability — sub-protocol owner only.
 */
export class TerminalRouter {
  private readonly host: Host;
  private readonly importTauri: () => Promise<TauriBundle>;
  private tauri: TauriBundle | null = null;
  private readonly sessions = new Map<string, SessionEntry>();
  private shutdownReason: string | null = null;

  constructor(opts: TerminalRouterOptions = {}) {
    this.host = opts.host ?? detectHost();
    this.importTauri = opts.importTauri ?? defaultImportTauri;
  }

  /** Predicate for the parent transport's dispatch fast-path. */
  static isFor(msg: MaybeEnvelope): boolean {
    return msg.type === 'extension.message' && msg.namespace === NAMESPACE;
  }

  /** Sub-protocol version this router speaks. */
  static get VERSION(): number {
    return VERSION;
  }

  /**
   * Handle a single sub-protocol payload.
   *
   * The transport has already unwrapped the `extension.message`
   * envelope; this method receives the bare payload plus the version
   * from the envelope (if any).
   */
  async handlePayload(
    payload: unknown,
    send: EnvelopeSender,
    opts: { extension_version?: number } = {},
  ): Promise<void> {
    if (this.shutdownReason !== null) return; // post-shutdown, drop silently

    const p = validatePayload(payload);
    if (!p) {
      const sid = extractSessionId(payload);
      send(this.err(sid, 'bad_payload', 'malformed workspace.terminal payload'));
      return;
    }

    const requested = opts.extension_version ?? 1;
    if (requested > VERSION) {
      send(
        this.err(
          p.session_id,
          'unsupported_version',
          `workspace.terminal v${VERSION} cannot handle extension_version=${requested}`,
        ),
      );
      return;
    }

    switch (p.type) {
      case 'open':
        return this.handleOpen(p.session_id, p.peer_id, p.cols, p.rows, send);
      case 'in':
        return this.handleIn(p.session_id, p.data, send);
      case 'resize':
        return this.handleResize(p.session_id, p.cols, p.rows, send);
      case 'pause':
        return this.handlePause(p.session_id);
      case 'resume':
        return this.handleResume(p.session_id, send);
      case 'close':
        return this.handleClose(p.session_id, send);
    }
  }

  /**
   * Tear everything down. Idempotent — safe to call from multiple
   * close paths (parent ws close, daemon shutdown, etc).
   */
  async shutdown(reason: string): Promise<void> {
    if (this.shutdownReason !== null) return;
    this.shutdownReason = reason;
    const sids = [...this.sessions.keys()];
    for (const sid of sids) {
      const entry = this.sessions.get(sid);
      if (!entry) continue;
      this.detachListeners(entry);
      try {
        if (this.tauri) {
          await this.tauri.invoke('pty_close', { ptySid: entry.ptySid });
        }
      } catch {
        // best-effort — process may already be gone
      }
      this.sessions.delete(sid);
    }
  }

  // -------------------------------------------------------------------------
  // Per-payload handlers
  // -------------------------------------------------------------------------

  private async handleOpen(
    session_id: string,
    _peer_id: string,
    cols: number,
    rows: number,
    send: EnvelopeSender,
  ): Promise<void> {
    if (this.host === 'unsupported') {
      send(this.err(session_id, 'not_supported', 'this daemon has no PTY surface'));
      return;
    }
    if (this.sessions.has(session_id)) {
      send(this.err(session_id, 'duplicate_session', 'session_id already open'));
      return;
    }
    if (this.sessions.size >= MAX_SESSIONS_PER_ROUTER) {
      send(
        this.err(
          session_id,
          'concurrency_limit',
          `per-router cap of ${MAX_SESSIONS_PER_ROUTER} sessions reached`,
        ),
      );
      return;
    }

    let tauri: TauriBundle;
    try {
      tauri = await this.ensureTauri();
    } catch (err) {
      send(
        this.err(
          session_id,
          'pty_open_failed',
          `tauri api load failed: ${describe(err)}`,
        ),
      );
      return;
    }

    let ptySid: string;
    try {
      ptySid = await tauri.invoke<string>('pty_open', {
        cols: clampDim(cols),
        rows: clampDim(rows),
      });
    } catch (err) {
      // No state stored, no listeners attached — failure is total.
      send(this.err(session_id, 'pty_open_failed', describe(err)));
      return;
    }

    const entry: SessionEntry = {
      ptySid,
      paused: false,
      pausedBuf: [],
      pausedBufBytes: 0,
      unlistenData: null,
      unlistenExit: null,
    };

    try {
      entry.unlistenData = await tauri.events.listen<PtyDataPayload>(
        'pty:data',
        (ev) => {
          if (ev.payload.ptySid !== ptySid) return;
          this.onPtyData(session_id, ev.payload.data, send);
        },
      );
      entry.unlistenExit = await tauri.events.listen<PtyExitPayload>(
        'pty:exit',
        (ev) => {
          if (ev.payload.ptySid !== ptySid) return;
          this.onPtyExit(session_id, ev.payload.code, ev.payload.signal ?? null, send);
        },
      );
    } catch (err) {
      // Listener attach failed — close the PTY we just opened so we
      // don't orphan a process. This is best-effort.
      try {
        await tauri.invoke('pty_close', { ptySid });
      } catch {
        /* ignore */
      }
      this.detachListeners(entry);
      send(this.err(session_id, 'pty_open_failed', `listener attach: ${describe(err)}`));
      return;
    }

    this.sessions.set(session_id, entry);
    send({ type: 'ready', session_id });
  }

  private async handleIn(
    session_id: string,
    data: string,
    send: EnvelopeSender,
  ): Promise<void> {
    const entry = this.sessions.get(session_id);
    if (!entry) return; // race with close, drop silently
    if (!this.tauri) return; // shouldn't happen if entry exists, defensive
    try {
      await this.tauri.invoke('pty_write', { ptySid: entry.ptySid, data });
    } catch (err) {
      // Stdin failure is non-fatal — stdout may still be useful, e.g.
      // user wants to see what the failing process printed before
      // closing.
      send(this.err(session_id, 'pty_write_failed', describe(err)));
    }
  }

  private async handleResize(
    session_id: string,
    cols: number,
    rows: number,
    send: EnvelopeSender,
  ): Promise<void> {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    if (!this.tauri) return;
    try {
      await this.tauri.invoke('pty_resize', {
        ptySid: entry.ptySid,
        cols: clampDim(cols),
        rows: clampDim(rows),
      });
    } catch (err) {
      send(this.err(session_id, 'pty_resize_failed', describe(err)));
    }
  }

  private handlePause(session_id: string): void {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    entry.paused = true;
  }

  private handleResume(session_id: string, send: EnvelopeSender): void {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    entry.paused = false;
    // Drain the rolling buffer in order. Each chunk is already base64
    // and was filtered by ptySid at enqueue time.
    const drained = entry.pausedBuf;
    entry.pausedBuf = [];
    entry.pausedBufBytes = 0;
    for (const data of drained) {
      send({ type: 'out', session_id, data });
    }
  }

  private async handleClose(
    session_id: string,
    _send: EnvelopeSender,
  ): Promise<void> {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    this.detachListeners(entry);
    this.sessions.delete(session_id);
    if (!this.tauri) return;
    try {
      await this.tauri.invoke('pty_close', { ptySid: entry.ptySid });
    } catch {
      // Process may have already exited via SIGCHLD; non-fatal.
    }
  }

  // -------------------------------------------------------------------------
  // Tauri event handlers
  // -------------------------------------------------------------------------

  private onPtyData(session_id: string, data: string, send: EnvelopeSender): void {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    if (entry.paused) {
      // Rolling 256 KiB cap, oldest chunks dropped first. We approximate
      // chunk byte length using the base64 string length (slightly
      // larger than the decoded bytes, but cheap and conservative).
      entry.pausedBuf.push(data);
      entry.pausedBufBytes += data.length;
      while (entry.pausedBufBytes > PAUSED_BUFFER_BYTES && entry.pausedBuf.length > 1) {
        const dropped = entry.pausedBuf.shift();
        if (dropped !== undefined) entry.pausedBufBytes -= dropped.length;
      }
      return;
    }
    send({ type: 'out', session_id, data });
  }

  private onPtyExit(
    session_id: string,
    code: number | null,
    signal: string | null,
    send: EnvelopeSender,
  ): void {
    const entry = this.sessions.get(session_id);
    if (!entry) return;
    this.detachListeners(entry);
    this.sessions.delete(session_id);
    send({ type: 'exit', session_id, code, signal });
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private async ensureTauri(): Promise<TauriBundle> {
    if (this.tauri) return this.tauri;
    this.tauri = await this.importTauri();
    return this.tauri;
  }

  private detachListeners(entry: SessionEntry): void {
    try {
      entry.unlistenData?.();
    } catch {
      /* ignore */
    }
    try {
      entry.unlistenExit?.();
    } catch {
      /* ignore */
    }
    entry.unlistenData = null;
    entry.unlistenExit = null;
  }

  private err(session_id: string, code: TerminalErrorCode, message: string): ErrPayload {
    return { type: 'err', session_id, code, message };
  }

  // Test helpers (intentionally not in `OutgoingPayload` — used by
  // unit tests to assert internal state without poking privates).
  /** @internal */
  _peekSessionCount(): number {
    return this.sessions.size;
  }
  /** @internal */
  _peekPausedBufferBytes(session_id: string): number {
    return this.sessions.get(session_id)?.pausedBufBytes ?? 0;
  }
  /** @internal */
  _injectPtyData(session_id: string, data: string, send: EnvelopeSender): void {
    this.onPtyData(session_id, data, send);
  }
  /** @internal */
  get _host(): Host {
    return this.host;
  }
}

// ---------------------------------------------------------------------------
// Default Tauri loader
// ---------------------------------------------------------------------------

async function defaultImportTauri(): Promise<TauriBundle> {
  // Dynamic import keeps `@tauri-apps/api` an optional peer-dep — the
  // module is only resolved on Tauri hosts where it's guaranteed
  // present. Plain Node daemons never reach this path because
  // `host === 'unsupported'`.
  const core = (await import('@tauri-apps/api/core')) as { invoke: TauriInvoke };
  const event = (await import('@tauri-apps/api/event')) as TauriEventApi;
  return { invoke: core.invoke, events: { listen: event.listen.bind(event) } };
}

// ---------------------------------------------------------------------------
// Payload validation
// ---------------------------------------------------------------------------

function validatePayload(p: unknown): IncomingPayload | null {
  if (!p || typeof p !== 'object') return null;
  const obj = p as Record<string, unknown>;
  const t = obj.type;
  const sid = obj.session_id;
  if (typeof t !== 'string' || typeof sid !== 'string' || sid.length === 0) return null;
  switch (t) {
    case 'open': {
      const peerId = obj.peer_id;
      const cols = obj.cols;
      const rows = obj.rows;
      if (typeof peerId !== 'string') return null;
      if (typeof cols !== 'number' || typeof rows !== 'number') return null;
      return { type: 'open', session_id: sid, peer_id: peerId, cols, rows };
    }
    case 'in': {
      const data = obj.data;
      if (typeof data !== 'string') return null;
      return { type: 'in', session_id: sid, data };
    }
    case 'resize': {
      const cols = obj.cols;
      const rows = obj.rows;
      if (typeof cols !== 'number' || typeof rows !== 'number') return null;
      return { type: 'resize', session_id: sid, cols, rows };
    }
    case 'pause':
      return { type: 'pause', session_id: sid };
    case 'resume':
      return { type: 'resume', session_id: sid };
    case 'close': {
      const reason = obj.reason;
      if (reason !== undefined && typeof reason !== 'string') return null;
      return { type: 'close', session_id: sid, reason };
    }
    default:
      return null;
  }
}

function extractSessionId(p: unknown): string {
  if (p && typeof p === 'object') {
    const sid = (p as Record<string, unknown>).session_id;
    if (typeof sid === 'string' && sid.length > 0) return sid;
  }
  return '';
}

function clampDim(n: number): number {
  if (!Number.isFinite(n)) return 80;
  return Math.max(MIN_DIM, Math.min(MAX_DIM, Math.trunc(n)));
}

function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}
