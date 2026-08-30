/**
 * The reverse WebSocket bridge (U2), TypeScript edition — the TRANSPORT, not
 * an entry point.
 *
 * Speaks the platform's REAL `/ws` contract: `session.create` with
 * `{agent, token}`, `input.text` in (with full conversation history and a
 * payment token), `response.delta` / `response.done` out. This replaces the
 * three dead paths that previously claimed the job: `portal_register` (a
 * body the route's schema rejects), `PortalWSSkill` and the old
 * `PortalTransportSkill.exposeAgent` (both spoke a `register` protocol the
 * platform's `/ws` never handled — the frames were dropped on the floor).
 *
 * `runPortalBridge()` is deliberately NOT the documented way to put an agent
 * on the platform. It used to be exported as `connect(agent)`, a one-word
 * call that hid the lifecycle; what developers write now is
 * `new PortalConnectSkill()` on the agent and `serve()` — the skill owns this
 * loop (src/skills/transport/portal-connect/skill.ts).
 */

import type { IAgent } from '../core/types';
import type { Message } from '../uamp/types';
import { createExtensionMessage } from '../uamp/events';

/** `workspace.terminal` — the namespace on the portal's terminal envelope
 *  (lib/terminal/backend-webagentsd.ts). Mirrors
 *  src/transport/terminal/types.ts NAMESPACE/VERSION without importing that
 *  module, which is only loaded when terminal support is on. */
const TERMINAL_NAMESPACE = 'workspace.terminal';
const TERMINAL_VERSION = 1;

export interface PortalBridgeOptions {
  /** Portal URL — http(s) accepted (converted to ws(s)); `/ws` appended to a bare origin. Falls back to WEBAGENTS_PORTAL_URL / PORTAL_WS_URL. */
  portalUrl?: string;
  /** Per-agent token. Falls back to WEBAGENTS_AGENT_TOKEN. */
  token?: string;
  /** Ping interval seconds (default 55). */
  pingIntervalS?: number;
  /** Abort to disconnect and resolve `runPortalBridge()`. */
  signal?: AbortSignal;
  /**
   * Workspace `terminal` node support (the `workspace.terminal`
   * `extension.message` envelope the portal sends onto THIS socket —
   * lib/terminal/backend-webagentsd.ts). `true` constructs a default
   * `TerminalRouter`; pass an instance to share one across transports.
   * Default false: a host that does not own a PTY surface should not claim
   * the namespace (the portal's session-open timeout turns that into a
   * `peer_offline` close reason). An envelope that arrives while this is off
   * is WARNED about once, never dropped in silence.
   */
  terminal?: boolean | TerminalRouterLike;
  /** Reconnect on an unexpected close (default true). */
  autoReconnect?: boolean;
  /** Seconds between reconnect attempts (default 5). */
  reconnectDelayS?: number;
  /** Max consecutive reconnect attempts before giving up (default 10). */
  maxReconnectAttempts?: number;
}

/** The slice of `TerminalRouter` `runPortalBridge()` uses (kept structural so the
 *  router module is only loaded when terminal support is actually on). */
export interface TerminalRouterLike {
  handlePayload(
    payload: unknown,
    send: (payload: unknown) => void,
    opts?: { extension_version?: number },
  ): Promise<void>;
  shutdown(reason: string): Promise<void>;
}

export class PortalCredentialError extends Error {}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

/** Resolve the portal WS URL from an option or the environment. */
export function resolvePortalWsUrl(configured?: string): string {
  let raw = (
    configured ||
    envVar('WEBAGENTS_PORTAL_URL') ||
    envVar('PORTAL_WS_URL') ||
    'wss://robutler.ai/ws'
  ).trim();
  if (raw.startsWith('https://')) raw = 'wss://' + raw.slice('https://'.length);
  else if (raw.startsWith('http://')) raw = 'ws://' + raw.slice('http://'.length);
  const schemeEnd = raw.indexOf('://');
  const rest = schemeEnd === -1 ? raw : raw.slice(schemeEnd + 3);
  if (!rest.includes('/')) raw = raw.replace(/\/+$/, '') + '/ws';
  return raw;
}

function decodeJwtClaims(token: string): Record<string, unknown> {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return {};
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4);
    const decoded =
      typeof Buffer !== 'undefined'
        ? Buffer.from(padded, 'base64').toString()
        : atob(padded);
    return JSON.parse(decoded) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/**
 * Refuse — at startup, with the fix in the message — the credential shape
 * that costs people days: an owner key with no agent binding (F-045). A
 * per-agent key from POST /api/agents/{id}/api-key carries an `agent_id`
 * claim; a generic owner key does not, and with it the socket registers
 * under the OWNER, the router never finds the agent's session, and the
 * daemon idles forever while every observable looks healthy.
 *
 * Set WEBAGENTS_ALLOW_UNBOUND_TOKEN=1 to bypass (custom deployments only).
 */
export function checkAgentToken(token: string | undefined): asserts token is string {
  if (!token) {
    throw new PortalCredentialError(
      'No agent token configured. Set WEBAGENTS_AGENT_TOKEN to a per-agent ' +
        'API key minted with POST /api/agents/{id}/api-key.',
    );
  }
  if (envVar('WEBAGENTS_ALLOW_UNBOUND_TOKEN') === '1') return;
  const claims = decodeJwtClaims(token);
  if (Object.keys(claims).length === 0) return; // not a readable JWT; let the platform judge
  if (!claims.agent_id) {
    throw new PortalCredentialError(
      `The configured token is not bound to an agent: its subject is '${String(
        claims.sub ?? '?',
      )}' and it carries no agent_id claim. This is an owner/account key — ` +
        'the platform will accept the connection and then never route a ' +
        'single message to this agent. Fix: mint a per-agent key with ' +
        'POST /api/agents/{id}/api-key (the returned JWT carries agent_id) ' +
        'and put THAT in WEBAGENTS_AGENT_TOKEN.',
    );
  }
}

interface WsLike {
  send(data: string): void;
  close(): void;
  on(event: string, cb: (...args: unknown[]) => void): void;
}

/**
 * Run the reverse WS bridge for one agent until `options.signal` aborts (or
 * the socket closes without reconnection); with no signal it runs for the
 * process lifetime. Callers own the lifecycle — `PortalConnectSkill` starts
 * this from the agent/server lifecycle and aborts it on cleanup.
 */
export async function runPortalBridge(
  agent: IAgent,
  options: PortalBridgeOptions = {},
): Promise<void> {
  const configuredToken = options.token ?? envVar('WEBAGENTS_AGENT_TOKEN');
  checkAgentToken(configuredToken);
  // Bound to a plain string: the assertion above narrows `configuredToken`
  // here, but not inside the per-connection closure below.
  const token: string = configuredToken;
  const wsUrl = resolvePortalWsUrl(options.portalUrl);
  const pingIntervalMs = (options.pingIntervalS ?? 55) * 1000;
  const autoReconnect = options.autoReconnect ?? true;
  const reconnectDelayMs = (options.reconnectDelayS ?? 5) * 1000;
  const maxReconnectAttempts = options.maxReconnectAttempts ?? 10;

  const { WebSocket } = (await import('ws')) as unknown as {
    WebSocket: new (url: string) => WsLike;
  };

  const initFn = (agent as { initialize?: () => Promise<void> }).initialize;
  if (typeof initFn === 'function') await initFn.call(agent);

  const sessions = new Map<string, string>(); // session_id -> agent string
  const inflight = new Map<string, AbortController>(); // session_id -> abort

  // Workspace terminal router: constructed at most once, lazily, and only
  // when terminal support is on — the module reaches for a Tauri PTY host.
  let terminal: TerminalRouterLike | null =
    options.terminal && typeof options.terminal === 'object' ? options.terminal : null;
  let terminalWarned = false;
  const ensureTerminal = async (): Promise<TerminalRouterLike | null> => {
    if (terminal) return terminal;
    if (options.terminal !== true) return null;
    const { TerminalRouter } = await import('../transport/terminal/index.js');
    terminal = new TerminalRouter() as unknown as TerminalRouterLike;
    return terminal;
  };

  let stopped = false;
  options.signal?.addEventListener('abort', () => {
    stopped = true;
  });

  const sleep = (ms: number): Promise<void> =>
    new Promise((resolve) => {
      const timer = setTimeout(resolve, ms);
      options.signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        resolve();
      });
    });

  let attempts = 0;
  for (;;) {
    await runOneConnection();
    if (stopped || !autoReconnect) break;
    if (attempts >= maxReconnectAttempts) {
      console.error(
        `[webagents] giving up after ${attempts} reconnect attempts to ${wsUrl}`,
      );
      break;
    }
    attempts += 1;
    console.warn(`[webagents] disconnected; reconnecting (attempt ${attempts})`);
    await sleep(reconnectDelayMs);
    if (stopped) break;
  }
  if (terminal) await terminal.shutdown('portal_disconnect');

  /**
   * One socket lifetime. Resolves when it closes (or the caller aborts);
   * never rejects — a dropped socket is a reconnect, not the end of the
   * bridge. Before this, a single `close` resolved `connect()` for the whole
   * process lifetime: the agent stayed up and silent forever.
   */
  function runOneConnection(): Promise<void> {
    return new Promise<void>((resolve) => {
      const ws = new WebSocket(`${wsUrl}?token=${encodeURIComponent(token)}`);
      let pingTimer: ReturnType<typeof setInterval> | null = null;
      let settled = false;

      const finish = () => {
        if (settled) return;
        settled = true;
        if (pingTimer) clearInterval(pingTimer);
        resolve();
      };

      const onAbort = () => {
        try {
          ws.close();
        } catch {
          /* closing */
        }
        finish();
      };
      options.signal?.addEventListener('abort', onAbort);

      ws.on('open', () => {
        attempts = 0; // a successful connection resets the backoff budget
        ws.send(
          JSON.stringify({
            type: 'session.create',
            event_id: `evt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
            timestamp: Date.now(),
            uamp_version: '1.0',
            session: { agent: agent.name, token },
          }),
        );
        pingTimer = setInterval(() => {
          try {
            ws.send(JSON.stringify({ type: 'ping', event_id: `evt_${Date.now().toString(36)}`, timestamp: Date.now() }));
          } catch {
            /* socket closing */
          }
        }, pingIntervalMs);
      });

      ws.on('error', (err: unknown) => {
        console.error(
          `[webagents] socket error: ${err instanceof Error ? err.message : String(err)}`,
        );
        finish();
      });
      ws.on('close', () => finish());

      ws.on('message', (raw: unknown) => {
        void handleFrame(String(raw));
      });

      async function handleFrame(text: string): Promise<void> {
        let m: Record<string, unknown>;
        try {
          m = JSON.parse(text) as Record<string, unknown>;
        } catch {
          return;
        }

        // BROADCAST discrimination: the wrapped chat envelope
        // {event, chatId, _origin, _portal} is a COPY of what happened, fanned
        // out to everyone on the user's channel. It must never start a run —
        // both shapes arrive for the same turn, so treating message.created as
        // work runs the model twice and bills twice.
        if (typeof m.type !== 'string') {
          return;
        }

        switch (m.type) {
          case 'session.created': {
            const sid = m.session_id as string | undefined;
            const session = (m.session ?? {}) as { agent?: string };
            if (sid && session.agent) sessions.set(sid, session.agent);
            console.log(`[webagents] session.created for ${session.agent ?? agent.name}`);
            return;
          }
          case 'session.error': {
            // The primary observable for a refused connect credential.
            const error = (m.error ?? {}) as { code?: string; message?: string };
            console.error(
              `[webagents] session.error code=${error.code}: ${error.message}`,
            );
            return;
          }
          case 'session.end': {
            const sid = m.session_id as string | undefined;
            if (sid) {
              sessions.delete(sid);
              inflight.get(sid)?.abort();
              inflight.delete(sid);
            }
            return;
          }
          case 'extension.message': {
            // The workspace `terminal` node. The portal sends exactly
            // `{type:'extension.message', namespace:'workspace.terminal',
            // extension_version, payload}` onto THIS socket
            // (lib/terminal/backend-webagentsd.ts -> sendToAgent). Without
            // this branch the frame hit `default: return` and the terminal
            // silently never opened.
            if (m.namespace !== 'workspace.terminal') {
              console.warn(`[webagents] unhandled extension namespace: ${String(m.namespace)}`);
              return;
            }
            const router = await ensureTerminal();
            if (!router) {
              if (!terminalWarned) {
                terminalWarned = true;
                console.warn(
                  '[webagents] received a workspace.terminal envelope but terminal ' +
                    'support is off. Pass `new PortalConnectSkill({ terminal: true })` (or a ' +
                    'TerminalRouter) on a host that owns a PTY surface; the portal will ' +
                    'otherwise time the session out as peer_offline.',
                );
              }
              return;
            }
            const payload = m.payload;
            if (!payload || typeof payload !== 'object') return; // malformed, drop
            const send = (out: unknown) => {
              try {
                ws.send(
                  JSON.stringify(
                    createExtensionMessage(TERMINAL_NAMESPACE, out, { version: TERMINAL_VERSION }),
                  ),
                );
              } catch (e) {
                console.warn(`[webagents] terminal send failed: ${(e as Error).message}`);
              }
            };
            try {
              await router.handlePayload(payload, send, {
                extension_version: (m.extension_version as number | undefined) ?? 1,
              });
            } catch (err) {
              console.warn(`[webagents] terminal handler error: ${(err as Error).message}`);
            }
            return;
          }
          case 'response.cancel': {
            const sid = m.session_id as string | undefined;
            if (sid) {
              inflight.get(sid)?.abort();
              inflight.delete(sid);
              ws.send(
                JSON.stringify({
                  type: 'response.cancelled',
                  event_id: `evt_${Date.now().toString(36)}`,
                  timestamp: Date.now(),
                  session_id: sid,
                }),
              );
            }
            return;
          }
          case 'input.text': {
            const sid = m.session_id as string | undefined;
            if (!sid) return;
            // Unknown sid: fall back to the frame's own `agent` field (the
            // multi-pod dispatch prerequisite — a per-request sid never
            // appears in a local session.created). INERT TODAY: no portal
            // emitter sets `agent` on the frame yet (sendInputToAgentSession
            // omits it; dispatchInputToAgent carries agentId on the Redis
            // wrapper, not the frame). Parser half only, until M5 lands.
            if (!sessions.has(sid)) {
              const frameAgent = m.agent as string | undefined;
              if (frameAgent) sessions.set(sid, frameAgent);
            }
            const abort = new AbortController();
            inflight.set(sid, abort);
            const rid = `resp_${Date.now().toString(36)}`;
            try {
              // Sanitise FIRST, then fall back. A history of only `role:'tool'`
              // rows (or assistant rows with no text) sanitises to nothing, and
              // the pre-sanitise check let that through as `runStreaming([])`.
              // Mirrors the Python guard in portal_connect/skill.py.
              let messages = sanitizeMessages(
                Array.isArray(m.messages) ? (m.messages as Message[]) : [],
              );
              if (messages.length === 0) {
                messages = [{ role: 'user', content: String(m.text ?? '') }] as Message[];
              }
              for await (const chunk of agent.runStreaming(messages)) {
                if (abort.signal.aborted) break;
                // TS BaseAgent yields StreamChunk {type:'delta', delta};
                // OpenAI-compatible generators yield choices[].delta.content.
                const c = chunk as {
                  type?: string;
                  delta?: string;
                  choices?: Array<{ delta?: { content?: string } }>;
                };
                const t =
                  (c.type === 'delta' && typeof c.delta === 'string' ? c.delta : undefined) ??
                  c.choices?.[0]?.delta?.content;
                if (t) {
                  ws.send(
                    JSON.stringify({
                      type: 'response.delta',
                      event_id: `evt_${Date.now().toString(36)}`,
                      timestamp: Date.now(),
                      session_id: sid,
                      response_id: rid,
                      delta: { type: 'text', text: t },
                    }),
                  );
                }
              }
              if (!abort.signal.aborted) {
                ws.send(
                  JSON.stringify({
                    type: 'response.done',
                    event_id: `evt_${Date.now().toString(36)}`,
                    timestamp: Date.now(),
                    session_id: sid,
                    response_id: rid,
                  }),
                );
              }
            } catch (err) {
              ws.send(
                JSON.stringify({
                  type: 'response.error',
                  event_id: `evt_${Date.now().toString(36)}`,
                  timestamp: Date.now(),
                  session_id: sid,
                  response_id: rid,
                  error: { code: 'agent_error', message: (err as Error).message },
                }),
              );
            } finally {
              if (inflight.get(sid) === abort) inflight.delete(sid);
            }
            return;
          }
          default:
            return; // pong, session.updated, payment.*, ...
        }
      }
    });
  }
}

/**
 * Reduce the platform's conversation history to plain OpenAI-shaped turns —
 * drops `role:'tool'` rows and non-text content parts, mirroring the Python
 * `sanitize_portal_messages`.
 */
export function sanitizeMessages(raw: Message[]): Message[] {
  const PLAIN = new Set(['system', 'user', 'assistant', 'developer']);
  const out: Message[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue;
    const role = (item as { role?: string }).role;
    if (!role || !PLAIN.has(role)) continue;
    let content = (item as { content?: unknown }).content;
    if (Array.isArray(content)) {
      content = content
        .filter((p): p is { type: string; text?: string } => !!p && typeof p === 'object' && (p as { type?: string }).type === 'text')
        .map((p) => p.text ?? '')
        .join('');
    }
    if (content == null) content = '';
    const text = typeof content === 'string' ? content : String(content);
    if (!text.trim()) continue;
    out.push({ role, content: text } as Message);
  }
  return out;
}
