/**
 * Portal Transport Skill
 * 
 * Native UAMP over WebSocket for Elaisium and agent mesh.
 * No protocol conversion needed - UAMP events flow directly.
 */

import { Skill } from '../../../core/skill';
import { websocket } from '../../../core/decorators';
import type { SkillConfig, Context, IAgent } from '../../../core/types';
import type { ClientEvent, ServerEvent } from '../../../uamp/events';
import {
  serializeEvent,
  generateEventId,
  createPaymentRequiredEvent,
  createPaymentAcceptedEvent,
  createResponseCancelledEvent,
  createExtensionMessage,
} from '../../../uamp/events';
import type { Capabilities } from '../../../uamp/types';
import { PaymentRequiredError } from '../../payments/x402';
import {
  TerminalRouter,
  NAMESPACE as TERMINAL_NS,
  VERSION as TERMINAL_VER,
} from '../../../transport/terminal/index';
import type {
  IncomingPayload as TerminalIn,
  OutgoingPayload as TerminalOut,
} from '../../../transport/terminal/index';

/**
 * Portal message types (server mode). The old outbound `register` /
 * `unregister` shapes are gone: the platform's /ws never handled them —
 * those frames were dropped on the floor. Outbound connectivity now goes
 * through the real session.create contract (see exposeAgent()).
 */
interface PortalUAMPMessage {
  type: 'uamp';
  events: ClientEvent[];
  requestId?: string;
}

interface PortalDiscoverMessage {
  type: 'discover';
  query?: string;
}

type PortalMessage = PortalUAMPMessage | PortalDiscoverMessage;

interface PortalAgentsMessage {
  type: 'agents';
  agents: Array<{
    name: string;
    id: string;
    capabilities: Capabilities;
  }>;
}

/**
 * Portal transport skill configuration
 */
export interface PortalTransportConfig extends SkillConfig {
  /** WebSocket path (default: '/uamp') */
  path?: string;
  /** Portal URL for outgoing connections */
  portalUrl?: string;
  /**
   * Workspace `terminal` node support. When enabled, this transport
   * unwraps `extension.message { namespace: 'workspace.terminal' }`
   * envelopes received from the portal and hands the inner payload to a
   * `TerminalRouter`, which bridges to the host's PTY surface (Tauri in
   * v1; everything else returns `not_supported`).
   *
   * Pass an instance to share one router across multiple transports, or
   * `true` to construct a default one. Defaults to `false` (no terminal
   * handling — daemons that don't host the namespace simply drop these
   * envelopes; the portal falls back to its `peer_offline` close
   * reason when no `ready` arrives).
   */
  terminal?: boolean | TerminalRouter;
}

/**
 * Portal Transport Skill
 * 
 * Bidirectional UAMP over WebSocket:
 * - Expose local agent to portal/Elaisium
 * - Connect to remote agents via portal
 */
export class PortalTransportSkill extends Skill {
  private agent: IAgent | null = null;
  private portalUrl?: string;
  private portalAbort: AbortController | null = null;
  private agentId: string;
  private connectedClients: Set<WebSocket> = new Set();
  /** Resolve payment token wait when client sends payment.submit (keyed by WebSocket) */
  private paymentResolvers = new Map<WebSocket, (token: string) => void>();
  /**
   * In-flight AbortController per WebSocket. When the client sends
   * `response.cancel` (or the underlying ws closes/errors) we abort the
   * controller; downstream `agent.processUAMP` reads `context.signal` and
   * tears down its tool loop. Without this, parent abort does not propagate
   * into a delegate sub-agent's processUAMP loop and the delegate keeps
   * running for tens of seconds — see plans/surface_platform_tool_history_*.
   */
  private inflightAborts = new Map<WebSocket, AbortController>();
  /**
   * Terminal envelope router (if enabled via config). Lazily constructed
   * so daemons that don't need it don't pay the dynamic-import cost.
   */
  private terminal: TerminalRouter | null = null;

  constructor(config: PortalTransportConfig = {}) {
    super({ ...config, name: config.name || 'portal' });
    this.portalUrl = config.portalUrl;
    this.agentId = generateEventId();
    if (config.terminal === true) this.terminal = new TerminalRouter();
    else if (config.terminal && typeof config.terminal === 'object') {
      this.terminal = config.terminal;
    }
  }
  
  /**
   * Set the agent to delegate to
   */
  setAgent(agent: IAgent): void {
    this.agent = agent;
  }
  
  /**
   * Connect this agent to the portal over the REAL `/ws` session contract.
   *
   * Retargeted (M4): the old implementation sent `{type:'register', ...}`,
   * a protocol the platform's /ws never handled — the frames were dropped
   * on the floor and no turn ever arrived. This now delegates to
   * `runPortalBridge()` (src/portal/connect.ts), which performs `session.create`
   * with a per-agent token and serves `input.text` turns.
   *
   * The `workspace.terminal` router configured on this skill is handed to
   * the bridge, so the bridge-mode terminal that lived in the deleted
   * `handlePortalMessage` keeps working through the delegation instead of
   * being dropped with it.
   *
   * Resolves when the bridge disconnects (use `disconnectFromPortal()`).
   */
  async exposeAgent(): Promise<void> {
    if (!this.portalUrl || !this.agent) {
      throw new Error('Portal URL and agent must be set');
    }
    const { runPortalBridge } = await import('../../../portal/connect');
    this.portalAbort = new AbortController();
    await runPortalBridge(this.agent, {
      portalUrl: this.portalUrl,
      token: (this.config as { token?: string } | undefined)?.token,
      signal: this.portalAbort.signal,
      ...(this.terminal ? { terminal: this.terminal } : {}),
    });
  }

  /**
   * Disconnect from portal
   */
  disconnectFromPortal(): void {
    this.portalAbort?.abort();
    this.portalAbort = null;
    // Drop any live PTYs we owned via the bridge — the portal-side gateway
    // has already lost the browser, so leaving processes alive just leaks.
    if (this.terminal) void this.terminal.shutdown('portal_disconnect');
  }

  /**
   * Fast-path detection for `response.cancel` events buried inside a
   * wrapped `{ type: 'uamp', events: [...] }` envelope (some clients
   * batch a cancel together with other events).
   */
  private eventsContainCancel(events: ClientEvent[] | undefined): boolean {
    if (!events) return false;
    for (const evt of events) {
      if ((evt as { type?: string }).type === 'response.cancel') return true;
    }
    return false;
  }

  /**
   * Set the agent's `context.signal` so `processUAMP` and downstream
   * tool/LLM calls observe the abort. Stores the previous signal so we
   * can restore (best-effort) on completion.
   */
  private setAgentSignal(signal: AbortSignal): void {
    if (!this.agent) return;
    const ctx = (this.agent as IAgent & { context?: Context }).context;
    if (!ctx) return;
    (ctx as Context & { signal?: AbortSignal }).signal = signal;
  }

  private clearAgentSignal(signal: AbortSignal): void {
    if (!this.agent) return;
    const ctx = (this.agent as IAgent & { context?: Context }).context;
    if (!ctx) return;
    const cur = (ctx as Context & { signal?: AbortSignal }).signal;
    if (cur === signal) {
      delete (ctx as Context & { signal?: AbortSignal }).signal;
    }
  }

  /**
   * If the agent context has a `_loadChatHistory(chatId)` hook installed by
   * the portal-side bridge AND the incoming events carry an `X-Chat-Id`,
   * load history and prime `_initial_conversation`. No-op otherwise.
   *
   * The hook is portal-side because webagents has no DB / chat service.
   */
  private async maybeSeedInitialConversation(events: ClientEvent[]): Promise<void> {
    if (!this.agent) return;
    const ctx = (this.agent as IAgent & { context?: Context }).context;
    if (!ctx) return;
    const loader = ctx.get<(chatId: string) => Promise<unknown[]>>('_loadChatHistory');
    if (typeof loader !== 'function') return;

    let chatId: string | undefined;
    for (const evt of events) {
      if ((evt as { type?: string }).type === 'session.create') {
        const sess = (evt as { session?: { extensions?: Record<string, unknown> } }).session;
        const ext = sess?.extensions;
        const cid = ext?.['X-Chat-Id'] ?? ext?.['x-chat-id'];
        if (typeof cid === 'string') chatId = cid;
      }
    }
    if (!chatId) return;
    try {
      const initial = await loader(chatId);
      if (Array.isArray(initial) && initial.length > 0) {
        // Seed as `_history_conversation`, NOT `_initial_conversation`:
        // the new user turn arrives via `input.text` events on the same
        // request and `processUAMP` will append it. Setting the
        // `_initial_conversation` key here would silently drop the new
        // turn (it short-circuits events-based conversation building).
        ctx.set('_history_conversation', initial);
        if (typeof process !== 'undefined' && process.env?.LOG_LOOP_DEBUG === '1') {
          console.log(`[loop-debug] portal-transport seeded _history_conversation chatId=${chatId} msgs=${initial.length}`);
        }
      }
    } catch (err) {
      console.warn(`[portal-transport] _loadChatHistory failed (non-fatal) chatId=${chatId}: ${(err as Error).message}`);
    }
  }
  
  /**
   * REMOVED (M4): `callRemoteAgent` rode the dead `register`/`uamp_response`
   * protocol over the outbound socket, which the platform never spoke.
   * Agent-to-agent calls go through the platform's HTTP surface
   * (`POST /api/agents/{id}/chat/completions` or NLI/delegation) instead.
   */
  async *callRemoteAgent(
    _agentName: string,
    _events: ClientEvent[]
  ): AsyncGenerator<ServerEvent, void, unknown> {
    throw new Error(
      'callRemoteAgent was removed: the portal never spoke this protocol. ' +
      "Dial the platform's HTTP surface (POST /api/agents/{id}/chat/completions) instead.",
    );
  }

  /**
   * Handle incoming WebSocket connection (server mode)
   */
  @websocket({ path: '/uamp' })
  handleConnection(ws: WebSocket, context: Context): void {
    this.connectedClients.add(ws);

    ws.onmessage = async (event) => {
      try {
        const data = typeof event.data === 'string' ? event.data : await (event.data as Blob).text();
        const msg = JSON.parse(data) as PortalMessage & { type?: string; payment?: { token?: string }; events?: ClientEvent[] };

        // Workspace terminal envelopes (server mode). Same dispatch as
        // handlePortalMessage but writes back through the per-connection
        // ws — used by hosts that accept inbound portal connections
        // (rare but supported).
        if (this.terminal && TerminalRouter.isFor(msg as { type?: string; namespace?: string })) {
          const env = msg as {
            payload?: unknown;
            extension_version?: number;
          };
          if (!env.payload || typeof env.payload !== 'object') return;
          const send = (payload: TerminalOut) => {
            try {
              const out = createExtensionMessage(TERMINAL_NS, payload, { version: TERMINAL_VER });
              ws.send(JSON.stringify(out));
            } catch {
              /* ws may be closing */
            }
          };
          try {
            await this.terminal.handlePayload(env.payload as TerminalIn, send, {
              extension_version: env.extension_version ?? 1,
            });
          } catch (err) {
            console.warn('[portal-transport] terminal handler error:', (err as Error).message);
          }
          return;
        }

        // Client sent payment.submit (standalone UAMP event or wrapper)
        if ((msg.type as string) === 'payment.submit' && this.paymentResolvers.has(ws)) {
          const token = msg.payment?.token ?? '';
          this.paymentResolvers.get(ws)!(token);
          this.paymentResolvers.delete(ws);
          return;
        }

        // Standalone response.cancel — abort the in-flight processUAMP loop
        // for this ws so a parent-initiated abort tears down sub-agent work.
        if ((msg.type as string) === 'response.cancel' || this.eventsContainCancel(msg.events)) {
          const ac = this.inflightAborts.get(ws);
          if (ac) {
            if (typeof process !== 'undefined' && process.env?.LOG_LOOP_DEBUG === '1') {
              console.log(`[loop-debug] portal-transport handleConnection: response.cancel → aborting in-flight processUAMP`);
            }
            ac.abort();
            this.inflightAborts.delete(ws);
            try {
              ws.send(serializeEvent(createResponseCancelledEvent(`resp_${Date.now().toString(36)}`)));
            } catch { /* ws may already be closed */ }
          }
          if ((msg.type as string) === 'response.cancel') return;
        }

        if (msg.type === 'uamp' && this.agent) {
          const uampEvents = msg.events ?? [];

          // A2A history preload (no-op if portal-side bridge hasn't wired
          // _loadChatHistory). See handlePortalMessage for the rationale.
          await this.maybeSeedInitialConversation(uampEvents);

          // Extract payment token from session.create extensions (sent by
          // NLI/UAMPClient callers) and merge into the agent context so
          // downstream skills (LLM proxy, payments) can access it.
          for (const evt of uampEvents) {
            if (evt.type === 'session.create') {
              const ext = (evt as any).session?.extensions;
              const token = ext?.['X-Payment-Token'] ?? ext?.['x-payment-token'];
              if (token && (this.agent as any).context) {
                const ctx = (this.agent as any).context;
                ctx.set('payment_token', token);
                ctx.payment = { ...ctx.payment, token };

                // Extract userId from JWT sub claim so downstream skills
                // (StoreMediaSkill, etc.) have auth.user_id available
                try {
                  const parts = token.split('.');
                  if (parts.length >= 2) {
                    let decoded: string;
                    if (typeof Buffer !== 'undefined') {
                      decoded = Buffer.from(parts[1], 'base64url').toString();
                    } else {
                      decoded = atob(parts[1].replace(/-/g, '+').replace(/_/g, '/'));
                    }
                    const payload = JSON.parse(decoded);
                    if (payload.sub) {
                      ctx.auth = { ...ctx.auth, user_id: payload.sub, authenticated: true };
                    }
                  }
                } catch (e) {
                  console.warn('[portal-transport] JWT sub extraction failed:', (e as Error).message);
                }
              }
            }
          }

          let retries = 0;
          const maxPaymentRetries = 1;

          while (true) {
            try {
              let paymentWasRequired = retries > 0;
              const abortController = new AbortController();
              this.inflightAborts.set(ws, abortController);
              this.setAgentSignal(abortController.signal);
              try {
                for await (const serverEvent of this.agent.processUAMP(uampEvents)) {
                  ws.send(serializeEvent(serverEvent));
                  if (abortController.signal.aborted) break;
                }
              } finally {
                if (this.inflightAborts.get(ws) === abortController) {
                  this.inflightAborts.delete(ws);
                }
                this.clearAgentSignal(abortController.signal);
              }
              if (paymentWasRequired) {
                ws.send(serializeEvent(createPaymentAcceptedEvent(`pay-${generateEventId()}`)));
              }
              break;
            } catch (err) {
              if (err instanceof PaymentRequiredError && retries < maxPaymentRetries) {
                retries++;
                const requirements = {
                  amount: '0',
                  currency: 'USD',
                  schemes: [{ scheme: 'token' as const }],
                  reason: 'agent_access',
                };
                if (Array.isArray(err.accepts) && err.accepts.length > 0) {
                  const first = err.accepts[0] as { amount?: string; currency?: string; scheme?: string };
                  requirements.amount = first.amount ?? requirements.amount;
                  requirements.currency = first.currency ?? requirements.currency;
                }
                ws.send(serializeEvent(createPaymentRequiredEvent(requirements)));

                const token = await new Promise<string>((resolve, reject) => {
                  this.paymentResolvers.set(ws, resolve);
                  setTimeout(() => {
                    if (this.paymentResolvers.has(ws)) {
                      this.paymentResolvers.delete(ws);
                      reject(new Error('Payment token not received in time'));
                    }
                  }, 60_000);
                });

                context.set('payment_token', token);
                continue;
              }
              throw err;
            }
          }
        } else if (msg.type === 'discover') {
          // Return agent capabilities
          if (this.agent) {
            const response: PortalAgentsMessage = {
              type: 'agents',
              agents: [{
                name: this.agent.name,
                id: this.agentId,
                capabilities: this.agent.getCapabilities(),
              }],
            };
            ws.send(JSON.stringify(response));
          }
        }
      } catch (error) {
        console.error('Error handling WebSocket message:', error);
        ws.send(JSON.stringify({
          type: 'error',
          message: (error as Error).message,
        }));
      }
    };

    ws.onclose = () => {
      if (this.paymentResolvers.has(ws)) {
        this.paymentResolvers.get(ws)!('');
        this.paymentResolvers.delete(ws);
      }
      const ac = this.inflightAborts.get(ws);
      if (ac) {
        ac.abort();
        this.inflightAborts.delete(ws);
      }
      this.connectedClients.delete(ws);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      const ac = this.inflightAborts.get(ws);
      if (ac) {
        ac.abort();
        this.inflightAborts.delete(ws);
      }
      this.connectedClients.delete(ws);
    };
  }
  
  /**
   * Broadcast event to all connected clients
   */
  broadcast(event: ServerEvent): void {
    const data = serializeEvent(event);
    for (const client of this.connectedClients) {
      if (client.readyState === 1 /* WebSocket.OPEN */) {
        client.send(data);
      }
    }
  }
  
  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.disconnectFromPortal();
    for (const client of this.connectedClients) {
      client.close();
    }
    this.connectedClients.clear();
  }
}
