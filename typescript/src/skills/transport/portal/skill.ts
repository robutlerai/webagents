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
} from '../../../uamp/events';
import type { Capabilities } from '../../../uamp/types';
import { PaymentRequiredError } from '../../payments/x402';

/**
 * Portal message types
 */
interface PortalRegisterMessage {
  type: 'register';
  agentName: string;
  agentId: string;
  capabilities: Capabilities;
}

interface PortalUnregisterMessage {
  type: 'unregister';
  agentId: string;
}

interface PortalUAMPMessage {
  type: 'uamp';
  events: ClientEvent[];
  requestId?: string;
}

interface PortalDiscoverMessage {
  type: 'discover';
  query?: string;
}

type PortalMessage = PortalRegisterMessage | PortalUnregisterMessage | PortalUAMPMessage | PortalDiscoverMessage;

interface PortalResponseMessage {
  type: 'uamp_response';
  event: ServerEvent;
  requestId?: string;
}

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
  private portalConnection: WebSocket | null = null;
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
  /** In-flight controller for portal-bridge mode (no per-ws keying needed). */
  private portalInflightAbort: AbortController | null = null;
  
  constructor(config: PortalTransportConfig = {}) {
    super({ ...config, name: config.name || 'portal' });
    this.portalUrl = config.portalUrl;
    this.agentId = generateEventId();
  }
  
  /**
   * Set the agent to delegate to
   */
  setAgent(agent: IAgent): void {
    this.agent = agent;
  }
  
  /**
   * Connect to portal and register agent
   */
  async exposeAgent(): Promise<void> {
    if (!this.portalUrl || !this.agent) {
      throw new Error('Portal URL and agent must be set');
    }
    
    return new Promise((resolve, reject) => {
      this.portalConnection = new WebSocket(this.portalUrl!);
      
      this.portalConnection.onopen = () => {
        const registerMsg: PortalRegisterMessage = {
          type: 'register',
          agentName: this.agent!.name,
          agentId: this.agentId,
          capabilities: this.agent!.getCapabilities(),
        };
        this.portalConnection!.send(JSON.stringify(registerMsg));
        resolve();
      };
      
      this.portalConnection.onerror = (error) => {
        reject(new Error(`Portal connection failed: ${error}`));
      };
      
      this.portalConnection.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data as string) as PortalMessage;
          await this.handlePortalMessage(msg);
        } catch (error) {
          console.error('Error handling portal message:', error);
        }
      };
      
      this.portalConnection.onclose = () => {
        this.portalConnection = null;
      };
    });
  }
  
  /**
   * Disconnect from portal
   */
  disconnectFromPortal(): void {
    if (this.portalConnection) {
      const unregisterMsg: PortalUnregisterMessage = {
        type: 'unregister',
        agentId: this.agentId,
      };
      this.portalConnection.send(JSON.stringify(unregisterMsg));
      this.portalConnection.close();
      this.portalConnection = null;
    }
  }
  
  /**
   * Handle incoming portal message
   */
  private async handlePortalMessage(msg: PortalMessage): Promise<void> {
    if (!this.agent) return;

    // Standalone response.cancel — abort the currently running portal-mode
    // processUAMP loop (if any) so a parent-initiated abort propagates.
    if ((msg as { type?: string }).type === 'response.cancel' || this.eventsContainCancel((msg as PortalUAMPMessage).events)) {
      if (this.portalInflightAbort) {
        if (typeof process !== 'undefined' && process.env?.LOG_LOOP_DEBUG === '1') {
          console.log(`[loop-debug] portal-transport handlePortalMessage: response.cancel → aborting in-flight processUAMP`);
        }
        this.portalInflightAbort.abort();
        this.portalInflightAbort = null;
      }
      if ((msg as { type?: string }).type === 'response.cancel') return;
    }

    if (msg.type === 'uamp') {
      // A2A parity with U2A: if the agent context has a `_loadChatHistory`
      // hook (wired by the portal-side bridge) and the events carry a chat
      // id, preload prior conversation so the delegate sees previously
      // executed text_editor/bash/delegate turns. Without this, the delegate
      // starts each turn with only the new prompt and the LLM redoes work
      // it already did — see plans/surface_platform_tool_history_3596ddbe.
      await this.maybeSeedInitialConversation(msg.events);

      const abortController = new AbortController();
      this.portalInflightAbort = abortController;
      this.setAgentSignal(abortController.signal);

      try {
        for await (const event of this.agent.processUAMP(msg.events)) {
          const response: PortalResponseMessage = {
            type: 'uamp_response',
            event,
            requestId: msg.requestId,
          };
          this.portalConnection?.send(JSON.stringify(response));
          if (abortController.signal.aborted) break;
        }
      } finally {
        if (this.portalInflightAbort === abortController) {
          this.portalInflightAbort = null;
        }
        this.clearAgentSignal(abortController.signal);
      }
    }
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
   * Call a remote agent via portal
   */
  async *callRemoteAgent(
    agentName: string,
    events: ClientEvent[]
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const ws = this.portalConnection;
    if (!ws) {
      throw new Error('Not connected to portal');
    }
    
    const requestId = generateEventId();
    
    // Send UAMP events to remote agent
    const msg: PortalUAMPMessage & { targetAgent: string } = {
      type: 'uamp',
      events,
      requestId,
      targetAgent: agentName,
    };
    ws.send(JSON.stringify(msg));
    
    // Create a promise-based event queue
    const eventQueue: ServerEvent[] = [];
    let resolveNext: ((event: ServerEvent | null) => void) | null = null;
    let done = false;
    
    const handler = (wsEvent: MessageEvent) => {
      try {
        const response = JSON.parse(wsEvent.data as string);
        if (response.type === 'uamp_response' && response.requestId === requestId) {
          const event = response.event as ServerEvent;
          const resolver = resolveNext;
          if (resolver) {
            resolveNext = null;
            resolver(event);
          } else {
            eventQueue.push(event);
          }
          
          if (event.type === 'response.done' || event.type === 'response.error') {
            done = true;
            const doneResolver = resolveNext;
            if (doneResolver) {
              resolveNext = null;
              doneResolver(null);
            }
          }
        }
      } catch {
        // Ignore parse errors
      }
    };
    
    ws.addEventListener('message', handler);
    
    try {
      while (!done) {
        const event = eventQueue.shift() || await new Promise<ServerEvent | null>(resolve => {
          resolveNext = resolve;
        });
        
        if (event) {
          yield event;
          if (event.type === 'response.done' || event.type === 'response.error') {
            break;
          }
        } else {
          break;
        }
      }
    } finally {
      ws.removeEventListener('message', handler);
    }
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
