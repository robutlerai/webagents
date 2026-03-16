/**
 * Portal Transport Skill
 * 
 * Native UAMP over WebSocket for Elaisium and agent mesh.
 * No protocol conversion needed - UAMP events flow directly.
 */

import { Skill } from '../../../core/skill.js';
import { websocket } from '../../../core/decorators.js';
import type { SkillConfig, Context, IAgent } from '../../../core/types.js';
import type { ClientEvent, ServerEvent } from '../../../uamp/events.js';
import {
  serializeEvent,
  generateEventId,
  createPaymentRequiredEvent,
  createPaymentAcceptedEvent,
} from '../../../uamp/events.js';
import type { Capabilities } from '../../../uamp/types.js';
import { PaymentRequiredError } from '../../payments/x402.js';

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
    if (msg.type === 'uamp' && this.agent) {
      // Process UAMP events from portal
      for await (const event of this.agent.processUAMP(msg.events)) {
        const response: PortalResponseMessage = {
          type: 'uamp_response',
          event,
          requestId: msg.requestId,
        };
        this.portalConnection?.send(JSON.stringify(response));
      }
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

        if (msg.type === 'uamp' && this.agent) {
          const uampEvents = msg.events ?? [];

          // Extract payment token from session.create extensions (sent by
          // NLI/UAMPClient callers) and merge into the agent context so
          // downstream skills (LLM proxy, payments) can access it.
          for (const evt of uampEvents) {
            if (evt.type === 'session.create') {
              const ext = (evt as any).session?.extensions;
              const token = ext?.['X-Payment-Token'] ?? ext?.['x-payment-token'];
              if (token && (this.agent as any).context) {
                (this.agent as any).context.set('payment_token', token);
                (this.agent as any).context.payment = {
                  ...(this.agent as any).context.payment,
                  token,
                };
              }
            }
          }

          let retries = 0;
          const maxPaymentRetries = 1;

          while (true) {
            try {
              let paymentWasRequired = retries > 0;
              for await (const serverEvent of this.agent.processUAMP(uampEvents)) {
                ws.send(serializeEvent(serverEvent));
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
      this.connectedClients.delete(ws);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.connectedClients.delete(ws);
    };
  }
  
  /**
   * Broadcast event to all connected clients
   */
  broadcast(event: ServerEvent): void {
    const data = serializeEvent(event);
    for (const client of this.connectedClients) {
      if (client.readyState === WebSocket.OPEN) {
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
