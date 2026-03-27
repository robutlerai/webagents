/**
 * UAMP WebSocket Transport Skill
 *
 * Native UAMP (Universal Agentic Message Protocol) WebSocket transport.
 * Speaks raw UAMP events directly over a WebSocket connection without
 * any wrapper protocol. Provides:
 * - Session lifecycle (create, update, close)
 * - Multimodal input (text, audio, image, video, file)
 * - Streaming response generation with tool calls
 * - Payment negotiation (payment.required → payment.submit → retry)
 * - Typing indicators and keepalive
 *
 * Protocol reference: https://uamp.dev/
 */

import { Skill } from '../../../core/skill';
import { websocket } from '../../../core/decorators';
import type { Context, IAgent } from '../../../core/types';
import type {
  ClientEvent,
  SessionCreateEvent,
  ResponseDelta,
} from '../../../uamp/events';
import {
  generateEventId,
  createBaseEvent,
  serializeEvent,
  createResponseErrorEvent,
  createPaymentRequiredEvent,
  createPaymentAcceptedEvent,
} from '../../../uamp/events';
import type {
  SessionConfig,
  ContentItem,
  ImageContent,
  AudioContent,
  VideoContent,
  FileContent,
} from '../../../uamp/types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UAMPTransportConfig {
  name?: string;
  enabled?: boolean;
  /** WebSocket path (default: '/uamp') */
  path?: string;
  /** Payment negotiation timeout (ms, default: 60_000) */
  paymentTimeout?: number;
}

interface UAMPSession {
  id: string;
  modalities: string[];
  instructions: string;
  createdAt: number;
  status: 'active' | 'closed';
  paymentBalance: string | null;
  paymentCurrency: string;
  paymentTokenExpiresAt: number | null;
  paymentToken: string | null;
  conversation: Array<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSessionId(): string {
  return `sess_${generateEventId().replace(/-/g, '').slice(0, 12)}`;
}

function makeResponseId(): string {
  return `resp_${generateEventId().replace(/-/g, '').slice(0, 12)}`;
}

function makePaymentId(): string {
  return `pay_${generateEventId().replace(/-/g, '').slice(0, 12)}`;
}

function nowMs(): number {
  return Date.now();
}

// ---------------------------------------------------------------------------
// UAMPTransportSkill
// ---------------------------------------------------------------------------

/**
 * Native UAMP WebSocket transport.
 *
 * Implements the UAMP event protocol directly over a single WebSocket
 * connection. Each connection holds exactly one session.
 *
 * @example
 * ```typescript
 * const agent = new BaseAgent({
 *   name: 'my-agent',
 *   skills: [new UAMPTransportSkill()],
 * });
 * // Clients connect via WebSocket to /agents/my-agent/uamp
 * ```
 */
export class UAMPTransportSkill extends Skill {
  private agent: IAgent | null = null;
  private sessions = new Map<string, UAMPSession>();
  private paymentResolvers = new Map<string, (token: string) => void>();
  private paymentTimeout: number;

  constructor(config: UAMPTransportConfig = {}) {
    super({ ...config, name: config.name || 'uamp-transport' });
    this.paymentTimeout = config.paymentTimeout ?? 60_000;
  }

  setAgent(agent: IAgent): void {
    this.agent = agent;
  }

  // =========================================================================
  // WebSocket endpoint
  // =========================================================================

  @websocket({ path: '/uamp' })
  handleConnection(ws: WebSocket, _context: Context): void {
    let session: UAMPSession | null = null;

    ws.onmessage = async (ev) => {
      try {
        const raw =
          typeof ev.data === 'string'
            ? ev.data
            : typeof Blob !== 'undefined' && ev.data instanceof Blob
              ? await ev.data.text()
              : String(ev.data);

        const message: Record<string, unknown> = JSON.parse(raw);
        const eventType = (message.type as string) || '';

        // ------------------------------------------------------------------
        // Session creation
        // ------------------------------------------------------------------
        if (eventType === 'session.create') {
          session = this._createSession(message);
          if (session.paymentToken && (this.agent as any)?.context) {
            (this.agent as any).context.set('payment_token', session.paymentToken);
            (this.agent as any).context.payment = {
              ...(this.agent as any).context.payment,
              token: session.paymentToken,
            };
          }
          await this._sendSessionCreated(ws, session);
          await this._sendCapabilities(ws, session);
          return;
        }

        // Require session for all other events
        if (!session) {
          this._sendError(ws, 'session_required', 'Session must be created first');
          return;
        }

        // Resolve pending wait-for-event (e.g. payment.submit)
        if (this.paymentResolvers.has(session.id) && eventType === 'payment.submit') {
          const token = ((message.payment as Record<string, unknown>)?.token as string) ?? '';
          const resolver = this.paymentResolvers.get(session.id);
          if (resolver) {
            this.paymentResolvers.delete(session.id);
            resolver(token);
          }
          // Also update session token
          session.paymentToken = token;
          session.paymentTokenExpiresAt = nowMs() + 3600_000;
          const paymentId = makePaymentId();
          const acceptedEvent = {
            ...createBaseEvent('payment.accepted'),
            type: 'payment.accepted',
            payment_id: paymentId,
            balance_remaining: session.paymentBalance ?? '0',
            expires_at: session.paymentTokenExpiresAt,
          };
          ws.send(JSON.stringify(acceptedEvent));
          return;
        }

        // ------------------------------------------------------------------
        // Event routing
        // ------------------------------------------------------------------
        switch (eventType) {
          case 'session.update':
            this._handleSessionUpdate(ws, session, message);
            break;
          case 'capabilities.query':
            await this._sendCapabilities(ws, session);
            break;
          case 'input.text':
            this._handleInputText(ws, session, message);
            break;
          case 'input.audio':
            this._handleInputAudio(ws, session, message);
            break;
          case 'input.image':
            this._handleInputImage(session, message);
            break;
          case 'input.video':
            this._handleInputVideo(session, message);
            break;
          case 'input.file':
            this._handleInputFile(session, message);
            break;
          case 'input.typing':
            // Typing indicators — no-op for server-side processing
            break;
          case 'response.create':
            this._spawnResponse(ws, session);
            break;
          case 'response.cancel':
            // Best-effort cancellation — not easily interruptable
            break;
          case 'tool.result':
            this._handleToolResult(ws, session, message);
            break;
          case 'ping':
            ws.send(JSON.stringify({ ...createBaseEvent('pong'), type: 'pong' }));
            break;
          default:
            this._sendError(ws, 'unknown_event', `Unknown event type: ${eventType}`);
        }
      } catch (err) {
        console.error('[UAMPTransport] message handler error:', err);
        this._sendError(ws, 'internal_error', (err as Error).message);
      }
    };

    ws.onclose = () => {
      if (session) {
        const resolver = this.paymentResolvers.get(session.id);
        if (resolver) {
          this.paymentResolvers.delete(session.id);
          resolver('');
        }
        session.status = 'closed';
        this.sessions.delete(session.id);
      }
    };

    ws.onerror = (error) => {
      console.error('[UAMPTransport] WebSocket error:', error);
    };
  }

  // =========================================================================
  // Session management
  // =========================================================================

  private _createSession(message: Record<string, unknown>): UAMPSession {
    const sessionConfig = (message.session as Record<string, unknown>) ?? {};
    const extensions = (sessionConfig.extensions as Record<string, unknown>) ?? {};
    const paymentToken =
      (message.payment_token as string) ??
      (sessionConfig.payment_token as string) ??
      (extensions['X-Payment-Token'] as string) ??
      null;

    const session: UAMPSession = {
      id: makeSessionId(),
      modalities: (sessionConfig.modalities as string[]) ?? ['text'],
      instructions: (sessionConfig.instructions as string) ?? '',
      createdAt: nowMs(),
      status: 'active',
      paymentBalance: null,
      paymentCurrency: 'USD',
      paymentTokenExpiresAt: null,
      paymentToken: paymentToken,
      conversation: [],
    };
    this.sessions.set(session.id, session);
    return session;
  }

  private async _sendSessionCreated(ws: WebSocket, session: UAMPSession): Promise<void> {
    const event = {
      ...createBaseEvent('session.created'),
      type: 'session.created',
      uamp_version: '1.0',
      session: {
        id: session.id,
        created_at: session.createdAt,
        status: session.status,
        config: {
          modalities: session.modalities,
          instructions: session.instructions,
        },
      },
    };
    ws.send(JSON.stringify(event));
  }

  private async _sendCapabilities(ws: WebSocket, session: UAMPSession): Promise<void> {
    const agentCaps = this.agent?.getCapabilities();
    const event = {
      ...createBaseEvent('capabilities'),
      type: 'capabilities',
      capabilities: {
        id: agentCaps?.id ?? 'unknown',
        provider: agentCaps?.provider ?? 'webagents',
        modalities: session.modalities,
        supports_streaming: agentCaps?.supports_streaming ?? true,
        supports_thinking: agentCaps?.supports_thinking ?? false,
        supports_caching: agentCaps?.supports_caching ?? false,
        tools: agentCaps?.tools,
      },
    };
    ws.send(JSON.stringify(event));
  }

  private _handleSessionUpdate(
    ws: WebSocket,
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const sessionConfig = (message.session as Record<string, unknown>) ?? {};
    const paymentToken =
      (message.payment_token as string) ??
      (sessionConfig.payment_token as string);

    if (sessionConfig.modalities) {
      session.modalities = sessionConfig.modalities as string[];
    }
    if (sessionConfig.instructions) {
      session.instructions = sessionConfig.instructions as string;
    }
    if (paymentToken) {
      session.paymentToken = paymentToken;
    }

    const event = {
      ...createBaseEvent('session.updated'),
      type: 'session.updated',
      session: {
        id: session.id,
        created_at: session.createdAt,
        status: session.status,
        config: {
          modalities: session.modalities,
          instructions: session.instructions,
        },
      },
    };
    ws.send(JSON.stringify(event));
  }

  // =========================================================================
  // Input handlers
  // =========================================================================

  private _handleInputText(
    _ws: WebSocket,
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const text = (message.text as string) ?? '';
    const role = (message.role as string) ?? 'user';
    const paymentToken = message.payment_token as string | undefined;

    session.conversation.push({
      role,
      content: null,
      content_items: [{ type: 'text', text } as ContentItem],
    });

    if (paymentToken) {
      session.paymentToken = paymentToken;
    }
  }

  private _handleInputAudio(
    _ws: WebSocket,
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const audio = message.audio ?? '';
    const format = (message.format as string) ?? 'webm';
    if (!audio) return;

    session.conversation.push({
      role: 'user',
      content: null,
      content_items: [{
        type: 'audio',
        audio,
        format,
      } as AudioContent],
    });
  }

  private _handleInputImage(
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const image = message.image ?? '';
    if (!image) return;

    session.conversation.push({
      role: 'user',
      content: null,
      content_items: [{
        type: 'image',
        image,
        format: message.format as string | undefined,
        detail: (message.detail as 'low' | 'high' | 'auto') ?? undefined,
      } as ImageContent],
    });
  }

  private _handleInputVideo(
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const video = message.video ?? '';
    if (!video) return;

    session.conversation.push({
      role: 'user',
      content: null,
      content_items: [{
        type: 'video',
        video,
        format: message.format as string | undefined,
      } as VideoContent],
    });
  }

  private _handleInputFile(
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const file = message.file ?? '';
    if (!file) return;

    session.conversation.push({
      role: 'user',
      content: null,
      content_items: [{
        type: 'file',
        file,
        filename: (message.filename as string) ?? 'document',
        mime_type: (message.mime_type as string) ?? 'application/octet-stream',
      } as FileContent],
    });
  }

  private _handleToolResult(
    ws: WebSocket,
    session: UAMPSession,
    message: Record<string, unknown>,
  ): void {
    const callId = (message.call_id as string) ?? '';
    const result = (message.result as string) ?? '';

    session.conversation.push({
      role: 'tool',
      tool_call_id: callId,
      content: result,
    });

    this._spawnResponse(ws, session);
  }

  // =========================================================================
  // Response generation
  // =========================================================================

  private _spawnResponse(
    ws: WebSocket,
    session: UAMPSession,
    opts?: {
      sessionId?: string;
      messages?: Array<Record<string, unknown>>;
      paymentToken?: string;
    },
  ): void {
    // Fire-and-forget so WS message loop stays unblocked
    this._generateResponse(ws, session, opts).catch((err) => {
      console.error('[UAMPTransport] Response generation failed:', err);
    });
  }

  private async _generateResponse(
    ws: WebSocket,
    session: UAMPSession,
    opts?: {
      sessionId?: string;
      messages?: Array<Record<string, unknown>>;
      paymentToken?: string;
    },
  ): Promise<void> {
    if (!this.agent) {
      this._sendError(ws, 'no_agent', 'No agent attached to transport');
      return;
    }

    const responseId = makeResponseId();
    const sessionId = opts?.sessionId;

    // Emit response.created
    ws.send(
      JSON.stringify({
        ...createBaseEvent('response.created'),
        type: 'response.created',
        response_id: responseId,
      }),
    );

    // Build UAMP client events from session conversation
    const clientEvents = this._buildClientEvents(session, opts?.paymentToken);

    let fullText = '';
    let paymentNegotiated = false;
    let retries = 0;
    const maxPaymentRetries = 1;

    while (true) {
      try {
        for await (const serverEvent of this.agent.processUAMP(clientEvents)) {
          // Stream events to client, enriching with response_id / session_id
          const enriched = { ...serverEvent } as Record<string, unknown>;
          if (!enriched.response_id) enriched.response_id = responseId;
          if (sessionId && !enriched.session_id) enriched.session_id = sessionId;

          if (serverEvent.type === 'response.delta') {
            const delta = (serverEvent as unknown as { delta: ResponseDelta }).delta;
            if (delta.text) fullText += delta.text;
          }

          ws.send(JSON.stringify(enriched));

          if (serverEvent.type === 'response.done' || serverEvent.type === 'response.error') {
            break;
          }
        }
        break; // success
      } catch (err) {
        // Payment negotiation: ask client for token and retry
        const isPaymentRequired = this._isPaymentError(err);
        if (isPaymentRequired && retries < maxPaymentRetries) {
          retries++;
          paymentNegotiated = true;

          const requirements = {
            amount: '0.01',
            currency: session.paymentCurrency,
            schemes: [{ scheme: 'token' as const }],
            reason: 'llm_usage',
          };
          ws.send(
            serializeEvent(createPaymentRequiredEvent(requirements, responseId)),
          );

          // Wait for payment.submit from client (resolved in onmessage handler)
          const token = await this._waitForPaymentToken(session.id);
          if (!token) {
            ws.send(
              JSON.stringify(
                createResponseErrorEvent(
                  'payment_timeout',
                  'Payment token not received in time',
                  responseId,
                ),
              ),
            );
            return;
          }
          session.paymentToken = token;
          continue;
        }

        // Non-retryable error
        ws.send(
          JSON.stringify(
            createResponseErrorEvent(
              'generation_error',
              (err as Error).message,
              responseId,
            ),
          ),
        );
        return;
      }
    }

    // Track conversation
    if (fullText) {
      session.conversation.push({ role: 'assistant', content: fullText });
    }

    if (paymentNegotiated) {
      ws.send(
        serializeEvent(
          createPaymentAcceptedEvent(
            makePaymentId(),
            session.paymentBalance ?? '0',
            session.paymentTokenExpiresAt ?? undefined,
          ),
        ),
      );
    }
  }

  // =========================================================================
  // Internal helpers
  // =========================================================================

  private _buildClientEvents(
    session: UAMPSession,
    paymentToken?: string,
  ): ClientEvent[] {
    const events: ClientEvent[] = [];

    const extensions: Record<string, unknown> = {};
    if (paymentToken || session.paymentToken) {
      extensions['X-Payment-Token'] = paymentToken || session.paymentToken;
    }

    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: session.modalities as SessionConfig['modalities'],
        instructions: session.instructions || undefined,
        ...(Object.keys(extensions).length > 0 ? { extensions } : {}),
      },
    } as SessionCreateEvent);

    for (const msg of session.conversation) {
      const role = msg.role as string;
      if (role === 'user' || role === 'system') {
        const items = msg.content_items as ContentItem[] | undefined;
        if (items) {
          for (const item of items) {
            if (item.type === 'text') {
              events.push({ type: 'input.text', event_id: generateEventId(), text: item.text, role } as ClientEvent);
            } else if (item.type === 'image') {
              events.push({ type: 'input.image', event_id: generateEventId(), image: (item as ImageContent).image } as ClientEvent);
            } else if (item.type === 'audio') {
              events.push({ type: 'input.audio', event_id: generateEventId(), audio: (item as AudioContent).audio, format: (item as AudioContent).format ?? 'webm' } as ClientEvent);
            } else if (item.type === 'video') {
              events.push({ type: 'input.video', event_id: generateEventId(), video: (item as VideoContent).video } as ClientEvent);
            } else if (item.type === 'file') {
              const f = item as FileContent;
              events.push({ type: 'input.file', event_id: generateEventId(), file: f.file, filename: f.filename, mime_type: f.mime_type } as ClientEvent);
            }
          }
        } else if (typeof msg.content === 'string') {
          events.push({ type: 'input.text', event_id: generateEventId(), text: msg.content, role } as ClientEvent);
        }
      } else if (role === 'tool') {
        // Tool results stay as-is in the conversation
      }
    }

    events.push({
      type: 'response.create',
      event_id: generateEventId(),
    } as ClientEvent);

    return events;
  }

  private _waitForPaymentToken(sessionId: string): Promise<string> {
    return new Promise<string>((resolve) => {
      this.paymentResolvers.set(sessionId, resolve);
      setTimeout(() => {
        if (this.paymentResolvers.has(sessionId)) {
          this.paymentResolvers.delete(sessionId);
          resolve('');
        }
      }, this.paymentTimeout);
    });
  }

  private _isPaymentError(err: unknown): boolean {
    if (!err || typeof err !== 'object') return false;
    const name = (err as Error).constructor?.name ?? '';
    if (name === 'PaymentRequiredError') return true;
    const message = ((err as Error).message ?? '').toLowerCase();
    return (
      message.includes('payment') &&
      (message.includes('required') || message.includes('insufficient'))
    );
  }

  private _sendError(ws: WebSocket, code: string, message: string): void {
    const event = {
      ...createBaseEvent('response.error'),
      type: 'response.error',
      error: { code, message },
    };
    ws.send(JSON.stringify(event));
  }

  // =========================================================================
  // Lifecycle
  // =========================================================================

  override async cleanup(): Promise<void> {
    for (const [_sessionId, resolver] of this.paymentResolvers) {
      resolver('');
    }
    this.paymentResolvers.clear();
    this.sessions.clear();
  }
}
