import {
  generateEventId,
  parseEvent,
  serializeEvent,
} from './events';

import type {
  SessionCreateConfig,
  SessionCreateEvent,
  InputTextEvent,
  ResponseCreateEvent,
  ResponseCancelEvent,
  PaymentSubmitEvent,
  ResponseDeltaEvent,
  ResponseDoneEvent,
  PaymentRequiredEvent,
  PaymentAcceptedEvent,
  ResponseErrorEvent,
  ResponseCancelledEvent,
  PaymentErrorEvent,
} from './events';

import type {
  ContentItem,
  ImageContent,
  AudioContent,
  VideoContent,
  FileContent,
  UsageStats,
  Capabilities,
} from './types';

interface WS {
  readyState: number;
  send(data: string): void;
  close(): void;
  addEventListener(event: string, handler: (...args: unknown[]) => void): void;
  removeEventListener(event: string, handler: (...args: unknown[]) => void): void;
}

const OPEN = 1;
const CLOSING = 2;
const CLOSED = 3;

export interface UAMPClientConfig {
  url: string;
  paymentToken?: string;
  signal?: AbortSignal;
  connectTimeout?: number;
  /** Per-message timeout during streaming (ms). Default 120000 (2 min). 0 to disable. */
  responseTimeout?: number;
  session?: Partial<SessionCreateConfig>;
  extensions?: Record<string, unknown>;
  /** Custom headers to send during WebSocket handshake (Node.js only) */
  headers?: Record<string, string>;
  /**
   * Client capabilities to announce in `session.create`. The remote agent uses
   * these to decide which capability-gated tools to register (e.g. `present`
   * and `read_content` are only registered when `supports_rich_display: true`).
   */
  clientCapabilities?: Partial<Capabilities>;
  /**
   * The in-band purchase hook (machine-purchase design section 6.1 and
   * 6.4, 2026-09-18). When a `payment.required` carries a scheme `mpp`
   * entry (the challenge, a purchase URL and the Terms notice) and a buyer
   * is configured, the client pays it over HTTP through the buyer and then
   * sends `payment.submit` with scheme `balance`, so the server re-reads
   * the agent's balance and resumes the run on this socket. A buyer that
   * refuses (policy, terms, method) hands the event to `paymentRequired`
   * listeners as before, so a token holder can still pay the token scheme
   * at index 0. `MppBuyer` from `skills/payments` satisfies this.
   *
   * An `mpp` entry WITHOUT a challenge is the platform's purchase pointer
   * (`UAMPInBandBuyer.purchaseAt`). After a pointer purchase the run resumes
   * the way this session pays: a session with no payment token sends
   * `payment.submit` scheme `balance`; a session that holds one hands the
   * event to the `paymentRequired` listeners with `purchased: true`, because
   * they own the token (a fresh one, or the same one re-submitted) and the
   * socket that sent the pointer accepts nothing else.
   */
  buyer?: UAMPInBandBuyer;
}

/** What the client needs from a buyer: pay this challenge at this URL, say whether it worked. */
export interface UAMPInBandBuyer {
  purchase(request: {
    url: string;
    challenge: string;
    terms?: { url?: string | null; version?: string | null } | null;
    /** One object per response: a buyer that counts purchases per call (`MppBuyer`, `maxPurchasesPerCall`) keys the count on it. */
    call?: object;
  }): Promise<{ ok: boolean; reason?: string; detail?: string; status?: number | null }>;
  /**
   * Act on a PURCHASE POINTER (2026-09-19): an `mpp` entry that names a
   * purchase URL and carries NO challenge, which is what a rail sends when it
   * has not verified who is asking (a token holder whose token ran dry on the
   * platform's `/llm` socket; the portal's lib/payments/purchase-pointer.ts).
   * The buyer asks the purchase URL itself, signed, is challenged there as
   * its own identity and pays under its policy. `from` is this socket's URL:
   * `MppBuyer` follows a pointer only when the purchase URL's host AND the
   * host that named it are on its realm allowlist, and only when its policy
   * sets a daily cap (`pointer_needs_daily_cap` otherwise), because a pointer
   * carries no secret and any peer can write one. Optional: a buyer without it never
   * receives an entry with no challenge, and the event reaches the
   * `paymentRequired` listeners exactly as before.
   */
  purchaseAt?(request: {
    url: string;
    from: string;
    call?: object;
  }): Promise<{ ok: boolean; reason?: string; detail?: string; status?: number | null }>;
  /**
   * The RFC 9421 headers that sign this socket's upgrade (2026-09-18). The
   * platform's socket door acts only on an upgrade carrying
   * `Signature-Input` and no payment token (`signedUpgradeEligible`,
   * lib/payments/machine-door-socket.ts); an unsigned one is proxied as
   * before and never meets an in-band `mpp` challenge, so without this the
   * `purchase` hook above only ever answered third-party peers. `MppBuyer`
   * returns an empty record for a host off its realm allowlist.
   */
  upgradeHeaders?(url: string): Promise<Record<string, string>>;
}

/**
 * A token is an explicit choice of how to pay, and the socket door passes
 * any upgrade that carries one (`X-Payment-Token`, `X-PAYMENT`, or the
 * Python NLI's `?payment_token=`) straight through, so such an upgrade is
 * never signed for the door.
 */
function carriesPaymentToken(config: UAMPClientConfig): boolean {
  if (config.paymentToken) return true;
  for (const name of Object.keys(config.headers ?? {})) {
    const lower = name.toLowerCase();
    if (lower === 'x-payment-token' || lower === 'x-payment') return true;
  }
  try {
    return new URL(config.url).searchParams.has('payment_token');
  } catch {
    return false;
  }
}

export interface UAMPClientEvents {
  delta: (text: string) => void;
  toolCall: (toolCall: { id: string; name: string; arguments: string }) => void;
  toolResult: (toolResult: { call_id: string; tool?: string; result?: string; command?: string; content_id?: string; path?: string; is_error?: boolean; content_items?: unknown[] }) => void;
  toolProgress: (progress: { call_id: string; text: string; replace?: boolean; media_type?: string; status?: string; progress_percent?: number; estimated_duration_ms?: number }) => void;
  file: (fileData: Record<string, unknown>) => void;
  thinking: (data: { content: string; stage?: string; redacted?: boolean; is_delta?: boolean }) => void;
  done: (response: { output: ContentItem[]; usage?: UsageStats; id: string; status: string; pre_executed_rounds?: import('./events').PreExecutedRound[] }) => void;
  error: (error: Error) => void;
  /** `purchased` is true only when the buyer has just bought through a purchase pointer on a session that pays by token: the listener's token submit is what resumes the run. */
  paymentRequired: (requirements: { amount: string; currency: string; schemes: Array<{ scheme: string; network?: string; challenge?: string; purchase_url?: string; terms?: { url: string; version: string } }>; reason?: string; purchased?: boolean }) => void;
  /** The buyer is paying an in-band `mpp` challenge over HTTP; `paymentAccepted` or `paymentRequired` (a refusal) follows. */
  paymentPurchasing: (data: { purchase_url: string; amount: string; currency: string; terms?: { url: string; version: string } }) => void;
  paymentAccepted: (data: { payment_id: string; balance_remaining?: string }) => void;
  cancelled: (data: { response_id: string; partial_output?: ContentItem[] }) => void;
}

export class UAMPClient {
  private ws: WS | null = null;
  private config: UAMPClientConfig;
  private listeners: Map<string, Set<Function>> = new Map();
  private connected: boolean = false;
  private responseTimer: ReturnType<typeof setTimeout> | null = null;
  /** One object per response asked for: the key a buyer counts this call's purchases on (`maxPurchasesPerCall`). */
  private call: object = {};

  constructor(config: UAMPClientConfig) {
    this.config = config;
  }

  on<K extends keyof UAMPClientEvents>(event: K, handler: UAMPClientEvents[K]): this {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return this;
  }

  off<K extends keyof UAMPClientEvents>(event: K, handler: UAMPClientEvents[K]): this {
    this.listeners.get(event)?.delete(handler);
    return this;
  }

  async connect(): Promise<void> {
    console.log(`[uamp-client] connect: url=${this.config.url} token=${this.config.paymentToken ? 'yes' : 'no'}`);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { default: WebSocket } = await import('ws' as any);
    console.log(`[uamp-client] ws module loaded`);
    const timeout = this.config.connectTimeout ?? 10000;

    // Sign the upgrade for the platform's socket door when a buyer can
    // (the `upgradeHeaders` doc above). A refusal the door writes before
    // the upgrade (401, 403) surfaces as a connect error, so the NLI's
    // `auto` transport falls back to HTTP exactly as for any failed dial.
    let handshakeHeaders = this.config.headers;
    if (this.config.buyer?.upgradeHeaders && !carriesPaymentToken(this.config)) {
      const signed = await this.config.buyer.upgradeHeaders(this.config.url);
      if (Object.keys(signed).length > 0) handshakeHeaders = { ...(this.config.headers ?? {}), ...signed };
    }

    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const wsOpts = handshakeHeaders ? { headers: handshakeHeaders } : undefined;
      const ws = new WebSocket(this.config.url, wsOpts) as unknown as WS;
      let timer: ReturnType<typeof setTimeout> | null = null;

      const cleanup = () => {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        this.connectResolve = null;
      };

      this.connectResolve = () => {
        cleanup();
        console.log(`[uamp-client] session.created received`);
        if (!settled) { settled = true; resolve(); }
      };

      timer = setTimeout(() => {
        cleanup();
        ws.close();
        if (!settled) { settled = true; reject(new Error(`Connection timeout after ${timeout}ms`)); }
      }, timeout);

      ws.addEventListener('message', ((...args: unknown[]) => {
        const raw = args[0];
        const data = typeof raw === 'string'
          ? raw
          : (raw as { data?: unknown })?.data != null
            ? String((raw as { data: unknown }).data)
            : String(raw);
        this.handleMessage(data);
      }) as (...args: unknown[]) => void);

      ws.addEventListener('close', ((...args: unknown[]) => {
        const code = typeof args[0] === 'number' ? args[0] : (args[0] as { code?: number })?.code;
        console.log(`[uamp-client] ws closed (code=${code ?? 'unknown'})`);
        this.connected = false;
        this.ws = null;
        if (!settled) { settled = true; reject(new Error(`WebSocket closed unexpectedly (code=${code})`)); }
      }) as (...args: unknown[]) => void);

      ws.addEventListener('error', ((...args: unknown[]) => {
        const err = args[0] instanceof Error ? args[0] : new Error('WebSocket error');
        console.error(`[uamp-client] ws error:`, err.message);
        cleanup();
        if (!settled) { settled = true; reject(err); }
        this.emit('error', err);
      }) as (...args: unknown[]) => void);

      ws.addEventListener('open', () => {
        this.ws = ws;
        this.connected = true;
        console.log(`[uamp-client] ws open, readyState=${ws.readyState}`);

        const sessionConfig: SessionCreateConfig = {
          modalities: this.config.session?.modalities ?? ['text'],
          ...this.config.session,
        };

        const extensions: Record<string, unknown> = {
          ...this.config.extensions,
        };
        if (this.config.paymentToken) {
          extensions['X-Payment-Token'] = this.config.paymentToken;
        }

        const sessionCreate: SessionCreateEvent = {
          type: 'session.create',
          event_id: generateEventId(),
          timestamp: Date.now(),
          uamp_version: '1.0',
          session: {
            ...sessionConfig,
            extensions: {
              ...sessionConfig.extensions,
              ...extensions,
            },
          },
          ...(this.config.clientCapabilities && {
            client_capabilities: this.config.clientCapabilities as Capabilities,
          }),
        };

        try {
          const payload = serializeEvent(sessionCreate);
          console.log(`[uamp-client] sending session.create (${payload.length} bytes)`);
          ws.send(payload);
          console.log(`[uamp-client] session.create sent, waiting for session.created…`);
        } catch (err) {
          console.error(`[uamp-client] session.create send FAILED:`, (err as Error).message);
          cleanup();
          if (!settled) { settled = true; reject(err as Error); }
          return;
        }
      });

      if (this.config.signal) {
        this.config.signal.addEventListener('abort', () => {
          this.cancel().catch(() => {});
          this.close();
        });
      }
    });
  }

  async sendInput(
    text: string,
    role: 'user' | 'system' = 'user',
    contentItems?: ContentItem[],
  ): Promise<void> {
    this.call = {};
    if (text) {
      this.send({
        type: 'input.text',
        event_id: generateEventId(),
        timestamp: Date.now(),
        text,
        role,
      } as InputTextEvent);
    }

    if (contentItems) {
      for (const item of contentItems) {
        if (item.type === 'image') {
          const img = item as ImageContent;
          this.send({ type: 'input.image', event_id: generateEventId(), timestamp: Date.now(), image: img.image, content_id: img.content_id } as unknown as InputTextEvent);
        } else if (item.type === 'audio') {
          const aud = item as AudioContent;
          this.send({ type: 'input.audio', event_id: generateEventId(), timestamp: Date.now(), audio: aud.audio, format: aud.format ?? 'webm', content_id: aud.content_id } as unknown as InputTextEvent);
        } else if (item.type === 'video') {
          const vid = item as VideoContent;
          this.send({ type: 'input.video', event_id: generateEventId(), timestamp: Date.now(), video: vid.video, content_id: vid.content_id } as unknown as InputTextEvent);
        } else if (item.type === 'file') {
          const f = item as FileContent;
          this.send({ type: 'input.file', event_id: generateEventId(), timestamp: Date.now(), file: f.file, filename: f.filename, mime_type: f.mime_type, content_id: f.content_id } as unknown as InputTextEvent);
        }
      }
    }

    this.send({
      type: 'response.create',
      event_id: generateEventId(),
      timestamp: Date.now(),
    } as ResponseCreateEvent);
  }

  async sendResponse(config: {
    messages?: Array<{ role: string; content?: string | null; content_items?: ContentItem[]; tool_calls?: unknown[]; tool_call_id?: string }>;
    model?: string;
    tools?: unknown[];
    temperature?: number;
    max_tokens?: number;
  }): Promise<void> {
    this.call = {};
    const responseCreate = {
      type: 'response.create' as const,
      event_id: generateEventId(),
      timestamp: Date.now(),
      response: config,
    };
    const payload = serializeEvent(responseCreate as unknown as Parameters<typeof serializeEvent>[0]);
    console.log(`[uamp-client] sending response.create (${payload.length} bytes, ${config.messages?.length ?? 0} messages, ${config.tools?.length ?? 0} tools)`);
    if (!this.ws || this.ws.readyState !== OPEN) {
      throw new Error('WebSocket is not connected');
    }
    this.ws.send(payload);
    console.log(`[uamp-client] response.create sent`);
  }

  async sendPayment(payment: { scheme: string; amount: string; token?: string; proof?: string }): Promise<void> {
    const paymentSubmit: PaymentSubmitEvent = {
      type: 'payment.submit',
      event_id: generateEventId(),
      timestamp: Date.now(),
      payment,
    };
    this.send(paymentSubmit);
  }

  async cancel(): Promise<void> {
    const cancelEvent: ResponseCancelEvent = {
      type: 'response.cancel',
      event_id: generateEventId(),
      timestamp: Date.now(),
    };
    this.send(cancelEvent);
  }

  close(): void {
    this.clearResponseTimer();
    if (this.ws && this.ws.readyState !== CLOSED && this.ws.readyState !== CLOSING) {
      this.ws.close();
    }
    this.ws = null;
    this.connected = false;
  }

  get isConnected(): boolean {
    return this.connected && this.ws !== null && this.ws.readyState === OPEN;
  }

  private emit<K extends keyof UAMPClientEvents>(event: K, ...args: Parameters<UAMPClientEvents[K]>): void {
    const handlers = this.listeners.get(event);
    if (!handlers) return;
    for (const handler of handlers) {
      try {
        (handler as (...a: unknown[]) => void)(...args);
      } catch {
        // Don't let listener errors propagate
      }
    }
  }

  private connectResolve: (() => void) | null = null;

  private resetResponseTimer(): void {
    if (this.responseTimer) clearTimeout(this.responseTimer);
    const timeout = this.config.responseTimeout ?? 120_000;
    if (timeout <= 0) return;
    this.responseTimer = setTimeout(() => {
      this.emit('error', new Error(`Response timeout after ${timeout}ms`));
    }, timeout);
  }

  private clearResponseTimer(): void {
    if (this.responseTimer) { clearTimeout(this.responseTimer); this.responseTimer = null; }
  }

  private handleMessage(data: string): void {
    let event: ReturnType<typeof parseEvent>;
    try {
      event = parseEvent(data);
    } catch {
      console.warn(`[uamp-client] unparseable message (${data.length} bytes)`);
      return;
    }
    console.log(`[uamp-client] ← ${event.type}`);

    switch (event.type) {
      case 'session.created': {
        if (this.connectResolve) {
          this.connectResolve();
          this.connectResolve = null;
        }
        break;
      }

      case 'response.delta': {
        this.resetResponseTimer();
        const e = event as ResponseDeltaEvent;
        const _dt = (e.delta as { type?: string }).type;
        if (e.delta.text != null) {
          console.log(`[uamp-client] delta emit: delta.type=${_dt} text=${JSON.stringify(e.delta.text)?.slice(0, 80)}`);
          this.emit('delta', e.delta.text);
        }
        if (e.delta.tool_call) {
          if (process.env.LOG_LOOP_DEBUG === '1') {
            const argsLen = typeof e.delta.tool_call.arguments === 'string' ? e.delta.tool_call.arguments.length : 0;
            console.log(`[loop-debug] uamp-client emit toolCall name=${e.delta.tool_call.name} args.len=${argsLen} (${argsLen === 0 ? 'tool_call_start' : 'final'})`);
          }
          this.emit('toolCall', e.delta.tool_call);
        }
        if (e.delta.tool_result) {
          this.emit('toolResult', e.delta.tool_result as Parameters<UAMPClientEvents['toolResult']>[0]);
        }
        if ((e.delta as { tool_progress?: Record<string, unknown> }).tool_progress) {
          if (process.env.LOG_LOOP_DEBUG === '1') {
            const tp = (e.delta as { tool_progress: { call_id?: string; status?: string; text?: string } }).tool_progress;
            console.log(`[loop-debug] uamp-client emit toolProgress call_id=${tp.call_id} status=${tp.status} text=${JSON.stringify(tp.text)?.slice(0, 60)}`);
          }
          this.emit('toolProgress', (e.delta as { tool_progress: Parameters<UAMPClientEvents['toolProgress']>[0] }).tool_progress);
        }
        if ((e.delta as { type?: string }).type === 'file') {
          console.log(`[uamp-client] file delta received: content_id=${(e.delta as any).content_id} filename=${(e.delta as any).filename}`);
          this.emit('file', e.delta as unknown as Record<string, unknown>);
        }
        break;
      }

      case 'tool.call': {
        this.resetResponseTimer();
        const tc = event as { call_id: string; name: string; arguments: string };
        this.emit('toolCall', { id: tc.call_id, name: tc.name, arguments: tc.arguments });
        break;
      }

      case 'thinking': {
        this.resetResponseTimer();
        const t = event as { content?: string; thinking?: { content?: string; stage?: string; redacted?: boolean; is_delta?: boolean }; stage?: string; redacted?: boolean; is_delta?: boolean };
        this.emit('thinking', {
          content: t.thinking?.content ?? t.content ?? '',
          stage: t.thinking?.stage ?? t.stage,
          redacted: t.thinking?.redacted ?? t.redacted,
          is_delta: t.thinking?.is_delta ?? t.is_delta,
        });
        break;
      }

      case 'response.done': {
        this.clearResponseTimer();
        const e = event as ResponseDoneEvent;
        this.emit('done', {
          output: e.response.output,
          usage: e.response.usage,
          id: e.response.id,
          status: e.response.status,
          ...(e.response.pre_executed_rounds && { pre_executed_rounds: e.response.pre_executed_rounds }),
        });
        break;
      }

      case 'response.error': {
        this.clearResponseTimer();
        const e = event as ResponseErrorEvent;
        this.emit('error', new Error(e.error.message));
        break;
      }

      case 'payment.required': {
        const e = event as PaymentRequiredEvent;
        const requirements = {
          amount: e.requirements.amount,
          currency: e.requirements.currency,
          schemes: e.requirements.schemes,
          reason: e.requirements.reason,
        };
        // In-band purchase (design section 6.1): the buyer pays the `mpp`
        // entry at its purchase URL and the run resumes with scheme
        // `balance`. The token scheme at index 0 is untouched; a refusal
        // falls through to the listeners exactly as if no buyer existed.
        // An entry with no challenge is the purchase pointer, followed only
        // by a buyer that implements `purchaseAt` (the config doc above).
        const buyer = this.config.buyer;
        const mpp = buyer ? findMppScheme(e.requirements.schemes) : null;
        const pointer = mpp !== null && mpp.challenge === null;
        if (buyer && mpp && (!pointer || typeof buyer.purchaseAt === 'function')) {
          this.emit('paymentPurchasing', {
            purchase_url: mpp.purchase_url,
            amount: e.requirements.amount,
            currency: e.requirements.currency,
            ...(mpp.terms ? { terms: mpp.terms } : {}),
          });
          const call = this.call;
          const paying =
            mpp.challenge === null
              ? buyer.purchaseAt!({ url: mpp.purchase_url, from: this.config.url, call })
              : buyer.purchase({ url: mpp.purchase_url, challenge: mpp.challenge, terms: mpp.terms ?? null, call });
          paying
            .then((outcome) => {
              if (outcome.ok && pointer && carriesPaymentToken(this.config)) {
                // The pointer went to a token holder, and the socket that
                // sent it resumes on a token and nothing else: the listeners
                // own that token, so the bought balance is theirs to use.
                this.emit('paymentRequired', { ...requirements, purchased: true });
                return undefined;
              }
              if (outcome.ok) {
                return this.sendPayment({ scheme: 'balance', amount: e.requirements.amount });
              }
              console.log(`[uamp-client] in-band purchase refused: ${outcome.reason ?? 'unknown'}${outcome.detail ? ` (${outcome.detail})` : ''}`);
              this.emit('paymentRequired', requirements);
              return undefined;
            })
            .catch((err: unknown) => {
              this.clearResponseTimer();
              this.emit('error', err instanceof Error ? err : new Error(String(err)));
            });
          break;
        }
        this.emit('paymentRequired', requirements);
        break;
      }

      case 'payment.accepted': {
        const e = event as PaymentAcceptedEvent;
        this.emit('paymentAccepted', {
          payment_id: e.payment_id,
          balance_remaining: e.balance_remaining,
        });
        break;
      }

      case 'response.cancelled': {
        this.clearResponseTimer();
        const e = event as ResponseCancelledEvent;
        this.emit('cancelled', {
          response_id: e.response_id,
          partial_output: e.partial_output,
        });
        break;
      }

      case 'payment.error': {
        this.clearResponseTimer();
        const e = event as PaymentErrorEvent;
        this.emit('error', new Error(e.message));
        break;
      }
    }
  }

  private send(event: object): void {
    if (!this.ws || this.ws.readyState !== OPEN) {
      throw new Error('WebSocket is not connected');
    }
    this.ws.send(serializeEvent(event as Parameters<typeof serializeEvent>[0]));
  }
}

/**
 * The `mpp` scheme entry of a `payment.required`: always a purchase URL, and
 * the challenge when the sender minted one. `challenge: null` is the purchase
 * pointer, an entry with NO `challenge` member. An entry whose `challenge` is
 * present and not a non-empty string is malformed and is never read as a
 * pointer.
 */
function findMppScheme(
  schemes: ReadonlyArray<{ scheme: string; challenge?: string | null; purchase_url?: string; terms?: { url: string; version: string } }>,
): { challenge: string | null; purchase_url: string; terms?: { url: string; version: string } } | null {
  for (const entry of schemes) {
    if (entry.scheme !== 'mpp') continue;
    if (typeof entry.purchase_url !== 'string' || !entry.purchase_url.trim()) continue;
    const hasChallenge = entry.challenge !== undefined && entry.challenge !== null;
    if (hasChallenge && (typeof entry.challenge !== 'string' || !entry.challenge.trim())) continue;
    return { challenge: hasChallenge ? (entry.challenge as string) : null, purchase_url: entry.purchase_url, ...(entry.terms ? { terms: entry.terms } : {}) };
  }
  return null;
}
