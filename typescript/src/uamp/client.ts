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
}

export interface UAMPClientEvents {
  delta: (text: string) => void;
  toolCall: (toolCall: { id: string; name: string; arguments: string }) => void;
  toolResult: (toolResult: { call_id: string; tool?: string; result?: string; command?: string; content_id?: string; path?: string; is_error?: boolean; content_items?: unknown[] }) => void;
  toolProgress: (progress: { call_id: string; text: string }) => void;
  file: (fileData: Record<string, unknown>) => void;
  thinking: (data: { content: string; stage?: string; redacted?: boolean; is_delta?: boolean }) => void;
  done: (response: { output: ContentItem[]; usage?: UsageStats; id: string; status: string }) => void;
  error: (error: Error) => void;
  paymentRequired: (requirements: { amount: string; currency: string; schemes: Array<{ scheme: string; network?: string }>; reason?: string }) => void;
  paymentAccepted: (data: { payment_id: string; balance_remaining?: string }) => void;
  cancelled: (data: { response_id: string; partial_output?: ContentItem[] }) => void;
}

export class UAMPClient {
  private ws: WS | null = null;
  private config: UAMPClientConfig;
  private listeners: Map<string, Set<Function>> = new Map();
  private connected: boolean = false;
  private responseTimer: ReturnType<typeof setTimeout> | null = null;

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

    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const wsOpts = this.config.headers ? { headers: this.config.headers } : undefined;
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
          this.emit('toolCall', e.delta.tool_call);
        }
        if (e.delta.tool_result) {
          this.emit('toolResult', e.delta.tool_result as Parameters<UAMPClientEvents['toolResult']>[0]);
        }
        if ((e.delta as { tool_progress?: { call_id: string; text: string } }).tool_progress) {
          this.emit('toolProgress', (e.delta as { tool_progress: { call_id: string; text: string } }).tool_progress);
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
        this.emit('paymentRequired', {
          amount: e.requirements.amount,
          currency: e.requirements.currency,
          schemes: e.requirements.schemes,
          reason: e.requirements.reason,
        });
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
