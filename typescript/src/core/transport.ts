/**
 * Transport Sinks for UAMP Router
 * 
 * Transport sinks are endpoints that receive response events from the router.
 * They are responsible for delivering events to clients via various protocols.
 */

import type { ServerEvent, TransportSink } from './router.js';

// ============================================================================
// WebSocket Transport Sink
// ============================================================================

/**
 * WebSocket sink for delivering events to WebSocket clients
 */
export class WebSocketSink implements TransportSink {
  /** Unique sink ID */
  public readonly id: string;
  
  /** The WebSocket connection */
  private ws: WebSocket;

  constructor(ws: WebSocket, id?: string) {
    this.ws = ws;
    this.id = id || `ws-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Whether the WebSocket connection is open
   */
  get isActive(): boolean {
    return this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Send an event through the WebSocket
   */
  async send(event: ServerEvent): Promise<void> {
    if (this.isActive) {
      this.ws.send(JSON.stringify(event));
    }
  }

  /**
   * Close the WebSocket connection
   */
  close(): void {
    if (this.ws.readyState !== WebSocket.CLOSED && this.ws.readyState !== WebSocket.CLOSING) {
      this.ws.close();
    }
  }

  /**
   * Get the underlying WebSocket
   */
  getSocket(): WebSocket {
    return this.ws;
  }
}

// ============================================================================
// Server-Sent Events (SSE) Transport Sink
// ============================================================================

/**
 * SSE response writer interface (Node.js compatible)
 */
export interface SSEResponseWriter {
  /** Write data to the response */
  write(data: string): boolean;
  /** End the response */
  end(): void;
  /** Whether the response is still writable */
  writableEnded?: boolean;
  /** Whether the response is finished */
  finished?: boolean;
}

/**
 * SSE sink for delivering events via Server-Sent Events
 */
export class SSESink implements TransportSink {
  /** Unique sink ID */
  public readonly id: string;
  
  /** The response writer */
  private response: SSEResponseWriter;
  
  /** Whether the sink has been closed */
  private closed = false;

  constructor(response: SSEResponseWriter, id?: string) {
    this.response = response;
    this.id = id || `sse-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Whether the SSE connection is active
   */
  get isActive(): boolean {
    if (this.closed) return false;
    if (this.response.writableEnded !== undefined) {
      return !this.response.writableEnded;
    }
    if (this.response.finished !== undefined) {
      return !this.response.finished;
    }
    return true;
  }

  /**
   * Send an event through SSE
   */
  async send(event: ServerEvent): Promise<void> {
    if (this.isActive) {
      const eventType = event.type || 'message';
      const data = JSON.stringify(event);
      
      // SSE format: event: <type>\ndata: <json>\n\n
      this.response.write(`event: ${eventType}\n`);
      this.response.write(`data: ${data}\n\n`);
    }
  }

  /**
   * Close the SSE connection
   */
  close(): void {
    if (!this.closed) {
      this.closed = true;
      this.response.end();
    }
  }
}

// ============================================================================
// Fetch API Response Sink (for Web Streams)
// ============================================================================

/**
 * Readable stream controller for Web Streams API
 */
export interface StreamController {
  enqueue(chunk: Uint8Array): void;
  close(): void;
  error(e: Error): void;
}

/**
 * Web Streams sink for browser environments using ReadableStream
 */
export class WebStreamSink implements TransportSink {
  /** Unique sink ID */
  public readonly id: string;
  
  /** The stream controller */
  private controller: StreamController | null = null;
  
  /** Text encoder for string to bytes conversion */
  private encoder = new TextEncoder();
  
  /** Whether the sink has been closed */
  private closed = false;

  constructor(id?: string) {
    this.id = id || `stream-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Create a ReadableStream that receives events from this sink
   */
  createStream(): ReadableStream<Uint8Array> {
    return new ReadableStream({
      start: (controller) => {
        this.controller = controller;
      },
      cancel: () => {
        this.closed = true;
        this.controller = null;
      },
    });
  }

  /**
   * Whether the stream is active
   */
  get isActive(): boolean {
    return !this.closed && this.controller !== null;
  }

  /**
   * Send an event through the stream (SSE format)
   */
  async send(event: ServerEvent): Promise<void> {
    if (this.isActive && this.controller) {
      const eventType = event.type || 'message';
      const data = JSON.stringify(event);
      const sseMessage = `event: ${eventType}\ndata: ${data}\n\n`;
      this.controller.enqueue(this.encoder.encode(sseMessage));
    }
  }

  /**
   * Close the stream
   */
  close(): void {
    if (!this.closed && this.controller) {
      this.closed = true;
      this.controller.close();
      this.controller = null;
    }
  }
}

// ============================================================================
// Callback Sink (for testing and programmatic use)
// ============================================================================

/**
 * Callback function type for receiving events
 */
export type EventCallback = (event: ServerEvent) => void | Promise<void>;

/**
 * Callback-based sink for testing and programmatic event handling
 */
export class CallbackSink implements TransportSink {
  /** Unique sink ID */
  public readonly id: string;
  
  /** Callback function */
  private callback: EventCallback;
  
  /** Whether the sink is active */
  private _isActive = true;

  constructor(callback: EventCallback, id?: string) {
    this.callback = callback;
    this.id = id || `callback-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Whether the sink is active
   */
  get isActive(): boolean {
    return this._isActive;
  }

  /**
   * Send an event to the callback
   */
  async send(event: ServerEvent): Promise<void> {
    if (this.isActive) {
      await this.callback(event);
    }
  }

  /**
   * Close the sink
   */
  close(): void {
    this._isActive = false;
  }
}

// ============================================================================
// Buffer Sink (for collecting events)
// ============================================================================

/**
 * Buffer sink that collects events for later retrieval
 */
export class BufferSink implements TransportSink {
  /** Unique sink ID */
  public readonly id: string;
  
  /** Collected events */
  private events: ServerEvent[] = [];
  
  /** Maximum buffer size */
  private maxSize: number;
  
  /** Whether the sink is active */
  private _isActive = true;

  constructor(options?: { id?: string; maxSize?: number }) {
    this.id = options?.id || `buffer-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.maxSize = options?.maxSize || 1000;
  }

  /**
   * Whether the sink is active
   */
  get isActive(): boolean {
    return this._isActive;
  }

  /**
   * Send an event to the buffer
   */
  async send(event: ServerEvent): Promise<void> {
    if (this.isActive) {
      this.events.push(event);
      // Trim buffer if it exceeds max size
      if (this.events.length > this.maxSize) {
        this.events = this.events.slice(-this.maxSize);
      }
    }
  }

  /**
   * Close the sink
   */
  close(): void {
    this._isActive = false;
  }

  /**
   * Get all collected events
   */
  getEvents(): ServerEvent[] {
    return [...this.events];
  }

  /**
   * Clear the buffer
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Get the number of collected events
   */
  get length(): number {
    return this.events.length;
  }
}
