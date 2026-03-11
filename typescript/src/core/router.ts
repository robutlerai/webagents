/**
 * UAMP Message Router
 * 
 * Central hub for capability-based message routing with auto-wiring,
 * observers, loop prevention, and system events.
 */

// ============================================================================
// Types
// ============================================================================

/**
 * UAMP Event with loop prevention metadata
 */
export interface UAMPEvent {
  /** Unique message ID (UUID) */
  id: string;
  /** Event type (e.g., 'input.text', 'response.delta') */
  type: string;
  /** Event payload */
  payload: Record<string, unknown>;
  // Loop prevention metadata:
  /** Handler that produced this message */
  source?: string;
  /** Time-to-live (max hops), default: 10 */
  ttl?: number;
  /** Handler IDs that have processed this message */
  seen?: string[];
}

/**
 * Server event for transport delivery
 */
export interface ServerEvent {
  type: string;
  payload: Record<string, unknown>;
  [key: string]: unknown;
}

/**
 * Execution context passed to handlers
 */
export interface RouterContext {
  /** Whether processing has been cancelled */
  cancelled?: boolean;
  /** Authentication token */
  authToken?: string;
  /** Session ID */
  sessionId?: string;
  /** Request scopes for access control (default: ['all']) */
  scopes?: string[];
  /** Additional context data */
  [key: string]: unknown;
}

/**
 * Route entry in routing table
 */
export interface Route {
  /** Event type or pattern */
  eventType: string;
  /** Handler for this route */
  handler: Handler;
  /** Priority (higher = preferred) */
  priority: number;
}

/**
 * Observer for non-consuming message listening
 */
export interface Observer {
  /** Observer name */
  name: string;
  /** Event types/patterns to observe ('*' for all) */
  subscribes: (string | RegExp)[];
  /** Required scopes for this observer (default: ['all']) */
  scopes?: string[];
  /** Handler function */
  handler: (event: UAMPEvent, context?: RouterContext) => Promise<void>;
}

/**
 * Transport sink for delivering responses
 */
export interface TransportSink {
  /** Unique sink ID */
  id: string;
  /** Send event to transport */
  send(event: ServerEvent): void | Promise<void>;
  /** Whether sink is active */
  isActive: boolean;
  /** Close the sink */
  close(): void;
}

/**
 * Handler for processing events
 */
export interface Handler {
  /** Handler name */
  name: string;
  /** Event types/patterns this handler accepts */
  subscribes: (string | RegExp)[];
  /** Event types this handler produces */
  produces: string[];
  /** Priority (higher = preferred) */
  priority: number;
  /** Required scopes for this handler (default: ['all']) */
  scopes?: string[];
  /** Process function (async generator) */
  process: (event: UAMPEvent, context: RouterContext) => AsyncGenerator<UAMPEvent, void, unknown>;
}

/**
 * Hook function types
 */
export type UnroutableHandler = (event: UAMPEvent, context?: RouterContext) => Promise<void>;
export type ErrorHandler = (error: Error, event: UAMPEvent, handler: Handler, context?: RouterContext) => Promise<void>;
export type RouteInterceptor = (event: UAMPEvent, handler: Handler | undefined, context?: RouterContext) => Promise<UAMPEvent | null>;

// ============================================================================
// Constants
// ============================================================================

/**
 * System event types for control flow
 */
export const SystemEvents = {
  ERROR: 'system.error',
  STOP: 'system.stop',
  CANCEL: 'system.cancel',
  PING: 'system.ping',
  PONG: 'system.pong',
  UNROUTABLE: 'system.unroutable',
} as const;

/**
 * UAMP Event Types
 */
export const UAMPEventTypes = {
  // Input events
  INPUT_TEXT: 'input.text',
  INPUT_AUDIO: 'input.audio',
  INPUT_IMAGE: 'input.image',
  // Response events
  RESPONSE_DELTA: 'response.delta',
  RESPONSE_DONE: 'response.done',
  // Audio events
  AUDIO_DELTA: 'audio.delta',
  TRANSCRIPT_DELTA: 'transcript.delta',
  // System events
  SYSTEM_ERROR: 'system.error',
  SYSTEM_STOP: 'system.stop',
  SYSTEM_CANCEL: 'system.cancel',
  SYSTEM_PING: 'system.ping',
  SYSTEM_PONG: 'system.pong',
} as const;

export type UAMPEventType = typeof UAMPEventTypes[keyof typeof UAMPEventTypes];

/** Default TTL for messages */
const DEFAULT_TTL = 10;

/** Lowest priority for default sink */
const DEFAULT_SINK_PRIORITY = -1000;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a UUID v4
 */
function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Check if an event type matches a subscription pattern
 */
export function matchesSubscription(eventType: string, patterns: (string | RegExp)[]): boolean {
  return patterns.some((pattern) => {
    if (pattern === '*') return true;
    if (pattern instanceof RegExp) return pattern.test(eventType);
    return pattern === eventType;
  });
}

/**
 * Check if request scopes match handler's required scopes
 * @param requestScopes - Scopes from the current request/context (strings or RegExp)
 * @param handlerScopes - Scopes required by the handler (strings or RegExp)
 * @returns True if access is allowed:
 *   - Empty/undefined scopes or 'all' or '' means no restriction
 *   - Any request scope matches any handler scope (supports regex)
 */
export function matchesScope(
  requestScopes: (string | RegExp)[],
  handlerScopes: (string | RegExp)[]
): boolean {
  // Normalize empty/undefined to ['all']
  const normReq = (!requestScopes || requestScopes.length === 0) 
    ? ['all'] 
    : requestScopes.map(s => s === '' ? 'all' : s);
  const normHandler = (!handlerScopes || handlerScopes.length === 0) 
    ? ['all'] 
    : handlerScopes.map(s => s === '' ? 'all' : s);

  // 'all' scope on handler means accessible to everyone
  if (normHandler.includes('all')) return true;
  // 'all' scope on request means superuser/admin access
  if (normReq.includes('all')) return true;

  // Check for matching scopes (supports regex patterns)
  for (const reqScope of normReq) {
    for (const handlerScope of normHandler) {
      // Both are regex patterns
      if (reqScope instanceof RegExp && handlerScope instanceof RegExp) {
        if (reqScope.source === handlerScope.source) return true;
      }
      // Request scope is regex
      else if (reqScope instanceof RegExp) {
        if (typeof handlerScope === 'string' && reqScope.test(handlerScope)) return true;
      }
      // Handler scope is regex
      else if (handlerScope instanceof RegExp) {
        if (typeof reqScope === 'string' && handlerScope.test(reqScope)) return true;
      }
      // Both are strings - exact match
      else if (reqScope === handlerScope) {
        return true;
      }
    }
  }

  return false;
}

// ============================================================================
// MessageRouter Class
// ============================================================================

/**
 * Central message router with capability-based routing
 */
export class MessageRouter {
  /** Routes by event type */
  private routes: Map<string, Route[]> = new Map();
  
  /** Registered handlers */
  private handlers: Map<string, Handler> = new Map();
  
  /** Non-consuming observers */
  private observers: Observer[] = [];
  
  /** Transport sinks */
  private sinks: Map<string, TransportSink> = new Map();
  
  /** Default handler for fallback routing */
  private _defaultHandler?: Handler;
  
  /** Active sink ID for responses */
  private _activeSinkId?: string;
  
  // Extensibility hooks
  private _onUnroutable?: UnroutableHandler;
  private _onError?: ErrorHandler;
  private _beforeRoute?: RouteInterceptor;
  private _afterRoute?: RouteInterceptor;

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Send a message into the router (main entry point for transports)
   */
  async send(event: UAMPEvent, context?: RouterContext): Promise<void> {
    // Ensure event has an ID
    const eventWithId: UAMPEvent = {
      ...event,
      id: event.id || generateId(),
    };

    // 1. TTL check - prevent deep chains
    const ttl = eventWithId.ttl ?? DEFAULT_TTL;
    if (ttl <= 0) {
      await this.emitSystemEvent(SystemEvents.ERROR, {
        error: 'TTL exceeded',
        originalEvent: eventWithId,
      });
      return;
    }

    // 2. Notify all observers (non-consuming, parallel)
    await this.notifyObservers(eventWithId, context);

    // 3. Handle system events
    if (this.isSystemEvent(eventWithId)) {
      await this.handleSystemEvent(eventWithId, context);
      return;
    }

    // 4. Find handler for event type (respecting scopes)
    const requestScopes = context?.scopes ?? ['all'];
    let handler = this.getHandler(eventWithId.type, requestScopes);
    
    // 5. Apply beforeRoute interceptor
    let processedEvent = eventWithId;
    if (this._beforeRoute) {
      const intercepted = await this._beforeRoute(eventWithId, handler, context);
      if (intercepted === null) {
        return; // Interceptor blocked the event
      }
      processedEvent = intercepted;
    }

    // 6. Use default handler if no specific handler found
    if (!handler) {
      handler = this._defaultHandler;
    }

    // 7. Check for wildcard handler if still no handler
    if (!handler) {
      const wildcardRoutes = this.routes.get('*');
      if (wildcardRoutes && wildcardRoutes.length > 0) {
        handler = wildcardRoutes[0].handler;
      }
    }

    // 8. If no handler found, emit unroutable event
    if (!handler) {
      if (this._onUnroutable) {
        await this._onUnroutable(processedEvent, context);
      }
      await this.emitSystemEvent(SystemEvents.UNROUTABLE, {
        originalEvent: processedEvent,
      });
      return;
    }

    // 9. Source check - don't route back to producer
    if (processedEvent.source === handler.name) {
      return; // Skip - would create loop
    }

    // 10. Seen check - don't process same message twice by same handler
    const seen = processedEvent.seen || [];
    if (seen.includes(handler.name)) {
      return; // Already processed by this handler
    }

    // 11. Process through handler
    await this.processHandlerOutput(handler, processedEvent, context);

    // 12. Apply afterRoute interceptor
    if (this._afterRoute) {
      await this._afterRoute(processedEvent, handler, context);
    }
  }

  /**
   * Register a handler (auto-wires based on capabilities)
   */
  registerHandler(handler: Handler): void {
    this.handlers.set(handler.name, handler);
    this.buildRoutingTable();
  }

  /**
   * Unregister a handler
   */
  unregisterHandler(name: string): void {
    this.handlers.delete(name);
    this.buildRoutingTable();
  }

  /**
   * Register an observer (non-consuming listener)
   */
  registerObserver(observer: Observer): void {
    this.observers.push(observer);
  }

  /**
   * Unregister an observer
   */
  unregisterObserver(name: string): void {
    this.observers = this.observers.filter((o) => o.name !== name);
  }

  /**
   * Add explicit route (overrides auto-wiring)
   */
  route(eventType: string, handlerName: string, priority?: number): void {
    const handler = this.handlers.get(handlerName);
    if (!handler) {
      throw new Error(`Handler '${handlerName}' not found`);
    }

    const routes = this.routes.get(eventType) || [];
    
    // Remove existing route for this handler
    const filtered = routes.filter((r) => r.handler.name !== handlerName);
    
    // Add new route
    filtered.push({
      eventType,
      handler,
      priority: priority ?? handler.priority,
    });
    
    // Sort by priority (higher first)
    filtered.sort((a, b) => b.priority - a.priority);
    
    this.routes.set(eventType, filtered);
  }

  /**
   * Register a transport sink for responses
   */
  registerSink(sink: TransportSink): void {
    this.sinks.set(sink.id, sink);
  }

  /**
   * Register a default sink that catches all unhandled response events
   */
  registerDefaultSink(sink: TransportSink): void {
    // Register as handler with lowest priority and wildcard pattern
    this.registerHandler({
      name: `default-sink-${sink.id}`,
      subscribes: ['*'],
      produces: [],
      priority: DEFAULT_SINK_PRIORITY,
      process: async function* (event) {
        await sink.send(event as unknown as ServerEvent);
      },
    });
    this.sinks.set(sink.id, sink);
  }

  /**
   * Unregister a sink
   */
  unregisterSink(id: string): void {
    this.sinks.delete(id);
    // Also remove any default sink handler
    this.handlers.delete(`default-sink-${id}`);
    this.buildRoutingTable();
  }

  /**
   * Set active sink (receives responses)
   */
  setActiveSink(sinkId: string): void {
    if (!this.sinks.has(sinkId)) {
      throw new Error(`Sink '${sinkId}' not registered`);
    }
    this._activeSinkId = sinkId;
  }

  /**
   * Get the active sink
   */
  get activeSink(): TransportSink | undefined {
    return this._activeSinkId ? this.sinks.get(this._activeSinkId) : undefined;
  }

  /**
   * Set default handler (fallback when no capability match)
   */
  setDefault(handlerName: string): void {
    const handler = this.handlers.get(handlerName);
    if (!handler) {
      throw new Error(`Handler '${handlerName}' not found`);
    }
    this._defaultHandler = handler;
  }

  /**
   * Get default handler
   */
  get defaultHandler(): Handler | undefined {
    return this._defaultHandler;
  }

  /**
   * Get all registered handlers
   */
  getHandlers(): Map<string, Handler> {
    return new Map(this.handlers);
  }

  /**
   * Get all registered observers
   */
  getObservers(): Observer[] {
    return [...this.observers];
  }

  // ============================================================================
  // Extensibility Hooks
  // ============================================================================

  /**
   * Set handler for unroutable events
   */
  onUnroutable(handler: UnroutableHandler): void {
    this._onUnroutable = handler;
  }

  /**
   * Set handler for errors during processing
   */
  onError(handler: ErrorHandler): void {
    this._onError = handler;
  }

  /**
   * Set interceptor called before routing
   * Return null to block the event, or modified event to continue
   */
  beforeRoute(interceptor: RouteInterceptor): void {
    this._beforeRoute = interceptor;
  }

  /**
   * Set interceptor called after routing
   */
  afterRoute(interceptor: RouteInterceptor): void {
    this._afterRoute = interceptor;
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Build routing table from handler capabilities
   */
  private buildRoutingTable(): void {
    this.routes.clear();

    for (const handler of this.handlers.values()) {
      for (const pattern of handler.subscribes) {
        // For string patterns, use exact match key
        const key = pattern instanceof RegExp ? pattern.source : pattern;
        const routes = this.routes.get(key) || [];
        
        routes.push({
          eventType: key,
          handler,
          priority: handler.priority,
        });
        
        // Sort by priority (higher first)
        routes.sort((a, b) => b.priority - a.priority);
        
        this.routes.set(key, routes);
      }
    }
  }

  /**
   * Get the best handler for an event type, respecting scopes
   * @param eventType - The event type to match
   * @param requestScopes - Scopes from the current request (default: ['all'])
   */
  private getHandler(eventType: string, requestScopes: string[] = ['all']): Handler | undefined {
    // 1. Check for exact match
    const exactRoutes = this.routes.get(eventType);
    if (exactRoutes && exactRoutes.length > 0) {
      // Find first handler that matches scopes
      for (const route of exactRoutes) {
        const handlerScopes = route.handler.scopes ?? ['all'];
        if (matchesScope(requestScopes, handlerScopes)) {
          return route.handler;
        }
      }
    }

    // 2. Check for regex matches
    for (const [_key, routes] of this.routes) {
      if (routes.length === 0) continue;
      
      for (const route of routes) {
        const handler = route.handler;
        const handlerScopes = handler.scopes ?? ['all'];
        
        // Check scope first
        if (!matchesScope(requestScopes, handlerScopes)) continue;
        
        // Then check pattern match
        for (const pattern of handler.subscribes) {
          if (pattern instanceof RegExp && pattern.test(eventType)) {
            return handler;
          }
        }
      }
    }

    return undefined;
  }

  /**
   * Notify all observers (non-consuming), respecting scopes
   */
  private async notifyObservers(event: UAMPEvent, context?: RouterContext): Promise<void> {
    const requestScopes = context?.scopes ?? ['all'];
    
    const notifications = this.observers
      .filter((observer) => {
        // Check scope access
        const observerScopes = observer.scopes ?? ['all'];
        if (!matchesScope(requestScopes, observerScopes)) return false;
        // Check subscription match
        return matchesSubscription(event.type, observer.subscribes);
      })
      .map(async (observer) => {
        try {
          await observer.handler(event, context);
        } catch (err) {
          // Observers should not break the flow
          console.error(`Observer ${observer.name} error:`, err);
        }
      });

    await Promise.all(notifications);
  }

  /**
   * Process handler output and route produced events
   */
  private async processHandlerOutput(
    handler: Handler,
    event: UAMPEvent,
    context?: RouterContext
  ): Promise<void> {
    const seen = [...(event.seen || []), handler.name];

    try {
      for await (const output of handler.process(event, context || {})) {
        // Route handler output back through router
        await this.send(
          {
            ...output,
            id: output.id || generateId(),
            source: handler.name,
            ttl: (event.ttl ?? DEFAULT_TTL) - 1,
            seen: [...seen],
          },
          context
        );
      }
    } catch (err) {
      if (this._onError) {
        await this._onError(err as Error, event, handler, context);
      } else {
        // Re-throw if no error handler
        throw err;
      }
    }
  }

  /**
   * Check if event is a system event
   */
  private isSystemEvent(event: UAMPEvent): boolean {
    return event.type.startsWith('system.');
  }

  /**
   * Handle system events
   */
  private async handleSystemEvent(event: UAMPEvent, context?: RouterContext): Promise<void> {
    switch (event.type) {
      case SystemEvents.STOP:
      case SystemEvents.CANCEL:
        // Signal handlers to stop via context
        if (context) {
          context.cancelled = true;
        }
        // Deliver to active sink
        await this.deliverToActiveSink(event);
        break;

      case SystemEvents.ERROR:
        // Deliver error to sink
        await this.deliverToActiveSink(event);
        break;

      case SystemEvents.PING:
        // Respond with pong
        await this.deliverToActiveSink({
          type: SystemEvents.PONG,
          payload: {},
        });
        break;

      case SystemEvents.UNROUTABLE:
        // Deliver unroutable notification
        await this.deliverToActiveSink(event);
        break;

      default:
        // Unknown system event - deliver as-is
        await this.deliverToActiveSink(event);
    }
  }

  /**
   * Deliver event to active sink
   */
  private async deliverToActiveSink(event: UAMPEvent | ServerEvent): Promise<void> {
    const sink = this.activeSink;
    if (sink && sink.isActive) {
      await sink.send(event as ServerEvent);
    }
  }

  /**
   * Emit a system event
   */
  private async emitSystemEvent(
    type: string,
    payload: Record<string, unknown>
  ): Promise<void> {
    const event: UAMPEvent = {
      id: generateId(),
      type,
      payload,
    };

    // Notify observers of system event
    await this.notifyObservers(event);

    // Handle the system event
    await this.handleSystemEvent(event);
  }
}
