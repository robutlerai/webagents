/**
 * Decorator System for WebAgents
 * 
 * TypeScript decorators for registering tools, hooks, handoffs, and endpoints.
 * Uses a simple Map-based metadata storage (no reflect-metadata dependency).
 */

import type {
  ToolConfig,
  HookConfig,
  HandoffConfig,
  HttpConfig,
  WebSocketConfig,
  ObserveConfig,
  PricingConfig,
  Tool,
  Hook,
  Handoff,
  HttpEndpoint,
  WebSocketEndpoint,
  Observer,
} from './types';

// ============================================================================
// Metadata Storage (Map-based, no reflect-metadata needed)
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const metadataStorage = new WeakMap<any, Map<symbol, unknown>>();

function getMetadata<T>(key: symbol, target: object): T | undefined {
  const targetMetadata = metadataStorage.get(target);
  return targetMetadata?.get(key) as T | undefined;
}

function defineMetadata<T>(key: symbol, value: T, target: object): void {
  let targetMetadata = metadataStorage.get(target);
  if (!targetMetadata) {
    targetMetadata = new Map();
    metadataStorage.set(target, targetMetadata);
  }
  targetMetadata.set(key, value);
}

// ============================================================================
// Metadata Keys (Symbol-based for privacy)
// ============================================================================

export const TOOLS_KEY = Symbol('webagents:tools');
export const HOOKS_KEY = Symbol('webagents:hooks');
export const HANDOFFS_KEY = Symbol('webagents:handoffs');
export const OBSERVERS_KEY = Symbol('webagents:observers');
export const HTTP_KEY = Symbol('webagents:http');
export const WEBSOCKET_KEY = Symbol('webagents:websocket');
export const PRICING_KEY = Symbol('webagents:pricing');

/** Default event types for handoffs */
const DEFAULT_SUBSCRIBES = ['input.text'];
const DEFAULT_PRODUCES = ['response.delta'];

// ============================================================================
// Tool Decorator
// ============================================================================

/**
 * Register a method as a tool
 * 
 * @example
 * ```typescript
 * class MySkill extends Skill {
 *   @tool({ provides: 'search' })
 *   async search(params: { query: string }) {
 *     return { results: [] };
 *   }
 * }
 * ```
 */
export function tool(config: ToolConfig = {}) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const tools: Map<string, Partial<Tool>> = 
      getMetadata(TOOLS_KEY, target.constructor) || new Map();
    
    tools.set(propertyKey, {
      name: config.name || propertyKey,
      description: config.description,
      parameters: config.parameters,
      provides: config.provides,
      scopes: config.scopes,
      enabled: config.enabled ?? true,
    });
    
    defineMetadata(TOOLS_KEY, tools, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered tools from a class
 */
export function getTools(target: object): Map<string, Partial<Tool>> {
  return getMetadata(TOOLS_KEY, target.constructor) || new Map();
}

// ============================================================================
// Hook Decorator
// ============================================================================

/**
 * Register a method as a lifecycle hook
 * 
 * @example
 * ```typescript
 * class MySkill extends Skill {
 *   @hook({ lifecycle: 'before_run', priority: 10 })
 *   async logRequest(data: HookData, ctx: Context) {
 *     console.log('Request:', data.messages);
 *   }
 * }
 * ```
 */
export function hook(config: HookConfig) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const hooks: Map<string, Partial<Hook>> = 
      getMetadata(HOOKS_KEY, target.constructor) || new Map();
    
    hooks.set(propertyKey, {
      lifecycle: config.lifecycle,
      priority: config.priority ?? 50,
      enabled: config.enabled ?? true,
    });
    
    defineMetadata(HOOKS_KEY, hooks, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered hooks from a class
 */
export function getHooks(target: object): Map<string, Partial<Hook>> {
  return getMetadata(HOOKS_KEY, target.constructor) || new Map();
}

// ============================================================================
// Handoff Decorator
// ============================================================================

/**
 * Register a method as a handoff target (LLM skill)
 * 
 * Handoffs are message handlers that subscribe to specific event types and
 * produce output events. The router uses these declarations for auto-wiring.
 * 
 * @example
 * ```typescript
 * class MyLLMSkill extends Skill {
 *   // Basic handoff (uses defaults: subscribes to 'input.text', produces 'response.delta')
 *   @handoff({ name: 'gpt-4', priority: 10 })
 *   async *processUAMP(events: ClientEvent[]) {
 *     // Process events and yield server events
 *   }
 * }
 * ```
 * 
 * @example
 * ```typescript
 * class STTSkill extends Skill {
 *   // Custom subscription for audio processing
 *   @handoff({ 
 *     name: 'speech-to-text',
 *     subscribes: ['input.audio'],
 *     produces: ['input.text'],
 *     priority: 100 
 *   })
 *   async *processAudio(events: ClientEvent[]) { ... }
 * }
 * ```
 * 
 * @example
 * ```typescript
 * class TranslationSkill extends Skill {
 *   // Regex pattern for dynamic routing
 *   @handoff({ 
 *     name: 'translator',
 *     subscribes: [/^translate\..+$/],  // Matches translate.en, translate.fr, etc.
 *     produces: ['response.delta']
 *   })
 *   async *translate(events: ClientEvent[]) { ... }
 * }
 * ```
 */
export function handoff(config: HandoffConfig) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const handoffs: Map<string, Partial<Handoff>> = 
      getMetadata(HANDOFFS_KEY, target.constructor) || new Map();
    
    handoffs.set(propertyKey, {
      name: config.name,
      description: config.description,
      priority: config.priority ?? 0,
      scopes: config.scopes,
      enabled: config.enabled ?? true,
      subscribes: config.subscribes ?? DEFAULT_SUBSCRIBES,
      produces: config.produces ?? DEFAULT_PRODUCES,
    });
    
    defineMetadata(HANDOFFS_KEY, handoffs, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered handoffs from a class
 */
export function getHandoffs(target: object): Map<string, Partial<Handoff>> {
  return getMetadata(HANDOFFS_KEY, target.constructor) || new Map();
}

// ============================================================================
// Observer Decorator
// ============================================================================

/**
 * Register a method as a non-consuming observer
 * 
 * Observers receive copies of events but do not consume them - the event
 * continues to be routed to handlers. Useful for logging, analytics, debugging.
 * 
 * @example
 * ```typescript
 * class LoggingSkill extends Skill {
 *   @observe({ name: 'message-logger', subscribes: ['*'] })
 *   async onMessage(event: UAMPEvent) {
 *     console.log(`[${event.type}]`, event.payload);
 *     // Does NOT consume - message continues to handlers
 *   }
 * }
 * ```
 * 
 * @example
 * ```typescript
 * class AnalyticsSkill extends Skill {
 *   @observe({ name: 'response-tracker', subscribes: ['response.delta', 'response.done'] })
 *   async trackResponse(event: UAMPEvent) {
 *     await this.analytics.track('response', event);
 *   }
 * }
 * ```
 */
export function observe(config: ObserveConfig) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const observers: Map<string, Partial<Observer>> = 
      getMetadata(OBSERVERS_KEY, target.constructor) || new Map();
    
    observers.set(propertyKey, {
      name: config.name,
      subscribes: config.subscribes,
      enabled: true,
    });
    
    defineMetadata(OBSERVERS_KEY, observers, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered observers from a class
 */
export function getObservers(target: object): Map<string, Partial<Observer>> {
  return getMetadata(OBSERVERS_KEY, target.constructor) || new Map();
}

// ============================================================================
// HTTP Decorator
// ============================================================================

/**
 * Register a method as an HTTP endpoint
 * 
 * @example
 * ```typescript
 * class MySkill extends Skill {
 *   @http({ path: '/api/data', method: 'GET' })
 *   async getData(request: Request) {
 *     return new Response(JSON.stringify({ data: [] }));
 *   }
 * }
 * ```
 */
export function http(config: HttpConfig) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const endpoints: Map<string, Partial<HttpEndpoint>> = 
      getMetadata(HTTP_KEY, target.constructor) || new Map();
    
    endpoints.set(propertyKey, {
      path: config.path,
      method: config.method ?? 'GET',
      scopes: config.scopes,
      content_type: config.content_type,
      enabled: config.enabled ?? true,
    });
    
    defineMetadata(HTTP_KEY, endpoints, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered HTTP endpoints from a class
 */
export function getHttpEndpoints(target: object): Map<string, Partial<HttpEndpoint>> {
  return getMetadata(HTTP_KEY, target.constructor) || new Map();
}

// ============================================================================
// WebSocket Decorator
// ============================================================================

/**
 * Register a method as a WebSocket endpoint
 * 
 * @example
 * ```typescript
 * class MySkill extends Skill {
 *   @websocket({ path: '/ws/stream' })
 *   handleConnection(ws: WebSocket, ctx: Context) {
 *     ws.onmessage = (msg) => { ... };
 *   }
 * }
 * ```
 */
export function websocket(config: WebSocketConfig) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const endpoints: Map<string, Partial<WebSocketEndpoint>> = 
      getMetadata(WEBSOCKET_KEY, target.constructor) || new Map();
    
    endpoints.set(propertyKey, {
      path: config.path,
      scopes: config.scopes,
      protocols: config.protocols,
      enabled: config.enabled ?? true,
    });
    
    defineMetadata(WEBSOCKET_KEY, endpoints, target.constructor);
    
    return descriptor;
  };
}

/**
 * Get registered WebSocket endpoints from a class
 */
export function getWebSocketEndpoints(target: object): Map<string, Partial<WebSocketEndpoint>> {
  return getMetadata(WEBSOCKET_KEY, target.constructor) || new Map();
}

// ============================================================================
// Pricing Decorator
// ============================================================================

/**
 * Register pricing metadata on a tool method.
 * Used by the payment skill's `before_toolcall` hook to pre-authorize
 * charges before tool execution.
 *
 * @example
 * ```typescript
 * class MySkill extends Skill {
 *   // Fixed pricing: 0.05 credits per call
 *   @pricing({ creditsPerCall: 0.05 })
 *   @tool({ description: 'Premium search' })
 *   async search(params: { query: string }, ctx: Context) {
 *     return await doSearch(params.query);
 *   }
 *
 *   // Dynamic pricing: return [result, PricingInfo]
 *   @pricing()
 *   @tool({ description: 'Variable cost operation' })
 *   async process(params: Record<string, unknown>, ctx: Context) {
 *     const result = await doWork(params);
 *     return [result, { credits: computeCost(result) }];
 *   }
 * }
 * ```
 */
export function pricing(config: PricingConfig = {}) {
  return function (
    target: object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const pricingMap: Map<string, PricingConfig> =
      getMetadata(PRICING_KEY, target.constructor) || new Map();

    pricingMap.set(propertyKey, config);

    defineMetadata(PRICING_KEY, pricingMap, target.constructor);

    return descriptor;
  };
}

/**
 * Get pricing metadata for a method on a class (by method name)
 */
export function getPricing(target: object): Map<string, PricingConfig> {
  return getMetadata(PRICING_KEY, target.constructor) || new Map();
}

/**
 * Get pricing config for a specific tool name across all skills on an agent
 */
export function getPricingForTool(
  skills: Array<{ constructor: Function }>,
  toolName: string
): PricingConfig | undefined {
  for (const skill of skills) {
    const pricingMap: Map<string, PricingConfig> | undefined =
      getMetadata(PRICING_KEY, skill.constructor);
    if (pricingMap) {
      for (const [methodName, config] of pricingMap) {
        const toolsMap: Map<string, Partial<Tool>> | undefined =
          getMetadata(TOOLS_KEY, skill.constructor);
        if (toolsMap) {
          const toolMeta = toolsMap.get(methodName);
          if (toolMeta && (toolMeta.name || methodName) === toolName) {
            return config;
          }
        }
      }
    }
  }
  return undefined;
}

// Internal exports for skill.ts
export { getMetadata, defineMetadata };
