/**
 * Base Skill Class
 * 
 * Skills are modular components that provide tools, hooks, handoffs,
 * and endpoints to agents.
 */

import type {
  Tool,
  Hook,
  Handoff,
  Prompt,
  HttpEndpoint,
  WebSocketEndpoint,
  SkillConfig,
  ISkill,
  ToolHandler,
  HookHandler,
  HandoffHandler,
  HttpHandler,
  WebSocketHandler,
  Context,
} from './types';

import {
  TOOLS_KEY,
  HOOKS_KEY,
  HANDOFFS_KEY,
  PROMPTS_KEY,
  HTTP_KEY,
  WEBSOCKET_KEY,
  getMetadata,
} from './decorators';

/**
 * Base class for all skills
 * 
 * @example
 * ```typescript
 * class SearchSkill extends Skill {
 *   @tool({ provides: 'search', description: 'Search the web' })
 *   async search(params: { query: string }) {
 *     const results = await webSearch(params.query);
 *     return { results };
 *   }
 * }
 * 
 * const agent = new BaseAgent({
 *   skills: [new SearchSkill()]
 * });
 * ```
 */
export abstract class Skill implements ISkill {
  /** Skill name */
  readonly name: string;
  
  /** Whether skill is enabled */
  enabled: boolean;

  /**
   * Names of skills this one depends on. Declarative contract — when
   * the agent is constructed, `BaseAgent` validates that every name
   * here corresponds to another skill the author passed in `skills`,
   * then topologically sorts so dependencies initialise first.
   * Dependencies are NEVER auto-materialised; missing names throw
   * with an actionable message.
   *
   * Default is no dependencies. Override in subclasses, e.g.:
   * ```ts
   * class CronSkill extends Skill {
   *   readonly dependencies = ['function-runtime'] as const;
   * }
   * ```
   */
  readonly dependencies: readonly string[] = [];
  
  /** Skill configuration */
  protected config: SkillConfig;
  
  /** Registered tools (populated from decorators) */
  private _tools: Tool[] = [];
  
  /** Registered hooks (populated from decorators) */
  private _hooks: Hook[] = [];
  
  /** Registered handoffs (populated from decorators) */
  private _handoffs: Handoff[] = [];
  
  /** Registered HTTP endpoints (populated from decorators) */
  private _httpEndpoints: HttpEndpoint[] = [];
  
  /** Registered WebSocket endpoints (populated from decorators) */
  private _wsEndpoints: WebSocketEndpoint[] = [];

  /** Registered prompts (populated from decorators) */
  private _prompts: Prompt[] = [];
  
  constructor(config: SkillConfig = {}) {
    this.config = config;
    this.name = config.name || this.constructor.name;
    this.enabled = config.enabled ?? true;
    
    // Collect decorated members
    this.collectDecorators();
  }
  
  /**
   * Get registered tools
   */
  get tools(): Tool[] {
    return this._tools.filter(t => t.enabled);
  }
  
  /**
   * Get registered hooks
   */
  get hooks(): Hook[] {
    return this._hooks.filter(h => h.enabled);
  }
  
  /**
   * Get registered handoffs
   */
  get handoffs(): Handoff[] {
    return this._handoffs.filter(h => h.enabled);
  }
  
  /**
   * Get registered HTTP endpoints
   */
  get httpEndpoints(): HttpEndpoint[] {
    return this._httpEndpoints.filter(e => e.enabled);
  }
  
  /**
   * Get registered WebSocket endpoints
   */
  get wsEndpoints(): WebSocketEndpoint[] {
    return this._wsEndpoints.filter(e => e.enabled);
  }

  /**
   * Get registered prompts
   */
  get prompts(): Prompt[] {
    return this._prompts;
  }
  
  /**
   * Initialize the skill (override in subclasses)
   */
  async initialize(): Promise<void> {
    // Default implementation does nothing
  }
  
  /**
   * Cleanup resources (override in subclasses)
   */
  async cleanup(): Promise<void> {
    // Default implementation does nothing
  }
  
  /**
   * Walk the prototype chain and merge decorator metadata from every
   * ancestor class (e.g. `LocalNotificationSkill → NotificationSkill →
   * Skill`). Without this, decorators applied on an abstract base
   * (`@hook` on `NotificationSkill.installContextHooks`) would not be
   * picked up when a concrete subclass is instantiated, because the
   * metadata is keyed on the constructor that *defined* the decorator,
   * not the leaf class. Subclass entries override ancestor entries by
   * method name so a subclass can replace a parent decoration.
   */
  private collectInheritedMetadata<T>(key: symbol): Map<string, T> {
    const merged = new Map<string, T>();
    let ctor: Function | null | undefined = this.constructor;
    const chain: Function[] = [];
    while (ctor && ctor !== Object && ctor !== Function.prototype) {
      chain.unshift(ctor);
      ctor = Object.getPrototypeOf(ctor.prototype)?.constructor;
      if (!ctor || ctor === chain[0]) break;
    }
    for (const klass of chain) {
      const m = getMetadata<Map<string, T>>(key, klass);
      if (!m) continue;
      for (const [k, v] of m) merged.set(k, v);
    }
    return merged;
  }

  /**
   * Collect decorated methods and populate registries
   */
  private collectDecorators(): void {
    // Collect tools
    const toolsMap: Map<string, Partial<Tool>> =
      this.collectInheritedMetadata<Partial<Tool>>(TOOLS_KEY);
    
    for (const [methodName, toolMeta] of toolsMap) {
      const method = (this as unknown as Record<string, ToolHandler>)[methodName];
      if (typeof method === 'function') {
        this._tools.push({
          name: toolMeta.name || methodName,
          description: toolMeta.description,
          parameters: toolMeta.parameters,
          provides: toolMeta.provides,
          scopes: toolMeta.scopes,
          enabled: toolMeta.enabled ?? true,
          handler: method.bind(this),
          requiresConfirmation: toolMeta.requiresConfirmation,
          requiresBridge: toolMeta.requiresBridge,
        });
      }
    }
    
    // Collect hooks
    const hooksMap: Map<string, Partial<Hook>> =
      this.collectInheritedMetadata<Partial<Hook>>(HOOKS_KEY);
    
    for (const [methodName, hookMeta] of hooksMap) {
      const method = (this as unknown as Record<string, HookHandler>)[methodName];
      if (typeof method === 'function' && hookMeta.lifecycle) {
        this._hooks.push({
          lifecycle: hookMeta.lifecycle,
          priority: hookMeta.priority ?? 50,
          enabled: hookMeta.enabled ?? true,
          handler: method.bind(this),
        });
      }
    }
    
    // Collect handoffs
    const handoffsMap: Map<string, Partial<Handoff>> =
      this.collectInheritedMetadata<Partial<Handoff>>(HANDOFFS_KEY);
    
    for (const [methodName, handoffMeta] of handoffsMap) {
      const method = (this as unknown as Record<string, HandoffHandler>)[methodName];
      if (typeof method === 'function' && handoffMeta.name) {
        this._handoffs.push({
          name: handoffMeta.name,
          description: handoffMeta.description,
          priority: handoffMeta.priority ?? 0,
          scopes: handoffMeta.scopes,
          enabled: handoffMeta.enabled ?? true,
          subscribes: handoffMeta.subscribes ?? ['input.text'],
          produces: handoffMeta.produces ?? ['response.delta'],
          handler: method.bind(this),
        });
      }
    }
    
    // Collect HTTP endpoints
    const httpMap: Map<string, Partial<HttpEndpoint>> =
      this.collectInheritedMetadata<Partial<HttpEndpoint>>(HTTP_KEY);
    
    for (const [methodName, httpMeta] of httpMap) {
      const method = (this as unknown as Record<string, HttpHandler>)[methodName];
      if (typeof method === 'function' && httpMeta.path) {
        this._httpEndpoints.push({
          path: httpMeta.path,
          method: httpMeta.method ?? 'GET',
          scopes: httpMeta.scopes,
          content_type: httpMeta.content_type,
          enabled: httpMeta.enabled ?? true,
          auth: httpMeta.auth ?? 'public',
          handler: method.bind(this),
        });
      }
    }
    
    // Collect WebSocket endpoints
    const wsMap: Map<string, Partial<WebSocketEndpoint>> =
      this.collectInheritedMetadata<Partial<WebSocketEndpoint>>(WEBSOCKET_KEY);
    
    for (const [methodName, wsMeta] of wsMap) {
      const method = (this as unknown as Record<string, WebSocketHandler>)[methodName];
      if (typeof method === 'function' && wsMeta.path) {
        this._wsEndpoints.push({
          path: wsMeta.path,
          scopes: wsMeta.scopes,
          protocols: wsMeta.protocols,
          enabled: wsMeta.enabled ?? true,
          handler: method.bind(this),
        });
      }
    }

    // Collect prompts
    const promptsMap: Map<string, Partial<Prompt>> =
      this.collectInheritedMetadata<Partial<Prompt>>(PROMPTS_KEY);

    for (const [methodName, promptMeta] of promptsMap) {
      const method = (this as unknown as Record<string, (ctx: Context) => string | Promise<string>>)[methodName];
      if (typeof method === 'function') {
        this._prompts.push({
          name: promptMeta.name || methodName,
          priority: promptMeta.priority ?? 50,
          scope: promptMeta.scope ?? 'all',
          handler: method.bind(this),
        });
      }
    }
    this._prompts.sort((a, b) => a.priority - b.priority);
  }
  
  /**
   * Manually register a tool (alternative to decorator)
   */
  protected registerTool(tool: Tool): void {
    this._tools.push(tool);
  }
  
  /**
   * Manually register a hook (alternative to decorator)
   */
  protected registerHook(hook: Hook): void {
    this._hooks.push(hook);
  }
  
  /**
   * Manually register a handoff (alternative to decorator)
   */
  protected registerHandoff(handoff: Handoff): void {
    this._handoffs.push(handoff);
  }
  
  /**
   * Manually register an HTTP endpoint (alternative to decorator)
   */
  protected registerHttpEndpoint(endpoint: HttpEndpoint): void {
    this._httpEndpoints.push(endpoint);
  }
  
  /**
   * Manually register a WebSocket endpoint (alternative to decorator)
   */
  protected registerWebSocketEndpoint(endpoint: WebSocketEndpoint): void {
    this._wsEndpoints.push(endpoint);
  }

  /**
   * Manually register a prompt (alternative to decorator)
   */
  protected registerPrompt(p: Prompt): void {
    this._prompts.push(p);
    this._prompts.sort((a, b) => a.priority - b.priority);
  }
  
  /**
   * Enable/disable a tool by name
   */
  setToolEnabled(name: string, enabled: boolean): void {
    const tool = this._tools.find(t => t.name === name);
    if (tool) {
      tool.enabled = enabled;
    }
  }
  
  /**
   * Get a tool by name
   */
  getTool(name: string): Tool | undefined {
    return this._tools.find(t => t.name === name);
  }
  
  /**
   * Get a handoff by name
   */
  getHandoff(name: string): Handoff | undefined {
    return this._handoffs.find(h => h.name === name);
  }
}
