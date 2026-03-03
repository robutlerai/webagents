/**
 * BaseAgent Implementation
 * 
 * The core agent class that processes UAMP events, manages skills,
 * and provides the run() and runStreaming() interfaces.
 */

import type {
  AgentConfig,
  IAgent,
  ISkill,
  Tool,
  Hook,
  Handoff,
  HttpEndpoint,
  WebSocketEndpoint,
  Context,
  RunOptions,
  RunResponse,
  StreamChunk,
  HookData,
  HookResult,
  HookLifecycle,
  Observer,
} from './types.js';

import type {
  Capabilities,
  Message,
  ToolDefinition,
  ContentItem,
  UsageStats,
} from '../uamp/types.js';

import type {
  ClientEvent,
  ServerEvent,
  SessionCreateEvent,
  InputTextEvent,
  ResponseCreateEvent,
  ResponseDelta,
} from '../uamp/events.js';

import {
  generateEventId,
  createResponseErrorEvent,
} from '../uamp/events.js';

import { createContext, ContextImpl } from './context.js';
import { MessageRouter, type TransportSink, type UAMPEvent, type RouterContext } from './router.js';
import { getObservers } from './decorators.js';

/**
 * Base agent implementation
 */
export class BaseAgent implements IAgent {
  /** Agent name */
  readonly name: string;
  
  /** Agent description */
  readonly description?: string;
  
  /** System instructions */
  protected instructions?: string;
  
  /** Default model */
  protected model?: string;
  
  /** Loaded skills */
  protected skills: ISkill[] = [];
  
  /** Aggregated tools from all skills */
  protected toolRegistry: Map<string, Tool> = new Map();
  
  /** Aggregated hooks by lifecycle */
  protected hookRegistry: Map<HookLifecycle, Hook[]> = new Map();
  
  /** Aggregated handoffs (LLM skills) */
  protected handoffRegistry: Map<string, Handoff> = new Map();
  
  /** Aggregated observers */
  protected observerRegistry: Map<string, Observer> = new Map();
  
  /** HTTP endpoints */
  protected httpRegistry: Map<string, HttpEndpoint> = new Map();
  
  /** WebSocket endpoints */
  protected wsRegistry: Map<string, WebSocketEndpoint> = new Map();
  
  /** Agent capabilities */
  protected capabilities: Capabilities;
  
  /** Current context */
  protected context: Context;
  
  /** Message router for capability-based routing */
  public readonly router: MessageRouter;
  
  /** Whether a default handler has been set */
  private _hasDefaultHandler = false;
  
  constructor(config: AgentConfig = {}) {
    this.name = config.name || 'agent';
    this.description = config.description;
    this.instructions = config.instructions;
    this.model = config.model;
    
    // Initialize context
    this.context = createContext();
    
    // Initialize router
    this.router = new MessageRouter();
    
    // Initialize capabilities
    this.capabilities = {
      id: this.name,
      provider: 'webagents',
      modalities: config.capabilities?.modalities || ['text'],
      supports_streaming: config.capabilities?.supports_streaming ?? true,
      supports_thinking: config.capabilities?.supports_thinking ?? false,
      supports_caching: config.capabilities?.supports_caching ?? false,
      ...config.capabilities,
    };
    
    // Load skills
    if (config.skills) {
      for (const skill of config.skills) {
        this.addSkill(skill);
      }
    }
  }
  
  /**
   * Add a skill to the agent
   */
  addSkill(skill: ISkill): void {
    this.skills.push(skill);
    
    // Register tools
    for (const tool of skill.tools) {
      if (this.toolRegistry.has(tool.name)) {
        console.warn(`Tool "${tool.name}" already registered, overwriting`);
      }
      this.toolRegistry.set(tool.name, tool);
    }
    
    // Register hooks
    for (const hook of skill.hooks) {
      const existing = this.hookRegistry.get(hook.lifecycle) || [];
      existing.push(hook);
      // Sort by priority (lower = earlier)
      existing.sort((a, b) => a.priority - b.priority);
      this.hookRegistry.set(hook.lifecycle, existing);
    }
    
    // Register handoffs (both in registry and router)
    for (const handoff of skill.handoffs) {
      if (this.handoffRegistry.has(handoff.name)) {
        console.warn(`Handoff "${handoff.name}" already registered, overwriting`);
      }
      this.handoffRegistry.set(handoff.name, handoff);
      
      // Register with router for capability-based routing
      this.router.registerHandler({
        name: handoff.name,
        subscribes: handoff.subscribes || ['input.text'],
        produces: handoff.produces || ['response.delta'],
        priority: handoff.priority,
        process: async function* (event, routerContext) {
          // Convert router event to ClientEvent array and call handoff handler
          const clientEvent: ClientEvent = {
            type: event.type,
            event_id: event.id,
            ...event.payload,
          } as ClientEvent;
          
          // Create context from router context
          const context = createContext();
          if (routerContext?.sessionId) {
            context.session.id = routerContext.sessionId;
          }
          if (routerContext?.authToken) {
            context.auth.authenticated = true;
          }
          
          // Process through handoff and convert results to UAMPEvents
          for await (const serverEvent of handoff.handler([clientEvent], context)) {
            yield {
              id: serverEvent.event_id || '',
              type: serverEvent.type,
              payload: serverEvent as unknown as Record<string, unknown>,
            };
          }
        },
      });
      
      // First LLM-like skill (subscribes to input.text) becomes default handler
      const subscribesToText = handoff.subscribes?.includes('input.text') ?? true;
      if (subscribesToText && !this._hasDefaultHandler) {
        this.router.setDefault(handoff.name);
        this._hasDefaultHandler = true;
      }
    }
    
    // Register observers from skill
    const observers = getObservers(skill);
    for (const [methodName, observerMeta] of observers) {
      if (observerMeta.name) {
        const observer: Observer = {
          name: observerMeta.name,
          subscribes: observerMeta.subscribes || ['*'],
          enabled: observerMeta.enabled ?? true,
          handler: (skill as unknown as Record<string, unknown>)[methodName] as Observer['handler'],
        };
        this.observerRegistry.set(observer.name, observer);
        
        // Register with router
        this.router.registerObserver({
          name: observer.name,
          subscribes: observer.subscribes,
          handler: async (event, routerContext) => {
            const context = createContext();
            await observer.handler({ type: event.type, payload: event.payload }, context);
          },
        });
      }
    }
    
    // Register HTTP endpoints
    for (const endpoint of skill.httpEndpoints) {
      const key = `${endpoint.method}:${endpoint.path}`;
      if (this.httpRegistry.has(key)) {
        console.warn(`HTTP endpoint "${key}" already registered, overwriting`);
      }
      this.httpRegistry.set(key, endpoint);
    }
    
    // Register WebSocket endpoints
    for (const endpoint of skill.wsEndpoints) {
      if (this.wsRegistry.has(endpoint.path)) {
        console.warn(`WebSocket endpoint "${endpoint.path}" already registered, overwriting`);
      }
      this.wsRegistry.set(endpoint.path, endpoint);
    }
    
    // Update capabilities
    this.updateCapabilities();
  }
  
  /**
   * Remove a skill from the agent
   */
  removeSkill(skillName: string): void {
    const index = this.skills.findIndex(s => s.name === skillName);
    if (index === -1) return;
    
    const skill = this.skills[index];
    this.skills.splice(index, 1);
    
    // Remove tools
    for (const tool of skill.tools) {
      this.toolRegistry.delete(tool.name);
    }
    
    // Remove hooks
    for (const hook of skill.hooks) {
      const existing = this.hookRegistry.get(hook.lifecycle) || [];
      const hookIndex = existing.indexOf(hook);
      if (hookIndex !== -1) {
        existing.splice(hookIndex, 1);
      }
    }
    
    // Remove handoffs (from both registry and router)
    for (const handoff of skill.handoffs) {
      this.handoffRegistry.delete(handoff.name);
      this.router.unregisterHandler(handoff.name);
    }
    
    // Remove observers
    const observers = getObservers(skill);
    for (const [_, observerMeta] of observers) {
      if (observerMeta.name) {
        this.observerRegistry.delete(observerMeta.name);
        this.router.unregisterObserver(observerMeta.name);
      }
    }
    
    // Remove HTTP endpoints
    for (const endpoint of skill.httpEndpoints) {
      this.httpRegistry.delete(`${endpoint.method}:${endpoint.path}`);
    }
    
    // Remove WebSocket endpoints
    for (const endpoint of skill.wsEndpoints) {
      this.wsRegistry.delete(endpoint.path);
    }
    
    // Update capabilities
    this.updateCapabilities();
  }
  
  /**
   * Update agent capabilities based on loaded skills
   */
  protected updateCapabilities(): void {
    const provides: string[] = [];
    const endpoints: string[] = [];
    const builtInTools: string[] = [];
    
    // Collect from tools
    for (const tool of this.toolRegistry.values()) {
      if (tool.provides) {
        provides.push(tool.provides);
      }
      builtInTools.push(tool.name);
    }
    
    // Collect endpoints
    for (const endpoint of this.httpRegistry.values()) {
      endpoints.push(endpoint.path);
    }
    for (const endpoint of this.wsRegistry.values()) {
      endpoints.push(endpoint.path);
    }
    
    this.capabilities.provides = provides;
    this.capabilities.endpoints = endpoints;
    this.capabilities.tools = {
      supports_tools: this.toolRegistry.size > 0,
      supports_parallel_tools: true,
      supports_streaming_tools: false,
      built_in_tools: builtInTools,
    };
  }
  
  /**
   * Get agent capabilities
   */
  getCapabilities(): Capabilities {
    return { ...this.capabilities };
  }
  
  /**
   * Get all tools as ToolDefinition array
   */
  getToolDefinitions(): ToolDefinition[] {
    const definitions: ToolDefinition[] = [];
    for (const tool of this.toolRegistry.values()) {
      definitions.push({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      });
    }
    return definitions;
  }
  
  /**
   * Initialize all skills
   */
  async initialize(): Promise<void> {
    for (const skill of this.skills) {
      if (skill.initialize) {
        await skill.initialize();
      }
    }
  }
  
  /**
   * Cleanup all skills
   */
  async cleanup(): Promise<void> {
    for (const skill of this.skills) {
      if (skill.cleanup) {
        await skill.cleanup();
      }
    }
  }
  
  /**
   * Run hooks for a lifecycle point
   */
  protected async runHooks(
    lifecycle: HookLifecycle,
    data: HookData
  ): Promise<HookResult | undefined> {
    const hooks = this.hookRegistry.get(lifecycle) || [];
    let result: HookResult | undefined;
    
    for (const hook of hooks) {
      if (!hook.enabled) continue;
      
      const hookResult = await hook.handler(data, this.context);
      if (hookResult) {
        result = { ...result, ...hookResult };
        if (hookResult.skip_remaining || hookResult.abort) {
          break;
        }
      }
    }
    
    return result;
  }
  
  /**
   * Execute a tool by name
   */
  async executeTool(name: string, params: Record<string, unknown>): Promise<unknown> {
    const tool = this.toolRegistry.get(name);
    if (!tool) {
      throw new Error(`Tool not found: ${name}`);
    }
    
    // Check scopes
    if (tool.scopes && tool.scopes.length > 0) {
      if (!this.context.hasScopes(tool.scopes)) {
        throw new Error(`Insufficient permissions for tool: ${name}`);
      }
    }
    
    // Run before_tool hooks
    const beforeResult = await this.runHooks('before_tool', {
      tool_name: name,
      tool_params: params,
    });
    
    if (beforeResult?.abort) {
      throw new Error(beforeResult.abort_reason || 'Tool execution aborted by hook');
    }
    
    const finalParams = beforeResult?.tool_params || params;
    
    // Execute tool
    let result: unknown;
    try {
      result = await tool.handler(finalParams, this.context);
    } catch (error) {
      // Run on_error hooks
      await this.runHooks('on_error', { error: error as Error, tool_name: name });
      throw error;
    }
    
    // Run after_tool hooks
    const afterResult = await this.runHooks('after_tool', {
      tool_name: name,
      tool_params: finalParams,
      tool_result: result,
    });
    
    return afterResult?.tool_result ?? result;
  }
  
  /**
   * Get the best handoff (LLM skill) for processing
   */
  protected getBestHandoff(): Handoff | undefined {
    let best: Handoff | undefined;
    let bestPriority = -Infinity;
    
    for (const handoff of this.handoffRegistry.values()) {
      if (!handoff.enabled) continue;
      
      // Check scopes
      if (handoff.scopes && handoff.scopes.length > 0) {
        if (!this.context.hasScopes(handoff.scopes)) continue;
      }
      
      if (handoff.priority > bestPriority) {
        best = handoff;
        bestPriority = handoff.priority;
      }
    }
    
    return best;
  }
  
  /**
   * Process UAMP events (main entry point for transport skills)
   */
  async *processUAMP(events: ClientEvent[]): AsyncGenerator<ServerEvent, void, unknown> {
    // Find the best LLM handoff
    const handoff = this.getBestHandoff();
    
    if (!handoff) {
      yield createResponseErrorEvent(
        'no_handoff',
        'No LLM skill available to process request'
      );
      return;
    }
    
    // Process client capabilities if present
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.client_capabilities) {
          (this.context as ContextImpl).setClientCapabilities(createEvent.client_capabilities);
        }
      }
    }
    
    // Run before_handoff hooks
    const beforeResult = await this.runHooks('before_handoff', {
      handoff_target: handoff.name,
    });
    
    if (beforeResult?.abort) {
      yield createResponseErrorEvent(
        'handoff_aborted',
        beforeResult.abort_reason || 'Handoff aborted by hook'
      );
      return;
    }
    
    // Delegate to handoff handler
    try {
      yield* handoff.handler(events, this.context);
    } catch (error) {
      // Run on_error hooks
      await this.runHooks('on_error', { error: error as Error });
      yield createResponseErrorEvent(
        'handoff_error',
        (error as Error).message
      );
    }
    
    // Run after_handoff hooks
    await this.runHooks('after_handoff', {
      handoff_target: handoff.name,
    });
  }
  
  /**
   * Run with messages (convenience method)
   */
  async run(messages: Message[], options: RunOptions = {}): Promise<RunResponse> {
    // Convert messages to UAMP events
    const events: ClientEvent[] = [];
    
    // Create session
    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
        instructions: options.instructions || this.instructions,
        tools: options.tools ? options.tools.map(t => ({
          type: 'function' as const,
          function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters,
          },
        })) : this.getToolDefinitions(),
      },
    } as SessionCreateEvent);
    
    // Add messages as input events
    for (const msg of messages) {
      if (msg.role === 'user' || msg.role === 'system') {
        events.push({
          type: 'input.text',
          event_id: generateEventId(),
          text: msg.content || '',
          role: msg.role,
        } as InputTextEvent);
      }
    }
    
    // Request response
    events.push({
      type: 'response.create',
      event_id: generateEventId(),
    } as ResponseCreateEvent);
    
    // Run before_run hooks
    const beforeResult = await this.runHooks('before_run', { messages });
    if (beforeResult?.abort) {
      throw new Error(beforeResult.abort_reason || 'Run aborted by hook');
    }
    
    // Process events and collect response
    let content = '';
    let contentItems: ContentItem[] = [];
    let usage: UsageStats | undefined;
    
    for await (const event of this.processUAMP(events)) {
      if (event.type === 'response.delta') {
        const delta = (event as { delta: ResponseDelta }).delta;
        if (delta.text) {
          content += delta.text;
        }
      } else if (event.type === 'response.done') {
        const done = event as { response: { output: ContentItem[]; usage?: UsageStats } };
        contentItems = done.response.output;
        usage = done.response.usage;
      } else if (event.type === 'response.error') {
        const error = (event as { error: { message: string } }).error;
        throw new Error(error.message);
      }
    }
    
    const response: RunResponse = {
      content,
      content_items: contentItems,
      usage,
    };
    
    // Run after_run hooks
    await this.runHooks('after_run', { messages, response: content });
    
    return response;
  }
  
  /**
   * Run with streaming (convenience method)
   */
  async *runStreaming(
    messages: Message[],
    options: RunOptions = {}
  ): AsyncGenerator<StreamChunk, void, unknown> {
    // Convert messages to UAMP events
    const events: ClientEvent[] = [];
    
    // Create session
    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
        instructions: options.instructions || this.instructions,
        tools: options.tools ? options.tools.map(t => ({
          type: 'function' as const,
          function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters,
          },
        })) : this.getToolDefinitions(),
      },
    } as SessionCreateEvent);
    
    // Add messages as input events
    for (const msg of messages) {
      if (msg.role === 'user' || msg.role === 'system') {
        events.push({
          type: 'input.text',
          event_id: generateEventId(),
          text: msg.content || '',
          role: msg.role,
        } as InputTextEvent);
      }
    }
    
    // Request response
    events.push({
      type: 'response.create',
      event_id: generateEventId(),
    } as ResponseCreateEvent);
    
    // Run before_run hooks
    const beforeResult = await this.runHooks('before_run', { messages });
    if (beforeResult?.abort) {
      yield {
        type: 'error',
        error: new Error(beforeResult.abort_reason || 'Run aborted by hook'),
      };
      return;
    }
    
    // Process events and stream chunks
    let content = '';
    let contentItems: ContentItem[] = [];
    let usage: UsageStats | undefined;
    
    for await (const event of this.processUAMP(events)) {
      if (event.type === 'response.delta') {
        const delta = (event as { delta: ResponseDelta }).delta;
        if (delta.text) {
          content += delta.text;
          yield { type: 'delta', delta: delta.text };
        }
        if (delta.tool_call) {
          yield {
            type: 'tool_call',
            tool_call: {
              id: delta.tool_call.id,
              name: delta.tool_call.name,
              arguments: delta.tool_call.arguments,
            },
          };
        }
      } else if (event.type === 'response.done') {
        const done = event as { response: { output: ContentItem[]; usage?: UsageStats } };
        contentItems = done.response.output;
        usage = done.response.usage;
        
        yield {
          type: 'done',
          response: {
            content,
            content_items: contentItems,
            usage,
          },
        };
      } else if (event.type === 'response.error') {
        const error = (event as { error: { message: string } }).error;
        yield {
          type: 'error',
          error: new Error(error.message),
        };
      }
    }
    
    // Run after_run hooks
    await this.runHooks('after_run', { messages, response: content });
  }
  
  /**
   * Get an HTTP handler for the agent
   */
  getHttpHandler(path: string, method: string): HttpEndpoint | undefined {
    return this.httpRegistry.get(`${method}:${path}`);
  }
  
  /**
   * Get a WebSocket handler for the agent
   */
  getWebSocketHandler(path: string): WebSocketEndpoint | undefined {
    return this.wsRegistry.get(path);
  }
  
  // ============================================================================
  // Router Integration
  // ============================================================================
  
  /**
   * Connect a transport sink to the agent's router
   * 
   * @example
   * ```typescript
   * const agent = new BaseAgent();
   * const ws = new WebSocket('...');
   * agent.connectTransport(new WebSocketSink(ws, 'client-1'));
   * ```
   */
  connectTransport(sink: TransportSink): void {
    this.router.registerSink(sink);
    this.router.setActiveSink(sink.id);
  }
  
  /**
   * Connect a transport sink as the default sink (catches all unhandled events)
   * 
   * @example
   * ```typescript
   * const agent = new BaseAgent();
   * const ws = new WebSocket('...');
   * agent.connectDefaultTransport(new WebSocketSink(ws, 'client-1'));
   * ```
   */
  connectDefaultTransport(sink: TransportSink): void {
    this.router.registerDefaultSink(sink);
    this.router.setActiveSink(sink.id);
  }
  
  /**
   * Disconnect a transport sink from the agent's router
   */
  disconnectTransport(sinkId: string): void {
    this.router.unregisterSink(sinkId);
  }
  
  /**
   * Process a UAMP event through the router
   * 
   * This is the router-based entry point for message processing.
   * Events are automatically routed to handlers based on their capabilities.
   * 
   * @example
   * ```typescript
   * await agent.processMessage({
   *   id: 'msg-1',
   *   type: 'input.text',
   *   payload: { text: 'Hello!' }
   * });
   * ```
   */
  async processMessage(event: UAMPEvent): Promise<void> {
    // Create router context from agent context
    const routerContext: RouterContext = {
      sessionId: this.context.session.id,
      authToken: this.context.auth.authenticated ? 'valid' : undefined,
    };
    
    await this.router.send(event, routerContext);
  }
  
  /**
   * Send a text message through the router
   * 
   * Convenience method for sending text input.
   * 
   * @example
   * ```typescript
   * await agent.sendText('Hello, world!');
   * ```
   */
  async sendText(text: string): Promise<void> {
    await this.processMessage({
      id: `msg-${Date.now()}`,
      type: 'input.text',
      payload: { text },
    });
  }
  
  /**
   * Send an audio message through the router
   * 
   * @example
   * ```typescript
   * await agent.sendAudio(audioData);
   * ```
   */
  async sendAudio(audio: ArrayBuffer | Uint8Array | string): Promise<void> {
    await this.processMessage({
      id: `msg-${Date.now()}`,
      type: 'input.audio',
      payload: { audio },
    });
  }
}
