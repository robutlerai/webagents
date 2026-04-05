/**
 * BaseAgent Implementation
 * 
 * The core agent class that processes UAMP events, manages skills,
 * and provides the run() and runStreaming() interfaces.
 */

import type {
  AgentConfig,
  AgenticMessage,
  StructuredToolResult,
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
} from './types';

import type {
  Capabilities,
  Message,
  ToolDefinition,
  ToolCall,
  ContentItem,
  TextContent,
  AudioContent,
  ImageContent,
  VideoContent,
  FileContent,
  UsageStats,
} from '../uamp/types';

import type {
  ClientEvent,
  ServerEvent,
  SessionCreateEvent,
  InputTextEvent,
  InputAudioEvent,
  InputImageEvent,
  InputVideoEvent,
  InputFileEvent,
  ResponseCreateEvent,
  ResponseDelta,
  ResponseDoneEvent,
} from '../uamp/events';

import {
  generateEventId,
  createResponseErrorEvent,
} from '../uamp/events';

import { ensureContentId } from '../uamp/content';

import { createContext, ContextImpl } from './context';
import { MessageRouter, type TransportSink, type UAMPEvent, type RouterContext } from './router';
import { getObservers } from './decorators';

/**
 * Async queue that allows a producer to push items while a consumer
 * pulls them via `for await`. Used to stream tool progress events
 * from within an awaited tool call back to the generator that yields
 * UAMP events.
 */
class AsyncQueue<T> implements AsyncIterableIterator<T> {
  private buffer: T[] = [];
  private waiting: ((r: IteratorResult<T>) => void) | null = null;
  private closed = false;

  push(item: T): void {
    if (this.closed) return;
    if (this.waiting) {
      const resolve = this.waiting;
      this.waiting = null;
      resolve({ value: item, done: false });
    } else {
      this.buffer.push(item);
    }
  }

  close(): void {
    this.closed = true;
    if (this.waiting) {
      const resolve = this.waiting;
      this.waiting = null;
      resolve({ value: undefined as unknown as T, done: true });
    }
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.buffer.length > 0) {
      return { value: this.buffer.shift()!, done: false };
    }
    if (this.closed) {
      return { value: undefined as unknown as T, done: true };
    }
    return new Promise<IteratorResult<T>>((resolve) => {
      this.waiting = resolve;
    });
  }

  [Symbol.asyncIterator](): AsyncIterableIterator<T> {
    return this;
  }
}

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
  
  /** Set of tool names overridden by external tools (client-executed) */
  private _overriddenTools: Set<string> = new Set();
  
  /** Max agentic loop iterations */
  protected maxToolIterations: number;
  
  /** Whether a default handler has been set */
  private _hasDefaultHandler = false;
  
  constructor(config: AgentConfig = {}) {
    this.name = config.name || 'agent';
    this.description = config.description;
    this.instructions = config.instructions;
    this.model = config.model;
    this.maxToolIterations = config.maxToolIterations ?? 10;
    
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

    if (typeof (skill as any).setAgent === 'function') {
      (skill as any).setAgent(this);
    }
    
    // Register tools (tolerant of non-conforming skills)
    for (const tool of (skill.tools ?? [])) {
      if (this.toolRegistry.has(tool.name)) {
        console.warn(`Tool "${tool.name}" already registered, overwriting`);
      }
      this.toolRegistry.set(tool.name, tool);
    }
    
    // Register hooks
    for (const hook of (skill.hooks ?? [])) {
      const existing = this.hookRegistry.get(hook.lifecycle) || [];
      existing.push(hook);
      // Sort by priority (lower = earlier)
      existing.sort((a, b) => a.priority - b.priority);
      this.hookRegistry.set(hook.lifecycle, existing);
    }
    
    // Register handoffs (both in registry and router)
    for (const handoff of (skill.handoffs ?? [])) {
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
          handler: async (event, _routerContext) => {
            const context = createContext();
            await observer.handler({ type: event.type, payload: event.payload }, context);
          },
        });
      }
    }
    
    // Register HTTP endpoints
    for (const endpoint of (skill.httpEndpoints ?? [])) {
      const key = `${endpoint.method}:${endpoint.path}`;
      if (this.httpRegistry.has(key)) {
        console.warn(`HTTP endpoint "${key}" already registered, overwriting`);
      }
      this.httpRegistry.set(key, endpoint);
    }
    
    // Register WebSocket endpoints
    for (const endpoint of (skill.wsEndpoints ?? [])) {
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
    for (const tool of (skill.tools ?? [])) {
      this.toolRegistry.delete(tool.name);
    }
    
    // Remove hooks
    for (const hook of (skill.hooks ?? [])) {
      const existing = this.hookRegistry.get(hook.lifecycle) || [];
      const hookIndex = existing.indexOf(hook);
      if (hookIndex !== -1) {
        existing.splice(hookIndex, 1);
      }
    }
    
    // Remove handoffs (from both registry and router)
    for (const handoff of (skill.handoffs ?? [])) {
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
   * Update agent capabilities based on loaded skills.
   *
   * Note on `built_in_tools` semantics: for agents, this lists all registered
   * tool names (what the agent exposes to callers). For LLM models/skills,
   * `built_in_tools` lists provider-native tools (web_search, code_sandbox, etc.).
   * The same field serves both contexts because `Capabilities` is unified.
   */
  protected updateCapabilities(): void {
    const provides: string[] = [];
    const endpoints: string[] = [];
    const builtInTools: string[] = [];
    
    for (const tool of this.toolRegistry.values()) {
      if (tool.provides) {
        provides.push(tool.provides);
      }
      builtInTools.push(tool.name);
    }
    
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
   * Mark a tool as overridden by an external tool (client-executed).
   * When the LLM calls this tool, it will be treated as external
   * and returned to the client instead of executed server-side.
   */
  overrideTool(name: string): void {
    this._overriddenTools.add(name);
  }

  /**
   * Check if a tool is registered internally (and not overridden by external).
   */
  private _isInternalTool(name: string): boolean {
    return this.toolRegistry.has(name) && !this._overriddenTools.has(name);
  }

  /**
   * Build conversation messages from initial UAMP events.
   */
  private _buildConversationFromEvents(events: ClientEvent[]): AgenticMessage[] {
    const messages: AgenticMessage[] = [];

    const pushOrMergeUser = (items: ContentItem[]) => {
      const last = messages[messages.length - 1];
      if (last && last.role === 'user') {
        last.content_items = [...(last.content_items || []), ...items];
      } else {
        messages.push({ role: 'user', content: null, content_items: items });
      }
    };

    for (const event of events) {
      if (event.type === 'session.create') {
        const e = event as SessionCreateEvent;
        if (e.session.instructions) {
          messages.push({ role: 'system', content: e.session.instructions });
        }
      } else if (event.type === 'input.text') {
        const e = event as InputTextEvent;
        const role = (e.role as 'user' | 'system') || 'user';
        if (role === 'system') {
          messages.push({ role: 'system', content: e.text });
        } else {
          pushOrMergeUser([{ type: 'text', text: e.text } as TextContent]);
        }
      } else if (event.type === 'input.image') {
        const e = event as InputImageEvent;
        pushOrMergeUser([ensureContentId({
          type: 'image',
          image: e.image,
          format: e.format,
          detail: e.detail,
          content_id: e.content_id,
        } as ImageContent)]);
      } else if (event.type === 'input.audio') {
        const e = event as InputAudioEvent;
        pushOrMergeUser([ensureContentId({
          type: 'audio',
          audio: e.audio,
          format: e.format,
          content_id: e.content_id,
        } as AudioContent)]);
      } else if (event.type === 'input.video') {
        const e = event as InputVideoEvent;
        pushOrMergeUser([ensureContentId({
          type: 'video',
          video: e.video,
          format: e.format,
          content_id: e.content_id,
        } as VideoContent)]);
      } else if (event.type === 'input.file') {
        const e = event as InputFileEvent;
        pushOrMergeUser([ensureContentId({
          type: 'file',
          file: e.file,
          filename: e.filename,
          mime_type: e.mime_type,
          content_id: e.content_id,
        } as FileContent)]);
      }
    }
    return messages;
  }

  /**
   * Extract tool calls from a response.done event's output.
   */
  private _extractToolCallsFromOutput(output: ContentItem[]): ToolCall[] {
    const toolCalls: ToolCall[] = [];
    for (const item of output) {
      if (item.type === 'tool_call' && item.tool_call) {
        toolCalls.push(item.tool_call);
      }
    }
    return toolCalls;
  }

  /**
   * Execute a single internal tool call and return the string result.
   */
  private async _executeInternalToolCall(tc: ToolCall): Promise<string | StructuredToolResult> {
    if (this.context.signal?.aborted) {
      return 'Tool execution cancelled: request was aborted';
    }

    let args: Record<string, unknown>;
    try {
      args = JSON.parse(tc.arguments);
    } catch {
      return `Error parsing tool arguments: ${tc.arguments}`;
    }

    const tool = this.toolRegistry.get(tc.name);
    if (!tool) {
      return `Tool not found: ${tc.name}`;
    }

    if (tool.scopes && tool.scopes.length > 0 && !this.context.hasScopes(tool.scopes)) {
      return `Insufficient permissions for tool: ${tc.name}`;
    }

    try {
      console.log(`[agent] executing tool: ${tc.name} args=${tc.arguments.slice(0, 500)}`);
      // Call tool.handler directly — processUAMP already manages before_tool/after_tool
      // hooks around this call. Going through executeTool() would fire hooks a second time.
      const result = await tool.handler(args, this.context);
      if (typeof result === 'string') return result;
      if (result && typeof result === 'object' && 'content_items' in result) {
        return result as StructuredToolResult;
      }
      return JSON.stringify(result);
    } catch (error) {
      if (this.context.signal?.aborted) {
        return 'Tool execution cancelled: request was aborted';
      }
      await this.runHooks('on_error', { error: error as Error, tool_name: tc.name });
      return `Tool execution error: ${(error as Error).message}`;
    }
  }

  /**
   * Process UAMP events through the agentic loop.
   * 
   * The loop invokes the LLM handoff, inspects the response for tool calls,
   * executes internal tools server-side, feeds results back to the LLM,
   * and repeats until the LLM produces a final text response or
   * returns external tool calls for the client.
   * 
   * Tool classification:
   * - Internal tools: registered in toolRegistry and NOT overridden.
   *   Executed server-side; results fed back to LLM for continuation.
   * - External tools: not registered, or overridden via overrideTool().
   *   Loop breaks and returns these tool calls to the client for execution.
   * - Mixed: internal tools execute first, then external tools are returned.
   */
  async *processUAMP(events: ClientEvent[]): AsyncGenerator<ServerEvent, void, unknown> {
    const handoff = this.getBestHandoff();
    const signal = this.context.signal;

    if (!handoff) {
      yield createResponseErrorEvent(
        'no_handoff',
        'No LLM skill available to process request'
      );
      return;
    }

    // Process client capabilities
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.client_capabilities) {
          (this.context as ContextImpl).setClientCapabilities(createEvent.client_capabilities);
        }
      }
    }

    // Run on_connection + before_handoff hooks
    await this.runHooks('on_connection', { metadata: {} });

    if (signal?.aborted) {
      await this.runHooks('finalize_connection', {});
      yield createResponseErrorEvent('aborted', 'Request was cancelled');
      return;
    }

    const beforeResult = await this.runHooks('before_handoff', {
      handoff_target: handoff.name,
    });

    if (beforeResult?.abort) {
      await this.runHooks('finalize_connection', {});
      yield createResponseErrorEvent(
        'handoff_aborted',
        beforeResult.abort_reason || 'Handoff aborted by hook'
      );
      return;
    }

    // Use pre-built conversation if set by run()/runStreaming() (preserves assistant/tool messages),
    // otherwise fall back to building from UAMP events (direct processUAMP callers).
    const prebuilt = this.context.get<AgenticMessage[]>('_initial_conversation');
    const conversation: AgenticMessage[] = prebuilt && prebuilt.length > 0
      ? [...prebuilt]
      : this._buildConversationFromEvents(events);
    this.context.delete('_initial_conversation');

    let iteration = 0;
    const collectedContentItems: ContentItem[] = [];

    while (iteration < this.maxToolIterations) {
      if (signal?.aborted) {
        yield createResponseErrorEvent('aborted', 'Request was cancelled');
        break;
      }

      iteration++;

      // Set conversation, tools, and skills in context so handoff/payment skills can access them
      this.context.set('_agentic_messages', conversation);
      this.context.set('_agentic_tools', this.getToolDefinitions());
      this.context.set('_skills', this.skills);

      // Run before_llm_call hooks
      await this.runHooks('before_llm_call', {
        metadata: { conversation_length: conversation.length },
        iteration,
      });

      // Call handoff and collect all events
      const collected: ServerEvent[] = [];
      try {
        for await (const event of handoff.handler(events, this.context)) {
          collected.push(event);

          // Fire on_chunk hook for each streaming delta
          if (event.type === 'response.delta') {
            const delta = (event as unknown as { delta: ResponseDelta }).delta;
            await this.runHooks('on_chunk', { chunk: delta, event });
          }
        }
      } catch (error) {
        await this.runHooks('on_error', { error: error as Error });
        yield createResponseErrorEvent(
          'handoff_error',
          (error as Error).message
        );
        await this.runHooks('after_handoff', { handoff_target: handoff.name });
        return;
      }

      // Run after_llm_call hooks
      await this.runHooks('after_llm_call', {
        metadata: { events_collected: collected.length },
        iteration,
      });

      // Find response.done to inspect tool calls
      const doneEvent = collected.find(
        (e): e is ResponseDoneEvent & ServerEvent => e.type === 'response.done'
      ) as (ResponseDoneEvent & { response: { output: ContentItem[]; usage?: UsageStats; id: string; status: string } }) | undefined;

      if (!doneEvent) {
        // No response.done -- yield everything (likely an error event)
        for (const event of collected) yield event;
        break;
      }

      const toolCalls = this._extractToolCallsFromOutput(doneEvent.response.output);

      // No tool calls -- LLM is done. Yield all events and exit.
      if (toolCalls.length === 0) {
        for (const event of collected) {
          if (event.type === 'response.done' && collectedContentItems.length > 0) {
            const done = event as { response: { output: ContentItem[] } };
            yield {
              ...event,
              response: { ...done.response, output: [...done.response.output, ...collectedContentItems] },
            } as ServerEvent;
          } else {
            yield event;
          }
        }
        break;
      }

      // Classify tool calls
      const internalCalls: ToolCall[] = [];
      const externalCalls: ToolCall[] = [];

      for (const tc of toolCalls) {
        if (this._isInternalTool(tc.name)) {
          internalCalls.push(tc);
        } else {
          externalCalls.push(tc);
        }
      }

      // If ANY external tool calls exist: execute internal tools first, then return external to client
      if (externalCalls.length > 0) {
        for (const tc of internalCalls) {
          let parsedArgs: Record<string, unknown> = {};
          try { parsedArgs = JSON.parse(tc.arguments || '{}'); } catch { /* proceed */ }

          this.context.set('tool_call', { id: tc.id, function: { name: tc.name, arguments: tc.arguments } });
          this.context.set('tool_name', tc.name);
          this.context.delete('tool_skipped');

          const beforeToolResult = await this.runHooks('before_tool', { tool_name: tc.name, tool_params: parsedArgs });
          const beforeToolcallResult = await this.runHooks('before_toolcall', { tool_name: tc.name, tool_params: parsedArgs });
          const toolSkipped = this.context.get<boolean>('tool_skipped');

          if (!beforeToolResult?.abort && !beforeToolcallResult?.abort && !toolSkipped) {
            const result = await this._executeInternalToolCall(tc);
            this.context.set('tool_result', result);
            await this.runHooks('after_tool', { tool_name: tc.name, tool_result: result });
            await this.runHooks('after_toolcall', { tool_name: tc.name, tool_result: result });
          }

          this.context.delete('tool_call');
          this.context.delete('tool_name');
          this.context.delete('tool_result');
          this.context.delete('tool_skipped');
        }

        // Yield all collected events, but filter response.done output to only external tool calls + text
        for (const event of collected) {
          if (event.type === 'response.done') {
            const filteredOutput = doneEvent.response.output.filter(item => {
              if (item.type === 'tool_call' && item.tool_call) {
                return externalCalls.some(ext => ext.id === item.tool_call!.id);
              }
              return true;
            });
            yield {
              ...event,
              response: { ...doneEvent.response, output: [...filteredOutput, ...collectedContentItems] },
            } as ServerEvent;
          } else {
            yield event;
          }
        }
        break;
      }

      // All tool calls are internal -- execute and continue loop

      // Yield streaming deltas from this iteration. Internal tool_call deltas
      // are suppressed here because processUAMP emits them during execution.
      // Platform tool_call/tool_result/file deltas (already handled by the
      // proxy) are forwarded so the browser can render their UI.
      const internalCallIds = new Set(internalCalls.map(tc => tc.id));
      for (const event of collected) {
        if (event.type === 'response.delta') {
          const delta = (event as unknown as { delta: ResponseDelta }).delta;
          if (delta.type === 'text' || delta.type === 'tool_result' || delta.type === 'tool_progress' || delta.type === 'file') {
            yield event;
          } else if (delta.type === 'tool_call' && delta.tool_call) {
            if (!internalCallIds.has(delta.tool_call.id)) {
              console.log(`[agent] yield non-internal tool_call from collected: name=${delta.tool_call.name} id=${delta.tool_call.id}`);
              yield event;
            } else {
              console.log(`[agent] suppressed internal tool_call from collected: name=${delta.tool_call.name} id=${delta.tool_call.id}`);
            }
          }
        } else if (event.type === 'thinking' || event.type === 'progress') {
          yield event;
        }
      }

      // Add assistant message (with tool calls) to conversation
      const assistantText = doneEvent.response.output
        .filter(item => item.type === 'text')
        .map(item => item.text || '')
        .join('');

      conversation.push({
        role: 'assistant',
        content: assistantText || null,
        tool_calls: toolCalls.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.name, arguments: tc.arguments },
        })),
      });

      // Execute each internal tool and add results to conversation
      for (const tc of internalCalls) {
        let parsedArgs: Record<string, unknown> = {};
        try {
          parsedArgs = JSON.parse(tc.arguments || '{}');
        } catch {
          // proceed with empty params — execution will also fail gracefully
        }

        // Set tool_call in context for Python-compatible hooks (PaymentSkill reads these)
        const toolCallDict = { id: tc.id, function: { name: tc.name, arguments: tc.arguments } };
        this.context.set('tool_call', toolCallDict);
        this.context.set('tool_name', tc.name);
        this.context.delete('tool_skipped');

        const beforeToolResult = await this.runHooks('before_tool', {
          tool_name: tc.name,
          tool_params: parsedArgs,
        });

        // Also fire before_toolcall (Python-compatible hook name)
        const beforeToolcallResult = await this.runHooks('before_toolcall', {
          tool_name: tc.name,
          tool_params: parsedArgs,
        });

        const toolSkipped = this.context.get<boolean>('tool_skipped');
        if (beforeToolResult?.abort || beforeToolcallResult?.abort || toolSkipped) {
          const reason = beforeToolResult?.abort_reason
            || beforeToolcallResult?.abort_reason
            || this.context.get<string>('tool_result')
            || 'Tool execution blocked by hook';
          conversation.push({
            role: 'tool',
            content: reason,
            tool_call_id: tc.id,
            name: tc.name,
          });
          this.context.delete('tool_call');
          this.context.delete('tool_name');
          this.context.delete('tool_skipped');
          continue;
        }

        // Notify client that tool execution is starting
        console.log(`[agent] yield internal tool_call before execution: name=${tc.name} id=${tc.id}`);
        yield {
          type: 'response.delta',
          event_id: generateEventId(),
          delta: { type: 'tool_call', tool_call: { id: tc.id, name: tc.name, arguments: tc.arguments } },
        } as unknown as ServerEvent;

        // Run tool with an AsyncQueue so streaming tools (e.g. delegate)
        // can push progress events that we yield in real-time.
        const progressQueue = new AsyncQueue<ServerEvent>();
        this.context.set('_toolProgressFn', (callId: string, text: string) => {
          progressQueue.push({
            type: 'response.delta',
            event_id: generateEventId(),
            delta: { type: 'tool_progress', tool_progress: { call_id: callId, text } },
          } as unknown as ServerEvent);
        });

        const resultPromise = this._executeInternalToolCall(tc).then((r) => {
          this.context.delete('_toolProgressFn');
          progressQueue.close();
          return r;
        });

        for await (const progressEvent of progressQueue) {
          yield progressEvent;
        }
        const result = await resultPromise;

        // Set tool_result in context for Python-compatible hooks
        this.context.set('tool_result', result);

        const afterResult = await this.runHooks('after_tool', {
          tool_name: tc.name,
          tool_result: result,
        });
        const finalResult = afterResult?.tool_result ?? result;

        // Also fire after_toolcall (Python-compatible hook name)
        await this.runHooks('after_toolcall', {
          tool_name: tc.name,
          tool_result: finalResult,
        });

        const resultText = typeof finalResult === 'string' ? finalResult : (finalResult as StructuredToolResult).text;
        const resultItems = (typeof finalResult === 'object' && finalResult !== null && 'content_items' in (finalResult as object))
          ? (finalResult as StructuredToolResult).content_items
          : undefined;
        console.log(`[agent] tool ${tc.name} result: hasContentItems=${!!resultItems} count=${resultItems?.length ?? 0}`);

        if (resultItems) {
          collectedContentItems.push(...resultItems);
        }

        // Notify client of tool result (include content_items so persisters can extract media)
        yield {
          type: 'response.delta',
          event_id: generateEventId(),
          delta: { type: 'tool_result', tool_result: { call_id: tc.id, result: resultText, content_items: resultItems } },
        } as unknown as ServerEvent;

        // Clean up context
        this.context.delete('tool_call');
        this.context.delete('tool_name');
        this.context.delete('tool_result');
        this.context.delete('tool_skipped');

        const toolMsg = {
          role: 'tool' as const,
          content: resultText,
          content_items: resultItems,
          tool_call_id: tc.id,
          name: tc.name,
        };
        console.log(`[agent] pushing tool result to conversation: tool=${tc.name} hasContentItems=${!!resultItems} count=${resultItems?.length ?? 0} contentItemTypes=${resultItems?.map(ci => ci.type).join(',') ?? 'none'}`);
        conversation.push(toolMsg);
      }

      // Loop continues: handoff will be called again with updated conversation in context
    }

    if (iteration >= this.maxToolIterations) {
      yield createResponseErrorEvent(
        'max_iterations',
        `Agent reached maximum tool iterations (${this.maxToolIterations}). Stopping to prevent infinite loop.`
      );
    }

    // Run after_handoff + finalize_connection hooks
    await this.runHooks('after_handoff', {
      handoff_target: handoff.name,
    });
    await this.runHooks('finalize_connection', {});
  }
  
  /**
   * Run with messages (convenience method)
   */
  async run(messages: Message[], options: RunOptions = {}): Promise<RunResponse> {
    if (options.signal) {
      (this.context as ContextImpl).signal = options.signal;
    }

    // Convert messages to UAMP events
    const events: ClientEvent[] = [];

    const runExtensions: Record<string, unknown> = {};
    if (options.paymentToken) {
      runExtensions['X-Payment-Token'] = options.paymentToken;
    }

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
        ...(Object.keys(runExtensions).length > 0 && { extensions: runExtensions }),
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
    
    // Set full conversation in context so processUAMP preserves assistant/tool messages
    const initialConv: AgenticMessage[] = [];
    const sysInstr = options.instructions || this.instructions;
    if (sysInstr) {
      initialConv.push({ role: 'system', content: sysInstr });
    }
    for (const msg of messages) {
      initialConv.push({
        role: msg.role as AgenticMessage['role'],
        content: msg.content || null,
        content_items: msg.content_items,
        tool_calls: msg.tool_calls?.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.name, arguments: tc.arguments },
        })),
        tool_call_id: msg.tool_call_id,
      });
    }
    this.context.set('_initial_conversation', initialConv);

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
    if (options.signal) {
      (this.context as ContextImpl).signal = options.signal;
    }

    // Convert messages to UAMP events
    const events: ClientEvent[] = [];

    const sessionExtensions: Record<string, unknown> = {};
    if (options.paymentToken) {
      sessionExtensions['X-Payment-Token'] = options.paymentToken;
    }

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
        ...(Object.keys(sessionExtensions).length > 0 && { extensions: sessionExtensions }),
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
    
    // Set full conversation in context so processUAMP preserves assistant/tool messages
    const initialConvS: AgenticMessage[] = [];
    const sysInstrS = options.instructions || this.instructions;
    if (sysInstrS) {
      initialConvS.push({ role: 'system', content: sysInstrS });
    }
    for (const msg of messages) {
      initialConvS.push({
        role: msg.role as AgenticMessage['role'],
        content: msg.content || null,
        content_items: msg.content_items,
        tool_calls: msg.tool_calls?.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.name, arguments: tc.arguments },
        })),
        tool_call_id: msg.tool_call_id,
      });
    }
    this.context.set('_initial_conversation', initialConvS);

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
        const toolResult = (delta as unknown as { tool_result?: { call_id: string; result: string; is_error?: boolean; content_items?: ContentItem[] } }).tool_result;
        if (toolResult) {
          yield {
            type: 'tool_result',
            tool_result: {
              call_id: toolResult.call_id,
              result: toolResult.result,
              is_error: toolResult.is_error,
              content_items: toolResult.content_items,
            },
          };
        }
        const toolProgress = (delta as unknown as { tool_progress?: { call_id: string; text: string } }).tool_progress;
        if (toolProgress) {
          yield {
            type: 'tool_progress',
            tool_progress: {
              call_id: toolProgress.call_id,
              text: toolProgress.text,
            },
          };
        }
        const fileDelta = delta as unknown as { type?: string; content_id?: string; filename?: string };
        if (fileDelta.type === 'file' && fileDelta.content_id) {
          console.log(`[agent] runStreaming: yielding file chunk content_id=${fileDelta.content_id} filename=${fileDelta.filename}`);
          yield { type: 'file', ...(delta as unknown as Record<string, unknown>) } as StreamChunk;
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
      } else if (event.type === 'thinking') {
        const t = event as { content?: string; stage?: string };
        yield {
          type: 'thinking',
          thinking: { content: t.content ?? '', stage: t.stage },
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
