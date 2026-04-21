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
  Prompt,
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
  DisplayHint,
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

import { ensureContentId, inferDisplayHint, isMediaContent } from '../uamp/content';

import { createContext, ContextImpl } from './context';
import { MessageRouter, type TransportSink, type UAMPEvent, type RouterContext } from './router';
import { getObservers, getPrompts } from './decorators';

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
 * Format an extracted text blob for `read_content` so the LLM gets a
 * structured, paginated, line-numbered view that mirrors `text_editor view`
 * and adds regex `search` with configurable context windows + pagination.
 *
 * Exported for unit tests; production callers go through the `read_content`
 * tool handler.
 */
export interface FormatExtractedTextOptions {
  filename: string;
  text: string;
  totalLines: number;
  byteSize: number;
  search?: string;
  viewRange?: [number, number];
  before?: number;
  after?: number;
  offset?: number;
  limit?: number;
}

export function formatExtractedText(opts: FormatExtractedTextOptions): string {
  const { filename, text, totalLines, byteSize } = opts;
  const lines = text.split('\n');
  const padW = String(totalLines || 1).length;
  const numberLine = (i1: number, raw: string) => `${String(i1).padStart(padW, ' ')}\t${raw}`;
  const sizeStr = formatBytesShort(byteSize);

  // search wins over view_range when both are provided.
  if (typeof opts.search === 'string' && opts.search.length > 0) {
    return formatSearch({
      filename,
      lines,
      totalLines,
      sizeStr,
      pattern: opts.search,
      before: clampInt(opts.before ?? 2, 0, 20),
      after: clampInt(opts.after ?? 2, 0, 20),
      offset: Math.max(0, opts.offset ?? 0),
      limit: clampInt(opts.limit ?? 30, 1, 200),
      viewRangeProvided: !!opts.viewRange,
      numberLine,
    });
  }

  if (opts.viewRange) {
    const [s, e] = opts.viewRange;
    const start = Math.max(1, Math.min(s, totalLines || 1));
    const end = Math.max(start, Math.min(e, totalLines || 1));
    const slice = lines.slice(start - 1, end);
    const numbered = slice.map((l, i) => numberLine(start + i, l)).join('\n');
    const header = `${filename} (${totalLines} lines, ${sizeStr}; showing ${start}-${end})`;
    return `${header}\n${numbered}`;
  }

  // Default: head 200 + tail 50, with a middle elision marker.
  const HEAD = 200;
  const TAIL = 50;
  if (totalLines <= HEAD + TAIL) {
    const numbered = lines.map((l, i) => numberLine(i + 1, l)).join('\n');
    const header = `${filename} (${totalLines} lines, ${sizeStr})`;
    return `${header}\n${numbered}`;
  }
  const head = lines.slice(0, HEAD).map((l, i) => numberLine(i + 1, l));
  const tailStart = totalLines - TAIL + 1;
  const tail = lines.slice(tailStart - 1, totalLines).map((l, i) => numberLine(tailStart + i, l));
  const elided = totalLines - HEAD - TAIL;
  const header = `${filename} (${totalLines} lines, ${sizeStr}; showing 1-${HEAD} + tail ${TAIL})`;
  return `${header}\n${head.join('\n')}\n... ${elided} lines elided; pass view_range or search to drill in ...\n${tail.join('\n')}`;
}

function formatBytesShort(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function clampInt(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, Math.floor(v)));
}

function formatSearch(args: {
  filename: string;
  lines: string[];
  totalLines: number;
  sizeStr: string;
  pattern: string;
  before: number;
  after: number;
  offset: number;
  limit: number;
  viewRangeProvided: boolean;
  numberLine: (i1: number, raw: string) => string;
}): string {
  const { filename, lines, totalLines, sizeStr, pattern, before, after, offset, limit, viewRangeProvided, numberLine } = args;

  let re: RegExp;
  try {
    re = new RegExp(pattern, 'gm');
  } catch (err) {
    return `Invalid regex: ${(err as Error).message}. Pattern: "${pattern}".`;
  }

  // Collect ALL hits (1-indexed line numbers) so the total count is accurate
  // across pages. Reset lastIndex per-line to avoid stateful matches across
  // line boundaries (we want a hit per matching line, not per match).
  const hitLines: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    re.lastIndex = 0;
    if (re.test(lines[i])) hitLines.push(i + 1);
  }
  const total = hitLines.length;

  const overrideNote = viewRangeProvided ? '; view_range ignored' : '';

  if (total === 0) {
    return `${filename} (${totalLines} lines, ${sizeStr}; search=/${pattern}/gm, 0 hits${overrideNote})\nNo matches. Try a broader pattern or pass view_range to browse.`;
  }

  const start = Math.min(offset, total);
  const end = Math.min(offset + limit, total);
  const pageHits = hitLines.slice(start, end);

  // Build per-hit windows then merge overlapping/adjacent ones (rg -A/-B
  // semantics): a window is [winStart, winEnd, hitLineSet]. Two windows
  // merge when winA.end + 1 >= winB.start.
  type Win = { start: number; end: number; hits: Set<number> };
  const windows: Win[] = pageHits.map((h) => ({
    start: Math.max(1, h - before),
    end: Math.min(totalLines, h + after),
    hits: new Set<number>([h]),
  }));
  const merged: Win[] = [];
  for (const w of windows) {
    const last = merged[merged.length - 1];
    if (last && w.start <= last.end + 1) {
      last.end = Math.max(last.end, w.end);
      for (const h of w.hits) last.hits.add(h);
    } else {
      merged.push({ start: w.start, end: w.end, hits: new Set(w.hits) });
    }
  }

  const blocks: string[] = [];
  for (const w of merged) {
    const block: string[] = [];
    for (let ln = w.start; ln <= w.end; ln++) {
      const isHit = w.hits.has(ln);
      const raw = lines[ln - 1] ?? '';
      // Mark hit lines with a `:` separator instead of `\t` so they're easy to
      // scan in a long block (mirrors ripgrep's distinction between `-` ctx
      // and `:` match prefixes).
      block.push(isHit ? `${String(ln).padStart(String(totalLines || 1).length, ' ')}:${raw}` : numberLine(ln, raw));
    }
    blocks.push(block.join('\n'));
  }

  const header = `${filename} (${totalLines} lines, ${sizeStr}; search=/${pattern}/gm, hits ${start + 1}-${end} of ${total}, before=${before}, after=${after}, limit=${limit}${overrideNote})`;
  const body = blocks.join('\n--\n');
  const remaining = total - end;
  const footer = remaining > 0
    ? `\n... ${remaining} more hits; pass offset=${end} for the next page (or raise limit, max 200).`
    : '';
  return `${header}\n${body}${footer}`;
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

  /** Aggregated prompts from all skills, sorted by priority */
  protected promptRegistry: Prompt[] = [];
  
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
    this.maxToolIterations = config.maxToolIterations ?? 50;
    
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

    // Register prompts from skill (explicit property + @prompt-decorated methods)
    for (const p of (skill.prompts ?? [])) {
      this.promptRegistry.push(p);
    }
    const decoratedPrompts = getPrompts(skill);
    for (const [methodName, promptMeta] of decoratedPrompts) {
      const handler = (skill as unknown as Record<string, unknown>)[methodName];
      if (typeof handler === 'function') {
        this.promptRegistry.push({
          name: promptMeta.name || methodName,
          priority: promptMeta.priority ?? 50,
          scope: promptMeta.scope ?? 'all',
          handler: handler.bind(skill) as Prompt['handler'],
        });
      }
    }
    this.promptRegistry.sort((a, b) => a.priority - b.priority);
    
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

    // Remove prompts contributed by this skill
    const skillPromptNames = new Set((skill.prompts ?? []).map(p => p.name));
    const decoratedPrompts = getPrompts(skill);
    for (const [methodName, meta] of decoratedPrompts) {
      skillPromptNames.add(meta.name || methodName);
    }
    this.promptRegistry = this.promptRegistry.filter(p => !skillPromptNames.has(p.name));
    
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
   *
   * System-prompt injection contract: agents reaching this path (direct
   * `processUAMP` callers — delegated sub-chats, A2A, portal transport) must
   * carry their own `this.instructions` into the conversation, because the
   * proxy / LLM adapter will not synthesize one. Any inbound system hints from
   * the caller (`session.instructions`, `input.text` with `role === 'system'`)
   * are appended *after* the agent's base instructions so callers can
   * supplement context without silently replacing the platform prompt.
   */
  private _buildConversationFromEvents(events: ClientEvent[]): AgenticMessage[] {
    const messages: AgenticMessage[] = [];
    const inboundInstructions: string[] = [];

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
          inboundInstructions.push(e.session.instructions);
        }
      } else if (event.type === 'input.text') {
        const e = event as InputTextEvent;
        const role = (e.role as 'user' | 'system') || 'user';
        if (role === 'system') {
          inboundInstructions.push(e.text);
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

    const systemContent = [this.instructions, ...inboundInstructions]
      .filter((s): s is string => typeof s === 'string' && s.length > 0)
      .join('\n\n');
    if (systemContent) {
      messages.unshift({ role: 'system', content: systemContent });
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
      if (
        result && typeof result === 'object'
        && ('content_items' in result || '_post_messages' in result)
      ) {
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

    // Conversation seeding has two flavors, used by different callers:
    //
    //  * `_initial_conversation` — full conversation (system + history + the
    //    new user turn). Replaces events-derived messages entirely. This is
    //    what `run()` / `runStreaming()` use because they already know every
    //    turn at call time.
    //
    //  * `_history_conversation` — prior chat-DB history only. The new turn
    //    still flows in via `events` (`input.text` etc.) and gets appended.
    //    Transport skills use this when an incoming WS connection carries
    //    `X-Chat-Id` so the delegate sub-agent sees prior text_editor / bash
    //    / delegate turns instead of starting cold.
    //
    // If both are set, `_initial_conversation` wins (preserves existing
    // semantics for `run()` callers).
    const prebuilt = this.context.get<AgenticMessage[]>('_initial_conversation');
    const history = this.context.get<AgenticMessage[]>('_history_conversation');
    let conversation: AgenticMessage[];
    if (prebuilt && prebuilt.length > 0) {
      conversation = [...prebuilt];
    } else {
      const fromEvents = this._buildConversationFromEvents(events);
      if (history && history.length > 0) {
        // Drop the events-built leading system message if history already
        // carries one; otherwise keep both. This avoids duplicate system
        // turns when the loader returns a chat that includes system rows.
        const histHasSystem = history[0]?.role === 'system';
        const trimmed = histHasSystem && fromEvents[0]?.role === 'system'
          ? fromEvents.slice(1)
          : fromEvents;
        conversation = [...history, ...trimmed];
      } else {
        conversation = fromEvents;
      }
    }
    this.context.delete('_initial_conversation');
    this.context.delete('_history_conversation');

    let iteration = 0;
    const collectedContentItems: ContentItem[] = [];
    const recentToolCalls: Array<{ key: string; count: number }> = [];
    const presentedIds = new Set<string>();

    // Index every content_id reachable from this turn: historical conversation messages
    // (so present/read_content can re-display or re-load prior media) plus items collected
    // from tool results in the current loop. LATEST-seen wins so that follow-up edits
    // (text_editor str_replace on a previously created file, delegate sub-agent edits,
    // etc.) overwrite the stale `metadata.command='create'` carried on the original
    // create-time content_item. Otherwise present() re-emits the file with the original
    // command, the parent's persisted message records command='create', and the
    // DocumentChip badge shows "Created" forever even after the file has been edited.
    // Walk newest -> oldest and stop at first match per content_id.
    const indexAvailableContent = (): Map<string, ContentItem> => {
      const out = new Map<string, ContentItem>();
      for (let i = collectedContentItems.length - 1; i >= 0; i--) {
        const ci = collectedContentItems[i]!;
        const cid = (ci as { content_id?: string }).content_id;
        if (cid && !out.has(cid)) out.set(cid, ci);
      }
      for (let mi = conversation.length - 1; mi >= 0; mi--) {
        const items = (conversation[mi] as { content_items?: ContentItem[] }).content_items;
        if (Array.isArray(items)) {
          for (let ci = items.length - 1; ci >= 0; ci--) {
            const item = items[ci]!;
            const cid = (item as { content_id?: string }).content_id;
            if (cid && !out.has(cid)) out.set(cid, item);
          }
        }
      }
      return out;
    };

    // `present` and `read_content` are the universal content-propagation
    // mechanism every agent needs:
    //   - `read_content` loads media into the agent's own LLM context
    //     (purely agent-internal; nothing to do with the client).
    //   - `present` marks an item as a deliverable AND emits a streaming
    //     `response.delta` so callers can render in-flight (works for any
    //     caller, browser or another agent).
    // Both are therefore registered unconditionally. The `supports_rich_display`
    // capability is still consulted further down to decide whether the final
    // `response.done.output` is **filtered to presented items only** (browser
    // clients want selective output) or whether **all collected items
    // auto-promote** (safety net for non-browser callers like delegate
    // sub-chats, in case the sub-agent forgets to call `present`).
    const hasPresentTool = !!(this.context.client_capabilities as Capabilities | undefined)?.supports_rich_display;
    {
      this.toolRegistry.set('present', {
        name: 'present',
        enabled: true,
        description: 'Display a piece of generated content (image, video, audio, 3D model, HTML page, file) to the user. Call once per content_id you want shown; content not passed through present is not rendered. The content_id is a UUID returned by a prior tool result (it appears as `content_id=<uuid>` or after `Media content_ids:`). Copy the exact UUID — do not guess, abbreviate, or fabricate IDs. Note: "HTML pages" means a standalone .html content item; do not pass raw HTML markup as a content_id, and write plain text or Markdown in your message body, not HTML tags.',
        parameters: {
          type: 'object',
          properties: {
            content_id: { type: 'string', description: 'A UUID content_id returned by a prior tool result. Must be an exact UUID — do not guess or fabricate.' },
            display_as: {
              type: 'string',
              enum: ['inline', 'attachment', 'sandbox'],
              description: 'Optional. inline: render in message (images, video, audio). attachment: downloadable chip (files). sandbox: interactive iframe (HTML). Auto-inferred from content type if omitted.',
            },
            caption: { type: 'string', description: 'Optional caption displayed with the content' },
          },
          required: ['content_id'],
        },
        handler: async (args: Record<string, unknown>) => {
          const id = args.content_id as string;
          const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
          if (!UUID_RE.test(id)) {
            const allItems = indexAvailableContent();
            const availableIds = [...allItems.keys()];
            return `Invalid content_id "${id}" — must be a UUID from a prior tool result (look for "content_id=..." in [Available ...] markers, NOT a filename). Available content_ids: ${availableIds.length > 0 ? availableIds.join(', ') : 'none'}.`;
          }
          const allItems = indexAvailableContent();
          const item = allItems.get(id);
          if (!item) {
            const availableIds = [...allItems.keys()];
            return `Content not found: "${id}". The content_id must be a UUID from tool results (not a filename). Available content_ids: ${availableIds.length > 0 ? availableIds.join(', ') : 'none'}. If this content came from an external URL, use save_content first to get a content_id.`;
          }
          const hint: DisplayHint = (args.display_as as DisplayHint) || inferDisplayHint(item.type);
          (item as { display_hint?: DisplayHint }).display_hint = hint;
          if (args.caption) (item as { caption?: string }).caption = args.caption as string;
          presentedIds.add(id);

          // If this is a historical item not yet in this turn's collection, splice it in so
          // the standard output/persistence/broadcast path includes it. The storage blob is
          // keyed by content_id, so this only creates a new content_items reference row,
          // not a duplicate binary.
          const alreadyCollected = collectedContentItems.some(
            (ci) => (ci as { content_id?: string }).content_id === id,
          );
          if (!alreadyCollected) {
            collectedContentItems.push(item);
          }

          // Push a content delta event directly to the caller via _presentDeltaFn
          const presentDeltaFn = this.context.get<(event: ServerEvent) => void>('_presentDeltaFn');
          if (presentDeltaFn) {
            const delta: Record<string, unknown> = { ...item, type: item.type };
            presentDeltaFn({
              type: 'response.delta',
              event_id: generateEventId(),
              delta,
            } as unknown as ServerEvent);
          }

          const dims = (item as { dimensions?: { width: number; height: number } }).dimensions;
          const desc = (item as { description?: string }).description || '';
          const filename = (item as { filename?: string }).filename;
          // Echo filename + content_id back so the model can unambiguously match
          // this success to the right item — otherwise a bare "Displayed text/html
          // to user." after several failed `text_editor create` attempts reads as
          // "something got presented but maybe not the file I was working on", and
          // the model loops back to recreate. Also state explicitly that no further
          // action is needed for this id.
          const label = filename ? `"${filename}" (${item.type})` : item.type;
          const dimStr = dims ? ` ${dims.width}x${dims.height}` : '';
          const descStr = desc ? ` — ${desc}` : '';
          return (
            `Displayed ${label}${dimStr} to the user (content_id=${id}).${descStr} ` +
            `The user can now see this content. Do not call present, text_editor, or any other tool ` +
            `to "create" or "show" content_id=${id} again — this id is done. ` +
            `Move on: write a brief reply to the user or call the next distinct tool.`
          );
        },
      });
    }

    // Register read_content() tool for explicit LLM media analysis.
    // Always registered — see note above; this is purely about the agent
    // loading content into its own context.
    {
      const providerModalities: Record<string, Set<string>> = {
        google:    new Set(['image', 'audio', 'video']),
        openai:    new Set(['image', 'audio']),
        anthropic: new Set(['image']),
        xai:       new Set(['image']),
        fireworks: new Set(['image']),
      };
      const currentProvider = this.model?.split('/')[0]?.toLowerCase() || '';
      const currentModalities = providerModalities[currentProvider];

      this.toolRegistry.set('read_content', {
        name: 'read_content',
        enabled: true,
        description: 'Load an existing content_id into your context.\nFor text-bearing files (code, HTML, markdown, txt, csv, json, log, PDF, DOCX): returns formatted text with 1-indexed line numbers and a header.\n  - view_range: [start, end]  show only those lines (preferred over loading the whole file before an edit).\n  - search:     JS regex (use plain text for literal matches). Returns matching lines with surrounding context, paginated.\n      before:   leading context lines per hit (default 2, max 20).\n      after:    trailing context lines per hit (default 2, max 20).\n      offset:   skip the first N hits (default 0).\n      limit:    max hits per call (default 30, max 200). Footer reports total hits and the next offset to use.\n  - default:    first 200 + last 50 lines + total count.\n  view_range and search are mutually exclusive (search wins).\nFor media (image/audio/video): attaches the item natively for analysis if the current model supports the modality; otherwise returns a modality error. Use present(content_id) instead if you only need to display content to the user. Do NOT call read_content on text files you just created/edited via text_editor (the contents are already in your prior tool call).\n\nSCOPE / WHEN TO USE: read_content addresses content by raw content_id and is subject to per-id ACL checks (creator / link / public). For files you can address by **path** (anything you created in this chat OR an attachment that arrived with a filename), prefer `text_editor view path="<path>" view_range=[a,b]` — it walks the chat tree, so it succeeds in delegate sub-chats where read_content by id is sometimes ACL-denied. Use read_content for items you ONLY have a content_id for (e.g. media just attached to you by another agent, or items returned in tool results without a path).\n\nREAD-FIRST WORKFLOW: before editing any existing file, read the relevant region first (`text_editor view view_range=` or `read_content search=` for big files); then apply a precise `text_editor str_replace`. Never re-create a file you intend to amend.',
        parameters: {
          type: 'object',
          properties: {
            content_id: { type: 'string', description: 'The UUID content_id to load.' },
            view_range: {
              type: 'array',
              items: { type: 'integer' },
              minItems: 2,
              maxItems: 2,
              description: 'For text files: [start, end] 1-indexed inclusive line range. Mutually exclusive with search.',
            },
            search: { type: 'string', description: 'For text files: JS regex pattern. Compiled with gm flags. Plain text matches literally if it has no metachars.' },
            before: { type: 'integer', description: 'Leading context lines per search hit (default 2, max 20).' },
            after: { type: 'integer', description: 'Trailing context lines per search hit (default 2, max 20).' },
            offset: { type: 'integer', description: 'Skip the first N search hits before paging (default 0).' },
            limit: { type: 'integer', description: 'Max search hits per call (default 30, max 200).' },
          },
          required: ['content_id'],
        },
        handler: async (args: Record<string, unknown>) => {
          const id = args.content_id as string;

          // Param validation up front so the LLM gets a clear error before we resolve anything.
          const viewRangeArg = args.view_range as unknown;
          const searchArg = args.search as unknown;
          const beforeArg = args.before as unknown;
          const afterArg = args.after as unknown;
          const offsetArg = args.offset as unknown;
          const limitArg = args.limit as unknown;

          const isNonNegInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 0;
          const isPosInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v > 0;

          if (beforeArg !== undefined && !isNonNegInt(beforeArg)) {
            return `Invalid argument: before must be a non-negative integer, got ${JSON.stringify(beforeArg)}.`;
          }
          if (afterArg !== undefined && !isNonNegInt(afterArg)) {
            return `Invalid argument: after must be a non-negative integer, got ${JSON.stringify(afterArg)}.`;
          }
          if (offsetArg !== undefined && !isNonNegInt(offsetArg)) {
            return `Invalid argument: offset must be a non-negative integer, got ${JSON.stringify(offsetArg)}.`;
          }
          if (limitArg !== undefined && !isPosInt(limitArg)) {
            return `Invalid argument: limit must be a positive integer, got ${JSON.stringify(limitArg)}.`;
          }
          if (viewRangeArg !== undefined) {
            if (!Array.isArray(viewRangeArg) || viewRangeArg.length !== 2 || !viewRangeArg.every((n) => Number.isInteger(n) && (n as number) >= 1)) {
              return `Invalid argument: view_range must be a 2-element array of positive integers [start, end], got ${JSON.stringify(viewRangeArg)}.`;
            }
            if ((viewRangeArg[0] as number) > (viewRangeArg[1] as number)) {
              return `Invalid argument: view_range start (${viewRangeArg[0]}) must be <= end (${viewRangeArg[1]}).`;
            }
          }

          const allItems = indexAvailableContent();
          let item = allItems.get(id);

          // DB fallback: when the content_id isn't already projected into our
          // conversation (e.g. the user references a file from a different
          // chat they can access, or the message it came from was pruned),
          // ask the runtime to resolve by id. The runtime applies an
          // ACL check (canAccessContent) before returning anything.
          if (!item) {
            const resolveById = this.context.get<(contentId: string, callerUserId?: string) => Promise<ContentItem | null>>('_resolveContentById');
            if (resolveById) {
              const callerUserId = this.context.auth?.user_id;
              try {
                const resolved = await resolveById(id, callerUserId);
                if (resolved) item = resolved;
              } catch (err) {
                console.warn(`[read_content] _resolveContentById threw for id=${id}:`, (err as Error).message);
              }
            }
          }

          if (!item) {
            const availableIds = [...allItems.keys()];
            const availableHint = availableIds.length > 0
              ? `Available content_ids in this chat: ${availableIds.join(', ')}.`
              : 'No other content_ids are visible in this chat.';
            return (
              `Content not found: "${id}". The id may be invalid, or you may not have access ` +
              `(the runtime ACL-checked DB lookup also returned nothing). Checks to try: ` +
              `(a) verify the id with the user — UUIDs are 36 hex chars; filenames and short prefixes ` +
              `are not accepted; (b) if the user just shared the id, confirm they own / have access in ` +
              `the source chat — they may need to re-share with you; (c) ls / lists every file already ` +
              `addressable in this chat with its content_id. ${availableHint}`
            );
          }

          const itemType = (item as { type?: string }).type || '';
          const itemFilename = (item as { filename?: string }).filename || `content_${id.slice(0, 8)}`;

          // Text-decodable branch: when the item is a `file` or `text` type,
          // ask the runtime to extract decoded text via _readContentText. If
          // it returns text, format it text_editor-view-style and return as
          // the tool result string (no native attach). If it returns null,
          // fall through to the native modality-gate branch below.
          if (itemType === 'file' || itemType === 'text') {
            const readText = this.context.get<(contentId: string, callerUserId?: string) => Promise<{ text: string; totalLines: number; byteSize: number; mimeType: string } | null>>('_readContentText');
            if (readText) {
              const callerUserId = this.context.auth?.user_id;
              let extracted: { text: string; totalLines: number; byteSize: number; mimeType: string } | null = null;
              try {
                extracted = await readText(id, callerUserId);
              } catch (err) {
                console.warn(`[read_content] _readContentText threw for id=${id}:`, (err as Error).message);
              }
              if (extracted) {
                return formatExtractedText({
                  filename: itemFilename,
                  text: extracted.text,
                  totalLines: extracted.totalLines,
                  byteSize: extracted.byteSize,
                  search: typeof searchArg === 'string' ? searchArg : undefined,
                  viewRange: viewRangeArg as [number, number] | undefined,
                  before: typeof beforeArg === 'number' ? beforeArg : undefined,
                  after: typeof afterArg === 'number' ? afterArg : undefined,
                  offset: typeof offsetArg === 'number' ? offsetArg : undefined,
                  limit: typeof limitArg === 'number' ? limitArg : undefined,
                });
              }
            }
          }

          // Native-attach branch: image/audio/video (or file/text where text
          // extraction failed, e.g. opaque binary mislabeled as file). Apply
          // the per-provider modality gate; file/text are always allowed
          // through here too — if extraction wasn't available the runtime is
          // already old or text wasn't extractable, and falling back to the
          // raw item is better than refusing.
          if (
            currentModalities
            && !currentModalities.has(itemType)
            && itemType !== 'text'
            && itemType !== 'file'
          ) {
            return `Cannot load ${itemType} content: this model (${currentProvider}) does not support ${itemType} analysis. The content metadata is already visible to you. To process ${itemType} with this model, use a transcription or conversion tool (if available) and re-read the resulting text.`;
          }

          // Splice into collectedContentItems so subsequent indexAvailableContent()
          // sweeps (e.g. when the LLM follows up with present()) find this item.
          const alreadyCollected = collectedContentItems.some(
            (ci) => (ci as { content_id?: string }).content_id === id,
          );
          if (!alreadyCollected) collectedContentItems.push(item);
          // CRITICAL: do NOT push the `_inline_for_llm` user message into
          // `conversation` here. The agent loop will append the `role: 'tool'`
          // result row immediately after this callback returns; if we push a
          // `role: 'user'` row first, Anthropic and OpenAI both reject the
          // next request with "tool_use without tool_result" because they
          // require the tool_result message to immediately follow the
          // assistant's tool_use turn. Instead, return the inline message via
          // `_post_messages` so the loop appends it AFTER the tool_result.
          return {
            text: `Content ${id} (${item.type}) loaded into your context. You can now see and analyze it.`,
            _post_messages: [{
              role: 'user' as const,
              content: `[Loaded content for analysis: ${id} (${item.type})]`,
              content_items: [item],
              _inline_for_llm: true,
            }],
          } as StructuredToolResult;
        },
      });
    }

    // Register save_content() tool when StoreMediaSkill is present (independent of present)
    const mediaSaver = this.context.get<{ save(base64: string, mimeType: string, meta?: Record<string, unknown>): Promise<string | { url: string; content_id: string }> }>('_media_saver');
    if (mediaSaver) {
      this.toolRegistry.set('save_content', {
        name: 'save_content',
        enabled: true,
        description: 'Save external content (URL or base64) to the user content library OR link an existing accessible file/folder into this chat at a local path. ALWAYS use the URL/base64 modes when you receive media from delegated agents, tools, or external sources. To work on an existing accessible file or folder by content_id, call save_content content_id=<uuid> as_path=<path> first; this links it into your chat at <path> (folders walked recursively, edits propagate to canonical bytes) so you can use text_editor/bash on it normally. Content produced by platform tools is auto-saved -- use this for external/unstructured sources or to import existing content.',
        parameters: {
          type: 'object',
          properties: {
            url: { type: 'string', description: 'URL of the content to save' },
            base64: { type: 'string', description: 'Base64-encoded content (alternative to url)' },
            content_id: { type: 'string', description: 'UUID of an existing accessible file or folder. When provided, links the item into this chat at as_path (folders walked recursively). Mutually exclusive with url and base64. Requires write access on the source content.' },
            as_path: { type: 'string', description: 'Local path under which to register the file or folder (e.g. /unicorn.html, /shared/). Required when content_id is provided; ignored otherwise.' },
            mime_type: { type: 'string', description: 'MIME type (e.g., image/png, video/mp4)' },
            description: { type: 'string', description: 'Human-readable description of the content' },
            filename: { type: 'string', description: 'Optional filename' },
          },
          required: ['description'],
        },
        handler: async (args: Record<string, unknown>) => {
          const urlArg = args.url as string | undefined;
          const base64Arg = args.base64 as string | undefined;
          const contentIdArg = args.content_id as string | undefined;
          const asPathArg = args.as_path as string | undefined;
          const descArg = args.description as string || '';

          // Link mode: import an existing accessible content_id at a local path.
          if (contentIdArg) {
            if (urlArg || base64Arg) {
              return 'Error: content_id is mutually exclusive with url/base64.';
            }
            if (!asPathArg) {
              return 'Error: as_path is required when content_id is provided.';
            }
            interface LinkedFileEntry {
              contentId: string;
              aliasOf: string;
              filename: string;
              mimeType: string;
              path: string;
              sizeBytes: number;
            }
            interface LinkResult {
              rootId: string;
              linkedFiles: number;
              linkedFolders: number;
              sourceContentId: string;
              sourceType: string;
              filename: string;
              linkedFileEntries: LinkedFileEntry[];
            }
            const linkFn = this.context.get<(args: { sourceContentId: string; asPath: string; callerUserId: string; targetChatId: string }) => Promise<LinkResult>>('_linkContentAtPath');
            if (!linkFn) {
              return 'Error: link mode is not available in this runtime (no _linkContentAtPath callback).';
            }
            const callerUserId = this.context.auth?.user_id;
            const targetChatId = this.context.metadata?.chatId as string | undefined;
            if (!callerUserId || !targetChatId) {
              return 'Error: link mode requires an authenticated user and an active chat context.';
            }
            try {
              const result = await linkFn({
                sourceContentId: contentIdArg,
                asPath: asPathArg,
                callerUserId,
                targetChatId,
              });
              const summary = result.sourceType === 'folder'
                ? `Linked folder content_id=${result.sourceContentId} at ${asPathArg} (${result.linkedFiles} file${result.linkedFiles === 1 ? '' : 's'}, ${result.linkedFolders} folder${result.linkedFolders === 1 ? '' : 's'}). Path-based tools (text_editor, bash) can now address files under ${asPathArg}; edits propagate to the canonical content.`
                : `Linked content_id=${result.sourceContentId} at ${asPathArg}. Path-based tools (text_editor, bash) can now read/edit ${asPathArg}; edits propagate to the canonical content.`;

              // Synthesize content_items for every newly linked file so the
              // planner's directory addendum (uamp-proxy.injectFileDirectoryAddendum)
              // and indexAvailableContent() see the linked paths immediately,
              // not just after a follow-up `ls`. For folder linking each
              // descendant gets one item; for single-file linking just the
              // root item is emitted. Folders themselves are not file/media
              // shaped so they're surfaced via ls/path-resolver instead.
              const entries = result.linkedFileEntries ?? [];
              if (entries.length > 0) {
                // FileContent doesn't formally type a `metadata` field, but
                // the planner-side directory addendum reads metadata.path
                // and metadata.aliasOf — see [lib/llm/uamp-proxy.ts:collectFileMarkersFromMessages].
                // Attach via cast so paths/alias are surfaced without
                // widening the public ContentItem type.
                const linkedItems = entries.map((entry) => ({
                  type: 'file' as const,
                  file: { url: '' },
                  filename: entry.filename,
                  mime_type: entry.mimeType || '',
                  content_id: entry.contentId,
                  size_bytes: entry.sizeBytes,
                  metadata: { path: entry.path, aliasOf: entry.aliasOf },
                  ...(entries.length === 1 && descArg ? { description: descArg } : {}),
                })) as unknown as ContentItem[];
                return {
                  text: summary,
                  content_items: linkedItems,
                } as StructuredToolResult;
              }
              return summary;
            } catch (err) {
              return `Error linking content_id=${contentIdArg}: ${(err as Error).message}`;
            }
          }

          if (!urlArg && !base64Arg) {
            return 'Either url, base64, or content_id (with as_path) must be provided.';
          }

          const progressFn = this.context.get<(callId: string, text: string) => void>('_toolProgressFn');
          const toolCall = this.context.get<{ id?: string }>('tool_call');
          const callId = toolCall?.id ?? '';

          let base64Data: string;
          let mimeType = args.mime_type as string || '';

          if (urlArg) {
            if (progressFn && callId) progressFn(callId, 'Downloading content...');
            try {
              const resp = await fetch(urlArg, { signal: AbortSignal.timeout(60_000) });
              if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
              const buffer = await resp.arrayBuffer();
              base64Data = Buffer.from(buffer).toString('base64');
              if (!mimeType) mimeType = resp.headers.get('content-type') || 'application/octet-stream';
            } catch (err) {
              const reason = (err as Error).message;
              return `Failed to download content from ${urlArg}: ${reason}. The URL may be expired or inaccessible.`;
            }
          } else {
            base64Data = base64Arg!.replace(/^data:[^;]+;base64,/, '');
            if (!mimeType) {
              const dataUriMatch = base64Arg!.match(/^data:([^;]+);base64,/);
              mimeType = dataUriMatch?.[1] || 'application/octet-stream';
            }
          }

          const chatId = this.context.metadata?.chatId as string | undefined;
          const agentId = this.context.metadata?.agentId as string | undefined;
          const userId = this.context.auth?.user_id;

          const savedResult = await mediaSaver.save(base64Data, mimeType, { chatId, agentId, userId } as Record<string, unknown>);
          const savedUrl = typeof savedResult === 'string' ? savedResult : savedResult.url;
          const contentId = typeof savedResult === 'string'
            ? (savedResult.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i)?.[1] || crypto.randomUUID())
            : savedResult.content_id;

          const mediaType = mimeType.startsWith('image/') ? 'image'
            : mimeType.startsWith('video/') ? 'video'
            : mimeType.startsWith('audio/') ? 'audio'
            : 'file';

          let contentItem: ContentItem;
          if (mediaType === 'image') {
            contentItem = { type: 'image', image: { url: savedUrl }, content_id: contentId, description: descArg } satisfies ImageContent;
          } else if (mediaType === 'video') {
            contentItem = { type: 'video', video: { url: savedUrl }, content_id: contentId, description: descArg } satisfies VideoContent;
          } else if (mediaType === 'audio') {
            contentItem = { type: 'audio', audio: { url: savedUrl }, content_id: contentId, description: descArg } satisfies AudioContent;
          } else {
            contentItem = { type: 'file', file: { url: savedUrl }, filename: (args.filename as string) || 'saved-content', mime_type: mimeType, content_id: contentId, description: descArg } satisfies FileContent;
          }

          return {
            text: `Saved ${mediaType} to content library (content_id: ${contentId}). ${descArg}`,
            content_items: [contentItem],
          } as StructuredToolResult;
        },
      });
    }

    // Warn earlier for short budgets (e.g. 10) so the "stop and summarize"
    // system message actually fires before the cap is hit. 50% works for
    // small budgets; clamp at 80% for large ones so chatty agents aren't
    // nudged too early.
    const warnFraction = this.maxToolIterations <= 20 ? 0.5 : 0.8;
    const warnAtIteration = Math.max(1, Math.ceil(this.maxToolIterations * warnFraction));
    let budgetWarned = false;

    console.log(
      `[agent] entering tool-call loop maxToolIterations=${this.maxToolIterations} ` +
      `warnAt=${warnAtIteration}`,
    );

    while (iteration < this.maxToolIterations) {
      if (signal?.aborted) {
        yield createResponseErrorEvent('aborted', 'Request was cancelled');
        break;
      }

      iteration++;
      console.log(`[agent] iteration ${iteration}/${this.maxToolIterations}`);

      if (!budgetWarned && iteration >= warnAtIteration) {
        budgetWarned = true;
        const budgetMsg = `You have used ${iteration}/${this.maxToolIterations} of your tool-call budget for this response. Stop delegating and calling tools — summarize what you have done so far and deliver the final answer to the user now. Do not start new workflows.`;
        conversation.push({ role: 'system', content: budgetMsg });
        console.log(
          `[agent] BUDGET-WARNING injected: agent=${this.name ?? '?'} iter=${iteration}/${this.maxToolIterations} ` +
          `(visible to LLM as system message; should appear in body.system for Anthropic / messages[].role=system for OpenAI)`,
        );
        if (process.env.LOG_LOOP_DEBUG === '1' || process.env.LOG_LLM_PAYLOAD === '1') {
          console.log(`[loop-debug] BUDGET-WARNING content: ${JSON.stringify(budgetMsg).slice(0, 200)}`);
        }
      }

      // Set conversation, tools, and skills in context so handoff/payment skills can access them
      this.context.set('_agentic_messages', conversation);
      const toolDefs = this.getToolDefinitions();
      this.context.set('_agentic_tools', toolDefs);
      this.context.set('_skills', this.skills);

      // Run before_llm_call hooks
      await this.runHooks('before_llm_call', {
        metadata: { conversation_length: conversation.length },
        iteration,
      });

      // Call handoff and collect all events, eagerly yielding streamable deltas
      const collected: ServerEvent[] = [];
      const eagerlyYielded = new Set<ServerEvent>();
      try {
        for await (const event of handoff.handler(events, this.context)) {
          collected.push(event);

          if (event.type === 'response.delta') {
            const delta = (event as unknown as { delta: ResponseDelta }).delta;
            await this.runHooks('on_chunk', { chunk: delta, event });

            // Platform tool file deltas (text_editor create/edit, save_content,
            // etc. handled server-side by the LLM proxy) need to be surfaced to
            // the agent's own content index so `present(content_id)` can resolve
            // them. Without this, the proxy creates a file with content_id X
            // and returns "call present(X)" in the tool result, but when the
            // LLM does exactly that, the local `present` handler walks
            // `conversation[*].content_items` + `collectedContentItems`, finds
            // neither, and returns "Content not found: X. Available: none" —
            // causing the classic create→present→retry loop.
            const fileDeltaForIndex = delta as unknown as { type?: string; content_id?: string };
            if (
              fileDeltaForIndex.type === 'file'
              && fileDeltaForIndex.content_id
              && isMediaContent(delta as unknown as ContentItem)
            ) {
              const ci = delta as unknown as ContentItem;
              const cid = (ci as { content_id?: string }).content_id;
              const existingIdx = collectedContentItems.findIndex(
                (o) => (o as { content_id?: string }).content_id === cid,
              );
              if (existingIdx === -1) {
                collectedContentItems.push(ci);
              } else {
                // Replace in place with the latest delta so subsequent
                // present() / final-output emission carries the freshest
                // metadata (e.g. command='str_replace' rather than the
                // original 'create'). Without this the DocumentChip badge
                // would freeze on "Created" even after edits.
                collectedContentItems[existingIdx] = ci;
              }
              console.log(
                `[agent] platform-file-indexed: content_id=${cid} type=${ci.type} ` +
                `filename=${(ci as { filename?: string }).filename ?? '?'} ` +
                `command=${((ci as { metadata?: { command?: string } }).metadata?.command) ?? '?'} ` +
                `${existingIdx === -1 ? '(new)' : '(updated)'}`,
              );
            }

            // Eager-yield contract (Phase 2 unification):
            // ALL response.delta events — text, thinking, tool_call (internal
            // OR platform), tool_result, tool_progress, file — are yielded
            // immediately in the order the model streamed them. The UI chip
            // for any tool_call is then anchored to its true stream position.
            //
            // Execution lifecycle for internal tools (handler runs after the
            // model finishes its turn) is signalled separately by the
            // `tool_progress` events emitted from `_toolProgressFn` and the
            // final `tool_result` delta (around line ~2094) — both correlate
            // back to the original `tool_call` by `call_id`. We deliberately
            // no longer re-yield the `tool_call` delta at execution time,
            // since the UI already has it.
            //
            // Why this matters: the previous design suppressed internal
            // `tool_call` deltas here and re-emitted them at execution time,
            // which silently assumed internal tools always come at end-of-
            // round. Any model that emitted `delegate` / `present` /
            // `read_content` / `memory` mid-text would have its chip rendered
            // at the bottom of the bubble. Memory exposed this first by being
            // dual-registered on portal-runtime agents (now reverted in
            // `lib/agents/factories.ts: PortalStorageFactory`).
            const isToolCallDelta = delta.type === 'tool_call';
            const toolName = isToolCallDelta ? delta.tool_call?.name : undefined;
            console.log(
              `[agent] eager-yield: delta.type=${delta.type} ` +
              `text=${JSON.stringify((delta as any).text)?.slice(0, 80)} ` +
              `hasToolProgress=${!!(delta as any).tool_progress}` +
              (toolName ? ` toolName=${toolName} internal=${this._isInternalTool(toolName)}` : ''),
            );
            eagerlyYielded.add(event);
            yield event;
          } else if (event.type === 'thinking' || event.type === 'progress') {
            eagerlyYielded.add(event);
            yield event;
          } else if (event.type === 'response.created') {
            // Metadata event carrying only response_id — yield immediately so
            // it precedes any response.delta on the wire. Without this, deltas
            // are eagerly yielded while response.created is flushed in the
            // tail loop below, landing AFTER the deltas (see portal-llm
            // streaming test). Clients rely on response.created arriving first
            // to bind subsequent deltas to a response_id.
            eagerlyYielded.add(event);
            yield event;
          }
        }
      } catch (error) {
        await this.runHooks('on_error', { error: error as Error });
        yield createResponseErrorEvent(
          'handoff_error',
          (error as Error).message
        );
        this.toolRegistry.delete('present');
        this.toolRegistry.delete('read_content');
        if (mediaSaver) this.toolRegistry.delete('save_content');
        await this.runHooks('after_handoff', { handoff_target: handoff.name });
        await this.runHooks('finalize_connection', {});
        return;
      }

      // Run after_llm_call hooks
      await this.runHooks('after_llm_call', {
        metadata: { events_collected: collected.length },
        iteration,
      });

      if (this.context.get<boolean>('_payment_exhausted')) {
        yield createResponseErrorEvent(
          'payment_exhausted',
          'Payment token exhausted. The conversation was stopped to avoid further charges.'
        );
        break;
      }

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

      // Track repeated identical tool calls so the soft nudge below can fire.
      // Hard loop-stopping is intentionally NOT done here — the iteration cap
      // (`maxToolIterations`) is the only termination signal; this counter is
      // purely advisory.
      //
      // Normalize certain tool arguments before comparison so semantically
      // identical calls are not counted as distinct just because the model
      // reworded the prompt slightly. This is especially important for:
      //   - `delegate`: LLM-generated free-form messages vary in wording
      //     across retries ("Create unicorn.html" vs "Please create the
      //     unicorn page") but drive the same sub-agent down the same path.
      //   - `text_editor`/`str_replace_based_edit_tool`: file_text bodies
      //     differ across retries but `{command, basename}` doesn't, and
      //     repeated `create` for the same basename is the loop signature.
      const normalizeArgsForDedup = (name: string, rawArgs: string): string => {
        try {
          const parsed = JSON.parse(rawArgs || '{}') as Record<string, unknown>;
          if (name === 'delegate') {
            const agent = String(parsed.agent ?? '').toLowerCase();
            const msg = String(parsed.message ?? '')
              .toLowerCase()
              .replace(/\s+/g, ' ')
              .trim()
              .slice(0, 160);
            return JSON.stringify({ agent, msg });
          }
          if (name === 'text_editor' || name === 'str_replace_based_edit_tool') {
            const cmd = String(parsed.command ?? '');
            const basename = String(parsed.path ?? '').replace(/^.*[\\/]/, '');
            return JSON.stringify({ cmd, basename });
          }
          return rawArgs;
        } catch {
          return rawArgs;
        }
      };

      if (toolCalls.length > 0) {
        const key = toolCalls.map(tc => `${tc.name}:${normalizeArgsForDedup(tc.name, tc.arguments)}`).join('|');
        const last = recentToolCalls[recentToolCalls.length - 1];
        if (last && last.key === key) {
          last.count++;
        } else {
          recentToolCalls.push({ key, count: 1 });
          if (recentToolCalls.length > 5) recentToolCalls.shift();
        }
        const tracked = recentToolCalls[recentToolCalls.length - 1];
        if (tracked && tracked.count >= 2) {
          const firstName = toolCalls[0]?.name ?? '?';
          const firstArgs = toolCalls[0]?.arguments ?? '';
          console.warn(
            `[agent] repeated-call: agent=${this.name ?? '?'} iter=${iteration}/${this.maxToolIterations} ` +
            `tool=${firstName} count=${tracked.count} args=${firstArgs.slice(0, 120)}`,
          );
        }
      }

      // Detect hallucinated tool calls: LLM wrote a tool call as text instead of
      // using the function calling mechanism. Re-prompt so the loop can recover.
      const HALLUCINATED_TOOL_RE = /\[Called tool \w+\([\s\S]*?\)\]/;
      if (toolCalls.length === 0) {
        const responseText = doneEvent.response.output
          .filter((item: ContentItem) => item.type === 'text')
          .map((item: ContentItem) => (item as { text: string }).text)
          .join('');

        if (HALLUCINATED_TOOL_RE.test(responseText)) {
          console.warn(`[agent] detected hallucinated tool call in text output (iter ${iteration}), re-prompting LLM`);
          conversation.push({
            role: 'assistant' as const,
            content: responseText,
          });
          conversation.push({
            role: 'user' as const,
            content: 'ERROR: You wrote a tool call as text instead of using the function calling mechanism. '
              + 'Do NOT write tool calls in your text response. Use the actual tool/function calling API. '
              + 'If you intended to call a tool, call it now using the proper mechanism. '
              + 'If you have already completed the task, respond with just your final message to the user.',
          });
          continue;
        }
      }

      // No tool calls -- LLM is done. Yield all events and exit.
      if (toolCalls.length === 0) {
        let outputContentItems: ContentItem[];
        if (!hasPresentTool) {
          outputContentItems = collectedContentItems;
        } else if (presentedIds.size > 0) {
          outputContentItems = collectedContentItems.filter(ci => presentedIds.has((ci as { content_id?: string }).content_id || ''));
        } else if (collectedContentItems.length > 0) {
          console.warn(`[agent] present tool available but never called — ${collectedContentItems.length} content item(s) not displayed`);
          outputContentItems = [];
        } else {
          outputContentItems = [];
        }

        for (const event of collected) {
          if (eagerlyYielded.has(event)) continue;
          if (event.type === 'response.done' && outputContentItems.length > 0) {
            const done = event as { response: { output: ContentItem[] } };
            yield {
              ...event,
              response: { ...done.response, output: [...done.response.output, ...outputContentItems] },
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
        const mixedPresentDeltas: ServerEvent[] = [];
        this.context.set('_presentDeltaFn', (event: ServerEvent) => {
          mixedPresentDeltas.push(event);
        });

        for (const tc of internalCalls) {
          if (signal?.aborted) break;

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

        this.context.delete('_presentDeltaFn');

        // Yield all collected events, but filter response.done output to only external tool calls + text
        let externalOutputContentItems: ContentItem[];
        if (!hasPresentTool) {
          externalOutputContentItems = collectedContentItems;
        } else if (presentedIds.size > 0) {
          externalOutputContentItems = collectedContentItems.filter(ci => presentedIds.has((ci as { content_id?: string }).content_id || ''));
        } else if (collectedContentItems.length > 0) {
          console.warn(`[agent] present tool available but never called (mixed path) — ${collectedContentItems.length} content item(s) not displayed`);
          externalOutputContentItems = [];
        } else {
          externalOutputContentItems = [];
        }

        for (const event of collected) {
          if (eagerlyYielded.has(event)) continue;
          if (event.type === 'response.done') {
            const filteredOutput = doneEvent.response.output.filter(item => {
              if (item.type === 'tool_call' && item.tool_call) {
                return externalCalls.some(ext => ext.id === item.tool_call!.id);
              }
              return true;
            });
            yield {
              ...event,
              response: { ...doneEvent.response, output: [...filteredOutput, ...externalOutputContentItems] },
            } as ServerEvent;
          } else {
            yield event;
          }
        }
        // Yield present deltas collected during mixed internal tool execution
        for (const delta of mixedPresentDeltas) yield delta;
        break;
      }

      // All tool calls are internal -- execute and continue loop.
      //
      // Post-handoff fallback (defensive): the eager-yield contract above
      // already streamed every response.delta / thinking / progress event in
      // model order, so this loop should be a no-op for properly-eagerly-
      // yielded events. We keep a defensive sweep for anything that slipped
      // through (e.g. a future event type that wasn't recognised in the
      // eager-yield branch) so it still reaches the client. tool_call deltas
      // are intentionally NOT re-yielded here — duplicating them would create
      // a second chip in the UI for the same call.
      for (const event of collected) {
        if (eagerlyYielded.has(event)) continue;
        if (event.type === 'response.delta') {
          const delta = (event as unknown as { delta: ResponseDelta }).delta;
          if (delta.type === 'tool_call') {
            console.log(`[agent] post-handoff: skipping non-eager tool_call (avoid UI dupe): name=${delta.tool_call?.name} id=${delta.tool_call?.id}`);
            continue;
          }
          console.log(`[agent] post-handoff-yield: delta.type=${delta.type} (defensive — eager-yield missed it)`);
          yield event;
        } else if (event.type === 'thinking' || event.type === 'progress') {
          yield event;
        }
      }

      // Replay any platform-tool rounds the proxy ran inside this single
      // LLM turn (text_editor, bash, etc.). Without this, the LLM's next
      // iteration sees a conversation that jumps straight from the user
      // prompt to a `present(content_id=X)` call with no record of the
      // create that produced X, and re-attempts the create — see the
      // create→present loop documented in data/logs/llm-payloads/.
      const preExecuted = (doneEvent.response as { pre_executed_rounds?: Array<{
        assistant: { role: 'assistant'; content?: string; tool_calls: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }> };
        tool_results: Array<{ role: 'tool'; tool_call_id: string; name: string; content: string }>;
      }> }).pre_executed_rounds;
      if (preExecuted && preExecuted.length > 0) {
        let appendedAsst = 0;
        let appendedResults = 0;
        for (const round of preExecuted) {
          conversation.push({
            role: 'assistant',
            content: round.assistant.content ?? null,
            tool_calls: round.assistant.tool_calls,
          });
          appendedAsst++;
          for (const tr of round.tool_results) {
            conversation.push({
              role: 'tool',
              content: tr.content,
              tool_call_id: tr.tool_call_id,
              name: tr.name,
            });
            appendedResults++;
          }
        }
        if (process.env.LOG_LOOP_DEBUG === '1' || process.env.LOG_LLM_PAYLOAD === '1') {
          console.log(`[loop-debug] agent replay iter=${iteration} appended assistant_turns=${appendedAsst} tool_results=${appendedResults} conversation.length=${conversation.length}`);
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

      // Per-iteration accumulator for the optional `_recordToolTurn` hook.
      // The portal route handler may set this hook to persist each iteration's
      // assistant tool_calls + tool result rows to the chat / sub-chat so
      // re-loading the conversation (U2A or A2A re-delegation) can rebuild
      // the LLM's view of prior actions and avoid the create→present loop.
      type RecordableToolCall = { id: string; name: string; arguments: string };
      type RecordableToolResult = {
        toolCallId: string;
        toolName: string;
        result: string;
        isError?: boolean;
        contentItems?: ContentItem[];
        data?: Record<string, unknown>;
      };
      const internalRecordableCalls: RecordableToolCall[] = internalCalls.map(tc => ({
        id: tc.id, name: tc.name, arguments: tc.arguments,
      }));
      const internalRecordableResults: RecordableToolResult[] = [];

      // Execute each internal tool and add results to conversation
      for (const tc of internalCalls) {
        if (signal?.aborted) {
          yield createResponseErrorEvent('aborted', 'Request was cancelled');
          break;
        }

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

        // The `tool_call` delta itself was already eagerly yielded in stream
        // position when the model emitted it (see eager-yield contract around
        // line 1599). Do NOT re-yield it here — that would duplicate the chip
        // in the UI. The `tool_progress` events from `_toolProgressFn` (set
        // up immediately below) and the final `tool_result` (around line
        // ~2100) correlate back to this tool_call by `call_id`, so the UI
        // can transition the existing chip through running → done states
        // without any additional `tool_call` emission.
        console.log(`[agent] starting internal tool execution: name=${tc.name} id=${tc.id}`);

        // Run tool with an AsyncQueue so streaming tools (e.g. delegate)
        // can push progress events that we yield in real-time.
        const progressQueue = new AsyncQueue<ServerEvent>();
        this.context.set('_toolProgressFn', (callId: string, text: string, opts?: { replace?: boolean; media_type?: string; status?: string; progress_percent?: number; estimated_duration_ms?: number }) => {
          progressQueue.push({
            type: 'response.delta',
            event_id: generateEventId(),
            delta: { type: 'tool_progress', tool_progress: { call_id: callId, text, ...opts } },
          } as unknown as ServerEvent);
        });
        this.context.set('_presentDeltaFn', (event: ServerEvent) => {
          progressQueue.push(event);
        });

        const resultPromise = this._executeInternalToolCall(tc).then((r) => {
          this.context.delete('_toolProgressFn');
          this.context.delete('_presentDeltaFn');
          progressQueue.close();
          return r;
        });

        for await (const progressEvent of progressQueue) {
          yield progressEvent;
        }
        const result = await resultPromise;

        // Set tool_result in context for Python-compatible hooks
        this.context.set('tool_result', result);

        const isToolError = typeof result === 'string' && result.startsWith('Tool execution error:');

        const afterResult = await this.runHooks('after_tool', {
          tool_name: tc.name,
          tool_result: result,
          error: isToolError ? new Error(result) : undefined,
        });
        const finalResult = afterResult?.tool_result ?? result;

        // Also fire after_toolcall (Python-compatible hook name)
        await this.runHooks('after_toolcall', {
          tool_name: tc.name,
          tool_result: finalResult,
          error: isToolError ? new Error(typeof finalResult === 'string' ? finalResult : '') : undefined,
        });

        let resultText = typeof finalResult === 'string' ? finalResult : (finalResult as StructuredToolResult).text;
        const resultItems = (typeof finalResult === 'object' && finalResult !== null && 'content_items' in (finalResult as object))
          ? (finalResult as StructuredToolResult).content_items
          : undefined;
        const postMessages = (typeof finalResult === 'object' && finalResult !== null && '_post_messages' in (finalResult as object))
          ? (finalResult as StructuredToolResult)._post_messages
          : undefined;
        // Optional structured metadata that flows through to the rendered
        // tool_result envelope. Used by the NLI delegate skill to surface
        // `subChatId` for the parent-side <DelegateSubChatPreview /> (plan §4).
        const resultData = (typeof finalResult === 'object' && finalResult !== null && 'data' in (finalResult as object))
          ? (finalResult as StructuredToolResult).data
          : undefined;

        const lastRepeat = recentToolCalls[recentToolCalls.length - 1];
        // Lower threshold for `delegate`: each delegation is an expensive
        // cross-agent call, and the orchestrator-loop pattern (re-delegating
        // a rephrased prompt to the same sub-agent after a failure) is
        // exactly what we want to short-circuit. For other tools, keep the
        // original 3x threshold so a model can still retry idempotent reads
        // once or twice without being nagged.
        const nudgeThreshold = tc.name === 'delegate' ? 2 : 3;
        if (lastRepeat && lastRepeat.count >= nudgeThreshold) {
          if (tc.name === 'delegate') {
            resultText =
              `You have delegated to this agent ${lastRepeat.count} times with essentially the same request. ` +
              `The previous attempt did not produce the requested artifact — retrying will not change that. ` +
              `Stop delegating. Either tell the user the task could not be completed and ask how to proceed, ` +
              `or pick a different agent via search. Do NOT call delegate again with a rephrased version of this message.`;
          } else {
            resultText = `You have called this tool ${lastRepeat.count} times with the same arguments. The result is unlikely to change. Please respond to the user or try a different approach.`;
          }
          console.log(`[agent] repeated tool nudge: tool=${tc.name} count=${lastRepeat.count} threshold=${nudgeThreshold}`);
        }
        console.log(`[agent] tool ${tc.name} result: hasContentItems=${!!resultItems} count=${resultItems?.length ?? 0} isError=${isToolError}`);

        if (resultItems) {
          collectedContentItems.push(...resultItems);
        }

        // Notify client of tool result (include content_items so persisters can extract media)
        yield {
          type: 'response.delta',
          event_id: generateEventId(),
          delta: {
            type: 'tool_result',
            tool_result: {
              call_id: tc.id,
              result: resultText,
              is_error: isToolError || undefined,
              content_items: resultItems,
              ...(resultData !== undefined ? { data: resultData } : {}),
            },
          },
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

        // Append any tool-supplied follow-up messages AFTER the tool_result
        // row. read_content uses this to inject its `_inline_for_llm` user
        // message without breaking Anthropic/OpenAI's tool_use→tool_result
        // adjacency requirement.
        if (postMessages?.length) {
          conversation.push(...postMessages);
          console.log(`[agent] appended ${postMessages.length} post_message(s) after tool=${tc.name} tool_result`);
        }

        internalRecordableResults.push({
          toolCallId: tc.id,
          toolName: tc.name,
          result: resultText,
          isError: isToolError || undefined,
          contentItems: resultItems,
          ...(resultData !== undefined ? { data: resultData } : {}),
        });
      }

      // Persist the iteration's internal tool turn (assistant tool_calls +
      // role=tool result rows) via the optional context hook. Platform tools
      // are persisted separately by the LLM proxy at execution time, so the
      // hook here only sees calls the agent itself ran.
      const recordToolTurnFn = this.context.get<(args: {
        assistantText?: string;
        toolCalls: RecordableToolCall[];
        toolResults: RecordableToolResult[];
      }) => Promise<void> | void>('_recordToolTurn');
      if (recordToolTurnFn && internalRecordableCalls.length > 0) {
        try {
          await recordToolTurnFn({
            assistantText: assistantText && assistantText.length > 0 ? assistantText : undefined,
            toolCalls: internalRecordableCalls,
            toolResults: internalRecordableResults,
          });
          if (process.env.LOG_LOOP_DEBUG === '1' || process.env.LOG_LLM_PAYLOAD === '1') {
            console.log(`[loop-debug] agent _recordToolTurn iter=${iteration} tool_calls=${internalRecordableCalls.length} tool_results=${internalRecordableResults.length}`);
          }
        } catch (err) {
          console.warn(`[agent] _recordToolTurn hook failed (non-fatal): ${(err as Error).message}`);
        }
      }

      // Loop continues: handoff will be called again with updated conversation in context
    }

    if (iteration >= this.maxToolIterations) {
      const dist = new Map<string, number>();
      for (const r of recentToolCalls) {
        const toolName = r.key.split(':')[0] ?? '?';
        dist.set(toolName, (dist.get(toolName) ?? 0) + r.count);
      }
      const summary = [...dist.entries()].map(([n, c]) => `${n}=${c}`).join(' ');
      console.warn(
        `[agent] max-iter bailout: agent=${this.name ?? '?'} iter=${iteration}/${this.maxToolIterations} ` +
        `tools=[${summary || '(none tracked)'}]`,
      );
      yield createResponseErrorEvent(
        'max_iterations',
        `Agent reached maximum tool iterations (${this.maxToolIterations}). Stopping to prevent infinite loop.`
      );
    }

    // Clean up built-in content tools
    this.toolRegistry.delete('present');
    this.toolRegistry.delete('read_content');
    if (mediaSaver) this.toolRegistry.delete('save_content');

    // Run after_handoff + finalize_connection hooks
    await this.runHooks('after_handoff', {
      handoff_target: handoff.name,
    });
    await this.runHooks('finalize_connection', {});
  }
  
  // ============================================================================
  // Prompt Execution
  // ============================================================================

  /**
   * Execute all registered prompts, filtered by scope, in priority order.
   * Returns concatenated prompt text.
   */
  private async _executePrompts(scope?: string): Promise<string> {
    if (this.promptRegistry.length === 0) return '';

    const scopeHierarchy: Record<string, number> = { admin: 3, owner: 2, all: 1 };
    const userLevel = scopeHierarchy[scope || 'all'] || 1;

    const parts: string[] = [];
    for (const p of this.promptRegistry) {
      const promptScopes = Array.isArray(p.scope) ? p.scope : [p.scope];
      const accessible = promptScopes.some(s => {
        const required = scopeHierarchy[s] || 1;
        return userLevel >= required;
      });
      if (!accessible) continue;

      try {
        const result = await p.handler(this.context);
        if (result) parts.push(result);
      } catch (err) {
        console.warn(`[agent] prompt "${p.name}" threw:`, (err as Error).message);
      }
    }
    return parts.join('\n\n');
  }

  /**
   * Enhance base instructions with dynamic prompt content from skills.
   */
  private async _enhanceInstructionsWithPrompts(baseInstructions: string | undefined): Promise<string | undefined> {
    const dynamic = await this._executePrompts();
    if (!dynamic) return baseInstructions;
    return baseInstructions ? `${baseInstructions}\n\n${dynamic}` : dynamic;
  }

  /**
   * Run with messages (convenience method)
   */
  async run(messages: Message[], options: RunOptions = {}): Promise<RunResponse> {
    if (options.signal) {
      (this.context as ContextImpl).signal = options.signal;
    }

    // Run before_run hooks first so they can modify messages
    const beforeResult = await this.runHooks('before_run', { messages });
    if (beforeResult?.abort) {
      throw new Error(beforeResult.abort_reason || 'Run aborted by hook');
    }
    const effectiveMessages = beforeResult?.messages ?? messages;

    // Convert messages to UAMP events
    const events: ClientEvent[] = [];

    const runExtensions: Record<string, unknown> = {};
    if (options.paymentToken) {
      runExtensions['X-Payment-Token'] = options.paymentToken;
    }
    // Forward the agent's bound chatId so the LLM proxy can scope platform-tool
    // operations (text_editor / fs / bash / file_search / memory) to this chat.
    // Without this, ToolSessionState is null at the proxy and platform tool
    // calls (e.g. text_editor) are silently dropped.
    {
      const ctxChatId = (this.context.metadata as Record<string, unknown> | undefined)?.chatId as string | undefined;
      if (ctxChatId) runExtensions['X-Chat-Id'] = ctxChatId;
    }

    // Enhance instructions with dynamic skill prompts
    const effectiveInstructions = await this._enhanceInstructionsWithPrompts(
      options.instructions || this.instructions,
    );

    // Create session
    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
        instructions: effectiveInstructions,
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
      ...(options.client_capabilities && { client_capabilities: options.client_capabilities }),
    } as SessionCreateEvent);
    
    // Add messages as input events
    for (const msg of effectiveMessages) {
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
    if (effectiveInstructions) {
      initialConv.push({ role: 'system', content: effectiveInstructions });
    }
    for (const msg of effectiveMessages) {
      // `name` is required so role='tool' rows round-trip to adapters'
      // functionResponse / tool_call name (Google falls back to 'unknown'
      // and the LLM cannot match the response to a prior call otherwise).
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
        name: msg.name,
      });
    }
    this.context.set('_initial_conversation', initialConv);
    
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

    // Run before_run hooks first so they can modify messages
    const beforeResult = await this.runHooks('before_run', { messages });
    if (beforeResult?.abort) {
      yield {
        type: 'error',
        error: new Error(beforeResult.abort_reason || 'Run aborted by hook'),
      };
      return;
    }
    const effectiveMessages = beforeResult?.messages ?? messages;

    // Convert messages to UAMP events
    const events: ClientEvent[] = [];

    const sessionExtensions: Record<string, unknown> = {};
    if (options.paymentToken) {
      sessionExtensions['X-Payment-Token'] = options.paymentToken;
    }
    // See the matching comment in run() above — forward chatId so platform
    // tools (text_editor, fs, bash, …) can execute against the right chat.
    {
      const ctxChatId = (this.context.metadata as Record<string, unknown> | undefined)?.chatId as string | undefined;
      if (ctxChatId) sessionExtensions['X-Chat-Id'] = ctxChatId;
    }

    // Enhance instructions with dynamic skill prompts
    const effectiveInstructionsS = await this._enhanceInstructionsWithPrompts(
      options.instructions || this.instructions,
    );

    // Create session
    const sessionTools = options.tools ? options.tools.map(t => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      },
    })) : this.getToolDefinitions();
    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
        instructions: effectiveInstructionsS,
        tools: sessionTools,
        ...(Object.keys(sessionExtensions).length > 0 && { extensions: sessionExtensions }),
      },
      ...(options.client_capabilities && { client_capabilities: options.client_capabilities }),
    } as SessionCreateEvent);
    
    // Add messages as input events
    for (const msg of effectiveMessages) {
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
    if (effectiveInstructionsS) {
      initialConvS.push({ role: 'system', content: effectiveInstructionsS });
    }
    for (const msg of effectiveMessages) {
      // See run() above — `name` is required so role='tool' rows round-trip
      // to adapters' functionResponse name (Google adapter falls back to
      // 'unknown' otherwise, which Gemini silently rejects).
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
        name: msg.name,
      });
    }
    this.context.set('_initial_conversation', initialConvS);
    
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
        const toolProgress = (delta as unknown as { tool_progress?: StreamChunk['tool_progress'] }).tool_progress;
        if (toolProgress) {
          yield {
            type: 'tool_progress',
            tool_progress: { ...toolProgress },
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
        const error = (event as { error: { code?: string; message: string; details?: unknown } }).error;
        const err = new Error(error.message);
        (err as any).code = error.code;
        (err as any).details = error.details;
        yield {
          type: 'error',
          error: err,
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
